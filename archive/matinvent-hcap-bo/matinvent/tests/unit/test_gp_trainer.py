"""Unit tests for GP training manager."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from rewards.gp_trainer import GPTrainingManager
from rewards.gp.surrogate import GPSurrogate
from rewards.calculators.orb.featurizer import ORBFeaturizer
from memory.ltm import LongTimeMem


@pytest.fixture
def featurizer(device):
    """Create ORB featurizer."""
    return ORBFeaturizer(n_components=10, device=device)


@pytest.fixture
def gp_model(device):
    """Create GP model."""
    return GPSurrogate(input_dim=10, task='bulk_modulus', device=device)


@pytest.fixture
def trainer(gp_model, featurizer, tmp_path):
    """Create GP training manager."""
    return GPTrainingManager(
        gp_model=gp_model,
        featurizer=featurizer,
        retrain_frequency=5,
        metrics_dir=str(tmp_path / 'gp_metrics'),
        min_samples=10,
        validation_split=0.2
    )


@pytest.fixture
def mock_ltm(structure_list, mock_property_values):
    """Create mock long-term memory with data."""
    ltm = LongTimeMem()

    # Add structures with property values and features
    n = len(structure_list)
    rewards = np.random.rand(n)
    property_values = mock_property_values[:n]
    features = [np.random.randn(10) for _ in range(n)]

    ltm.extend(
        structure_list,
        rewards,
        step=0,
        property_values=property_values,
        features=features,
        calculators_used=['orb'] * n
    )

    return ltm


def test_trainer_initialization(trainer):
    """Test trainer initializes correctly."""
    assert trainer.retrain_frequency == 5
    assert trainer.min_samples == 10
    assert trainer.validation_split == 0.2
    assert trainer.last_retrain_step == -1
    assert len(trainer.training_history) == 0


def test_should_retrain(trainer):
    """Test retrain timing logic."""
    # Should retrain at step 0
    assert trainer.should_retrain(0)

    # Simulate retraining at step 0
    trainer.last_retrain_step = 0

    # Should not retrain at steps 1-4
    for step in range(1, 5):
        assert not trainer.should_retrain(step)

    # Should retrain at step 5
    assert trainer.should_retrain(5)


def test_collect_training_data_insufficient_samples(trainer):
    """Test data collection with insufficient samples."""
    ltm = LongTimeMem()

    # Add only 5 samples (less than min_samples=10)
    from tests.fixtures.structures import fcc_structure
    ltm.extend(
        [fcc_structure()] * 5,
        np.random.rand(5),
        step=0,
        property_values=np.random.rand(5) * 100,
        features=[np.random.randn(10) for _ in range(5)]
    )

    X, y = trainer.collect_training_data(ltm, 'property_value')

    assert X is None
    assert y is None


def test_collect_training_data_sufficient_samples(trainer, mock_ltm):
    """Test data collection with sufficient samples."""
    # Add more samples to reach threshold
    from tests.fixtures.structures import bcc_structure
    ltm = mock_ltm
    ltm.extend(
        [bcc_structure()] * 10,
        np.random.rand(10),
        step=1,
        property_values=np.random.rand(10) * 100,
        features=[np.random.randn(10) for _ in range(10)]
    )

    X, y = trainer.collect_training_data(ltm, 'property_value')

    assert X is not None
    assert y is not None
    assert X.shape[1] == 10  # Feature dimension
    assert len(X) == len(y)
    assert len(X) >= trainer.min_samples


def test_collect_training_data_with_cached_features(trainer, mock_ltm):
    """Test data collection uses cached features from LTM."""
    X, y = trainer.collect_training_data(mock_ltm, 'property_value')

    assert X is not None
    # Features should be extracted from LTM (cached)
    assert X.shape[0] == len(mock_ltm)


def test_retrain_insufficient_data(trainer):
    """Test retraining with insufficient data."""
    ltm = LongTimeMem()

    # Add only a few samples
    from tests.fixtures.structures import fcc_structure
    ltm.extend(
        [fcc_structure()] * 3,
        np.random.rand(3),
        step=0,
        property_values=np.random.rand(3) * 100
    )

    metrics = trainer.retrain(ltm, current_step=0)

    assert metrics == {}  # Should return empty dict
    assert len(trainer.training_history) == 0


def test_retrain_sufficient_data(trainer, mock_ltm):
    """Test retraining with sufficient data."""
    # Add more data
    from tests.fixtures.structures import bcc_structure
    mock_ltm.extend(
        [bcc_structure()] * 10,
        np.random.rand(10),
        step=1,
        property_values=np.random.rand(10) * 100,
        features=[np.random.randn(10) for _ in range(10)]
    )

    metrics = trainer.retrain(mock_ltm, current_step=5)

    assert metrics != {}
    assert 'step' in metrics
    assert metrics['step'] == 5
    assert 'train_R2' in metrics
    assert 'train_RMSE' in metrics
    assert len(trainer.training_history) == 1
    assert trainer.last_retrain_step == 5


def test_retrain_creates_train_val_split(trainer, mock_ltm):
    """Test that retraining creates train/val split."""
    # Add enough data for meaningful split
    from tests.fixtures.structures import bcc_structure, hcp_structure
    mock_ltm.extend(
        [bcc_structure()] * 20,
        np.random.rand(20),
        step=1,
        property_values=np.random.rand(20) * 100,
        features=[np.random.randn(10) for _ in range(20)]
    )

    metrics = trainer.retrain(mock_ltm, current_step=5)

    # Should have both train and val metrics
    assert 'n_train_samples' in metrics
    assert 'n_val_samples' in metrics
    assert metrics['n_val_samples'] > 0


def test_retrain_improves_gp(trainer, mock_ltm):
    """Test that retraining improves GP model."""
    # Add more data
    from tests.fixtures.structures import bcc_structure
    mock_ltm.extend(
        [bcc_structure()] * 15,
        np.random.rand(15),
        step=1,
        property_values=np.random.rand(15) * 100 + 150,
        features=[np.random.randn(10) for _ in range(15)]
    )

    # GP should not be trained initially
    assert not trainer.gp_model.is_trained

    # Retrain
    metrics = trainer.retrain(mock_ltm, current_step=5)

    # GP should now be trained
    assert trainer.gp_model.is_trained


def test_retrain_saves_metrics(trainer, mock_ltm, tmp_path):
    """Test that metrics are saved to CSV."""
    # Add more data
    from tests.fixtures.structures import bcc_structure
    mock_ltm.extend(
        [bcc_structure()] * 15,
        np.random.rand(15),
        step=1,
        property_values=np.random.rand(15) * 100,
        features=[np.random.randn(10) for _ in range(15)]
    )

    trainer.retrain(mock_ltm, current_step=5)
    trainer.retrain(mock_ltm, current_step=10)

    # Check metrics file exists
    metrics_file = tmp_path / 'gp_metrics' / 'gp_training_metrics.csv'
    assert metrics_file.exists()

    # Load and verify
    df = pd.read_csv(metrics_file)
    assert len(df) == 2  # Two retraining events
    assert 'step' in df.columns
    assert list(df['step']) == [5, 10]


def test_evaluate_computes_metrics(trainer, seed):
    """Test that evaluation computes all required metrics."""
    # Create synthetic data
    np.random.seed(seed)
    X = np.random.randn(20, 10)
    y_true = np.random.randn(20) * 20 + 150

    # Train GP
    trainer.gp_model.fit(X, y_true)

    # Evaluate
    metrics = trainer._evaluate(X, y_true, prefix='test')

    assert 'test_RMSE' in metrics
    assert 'test_MAE' in metrics
    assert 'test_R2' in metrics
    assert 'test_SMAPE' in metrics
    assert 'test_Spearman' in metrics


def test_multiple_retraining_cycles(trainer, mock_ltm):
    """Test multiple retraining cycles."""
    from tests.fixtures.structures import bcc_structure

    for step in [0, 5, 10, 15]:
        # Add new data each cycle
        mock_ltm.extend(
            [bcc_structure()] * 5,
            np.random.rand(5),
            step=step,
            property_values=np.random.rand(5) * 100,
            features=[np.random.randn(10) for _ in range(5)]
        )

        if trainer.should_retrain(step):
            metrics = trainer.retrain(mock_ltm, current_step=step)

            if metrics:  # If training occurred
                assert metrics['step'] == step

    # Should have multiple entries in history
    assert len(trainer.training_history) >= 1


def test_trainer_validation_split_parameter():
    """Test different validation split values."""
    gp_model = GPSurrogate(input_dim=10, task='test', device='cpu')
    featurizer = ORBFeaturizer(n_components=10, device='cpu')

    for val_split in [0.1, 0.2, 0.3]:
        trainer = GPTrainingManager(
            gp_model=gp_model,
            featurizer=featurizer,
            validation_split=val_split
        )
        assert trainer.validation_split == val_split


def test_collect_data_missing_property_column(trainer):
    """Test data collection when property column is missing."""
    ltm = LongTimeMem()

    # Add data without property_value column
    from tests.fixtures.structures import fcc_structure
    ltm.extend(
        [fcc_structure()] * 10,
        np.random.rand(10),
        step=0
        # No property_values provided
    )

    X, y = trainer.collect_training_data(ltm, 'property_value')

    # Should return None when property values are missing/null
    assert X is None or y is None
