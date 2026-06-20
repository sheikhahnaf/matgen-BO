"""Unit tests for GP surrogate model."""

import pytest
import numpy as np
import torch

from rewards.gp.surrogate import GPSurrogate


@pytest.fixture
def gp_model(device):
    """Create GP surrogate instance."""
    return GPSurrogate(input_dim=10, task='bulk_modulus', device=device)


@pytest.fixture
def training_data(seed):
    """Generate synthetic training data."""
    np.random.seed(seed)
    n_samples = 50
    X = np.random.randn(n_samples, 10)
    y = 150 + 20 * X[:, 0] - 10 * X[:, 1] + np.random.randn(n_samples) * 5
    return X, y


@pytest.fixture
def test_data(seed):
    """Generate synthetic test data."""
    np.random.seed(seed + 1)
    n_samples = 20
    X = np.random.randn(n_samples, 10)
    y = 150 + 20 * X[:, 0] - 10 * X[:, 1] + np.random.randn(n_samples) * 5
    return X, y


def test_gp_initialization(gp_model):
    """Test GP model initializes correctly."""
    assert gp_model.input_dim == 10
    assert gp_model.task == 'bulk_modulus'
    assert gp_model.device == 'cpu'
    assert not gp_model.is_trained
    assert gp_model.model is None


def test_gp_fit(gp_model, training_data):
    """Test GP model fitting."""
    X, y = training_data
    gp_model.fit(X, y)

    assert gp_model.is_trained
    assert gp_model.model is not None
    assert gp_model.feature_scaler is not None
    assert gp_model.target_scaler is not None


def test_gp_predict(gp_model, training_data, test_data):
    """Test GP prediction."""
    X_train, y_train = training_data
    X_test, y_test = test_data

    gp_model.fit(X_train, y_train)
    mean, std = gp_model.predict(X_test, return_std=True)

    assert mean.shape == (len(X_test),)
    assert std.shape == (len(X_test),)
    assert np.all(std > 0)  # Uncertainties should be positive
    assert not np.any(np.isnan(mean))
    assert not np.any(np.isnan(std))


def test_gp_predict_without_std(gp_model, training_data, test_data):
    """Test GP prediction without uncertainty."""
    X_train, y_train = training_data
    X_test, y_test = test_data

    gp_model.fit(X_train, y_train)
    mean = gp_model.predict(X_test, return_std=False)

    assert mean.shape == (len(X_test),)
    assert not np.any(np.isnan(mean))


def test_gp_prediction_accuracy(gp_model, training_data, test_data):
    """Test that GP predictions are reasonably accurate."""
    X_train, y_train = training_data
    X_test, y_test = test_data

    gp_model.fit(X_train, y_train)
    mean, std = gp_model.predict(X_test, return_std=True)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((mean - y_test) ** 2))

    # Should be better than just predicting the mean
    baseline_rmse = np.sqrt(np.mean((y_train.mean() - y_test) ** 2))
    assert rmse < baseline_rmse


def test_gp_uncertainty_calibration(gp_model, training_data):
    """Test that uncertainty is higher for points far from training data."""
    X_train, y_train = training_data
    gp_model.fit(X_train, y_train)

    # Test on training data (should have low uncertainty)
    mean_train, std_train = gp_model.predict(X_train[:10], return_std=True)

    # Test on far-away points (should have higher uncertainty)
    X_far = X_train[:10] + 10 * np.random.randn(10, 10)
    mean_far, std_far = gp_model.predict(X_far, return_std=True)

    # Far points should generally have higher uncertainty
    assert std_far.mean() > std_train.mean()


def test_gp_get_training_data_size(gp_model, training_data):
    """Test getting training data size."""
    X, y = training_data

    assert gp_model.get_training_data_size() == 0  # Before training

    gp_model.fit(X, y)
    assert gp_model.get_training_data_size() == len(X)


def test_gp_save_load(gp_model, training_data, tmp_path):
    """Test saving and loading GP model."""
    X, y = training_data
    gp_model.fit(X, y)

    # Make predictions before saving
    mean_before, std_before = gp_model.predict(X[:5], return_std=True)

    # Save model
    save_path = tmp_path / "gp_model.pt"
    gp_model.save(str(save_path))

    # Load into new model
    gp_model_loaded = GPSurrogate(input_dim=10, task='bulk_modulus', device='cpu')
    gp_model_loaded.load(str(save_path))

    # Predictions should be identical
    mean_after, std_after = gp_model_loaded.predict(X[:5], return_std=True)

    np.testing.assert_array_almost_equal(mean_before, mean_after, decimal=5)
    np.testing.assert_array_almost_equal(std_before, std_after, decimal=5)


def test_gp_scaling(gp_model, training_data):
    """Test that scaling is applied correctly."""
    X, y = training_data
    gp_model.fit(X, y)

    # Features should be standardized (mean ~0, std ~1)
    assert hasattr(gp_model.feature_scaler, 'mean_')
    assert hasattr(gp_model.target_scaler, 'mean_')


def test_gp_retrain(gp_model, training_data):
    """Test retraining GP with new data."""
    X1, y1 = training_data
    gp_model.fit(X1, y1)

    size_before = gp_model.get_training_data_size()

    # Generate new data
    X2 = np.random.randn(30, 10)
    y2 = 150 + 20 * X2[:, 0] - 10 * X2[:, 1] + np.random.randn(30) * 5

    # Retrain (should replace old data in current implementation)
    gp_model.fit(X2, y2)

    size_after = gp_model.get_training_data_size()
    assert size_after == len(X2)


@pytest.mark.parametrize("n_samples,n_features", [(10, 5), (50, 10), (100, 20)])
def test_gp_different_sizes(device, n_samples, n_features, seed):
    """Test GP with different data sizes."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    gp_model = GPSurrogate(input_dim=n_features, task='test', device=device)
    gp_model.fit(X, y)

    mean, std = gp_model.predict(X[:5], return_std=True)
    assert mean.shape == (5,)
    assert std.shape == (5,)
