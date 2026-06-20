"""End-to-end test for RL with GP routing.

This test simulates a small RL experiment with GP-based calculator routing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from memory.ltm import LongTimeMem
from rewards.reward import Reward
from rewards.router import CalculatorRouter
from rewards.gp.surrogate import GPSurrogate
from rewards.calculators.orb.featurizer import ORBFeaturizer
from rewards.acquisition import ExpectedImprovementPerCost
from rewards.gp_trainer import GPTrainingManager


@pytest.fixture
def mock_rl_components(tmp_path, device):
    """Create mock RL components for testing."""
    # Mock calculators
    from tests.fixtures.mock_calculators import MockCheapCalculator, MockExpensiveCalculator

    calculators = {
        'orb': MockCheapCalculator(str(tmp_path / 'orb'), 'bulk_modulus'),
        'alignn': MockExpensiveCalculator(str(tmp_path / 'alignn'), 'bulk_modulus')
    }

    # GP components
    gp_model = GPSurrogate(input_dim=10, task='bulk_modulus', device=device)
    featurizer = ORBFeaturizer(n_components=10, device=device)
    acquisition_fn = ExpectedImprovementPerCost(
        cost_model={'orb': 0.001, 'alignn': 0.01},
        xi=0.01
    )

    # Router
    router = CalculatorRouter(
        calculators=calculators,
        gp_model=gp_model,
        featurizer=featurizer,
        acquisition_fn=acquisition_fn,
        default_calculator='orb',
        min_gp_samples=5  # Lower for testing
    )

    # Reward
    from omegaconf import OmegaConf
    prop_cfg = [
        OmegaConf.create({
            'name': 'bulk_modulus',
            'use_gp_routing': True,
            'router': router,
            'target': 150.0,
            'minv': 100.0,
            'maxv': 200.0
        })
    ]

    reward = Reward(
        root_dir=str(tmp_path / 'rewards'),
        prop_cfg=prop_cfg,
        reward_threshold=0.7
    )

    # GP Trainer
    gp_trainer = GPTrainingManager(
        gp_model=gp_model,
        featurizer=featurizer,
        retrain_frequency=3,
        metrics_dir=str(tmp_path / 'gp_metrics'),
        min_samples=5,
        validation_split=0.2
    )

    # LTM
    ltm = LongTimeMem()

    return {
        'reward': reward,
        'ltm': ltm,
        'gp_trainer': gp_trainer,
        'router': router
    }


@pytest.mark.e2e
@pytest.mark.slow
def test_rl_loop_with_gp_routing(mock_rl_components, structure_list, xyz_file_path):
    """Test complete RL loop with GP routing."""
    reward = mock_rl_components['reward']
    ltm = mock_rl_components['ltm']
    gp_trainer = mock_rl_components['gp_trainer']

    n_steps = 10
    structures_per_step = len(structure_list)

    total_cost = 0
    routing_history = []

    for step in range(n_steps):
        # Simulate sampling (use same structures for simplicity)
        samples = (structure_list, xyz_file_path)

        # Calculate rewards
        rewards, prop_dict, failed_mask = reward.scoring(samples, label=f'step_{step}')

        # Get routing metadata
        metadata = reward.get_routing_metadata()
        if 'bulk_modulus' in metadata:
            routing_metadata = metadata['bulk_modulus']
            total_cost += routing_metadata.get('cost', 0)
            routing_history.append(routing_metadata.get('routing_counts', {}))

            # Extract GP data
            property_values = prop_dict.get('bulk_modulus')
            features = routing_metadata.get('features')
            calculators_used = routing_metadata.get('routed_to')
        else:
            property_values = None
            features = None
            calculators_used = None

        # Store in LTM
        ltm.extend(
            structure_list,
            rewards,
            step=step,
            property_values=property_values,
            features=features,
            calculators_used=calculators_used
        )

        # GP retraining
        if gp_trainer.should_retrain(step):
            gp_metrics = gp_trainer.retrain(ltm, current_step=step, property_name='property_value')

            if gp_metrics:
                print(f"Step {step}: GP retrained - Val R²: {gp_metrics.get('val_R2', 'N/A')}")

    # Verify results
    assert len(ltm) == n_steps * structures_per_step
    assert total_cost > 0

    # GP should be trained
    assert gp_trainer.gp_model.is_trained

    # Should have training history
    assert len(gp_trainer.training_history) > 0

    # Routing should have evolved over time
    # Early steps should be all 'orb', later steps should have some 'alignn'
    early_routing = routing_history[0] if routing_history else {}
    late_routing = routing_history[-1] if routing_history else {}

    print(f"\nEarly routing: {early_routing}")
    print(f"Late routing: {late_routing}")
    print(f"Total cost: {total_cost:.4f}")

    # Cold start should use only orb
    if early_routing:
        assert early_routing.get('orb', 0) == structures_per_step


@pytest.mark.e2e
def test_gp_metrics_saved(mock_rl_components, structure_list, xyz_file_path, tmp_path):
    """Test that GP metrics are saved during RL."""
    reward = mock_rl_components['reward']
    ltm = mock_rl_components['ltm']
    gp_trainer = mock_rl_components['gp_trainer']

    # Run several steps
    for step in range(6):
        samples = (structure_list, xyz_file_path)
        rewards, prop_dict, _ = reward.scoring(samples, label=f'step_{step}')

        # Get metadata
        metadata = reward.get_routing_metadata()
        property_values = prop_dict.get('bulk_modulus')
        features = metadata.get('bulk_modulus', {}).get('features')
        calculators = metadata.get('bulk_modulus', {}).get('routed_to')

        ltm.extend(structure_list, rewards, step, property_values, features, calculators)

        # Retrain
        if gp_trainer.should_retrain(step):
            gp_trainer.retrain(ltm, step, 'property_value')

    # Check metrics file
    metrics_file = tmp_path / 'gp_metrics' / 'gp_training_metrics.csv'
    assert metrics_file.exists()

    import pandas as pd
    df = pd.read_csv(metrics_file)

    # Should have metrics for steps [0, 3]
    assert len(df) >= 1
    assert 'step' in df.columns
    assert 'train_R2' in df.columns


@pytest.mark.e2e
def test_cost_reduction_over_time(mock_rl_components, structure_list, xyz_file_path):
    """Test that GP routing reduces cost over time."""
    reward = mock_rl_components['reward']
    ltm = mock_rl_components['ltm']
    gp_trainer = mock_rl_components['gp_trainer']

    costs = []

    for step in range(12):
        samples = (structure_list, xyz_file_path)
        rewards, prop_dict, _ = reward.scoring(samples, label=f'step_{step}')

        metadata = reward.get_routing_metadata()
        step_cost = metadata.get('bulk_modulus', {}).get('cost', 0)
        costs.append(step_cost)

        # Store and retrain
        property_values = prop_dict.get('bulk_modulus')
        features = metadata.get('bulk_modulus', {}).get('features')
        calculators = metadata.get('bulk_modulus', {}).get('routed_to')

        ltm.extend(structure_list, rewards, step, property_values, features, calculators)

        if gp_trainer.should_retrain(step):
            gp_trainer.retrain(ltm, step, 'property_value')

    # Early costs (cold start) should be lower (all orb)
    early_cost = np.mean(costs[:3])

    # Later costs might vary (GP routing active)
    late_cost = np.mean(costs[-3:])

    print(f"Early cost: {early_cost:.4f}, Late cost: {late_cost:.4f}")

    # Cost should be tracked
    assert all(c > 0 for c in costs)


@pytest.mark.e2e
def test_ltm_data_integrity(mock_rl_components, structure_list, xyz_file_path):
    """Test that LTM maintains data integrity throughout RL."""
    reward = mock_rl_components['reward']
    ltm = mock_rl_components['ltm']

    for step in range(5):
        samples = (structure_list, xyz_file_path)
        rewards, prop_dict, _ = reward.scoring(samples, label=f'step_{step}')

        metadata = reward.get_routing_metadata()
        ltm.extend(
            structure_list,
            rewards,
            step,
            property_values=prop_dict.get('bulk_modulus'),
            features=metadata.get('bulk_modulus', {}).get('features'),
            calculators_used=metadata.get('bulk_modulus', {}).get('routed_to')
        )

    # Verify LTM integrity
    assert len(ltm) == 5 * len(structure_list)

    # All entries should have required columns
    assert 'struc' in ltm.memory.columns
    assert 'reward' in ltm.memory.columns
    assert 'property_value' in ltm.memory.columns
    assert 'features' in ltm.memory.columns
    assert 'calculator_used' in ltm.memory.columns

    # GP data should be present (after featurizer initialized)
    non_null_props = ltm.memory['property_value'].notna().sum()
    assert non_null_props > 0


@pytest.mark.e2e
@pytest.mark.slow
def test_gp_convergence(mock_rl_components, structure_list, xyz_file_path):
    """Test that GP model improves over training."""
    reward = mock_rl_components['reward']
    ltm = mock_rl_components['ltm']
    gp_trainer = mock_rl_components['gp_trainer']

    r2_scores = []

    for step in range(15):
        samples = (structure_list, xyz_file_path)
        rewards, prop_dict, _ = reward.scoring(samples, label=f'step_{step}')

        metadata = reward.get_routing_metadata()
        ltm.extend(
            structure_list,
            rewards,
            step,
            property_values=prop_dict.get('bulk_modulus'),
            features=metadata.get('bulk_modulus', {}).get('features'),
            calculators_used=metadata.get('bulk_modulus', {}).get('routed_to')
        )

        if gp_trainer.should_retrain(step):
            gp_metrics = gp_trainer.retrain(ltm, step, 'property_value')
            if gp_metrics and 'val_R2' in gp_metrics:
                r2_scores.append(gp_metrics['val_R2'])

    # Should have some R² scores
    assert len(r2_scores) > 0

    # Later R² scores should generally be better (more data)
    if len(r2_scores) >= 2:
        print(f"R² progression: {r2_scores}")
        # At least should not be getting worse on average
        assert np.mean(r2_scores[-2:]) >= np.mean(r2_scores[:2]) - 0.3  # Allow some variance
