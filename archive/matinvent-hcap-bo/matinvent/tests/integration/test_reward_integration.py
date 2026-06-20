"""Integration tests for reward system with GP routing."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from omegaconf import OmegaConf

from rewards.reward import Reward
from rewards.router import CalculatorRouter
from rewards.gp.surrogate import GPSurrogate
from rewards.calculators.orb.featurizer import ORBFeaturizer
from rewards.acquisition import ExpectedImprovementPerCost


@pytest.fixture
def gp_routing_config(tmp_path, device, mock_cost_model):
    """Create GP routing configuration."""
    config = OmegaConf.create({
        'root_dir': str(tmp_path / 'rewards'),
        'prop_cfg': [
            {
                'name': 'bulk_modulus',
                'use_gp_routing': True,
                'router': {
                    'calculators': {
                        'orb': Mock(),
                        'alignn': Mock()
                    },
                    'gp_model': GPSurrogate(input_dim=10, task='bulk_modulus', device=device),
                    'featurizer': ORBFeaturizer(n_components=10, device=device),
                    'acquisition_fn': ExpectedImprovementPerCost(cost_model=mock_cost_model),
                    'default_calculator': 'orb',
                    'min_gp_samples': 10
                },
                'target': 300.0,
                'minv': 0.0,
                'maxv': 250.0
            }
        ],
        'reward_threshold': 0.8,
        'reduce': 'mean'
    })
    return config


@pytest.fixture
def reward_with_routing(gp_routing_config, tmp_path, device, mock_calculator_dict):
    """Create Reward instance with GP routing."""
    # Setup router with mock calculators
    router = CalculatorRouter(
        calculators=mock_calculator_dict,
        gp_model=GPSurrogate(input_dim=10, task='bulk_modulus', device=device),
        featurizer=ORBFeaturizer(n_components=10, device=device),
        acquisition_fn=ExpectedImprovementPerCost(cost_model={'orb': 0.001, 'alignn': 0.01, 'vasp': 1.0}),
        default_calculator='orb',
        min_gp_samples=10
    )

    # Create reward with routing
    prop_cfg = [
        OmegaConf.create({
            'name': 'bulk_modulus',
            'use_gp_routing': True,
            'router': router,
            'target': 300.0,
            'minv': 0.0,
            'maxv': 250.0
        })
    ]

    reward = Reward(
        root_dir=str(tmp_path / 'rewards'),
        prop_cfg=prop_cfg,
        reward_threshold=0.8
    )

    return reward


@pytest.mark.integration
def test_reward_calc_props_with_routing(reward_with_routing, structure_list, xyz_file_path):
    """Test reward calculation with GP routing."""
    samples = (structure_list, xyz_file_path)

    prop_dict, failed_mask = reward_with_routing.calc_props(samples, label='test')

    # Should return property values
    assert 'bulk_modulus' in prop_dict
    assert len(prop_dict['bulk_modulus']) == len(structure_list)
    assert len(failed_mask) == len(structure_list)


@pytest.mark.integration
def test_reward_get_routing_metadata(reward_with_routing, structure_list, xyz_file_path):
    """Test that routing metadata is stored and retrievable."""
    samples = (structure_list, xyz_file_path)

    reward_with_routing.calc_props(samples, label='test')

    metadata = reward_with_routing.get_routing_metadata()

    assert 'bulk_modulus' in metadata
    assert 'routed_to' in metadata['bulk_modulus']
    assert 'routing_counts' in metadata['bulk_modulus']
    assert 'cost' in metadata['bulk_modulus']


@pytest.mark.integration
def test_reward_scoring_with_routing(reward_with_routing, structure_list, xyz_file_path):
    """Test full scoring pipeline with GP routing."""
    samples = (structure_list, xyz_file_path)

    rewards, prop_dict, failed_mask = reward_with_routing.scoring(samples, label='test')

    # Should return rewards
    assert len(rewards) == len(structure_list)
    assert np.all((rewards >= 0) & (rewards <= 1))  # Rewards should be scaled

    # Property dict should have values
    assert 'bulk_modulus' in prop_dict

    # Metadata should be available
    metadata = reward_with_routing.get_routing_metadata()
    assert metadata is not None


@pytest.mark.integration
def test_reward_routing_cold_start(reward_with_routing, structure_list, xyz_file_path):
    """Test routing behavior before GP is trained (cold start)."""
    samples = (structure_list, xyz_file_path)

    # Before GP training, should use default calculator
    prop_dict, failed_mask = reward_with_routing.calc_props(samples, label='cold_start')

    metadata = reward_with_routing.get_routing_metadata()

    # All should route to default calculator
    assert all(calc == 'orb' for calc in metadata['bulk_modulus']['routed_to'])


@pytest.mark.integration
def test_reward_routing_after_gp_training(reward_with_routing, structure_list, xyz_file_path):
    """Test routing behavior after GP is trained."""
    # Train GP first
    router = reward_with_routing.prop_cfg[0].router
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20) * 20 + 150
    router.gp_model.fit(X_train, y_train)

    samples = (structure_list, xyz_file_path)
    prop_dict, failed_mask = reward_with_routing.calc_props(samples, label='after_training')

    metadata = reward_with_routing.get_routing_metadata()

    # Should have GP predictions
    assert metadata['bulk_modulus']['gp_predictions'] is not None
    assert metadata['bulk_modulus']['uncertainties'] is not None


@pytest.mark.integration
def test_reward_without_routing(tmp_path, structure_list, xyz_file_path):
    """Test traditional reward calculation without GP routing."""
    # Create reward without routing
    from tests.fixtures.mock_calculators import MockCalculator

    prop_cfg = [
        OmegaConf.create({
            'name': 'bulk_modulus',
            'calculator': MockCalculator(str(tmp_path / 'alignn'), 'bulk_modulus'),
            'target': 300.0,
            'minv': 0.0,
            'maxv': 250.0
        })
    ]

    reward = Reward(
        root_dir=str(tmp_path / 'rewards'),
        prop_cfg=prop_cfg,
        reward_threshold=0.8
    )

    samples = (structure_list, xyz_file_path)
    rewards, prop_dict, failed_mask = reward.scoring(samples, label='no_routing')

    # Should work without routing
    assert len(rewards) == len(structure_list)
    assert 'bulk_modulus' in prop_dict


@pytest.mark.integration
def test_reward_multiple_properties(tmp_path, device, structure_list, xyz_file_path):
    """Test reward with multiple properties (some routed, some not)."""
    from tests.fixtures.mock_calculators import MockCalculator

    # One property with routing, one without
    router = CalculatorRouter(
        calculators={'orb': MockCalculator(str(tmp_path / 'orb'), 'bulk_modulus')},
        gp_model=GPSurrogate(input_dim=10, task='bulk_modulus', device=device),
        featurizer=ORBFeaturizer(n_components=10, device=device),
        acquisition_fn=ExpectedImprovementPerCost(cost_model={'orb': 0.001}),
        default_calculator='orb'
    )

    prop_cfg = [
        OmegaConf.create({
            'name': 'bulk_modulus',
            'use_gp_routing': True,
            'router': router,
            'target': 300.0,
            'minv': 0.0,
            'maxv': 250.0,
            'weight': 0.5
        }),
        OmegaConf.create({
            'name': 'formation_energy',
            'calculator': MockCalculator(str(tmp_path / 'form_e'), 'formation_energy', mean_value=-2.0),
            'target': 'descending',
            'minv': -3.5,
            'maxv': -1.0,
            'weight': 0.5
        })
    ]

    reward = Reward(
        root_dir=str(tmp_path / 'rewards'),
        prop_cfg=prop_cfg,
        reward_threshold=0.8,
        reduce='weight'
    )

    samples = (structure_list, xyz_file_path)
    rewards, prop_dict, failed_mask = reward.scoring(samples, label='multi_prop')

    # Should calculate both properties
    assert 'bulk_modulus' in prop_dict
    assert 'formation_energy' in prop_dict

    # Should have metadata for routed property only
    metadata = reward.get_routing_metadata()
    assert 'bulk_modulus' in metadata
    assert 'formation_energy' not in metadata  # Not routed


@pytest.mark.integration
def test_reward_cost_tracking(reward_with_routing, structure_list, xyz_file_path):
    """Test that computational cost is tracked correctly."""
    samples = (structure_list, xyz_file_path)

    reward_with_routing.calc_props(samples, label='cost_tracking')

    metadata = reward_with_routing.get_routing_metadata()

    # Should track cost
    assert 'cost' in metadata['bulk_modulus']
    assert metadata['bulk_modulus']['cost'] > 0


@pytest.mark.integration
def test_reward_features_cached(reward_with_routing, structure_list, xyz_file_path):
    """Test that ORB features are computed and returned."""
    samples = (structure_list, xyz_file_path)

    reward_with_routing.calc_props(samples, label='features')

    metadata = reward_with_routing.get_routing_metadata()

    # Features should be in metadata
    assert 'features' in metadata['bulk_modulus']
    features = metadata['bulk_modulus']['features']

    if features is not None:  # After GP training
        assert features.shape == (len(structure_list), 10)
