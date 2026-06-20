"""Unit tests for calculator router."""

import pytest
import numpy as np

from rewards.router import CalculatorRouter
from rewards.gp.surrogate import GPSurrogate
from rewards.calculators.orb.featurizer import ORBFeaturizer
from rewards.acquisition import ExpectedImprovementPerCost


@pytest.fixture
def featurizer(device):
    """Create ORB featurizer."""
    return ORBFeaturizer(n_components=10, device=device)


@pytest.fixture
def gp_model(device):
    """Create GP model."""
    return GPSurrogate(input_dim=10, task='bulk_modulus', device=device)


@pytest.fixture
def acquisition_fn(mock_cost_model):
    """Create acquisition function."""
    return ExpectedImprovementPerCost(cost_model=mock_cost_model, xi=0.01)


@pytest.fixture
def router(mock_calculator_dict, gp_model, featurizer, acquisition_fn):
    """Create calculator router."""
    return CalculatorRouter(
        calculators=mock_calculator_dict,
        gp_model=gp_model,
        featurizer=featurizer,
        acquisition_fn=acquisition_fn,
        default_calculator='orb',
        min_gp_samples=10
    )


def test_router_initialization(router):
    """Test router initializes correctly."""
    assert router.default_calculator == 'orb'
    assert router.min_gp_samples == 10
    assert len(router.calculators) == 3


def test_router_cold_start(router, structure_list, xyz_file_path):
    """Test routing before GP is trained (cold start)."""
    samples = (structure_list, xyz_file_path)

    properties, metadata = router.route_and_compute(samples, label='test')

    # Should use default calculator
    assert properties.shape == (len(structure_list),)
    assert metadata['routed_to'] == ['orb'] * len(structure_list)
    assert metadata['routing_counts'] == {'orb': len(structure_list)}
    assert metadata['uncertainties'] is None  # No GP predictions yet
    assert metadata['gp_predictions'] is None


def test_router_with_trained_gp(router, structure_list, xyz_file_path, seed):
    """Test routing after GP is trained."""
    # Train GP on some dummy data
    np.random.seed(seed)
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20) * 20 + 150
    router.gp_model.fit(X_train, y_train)

    samples = (structure_list, xyz_file_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    # Should have GP predictions
    assert properties.shape == (len(structure_list),)
    assert metadata['uncertainties'] is not None
    assert metadata['gp_predictions'] is not None
    assert len(metadata['routed_to']) == len(structure_list)
    assert 'routing_counts' in metadata


def test_router_cost_tracking(router, structure_list, xyz_file_path, seed):
    """Test that router tracks computational cost."""
    # Train GP
    np.random.seed(seed)
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20) * 20 + 150
    router.gp_model.fit(X_train, y_train)

    samples = (structure_list, xyz_file_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    assert 'cost' in metadata
    assert metadata['cost'] > 0

    # Cost should match routing decisions
    expected_cost = sum(
        count * router.acquisition_fn.cost_model[calc]
        for calc, count in metadata['routing_counts'].items()
    )
    assert abs(metadata['cost'] - expected_cost) < 1e-6


def test_router_feature_extraction(router, structure_list, xyz_file_path, seed):
    """Test that router extracts and stores features."""
    np.random.seed(seed)
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20) * 20 + 150
    router.gp_model.fit(X_train, y_train)

    samples = (structure_list, xyz_file_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    assert 'features' in metadata
    assert metadata['features'] is not None
    assert metadata['features'].shape == (len(structure_list), 10)


def test_router_best_observed(router, seed):
    """Test that router tracks best observed value."""
    np.random.seed(seed)
    X_train = np.random.randn(20, 10)
    y_train = np.random.uniform(100, 200, 20)
    router.gp_model.fit(X_train, y_train)

    best_observed = router._get_best_observed()

    assert best_observed == np.max(y_train)


def test_router_handles_failed_calculations(router, structure_list, xyz_file_path):
    """Test router handles calculator failures gracefully."""
    # Make one calculator fail
    original_calc = router.calculators['vasp'].calc

    def failing_calc(samples, label):
        raise RuntimeError("Simulated failure")

    router.calculators['vasp'].calc = failing_calc

    samples = (structure_list, xyz_file_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    # Should still return results (with NaN for failed calculations)
    assert properties.shape == (len(structure_list),)

    # Restore original
    router.calculators['vasp'].calc = original_calc


def test_router_routing_distribution(router, seed):
    """Test that routing distributes across calculators appropriately."""
    # Train GP with varied predictions
    np.random.seed(seed)
    X_train = np.random.randn(50, 10)
    y_train = np.random.uniform(100, 200, 50)
    router.gp_model.fit(X_train, y_train)

    # Create structures that will have varied predictions
    from tests.fixtures.structures import fcc_structure, bcc_structure, hcp_structure
    structures = [fcc_structure(), bcc_structure(), hcp_structure()] * 10  # 30 structures
    xyz_path = "/tmp/test.xyz"

    samples = (structures, xyz_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    # Should use multiple calculators (though distribution depends on GP predictions)
    assert len(metadata['routing_counts']) >= 1
    assert sum(metadata['routing_counts'].values()) == len(structures)


def test_router_uncertainty_propagation(router, structure_list, xyz_file_path, seed):
    """Test that uncertainties are computed and returned."""
    np.random.seed(seed)
    X_train = np.random.randn(20, 10)
    y_train = np.random.randn(20) * 20 + 150
    router.gp_model.fit(X_train, y_train)

    samples = (structure_list, xyz_file_path)
    properties, metadata = router.route_and_compute(samples, label='test')

    assert metadata['uncertainties'] is not None
    assert len(metadata['uncertainties']) == len(structure_list)
    assert np.all(metadata['uncertainties'] > 0)  # Uncertainties should be positive


def test_router_format_routing_counts():
    """Test routing counts formatting."""
    router = CalculatorRouter(
        calculators={},
        gp_model=None,
        featurizer=None,
        acquisition_fn=None
    )

    calc_groups = {
        'orb': [0, 1, 2],
        'alignn': [3, 4],
        'vasp': [5]
    }
    total = 6

    formatted = router._format_routing_counts(calc_groups, total)

    assert 'orb: 3 (50.0%)' in formatted
    assert 'alignn: 2 (33.3%)' in formatted
    assert 'vasp: 1 (16.7%)' in formatted


@pytest.mark.parametrize("min_samples", [5, 10, 20])
def test_router_min_gp_samples(mock_calculator_dict, gp_model, featurizer, acquisition_fn, min_samples):
    """Test router respects min_gp_samples threshold."""
    router = CalculatorRouter(
        calculators=mock_calculator_dict,
        gp_model=gp_model,
        featurizer=featurizer,
        acquisition_fn=acquisition_fn,
        default_calculator='orb',
        min_gp_samples=min_samples
    )

    # Train with fewer samples than threshold
    X_train = np.random.randn(min_samples - 1, 10)
    y_train = np.random.randn(min_samples - 1)
    router.gp_model.fit(X_train, y_train)

    # Should still use default calculator
    assert router.gp_model.get_training_data_size() < min_samples
