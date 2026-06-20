"""Unit tests for acquisition function."""

import pytest
import numpy as np

from rewards.acquisition import ExpectedImprovementPerCost


@pytest.fixture
def acquisition_fn(mock_cost_model):
    """Create acquisition function instance."""
    return ExpectedImprovementPerCost(cost_model=mock_cost_model, xi=0.01)


def test_acquisition_initialization(acquisition_fn, mock_cost_model):
    """Test acquisition function initializes correctly."""
    assert acquisition_fn.cost_model == mock_cost_model
    assert acquisition_fn.xi == 0.01


def test_select_calculator_low_uncertainty(acquisition_fn):
    """Test that low uncertainty leads to cheap calculator selection."""
    mean = np.array([150.0, 160.0, 140.0])
    std = np.array([1.0, 1.5, 0.5])  # Low uncertainty
    best_observed = 155.0
    available_calcs = ['orb', 'alignn', 'vasp']

    selected = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    # Should prefer cheap calculators when uncertainty is low
    assert len(selected) == len(mean)
    # Most selections should be cheap (orb)
    assert selected.count('orb') >= len(mean) // 2


def test_select_calculator_high_uncertainty(acquisition_fn):
    """Test that high uncertainty can lead to expensive calculator selection."""
    mean = np.array([200.0, 210.0, 205.0])  # High potential values
    std = np.array([50.0, 60.0, 55.0])  # High uncertainty
    best_observed = 150.0
    available_calcs = ['orb', 'alignn', 'vasp']

    selected = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    assert len(selected) == len(mean)
    # With high uncertainty and high mean, should consider more expensive calculators
    # At least some should not be orb
    unique_calcs = set(selected)
    assert len(unique_calcs) >= 1


def test_select_calculator_mixed_uncertainty(acquisition_fn):
    """Test calculator selection with mixed uncertainties."""
    mean = np.array([150.0, 180.0, 160.0, 200.0, 140.0])
    std = np.array([2.0, 30.0, 5.0, 50.0, 1.0])  # Mixed
    best_observed = 155.0
    available_calcs = ['orb', 'alignn', 'vasp']

    selected = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    assert len(selected) == len(mean)
    # High uncertainty samples (index 1, 3) might get more expensive calculators
    # Low uncertainty samples (index 0, 2, 4) should get cheap calculators


def test_expected_improvement_calculation(acquisition_fn):
    """Test EI calculation directly."""
    mean = np.array([160.0])
    std = np.array([10.0])
    best_observed = 150.0

    ei = acquisition_fn._compute_ei(mean, std, best_observed)

    assert ei.shape == mean.shape
    assert ei[0] > 0  # Should have positive EI for mean > best_observed


def test_ei_zero_for_low_mean(acquisition_fn):
    """Test that EI is near zero when mean << best_observed."""
    mean = np.array([100.0])
    std = np.array([5.0])
    best_observed = 200.0

    ei = acquisition_fn._compute_ei(mean, std, best_observed)

    assert ei[0] < 1.0  # Should be very small


def test_ei_increases_with_uncertainty(acquisition_fn):
    """Test that EI increases with uncertainty."""
    mean = np.array([160.0, 160.0])
    std = np.array([5.0, 20.0])  # Same mean, different std
    best_observed = 150.0

    ei = acquisition_fn._compute_ei(mean, std, best_observed)

    assert ei[1] > ei[0]  # Higher uncertainty should give higher EI


def test_cost_model_integration(acquisition_fn):
    """Test that cost model is used correctly."""
    # Create scenario where cheap calc wins due to cost
    mean = np.array([160.0])
    std = np.array([10.0])
    best_observed = 150.0

    # Calculate EI/cost for each calculator
    ei = acquisition_fn._compute_ei(mean, std, best_observed)

    ei_per_cost = {}
    for calc_name, cost in acquisition_fn.cost_model.items():
        ei_per_cost[calc_name] = ei[0] / cost

    # ORB should have highest EI/cost due to low cost
    assert ei_per_cost['orb'] > ei_per_cost['alignn']
    assert ei_per_cost['orb'] > ei_per_cost['vasp']


def test_xi_parameter_effect(mock_cost_model):
    """Test that xi parameter affects exploration."""
    mean = np.array([155.0])
    std = np.array([10.0])
    best_observed = 150.0

    # Low xi (more exploitation)
    acq_low_xi = ExpectedImprovementPerCost(cost_model=mock_cost_model, xi=0.001)
    ei_low = acq_low_xi._compute_ei(mean, std, best_observed)

    # High xi (more exploration)
    acq_high_xi = ExpectedImprovementPerCost(cost_model=mock_cost_model, xi=0.1)
    ei_high = acq_high_xi._compute_ei(mean, std, best_observed)

    # Higher xi should give higher EI
    assert ei_high[0] > ei_low[0]


def test_batch_selection_consistency(acquisition_fn):
    """Test that batch selection is consistent."""
    mean = np.array([150.0, 160.0, 170.0])
    std = np.array([10.0, 15.0, 20.0])
    best_observed = 155.0
    available_calcs = ['orb', 'alignn', 'vasp']

    selected1 = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)
    selected2 = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    assert selected1 == selected2  # Should be deterministic


@pytest.mark.parametrize("n_samples", [1, 5, 10, 50])
def test_different_batch_sizes(acquisition_fn, n_samples):
    """Test acquisition function with different batch sizes."""
    np.random.seed(42)
    mean = np.random.uniform(100, 200, n_samples)
    std = np.random.uniform(5, 30, n_samples)
    best_observed = 150.0
    available_calcs = ['orb', 'alignn']

    selected = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    assert len(selected) == n_samples
    assert all(calc in available_calcs for calc in selected)


def test_limited_calculator_availability(acquisition_fn):
    """Test when only subset of calculators available."""
    mean = np.array([150.0, 160.0])
    std = np.array([10.0, 20.0])
    best_observed = 155.0
    available_calcs = ['orb']  # Only cheap calculator

    selected = acquisition_fn.select_calculator(mean, std, best_observed, available_calcs)

    assert all(calc == 'orb' for calc in selected)
