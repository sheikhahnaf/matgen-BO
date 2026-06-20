"""Unit tests for GP metrics module."""

import pytest
import numpy as np
from scipy.stats import spearmanr

from rewards.gp.metrics import GPMetrics


@pytest.fixture
def perfect_predictions():
    """Perfect predictions for testing."""
    y_true = np.array([100.0, 150.0, 200.0, 120.0, 180.0])
    y_pred = y_true.copy()
    y_std = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    return y_true, y_pred, y_std


@pytest.fixture
def noisy_predictions(seed):
    """Noisy predictions for testing."""
    np.random.seed(seed)
    y_true = np.random.uniform(100, 200, 20)
    y_pred = y_true + np.random.randn(20) * 10  # Add noise
    y_std = np.random.uniform(5, 15, 20)
    return y_true, y_pred, y_std


def test_rmse_perfect(perfect_predictions):
    """Test RMSE with perfect predictions."""
    y_true, y_pred, _ = perfect_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'RMSE' in metrics
    assert metrics['RMSE'] < 1e-6  # Should be near zero


def test_rmse_noisy(noisy_predictions):
    """Test RMSE with noisy predictions."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'RMSE' in metrics
    assert metrics['RMSE'] > 0


def test_mae_perfect(perfect_predictions):
    """Test MAE with perfect predictions."""
    y_true, y_pred, _ = perfect_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'MAE' in metrics
    assert metrics['MAE'] < 1e-6


def test_mae_noisy(noisy_predictions):
    """Test MAE with noisy predictions."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'MAE' in metrics
    assert metrics['MAE'] > 0
    # MAE should be less than or equal to RMSE
    assert metrics['MAE'] <= metrics['RMSE']


def test_r2_perfect(perfect_predictions):
    """Test R² with perfect predictions."""
    y_true, y_pred, _ = perfect_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'R2' in metrics
    assert abs(metrics['R2'] - 1.0) < 1e-6  # Should be 1.0


def test_r2_noisy(noisy_predictions):
    """Test R² with noisy predictions."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'R2' in metrics
    assert -1.0 <= metrics['R2'] <= 1.0  # Should be in valid range


def test_smape_perfect(perfect_predictions):
    """Test SMAPE with perfect predictions."""
    y_true, y_pred, _ = perfect_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'SMAPE' in metrics
    assert metrics['SMAPE'] < 1e-6  # Should be near zero


def test_smape_noisy(noisy_predictions):
    """Test SMAPE with noisy predictions."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'SMAPE' in metrics
    assert 0 <= metrics['SMAPE'] <= 100  # Should be percentage


def test_spearman_perfect(perfect_predictions):
    """Test Spearman correlation with perfect predictions."""
    y_true, y_pred, _ = perfect_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'Spearman' in metrics
    assert abs(metrics['Spearman'] - 1.0) < 1e-6  # Should be 1.0


def test_spearman_noisy(noisy_predictions):
    """Test Spearman correlation with noisy predictions."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'Spearman' in metrics
    assert -1.0 <= metrics['Spearman'] <= 1.0


def test_calibration_error_with_std(noisy_predictions):
    """Test calibration error when uncertainty is provided."""
    y_true, y_pred, y_std = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred, y_std)

    assert 'calibration_error' in metrics
    assert metrics['calibration_error'] >= 0


def test_calibration_error_without_std(noisy_predictions):
    """Test that calibration error is not computed without uncertainty."""
    y_true, y_pred, _ = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'calibration_error' not in metrics


def test_all_metrics_computed(noisy_predictions):
    """Test that all expected metrics are computed."""
    y_true, y_pred, y_std = noisy_predictions

    metrics = GPMetrics.compute_metrics(y_true, y_pred, y_std)

    expected_metrics = ['RMSE', 'MAE', 'R2', 'SMAPE', 'Spearman', 'calibration_error']

    for metric_name in expected_metrics:
        assert metric_name in metrics


def test_metrics_with_constant_predictions():
    """Test metrics when all predictions are the same."""
    y_true = np.array([100.0, 150.0, 200.0, 120.0, 180.0])
    y_pred = np.array([150.0, 150.0, 150.0, 150.0, 150.0])  # Constant

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    # RMSE and MAE should be non-zero
    assert metrics['RMSE'] > 0
    assert metrics['MAE'] > 0

    # R² should be 0 (variance of predictions is 0)
    # Note: sklearn might return 0 or negative for this case


def test_metrics_with_single_sample():
    """Test metrics with only one sample."""
    y_true = np.array([150.0])
    y_pred = np.array([155.0])

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    # Spearman correlation undefined for single sample


def test_metrics_numerical_stability():
    """Test metrics with extreme values."""
    y_true = np.array([1e-10, 1e10, 1e-5, 1e5])
    y_pred = y_true * 1.01  # 1% error

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    # Should not raise errors or return NaN
    assert not np.isnan(metrics['RMSE'])
    assert not np.isnan(metrics['MAE'])


def test_rmse_calculation_manual():
    """Test RMSE calculation manually."""
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 145.0, 195.0])

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    # Manual RMSE calculation
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    np.testing.assert_almost_equal(metrics['RMSE'], expected_rmse, decimal=5)


def test_mae_calculation_manual():
    """Test MAE calculation manually."""
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 145.0, 195.0])

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    # Manual MAE calculation
    expected_mae = np.mean(np.abs(y_true - y_pred))

    np.testing.assert_almost_equal(metrics['MAE'], expected_mae, decimal=5)


def test_smape_calculation_manual():
    """Test SMAPE calculation manually."""
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([110.0, 145.0, 195.0])

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    # Manual SMAPE calculation
    expected_smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))
    )

    np.testing.assert_almost_equal(metrics['SMAPE'], expected_smape, decimal=5)


def test_calibration_error_calculation():
    """Test calibration error calculation."""
    y_true = np.array([100.0, 150.0, 200.0, 120.0, 180.0])
    y_pred = np.array([105.0, 155.0, 195.0, 125.0, 175.0])
    y_std = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    metrics = GPMetrics.compute_metrics(y_true, y_pred, y_std)

    # Calibration error should be close to 0 if uncertainties are well-calibrated
    # In this case, errors are within ~1 std, so calibration should be reasonable
    assert metrics['calibration_error'] >= 0


@pytest.mark.parametrize("n_samples", [5, 10, 50, 100])
def test_metrics_with_different_sample_sizes(n_samples, seed):
    """Test metrics computation with different sample sizes."""
    np.random.seed(seed)
    y_true = np.random.uniform(100, 200, n_samples)
    y_pred = y_true + np.random.randn(n_samples) * 10

    metrics = GPMetrics.compute_metrics(y_true, y_pred)

    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R2' in metrics
    assert 'SMAPE' in metrics

    if n_samples > 1:
        assert 'Spearman' in metrics
