"""GP Metrics calculation.

Provides comprehensive regression metrics for evaluating GP surrogate performance.
Exactly matches user's metric calculation code.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Exactly matches user's implementation with:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of Determination
    - SMAPE: Symmetric Mean Absolute Percentage Error
    - Spearman: Spearman rank correlation

    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Predicted standard deviations (optional, for calibration)

    Returns:
        dict: Dictionary of metric names and values
    """
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # R²
    r2 = r2_score(y_true, y_pred)

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denom = np.abs(y_pred) + np.abs(y_true)
    denom[denom == 0] = 1e-8  # Avoid division by zero
    smape = np.mean(2 * np.abs(y_pred - y_true) / denom) * 100

    # Spearman correlation
    spearman, _ = spearmanr(y_true, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'SMAPE': smape,
        'Spearman': spearman,
    }

    # Calibration metrics (if uncertainty provided)
    if y_std is not None:
        ece = calculate_calibration_error(y_true, y_pred, y_std)
        metrics['ECE'] = ece

        # Average predicted uncertainty
        metrics['avg_uncertainty'] = np.mean(y_std)

    return metrics


def calculate_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Measures how well the predicted uncertainties match actual errors.
    Lower ECE indicates better calibrated uncertainty estimates.

    Args:
        y_true: True target values
        y_pred: Predicted values
        y_std: Predicted standard deviations
        n_bins: Number of bins for calibration curve

    Returns:
        float: Expected Calibration Error
    """
    # Calculate z-scores (normalized errors)
    z_scores = np.abs((y_true - y_pred) / (y_std + 1e-9))

    # Expected fractions within confidence intervals
    confidence_levels = np.linspace(0, 1, n_bins + 1)[1:]

    ece = 0.0
    for conf in confidence_levels:
        # Theoretical: conf fraction should be within conf*sigma
        # Actual: count how many are actually within
        expected_fraction = conf
        actual_fraction = np.mean(z_scores <= conf)

        # Accumulate calibration error
        ece += np.abs(expected_fraction - actual_fraction)

    ece /= len(confidence_levels)

    return ece


def aggregate_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple splits/folds.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        dict: Dictionary with mean and std for each metric
    """
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return aggregated


class GPMetrics:
    """
    GP Metrics tracker for logging and analysis.

    Provides static methods for metric computation and
    maintains history for plotting/analysis.
    """

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute metrics (wrapper around calculate_metrics).

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_std: Predicted standard deviations (optional)

        Returns:
            dict: Dictionary of metric names and values
        """
        return calculate_metrics(y_true, y_pred, y_std)

    @staticmethod
    def print_metrics(
        metrics: Dict[str, float],
        prefix: str = ""
    ):
        """
        Print metrics in formatted way.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for printing (e.g., "Train" or "Val")
        """
        print(f"\n{prefix} Metrics:")
        print(f"  R² = {metrics['R2']:.4f}")
        print(f"  RMSE = {metrics['RMSE']:.4f}")
        print(f"  MAE = {metrics['MAE']:.4f}")
        print(f"  SMAPE = {metrics['SMAPE']:.2f}%")
        print(f"  Spearman = {metrics['Spearman']:.4f}")

        if 'ECE' in metrics:
            print(f"  ECE = {metrics['ECE']:.4f}")
        if 'avg_uncertainty' in metrics:
            print(f"  Avg Uncertainty = {metrics['avg_uncertainty']:.4f}")
