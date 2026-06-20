"""Uncertainty calibration metrics for GP posteriors.

Closes the gap explicitly noted in the FME paper (no calibration audit).

Metrics:
    ence:     Expected Normalized Calibration Error (Levi et al., 2020).
    picp:     Prediction Interval Coverage Probability at given level.
    nll_gauss: Gaussian negative log-likelihood under (mu, sigma).
    reliability_diagram: per-bin (rmse, mean_predicted_std) for plotting.
"""

from __future__ import annotations

import numpy as np


def _bin_by_predicted_std(sigma: np.ndarray, n_bins: int):
    order = np.argsort(sigma)
    chunks = np.array_split(order, n_bins)
    return chunks


def ence(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Normalized Calibration Error (Levi et al., 2020).

    For each bin (sorted by predicted σ):
        RMV = √(mean(σ²))     — root-mean-variance of predicted std in bin
        RMSE = √(mean((y-μ)²)) — empirical RMSE in bin
        bin_ence = |RMSE − RMV| / RMV
    ENCE = mean over bins of bin_ence. Lower is better; 0 = perfectly calibrated.

    Per the original paper: comparing RMSE to RMV (not mean σ) is the correct
    formulation — they coincide only when σ is constant within a bin, which it
    typically is not.
    """
    sigma = np.maximum(sigma, 1e-8)
    chunks = _bin_by_predicted_std(sigma, n_bins)
    vals = []
    for idx in chunks:
        if len(idx) == 0:
            continue
        rmse = float(np.sqrt(np.mean((y[idx] - mu[idx]) ** 2)))
        rmv = float(np.sqrt(np.mean(sigma[idx] ** 2)))
        vals.append(abs(rmse - rmv) / rmv)
    return float(np.mean(vals))


def picp(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, level: float = 0.95) -> float:
    """Prediction Interval Coverage Probability at the given level (Gaussian)."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2.0)
    lo = mu - z * sigma
    hi = mu + z * sigma
    inside = (y >= lo) & (y <= hi)
    return float(inside.mean())


def nll_gauss(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood. Lower is better."""
    sigma = np.maximum(sigma, 1e-8)
    return float(
        np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * ((y - mu) / sigma) ** 2)
    )


def reliability_diagram(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> dict:
    """Returns per-bin {rmv, rmse, n} arrays for plotting (RMV vs RMSE per Levi)."""
    chunks = _bin_by_predicted_std(np.maximum(sigma, 1e-8), n_bins)
    rmv_vals, rmse_vals, ns = [], [], []
    for idx in chunks:
        if len(idx) == 0:
            rmv_vals.append(np.nan); rmse_vals.append(np.nan); ns.append(0)
            continue
        rmv_vals.append(float(np.sqrt(np.mean(sigma[idx] ** 2))))
        rmse_vals.append(float(np.sqrt(np.mean((y[idx] - mu[idx]) ** 2))))
        ns.append(int(len(idx)))
    return {
        "rmv": np.array(rmv_vals),
        "rmse": np.array(rmse_vals),
        "n_per_bin": np.array(ns),
    }


def regression_metrics(mu: np.ndarray, y: np.ndarray) -> dict:
    """RMSE, MAE, R^2, Spearman ρ — the FME-paper headline four."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y, mu)))
    mae = float(np.mean(np.abs(y - mu)))
    r2 = float(r2_score(y, mu))
    rho, _ = spearmanr(y, mu)
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": float(rho)}
