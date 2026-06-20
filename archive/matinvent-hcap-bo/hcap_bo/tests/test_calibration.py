"""Unit tests for src/calibration.py (committee P2 fix verification)."""

import numpy as np
import pytest

from src.calibration import ence, picp, nll_gauss, reliability_diagram, regression_metrics


def _well_calibrated(n=500, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    sigma = rng.uniform(0.1, 1.0, n)
    y = mu + rng.normal(0, sigma)
    return mu, sigma, y


def _underconfident(n=500, seed=0):
    """sigma too large → PICP@90 ≈ 1.0, ENCE > 0."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    sigma_true = rng.uniform(0.1, 1.0, n)
    sigma = sigma_true * 3.0
    y = mu + rng.normal(0, sigma_true)
    return mu, sigma, y


def _overconfident(n=500, seed=0):
    """sigma too small → PICP@90 << 0.9, ENCE > 0."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    sigma_true = rng.uniform(0.1, 1.0, n)
    sigma = sigma_true * 0.3
    y = mu + rng.normal(0, sigma_true)
    return mu, sigma, y


def test_ence_uses_rmv_not_mean_sigma():
    """ENCE on well-calibrated data ≈ 0; on miscalibrated data, > 0."""
    mu, sigma, y = _well_calibrated(n=2000, seed=1)
    e_ok = ence(mu, sigma, y, n_bins=15)
    assert e_ok < 0.20, f"ENCE on well-calibrated data should be ≤ 0.2, got {e_ok}"

    mu_o, sig_o, y_o = _overconfident(n=2000, seed=2)
    e_bad = ence(mu_o, sig_o, y_o, n_bins=15)
    assert e_bad > e_ok, f"ENCE should be larger on overconfident data: ok={e_ok}, bad={e_bad}"


def test_picp_matches_nominal_on_well_calibrated():
    mu, sigma, y = _well_calibrated(n=5000, seed=3)
    for level in (0.50, 0.90, 0.95):
        p = picp(mu, sigma, y, level=level)
        assert abs(p - level) < 0.03, f"PICP@{level} should be ~{level}, got {p}"


def test_picp_low_on_overconfident():
    mu, sigma, y = _overconfident(n=2000, seed=4)
    p = picp(mu, sigma, y, level=0.90)
    assert p < 0.7, f"PICP@90 on overconfident data should be < 0.7, got {p}"


def test_picp_high_on_underconfident():
    mu, sigma, y = _underconfident(n=2000, seed=5)
    p = picp(mu, sigma, y, level=0.90)
    assert p > 0.95, f"PICP@90 on underconfident data should be > 0.95, got {p}"


def test_nll_lower_on_better_calibration():
    mu, sigma, y = _well_calibrated(n=2000, seed=6)
    mu_o, sig_o, y_o = _overconfident(n=2000, seed=6)
    nll_ok = nll_gauss(mu, sigma, y)
    nll_bad = nll_gauss(mu_o, sig_o, y_o)
    assert nll_ok < nll_bad, f"NLL on overconfident should be larger: ok={nll_ok}, bad={nll_bad}"


def test_reliability_diagram_returns_rmv_key():
    """Committee fix: reliability_diagram should expose RMV (not mean_sigma)."""
    mu, sigma, y = _well_calibrated(n=300, seed=7)
    rd = reliability_diagram(mu, sigma, y, n_bins=10)
    assert "rmv" in rd, "reliability_diagram should contain 'rmv'"
    assert "rmse" in rd
    assert "n_per_bin" in rd
    # On well-calibrated data, RMV ≈ RMSE per bin.
    valid = ~np.isnan(rd["rmv"]) & ~np.isnan(rd["rmse"])
    if valid.sum() >= 5:
        ratio = rd["rmse"][valid] / rd["rmv"][valid]
        assert (ratio.mean() > 0.6) and (ratio.mean() < 1.4), \
            f"RMSE/RMV mean ratio off: {ratio.mean()}"


def test_regression_metrics_keys():
    rng = np.random.default_rng(8)
    y = rng.normal(0, 1, 200)
    m = regression_metrics(y + 0.1 * rng.normal(0, 1, 200), y)
    assert set(m.keys()) == {"rmse", "mae", "r2", "spearman"}
    assert m["r2"] > 0.95
    assert m["spearman"] > 0.95
