"""Unit tests for src/surrogate.py — including committee P1 fix for save/load
preserving heteroscedastic noise."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.surrogate import HCapSurrogate


def _toy_data(n=60, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(0, 1, n)
    return X, y


def test_fit_predict_basic():
    X, y = _toy_data()
    sur = HCapSurrogate(device="cpu")
    sur.fit(X, y)
    mu, sigma = sur.predict(X)
    assert mu.shape == (60,) and sigma.shape == (60,)
    assert np.all(sigma > 0)
    # Within reasonable RMSE on training data
    assert np.sqrt(np.mean((mu - y) ** 2)) < 0.5


def test_predict_in_original_y_scale():
    """y_scaler should be inverted before returning μ; large absolute target → large μ."""
    X, y = _toy_data()
    y = y * 100.0 + 50.0  # shift + scale
    sur = HCapSurrogate(device="cpu")
    sur.fit(X, y)
    mu, sigma = sur.predict(X)
    # μ should be in the original-y range (~50), not z-score range
    assert mu.mean() > 30 and mu.mean() < 80
    assert sigma.mean() > 1.0  # σ also in original scale, not unit


def test_save_load_roundtrip_homoscedastic():
    X, y = _toy_data()
    sur = HCapSurrogate(device="cpu")
    sur.fit(X, y)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sur.pkl"
        sur.save(path)
        sur2 = HCapSurrogate.load(path, device="cpu")
        mu1, sig1 = sur.predict(X[:10])
        mu2, sig2 = sur2.predict(X[:10])
        np.testing.assert_allclose(mu1, mu2, atol=1e-5)
        np.testing.assert_allclose(sig1, sig2, atol=1e-5)


def test_save_load_roundtrip_heteroscedastic():
    """Committee fix: train_Yvar should be properly restored, not overwritten with 1e-4."""
    X, y = _toy_data(n=40, seed=2)
    rng = np.random.default_rng(99)
    y_var = rng.uniform(0.01, 0.5, len(y))  # per-sample noise variance
    sur = HCapSurrogate(device="cpu")
    sur.fit(X, y, y_var=y_var)
    assert sur._is_fixed_noise is True

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sur_het.pkl"
        sur.save(path)
        sur2 = HCapSurrogate.load(path, device="cpu")

    # Predictions must match
    mu1, sig1 = sur.predict(X[:10])
    mu2, sig2 = sur2.predict(X[:10])
    np.testing.assert_allclose(mu1, mu2, atol=1e-5)
    np.testing.assert_allclose(sig1, sig2, atol=1e-5)

    # And the noise vector itself must round-trip (not be the dummy 1e-4)
    n1 = sur.model.likelihood.noise_covar.noise.detach().cpu().numpy()
    n2 = sur2.model.likelihood.noise_covar.noise.detach().cpu().numpy()
    np.testing.assert_allclose(n1, n2, atol=1e-5)
    # Noise should NOT all be 1e-4 (the previous buggy default)
    assert not np.allclose(n2, 1e-4), "load() restored dummy noise — committee bug returned"


def test_predict_before_fit_raises():
    sur = HCapSurrogate(device="cpu")
    X = np.zeros((1, 5))
    with pytest.raises(ValueError):
        sur.predict(X)
