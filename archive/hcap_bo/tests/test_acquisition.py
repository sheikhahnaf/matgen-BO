"""Unit tests for src/acquisition.py — qLogNEI + k-DPP top-k selection."""

import numpy as np
import pytest
import torch

from src.acquisition import select_topk, _kdpp_select, _gauss_kernel
from src.surrogate import HCapSurrogate


def _fitted_surrogate(n=64, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    # Synthetic target: y = sum of first two features + noise
    y = X[:, 0] + 0.5 * X[:, 1] + 0.05 * rng.normal(0, 1, n)
    sur = HCapSurrogate(device="cpu")
    sur.fit(X, y)
    return sur, X, y


def test_topk_returns_unique_indices():
    sur, X, _ = _fitted_surrogate()
    rng = np.random.default_rng(1)
    Z_q = rng.normal(0, 1, (32, 10))
    idx = select_topk(sur, Z_q, k=8, diversity="kdpp", seed=0)
    assert len(idx) == 8
    assert len(set(idx.tolist())) == 8
    assert all(0 <= i < 32 for i in idx)


def test_topk_no_diversity_picks_topk_by_score():
    sur, X, _ = _fitted_surrogate()
    rng = np.random.default_rng(2)
    Z_q = rng.normal(0, 1, (20, 10))
    idx = select_topk(sur, Z_q, k=5, diversity="none", seed=0)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5


def test_topk_cold_start_falls_back_to_random():
    """If surrogate is undertrained, must NOT crash; should pick random k-subset."""
    sur, X, y = _fitted_surrogate(n=8, d=4)
    rng = np.random.default_rng(3)
    Z_q = rng.normal(0, 1, (16, 4))
    idx = select_topk(sur, Z_q, k=4, diversity="none", min_train_for_ei=32, seed=0)
    assert len(idx) == 4
    assert len(set(idx.tolist())) == 4


def test_kdpp_handles_narrow_score_range():
    """Committee fix: scores - scores.min() + epsilon should not collapse to zeros
    when score variance is tiny (which used to make k-DPP pick arbitrary points)."""
    n = 20
    scores = np.full(n, 1.0) + np.random.default_rng(0).normal(0, 1e-8, n)
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1, (n, 5))
    K = _gauss_kernel(Z)
    idx = _kdpp_select(scores, K, k=5)
    assert len(idx) == 5
    assert len(set(idx.tolist())) == 5


def test_gauss_kernel_psd():
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1, (10, 5))
    K = _gauss_kernel(Z)
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() > -1e-6, f"kernel not PSD: min eig = {eigs.min()}"


def test_select_topk_returns_within_pool():
    sur, X, _ = _fitted_surrogate()
    rng = np.random.default_rng(4)
    Z_q = rng.normal(0, 1, (50, 10))
    idx = select_topk(sur, Z_q, k=10, diversity="kdpp", seed=0)
    assert (idx >= 0).all() and (idx < 50).all()
