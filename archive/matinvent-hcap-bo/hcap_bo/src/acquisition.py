"""Batched acquisition for pre-oracle screening.

Strategy: rank candidates by qLogNoisyExpectedImprovement (BoTorch). For
larger batches, optionally apply a k-DPP diversity guard over the posterior
samples to avoid redundant picks (common when MatterGen produces many
near-duplicates of a high-EI motif).

Key behaviors:
    - Cold-start fallback: if surrogate has fewer than `min_train_for_ei`
      observations or candidate posterior std collapses near zero, returns
      a random selection (avoids the upstream EI-degenerate-at-σ→0 bug).
    - Operates on already-PCA-reduced features; surrogate.predict gives the
      true-scale (mu, sigma) used for fallback inspection.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement


def _kdpp_select(scores: np.ndarray, kernel: np.ndarray, k: int) -> np.ndarray:
    """Greedy k-DPP MAP: pick highest-scoring item, then iteratively add the
    item that maximizes (score * sqrt(det(K_S))) using a Gram-Schmidt update.

    `scores` shape (N,) ≥ 0 (we'll shift if needed); `kernel` shape (N, N).
    """
    n = len(scores)
    if k >= n:
        return np.arange(n)

    # Shift to non-negative; add a tiny floor so all items have nonzero
    # selection probability even when the score range is very narrow.
    score_range = float(scores.max() - scores.min())
    s = scores - scores.min() + max(1e-9, 1e-3 * score_range)
    selected = []
    di2 = np.diag(kernel).astype(np.float64).copy()
    e_vecs = np.zeros((n, k))

    for it in range(k):
        objective = s * np.sqrt(np.maximum(di2, 0.0))
        # mask already-selected
        if selected:
            objective[selected] = -np.inf
        idx = int(np.argmax(objective))
        selected.append(idx)

        if it < k - 1:
            ei = (kernel[:, idx] - e_vecs[:, :it] @ e_vecs[idx, :it]) / (
                np.sqrt(max(di2[idx], 1e-12))
            )
            e_vecs[:, it] = ei
            di2 = di2 - ei ** 2

    return np.array(selected)


def select_topk(
    surrogate,
    Z_candidates: np.ndarray,
    k: int,
    diversity: str = "kdpp",
    raw_samples: int = 4096,
    num_restarts: int = 10,
    min_train_for_ei: int = 32,
    sigma_floor_for_ei: float = 1e-4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return indices of the top-k candidates ranked by qLogNEI (+optional k-DPP).

    Args:
        surrogate: HCapSurrogate (already fitted).
        Z_candidates: (N, d) PCA features (np.ndarray, original feature scale).
        k: how many to return.
        diversity: "kdpp" or "none".
        seed: reproducibility seed for raw-sample initial conditions.

    Behavior on cold start / σ-collapse: returns a random k-subset.
    """
    if surrogate.model is None:
        rng = np.random.default_rng(seed)
        return rng.choice(len(Z_candidates), size=min(k, len(Z_candidates)), replace=False)

    n_train = int(surrogate.model.train_inputs[0].shape[0])
    mu, sigma = surrogate.predict(Z_candidates)

    if n_train < min_train_for_ei or float(np.median(sigma)) < sigma_floor_for_ei:
        # qLogNEI is unreliable; fall back to UCB on (mu, sigma).
        warnings.warn(
            f"Falling back to UCB selection (n_train={n_train}, median sigma={np.median(sigma):.2e})."
        )
        kappa = 2.0
        score = mu + kappa * sigma
        order = np.argsort(-score)
        if diversity == "kdpp":
            kernel = _gauss_kernel(Z_candidates)
            return _kdpp_select(score - score.min(), kernel, k)
        return order[:k]

    # Normal path: qLogNEI on the surrogate's *scaled* model
    Xs = surrogate.x_scaler.transform(Z_candidates)
    Xt = torch.as_tensor(Xs, dtype=surrogate.dtype, device=surrogate.device)

    # X_baseline = surrogate's training inputs
    X_baseline = surrogate.model.train_inputs[0]

    acq = qLogNoisyExpectedImprovement(
        model=surrogate.model,
        X_baseline=X_baseline,
        prune_baseline=True,
    )

    with torch.no_grad():
        scores_t = acq(Xt.unsqueeze(1))  # shape (N,)
    scores = scores_t.detach().cpu().numpy()

    if diversity == "kdpp":
        kernel = _gauss_kernel(Z_candidates)
        return _kdpp_select(scores - scores.min(), kernel, k)

    order = np.argsort(-scores)
    return order[:k]


def _gauss_kernel(Z: np.ndarray, lengthscale: Optional[float] = None) -> np.ndarray:
    """Simple RBF kernel for k-DPP diversity. Lengthscale = median pairwise dist."""
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(Z))
    if lengthscale is None:
        lengthscale = float(np.median(D[D > 0])) or 1.0
    K = np.exp(-(D ** 2) / (2.0 * lengthscale ** 2))
    return K
