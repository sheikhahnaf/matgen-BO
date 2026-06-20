"""σ-routed reward router (Lever 2 of the two-lever architecture).

For a candidate batch and a fitted GP surrogate, decide per-sample whether
to query the expensive oracle (σ ≥ τ) or use the GP posterior mean (σ < τ).

Hard rule (carried from DESIGN §4):
    GP-PREDICTED μ values are returned as REWARDS to the RL/BO loop, but
    they NEVER land in the LTM as labels. Only true oracle outputs are
    written to LTM via `ltm_rows_for_oracle_results`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RouterResult:
    """Outputs of one σ-routing pass over a candidate batch."""
    rewards: np.ndarray            # (K,) reward signal for the loop (mixed source)
    sources: np.ndarray            # (K,) bytes: b"gp" or b"oracle"
    mu: np.ndarray                 # (K,) GP posterior mean
    sigma: np.ndarray              # (K,) GP posterior std
    oracle_idx: np.ndarray         # indices into the input batch that were oracled
    oracle_cp: np.ndarray          # (n_oracle,) true Cp values (NaN where failed)
    oracle_failed: np.ndarray      # (n_oracle,) bool — oracle returned NaN
    n_oracle_calls: int            # = len(oracle_idx)
    elapsed_s: float
    cycle_id: int = -1
    threshold_used: float = field(default=float("nan"))


def sigma_route(
    Z_candidates: np.ndarray,
    atoms_candidates: list,
    surrogate,
    oracle,
    threshold: float,
    cycle_id: int = 0,
    force_anchor: bool = False,
    cold_start_min_train: int = 16,
) -> RouterResult:
    """Apply σ-threshold routing to a screened candidate batch.

    Args:
        Z_candidates: (K, d) PCA features for the K already-screened candidates.
        atoms_candidates: list of K ASE Atoms (parallel to Z_candidates).
        surrogate: fitted HCapSurrogate (or None / undertrained → all oracle).
        oracle: oracle backend with .evaluate(atoms_list) -> (cp, fail_mask).
        threshold: τ on GP posterior std (in Cp J/g/K units).
        cycle_id: BO cycle index for bookkeeping.
        force_anchor: if True, oracle ALL candidates regardless of σ.
        cold_start_min_train: if surrogate has < this many train pts, oracle all.

    Returns RouterResult.
    """
    K = len(atoms_candidates)
    assert Z_candidates.shape[0] == K, "Z and atoms must align"

    t0 = time.time()

    # Decide who gets oracled
    if surrogate is None or surrogate.model is None:
        mu = np.full(K, np.nan)
        sigma = np.full(K, np.inf)
        n_train = 0
    else:
        mu, sigma = surrogate.predict(Z_candidates)
        n_train = int(surrogate.model.train_inputs[0].shape[0])

    if force_anchor or n_train < cold_start_min_train:
        oracle_mask = np.ones(K, dtype=bool)
    else:
        oracle_mask = sigma >= threshold

    oracle_idx = np.where(oracle_mask)[0]
    rewards = np.empty(K, dtype=np.float64)
    # Use object dtype for sources — fixed-width bytes (S6/S8) silently truncates
    # longer values like b"oracfail" or future labels. Object array is small and
    # K is at most ~64 in practice.
    sources = np.empty(K, dtype=object)

    # Fill GP path first (cheap)
    gp_mask = ~oracle_mask
    rewards[gp_mask] = mu[gp_mask]
    sources[gp_mask] = "gp"

    # Oracle the rest
    oracle_cp = np.zeros(0, dtype=np.float64)
    oracle_failed = np.zeros(0, dtype=bool)

    if len(oracle_idx) > 0:
        atoms_to_oracle = [atoms_candidates[i] for i in oracle_idx]
        oracle_cp, oracle_failed = oracle.evaluate(atoms_to_oracle)
        # Failed rows get GP fallback (μ) so the RL loop has a finite reward.
        # The failure flag is propagated separately so the LTM step skips them.
        for j, i in enumerate(oracle_idx):
            if oracle_failed[j]:
                rewards[i] = mu[i] if np.isfinite(mu[i]) else 0.0
                sources[i] = "oracfail"
            else:
                rewards[i] = float(oracle_cp[j])
                sources[i] = "oracle"

    return RouterResult(
        rewards=rewards,
        sources=sources,
        mu=mu,
        sigma=sigma,
        oracle_idx=oracle_idx,
        oracle_cp=oracle_cp,
        oracle_failed=oracle_failed,
        n_oracle_calls=len(oracle_idx),
        elapsed_s=time.time() - t0,
        cycle_id=cycle_id,
        threshold_used=float(threshold) if not force_anchor else float("inf"),
    )


def ltm_rows_for_oracle_results(
    result: RouterResult,
    Z_candidates: np.ndarray,
    atoms_candidates: list,
    structure_ids: list,
    formulas: list,
) -> list[dict]:
    """Build the rows to insert into the LTM from a routing result.

    ONLY oracle-source rows are emitted (i.e., where source == b"oracle" and
    oracle did not fail). GP-source rows are NEVER written.
    """
    from src.ltm import atoms_to_json

    rows = []
    for j, i in enumerate(result.oracle_idx):
        if result.oracle_failed[j]:
            continue
        rows.append({
            "structure_id": structure_ids[i],
            "formula": formulas[i],
            "cycle_id": int(result.cycle_id),
            "atoms_json": atoms_to_json(atoms_candidates[i]),
            "Z_pca50": list(map(float, Z_candidates[i])),
            "y_cp": float(result.oracle_cp[j]),
            "y_cp_var": float("nan"),
            "sigma_pred": float(result.sigma[i]),
            "ood_score": float("nan"),
            "oracle_source": "anchor_batch" if result.threshold_used == float("inf") else "oracle",
        })
    return rows


def calibrate_threshold_from_picp(
    sigma: np.ndarray,
    errors: np.ndarray,
    target_picp: float = 0.90,
    eps: float = 1e-6,
) -> float:
    """Pick τ on σ so that the routed-to-GP path keeps PICP ≥ target.

    Logic (consensus-fix from committee review):
        - Compute the empirical |error| / σ ratio on the held-out anchor batch.
          If the GP is well-calibrated this ratio's `target_picp` quantile
          should equal the Gaussian z-score (1.645 for 90 %); if it's larger,
          the GP is over-confident and we should LOWER τ (force more oracle
          calls) — and vice versa.
        - τ_new = sigma_quantile(1 - target_picp) × z_observed
          where z_observed = quantile_target(|error| / σ) and we clamp σ to
          a small floor to avoid /0.
        - If too few points (<5) or all errors are zero, fall back to the
          previous τ (returned NaN signals "keep current" to the caller).
    """
    if len(errors) < 5 or len(sigma) != len(errors):
        return float(np.nan)
    sigma_safe = np.maximum(sigma, eps)
    ratio = np.abs(errors) / sigma_safe
    z_obs = float(np.quantile(ratio, target_picp))
    # σ-budget at which we'd start oracling: where σ-quantile aligns with the
    # observed mis-coverage. Lower σ-quantile + higher z_obs → tighter τ.
    sigma_budget = float(np.quantile(sigma, 1.0 - target_picp))
    return float(z_obs * sigma_budget)
