"""LocalESEN_GPRoutedV3 — v3 BO+RL accelerator.

Sibling of LocalESEN_GPRouted (does NOT replace it). Selected via Hydra:
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRoutedV3

Improvements over v2 (committee findings + observed late-cycle ACC collapse):

    (1) Honest reward — non-oracle samples return NaN as Cp.
        Reason: v2 sets reward[gp_idx] = mu (GP prediction). RL then learns to
        maximise GP-predicted Cp, NOT real Cp; over cycles RL drifts toward
        GP-error regions.
        Mechanism: NaN propagates through rewards/reward.py:calc_props →
        none_ids[gp_idx] = True → rewards[gp_idx] = 0 → pipeline/base.py
        success_mask filters them out of REINFORCE gradient. Ground-truth-only
        RL update.

    (2) DPP batch acquisition for top-K — replace argmax-of-EI with greedy
        diversity-aware selection. Avoids picking K near-duplicate structures
        per cycle (mode collapse).

    (4) Warm-start importance decay — older LTM rows (warm-start, cycle_id=-1)
        get heavier noise variance in the FixedNoiseGP fit, fading their
        influence over training. Live oracle labels are full-weight.

    (6) GP recalibration logging — every fit, prints LOO-CV RMSE / MAE plus
        train RMSE. Lets us diagnose whether GP error tracks the live
        distribution or has gone stale (H1 vs H2 from the v3 plan).

Knobs (env vars, all optional):
    GP_ACQUISITION         "ei" (default) | "pi"
    GP_TOP_K               default 8
    GP_TARGET_CP           default 1.5
    GP_ANCHOR_EVERY        default 5
    GP_COLD_START_MIN      default 50  (was 16) — user's hunch: GP needs more
                           data before routing kicks in
    GP_PCA_COMPONENTS      default 50
    GP_LTM_PATH            persistence path
    GP_LTM_SEED            warm-start seed pool path
    GP_DPP_LAMBDA          diversity weight in greedy DPP, default 0.5
    GP_WARMSTART_DECAY     fade rate per cycle, default 0.9 (1.0 = no decay)
    GP_WARMSTART_NOISE_MIN minimum noise floor, default 1e-6
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from ase.io import read as ase_read

from src.calculators.local_esen import LocalESEN
from src.featurizer import ORBFeaturizer
from src.surrogate import HCapSurrogate
from src.ltm import LTM, canonical_atoms_id, atoms_to_json
from src import calibration as cal


def _scratch_path(*parts) -> str:
    sc = os.environ.get("SCRATCH", "/tmp")
    return os.path.join(sc, *parts)


def _probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, target: float) -> np.ndarray:
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    return 1.0 - norm.cdf((target - mu) / sigma)


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - f_best) / sigma
    return (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)


def _greedy_dpp_topk(
    scores: np.ndarray,
    features: np.ndarray,
    K: int,
    lambda_div: float = 0.5,
) -> np.ndarray:
    """Greedy DPP MAP selection: pick K samples maximising acquisition while
    spreading across feature space.

    Step i: pick j = argmax_j (score[j] - lambda_div * max_{s in selected} cosine(features[j], features[s])).
    Cosine similarity on standardised PCA features ∈ [-1, 1].
    """
    if len(scores) <= K:
        return np.argsort(-scores)

    # Cosine similarity matrix
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    feat_n = features / norms
    sim = feat_n @ feat_n.T  # (N, N)

    selected = []
    remaining = set(range(len(scores)))
    # First pick: highest acquisition score
    first = int(np.argmax(scores))
    selected.append(first)
    remaining.discard(first)

    for _ in range(K - 1):
        if not remaining:
            break
        rem = np.array(sorted(remaining))
        # Max similarity to any already-selected sample
        max_sim_to_sel = sim[rem][:, selected].max(axis=1)
        adjusted = scores[rem] - lambda_div * max_sim_to_sel
        pick = int(rem[np.argmax(adjusted)])
        selected.append(pick)
        remaining.discard(pick)

    return np.array(selected, dtype=int)


def _gp_loo_metrics(sur: "HCapSurrogate", Z: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Leave-one-out CV RMSE / MAE on the LTM. Plus train RMSE for reference.

    For LOO at every cycle this is O(N²) — fine while N≲500. We use the
    closed-form GP LOO trick: Σ⁻¹y / [Σ⁻¹]_ii, but botorch doesn't expose Σ⁻¹
    cleanly. Approximate by 5-fold CV when N≥25, else LOO via predict-on-held.
    """
    if len(y) < 4:
        return float("nan"), float("nan"), float("nan")

    # Train RMSE on all samples (sanity)
    mu_train, _ = sur.predict(Z)
    train_rmse = float(np.sqrt(np.mean((mu_train - y) ** 2)))

    # 5-fold (or LOO) CV using the same surrogate class
    n = len(y)
    n_splits = min(5, n)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_splits)

    cv_preds = np.full(n, np.nan)
    for fold_idx, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for k, f in enumerate(folds) if k != fold_idx])
        if len(train_idx) < 3:
            continue
        try:
            sur_cv = HCapSurrogate(device=sur.device)
            sur_cv.fit(Z[train_idx], y[train_idx])
            mu_v, _ = sur_cv.predict(Z[val_idx])
            cv_preds[val_idx] = mu_v
        except Exception:
            pass

    ok = np.isfinite(cv_preds)
    if ok.sum() < 3:
        return train_rmse, float("nan"), float("nan")
    cv_rmse = float(np.sqrt(np.mean((cv_preds[ok] - y[ok]) ** 2)))
    cv_mae = float(np.mean(np.abs(cv_preds[ok] - y[ok])))
    return train_rmse, cv_rmse, cv_mae


class LocalESEN_GPRoutedV3:
    """v3 GP-routed wrapper with honest reward, DPP top-K, warm-start decay,
    and CV calibration logging."""

    def __init__(
        self,
        root_dir: str,
        task: str = "heat_capacity",
        env_name: str | None = None,
        worker: int = 1,
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "heat_capacity":
            raise ValueError(f"v3 only supports task='heat_capacity', got {task!r}")
        self.worker = int(worker)
        self._oracle = LocalESEN(root_dir=root_dir, task=task, env_name=env_name, worker=worker)

        import torch
        self.pca_components = int(os.environ.get("GP_PCA_COMPONENTS", "50"))
        self.top_k = int(os.environ.get("GP_TOP_K", "8"))
        self.target_cp = float(os.environ.get("GP_TARGET_CP", "1.5"))
        self.anchor_every = int(os.environ.get("GP_ANCHOR_EVERY", "5"))
        # v3 default: 50 (was 16) — user's hunch: GP needs more data first
        self.cold_start_min = int(os.environ.get("GP_COLD_START_MIN", "50"))
        self.dpp_lambda = float(os.environ.get("GP_DPP_LAMBDA", "0.5"))
        self.ws_decay = float(os.environ.get("GP_WARMSTART_DECAY", "0.9"))
        self.ws_noise_min = float(os.environ.get("GP_WARMSTART_NOISE_MIN", "1e-6"))
        self.device = os.environ.get(
            "GP_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        ltm_path = os.environ.get(
            "GP_LTM_PATH",
            _scratch_path("matinvent-hcap-bo", "data", "ltm_phase3_v3.parquet"),
        )

        self._seed_pending = []
        seed_path = os.environ.get("GP_LTM_SEED", "").strip()
        if seed_path and os.path.exists(seed_path) and not os.path.exists(ltm_path):
            try:
                import pandas as _pd
                from src.ltm import atoms_from_json
                df = _pd.read_parquet(seed_path)
                df = df.dropna(subset=["y_cp"])
                for _, row in df.iterrows():
                    try:
                        atoms = atoms_from_json(row["atoms_json"])
                        self._seed_pending.append((atoms, float(row["y_cp"])))
                    except Exception:
                        pass
                print(f"[GPRoutedV3] queued {len(self._seed_pending)} warm-start "
                      f"samples from {seed_path} (re-embed at first calc)",
                      file=sys.stderr)
            except Exception as e:
                print(f"[GPRoutedV3] warm-start load failed: {e}", file=sys.stderr)

        self._ltm = LTM(ltm_path)
        self.acquisition = os.environ.get("GP_ACQUISITION", "ei").lower()
        if self.acquisition not in ("ei", "pi"):
            self.acquisition = "ei"

        self._feat: Optional[ORBFeaturizer] = None
        self._sur: Optional[HCapSurrogate] = None
        self._cycle = 0

        self._log_path = os.path.join(root_dir, "gp_routed_v3_log.csv")
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w") as f:
                f.write(
                    "cycle,n_input,K,n_oracle,n_gp,is_anchor,target_cp,ltm_size,"
                    "mean_oracle_cp,gp_rmse_train,gp_rmse_cv5,gp_mae_cv5,"
                    "elapsed_s\n"
                )

        print(
            f"[GPRoutedV3] init: K={self.top_k} target={self.target_cp} "
            f"acq={self.acquisition} cold_min={self.cold_start_min} "
            f"dpp_λ={self.dpp_lambda} ws_decay={self.ws_decay}",
            file=sys.stderr,
        )

    # --------------------------------------------------------------------

    def calc(self, samples, label: str = "tmp") -> np.ndarray:
        t0 = time.time()
        atoms_list = self._read_atoms(samples)
        N = len(atoms_list)
        if N == 0:
            return np.zeros(0, dtype=np.float64)

        if self._feat is None:
            self._feat = ORBFeaturizer(n_components=self.pca_components, device=self.device)

        # Featurize (and re-embed warm-start under live PCA basis on first call)
        if not self._feat.is_fitted and self._seed_pending:
            seed_atoms = [a for a, _ in self._seed_pending]
            seed_y = np.array([y for _, y in self._seed_pending], dtype=np.float64)
            combined_atoms = seed_atoms + atoms_list
            Z_combined = self._feat.fit_transform(combined_atoms)
            Z_seed = Z_combined[: len(seed_atoms)]
            Z = Z_combined[len(seed_atoms):]
            for atoms, y, z in zip(seed_atoms, seed_y, Z_seed):
                self._ltm.append([{
                    "structure_id": canonical_atoms_id(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "cycle_id": -1,
                    "atoms_json": atoms_to_json(atoms),
                    "Z_pca50": list(map(float, z)),
                    "y_cp": float(y),
                    "y_cp_var": float("nan"),
                    "sigma_pred": float("nan"),
                    "ood_score": float("nan"),
                    "oracle_source": "warm_start",
                }])
            print(f"[GPRoutedV3] re-embedded {len(seed_atoms)} warm-start rows",
                  file=sys.stderr)
            self._seed_pending = []
        elif not self._feat.is_fitted:
            Z = self._feat.fit_transform(atoms_list)
        else:
            Z = self._feat.transform(atoms_list)

        # Decide oracle subset
        is_anchor = (self._cycle > 0) and (self._cycle % self.anchor_every == 0)
        is_cold = self._sur is None or self._ltm.size() < self.cold_start_min
        K = self.top_k

        if is_cold or is_anchor or K >= N:
            mu = np.full(N, np.nan)
            sigma = np.full(N, np.inf)
            oracle_idx = np.arange(N)
            gp_idx = np.array([], dtype=int)
        else:
            mu, sigma = self._sur.predict(Z)
            if self.acquisition == "ei":
                _, y_train = self._ltm.features_and_targets()
                f_best = float(np.nanmax(y_train)) if len(y_train) else 0.0
                pi = _expected_improvement(mu, sigma, f_best)
            else:
                pi = _probability_of_improvement(mu, sigma, self.target_cp)
            # (2) DPP-diverse top-K
            oracle_idx = _greedy_dpp_topk(pi, Z, K, lambda_div=self.dpp_lambda)
            oracle_idx = np.sort(oracle_idx)
            mask_o = np.zeros(N, dtype=bool)
            mask_o[oracle_idx] = True
            gp_idx = np.where(~mask_o)[0]

        rewards = np.empty(N, dtype=np.float64)

        # (1) HONEST REWARD: non-oracle samples → NaN, propagates through
        #     rewards/reward.py:calc_props → failed_mask → excluded from gradient.
        if len(gp_idx) > 0:
            rewards[gp_idx] = np.nan

        oracle_cp = np.full(len(oracle_idx), np.nan)
        if len(oracle_idx) > 0:
            atoms_o = [atoms_list[int(i)] for i in oracle_idx]
            cp_list = []
            for a in atoms_o:
                try:
                    cp_list.append(self._oracle._phonon_task(a))
                except Exception as e:
                    print(f"[GPRoutedV3] inner _phonon_task failed: "
                          f"{type(e).__name__}: {e}", file=sys.stderr)
                    cp_list.append(float("nan"))
            cp_subset = np.array(cp_list, dtype=np.float64)
            oracle_cp[: len(cp_subset)] = cp_subset
            for j, i in enumerate(oracle_idx):
                cp_val = oracle_cp[j]
                # Failed oracle → NaN (excluded from gradient just like gp_idx).
                # No fallback to GP μ (that would re-introduce contamination).
                rewards[i] = float(cp_val) if np.isfinite(cp_val) else np.nan

        # Append true labels
        for j, i in enumerate(oracle_idx):
            cp_val = oracle_cp[j]
            if not np.isfinite(cp_val):
                continue
            atoms = atoms_list[int(i)]
            self._ltm.append([{
                "structure_id": canonical_atoms_id(atoms),
                "formula": atoms.get_chemical_formula(),
                "cycle_id": int(self._cycle),
                "atoms_json": atoms_to_json(atoms),
                "Z_pca50": list(map(float, Z[i])),
                "y_cp": float(cp_val),
                "y_cp_var": float("nan"),
                "sigma_pred": float(sigma[i]) if np.isfinite(sigma[i]) else float("nan"),
                "ood_score": float("nan"),
                "oracle_source": "anchor_batch" if is_anchor else (
                    "seed_pool" if is_cold else "oracle"
                ),
            }])

        # (4) WARM-START DECAY: re-fit GP with FixedNoiseGP, per-row noise
        #     scaled by (1 - decay) ** age. Live oracle = age 0 → small noise;
        #     warm-start cycle_id=-1 → age = current_cycle + 1 → large noise.
        gp_rmse_train = gp_rmse_cv = gp_mae_cv = float("nan")
        df_full = self._ltm.load()
        if len(df_full) >= 4:
            Z_train = np.stack(df_full["Z_pca50"].apply(np.asarray).tolist())
            y_train = df_full["y_cp"].to_numpy(dtype=np.float64)
            cyc = df_full["cycle_id"].to_numpy()
            # Warm-start has cycle_id == -1 → treat as oldest.
            ages = np.where(cyc < 0, self._cycle + 1, self._cycle - cyc).astype(float)
            ages = np.maximum(ages, 0.0)
            weights = self.ws_decay ** ages  # ∈ (0, 1]
            # Convert weight → noise variance: high weight → low noise.
            # noise_var = noise_floor / weight. y is on raw Cp scale; scale ~ 0.1-1 J/g/K
            y_var = self.ws_noise_min / np.maximum(weights, 1e-3)
            try:
                self._sur = HCapSurrogate(device=self.device)
                self._sur.fit(Z_train, y_train, y_var=y_var)
                # (6) GP RMSE/MAE diagnostic
                gp_rmse_train, gp_rmse_cv, gp_mae_cv = _gp_loo_metrics(
                    self._sur, Z_train, y_train
                )
            except Exception as e:
                print(f"[GPRoutedV3] GP fit failed: {type(e).__name__}: {e}",
                      file=sys.stderr)

        # Log and print
        elapsed = time.time() - t0
        finite_oracle = oracle_cp[np.isfinite(oracle_cp)]
        mean_oracle = float(finite_oracle.mean()) if len(finite_oracle) else float("nan")
        with open(self._log_path, "a") as f:
            f.write(
                f"{self._cycle},{N},{K},{len(oracle_idx)},{len(gp_idx)},"
                f"{int(is_anchor)},{self.target_cp:.4f},{self._ltm.size()},"
                f"{mean_oracle:.4f},{gp_rmse_train:.4f},{gp_rmse_cv:.4f},"
                f"{gp_mae_cv:.4f},{elapsed:.2f}\n"
            )
        print(
            f"[GPRoutedV3] cycle={self._cycle} N={N} oracled={len(oracle_idx)} "
            f"masked={len(gp_idx)} ltm={self._ltm.size()} "
            f"GP_RMSE_train={gp_rmse_train:.3f} GP_RMSE_cv5={gp_rmse_cv:.3f} "
            f"GP_MAE_cv5={gp_mae_cv:.3f} mean_oracle_Cp={mean_oracle:.3f} "
            f"anchor={is_anchor} cold={is_cold} {elapsed:.1f}s",
            flush=True,
        )
        self._cycle += 1
        return rewards

    # --------------------------------------------------------------------

    def _read_atoms(self, samples) -> list:
        if isinstance(samples, (tuple, list)) and len(samples) >= 2:
            structures, xyz_path = samples[0], samples[1]
        else:
            structures, xyz_path = samples, ""
        if xyz_path and os.path.isfile(xyz_path):
            atoms = ase_read(xyz_path, index=":")
            if not isinstance(atoms, list):
                atoms = [atoms]
            return atoms
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        return [adaptor.get_atoms(s) for s in structures]
