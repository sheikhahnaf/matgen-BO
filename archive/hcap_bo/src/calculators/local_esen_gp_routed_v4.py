"""LocalESEN_GPRoutedV4 — v4 BO+RL with TRUE oracle-call savings.

Sibling of v3 (v3 fixed reward quality; v4 fixes oracle COUNT).
Hydra: reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRoutedV4

Differences vs v3:
    (A) GP_ANCHOR_EVERY default = 999  (no recalibration anchors → 0 anchor oracles)
    (B) GP_TOP_K default = 4 (was 8 in v2/v3)
    (C) GP_K_RATIO default = 0.5 → cap K at min(K, ceil(0.5*N_post)).
        Ensures K < N_post even when filter is loose.
    (D) GP_COLD_START_MIN default = 0 (warm-start pool of 446 covers cold-start
        fully — never trigger force-oracle-all-on-cold).

Inherited from v3:
    - Honest reward (NaN for non-oracle samples → masked from RL gradient)
    - DPP top-K diversity selection
    - Warm-start importance decay (FixedNoiseGP per-row noise = noise_min / 0.9^age)
    - GP RMSE/MAE 5-fold CV logging every cycle

Total oracle calls expected per 20-cycle run with K=4, K_ratio=0.5, no anchors:
    ≤ 4 × 20 = 80 (vs ≥ 180 in v2/v3 accel; ~91 in cf BASE).
"""

from __future__ import annotations

import os
import sys
import time
import math
from typing import List, Optional

import numpy as np
from ase.io import read as ase_read

from src.calculators.local_esen import LocalESEN
from src.featurizer import ORBFeaturizer
from src.surrogate import HCapSurrogate
from src.ltm import LTM, canonical_atoms_id, atoms_to_json


def _scratch_path(*parts) -> str:
    sc = os.environ.get("SCRATCH", "/tmp")
    return os.path.join(sc, *parts)


def _probability_of_improvement(mu, sigma, target):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    return 1.0 - norm.cdf((target - mu) / sigma)


def _expected_improvement(mu, sigma, f_best):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - f_best) / sigma
    return (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)


def _greedy_dpp_topk(scores, features, K, lambda_div=0.5):
    if len(scores) <= K:
        return np.argsort(-scores)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
    fn = features / norms
    sim = fn @ fn.T
    selected = [int(np.argmax(scores))]
    remaining = set(range(len(scores)))
    remaining.discard(selected[0])
    for _ in range(K - 1):
        if not remaining:
            break
        rem = np.array(sorted(remaining))
        max_sim = sim[rem][:, selected].max(axis=1)
        adjusted = scores[rem] - lambda_div * max_sim
        pick = int(rem[np.argmax(adjusted)])
        selected.append(pick)
        remaining.discard(pick)
    return np.array(selected, dtype=int)


def _gp_loo_metrics(sur, Z, y):
    if len(y) < 4:
        return float("nan"), float("nan"), float("nan")
    mu_train, _ = sur.predict(Z)
    train_rmse = float(np.sqrt(np.mean((mu_train - y) ** 2)))
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


class LocalESEN_GPRoutedV4:
    """v4 — fewer oracle calls than v3 by killing anchor cycles + capping K."""

    def __init__(self, root_dir, task="heat_capacity", env_name=None, worker=1):
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "heat_capacity":
            raise ValueError(f"v4 only supports task='heat_capacity'")
        self.worker = int(worker)
        self._oracle = LocalESEN(root_dir=root_dir, task=task, env_name=env_name, worker=worker)

        import torch
        self.pca_components = int(os.environ.get("GP_PCA_COMPONENTS", "50"))
        # v4 defaults: aggressively cap oracle calls
        self.top_k = int(os.environ.get("GP_TOP_K", "4"))                   # was 8
        self.k_ratio = float(os.environ.get("GP_K_RATIO", "0.5"))           # cap K at 50% of N_post
        self.target_cp = float(os.environ.get("GP_TARGET_CP", "1.5"))
        self.anchor_every = int(os.environ.get("GP_ANCHOR_EVERY", "999"))   # was 5 → effectively disabled
        self.cold_start_min = int(os.environ.get("GP_COLD_START_MIN", "0")) # was 16/50 → 0 (warm-start covers)
        self.dpp_lambda = float(os.environ.get("GP_DPP_LAMBDA", "0.5"))
        self.ws_decay = float(os.environ.get("GP_WARMSTART_DECAY", "0.9"))
        self.ws_noise_min = float(os.environ.get("GP_WARMSTART_NOISE_MIN", "1e-6"))
        self.device = os.environ.get("GP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        ltm_path = os.environ.get(
            "GP_LTM_PATH",
            _scratch_path("matinvent-hcap-bo", "data", "ltm_phase3_v4.parquet"),
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
                print(f"[GPRoutedV4] queued {len(self._seed_pending)} warm-start "
                      f"samples from {seed_path}", file=sys.stderr)
            except Exception as e:
                print(f"[GPRoutedV4] warm-start load failed: {e}", file=sys.stderr)

        self._ltm = LTM(ltm_path)
        self.acquisition = os.environ.get("GP_ACQUISITION", "ei").lower()
        if self.acquisition not in ("ei", "pi"):
            self.acquisition = "ei"

        self._feat: Optional[ORBFeaturizer] = None
        self._sur: Optional[HCapSurrogate] = None
        self._cycle = 0

        self._log_path = os.path.join(root_dir, "gp_routed_v4_log.csv")
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w") as f:
                f.write(
                    "cycle,n_input,K,n_oracle,n_gp,is_anchor,target_cp,ltm_size,"
                    "mean_oracle_cp,gp_rmse_train,gp_rmse_cv5,gp_mae_cv5,"
                    "elapsed_s\n"
                )

        print(
            f"[GPRoutedV4] init: K={self.top_k} K_ratio={self.k_ratio} "
            f"anchor_every={self.anchor_every} cold_min={self.cold_start_min} "
            f"target={self.target_cp} acq={self.acquisition}",
            file=sys.stderr,
        )

    def calc(self, samples, label="tmp"):
        t0 = time.time()
        atoms_list = self._read_atoms(samples)
        N = len(atoms_list)
        if N == 0:
            return np.zeros(0, dtype=np.float64)

        if self._feat is None:
            self._feat = ORBFeaturizer(n_components=self.pca_components, device=self.device)

        # Featurize + re-embed warm-start under live PCA basis
        if not self._feat.is_fitted and self._seed_pending:
            seed_atoms = [a for a, _ in self._seed_pending]
            seed_y = np.array([y for _, y in self._seed_pending], dtype=np.float64)
            combined = seed_atoms + atoms_list
            Z_combined = self._feat.fit_transform(combined)
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
            print(f"[GPRoutedV4] re-embedded {len(seed_atoms)} warm-start rows",
                  file=sys.stderr)
            self._seed_pending = []
        elif not self._feat.is_fitted:
            Z = self._feat.fit_transform(atoms_list)
        else:
            Z = self._feat.transform(atoms_list)

        # K-cap: min(top_k, ceil(k_ratio * N))
        K_eff = min(self.top_k, max(1, int(math.ceil(self.k_ratio * N))))

        is_anchor = (self._cycle > 0) and (self._cycle % self.anchor_every == 0)
        is_cold = self._sur is None or self._ltm.size() < self.cold_start_min
        # Crucially: if K_eff >= N we still oracle all (can't help that), but K_eff is bounded.
        if is_cold or is_anchor or K_eff >= N:
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
            oracle_idx = _greedy_dpp_topk(pi, Z, K_eff, lambda_div=self.dpp_lambda)
            oracle_idx = np.sort(oracle_idx)
            mask_o = np.zeros(N, dtype=bool)
            mask_o[oracle_idx] = True
            gp_idx = np.where(~mask_o)[0]

        rewards = np.empty(N, dtype=np.float64)
        # Honest reward: non-oracle → NaN
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
                    print(f"[GPRoutedV4] _phonon_task failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    cp_list.append(float("nan"))
            cp_subset = np.array(cp_list, dtype=np.float64)
            oracle_cp[: len(cp_subset)] = cp_subset
            for j, i in enumerate(oracle_idx):
                cp_val = oracle_cp[j]
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

        # Warm-start decay GP refit
        gp_rmse_train = gp_rmse_cv = gp_mae_cv = float("nan")
        df_full = self._ltm.load()
        if len(df_full) >= 4:
            Z_train = np.stack(df_full["Z_pca50"].apply(np.asarray).tolist())
            y_train = df_full["y_cp"].to_numpy(dtype=np.float64)
            cyc = df_full["cycle_id"].to_numpy()
            ages = np.where(cyc < 0, self._cycle + 1, self._cycle - cyc).astype(float)
            ages = np.maximum(ages, 0.0)
            weights = self.ws_decay ** ages
            y_var = self.ws_noise_min / np.maximum(weights, 1e-3)
            try:
                self._sur = HCapSurrogate(device=self.device)
                self._sur.fit(Z_train, y_train, y_var=y_var)
                gp_rmse_train, gp_rmse_cv, gp_mae_cv = _gp_loo_metrics(
                    self._sur, Z_train, y_train
                )
            except Exception as e:
                print(f"[GPRoutedV4] GP fit failed: {type(e).__name__}: {e}",
                      file=sys.stderr)

        elapsed = time.time() - t0
        finite_oracle = oracle_cp[np.isfinite(oracle_cp)]
        mean_oracle = float(finite_oracle.mean()) if len(finite_oracle) else float("nan")
        with open(self._log_path, "a") as f:
            f.write(
                f"{self._cycle},{N},{K_eff},{len(oracle_idx)},{len(gp_idx)},"
                f"{int(is_anchor)},{self.target_cp:.4f},{self._ltm.size()},"
                f"{mean_oracle:.4f},{gp_rmse_train:.4f},{gp_rmse_cv:.4f},"
                f"{gp_mae_cv:.4f},{elapsed:.2f}\n"
            )
        print(
            f"[GPRoutedV4] cycle={self._cycle} N={N} K_eff={K_eff} "
            f"oracled={len(oracle_idx)} masked={len(gp_idx)} ltm={self._ltm.size()} "
            f"GP_RMSE_train={gp_rmse_train:.3f} GP_RMSE_cv5={gp_rmse_cv:.3f} "
            f"GP_MAE_cv5={gp_mae_cv:.3f} mean_oracle_Cp={mean_oracle:.3f} "
            f"anchor={is_anchor} cold={is_cold} {elapsed:.1f}s",
            flush=True,
        )
        self._cycle += 1
        return rewards

    def _read_atoms(self, samples):
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
