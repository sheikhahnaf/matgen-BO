"""LocalESEN_BM_GPRoutedV4 — v4 BM accel (mirrors v4 Cp).

Same v4 mechanics:
    - Honest reward (NaN for non-oracle samples)
    - DPP top-K diversity selection
    - Warm-start importance decay
    - GP RMSE/MAE 5-fold CV log per cycle
    - K=4 default, K_RATIO=0.5, ANCHOR_EVERY=999, COLD_START_MIN=0
Different oracle: LocalESEN_BM (Birch-Murnaghan EOS bulk modulus).

LTM column `y_cp` is reused for K_VRH (GPa) — keeps the surrogate code generic;
the column name is just a label.

Hydra:
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_BM_GPRoutedV4
"""

from __future__ import annotations

import os
import sys
import time
import math
from typing import Optional

import numpy as np
from ase.io import read as ase_read

from src.calculators.local_esen_bm import LocalESEN_BM
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


class LocalESEN_BM_GPRoutedV4:
    """v4 BM accel — mirrors v4 Cp but oracles bulk modulus via Birch-Murnaghan."""

    def __init__(self, root_dir, task="bulk_modulus", env_name=None, worker=1):
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "bulk_modulus":
            raise ValueError(f"v4 BM only supports task='bulk_modulus'")
        self.worker = int(worker)
        self._oracle = LocalESEN_BM(root_dir=root_dir, task=task,
                                    env_name=env_name, worker=worker)

        import torch
        self.pca_components = int(os.environ.get("GP_PCA_COMPONENTS", "50"))
        # v4 oracle-saving defaults
        self.top_k = int(os.environ.get("GP_TOP_K", "4"))
        self.k_ratio = float(os.environ.get("GP_K_RATIO", "0.5"))
        # BM target: ascending → use a high target value for PI; EI uses incumbent
        self.target_cp = float(os.environ.get("GP_TARGET_CP", "300.0"))
        self.anchor_every = int(os.environ.get("GP_ANCHOR_EVERY", "999"))
        self.cold_start_min = int(os.environ.get("GP_COLD_START_MIN", "0"))
        self.dpp_lambda = float(os.environ.get("GP_DPP_LAMBDA", "0.5"))
        self.ws_decay = float(os.environ.get("GP_WARMSTART_DECAY", "0.9"))
        self.ws_noise_min = float(os.environ.get("GP_WARMSTART_NOISE_MIN", "1e-6"))
        self.device = os.environ.get("GP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        ltm_path = os.environ.get(
            "GP_LTM_PATH",
            _scratch_path("matinvent-hcap-bo", "data", "bm", "ltm_phase3_v4_bm.parquet"),
        )

        self._seed_pending = []
        seed_path = os.environ.get("GP_LTM_SEED", "").strip()
        if seed_path and os.path.exists(seed_path) and not os.path.exists(ltm_path):
            try:
                import pandas as _pd
                from src.ltm import atoms_from_json
                df = _pd.read_parquet(seed_path)
                df = df.dropna(subset=["y_cp"])  # column reused as y_K
                for _, row in df.iterrows():
                    try:
                        atoms = atoms_from_json(row["atoms_json"])
                        self._seed_pending.append((atoms, float(row["y_cp"])))
                    except Exception:
                        pass
                print(f"[BMGPRoutedV4] queued {len(self._seed_pending)} warm-start "
                      f"BM samples from {seed_path}", file=sys.stderr)
            except Exception as e:
                print(f"[BMGPRoutedV4] warm-start load failed: {e}", file=sys.stderr)

        self._ltm = LTM(ltm_path)
        self.acquisition = os.environ.get("GP_ACQUISITION", "ei").lower()
        if self.acquisition not in ("ei", "pi"):
            self.acquisition = "ei"

        self._feat: Optional[ORBFeaturizer] = None
        self._sur: Optional[HCapSurrogate] = None
        self._cycle = 0

        self._log_path = os.path.join(root_dir, "bm_gp_routed_v4_log.csv")
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w") as f:
                f.write(
                    "cycle,n_input,K,n_oracle,n_gp,is_anchor,target_K,ltm_size,"
                    "mean_oracle_K_GPa,gp_rmse_train,gp_rmse_cv5,gp_mae_cv5,"
                    "elapsed_s\n"
                )

        print(
            f"[BMGPRoutedV4] init: K={self.top_k} K_ratio={self.k_ratio} "
            f"anchor_every={self.anchor_every} cold_min={self.cold_start_min} "
            f"target_K={self.target_cp} acq={self.acquisition}",
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
                    "y_cp": float(y),  # column reused as K_VRH (GPa)
                    "y_cp_var": float("nan"),
                    "sigma_pred": float("nan"),
                    "ood_score": float("nan"),
                    "oracle_source": "warm_start",
                }])
            print(f"[BMGPRoutedV4] re-embedded {len(seed_atoms)} warm-start rows",
                  file=sys.stderr)
            self._seed_pending = []
        elif not self._feat.is_fitted:
            Z = self._feat.fit_transform(atoms_list)
        else:
            Z = self._feat.transform(atoms_list)

        K_eff = min(self.top_k, max(1, int(math.ceil(self.k_ratio * N))))
        is_anchor = (self._cycle > 0) and (self._cycle % self.anchor_every == 0)
        is_cold = self._sur is None or self._ltm.size() < self.cold_start_min

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
        if len(gp_idx) > 0:
            rewards[gp_idx] = np.nan

        oracle_K = np.full(len(oracle_idx), np.nan)
        if len(oracle_idx) > 0:
            atoms_o = [atoms_list[int(i)] for i in oracle_idx]
            K_list = []
            for a in atoms_o:
                try:
                    K_list.append(self._oracle._bm_task(a))
                except Exception as e:
                    print(f"[BMGPRoutedV4] _bm_task failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    K_list.append(float("nan"))
            K_subset = np.array(K_list, dtype=np.float64)
            oracle_K[: len(K_subset)] = K_subset
            for j, i in enumerate(oracle_idx):
                K_val = oracle_K[j]
                rewards[i] = float(K_val) if np.isfinite(K_val) else np.nan

        for j, i in enumerate(oracle_idx):
            K_val = oracle_K[j]
            if not np.isfinite(K_val):
                continue
            atoms = atoms_list[int(i)]
            self._ltm.append([{
                "structure_id": canonical_atoms_id(atoms),
                "formula": atoms.get_chemical_formula(),
                "cycle_id": int(self._cycle),
                "atoms_json": atoms_to_json(atoms),
                "Z_pca50": list(map(float, Z[i])),
                "y_cp": float(K_val),  # K_VRH stored in y_cp column
                "y_cp_var": float("nan"),
                "sigma_pred": float(sigma[i]) if np.isfinite(sigma[i]) else float("nan"),
                "ood_score": float("nan"),
                "oracle_source": "anchor_batch" if is_anchor else (
                    "seed_pool" if is_cold else "oracle"
                ),
            }])

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
                print(f"[BMGPRoutedV4] GP fit failed: {type(e).__name__}: {e}",
                      file=sys.stderr)

        elapsed = time.time() - t0
        finite_K = oracle_K[np.isfinite(oracle_K)]
        mean_K = float(finite_K.mean()) if len(finite_K) else float("nan")
        with open(self._log_path, "a") as f:
            f.write(
                f"{self._cycle},{N},{K_eff},{len(oracle_idx)},{len(gp_idx)},"
                f"{int(is_anchor)},{self.target_cp:.4f},{self._ltm.size()},"
                f"{mean_K:.4f},{gp_rmse_train:.4f},{gp_rmse_cv:.4f},"
                f"{gp_mae_cv:.4f},{elapsed:.2f}\n"
            )
        print(
            f"[BMGPRoutedV4] cycle={self._cycle} N={N} K_eff={K_eff} "
            f"oracled={len(oracle_idx)} masked={len(gp_idx)} ltm={self._ltm.size()} "
            f"GP_RMSE_train={gp_rmse_train:.3f} GP_RMSE_cv5={gp_rmse_cv:.3f} "
            f"GP_MAE_cv5={gp_mae_cv:.3f} mean_oracle_K_GPa={mean_K:.2f} "
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
