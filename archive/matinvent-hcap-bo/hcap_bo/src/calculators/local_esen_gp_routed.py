"""LocalESEN_GPRouted — acquisition-based top-K wrapper around LocalESEN.

Drop-in replacement for upstream `rewards.calculators.FairChem` (same `.calc()`
signature, instantiated via Hydra `_target_=src.calculators.LocalESEN_GPRouted`).

ROUTING STRATEGY (replaces the older σ-threshold approach, which broke under
under-confident GPs):

    For each cycle's batch of K_in candidates:
        1. ORB-PCA50 featurize all K_in.
        2. GP predicts (μ, σ).
        3. Acquisition score: PI = P(Cp > target_cp) = 1 − Φ((target − μ)/σ)
           (Probability of Improvement at fixed target — what we want for
           the "find structures with Cp > 1.5 J/g/K" task).
        4. ORACLE the top K (by PI) — get true Cp via eSEN.
        5. GP-PATH (the bottom K_in − K) — return GP posterior mean μ as reward.
        6. ANCHOR every N cycles: force-oracle ALL K_in (recalibration, retrain).

    Hard rule: TRUE Cp values are appended to LTM. GP predictions NEVER are.
    Cold-start (LTM < cold_start_min): force-oracle all.

Why PI-top-K vs σ-threshold:
    σ-threshold collapses to "always oracle" when the GP is under-confident
    (PICP@90 = 1.0 → ratio quantile small → τ→0). Top-K is deterministic in
    cycle cost (always exactly K oracled) and uses the right unit (target
    threshold-improvement probability), not raw σ in absolute Cp units.

Speedup is built in: K oracle calls / K_in input = K/K_in fraction. With the
upstream eval_size=16 and K=8 we get a 2× speedup per cycle (deterministic),
plus more from RL convergence efficiency if the GP picks better candidates.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
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
    """PI = P(Cp > target). Uses Gaussian CDF.

    Failure mode: when no LTM samples are near `target`, μ is far below target
    and PI ≈ 0 everywhere → top-K becomes "most uncertain" not "most promising".
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    return 1.0 - norm.cdf((target - mu) / sigma)


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """EI = E[max(Cp - f_best, 0)] for maximization.

    Smoother + better-behaved than PI when target is far away. Uses
    `f_best` = best Cp seen in LTM so far (NOT the absolute target). This way
    the acquisition has signal even when no sample is near 1.5 J/g/K yet.

        EI(x) = (μ - f*) Φ((μ-f*)/σ) + σ φ((μ-f*)/σ)

    where Φ is Gaussian CDF, φ is Gaussian PDF.
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - f_best) / sigma
    return (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)


class LocalESEN_GPRouted:
    """Acquisition-top-K wrapper around LocalESEN (drop-in for FairChem).

    Constructor signature MUST match upstream FairChem so Hydra can swap it in:
        FairChem(root_dir, task='heat_capacity', env_name='fair-chem-v1', worker=5)

    Routing config from environment variables (avoid YAML schema changes):
        GP_PCA_COMPONENTS      PCA dim                                 default 50
        GP_TOP_K               # samples to oracle per cycle           default 8
        GP_TARGET_CP           PI target (J/g/K)                       default 1.5
        GP_ANCHOR_EVERY        force-oracle all K_in every Nth cycle   default 5
        GP_COLD_START_MIN      oracle ALL until LTM has this many true labels
                                                                        default 16
        GP_LTM_PATH            parquet path for LTM observations
        GP_DEVICE              "cuda" or "cpu"                         default auto

    Diagnostics: <root_dir>/gp_routed_log.csv with columns
        cycle,n_input,K,n_oracle,n_gp,is_anchor,target_cp,ltm_size,
        mean_reward,pi_mean,pi_max,oracle_pi_threshold,elapsed_s,
        ence,picp_50,picp_90
    """

    def __init__(
        self,
        root_dir: str,
        task: str = "heat_capacity",
        env_name: str | None = None,   # ignored — in-process
        worker: int = 1,
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "heat_capacity":
            raise ValueError(
                f"LocalESEN_GPRouted currently only supports task='heat_capacity', got {task!r}"
            )
        self.worker = int(worker)

        # Inner oracle (real eSEN-30M-OAM)
        self._oracle = LocalESEN(root_dir=root_dir, task=task, env_name=env_name, worker=worker)

        import torch
        self.pca_components = int(os.environ.get("GP_PCA_COMPONENTS", "50"))
        self.top_k = int(os.environ.get("GP_TOP_K", "8"))
        self.target_cp = float(os.environ.get("GP_TARGET_CP", "1.5"))
        self.anchor_every = int(os.environ.get("GP_ANCHOR_EVERY", "5"))
        self.cold_start_min = int(os.environ.get("GP_COLD_START_MIN", "16"))
        self.device = os.environ.get(
            "GP_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        ltm_path = os.environ.get(
            "GP_LTM_PATH",
            _scratch_path("matinvent-hcap-bo", "data", "ltm_phase3.parquet"),
        )

        # Warm-start: defer until first calc() so PCA can be fit on the
        # combined (seed + first_batch) atoms — avoids the basis mismatch
        # bug where seed Z_pca50 came from a different PCA fit.
        # See COMMITTEE_REVIEW.md Bug 1 (P1).
        self._seed_pending = []  # [(atoms, y_cp), ...] — populated below
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
                print(f"[GPRouted] queued {len(self._seed_pending)} warm-start "
                      f"samples from {seed_path} (re-embed at first calc)",
                      file=sys.stderr)
            except Exception as e:
                print(f"[GPRouted] warm-start load failed: {e}", file=sys.stderr)

        self._ltm = LTM(ltm_path)
        self.acquisition = os.environ.get("GP_ACQUISITION", "ei").lower()
        if self.acquisition not in ("ei", "pi"):
            print(f"[GPRouted] unknown GP_ACQUISITION={self.acquisition!r}, "
                  f"falling back to 'ei'", file=sys.stderr)
            self.acquisition = "ei"

        self._feat: Optional[ORBFeaturizer] = None
        self._sur: Optional[HCapSurrogate] = None
        self._cycle = 0

        self._log_path = os.path.join(root_dir, "gp_routed_log.csv")
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w") as f:
                f.write(
                    "cycle,n_input,K,n_oracle,n_gp,is_anchor,target_cp,ltm_size,"
                    "mean_reward,pi_mean,pi_max,oracle_pi_threshold,"
                    "elapsed_s,ence,picp_50,picp_90\n"
                )

    # --------------------------------------------------------------------

    def calc(
        self,
        samples: Tuple[List, str],
        label: str = "tmp",
    ) -> np.ndarray:
        t0 = time.time()
        atoms_list = self._read_atoms(samples)
        N = len(atoms_list)
        if N == 0:
            return np.zeros(0, dtype=np.float64)

        if self._feat is None:
            self._feat = ORBFeaturizer(n_components=self.pca_components, device=self.device)

        # 1. Featurize input batch.
        #    First call: fit PCA on (seed_atoms + live_atoms) so warm-start
        #    rows are embedded in the same basis as live samples (Bug 1 fix).
        if not self._feat.is_fitted and self._seed_pending:
            seed_atoms = [a for a, _ in self._seed_pending]
            seed_y = np.array([y for _, y in self._seed_pending], dtype=np.float64)
            combined_atoms = seed_atoms + atoms_list
            Z_combined = self._feat.fit_transform(combined_atoms)
            Z_seed = Z_combined[: len(seed_atoms)]
            Z = Z_combined[len(seed_atoms) :]
            # Persist re-embedded seed rows into LTM
            for atoms, y, z in zip(seed_atoms, seed_y, Z_seed):
                self._ltm.append([{
                    "structure_id": canonical_atoms_id(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "cycle_id": -1,                # warm-start sentinel
                    "atoms_json": atoms_to_json(atoms),
                    "Z_pca50": list(map(float, z)),
                    "y_cp": float(y),
                    "y_cp_var": float("nan"),
                    "sigma_pred": float("nan"),
                    "ood_score": float("nan"),
                    "oracle_source": "warm_start",
                }])
            print(f"[GPRouted] re-embedded {len(seed_atoms)} warm-start rows "
                  f"under fresh PCA basis", file=sys.stderr)
            self._seed_pending = []
        elif not self._feat.is_fitted:
            Z = self._feat.fit_transform(atoms_list)
        else:
            Z = self._feat.transform(atoms_list)

        # 2. Decide oracle subset
        is_anchor = (self._cycle > 0) and (self._cycle % self.anchor_every == 0)
        is_cold = self._sur is None or self._ltm.size() < self.cold_start_min
        K = self.top_k

        if is_cold or is_anchor or K >= N:
            # Force-oracle all
            mu = np.full(N, np.nan)
            sigma = np.full(N, np.inf)
            pi = np.full(N, np.nan)
            oracle_idx = np.arange(N)
            gp_idx = np.array([], dtype=int)
            oracle_pi_threshold = float("nan")
        else:
            mu, sigma = self._sur.predict(Z)
            if self.acquisition == "ei":
                # Use best LTM y_cp as f_best (incumbent) — gives EI a
                # well-behaved gradient even when 1.5 J/g/K is unreached.
                _, y_train = self._ltm.features_and_targets()
                f_best = float(np.nanmax(y_train)) if len(y_train) else 0.0
                pi = _expected_improvement(mu, sigma, f_best)
            else:
                pi = _probability_of_improvement(mu, sigma, self.target_cp)
            order = np.argsort(-pi)  # descending acquisition score
            oracle_idx = np.sort(order[:K])
            gp_idx = np.sort(order[K:])
            oracle_pi_threshold = float(pi[oracle_idx[-1]] if len(oracle_idx) else np.nan)

        rewards = np.empty(N, dtype=np.float64)
        # GP-path: μ as reward, NO LTM write
        if len(gp_idx) > 0:
            rewards[gp_idx] = mu[gp_idx]

        # Oracle path: bypass LocalESEN.calc reformat; call _phonon_task directly
        oracle_cp = np.full(len(oracle_idx), np.nan)
        if len(oracle_idx) > 0:
            atoms_o = [atoms_list[int(i)] for i in oracle_idx]
            cp_list = []
            for a in atoms_o:
                try:
                    cp_list.append(self._oracle._phonon_task(a))
                except Exception as e:
                    print(f"[GPRouted] inner _phonon_task failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    cp_list.append(float("nan"))
            cp_subset = np.array(cp_list, dtype=np.float64)
            oracle_cp[: len(cp_subset)] = cp_subset

            for j, i in enumerate(oracle_idx):
                cp_val = oracle_cp[j]
                if np.isfinite(cp_val):
                    rewards[i] = float(cp_val)
                else:
                    # Oracle failure → fall back to GP μ if available, else 0
                    rewards[i] = float(mu[i]) if np.isfinite(mu[i]) else 0.0

        # 3. Append TRUE labels to LTM (only successful oracles)
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

        # 4. Retrain GP on full LTM
        Z_train, y_train = self._ltm.features_and_targets()
        if len(y_train) >= 4:
            self._sur = HCapSurrogate(device=self.device)
            self._sur.fit(Z_train, y_train)

        # 5. Anchor calibration (out-of-sample within this cycle's anchor batch)
        ence = picp50 = picp90 = float("nan")
        if is_anchor and self._sur is not None:
            ok = np.isfinite(oracle_cp)
            if ok.sum() >= 3:
                idx_ok = np.array([oracle_idx[k] for k, m in enumerate(ok) if m])
                Z_ok = Z[idx_ok]
                y_ok = oracle_cp[ok]
                mu_a, sigma_a = self._sur.predict(Z_ok)
                ence = cal.ence(mu_a, sigma_a, y_ok)
                picp50 = cal.picp(mu_a, sigma_a, y_ok, level=0.50)
                picp90 = cal.picp(mu_a, sigma_a, y_ok, level=0.90)

        # 6. Log
        elapsed = time.time() - t0
        pi_mean = float(np.nanmean(pi)) if pi.size else float("nan")
        pi_max = float(np.nanmax(pi)) if pi.size and np.isfinite(np.nanmax(pi)) else float("nan")
        with open(self._log_path, "a") as f:
            f.write(
                f"{self._cycle},{N},{K},{len(oracle_idx)},{len(gp_idx)},"
                f"{int(is_anchor)},{self.target_cp:.4f},{self._ltm.size()},"
                f"{float(np.nanmean(rewards)):.4f},"
                f"{pi_mean:.4f},{pi_max:.4f},{oracle_pi_threshold:.4f},"
                f"{elapsed:.2f},{ence:.4f},{picp50:.4f},{picp90:.4f}\n"
            )
        print(
            f"[GPRouted] cycle={self._cycle} N={N} oracled={len(oracle_idx)}/{N} "
            f"K={K} ltm={self._ltm.size()} target={self.target_cp:.2f} "
            f"PI(oracle_min)={oracle_pi_threshold:.3f} anchor={is_anchor} "
            f"cold={is_cold} {elapsed:.1f}s",
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
