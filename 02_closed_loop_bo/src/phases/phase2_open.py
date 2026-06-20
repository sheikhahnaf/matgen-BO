"""Phase 2 — open-loop two-lever BO over MatterGen samples.

NO RL update. We just run the pipeline:
    - Generate (or load) candidate batches
    - Screen via GP + qLogNEI + k-DPP (Lever 1) → top K=16
    - σ-route per sample (Lever 2) → eSEN or GP μ
    - Append true Cp rows to LTM
    - Anchor batch every 5 cycles (force-eSEN all 16)
    - Retrain GP each cycle
    - Log calibration metrics every cycle

Outputs:
    results/<run>/
        ltm.parquet
        cycles.csv     per-cycle metrics
        calib_<cycle>.json
        config.yaml    snapshot
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.featurizer import get_featurizer
from src.surrogate import HCapSurrogate
from src.acquisition import select_topk
from src.oracle import get_oracle
from src.router import sigma_route, ltm_rows_for_oracle_results, calibrate_threshold_from_picp
from src.ltm import LTM, canonical_atoms_id
from src import calibration as cal


def _instantiate(cfg_block):
    """Tiny `_target_`-aware instantiator (Hydra-compatible subset).

    Resolves `_target_: 'pkg.mod.ClassName'` plus all other keys as kwargs.
    Skips OmegaConf interpolation by converting to a plain dict first.
    """
    from omegaconf import OmegaConf as _OC
    import importlib as _il
    d = _OC.to_container(cfg_block, resolve=True)
    target = d.pop("_target_")
    mod_path, _, name = target.rpartition(".")
    cls = getattr(_il.import_module(mod_path), name)
    return cls(**d)


def _make_pool_provider(cfg, n_total: int, output_dir: Path):
    """Returns a callable `provider(cycle_idx, n) -> list[Atoms]`.

    Three sources, in priority order:
    (1) `cfg.generation.adapter` (a Hydra-style `_target_` block)  — generates
        per cycle via the adapter, saving cycle samples to disk.
    (2) `cfg.generation.pool_path`                                 — slices a
        pre-generated extxyz pool.
    (3) synthetic random-bulk fallback (plumbing only).
    """
    adapter_cfg = None
    if hasattr(cfg.generation, "adapter") and cfg.generation.adapter is not None:
        try:
            _ = cfg.generation.adapter._target_
            adapter_cfg = cfg.generation.adapter
        except Exception:
            adapter_cfg = None

    if adapter_cfg is not None:
        adapter = _instantiate(adapter_cfg)
        print(f"[phase2] using diffusion adapter: {getattr(adapter, 'name', adapter.__class__.__name__)}")

        def _provider(cycle_idx: int, n: int):
            seed = int(cfg.experiment.seed) + int(cycle_idx)
            atoms_list = adapter.sample(n=int(n), seed=seed)
            try:
                from ase.io import write
                write(output_dir / f"cycle{cycle_idx:03d}_samples.xyz",
                      atoms_list, format="extxyz")
            except Exception as e:
                print(f"[phase2]   warning: could not save cycle samples: {e}")
            return atoms_list

        return _provider, "adapter"

    src_path = cfg.generation.get("pool_path", None) if hasattr(cfg.generation, "get") else None
    if src_path and Path(src_path).exists():
        from ase.io import read
        full_pool = read(src_path, index=":")

        def _provider(cycle_idx: int, n: int):
            return full_pool[cycle_idx * n : (cycle_idx + 1) * n]

        return _provider, "pool_path"

    from ase.build import bulk
    rng = np.random.default_rng(cfg.experiment.seed)
    elements = ["Si", "Al", "Mg", "Cu", "Fe", "Ti", "Zn", "Na", "Ca", "Sr"]
    synthetic_pool = []
    for _ in range(n_total):
        e = rng.choice(elements)
        a = float(rng.uniform(2.8, 4.5))
        try:
            synthetic_pool.append(bulk(e, "fcc", a=a, cubic=True))
        except Exception:
            synthetic_pool.append(bulk(e, "bcc", a=a, cubic=True))

    def _provider(cycle_idx: int, n: int):
        return synthetic_pool[cycle_idx * n : (cycle_idx + 1) * n]

    return _provider, "synthetic"


def run(config_path: str, output_dir: str) -> int:
    cfg = OmegaConf.load(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase2] device={device}")

    cycles = int(cfg.generation.cycles)
    Bp = int(cfg.generation.batch_size)        # candidates generated per cycle (Lever 1 input)
    K = int(cfg.acquisition.q)                 # screened top-K (Lever 1 output)
    anchor_every = int(cfg.oracle.anchor_every)
    tau = float(cfg.acquisition.get("sigma_threshold", 0.05))  # initial τ in Cp units

    # Candidate provider (real diffusion adapter, pre-saved pool, or synthetic).
    provider, src_kind = _make_pool_provider(cfg, n_total=cycles * Bp, output_dir=out)
    print(f"[phase2] pool source = {src_kind}; cycles={cycles}, B'={Bp}, K={K}, τ₀={tau:.3f}")

    feat = get_featurizer(
        kind=cfg.featurizer.kind,
        n_components=cfg.featurizer.pca_components,
        device=device,
    )

    ltm = LTM(out / "ltm.parquet")

    sur: Optional[HCapSurrogate] = None
    oracle = get_oracle(
        kind=cfg.oracle.kind,
        env_prefix=str(cfg.oracle.env_prefix),
        n_workers=int(cfg.oracle.get("n_workers", 1)),
        scratch_dir=str(out),
    )

    cycles_log = []

    for cycle in range(cycles):
        t0 = time.time()
        batch = provider(cycle, Bp)
        if not batch:
            break

        # Featurize the whole batch (Z for screening + routing)
        if not feat.is_fitted:
            Z_batch = feat.fit_transform(batch)
        else:
            Z_batch = feat.transform(batch)

        # --- Lever 1: GP-screen down to K
        if sur is None or sur.model is None:
            # cold start: pick first K
            sel = np.arange(min(K, len(batch)))
        else:
            sel = select_topk(
                sur, Z_batch, k=min(K, len(batch)),
                diversity="kdpp", seed=cycle,
            )
        atoms_K = [batch[int(i)] for i in sel]
        Z_K = Z_batch[sel]

        # --- Lever 2: σ-route per sample
        is_anchor = (cycle > 0) and (cycle % anchor_every == 0)
        rr = sigma_route(
            Z_candidates=Z_K,
            atoms_candidates=atoms_K,
            surrogate=sur,
            oracle=oracle,
            threshold=tau,
            cycle_id=cycle,
            force_anchor=is_anchor,
        )

        # --- Calibration on the anchor batch FIRST (out-of-sample; before retrain)
        # On anchor cycles, every K candidate has a true oracle Cp from rr,
        # and the GP at this point was trained only on PRIOR cycles' LTM →
        # mu_a/sigma_a here are out-of-sample for the anchor set.
        cal_rec = {}
        if is_anchor and sur is not None and rr.n_oracle_calls == len(atoms_K):
            ok_mask = ~rr.oracle_failed  # only successful oracle points
            if ok_mask.sum() >= 3:
                idx_ok = rr.oracle_idx[ok_mask]
                Z_ok = Z_K[idx_ok]
                y_ok = rr.oracle_cp[ok_mask]
                mu_a, sigma_a = sur.predict(Z_ok)
                cal_rec = {
                    "ence": cal.ence(mu_a, sigma_a, y_ok),
                    "picp_50": cal.picp(mu_a, sigma_a, y_ok, level=0.50),
                    "picp_90": cal.picp(mu_a, sigma_a, y_ok, level=0.90),
                    "nll": cal.nll_gauss(mu_a, sigma_a, y_ok),
                }
                # Recalibrate τ from the *out-of-sample* anchor errors.
                # NaN → "no update" (too few points or undefined ratio).
                errs = np.abs(mu_a - y_ok)
                tau_new = calibrate_threshold_from_picp(sigma_a, errs, target_picp=0.90)
                if np.isfinite(tau_new) and tau_new > 0:
                    tau = float(tau_new)
                with open(out / f"calib_cycle{cycle:03d}.json", "w") as f:
                    json.dump(cal_rec, f, indent=2)

        # --- LTM update (TRUE labels only)
        sids = [canonical_atoms_id(a) for a in atoms_K]
        formulas = [a.get_chemical_formula() for a in atoms_K]
        rows = ltm_rows_for_oracle_results(rr, Z_K, atoms_K, sids, formulas)
        n_added = ltm.append(rows)

        # --- Retrain GP on full LTM (now includes the anchor labels)
        Z_train, y_train = ltm.features_and_targets()
        if len(y_train) >= 4:
            sur = HCapSurrogate(device=device)
            sur.fit(Z_train, y_train)

        cycles_log.append({
            "cycle": cycle,
            "n_oracle": int(rr.n_oracle_calls),
            "n_added_to_ltm": int(n_added),
            "ltm_size": int(ltm.size()),
            "mean_reward": float(np.nanmean(rr.rewards)),
            "max_reward": float(np.nanmax(rr.rewards)),
            "is_anchor": bool(is_anchor),
            "tau": float(tau),
            **{f"cal_{k}": v for k, v in cal_rec.items()},
            "elapsed_s": float(time.time() - t0),
        })
        print(
            f"[phase2] cycle {cycle:02d}  oracled={rr.n_oracle_calls:2d}/{K}  "
            f"ltm={ltm.size():3d}  reward μ={np.nanmean(rr.rewards):.3f}  "
            f"τ={tau:.3f}  anchor={is_anchor}  {time.time()-t0:.1f}s"
        )

        # save intermediate cycles.csv every cycle
        pd.DataFrame(cycles_log).to_csv(out / "cycles.csv", index=False)

    print(f"[phase2] done. final LTM size = {ltm.size()}")
    return 0
