"""3-way oracle cross-comparison: eSEN vs ORB-phonon vs UMA-phonon on a fixed
small validation pool (Si/MgO/Cu/Al2O3/AlN). Quantifies the systematic Cp
offset between the three backends so we know how to reconcile any cross-
oracle results downstream.

Each oracle returns Cp@300K in J/g/K. We expect:
    - All three within 10-20% on simple solids (Si, Cu).
    - Larger spread on complex solids (Al2O3, AlN) — interesting datapoint.
    - Imaginary modes → NaN for ORB/UMA pipelines (plain phonopy).

Outputs:
    results/<run>/cp_3way.csv     formula × oracle table
    results/<run>/runtimes.csv    per-structure runtimes
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.build import bulk
from omegaconf import OmegaConf

from src.oracle import get_oracle


def _validation_pool() -> tuple[list, list[str]]:
    """Five canonical small crystals — varied bonding + simple cells.

    All ASE-build-supported. Mix of: covalent (Si), ionic (MgO, CaF2),
    metallic (Cu), and polar covalent (AlN). 2-5 atoms/cell each.
    """
    structs = [
        bulk("Si", "diamond", a=5.43),
        bulk("MgO", "rocksalt", a=4.21),
        bulk("Cu", "fcc", a=3.61, cubic=True),
        bulk("AlN", "wurtzite", a=3.11, c=4.98),
        bulk("CaF2", "fluorite", a=5.46),
    ]
    labels = ["Si-diamond", "MgO-rocksalt", "Cu-fcc", "AlN-wurtzite", "CaF2-fluorite"]
    return structs, labels


def run(config_path: str, output_dir: str) -> int:
    cfg = OmegaConf.load(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-oracle env routing:
    #   - eSEN: needs fairchem 1.10 → matinvent-hcap-bo (where we just installed it)
    #   - ORB:  needs orb_models    → matinvent-hcap-bo (already has it)
    #   - UMA:  needs fairchem 2.14 → matinvent-hcap-fairchem (UMA-S-1p1 in registry there)
    scratch = os.environ.get("SCRATCH", "/scratch/user/ahnafalvi")
    env_for = {
        "esen": f"{scratch}/envs/matinvent-hcap-bo",
        "orb":  f"{scratch}/envs/matinvent-hcap-bo",
        "uma":  f"{scratch}/envs/matinvent-hcap-fairchem",
    }
    backends = ("esen", "orb", "uma")

    atoms_list, labels = _validation_pool()
    print(f"[3way] {len(atoms_list)} structures × {len(backends)} oracles")

    rows = []
    for kind in backends:
        try:
            oracle = get_oracle(
                kind=kind,
                env_prefix=env_for[kind],
                n_workers=1,
                scratch_dir=str(out),
            )
        except Exception as e:
            print(f"[3way] {kind} init FAILED: {e}")
            for lbl in labels:
                rows.append({"oracle": kind, "formula": lbl,
                             "cp_jpgK": np.nan, "elapsed_s": np.nan,
                             "error": f"{type(e).__name__}: {e}"})
            continue

        print(f"\n[3way] === {kind.upper()} ===")
        t0 = time.time()
        try:
            cp, mask = oracle.evaluate(atoms_list)
        except Exception as e:
            print(f"[3way] {kind} eval FAILED: {e}")
            cp = np.full(len(atoms_list), np.nan)
            mask = np.ones(len(atoms_list), dtype=bool)
        dt = time.time() - t0
        per_struct = dt / max(1, len(atoms_list))

        for lbl, c, m in zip(labels, cp, mask):
            print(f"  {lbl:18s}  {('FAIL' if m else f'{c:.4f} J/g/K')}")
            rows.append({"oracle": kind, "formula": lbl,
                         "cp_jpgK": float(c) if not m else np.nan,
                         "elapsed_s_total": dt,
                         "elapsed_s_per_struct": per_struct,
                         "failed": bool(m)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "cp_3way.csv", index=False)

    # Pivot to wide format for visual inspection
    pivot = df.pivot_table(index="formula", columns="oracle", values="cp_jpgK")
    pivot.to_csv(out / "cp_3way_wide.csv")
    print("\n[3way] Cp (J/g/K) — wide table:")
    print(pivot.to_string())

    # Pairwise relative differences between oracles (if both succeeded)
    summary = {}
    available = [b for b in backends if b in pivot.columns]
    for a in available:
        for b in available:
            if a >= b:
                continue
            mask = pivot[a].notna() & pivot[b].notna()
            if mask.sum() < 2:
                continue
            d = (pivot[a][mask] - pivot[b][mask])
            rel = d / pivot[b][mask]
            summary[f"{a}_vs_{b}"] = {
                "n": int(mask.sum()),
                "mean_abs_diff": float(d.abs().mean()),
                "mean_rel_diff_pct": float((rel * 100).abs().mean()),
                "max_abs_diff": float(d.abs().max()),
            }
    # Also report which oracles entirely failed (informational)
    summary["_failed_oracles"] = sorted(set(backends) - set(available))
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[3way] pairwise summary:")
    print(json.dumps(summary, indent=2))

    return 0
