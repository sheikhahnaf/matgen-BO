"""Phase 0+: smoke test for HCapOracle (FairChem eSEN-30M-OAM heat-capacity).

Builds 3 tiny ASE Atoms (Si diamond, MgO rocksalt, Cu fcc) and runs the
production oracle subprocess wrapper. Verifies:
    - Env prefix + runner script wiring
    - Multi-image extxyz IO
    - Cp values land in J/g/K of the right magnitude (~0.4-1.0 typical)
    - Runtime per structure (informs Phase 1 budget)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from ase.build import bulk

from src.oracle import HCapOracle_eSEN as HCapOracle


def run(config_path: str, output_dir: str) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_prefix = os.environ.get(
        "ENV_PREFIX",
        f"{os.environ['SCRATCH']}/envs/matinvent-hcap-bo",
    )

    # Three small canonical crystals
    atoms_list = [
        bulk("Si", "diamond", a=5.43),
        bulk("MgO", "rocksalt", a=4.21),
        bulk("Cu", "fcc", a=3.61, cubic=True),
    ]
    labels = ["Si-diamond", "MgO-rocksalt", "Cu-fcc"]
    print(f"[oracle-smoke] {len(atoms_list)} structures")

    oracle = HCapOracle(
        env_prefix=env_prefix,
        n_workers=1,
        scratch_dir=str(out),
    )
    print(f"[oracle-smoke] env_prefix={env_prefix}")
    print(f"[oracle-smoke] runner={oracle.runner_script}")

    t0 = time.time()
    cp, mask = oracle.evaluate(atoms_list)
    dt = time.time() - t0
    print(f"[oracle-smoke] elapsed {dt:.1f}s ({dt / max(1, len(atoms_list)):.1f}s/structure)")

    for lbl, c, fail in zip(labels, cp, mask):
        flag = "FAIL" if fail else f"{c:.4f} J/g/K"
        print(f"  {lbl:18s}  {flag}")

    # Save raw output
    np.savetxt(out / "cp.txt", cp)
    with open(out / "summary.txt", "w") as f:
        f.write(f"elapsed_seconds: {dt:.2f}\n")
        for lbl, c, fail in zip(labels, cp, mask):
            f.write(f"{lbl}\t{c}\t{int(fail)}\n")

    n_ok = int((~mask).sum())
    print(f"[oracle-smoke] success rate: {n_ok}/{len(atoms_list)}")
    if n_ok == 0:
        print("[oracle-smoke] FAIL: no successful evaluations")
        return 1
    return 0
