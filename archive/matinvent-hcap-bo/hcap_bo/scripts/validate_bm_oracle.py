"""Validate LocalESEN_BM against known experimental bulk moduli.

Run:
    PYTHONPATH=$SCRATCH/matinvent-hcap-bo:$SCRATCH/matinvent-pristine \
        python scripts/validate_bm_oracle.py

Materials and reference K_VRH (GPa) — experimental / DFT consensus values:
    diamond  443  (sp3 C, F m-3m → diamond cubic Fd-3m, 2 atoms/cell)
    Si       97.6 (Fd-3m)
    Cu       137  (Fm-3m)
    Al       76   (Fm-3m)
    MgO      160  (Fm-3m, rocksalt)
    NaCl     24   (Fm-3m)
    GaN      210  (P63mc, wurtzite)

Pass criterion: |predicted - reference| / reference < 0.25 for ALL materials.
"""
import os
import sys
import numpy as np
from ase import Atoms
from ase.build import bulk

REFS = {
    "diamond": {"K": 443.0, "make": lambda: bulk("C", "diamond", a=3.567)},
    "Si":      {"K":  97.6, "make": lambda: bulk("Si", "diamond", a=5.431)},
    "Cu":      {"K": 137.0, "make": lambda: bulk("Cu", "fcc", a=3.615)},
    "Al":      {"K":  76.0, "make": lambda: bulk("Al", "fcc", a=4.050)},
    "MgO":     {"K": 160.0, "make": lambda: bulk("MgO", "rocksalt", a=4.212)},
    "NaCl":    {"K":  24.0, "make": lambda: bulk("NaCl", "rocksalt", a=5.640)},
    "GaN":     {"K": 210.0, "make": lambda: bulk("GaN", "wurtzite", a=3.189, c=5.185)},
}

TOLERANCE = 0.25  # 25%


def main():
    sys.path.insert(0, os.path.expandvars("$SCRATCH/matinvent-hcap-bo"))
    from src.calculators.local_esen_bm import LocalESEN_BM

    out_dir = os.path.expandvars("$SCRATCH/matinvent-hcap-bo/results_bm/_validation")
    os.makedirs(out_dir, exist_ok=True)
    bm = LocalESEN_BM(root_dir=out_dir, task="bulk_modulus", env_name=None, worker=1)

    print(f"{'material':>10}  {'predicted':>10}  {'reference':>10}  {'rel_err':>10}  status")
    print("-" * 60)
    fails = []
    rows = []
    for mat, info in REFS.items():
        atoms = info["make"]()
        try:
            K_pred = bm._bm_task(atoms)
        except Exception as e:
            print(f"{mat:>10}  {'CRASH':>10}  {info['K']:>10.1f}  {'—':>10}  FAIL ({type(e).__name__}: {e})")
            fails.append(mat)
            rows.append({"material": mat, "predicted": np.nan, "reference": info["K"], "rel_err": np.nan})
            continue
        if not np.isfinite(K_pred):
            print(f"{mat:>10}  {'NaN':>10}  {info['K']:>10.1f}  {'—':>10}  FAIL")
            fails.append(mat)
            rows.append({"material": mat, "predicted": np.nan, "reference": info["K"], "rel_err": np.nan})
            continue
        rel = abs(K_pred - info["K"]) / info["K"]
        ok = rel <= TOLERANCE
        status = "OK" if ok else f"FAIL (>{TOLERANCE:.0%})"
        print(f"{mat:>10}  {K_pred:>10.2f}  {info['K']:>10.1f}  {rel:>10.1%}  {status}")
        if not ok:
            fails.append(mat)
        rows.append({"material": mat, "predicted": K_pred, "reference": info["K"], "rel_err": rel})

    # Save table
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "bm_oracle_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    if fails:
        print(f"\n❌ FAILED {len(fails)}/{len(REFS)}: {fails}", file=sys.stderr)
        sys.exit(1)
    print(f"\n✅ All {len(REFS)} materials within ±{TOLERANCE:.0%}.")


if __name__ == "__main__":
    main()
