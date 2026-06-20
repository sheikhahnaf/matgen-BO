"""Merge the incremental Zenodo per-tarball row-pickles into the canonical pheasy
benchmark DataFrame.

The pheasy arm was assembled in two passes from Zenodo record 20196565:
  * pass 1 -- 70 tarballs extracted at once -> data/pheasy_phonon_thermo.pkl (6,629 rows)
  * pass 2 -- the 17 tarballs that overflowed the $SCRATCH inode quota were fetched
    incrementally to node-local /tmp and parsed one-by-one into
    data/zenodo/rows/<tarball>.pkl (one small pickle each).

The two passes cover disjoint mp-id ranges, so the union is a strict superset of the
pass-1 set. We still dedup on material_id (keep first) as a guard, re-validate that all
four targets are finite, and archive the pass-1 pickle to data/legacy/ before writing the
merged canonical file -- no data is ever lost (non-destructive: old pickle preserved).
"""

import glob
import os
import shutil

import numpy as np
import pandas as pd

ROOT = "/scratch/user/u.sa119259/phonon_thermo_benchmark"
CANON = os.path.join(ROOT, "data", "pheasy_phonon_thermo.pkl")
ROWS = os.path.join(ROOT, "data", "zenodo", "rows")
LEGACY = os.path.join(ROOT, "data", "legacy")
TARGETS = ("Cv_300K", "S_300K", "F_300K", "max_phonon_freq")


def main():
    base = pd.read_pickle(CANON)
    print(f"pass-1 canonical: {len(base)} rows")

    parts = [base]
    for p in sorted(glob.glob(os.path.join(ROWS, "*.pkl"))):
        df = pd.read_pickle(p)
        parts.append(df)
        print(f"  + {os.path.basename(p)}: {len(df)} rows")

    merged = pd.concat(parts, ignore_index=True)
    n_concat = len(merged)
    merged = merged.drop_duplicates(subset="material_id", keep="first").reset_index(drop=True)
    n_dedup = len(merged)

    # Re-validate finiteness (the per-tarball parser already guards, but defend against
    # any stray non-finite target slipping through a schema change).
    finite_mask = np.isfinite(merged[list(TARGETS)].to_numpy()).all(axis=1)
    merged = merged[finite_mask].reset_index(drop=True)
    n_final = len(merged)

    print(f"concat: {n_concat} | after dedup: {n_dedup} | after finite-guard: {n_final}")
    for c in TARGETS:
        lo, hi = float(merged[c].min()), float(merged[c].max())
        print(f"  {c}: [{lo:.4g}, {hi:.4g}]")

    # Archive pass-1 pickle (non-destructive) before writing the merged superset.
    os.makedirs(LEGACY, exist_ok=True)
    archive = os.path.join(LEGACY, "pheasy_phonon_thermo_pass1_6629.pkl")
    if not os.path.exists(archive):
        shutil.copy2(CANON, archive)
        print(f"archived pass-1 -> {archive}")
    else:
        print(f"archive already present -> {archive} (left as-is)")

    merged.to_pickle(CANON)
    print(f"wrote merged superset -> {CANON} ({n_final} rows)")


if __name__ == "__main__":
    main()
