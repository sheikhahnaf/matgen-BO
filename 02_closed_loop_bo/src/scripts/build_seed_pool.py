"""Combine LTMs from previous Phase-3 accel runs into a single seed pool.

Output: $SCRATCH/matinvent-hcap-bo/data/ltm_seed_pool.parquet

This pool warm-starts the GP for new accel runs (cold-start mitigation).
We use ALL prior labels (seeds 42 + 17 + 99) since rewards are
seed-independent properties of the structure.
"""
import os
import pandas as pd

R = os.path.expandvars("$SCRATCH/matinvent-hcap-bo/data")
SOURCES = [
    "ltm_phase3_2999988.parquet",                    # seed=42 accel (best run, has Li-Mg-Zn/Al hits)
    "ltm_hcap_p2_accel_seed17_3001432.parquet",      # seed=17 accel
    "ltm_hcap_p2_accel_seed99_3001434.parquet",      # seed=99 accel
]

dfs = []
for fname in SOURCES:
    p = os.path.join(R, fname)
    if not os.path.exists(p):
        print(f"  skip missing: {p}")
        continue
    df = pd.read_parquet(p)
    print(f"  + {fname}: {len(df)} rows")
    dfs.append(df)

if not dfs:
    raise SystemExit("no source LTMs found")

merged = pd.concat(dfs, ignore_index=True)

# Dedupe by structure_id (keep first occurrence)
before = len(merged)
merged = merged.drop_duplicates(subset=["structure_id"], keep="first")
print(f"  merged: {before} → {len(merged)} unique")

# Drop NaN y_cp (failed oracles)
merged = merged.dropna(subset=["y_cp"])
print(f"  with valid y_cp: {len(merged)}")
print(f"  y_cp: min={merged['y_cp'].min():.3f}  max={merged['y_cp'].max():.3f}  "
      f"mean={merged['y_cp'].mean():.3f}")

# Sort by descending y_cp for visibility
merged = merged.sort_values("y_cp", ascending=False).reset_index(drop=True)
print("  top-10:")
print(merged[["formula", "y_cp", "cycle_id"]].head(10).to_string(index=False))

out = os.path.join(R, "ltm_seed_pool.parquet")
merged.to_parquet(out, index=False)
print(f"\nwrote seed pool to {out}  ({len(merged)} rows)")
