"""Build v3 seed pool — union of EVERY ltm_*.parquet in data/, dedup by id.

Goal: give v3 GP cold-start min ≥ 50 with the broadest possible label set
across all prior phase-2/phase-3 runs (seed-independent property).

Output: $SCRATCH/matinvent-hcap-bo/data/ltm_seed_pool_v3.parquet
"""
import glob
import os
import pandas as pd

R = os.path.expandvars("$SCRATCH/matinvent-hcap-bo/data")
out = os.path.join(R, "ltm_seed_pool_v3.parquet")

paths = sorted(glob.glob(os.path.join(R, "ltm_*.parquet")))
# Exclude any prior pool files
paths = [p for p in paths if "seed_pool" not in os.path.basename(p)]
print(f"Found {len(paths)} LTM files")

dfs = []
for p in paths:
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"  skip {os.path.basename(p)}: {e}")
        continue
    if "y_cp" not in df.columns:
        print(f"  skip {os.path.basename(p)}: no y_cp column")
        continue
    df = df.dropna(subset=["y_cp"])
    if len(df):
        print(f"  + {os.path.basename(p)}: {len(df)} rows")
        dfs.append(df)

if not dfs:
    raise SystemExit("no usable LTM files found")

merged = pd.concat(dfs, ignore_index=True)
before = len(merged)
merged = merged.drop_duplicates(subset=["structure_id"], keep="first")
print(f"\nmerged: {before} → {len(merged)} unique rows")
print(f"y_cp:  min={merged['y_cp'].min():.3f}  max={merged['y_cp'].max():.3f}  "
      f"mean={merged['y_cp'].mean():.3f}  median={merged['y_cp'].median():.3f}")

# Sort high → low for visibility, keep all
merged = merged.sort_values("y_cp", ascending=False).reset_index(drop=True)
print("\ntop-15:")
print(merged[["formula", "y_cp", "cycle_id"]].head(15).to_string(index=False))

merged.to_parquet(out, index=False)
print(f"\nwrote {out}")
print(f"final rows: {len(merged)}  (cold_start_min target: 50 — "
      f"{'OK' if len(merged) >= 50 else 'WARN: still below 50'})")
