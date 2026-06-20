"""Cross-run best-Cp comparison.

Reads each run's data files and produces a comparison table:
  - Phase-3 runs (MatterGen baseline+accel × 3 seeds): use samples/long_term_memory.csv
  - Phase-2 runs (ADiT, CrystalFlow, CrysBFN): use ltm.parquet

Run on FASTER:
  /scratch/user/ahnafalvi/envs/matinvent-hcap-bo/bin/python scripts/compare_runs.py
"""

import glob
import os

import pandas as pd

RESULTS_DIR = "/scratch/user/ahnafalvi/matinvent-hcap-bo/results"

# Reward scale derived from seed=42 accel: Li12Mg7Zn at reward 0.696 → Cp 1.468 J/g/K → factor 2.11.
REWARD_TO_CP = 2.11


def analyze_phase3(run_dir, label):
    cands = glob.glob(os.path.join(RESULTS_DIR, run_dir + "*", "samples", "long_term_memory.csv"))
    if not cands:
        return (label, 0, None, None, None, "NOT FOUND")
    df = pd.read_csv(cands[0])
    n = len(df)
    if n == 0:
        return (label, 0, None, None, 0, "empty")
    mx = float(df["reward"].max())
    n_above = int((df["reward"] > 0.5).sum())
    cp_est = REWARD_TO_CP * mx
    top = df.nlargest(1, "reward")["comp"].values[0].strip().replace("\n", " ")[:32]
    return (label, n, mx, cp_est, n_above, top)


def analyze_phase2(run_dir, label):
    path = os.path.join(RESULTS_DIR, run_dir, "ltm.parquet")
    if not os.path.exists(path):
        return (label, 0, None, None, None, "NOT FOUND")
    df = pd.read_parquet(path)
    n = len(df)
    if n == 0:
        return (label, 0, None, None, 0, "empty")
    cp_cols = [c for c in df.columns if c.lower() in ("cp", "y_cp") or "heat" in c.lower() or "capacity" in c.lower()]
    if not cp_cols:
        return (label, n, None, None, None, f"cols={list(df.columns)[:5]}")
    cp_col = cp_cols[0]
    df_ok = df.dropna(subset=[cp_col])
    if len(df_ok) == 0:
        return (label, n, None, None, 0, "all NaN")
    mx_cp = float(df_ok[cp_col].max())
    rew = min(mx_cp / 1.5, 1.0)
    n_above = int((df_ok[cp_col] / 1.5 > 0.5).sum())
    fcol = next((c for c in ["formula", "comp", "composition", "sid"] if c in df.columns), df.columns[0])
    top = str(df_ok.nlargest(1, cp_col)[fcol].values[0])[:32]
    return (label, n, rew, mx_cp, n_above, top)


def main():
    rows = []
    rows.append(("=== Phase-3 (RL+BO, 10 cycles, eval_size=16, K=8) ===", 0, None, None, None, ""))
    for d, lab in [
        ("hcap_p1_baseline_2999987", "MatterGen-baseline-seed42"),
        ("hcap_p2_accel_2999988", "MatterGen-accel-seed42"),
        ("hcap_p1_baseline_seed17_3001431", "MatterGen-baseline-seed17"),
        ("hcap_p2_accel_seed17_3001432", "MatterGen-accel-seed17"),
        ("hcap_p1_baseline_seed99_3001433", "MatterGen-baseline-seed99"),
        ("hcap_p2_accel_seed99_3001434", "MatterGen-accel-seed99"),
    ]:
        rows.append(analyze_phase3(d, lab))

    rows.append(("=== Phase-2 (frozen-generator BO, 10 cycles, K=8 PI) ===", 0, None, None, None, ""))
    for d, lab in [
        ("p2_adit_3001430", "ADiT-Phase2"),
        ("p2_crystalflow_3001428", "CrystalFlow-Phase2"),
        ("p2_crysbfn_3001429", "CrysBFN-Phase2(crash)"),
    ]:
        rows.append(analyze_phase2(d, lab))

    print()
    print("{:<32} {:>5} {:>7} {:>9} {:>5}  {}".format("Run", "n", "MaxRew", "BestCp", ">0.5", "TopHit"))
    print("-" * 100)
    for label, n, mxrew, cp, na, top in rows:
        if mxrew is None and not str(label).startswith("==="):
            print("{:<32} {:>5} {:>7} {:>9} {:>5}  {}".format(label, n, "?", "?", "?", str(top)))
        elif str(label).startswith("==="):
            print()
            print(label)
        else:
            print("{:<32} {:>5} {:>7.3f} {:>9.3f} {:>5}  {}".format(label, n, mxrew, cp, na, str(top)))


if __name__ == "__main__":
    main()
