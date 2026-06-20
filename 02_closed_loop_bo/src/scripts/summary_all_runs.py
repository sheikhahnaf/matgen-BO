"""Summarize all experimental runs for the paper.

Reads MatterGen Phase-3 (long_term_memory.csv) and Phase-2 ablation
(ltm.parquet) results, prints a comprehensive table.
"""
import os, glob, pandas as pd

R = "/scratch/user/ahnafalvi/matinvent-hcap-bo/results"
RW2CP = 2.11   # Cp ≈ 2.11 × reward (calibrated from Li12Mg7Zn ref point)


def p3_summary(d):
    ltm = os.path.join(d, "samples", "long_term_memory.csv")
    if not os.path.exists(ltm):
        return None
    df = pd.read_csv(ltm)
    if not len(df):
        return None
    mx = float(df["reward"].max())
    n_above_05 = int((df["reward"] > 0.5).sum())
    n_above_07 = int((df["reward"] > 0.7).sum())
    top_comp = df.nlargest(1, "reward")["comp"].values[0].strip().replace("\n", " ")[:30]
    return dict(n=len(df), max_reward=mx, est_cp=RW2CP * mx,
                n_gt_05=n_above_05, n_gt_07=n_above_07, top=top_comp)


def p2_summary(d):
    parq = os.path.join(d, "ltm.parquet")
    if not os.path.exists(parq):
        return None
    df = pd.read_parquet(parq)
    cp = df.dropna(subset=["y_cp"])
    if not len(cp):
        return None
    mx = float(cp["y_cp"].max())
    n_15 = int((cp["y_cp"] >= 1.5).sum())
    n_10 = int((cp["y_cp"] >= 1.0).sum())
    top = str(cp.nlargest(1, "y_cp")["formula"].values[0])[:30]
    return dict(n=len(cp), best_cp=mx, n_above_target=n_15, n_above_1=n_10, top=top)


print("\n" + "=" * 100)
print("=== Phase-3 MatterGen (RL+BO, 10 cycles each) ===")
print("=" * 100)
print(f"{'Run':<42} {'n':>4} {'maxRew':>8} {'estCp':>8} {'>0.5':>5} {'>0.7':>5}  TopHit")
print("-" * 100)

p3_runs = [
    ("hcap_p1_baseline_2999987",          "MatterGen-baseline-seed42"),
    ("hcap_p2_accel_2999988",             "MatterGen-accel-seed42"),
    ("hcap_p1_baseline_seed17_3001431",   "MatterGen-baseline-seed17"),
    ("hcap_p2_accel_seed17_3001432",      "MatterGen-accel-seed17"),
    ("hcap_p1_baseline_seed99_3001433",   "MatterGen-baseline-seed99"),
    ("hcap_p2_accel_seed99_3001434",      "MatterGen-accel-seed99"),
]
for rdir, label in p3_runs:
    cands = glob.glob(os.path.join(R, rdir + "*"))
    if not cands:
        print(f"  {label:<40} NOT FOUND")
        continue
    res = p3_summary(cands[0])
    if res is None:
        print(f"  {label:<40} (incomplete)")
        continue
    print(f"  {label:<40} {res['n']:>4} {res['max_reward']:>8.4f} {res['est_cp']:>8.3f} "
          f"{res['n_gt_05']:>5} {res['n_gt_07']:>5}  {res['top']}")

print()
print("=" * 100)
print("=== Phase-2 ablation (frozen generator, 10 cycles, raw Cp from y_cp) ===")
print("=" * 100)
print(f"{'Run':<42} {'n':>4} {'bestCp':>8} {'>=1.5':>6} {'>=1.0':>6}  TopHit")
print("-" * 100)

# Phase-2: K=8 PI top-K accel; B=K=8 anchor=1 baseline
p2_runs = [
    ("p2_adit_3001430",                    "ADiT-Phase2-accel"),
    ("p2_adit_baseline_3001783",           "ADiT-Phase2-baseline"),
    ("p2_crystalflow_3001428",             "CrystalFlow-Phase2-accel"),
    ("p2_crystalflow_baseline_3001784",    "CrystalFlow-Phase2-baseline"),
    ("p2_crysbfn_3001429",                 "CrysBFN-Phase2-accel"),
]
for rdir, label in p2_runs:
    d = os.path.join(R, rdir)
    if not os.path.isdir(d):
        print(f"  {label:<40} NOT FOUND")
        continue
    res = p2_summary(d)
    if res is None:
        print(f"  {label:<40} (no Cp data)")
        continue
    print(f"  {label:<40} {res['n']:>4} {res['best_cp']:>8.3f} "
          f"{res['n_above_target']:>6} {res['n_above_1']:>6}  {res['top']}")

print()
