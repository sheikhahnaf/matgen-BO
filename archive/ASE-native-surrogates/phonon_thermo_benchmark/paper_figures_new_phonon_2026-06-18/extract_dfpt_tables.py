"""Derive the exact DFPT phonon-thermo table values for the paper swap (n_train=500), so no
number is hand-typed. Mirrors the original table logic:
  * best PCA per (surrogate, descriptor) = the PCA maximizing avg R2 across the 4 properties
  * Table 5 (best_configs_phonon): rows per surrogate x descriptor with best PCA + avg R2/RMSE/rho
  * cross-dataset phonon row: single best (surr,desc,pca) by avg R2 -> avg R2, avg rho, easiest/hardest prop
  * Table S3 (property_difficulty): per property, best (desc,surr,pca) by R2 with R2 + rho
"""
import pandas as pd
from pathlib import Path

CSV = Path(__file__).parent / "arm_a_dfpt" / "aggregated_results.csv"
df = pd.read_csv(CSV)
df = df[df["n_train"] == 500].copy()
PROPS = ["Cv_300K", "S_300K", "F_300K", "max_phonon_freq"]
MODELS = ["gp", "mtgp_2", "dgp"]
DESCS = ["mace", "orb", "soap", "uma"]
MLAB = {"gp": "GP", "mtgp_2": "MTGP", "dgp": "DGP"}


def val(model, desc, pca, prop, metric):
    r = df[(df.model == model) & (df.descriptor == desc) & (df.pca_components == pca)
           & (df["property"] == prop) & (df.metric == metric)]
    return float(r["mean"].iloc[0]) if len(r) else float("nan")


def avg_over_props(model, desc, pca, metric):
    vals = [val(model, desc, pca, p, metric) for p in PROPS]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def best_pca(model, desc):
    cand = [(pca, avg_over_props(model, desc, pca, "R2")) for pca in (10, 25, 50)]
    cand = [(p, r) for p, r in cand if r == r]
    return max(cand, key=lambda t: t[1])[0] if cand else None


print("=== TABLE 5 (best_configs_phonon, n=500): model desc bestPCA avgR2 avgRMSE avgRho ===")
best_overall = None
for m in MODELS:
    for d in DESCS:
        bp = best_pca(m, d)
        if bp is None:
            continue
        r2 = avg_over_props(m, d, bp, "R2")
        rmse = avg_over_props(m, d, bp, "RMSE")
        rho = avg_over_props(m, d, bp, "Spearman")
        print(f"{MLAB[m]:5s} {d:5s} PCA{bp:<2d}  R2={r2:.3f}  RMSE={rmse:.3f}  rho={rho:.3f}")
        if best_overall is None or r2 > best_overall[3]:
            best_overall = (m, d, bp, r2, rho)

print("\n=== CROSS-DATASET phonon row (best config by avg R2) ===")
m, d, bp, r2, rho = best_overall
# easiest / hardest property at that config
prop_r2 = sorted(((p, val(m, d, bp, p, "R2")) for p in PROPS), key=lambda t: t[1])
easiest = prop_r2[-1]
hardest = prop_r2[0]
print(f"config: {MLAB[m]}+{d.upper()}+PCA{bp}  avgR2={r2:.3f}  avgRho={rho:.3f}")
print(f"easiest: {easiest[0]} (R2={easiest[1]:.3f})   hardest: {hardest[0]} (R2={hardest[1]:.3f})")

print("\n=== TABLE S3 (property_difficulty): per property best (desc,surr,pca) by R2 ===")
for p in PROPS:
    best = None
    for m in MODELS:
        for d in DESCS:
            for pca in (10, 25, 50):
                r2 = val(m, d, pca, p, "R2")
                if r2 == r2 and (best is None or r2 > best[0]):
                    best = (r2, val(m, d, pca, p, "Spearman"), d, m, pca)
    if best:
        r2, rho, d, m, pca = best
        print(f"{p:16s}  {d:5s} {MLAB[m]:5s} PCA{pca:<2d}  R2={r2:.3f}  rho={rho:.3f}")

print("\n=== S12c parity: best-predicted property overall (for the parity panel) ===")
flat = [(val(m, d, pca, p, "R2"), p, d, m, pca) for p in PROPS for m in MODELS for d in DESCS for pca in (10, 25, 50)]
flat = [t for t in flat if t[0] == t[0]]
# paper convention: ORB+GP best-predicted property
orbgp = sorted([t for t in flat if t[2] == "orb" and t[3] == "gp"], key=lambda t: t[0])
if orbgp:
    r2, p, d, m, pca = orbgp[-1]
    print(f"ORB+GP best property: {p} (R2={r2:.3f}, PCA{pca})")
