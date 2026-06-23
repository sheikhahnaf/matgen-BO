#!/usr/bin/env python
"""eSEN-oracle vs ground-truth DFT for K0 (the oracle-integrity leg of the three-way).

DFT K0 = our oracle-parity Birch-Murnaghan campaign (dft_validation eos_<stem>/K0.json, B0_GPa).
eSEN K0 = the value the bm closed loop optimized (bm leaderboard 'value', by global rank).
Only the bm-leaderboard structures have an eSEN K0 (the cp loop optimized Cv, not K0); the cp
eSEN-vs-DFT(Cv) leg waits on the phonon Cv campaign. Writes a per-structure table + a scatter.
"""
import glob, os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DV = "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation"
ESEN = "/Volumes/SSD1_SMAAA/matinvent-hcap-bo/hcap_bo/analysis/top_structures/bm/global_top20.csv"
RES = "/Volumes/SSD1_SMAAA/matinvent-bo/three_way_comparison/results"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.titlesize": 11.5,
                     "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "figure.dpi": 150, "axes.linewidth": 0.8})

esen = pd.read_csv(ESEN)
rows = []
for cif in sorted(glob.glob(DV + "/structures/bm_*.cif")):
    stem = os.path.basename(cif)[:-4]
    m = re.search(r"bm_top(\d+)_(adit|cf|mg)_(ACC|BASE)_seed\d+_", stem)
    if not m:
        continue
    rank, setup = int(m.group(1)), m.group(3)
    j = os.path.join(DV, "results/faster", "eos_" + stem, "K0.json")
    if not os.path.exists(j) or rank > len(esen):
        continue
    d = json.load(open(j))
    rows.append(dict(formula=esen.iloc[rank - 1]["reduced_formula"], rank=rank, setup=setup,
                     dft_K0=round(float(d["B0_GPa"]), 1), esen_K0=round(float(esen.iloc[rank - 1]["value"]), 1),
                     V0=round(float(d["V0"]), 1), B0prime=round(float(d["B0prime"]), 2),
                     inwindow=bool(d["inwindow"])))
df = pd.DataFrame(rows).sort_values("esen_K0", ascending=False).reset_index(drop=True)
df.to_csv(RES + "/esen_vs_dft_bm.csv", index=False)

a = df["esen_K0"].values.astype(float); b = df["dft_K0"].values.astype(float); e = a - b
print("=== eSEN K0 vs DFT K0 — oracle integrity (n=%d bm structures) ===" % len(df))
print(df.to_string(index=False))
print("\nMAE=%.1f  RMSE=%.1f  MAPE=%.1f%%  bias(eSEN-DFT)=%+.1f  Pearson r=%.3f  Spearman rho=%.3f"
      % (np.abs(e).mean(), (e**2).mean()**.5, (100*np.abs(e)/b).mean(), e.mean(),
         pearsonr(a, b)[0], spearmanr(a, b).correlation))

fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.9))
# all bm winners are hard refractory -> zoom to the data band; identities live in the table
lo = min(a.min(), b.min()) - 12; hi = max(a.max(), b.max()) + 12
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, label="parity", zorder=1)
ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color="gray", alpha=0.10,
                zorder=0, label="±10 GPa")
for setup, c, mk in [("ACC", "#b5482a", "o"), ("BASE", "#3a6ea5", "s")]:
    s = df[df["setup"] == setup]
    ax.scatter(s["dft_K0"], s["esen_K0"], c=c, marker=mk, s=70, edgecolors="black",
               linewidths=0.6, label="%s run" % setup, zorder=3, alpha=0.9)
ax.set_xlabel("DFT K$_0$ (oracle-parity Birch–Murnaghan, GPa)"); ax.set_ylabel("eSEN oracle K$_0$ (GPa)")
ax.set_title("Oracle integrity: eSEN vs ground-truth DFT\nMAE=%.1f GPa, MAPE=%.1f%%, r=%.2f, ρ=%.2f (n=%d, all hard refractory)"
             % (np.abs(e).mean(), (100*np.abs(e)/b).mean(), pearsonr(a, b)[0], spearmanr(a, b).correlation, len(df)))
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.legend(frameon=False, loc="upper left", fontsize=8); ax.grid(True, alpha=0.2, lw=0.5)
fig.tight_layout(); fig.savefig(RES + "/esen_vs_dft_bm.png", bbox_inches="tight"); plt.close(fig)
print("\nsaved -> %s/esen_vs_dft_bm.{csv,png}" % RES)
