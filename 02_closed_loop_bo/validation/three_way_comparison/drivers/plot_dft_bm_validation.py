#!/usr/bin/env python
"""Paper figure: DFT validation of the bulk-modulus discoveries (2 panels).
(a) eSEN oracle vs ground-truth DFT K0 parity (oracle fidelity, n=15).
(b) GP causal step-ahead prediction vs eSEN over all generated structures, colored by
    loop-accumulated training step, with the DFT-validated winners circled (surrogate
    generalization / ranking, n=1209). Panel (b) reproduces the design of
    results/three_way_property_bm_seeded.png. Outputs to the FME paper figures/ dir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "savefig.dpi": 900,
    "figure.dpi": 900,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.expanduser("~/fme_paper_work/FoundationalEmbeddings_2026/figures/fig_dft_bm_validation.png")

e = pd.read_csv(os.path.join(RES, "esen_vs_dft_bm.csv"))
g = pd.read_csv(os.path.join(RES, "gp_causal_allpreds_bm_seeded.csv"))
w = pd.read_csv(os.path.join(RES, "three_way_causal_bm_seeded.csv"))

BLUE = "#2c6fbb"
GRAY = "#777777"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)

# --- (a) eSEN vs DFT parity (oracle fidelity) ---
d, s = e.dft_K0.values, e.esen_K0.values
lo, hi = 285, 365
ax1.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10],
                 color=GRAY, alpha=0.12, lw=0)
ax1.plot([lo, hi], [lo, hi], "--", color=GRAY, lw=1.0, zorder=1)
ax1.scatter(d, s, s=40, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi); ax1.set_aspect("equal", "box")
ax1.set_xlabel(r"DFT $K_0$ (GPa)")
ax1.set_ylabel(r"eSEN oracle $K_0$ (GPa)")
ax1.set_title("(a) Oracle fidelity", loc="left", fontweight="bold")
ax1.text(0.04, 0.96, "MAE 8.5 GPa\nMAPE 2.5%\n" + r"$\rho$ = 0.87  ($n$=15)",
         transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY, lw=0.6, alpha=0.9))

# --- (b) GP causal vs eSEN, colored by training step, winners circled ---
gp, es, nt = g.gp_pred.values, g.esen.values, g.n_train_accum.values
rho = spearmanr(gp, es)[0]; r = pearsonr(gp, es)[0]; rmse = np.sqrt(np.mean((gp - es) ** 2))
top = max(es.max(), gp.max()) * 1.02
ax2.plot([0, top], [0, top], "--", color="black", lw=1.0, alpha=0.7, zorder=1, label="parity")
sc = ax2.scatter(es, gp, c=nt, cmap="viridis", s=10, alpha=0.45, linewidths=0, zorder=2)
ax2.scatter(w.esen_K0, w.gp_causal_K0, facecolors="none", edgecolors="red",
            s=70, linewidths=1.4, zorder=4, label="DFT-validated winners (ACC)")
cb = fig.colorbar(sc, ax=ax2, pad=0.02)
cb.set_label("loop-accumulated $n_\\mathrm{train}$ (seed excl.)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax2.set_xlim(0, top); ax2.set_ylim(0, top); ax2.set_aspect("equal", "box")
ax2.set_xlabel(r"eSEN oracle $K_0$ (GPa)")
ax2.set_ylabel(r"GP causal prediction $K_0$ (GPa)")
ax2.set_title("(b) Surrogate generalization", loc="left", fontweight="bold")
ax2.text(0.04, 0.96, r"$\rho$ = 0.944,  $r$ = 0.95" + "\nRMSE 19.5 GPa\n" + r"($n$=1209)",
         transform=ax2.transAxes, va="top", ha="left", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRAY, lw=0.6, alpha=0.9))
ax2.legend(frameon=False, loc="lower right", fontsize=8)

for ax in (ax1, ax2):
    ax.tick_params(direction="out", length=3)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("saved ->", OUT)
print("verify: MAE=%.1f MAPE=%.2f rho_b=%.3f r_b=%.3f rmse_b=%.1f"
      % (np.mean(np.abs(s - d)), np.mean(np.abs(s - d) / d) * 100, rho, r, rmse))
