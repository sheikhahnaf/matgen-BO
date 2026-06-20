"""Figures for the A-PU vs CGNF synthesizability comparison on generated structures.

Reads results/apu_grace/generated_scores/scores.csv and writes two PNGs:
  fig_apu_vs_cgnf_scatter.png   - per-structure A-PU vs CGNF, by backbone, OOD-flagged
  fig_per_backbone_panel.png    - mean scores + OOD/abstain rate per (backbone, policy)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

from pathlib import Path
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())
_SUB = _ROOT / "03_synthesizability"
_OUT = _SUB / "figures" / "regenerated"
_OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(_SUB / "results" / "generated_scores" / "scores.csv")

COLORS = {"ADiT": "#1b7837", "CrystalFlow": "#2166ac", "MatterGen": "#b2182b"}
order = ["ADiT", "CrystalFlow", "MatterGen"]

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(5.6, 5.4))
for bb in order:
    sub = df[df.backbone_name == bb]
    kept = sub[~sub.abstain_ood]
    ood = sub[sub.abstain_ood]
    ax.scatter(kept.cgnf_score, kept.apu_score, s=46, c=COLORS[bb],
               edgecolors="white", linewidths=0.5, label=bb, zorder=3)
    ax.scatter(ood.cgnf_score, ood.apu_score, s=46, facecolors="none",
               edgecolors=COLORS[bb], linewidths=1.4, zorder=3)
ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1.0, zorder=1)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("CGNF synthesizability score")
ax.set_ylabel("A-PU synthesizability probability")

rho, _ = spearmanr(df.apu_score, df.cgnf_score)
agree = float(np.mean((df.apu_score > 0.5) == (df.cgnf_score > 0.5)))
ax.set_title(f"A-PU vs CGNF on 40 generated structures\nSpearman $\\rho$={rho:.2f}, "
             f"agreement at 0.5 = {agree*100:.0f}%")

bb_handles = [Line2D([], [], marker="o", ls="", mfc=COLORS[b], mec="white", ms=8, label=b) for b in order]
flag_handles = [
    Line2D([], [], marker="o", ls="", mfc="0.4", mec="white", ms=8, label="in-distribution"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", mew=1.4, ms=8, label="OOD (abstained)"),
]
leg1 = ax.legend(handles=bb_handles, title="backbone", loc="upper left", frameon=False)
ax.add_artist(leg1)
ax.legend(handles=flag_handles, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(_OUT / "fig_apu_vs_cgnf_scatter.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2
g = (df.groupby(["backbone_name", "policy"])
       .agg(n=("apu_score", "size"), apu=("apu_score", "mean"),
            cgnf=("cgnf_score", "mean"), ood=("abstain_ood", "mean"))
       .reset_index())
g["bborder"] = g.backbone_name.map({b: i for i, b in enumerate(order)})
g = g.sort_values(["bborder", "policy"]).reset_index(drop=True)
labels = [f"{r.backbone_name}\n{r.policy} (n={r.n})" for r in g.itertuples()]
x = np.arange(len(g)); w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax0 = axes[0]
ax0.bar(x - w/2, g.apu, w, label="A-PU", color="#4575b4", edgecolor="white")
ax0.bar(x + w/2, g.cgnf, w, label="CGNF", color="#d6604d", edgecolor="white")
ax0.set_ylim(0, 1.0); ax0.set_ylabel("mean synthesizability score")
ax0.set_xticks(x); ax0.set_xticklabels(labels)
ax0.set_title("Mean score per backbone and policy")
ax0.legend(frameon=False, loc="upper right")
ax0.grid(axis="y", ls=":", c="0.85", zorder=0)

ax1 = axes[1]
ax1.bar(x, g.ood * 100, 0.6, color="#7f7f7f", edgecolor="white")
ax1.set_ylim(0, 105); ax1.set_ylabel("OOD-abstained fraction (%)")
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_title("Out-of-distribution rate (A-PU OOD layer)")
ax1.grid(axis="y", ls=":", c="0.85", zorder=0)
for xi, v in zip(x, g.ood * 100):
    ax1.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)

fig.tight_layout()
fig.savefig(_OUT / "fig_per_backbone_panel.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_apu_vs_cgnf_scatter.png and fig_per_backbone_panel.png")
print(f"rho={rho:.4f} agree={agree:.3f}")
