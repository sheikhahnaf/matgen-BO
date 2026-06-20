"""Synthesizability comparison figures for the FME paper (closed-loop block).

Generates, in the paper's house style (plot_style.apply_style):
  fig_bogen_synth_scatter.png       - per-structure A-PU vs CGNF, by backbone, OOD-flagged
  fig_bogen_synth_per_backbone.png  - mean scores + OOD/abstain rate per (backbone, policy)

Reads the per-structure scores produced by
matinvent-hcap-bo/src/apu_synthesizability/score_generated.py and writes PNGs into
the FME paper figures/ directory.
"""
import sys
from pathlib import Path
import os as _os
from pathlib import Path as _Path
# Repo-relative defaults; override with MBO_REPO_ROOT / MBO_FIG_DIR / MBO_RESULTS_ROOT.
_REPO = _Path(_os.environ.get("MBO_REPO_ROOT", _Path(__file__).resolve().parent.parent))
_FIGS = _Path(_os.environ.get("MBO_FIG_DIR", _REPO / "figures"))
_RES = _Path(_os.environ.get("MBO_RESULTS_ROOT", _REPO / "hcap_bo" / "results-paper-v4"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_style  # noqa: E402

apply_style()

SCORES = str(_REPO / "hcap_bo" / "results" / "apu_grace" / "generated_scores" / "scores.csv")
OUT = _FIGS

df = pd.read_csv(SCORES)
order = ["ADiT", "CrystalFlow", "MatterGen"]
# Backbone palette matched to fig:closed_loop_zoo (blue=MatterGen, green=CrystalFlow, red=ADiT)
COLORS = {"MatterGen": "#1f77b4", "CrystalFlow": "#2ca02c", "ADiT": "#d62728"}

# ---------------------------------------------------------------- scatter
fig, ax = plt.subplots(figsize=(5.4, 5.0))
for bb in order:
    sub = df[df.backbone_name == bb]
    kept = sub[~sub.abstain_ood]
    ood = sub[sub.abstain_ood]
    ax.scatter(kept.cgnf_score, kept.apu_score, s=46, c=COLORS[bb],
               edgecolors="white", linewidths=0.5, label=bb, zorder=3)
    ax.scatter(ood.cgnf_score, ood.apu_score, s=46, facecolors="none",
               edgecolors=COLORS[bb], linewidths=1.4, zorder=3)
ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1.0, zorder=1)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("CGNF synthesizability score")
ax.set_ylabel("ORB-PU synthesizability probability")
rho, _ = spearmanr(df.apu_score, df.cgnf_score)
agree = float(np.mean((df.apu_score > 0.5) == (df.cgnf_score > 0.5)))
ax.set_title(rf"Spearman $\rho$={rho:.2f}, agreement at 0.5 = {agree*100:.0f}%")
bb_handles = [Line2D([], [], marker="o", ls="", mfc=COLORS[b], mec="white", ms=8, label=b) for b in order]
flag_handles = [
    Line2D([], [], marker="o", ls="", mfc="0.4", mec="white", ms=8, label="in-distribution"),
    Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", mew=1.4, ms=8, label="OOD (abstained)"),
]
leg1 = ax.legend(handles=bb_handles, title="backbone", loc="upper left")
ax.add_artist(leg1)
ax.legend(handles=flag_handles, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig_bogen_synth_scatter.png", bbox_inches="tight", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- per-backbone panel
g = (df.groupby(["backbone_name", "policy"])
       .agg(n=("apu_score", "size"), apu=("apu_score", "mean"),
            cgnf=("cgnf_score", "mean"), ood=("abstain_ood", "mean"))
       .reset_index())
g["o"] = g.backbone_name.map({b: i for i, b in enumerate(order)})
g = g.sort_values(["o", "policy"]).reset_index(drop=True)
labels = [f"{r.backbone_name} {r.policy}" for r in g.itertuples()]
x = np.arange(len(g)); w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
a0 = axes[0]
a0.bar(x - w/2, g.apu, w, label="ORB-PU", color="#6a51a3", edgecolor="white")
a0.bar(x + w/2, g.cgnf, w, label="CGNF", color="#e6550d", edgecolor="white")
a0.set_ylim(0, 1.0)
a0.set_ylabel("mean synthesizability score")
a0.set_xticks(x); a0.set_xticklabels(labels, rotation=25, ha="right")
a0.set_title("Mean score per backbone and policy")
a0.legend(loc="upper right")
a0.grid(axis="y", ls=":", c="0.85", zorder=0)

a1 = axes[1]
a1.bar(x, g.ood * 100, 0.6, color="#7f7f7f", edgecolor="white")
a1.set_ylim(0, 108)
a1.set_ylabel("OOD-abstained fraction (%)")
a1.set_xticks(x); a1.set_xticklabels(labels, rotation=25, ha="right")
a1.set_title("Out-of-distribution rate (ORB-PU OOD criterion)")
a1.grid(axis="y", ls=":", c="0.85", zorder=0)
for xi, v in zip(x, g.ood * 100):
    a1.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_bogen_synth_per_backbone.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote fig_bogen_synth_scatter.png and fig_bogen_synth_per_backbone.png to", OUT)
print(f"rho={rho:.4f} agree={agree:.3f}")
