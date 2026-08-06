#!/usr/bin/env python
"""Regenerate the manuscript synthesizability figures with the corrected OOD flag.

Paper artifacts (final manuscript versions):
  Fig 10 (main)  fig_bogen_synth_per_backbone_oodfix   per-(backbone, policy) scores + OOD rate
  Fig S18 (SI)   fig_bogen_synth_scatter_oodfix        A-PU vs CGNF scatter, OOD-flagged

The OOD detector's dist95_ threshold was originally fit with a kNN self-match
(each positive's nearest neighbor is itself, distance 0), deflating the
threshold; new structures at score time have no self-match, so ood_score was
inflated -> over-abstention. Fix: recompute dist95_ excluding the self-neighbor
(query k+1, drop col 0). Effect on the published 40-structure set:
abstain_ood 30/40 (75%) -> 21/40 (52.5%). Spearman rho and 0.5-agreement are
computed on all 40 scores and are UNCHANGED. The in-tree scores.csv carries the
pre-fix flags; this generator recomputes the corrected flags from the trained
model, exactly as the manuscript figures were made.

Model resolution (the trained ORB-PU model is not stored in git): (1) an
in-tree copy at 03_synthesizability/results/apu_optuna/orb_mag__xgboost/
model.joblib if present; (2) the MATGEN_BO_APU_MODEL environment variable;
(3) automatic download from the paper's Hugging Face archive
(SheikhAhnaf/apu-synthesizability-checkpoints, DOI 10.57967/hf/9893).

Provenance: verbatim transform of fig_redesign_20260611/make_synth_oodfix.py
(2026-06-19); only the path resolution differs. Outputs are written to
figures/regenerated/oodfix_regen/ and never overwrite figures/rendered/.
"""
import os, sys, warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "03_synthesizability" / "src"))
OUT = ROOT / "03_synthesizability" / "figures" / "regenerated" / "oodfix_regen"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

df = pd.read_csv(ROOT / "03_synthesizability" / "results" / "generated_scores" / "scores.csv")

# --- locate the trained ORB-PU model (in-tree -> env var -> Hugging Face) ---
_cands = [ROOT / "03_synthesizability" / "results" / "apu_optuna" / "orb_mag__xgboost" / "model.joblib"]
if os.environ.get("MATGEN_BO_APU_MODEL"):
    _cands.append(Path(os.environ["MATGEN_BO_APU_MODEL"]))
model_path = next((p for p in _cands if p.exists()), None)
if model_path is None:
    from huggingface_hub import hf_hub_download
    model_path = Path(hf_hub_download("SheikhAhnaf/apu-synthesizability-checkpoints",
                                      "apu_optuna/orb_mag__xgboost/model.joblib"))
print(f"model: {model_path}")

# --- corrected OOD flag (self-match removed) ---
m = joblib.load(model_path)
nn, d95_buggy, thr = m.nn_, float(m.dist95_), float(getattr(m, "ood_threshold", 1.0))
Xp, k = nn._fit_X, nn.n_neighbors
nn2 = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Xp)
d_all, _ = nn2.kneighbors(Xp)
d95_corr = float(np.percentile(d_all[:, 1:].mean(axis=1), 95)) + 1e-6
df["ood_score_corr"] = df["ood_score"] * (d95_buggy / d95_corr)
df["abstain_ood_corr"] = df["ood_score_corr"] > thr
print(f"dist95 buggy={d95_buggy:.4f} corrected={d95_corr:.4f}")
print(f"abstain_ood: was {int(df.abstain_ood.sum())}/40 ({100*df.abstain_ood.mean():.1f}%) "
      f"-> corrected {int(df.abstain_ood_corr.sum())}/40 ({100*df.abstain_ood_corr.mean():.1f}%)")

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 200})
COLORS = {"ADiT": "#1b7837", "CrystalFlow": "#2166ac", "MatterGen": "#b2182b"}
order = ["ADiT", "CrystalFlow", "MatterGen"]
OODCOL = "abstain_ood_corr"

# ---- Figure 1: parity scatter (open = corrected OOD) ----
fig, ax = plt.subplots(figsize=(5.6, 5.4))
for bb in order:
    sub = df[df.backbone_name == bb]
    kept, ood = sub[~sub[OODCOL]], sub[sub[OODCOL]]
    ax.scatter(kept.cgnf_score, kept.apu_score, s=46, c=COLORS[bb],
               edgecolors="white", linewidths=0.5, label=bb, zorder=3)
    ax.scatter(ood.cgnf_score, ood.apu_score, s=46, facecolors="none",
               edgecolors=COLORS[bb], linewidths=1.4, zorder=3)
ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1.0, zorder=1)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0, .25, .5, .75, 1.]); ax.set_yticks([0, .25, .5, .75, 1.])
ax.set_xlabel("CGNF synthesizability score")
ax.set_ylabel("A-PU synthesizability probability")
rho, _ = spearmanr(df.apu_score, df.cgnf_score)
agree = float(np.mean((df.apu_score > 0.5) == (df.cgnf_score > 0.5)))
ax.set_title(f"A-PU vs CGNF on 40 generated structures\nSpearman $\\rho$={rho:.2f}, "
             f"agreement at 0.5 = {agree*100:.0f}%")
bb_handles = [Line2D([], [], marker="o", ls="", mfc=COLORS[b], mec="white", ms=8, label=b) for b in order]
flag_handles = [Line2D([], [], marker="o", ls="", mfc="0.4", mec="white", ms=8, label="in-distribution"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="0.4", mew=1.4, ms=8, label="OOD (abstained)")]
leg1 = ax.legend(handles=bb_handles, title="backbone", loc="upper left", frameon=False)
ax.add_artist(leg1)
ax.legend(handles=flag_handles, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_bogen_synth_scatter_oodfix.png", bbox_inches="tight")
fig.savefig(OUT / "fig_bogen_synth_scatter_oodfix.pdf", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: per-backbone panel (OOD rate uses corrected) ----
g = (df.groupby(["backbone_name", "policy"])
       .agg(n=("apu_score", "size"), apu=("apu_score", "mean"),
            cgnf=("cgnf_score", "mean"), ood=(OODCOL, "mean")).reset_index())
g["bborder"] = g.backbone_name.map({b: i for i, b in enumerate(order)})
g = g.sort_values(["bborder", "policy"]).reset_index(drop=True)
labels = [f"{r.backbone_name}\n{r.policy} (n={r.n})" for r in g.itertuples()]
x = np.arange(len(g)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax0 = axes[0]
ax0.bar(x - w/2, g.apu, w, label="A-PU", color="#4575b4", edgecolor="white")
ax0.bar(x + w/2, g.cgnf, w, label="CGNF", color="#d6604d", edgecolor="white")
ax0.set_ylim(0, 1.0); ax0.set_ylabel("mean synthesizability score")
ax0.set_xticks(x); ax0.set_xticklabels(labels); ax0.set_title("Mean score per backbone and policy")
ax0.legend(frameon=False, loc="upper right"); ax0.grid(axis="y", ls=":", c="0.85", zorder=0)
ax1 = axes[1]
ax1.bar(x, g.ood * 100, 0.6, color="#7f7f7f", edgecolor="white")
ax1.set_ylim(0, 105); ax1.set_ylabel("OOD-abstained fraction (%)")
ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_title("Out-of-distribution rate (A-PU OOD layer, corrected)")
ax1.grid(axis="y", ls=":", c="0.85", zorder=0)
for xi, v in zip(x, g.ood * 100):
    ax1.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_bogen_synth_per_backbone_oodfix.png", bbox_inches="tight")
fig.savefig(OUT / "fig_bogen_synth_per_backbone_oodfix.pdf", bbox_inches="tight")
plt.close(fig)

print(f"\nSpearman rho={rho:.4f}  agreement={agree:.3f}  (unchanged by OOD fix)")
print("\nper-(backbone,policy) OOD rate, OLD vs CORRECTED:")
g_old = df.groupby(["backbone_name", "policy"])["abstain_ood"].mean()
for (bb, pol), v_old in g_old.items():
    v_new = df[(df.backbone_name == bb) & (df.policy == pol)][OODCOL].mean()
    print(f"  {bb:12s} {pol:5s}: {100*v_old:5.1f}% -> {100*v_new:5.1f}%")
print("\nsaved fig_bogen_synth_scatter_oodfix + fig_bogen_synth_per_backbone_oodfix")
