"""Poster-scaled benchmark figure: best held-out R^2 (n_train = 500) per featurizer on
each dataset, best surrogate labelled in the bar. Values are those of the manuscript
figure fig_cross_dataset_R2_by_descriptor.png (FoundationalEmbeddings_2026)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Helvetica", "font.size": 13.5, "axes.titlesize": 15,
                     "axes.labelsize": 14.5, "xtick.labelsize": 13.5, "ytick.labelsize": 13,
                     "legend.fontsize": 12.5, "axes.linewidth": 1.0})
DATASETS = ["Elastic", "Dielectric", "Phonon"]
BEST = {  # featurizer: [(R2, best surrogate) per dataset]
    "SOAP": [(0.445, "GP"), (0.177, "MTGP"), (0.837, "GP")],
    "MACE": [(0.731, "GP"), (0.239, "DGP"), (0.928, "GP")],
    "UMA":  [(0.734, "GP"), (0.179, "DGP"), (0.909, "DGP")],
    "ORB":  [(0.807, "DGP"), (0.336, "DGP"), (0.968, "GP")],
}
COL = {"SOAP": "#b9b9b9", "MACE": "#6b8fb5", "UMA": "#c58a3a", "ORB": "#500000"}

fig, ax = plt.subplots(figsize=(8.2, 5.2))
x = np.arange(len(DATASETS)); w = 0.2
for i, (feat, vals) in enumerate(BEST.items()):
    pos = x + (i - 1.5) * w
    r2 = [v[0] for v in vals]
    ax.bar(pos, r2, w * 0.92, color=COL[feat], edgecolor="black", lw=0.6, label=feat)
    for p, (v, s) in zip(pos, vals):
        ax.text(p, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)
        ax.text(p, 0.02, s, ha="center", va="bottom", fontsize=9.5, rotation=90,
                color="white" if feat in ("ORB", "MACE") else "black")
ax.set_xticks(x); ax.set_xticklabels(DATASETS)
ax.set_ylabel("Best held-out R$^2$  (n$_{train}$ = 500)")
ax.set_ylim(0, 1.08)
ax.set_title("Featurizer comparison across three property datasets")
ax.legend(ncol=4, frameon=False, loc="upper left")
ax.grid(True, axis="y", alpha=0.25, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "figs" / "fig_benchmark_poster.png", dpi=220, bbox_inches="tight")
print("wrote figs/fig_benchmark_poster.png")
