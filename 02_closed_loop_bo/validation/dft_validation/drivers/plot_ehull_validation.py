#!/usr/bin/env python
"""SI figure: DFT energy-above-hull of the generated structures (MP2020 anion-corrected).
(a) uncorrected vs corrected E_hull -- the correction is one-signed and touches only the 5 anion
    phases; the 23 anion-free phases lie on the diagonal (unchanged).
(b) corrected-E_hull distribution against the on/near-hull (<=0.10) and metastable-ceiling (0.20)
    tiers; Na2BO3 (charge-imbalanced bad generation) flagged and excluded from the tally.
Reads dft_validation/results/ehull_summary_corrected.csv. Outputs to the FME paper figures/ dir.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "axes.labelsize": 10.5,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 900, "figure.dpi": 900,
})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "ehull_summary_corrected.csv")
OUT = os.path.expanduser("~/fme_paper_work/FoundationalEmbeddings_2026/figures/fig_ehull_validation.png")

rows = list(csv.DictReader(open(CSV)))
for r in rows:
    r["old"] = float(r["e_hull_old"]); r["new"] = float(r["e_hull_new"])
    r["is_anion"] = r["anion"].strip() == "yes"
    r["bad"] = r["formula"] == "Na2BO3"

BLUE, RED, GRAY = "#2c6fbb", "#c0392b", "#777777"
GREEN, AMBER = "#3a8f5a", "#d99a2b"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)

# --- (a) uncorrected vs corrected ---
hi = 0.40
ax1.plot([0, hi], [0, hi], "--", color=GRAY, lw=1.0, zorder=1)
af = [r for r in rows if not r["is_anion"]]
an = [r for r in rows if r["is_anion"]]
ax1.scatter([r["old"] for r in af], [r["new"] for r in af], s=34, color=BLUE,
            edgecolor="white", linewidth=0.5, zorder=3, label="anion-free (23): unchanged")
ax1.scatter([r["old"] for r in an], [r["new"] for r in an], s=46, color=RED,
            edgecolor="white", linewidth=0.6, zorder=4, label="anion phase (5): corrected")
lab = {"MoN": (8, 5), "TiVN2": (-40, 2), "Li6TiO5": (7, -1), "Li3PO4": (-44, 9), "Na2BO3": (5, 17)}
for r in an:
    dx, dy = lab.get(r["formula"], (5, 5))
    ax1.annotate(r["formula"], (r["old"], r["new"]), textcoords="offset points",
                 xytext=(dx, dy), fontsize=7.5, color=RED,
                 arrowprops=dict(arrowstyle="-", color=RED, lw=0.5, shrinkA=0, shrinkB=2))
ax1.set_xlim(0, hi); ax1.set_ylim(0, hi); ax1.set_aspect("equal", "box")
ax1.set_xlabel(r"uncorrected $E_\mathrm{hull}$ (eV/atom)")
ax1.set_ylabel(r"MP2020-corrected $E_\mathrm{hull}$ (eV/atom)")
ax1.set_title("(a) effect of the anion correction", loc="left", fontweight="bold")
ax1.legend(frameon=False, loc="upper left", fontsize=8)

# --- (b) corrected-E_hull distribution vs tiers ---
ax2.axvspan(0.0, 0.10, color=GREEN, alpha=0.10)
ax2.axvspan(0.10, 0.20, color=AMBER, alpha=0.12)
ax2.axvline(0.10, color=GREEN, lw=1.0, ls="--")
ax2.axvline(0.20, color=AMBER, lw=1.0, ls="--")
rng = np.random.default_rng  # not used; deterministic jitter below
good = [r for r in rows if not r["bad"]]
# deterministic vertical jitter by index so points don't overlap
ys = [(i % 7) * 0.12 + 0.1 for i in range(len(good))]
cols = [GREEN if r["new"] <= 0.10 else AMBER for r in good]
ax2.scatter([r["new"] for r in good], ys, s=34, c=cols, edgecolor="white", linewidth=0.5, zorder=3)
# Na2BO3 flagged separately (open marker, excluded)
bad = [r for r in rows if r["bad"]][0]
ax2.scatter([bad["new"]], [0.95], s=52, facecolors="none", edgecolors=RED, linewidth=1.4, zorder=4)
ax2.annotate("Na$_2$BO$_3$\n(charge-imbalanced,\nexcluded)", (bad["new"], 0.95),
             textcoords="offset points", xytext=(8, -2), fontsize=7, color=RED, va="center")
ax2.set_xlim(-0.01, 0.24); ax2.set_ylim(0, 1.15)
ax2.set_yticks([])
ax2.set_xlabel(r"MP2020-corrected $E_\mathrm{hull}$ (eV/atom)")
ax2.set_title("(b) corrected stability distribution", loc="left", fontweight="bold")
ax2.text(0.05, 1.08, "on/near hull\n$\\leq$0.10 (21)", fontsize=7.5, color=GREEN, ha="center", va="top")
ax2.text(0.15, 1.08, "metastable\n0.10$-$0.20 (6)", fontsize=7.5, color="#b07d1a", ha="center", va="top")

fig.savefig(OUT, bbox_inches="tight")
print("saved ->", OUT)
n_le = sum(1 for r in good if r["new"] <= 0.10); n_meta = sum(1 for r in good if 0.10 < r["new"] <= 0.20)
print("valid (excl Na2BO3)=%d  <=0.10: %d  0.10-0.20: %d  >0.20: %d"
      % (len(good), n_le, n_meta, sum(1 for r in good if r["new"] > 0.20)))
