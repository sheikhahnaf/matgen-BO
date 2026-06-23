#!/usr/bin/env python
"""Ranking analysis for the FAITHFUL (seed-inclusive, ACC-only) causal replay.

Reads gp_causal_allpreds_<prop>_seeded.csv (from three_way_causal_seeded.py). The GP is
trained WITH the warm-start seed, but per the plotting rule the x-axis (n_train_accum)
counts only the loop-gathered structures, EXCLUDING the ~500-structure warm start. So the
learning curve answers: "given the warm start, does extra loop-gathered data change the
ranking?" (Expected: largely flat/high — the seed does the work.)

Usage: python rank_analysis_seeded.py <bm|cp>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROP = sys.argv[1] if len(sys.argv) > 1 else "bm"
RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
ALL = os.path.join(RES, "gp_causal_allpreds_%s_seeded.csv" % PROP)
UNITS = "K$_0$ (GPa)" if PROP == "bm" else "C$_v$ (J/g/K)"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 8.5, "figure.dpi": 150, "axes.linewidth": 0.8})
# accumulated (seed-excluded) bins, sized to the loop-gathered range
ACC_BINS = [0, 5, 15, 30, 50, 10**9]
ACC_LAB = ["0–5", "5–15", "15–30", "30–50", "50+"]
DISTINCT_MIN = 3


def main():
    d = pd.read_csv(ALL)
    print("loaded %d causal preds, %d ACC runs, n_train_accum 0..%d (seed excluded from axis)"
          % (len(d), d["run"].nunique(), d["n_train_accum"].max()))
    overall = spearmanr(d["gp_pred"], d["esen"]).correlation
    overall_r = pearsonr(d["gp_pred"], d["esen"])[0]
    rmse = ((d["gp_pred"] - d["esen"]) ** 2).mean() ** .5
    print("overall GP(causal) vs eSEN: rho=%.3f r=%.3f RMSE=%.3f" % (overall, overall_r, rmse))

    # A: within-step rho
    rows = []
    for (run, s), g in d.groupby(["run", "cycle_id"]):
        nd = g["gp_pred"].round(4).nunique()
        rho = spearmanr(g["gp_pred"], g["esen"]).correlation if (len(g) >= 3 and nd >= DISTINCT_MIN) else np.nan
        rows.append(dict(run=run, cycle_id=int(s), n=len(g), rho=rho))
    within = pd.DataFrame(rows); within.to_csv(os.path.join(RES, "rank_within_step_%s_seeded.csv" % PROP), index=False)
    rk = within.dropna(subset=["rho"])
    print("within-step rho: mean=%.3f median=%.3f frac>0=%.2f (rankable=%d)"
          % (rk["rho"].mean(), rk["rho"].median(), (rk["rho"] > 0).mean(), len(rk)))

    # B: by accumulated (seed-excluded) bin
    d["accbin"] = pd.cut(d["n_train_accum"], bins=ACC_BINS, right=False, labels=ACC_LAB)
    byb = d.groupby("accbin").apply(lambda g: pd.Series(dict(
        n=len(g), rho=spearmanr(g["gp_pred"], g["esen"]).correlation if len(g) >= 3 else np.nan))).reset_index()
    byb.to_csv(os.path.join(RES, "rank_by_accum_%s_seeded.csv" % PROP), index=False)
    print("\nranking vs loop-accumulated (seed excluded from count):"); print(byb.round(3).to_string(index=False))

    # C: by cycle
    bystep = d.groupby("cycle_id").apply(lambda g: pd.Series(dict(
        n=len(g), rho=spearmanr(g["gp_pred"], g["esen"]).correlation if len(g) >= 3 else np.nan))).reset_index()
    bystep.to_csv(os.path.join(RES, "rank_by_cycle_%s_seeded.csv" % PROP), index=False)

    # ---- figure: ranking ----
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7))
    b = byb.dropna(subset=["rho"])
    ax[0].bar(range(len(b)), b["rho"], color="#3a6ea5", width=0.62, edgecolor="black", linewidth=0.6)
    for i, (rho, n) in enumerate(zip(b["rho"], b["n"])):
        ax[0].text(i, rho + 0.02, "%.2f\n(n=%d)" % (rho, int(n)), ha="center", va="bottom", fontsize=7.5)
    ax[0].set_xticks(range(len(b))); ax[0].set_xticklabels(b["accbin"])
    ax[0].set_xlabel("loop-accumulated training points\n(warm-start seed EXCLUDED)")
    ax[0].set_ylabel("Spearman ρ (GP vs eSEN)")
    ax[0].set_title("(a) Warm-started: ranking ~flat\nvs loop-accumulated data")
    ax[0].set_ylim(0, 1.0); ax[0].axhline(0, color="gray", lw=0.6)
    bs = bystep.dropna(subset=["rho"])
    ax[1].plot(bs["cycle_id"], bs["rho"], "o-", color="#b5482a", ms=4, lw=1.4)
    ax[1].set_xlabel("RL step (cycle)"); ax[1].set_ylabel("pooled Spearman ρ")
    ax[1].set_title("(b) Ranking quality across cycles")
    ax[1].set_ylim(0, 1.0); ax[1].axhline(0, color="gray", lw=0.6); ax[1].grid(True, alpha=0.25, lw=0.5)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].hist(rk["rho"], bins=np.linspace(-1, 1, 21), color="#5a8f5a", edgecolor="black", linewidth=0.5)
    ax[2].axvline(rk["rho"].mean(), color="black", ls="--", lw=1.2, label="mean = %.2f" % rk["rho"].mean())
    ax[2].set_xlabel("within-step Spearman ρ"); ax[2].set_ylabel("number of step-batches")
    ax[2].set_title("(c) Per-step ranking spread\n(%d rankable batches)" % len(rk))
    ax[2].legend(frameon=False, loc="upper left")
    fig.suptitle("Faithful causal ranking — %s (GP warm-started on seed; axis = loop-gathered only)" % UNITS,
                 fontsize=12, y=1.04)
    fig.tight_layout()
    f1 = os.path.join(RES, "three_way_ranking_%s_seeded.png" % PROP)
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig); print("\nsaved -> %s" % f1)

    # ---- figure: property scatter ----
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
    sc = ax.scatter(d["esen"], d["gp_pred"], c=d["n_train_accum"], cmap="viridis", s=10, alpha=0.45, linewidths=0)
    hi = max(d["esen"].max(), d["gp_pred"].max()) * 1.02
    ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.7, label="parity")
    tw = os.path.join(RES, "three_way_causal_%s_seeded.csv" % PROP)
    if os.path.exists(tw):
        t = pd.read_csv(tw)
        ax.scatter(t["esen_K0"], t["gp_causal_K0"], facecolors="none", edgecolors="red", s=70, linewidths=1.4,
                   label="DFT-validated winners (ACC)")
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("loop-accumulated n_train (seed excl.)", fontsize=8.5)
    ax.set_xlabel("eSEN oracle %s" % UNITS); ax.set_ylabel("GP causal prediction %s" % UNITS)
    ax.set_title("Faithful causal GP vs eSEN (seed-warm)\nρ=%.2f r=%.2f RMSE=%.2f (n=%d)" % (overall, overall_r, rmse, len(d)))
    ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    f2 = os.path.join(RES, "three_way_property_%s_seeded.png" % PROP)
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig); print("saved -> %s" % f2)


if __name__ == "__main__":
    main()
