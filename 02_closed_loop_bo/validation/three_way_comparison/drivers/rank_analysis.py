#!/usr/bin/env python
"""Ranking quality of the GP, step-by-step, on freshly-generated structures.

Companion to the property-value comparison (three_way_causal.py): instead of asking how
close the GP's predicted K0 is to eSEN, this asks how well the GP ORDERS the candidates it
sees at each cycle, which is what the EI+DPP acquisition gate actually consumes. Reads the
causal step-ahead predictions (gp_causal_allpreds.csv: one row per generated structure, with
the GP's prediction made when only earlier-step structures were known).

Metrics:
  A. Within-step Spearman  — per (run, RL_step) batch, rho(gp_pred, eSEN). The direct "rank
     at each step" measure. Batches where the GP returned a near-constant mean (no
     discrimination) are reported separately, not silently scored as rho=0.
  B. Ranking vs accumulated data — pooled Spearman binned by n_train (how well the GP ranks
     once it has seen ~N labels). This is the honest learning curve.
  C. Ranking vs cycle — pooled Spearman per RL_step.
  D. Acquisition relevance — per-step recall@1/@3 (does the GP's top pick(s) coincide with
     the step's true best by eSEN) vs the random baseline, and a global precision@k.

Outputs (in ../results): rank_within_step.csv, rank_by_ntrain.csv, rank_by_step.csv,
three_way_ranking.png, three_way_property.png. Pure pandas/scipy — no ORB, runs anywhere.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ALL = os.path.join(RES, "gp_causal_allpreds.csv")
TW = os.path.join(RES, "three_way_causal.csv")

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "axes.linewidth": 0.8,
})
NTBINS = [10, 25, 50, 100, 200, 10**9]
NTLAB = ["10–25", "25–50", "50–100", "100–200", "200+"]
DISTINCT_MIN = 3   # a batch must have >=3 distinct GP predictions to be rankable


def pooled_rho(g):
    return spearmanr(g["gp_pred_K0"], g["esen_K0"]).correlation if len(g) >= 3 else np.nan


def main():
    d = pd.read_csv(ALL)
    print("loaded %d causal predictions, %d runs, RL_step %d-%d"
          % (len(d), d["run"].nunique(), d["RL_step"].min(), d["RL_step"].max()))

    # ---------- A. within-step ranking ----------
    rows = []
    for (run, s), g in d.groupby(["run", "RL_step"]):
        ndist = g["gp_pred_K0"].round(2).nunique()
        rho = spearmanr(g["gp_pred_K0"], g["esen_K0"]).correlation if (len(g) >= 3 and ndist >= DISTINCT_MIN) else np.nan
        rows.append(dict(run=run, RL_step=int(s), n=len(g), n_distinct_pred=ndist,
                         n_train=int(g["n_train"].iloc[0]), rho=rho))
    within = pd.DataFrame(rows)
    within.to_csv(os.path.join(RES, "rank_within_step.csv"), index=False)
    rankable = within.dropna(subset=["rho"])
    ge3 = within[within["n"] >= 3]
    print("\n[A] within-step ranking:")
    print("  batches total=%d, with>=3 structs=%d, rankable(>=3 distinct preds)=%d, "
          "non-discriminating(GP~const mean)=%d"
          % (len(within), len(ge3), len(rankable), len(ge3) - len(rankable)))
    print("  mean within-step rho=%.3f  median=%.3f  frac(rho>0)=%.2f"
          % (rankable["rho"].mean(), rankable["rho"].median(), (rankable["rho"] > 0).mean()))

    # ---------- B. ranking vs accumulated data ----------
    d["ntbin"] = pd.cut(d["n_train"], bins=NTBINS, right=False, labels=NTLAB)
    byb = d.groupby("ntbin").apply(lambda g: pd.Series(dict(
        n=len(g), rho=pooled_rho(g), r=pearsonr(g["gp_pred_K0"], g["esen_K0"])[0],
        rmse=((g["gp_pred_K0"] - g["esen_K0"]) ** 2).mean() ** .5))).reset_index()
    byb.to_csv(os.path.join(RES, "rank_by_ntrain.csv"), index=False)
    print("\n[B] ranking vs accumulated data (pooled within n_train bin):")
    print(byb.round(2).to_string(index=False))

    # ---------- C. ranking vs cycle ----------
    bystep = d.groupby("RL_step").apply(lambda g: pd.Series(dict(
        n=len(g), rho=pooled_rho(g)))).reset_index()
    bystep.to_csv(os.path.join(RES, "rank_by_step.csv"), index=False)

    # ---------- D. acquisition relevance ----------
    def recall_at(g, k):
        if len(g) <= k:
            return np.nan
        gp_top = set(g.nlargest(k, "gp_pred_K0").index)
        es_top = set(g.nlargest(k, "esen_K0").index)
        return len(gp_top & es_top) / k
    r1 = d.groupby(["run", "RL_step"]).apply(lambda g: recall_at(g, 1)).dropna()
    r3 = d.groupby(["run", "RL_step"]).apply(lambda g: recall_at(g, 3)).dropna()
    base1 = d.groupby(["run", "RL_step"]).apply(lambda g: 1.0 / len(g) if len(g) > 1 else np.nan).dropna()
    print("\n[D] acquisition relevance (per-step):")
    print("  recall@1 (GP top-1 == eSEN top-1): %.2f   (random baseline %.2f, n=%d steps)"
          % (r1.mean(), base1.mean(), len(r1)))
    print("  recall@3 (overlap of top-3 picks):  %.2f   (n=%d steps)" % (r3.mean(), len(r3)))
    # global precision@k by cif identity
    gp_topk = {k: set(d.sort_values("gp_pred_K0", ascending=False).head(k)["cif"]) for k in (50, 100, 200)}
    es_topk = {k: set(d.sort_values("esen_K0", ascending=False).head(k)["cif"]) for k in (50, 100, 200)}
    print("  global precision@k (GP-ordered top-k ∩ eSEN top-k):")
    for k in (50, 100, 200):
        print("    @%-4d %.2f" % (k, len(gp_topk[k] & es_topk[k]) / k))

    overall = spearmanr(d["gp_pred_K0"], d["esen_K0"]).correlation
    overall_r = pearsonr(d["gp_pred_K0"], d["esen_K0"])[0]
    print("\noverall GP(causal) vs eSEN: rho=%.3f r=%.3f n=%d" % (overall, overall_r, len(d)))

    # ================= FIGURE 1: ranking analysis =================
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7))
    # B: rho vs n_train bin
    b = byb.dropna(subset=["rho"])
    ax[0].bar(range(len(b)), b["rho"], color="#3a6ea5", width=0.62, edgecolor="black", linewidth=0.6)
    for i, (rho, n) in enumerate(zip(b["rho"], b["n"])):
        ax[0].text(i, rho + 0.02, "%.2f\n(n=%d)" % (rho, int(n)), ha="center", va="bottom", fontsize=7.5)
    ax[0].set_xticks(range(len(b))); ax[0].set_xticklabels(b["ntbin"], rotation=0)
    ax[0].set_xlabel("GP training size when predicting (n_train)")
    ax[0].set_ylabel("Spearman ρ (GP vs eSEN)")
    ax[0].set_title("(a) Ranking improves as the GP\naccumulates data")
    ax[0].set_ylim(0, max(0.85, b["rho"].max() + 0.15)); ax[0].axhline(0, color="gray", lw=0.6)
    # C: rho vs RL_step
    bs = bystep.dropna(subset=["rho"])
    ax[1].plot(bs["RL_step"], bs["rho"], "o-", color="#b5482a", ms=4, lw=1.4)
    ax[1].set_xlabel("RL step (cycle)"); ax[1].set_ylabel("pooled Spearman ρ")
    ax[1].set_title("(b) Ranking quality across cycles")
    ax[1].set_ylim(0, max(0.85, bs["rho"].max() + 0.15)); ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].grid(True, alpha=0.25, lw=0.5)
    # RL cycle is integer-valued; fractional ticks (2.5) are meaningless
    ax[1].set_xticks(range(int(bs["RL_step"].min()), int(bs["RL_step"].max()) + 1, 2))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # A: within-step rho distribution
    ax[2].hist(rankable["rho"], bins=np.linspace(-1, 1, 21), color="#5a8f5a", edgecolor="black", linewidth=0.5)
    ax[2].axvline(rankable["rho"].mean(), color="black", ls="--", lw=1.2,
                  label="mean = %.2f" % rankable["rho"].mean())
    ax[2].set_xlabel("within-step Spearman ρ"); ax[2].set_ylabel("number of step-batches")
    ax[2].set_title("(c) Per-step ranking spread\n(%d rankable batches)" % len(rankable))
    ax[2].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    f1 = os.path.join(RES, "three_way_ranking.png")
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig)
    print("\nsaved figure -> %s" % f1)

    # ================= FIGURE 2: property-value comparison =================
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
    sc = ax.scatter(d["esen_K0"], d["gp_pred_K0"], c=d["n_train"], cmap="viridis",
                    s=10, alpha=0.45, linewidths=0)
    lo, hi = 0, max(d["esen_K0"].max(), d["gp_pred_K0"].max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7, label="parity")
    # overlay validated winners
    if os.path.exists(TW):
        tw = pd.read_csv(TW)
        ax.scatter(tw["esen_K0"], tw["gp_causal_K0"], facecolors="none", edgecolors="red",
                   s=70, linewidths=1.4, label="DFT-validated winners")
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("n_train at prediction", fontsize=9)
    ax.set_xlabel("eSEN oracle K$_0$ (GPa)"); ax.set_ylabel("GP causal prediction K$_0$ (GPa)")
    ax.set_title("Causal GP prediction vs eSEN\nρ=%.2f, r=%.2f, RMSE=%.0f GPa (n=%d)"
                 % (overall, overall_r, ((d["gp_pred_K0"] - d["esen_K0"]) ** 2).mean() ** .5, len(d)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    f2 = os.path.join(RES, "three_way_property.png")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
    print("saved figure -> %s" % f2)


if __name__ == "__main__":
    main()
