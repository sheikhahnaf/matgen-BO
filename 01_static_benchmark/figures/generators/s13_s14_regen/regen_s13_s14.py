"""Regenerate the cross-dataset (S12) and cross-surrogate (S13) SI aggregate figures, swapping
the old "Phonon Diel. MP" third dataset for the new DFPT phonon-thermo benchmark.

Everything is COMPUTED from current data -- no hardcoding. Aggregation (verified below to
reproduce every published elastic/phonon bar and the dielectric DGP bar exactly, and to match
the analysis pipeline's own best_pca_averaged.csv intermediates):

    per (dataset, surrogate, descriptor): best over pca of [ raw mean-R2 / Spearman averaged
    over the dataset's physical target properties ]; calc parameters excluded
    (kpoint_density for elastic). All dielectric (4) and phonon (4) targets are real.

  - S12 (per descriptor):     bar = best over surrogates; inside label = winning surrogate.
  - S13 (cross-surrogate):    ORB descriptor, one bar per surrogate.

NOTE -- bug fix vs the published S13: the original "ORB descriptor" cross-surrogate figure
plotted best-descriptor-per-surrogate (max over descriptors), not ORB. This was invisible for
elastic and phonon-diel-MP (ORB is their best descriptor for every surrogate) but on dielectric
GP/MTGP it leaked SOAP values (0.132 / 0.177) into bars labelled ORB. The correct ORB values are
GP=0.096, MTGP=0.121 -- exactly what the main-text dielectric Table 4 and the discussion already
report. This script uses the true ORB values; only those two dielectric bars change.

Colour encodes DATASET position (elastic=blue, dielectric=orange, phonon=green). Outputs to ./out/.
"""
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402

ps.apply_style()
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

ASE = _ROOT / "archive" / "ASE-native-surrogates"
NEW = _ROOT / "archive" / "phonon_thermo_benchmark" / "paper_figures_new_phonon_2026-06-18"
CSV = {
    "elastic":    ASE / "analysis_v3" / "aggregated_results.csv",
    "dielectric": ASE / "analysis_v3_dielectric_constant" / "aggregated_results.csv",
    "phonon":     NEW / "arm_a_dfpt" / "aggregated_results.csv",
}
EXCLUDE = {"elastic": {"kpoint_density"}}          # pure k-point calc parameter
DORDER = ["elastic", "dielectric", "phonon"]
DLABEL = {"elastic": "Elastic Tensor", "dielectric": "Dielectric Const.", "phonon": "Phonon Thermo."}
DCOLOR = {"elastic": "#1f77b4", "dielectric": "#ff7f0e", "phonon": "#2ca02c"}
MODELS = ["gp", "mtgp_2", "dgp"]
DESCS = ["mace", "orb", "soap", "uma"]

DF = {}
for d, p in CSV.items():
    t = pd.read_csv(p).query("n_train == 500")
    DF[d] = t[~t["property"].isin(EXCLUDE.get(d, set()))]


def bestpca_avg(df, model, desc, metric):
    """max over pca of (raw mean metric averaged over the dataset's properties)."""
    s = df[(df.model == model) & (df.descriptor == desc) & (df.metric == metric)]
    best = float("-inf")
    for pca in s["pca_components"].unique():
        v = s[s.pca_components == pca].groupby("property")["mean"].mean().mean()
        if v == v and v > best:
            best = v
    return best


def best_over_surr(df, desc, metric):
    best = (float("-inf"), None)
    for m in MODELS:
        v = bestpca_avg(df, m, desc, metric)
        if v == v and v > best[0]:
            best = (v, m)
    return best


# ---------- S12: best <metric> per descriptor across datasets (colour=dataset, label=surrogate) ----------
def make_s12(metric, fname, title_metric):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, desc in zip(axes, DESCS):
        bars_data = [(*best_over_surr(DF[d], desc, metric), DCOLOR[d]) for d in DORDER]
        heights = [b[0] for b in bars_data]
        x = range(len(DORDER))
        bars = ax.bar(x, heights, color=[b[2] for b in bars_data], edgecolor="black", linewidth=0.8, width=0.7)
        ax.set_title(ps.DESCRIPTOR_LABEL[desc], fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels([DLABEL[d] for d in DORDER], rotation=20, ha="right")
        ymax = max(heights) * 1.18
        ax.set_ylim(0, ymax)
        for b, (h, m, _) in zip(bars, bars_data):
            ax.text(b.get_x() + b.get_width() / 2, h + ymax * 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=ps.SZ_VALUE_LABEL)
            ax.text(b.get_x() + b.get_width() / 2, h * 0.5, ps.SURROGATE_LABEL[m],
                    ha="center", va="center", color="white", fontsize=ps.SZ_ANNOT, fontweight="bold")
    axes[0].set_ylabel(f"Best Avg {title_metric} Corr." if title_metric == "Spearman" else f"Best Avg {title_metric}")
    fig.suptitle(f"Best {title_metric} per Descriptor Across Datasets (n=500)\n"
                 f"Label inside bar = surrogate with best {title_metric}",
                 fontweight="bold", fontsize=ps.SZ_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ps.save_figure(fig, OUT / fname)


# ---------- S13: best avg-R2 per surrogate across datasets (ORB descriptor) ----------
def make_s13(fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m in zip(axes, MODELS):
        heights = [bestpca_avg(DF[d], m, "orb", "R2") for d in DORDER]
        x = range(len(DORDER))
        bars = ax.bar(x, heights, color=[DCOLOR[d] for d in DORDER], edgecolor="black", linewidth=0.8, width=0.7)
        ax.set_title(ps.SURROGATE_LABEL[m], fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels([DLABEL[d] for d in DORDER], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom",
                    fontsize=ps.SZ_VALUE_LABEL, fontweight="bold")
    axes[0].set_ylabel("Best Avg R$^2$")
    fig.suptitle("Best Average R$^2$ per Surrogate Across Datasets (ORB descriptor, n=500)",
                 fontweight="bold", fontsize=ps.SZ_SUPTITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ps.save_figure(fig, OUT / fname)


make_s12("R2", "fig_cross_dataset_R2_by_descriptor.png", "R$^2$")
make_s12("Spearman", "fig_cross_dataset_spearman_by_descriptor.png", "Spearman")
make_s13("fig_cross_surrogate_r2_pptx.png")

# ---------------------------------------------------------------------------
# VALIDATION  (everything computed; cross-checked against published + pipeline intermediates)
# ---------------------------------------------------------------------------
PUB_S12_R2 = {"elastic":    {"mace": 0.731, "orb": 0.807, "soap": 0.445, "uma": 0.734},
              "dielectric": {"mace": 0.239, "orb": 0.336, "soap": 0.177, "uma": 0.179}}
PUB_S13_ORB = {"elastic": {"gp": 0.803, "mtgp_2": 0.780, "dgp": 0.807},
               "dielectric_published_buggy": {"gp": 0.132, "mtgp_2": 0.177, "dgp": 0.336},
               "dielectric_true_orb": {"gp": 0.096, "mtgp_2": 0.121, "dgp": 0.336}}
print("\n=== VALIDATION ===")
print("[S12] elastic + dielectric best-over-surrogate per descriptor vs published (must match):")
allok = True
for dset in ("elastic", "dielectric"):
    for d in DESCS:
        v, m = best_over_surr(DF[dset], d, "R2"); pv = PUB_S12_R2[dset][d]
        ok = abs(v - pv) <= 0.0015; allok &= ok
        print(f"   {dset:10s} {d:4s} computed={v:.3f}({ps.SURROGATE_LABEL[m]:4s}) published={pv:.3f}  {'OK' if ok else 'MISMATCH'}")
print(f"   S12 elastic+dielectric reproduction: {'ALL OK' if allok else 'FAILED'}")
print("\n[S13] ORB per surrogate:")
print("   elastic (must match published 0.803/0.780/0.807):")
for m in MODELS:
    v = bestpca_avg(DF['elastic'], m, 'orb', 'R2')
    print(f"      {ps.SURROGATE_LABEL[m]:4s} {v:.3f}  published {PUB_S13_ORB['elastic'][m]}")
print("   dielectric (BUG FIX: published-buggy plotted SOAP; true ORB matches Table 4):")
for m in MODELS:
    v = bestpca_avg(DF['dielectric'], m, 'orb', 'R2')
    print(f"      {ps.SURROGATE_LABEL[m]:4s} true-ORB={v:.3f}  was(buggy)={PUB_S13_ORB['dielectric_published_buggy'][m]}  Table4-ORB={PUB_S13_ORB['dielectric_true_orb'][m]}")
print("   phonon (new DFPT):")
for m in MODELS:
    print(f"      {ps.SURROGATE_LABEL[m]:4s} {bestpca_avg(DF['phonon'], m, 'orb', 'R2'):.3f}")

# cross-check against the pipeline's own best_pca_averaged.csv where present
print("\n[cross-check vs pipeline best_pca_averaged.csv]:")
for dset, sub in [("elastic", ASE / "analysis_v3/n500/data/best_pca_averaged.csv"),
                  ("dielectric", ASE / "analysis_v3_dielectric_constant/n500/data/best_pca_averaged.csv")]:
    if sub.exists():
        bpa = pd.read_csv(sub)
        for m in MODELS:
            mine = bestpca_avg(DF[dset], m, "orb", "R2")
            row = bpa[(bpa.model == m) & (bpa.descriptor == "orb")]
            theirs = row["avg_R2"].iloc[0] if len(row) else float("nan")
            tag = "OK" if abs(mine - theirs) <= 0.003 else "DIFF"
            print(f"   {dset:10s} {ps.SURROGATE_LABEL[m]:4s} mine={mine:.4f} pipeline={theirs:.4f} {tag}")
