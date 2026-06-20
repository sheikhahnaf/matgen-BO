"""Figure: R² vs Spearman ρ disagreement across properties.

Visualises why R² and Spearman disagree on outlier-heavy properties.
Each point is one (dataset, property) combo at ORB+GP, PCA=50, n_train=500
(the same config as the paper's parity figure).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style, save_figure

apply_style()

import os as _os
_REPO = Path(_os.environ.get("ASE_REPO_ROOT", Path(__file__).resolve().parent.parent))
ROOT = _REPO / "results" / "gp"
OUT = Path(_os.environ.get("ASE_FIG_DIR", _REPO / "figures")) / "fig_metric_disagreement.png"

DATASETS = {
    "elastic_tensor_2015": ("Elastic",  "#1f77b4"),
    "dielectric_constant": ("Dielectric", "#ff7f0e"),
    "phonon_dielectric_mp": ("Phonon", "#2ca02c"),
}

PROP_LABEL = {
    "K_VRH":   r"$K_{\mathrm{VRH}}$",
    "K_Voigt": r"$K_{\mathrm{Voigt}}$",
    "K_Reuss": r"$K_{\mathrm{Reuss}}$",
    "G_VRH":   r"$G_{\mathrm{VRH}}$",
    "G_Voigt": r"$G_{\mathrm{Voigt}}$",
    "G_Reuss": r"$G_{\mathrm{Reuss}}$",
    "poisson_ratio": r"$\nu$",
    "elastic_anisotropy": r"$A_{U}$",
    "kpoint_density": "k-pt density",
    "band_gap": "band gap",
    "n": "n",
    "poly_electronic": r"$\epsilon^{\infty}_{\mathrm{poly}}$",
    "poly_total":      r"$\epsilon^{0}_{\mathrm{poly}}$",
    "eps_electronic":  r"$\epsilon^{\infty}$",
    "eps_total":       r"$\epsilon^{0}$",
    "last phdos peak": "phdos peak",
}


def load() -> pd.DataFrame:
    rows = []
    for ds_dir, (label, _) in DATASETS.items():
        df = pd.read_csv(ROOT / f"{ds_dir}_pca50_n500" / "gp_orb_holdout_summary.csv")
        # kpoint_density is a calculation setting, not one of the 8 elastic target
        # properties counted in the paper (and in the 8,100-fit factorial); exclude it.
        df = df[df["Property"] != "kpoint_density"]
        for _, r in df.iterrows():
            rows.append({
                "dataset": label,
                "property": r["Property"],
                "R2": r["R2_mean"],
                "rho": r["Spearman_mean"],
                "R2_std": r["R2_std"],
                "rho_std": r["Spearman_std"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    fig, ax = plt.subplots(figsize=(7.4, 6.2))

    # Reference: where R² and ρ agree
    ax.plot([-1.05, 1.0], [-1.05, 1.0], "--", color="gray", alpha=0.55,
            lw=1.0, label=r"$\rho = R^{2}$", zorder=1)

    # R² = 0 line (predicting the mean)
    ax.axvline(0, color="black", alpha=0.25, lw=0.9, ls=":", zorder=1)

    # Highlight the metric-mismatch zone (low R², high ρ)
    ax.axvspan(-1.05, 0.2, ymin=0.6, ymax=1.0, color="#d62728", alpha=0.06, zorder=0)
    ax.text(-1.0, 0.96, "high ρ, low R²\n(outlier-dominated)",
            fontsize=8, color="#7a1f1f", style="italic", ha="left", va="top")

    # Scatter per dataset
    for ds, (label, c) in DATASETS.items():
        sub = df[df["dataset"] == label]
        ax.errorbar(sub["R2"], sub["rho"],
                    xerr=sub["R2_std"], yerr=sub["rho_std"],
                    fmt="o", color=c, ecolor=c,
                    elinewidth=0.8, capsize=2, capthick=0.7,
                    markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                    label=label, alpha=0.9, zorder=3)

    # Annotate: extreme outliers (low R² or large gap)
    extremes = df[(df["R2"] < 0.3) | (df["rho"] - df["R2"] > 0.4)]
    for _, r in extremes.iterrows():
        prop = PROP_LABEL.get(r["property"], r["property"])
        ax.annotate(prop, (r["R2"], r["rho"]),
                    fontsize=8.5, ha="left", va="center",
                    xytext=(8, 0), textcoords="offset points",
                    color="#3a3a3a")

    ax.set_xlabel(r"Mean $R^{2}$ (5-split CV)")
    ax.set_ylabel(r"Mean Spearman $\rho$ (5-split CV)")
    ax.set_xlim(-1.05, 1.0)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, ls="--", lw=0.5, color="lightgray", alpha=0.7, zorder=0)

    ax.set_title("Ranking quality vs. absolute prediction error", pad=14, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.91,
             r"ORB·GP, PCA = 50, $n_{\mathrm{train}}$ = 500   |   one point per (dataset, property)",
             ha="center", va="bottom", fontsize=9.5, color="#5a5a5a")

    ax.legend(loc="lower right", title="Dataset", title_fontsize=9.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, OUT)
    print("\nExtremes annotated:")
    print(extremes[["dataset", "property", "R2", "rho"]].to_string(index=False))


if __name__ == "__main__":
    main()
