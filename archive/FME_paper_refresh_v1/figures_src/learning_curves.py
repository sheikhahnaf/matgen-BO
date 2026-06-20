"""Refresh learning-curve figures for the FME paper (Phase 1, figures only).

Outputs (legacy → figures/legacy/, new → figures/<same name>):
  fig_combined_dielectric_R2_learning_curve.png       (dielectric_constant data)
  fig_combined_dielectric_RMSE_learning_curve.png
  fig_combined_dielectric_Spearman_learning_curve.png
  fig_combined_phonon_R2_learning_curve.png           (phonon_dielectric_mp data)
  fig_combined_phonon_RMSE_learning_curve.png
  fig_combined_phonon_Spearman_learning_curve.png
  fig5_learning_curve_R2.png                          (analysis_v3/combined elastic data)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    SURROGATE_COLOURS,
    SURROGATE_LABEL,
    SZ_AXIS,
    SZ_LEGEND,
    SZ_SUPTITLE,
    apply_style,
    save_figure,
)


REPO_ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-bo")
FIG_DIR = REPO_ROOT / "FME_paper_refresh_v1" / "figures"

DATASETS = {
    "dielectric": REPO_ROOT
    / "ASE_regression_test"
    / "combined_dielectric_constant"
    / "data"
    / "learning_curves_orb.csv",
    "phonon": REPO_ROOT
    / "ASE_regression_test"
    / "combined_phonon_dielectric_mp"
    / "data"
    / "learning_curves_orb.csv",
    "elastic": REPO_ROOT
    / "ASE_regression_test"
    / "analysis_v3"
    / "combined"
    / "data"
    / "learning_curves_orb.csv",
}

DATASET_LABEL = {
    "dielectric": "Dielectric properties",
    "phonon": "Phonon / dielectric (MP)",
    "elastic": "Elastic properties",
}

METRIC_FORMAT = {
    "R2": {
        "ylabel": r"Mean R$^{2}$ across properties",
        "title_metric": r"R$^{2}$",
        "ylim_low_clip": 0.0,
    },
    "RMSE": {
        "ylabel": "Mean RMSE across properties",
        "title_metric": "RMSE",
        "ylim_low_clip": None,
    },
    "Spearman": {
        "ylabel": "Mean Spearman across properties",
        "title_metric": "Spearman",
        "ylim_low_clip": 0.0,
    },
}

MODELS = ["gp", "mtgp_2", "dgp"]
JITTER = {"gp": -8.0, "mtgp_2": 0.0, "dgp": 8.0}  # x-offset to separate error bars


def _aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "n_train"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["model", "n_train"])
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _plot_single(df: pd.DataFrame, metric: str, dataset_key: str, out_name: str) -> Path:
    fmt = METRIC_FORMAT[metric]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    grouped = _aggregate(df, metric)

    n_props = int(df.groupby("model")["property"].nunique().max())

    for model in MODELS:
        sub = grouped[grouped["model"] == model]
        if sub.empty:
            continue
        x = sub["n_train"].to_numpy(dtype=float) + JITTER[model]
        y = sub["mean"].to_numpy(dtype=float)
        yerr = sub["std"].to_numpy(dtype=float)

        colour = SURROGATE_COLOURS[model]
        label = SURROGATE_LABEL[model]

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=colour,
            label=label,
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linewidth=1.6,
            elinewidth=1.0,
            capsize=3.5,
            capthick=1.0,
            zorder=3,
        )

    ax.set_xlabel("Training set size  $n_{\\mathrm{train}}$", fontsize=SZ_AXIS)
    ax.set_ylabel(fmt["ylabel"], fontsize=SZ_AXIS)

    ax.set_xticks([100, 250, 500])
    ax.set_xticklabels(["100", "250", "500"])

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    if fmt["ylim_low_clip"] is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(bottom=max(fmt["ylim_low_clip"], cur_lo - 0.05), top=cur_hi)

    title = f"{DATASET_LABEL[dataset_key]} – {fmt['title_metric']} vs. $n_{{\\mathrm{{train}}}}$"
    subtitle = f"ORB descriptor; mean ± std across {n_props} properties"

    leg = ax.legend(
        loc="best",
        fontsize=SZ_LEGEND,
        ncol=3,
        handlelength=1.4,
        columnspacing=1.4,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.85",
        facecolor="white",
    )
    if leg is not None:
        leg.set_zorder(5)
        leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.text(0.5, 0.96, title, ha="center", va="top",
             fontsize=SZ_SUPTITLE, fontweight="bold")
    fig.text(0.5, 0.905, subtitle, ha="center", va="top",
             fontsize=8.5, color="0.45")
    out_path = FIG_DIR / out_name
    save_figure(fig, out_path)
    return out_path


def main() -> None:
    apply_style()

    plan: list[tuple[str, str, str]] = [
        ("dielectric", "R2", "fig_combined_dielectric_R2_learning_curve.png"),
        ("dielectric", "RMSE", "fig_combined_dielectric_RMSE_learning_curve.png"),
        ("dielectric", "Spearman", "fig_combined_dielectric_Spearman_learning_curve.png"),
        ("phonon", "R2", "fig_combined_phonon_R2_learning_curve.png"),
        ("phonon", "RMSE", "fig_combined_phonon_RMSE_learning_curve.png"),
        ("phonon", "Spearman", "fig_combined_phonon_Spearman_learning_curve.png"),
        ("elastic", "R2", "fig5_learning_curve_R2.png"),
    ]

    cached: dict[str, pd.DataFrame] = {}
    for dataset_key, metric, out_name in plan:
        if dataset_key not in cached:
            csv = DATASETS[dataset_key]
            cached[dataset_key] = pd.read_csv(csv)
            print(f"loaded {csv.name} ({cached[dataset_key].shape})")
        _plot_single(cached[dataset_key], metric, dataset_key, out_name)


if __name__ == "__main__":
    main()
