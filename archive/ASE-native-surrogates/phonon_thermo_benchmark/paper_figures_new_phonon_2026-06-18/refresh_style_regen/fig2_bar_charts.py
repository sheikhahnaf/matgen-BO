"""Refresh Fig 2 (3 dataset bar charts) with paper-quality typography.

Replicates the panel structure of
ASE_regression_test/analysis_v3/n500/scripts/bar_charts_averaged.py but emits
the three dataset-specific filenames the LaTeX paper expects:

    figures/fig2_bar_elastic_R2_grouped.png
    figures/fig2_bar_dielectric_R2_grouped.png
    figures/fig2_bar_phonon_R2_grouped.png

Run from FME_paper_refresh_v1/figures_src/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_loaders import (
    DATASET_LABEL,
    best_pca_per_surrogate,
    load_filtered_n500,
)
from plot_style import (
    DESCRIPTOR_LABEL,
    DESCRIPTOR_ORDER,
    SURROGATE_COLOURS,
    SURROGATE_LABEL,
    SZ_ANNOT,
    SZ_AXIS,
    SZ_LEGEND,
    SZ_SUPTITLE,
    SZ_VALUE_LABEL,
    apply_style,
    save_figure,
)


FIGURES_DIR = Path(__file__).resolve().parent / "out"

DATASETS = [
    ("elastic", "fig2_bar_elastic_R2_grouped.png"),
    ("dielectric", "fig2_bar_dielectric_R2_grouped.png"),
    ("phonon", "fig2_bar_phonon_R2_grouped.png"),
]

MODEL_ORDER = ["gp", "mtgp_2", "dgp"]


def plot_one_dataset(dataset: str, out_filename: str) -> None:
    df = load_filtered_n500(dataset)
    best = best_pca_per_surrogate(df, optimize_for="R2")

    descriptors = [d for d in DESCRIPTOR_ORDER if d in best["descriptor"].unique()]
    models = [m for m in MODEL_ORDER if m in best["model"].unique()]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(descriptors))
    width = 0.26

    for i, model in enumerate(models):
        sub = best[best["model"] == model].set_index("descriptor")
        values = [sub.loc[d, "avg_R2"] if d in sub.index else np.nan for d in descriptors]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=SURROGATE_LABEL[model],
            color=SURROGATE_COLOURS[model],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            if val >= 0:
                y, va = bar.get_height() + 0.012, "bottom"
            else:
                y, va = bar.get_height() - 0.012, "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=SZ_VALUE_LABEL,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([DESCRIPTOR_LABEL[d] for d in descriptors])
    ax.set_xlabel("Featurizer", fontsize=SZ_AXIS)
    ax.set_ylabel(r"Averaged $R^{2}$", fontsize=SZ_AXIS)
    ax.set_ylim(_y_limits(best, descriptors, models))
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", ncol=len(models), fontsize=SZ_LEGEND, frameon=False)
    ax.set_title(
        rf"Averaged $R^{{2}}$ at $n_{{\mathrm{{train}}}}=500$",
        fontsize=SZ_SUPTITLE,
        fontweight="bold",
        pad=10,
    )

    pca_lines = _pca_caption_lines(best, descriptors, models)
    for i, line in enumerate(pca_lines):
        ax.text(
            0.5,
            -0.20 - i * 0.05,
            line,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=SZ_ANNOT,
            style="italic",
            color="0.35",
        )

    fig.tight_layout()
    save_figure(fig, FIGURES_DIR / out_filename)


def _y_limits(best, descriptors, models) -> tuple[float, float]:
    vals = []
    for model in models:
        sub = best[best["model"] == model].set_index("descriptor")
        for d in descriptors:
            if d in sub.index:
                vals.append(sub.loc[d, "avg_R2"])
    lo = min(min(vals), 0.0) - 0.05
    hi = max(max(vals), 0.0) + 0.10
    if lo > -0.05:
        lo = 0.0
    return (lo, min(hi, 1.05))


def _pca_caption_lines(best, descriptors, models) -> list[str]:
    parts = []
    for d in descriptors:
        sub = best[best["descriptor"] == d].set_index("model")
        chunks = [
            f"{SURROGATE_LABEL[m]}:{int(sub.loc[m, 'best_pca'])}"
            for m in models
            if m in sub.index
        ]
        parts.append(f"{DESCRIPTOR_LABEL[d]} ({', '.join(chunks)})")
    half = (len(parts) + 1) // 2
    return [
        "Best PCA per surrogate:  " + "   ".join(parts[:half]),
        "  " + "   ".join(parts[half:]),
    ]


def main() -> None:
    apply_style()
    for dataset, fname in DATASETS:
        plot_one_dataset(dataset, fname)


if __name__ == "__main__":
    main()
