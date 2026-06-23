"""Refresh heatmap + PCA-sensitivity figures (Phase 1).

Outputs (filenames match originals so LaTeX includegraphics paths still work):
  figures/fig3_heatmap_R2_n500.pdf            (elastic)
  figures/fig_heatmap_dielectric_R2_n500.pdf
  figures/fig_heatmap_phonon_R2_n500.pdf
  figures/fig4_pca_sensitivity_n500.pdf       (elastic)
  figures/fig_pca_dielectric_R2_n500.pdf
  figures/fig_pca_phonon_R2_n500.pdf

Style template: plot_style.apply_style() — Helvetica, 11/10/13/9/8 pt sizes,
dashed light gridlines, bbox_inches="tight".

Per quality memo: archive originals to legacy/ before overwriting (handled by
plot_style.save_figure).
"""

from __future__ import annotations

from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    DESCRIPTOR_LABEL,
    SURROGATE_COLOURS,
    SURROGATE_LABEL,
    SZ_ANNOT,
    SZ_AXIS,
    SZ_LEGEND,
    SZ_SUBTITLE,
    SZ_SUPTITLE,
    SZ_TICK,
    SZ_VALUE_LABEL,
    apply_style,
    save_figure,
)


ANALYSIS_ROOT = _ROOT / "archive" / "ASE-native-surrogates"
DFPT = _ROOT / "01_static_benchmark" / "data" / "arm_b_pheasy_full"  # PHEASY-SI: big phonon dataset w/ DGP
FIG_DIR = Path(__file__).resolve().parent / "out_pheasy"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "elastic": {
        "data_dir": ANALYSIS_ROOT / "analysis_v3" / "n500" / "data",
        "title": "Elastic moduli",
        "heatmap_out": FIG_DIR / "fig3_heatmap_R2_n500.pdf",
        "pca_out": FIG_DIR / "fig4_pca_sensitivity_n500.pdf",
    },
    "dielectric": {
        "data_dir": ANALYSIS_ROOT / "analysis_v3_dielectric_constant" / "n500" / "data",
        "title": "Dielectric constants",
        "heatmap_out": FIG_DIR / "fig_heatmap_dielectric_R2_n500.pdf",
        "pca_out": FIG_DIR / "fig_pca_dielectric_R2_n500.pdf",
    },
    "phonon": {
        "data_dir": DFPT / "n500" / "data",
        "title": "Phonon thermodynamics — Pheasy (11,818)",
        "heatmap_out": FIG_DIR / "fig_heatmap_pheasy_R2_n500.pdf",
        "pca_out": FIG_DIR / "fig_pca_pheasy_R2_n500.pdf",
    },
}

DESCRIPTOR_ROW_ORDER = ["orb", "mace", "uma", "soap"]
SURROGATE_COL_ORDER = ["gp", "mtgp_2", "dgp"]


def plot_heatmap(best_df: pd.DataFrame, dataset_title: str, out_path: Path) -> None:
    pivot = best_df.pivot(index="descriptor", columns="model", values="avg_R2").loc[
        DESCRIPTOR_ROW_ORDER, SURROGATE_COL_ORDER
    ]
    pca_pivot = best_df.pivot(index="descriptor", columns="model", values="best_pca").loc[
        DESCRIPTOR_ROW_ORDER, SURROGATE_COL_ORDER
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(SURROGATE_COL_ORDER)))
    ax.set_xticklabels([SURROGATE_LABEL[m] for m in SURROGATE_COL_ORDER], fontsize=SZ_TICK)
    ax.set_yticks(range(len(DESCRIPTOR_ROW_ORDER)))
    ax.set_yticklabels([DESCRIPTOR_LABEL[d] for d in DESCRIPTOR_ROW_ORDER], fontsize=SZ_TICK)

    ax.set_xlabel("Surrogate", fontsize=SZ_AXIS)
    ax.set_ylabel("Featurizer", fontsize=SZ_AXIS)
    ax.set_title(
        f"Averaged R² at $\\mathit{{n}}_{{\\mathrm{{train}}}}=500$",
        fontsize=SZ_SUPTITLE,
        fontweight="bold",
        pad=8,
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            pca = int(pca_pivot.values[i, j])
            txt_color = "black" if 0.25 <= v <= 0.75 else "white"
            ax.text(
                j,
                i,
                f"{v:.3f}\nPCA={pca}",
                ha="center",
                va="center",
                fontsize=SZ_VALUE_LABEL,
                color=txt_color,
            )

    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Averaged R²", fontsize=SZ_AXIS)
    cbar.ax.tick_params(labelsize=SZ_TICK)
    cbar.outline.set_visible(False)

    fig.text(
        0.5,
        -0.02,
        f"{dataset_title} — best PCA per (featurizer × surrogate)",
        ha="center",
        fontsize=SZ_ANNOT,
        color="0.35",
    )

    fig.tight_layout()
    save_figure(fig, out_path)


def plot_pca_sensitivity(sens_df: pd.DataFrame, dataset_title: str, out_path: Path) -> None:
    descriptors = ["orb", "mace", "uma", "soap"]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4), sharey=True)

    pca_values = sorted(sens_df["pca"].unique())

    for idx, descriptor in enumerate(descriptors):
        ax = axes[idx]
        desc_data = sens_df[sens_df["descriptor"] == descriptor]
        for model in SURROGATE_COL_ORDER:
            md = desc_data[desc_data["model"] == model].sort_values("pca")
            ax.plot(
                md["pca"].values,
                md["avg_R2"].values,
                marker="o",
                markersize=5.5,
                linewidth=1.6,
                color=SURROGATE_COLOURS[model],
                label=SURROGATE_LABEL[model],
            )
        ax.set_title(DESCRIPTOR_LABEL[descriptor], fontsize=SZ_SUBTITLE, fontweight="bold")
        ax.set_xlabel("PCA components", fontsize=SZ_AXIS)
        if idx == 0:
            ax.set_ylabel("Averaged R²", fontsize=SZ_AXIS)
        ax.set_xticks(pca_values)
        ax.tick_params(axis="both", labelsize=SZ_TICK)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6, color="0.7")
        ax.set_axisbelow(True)
        if idx == len(descriptors) - 1:
            ax.legend(
                loc="best",
                fontsize=SZ_LEGEND,
                frameon=False,
                handlelength=1.4,
            )

    fig.suptitle(
        f"PCA sensitivity — {dataset_title} (averaged R², $\\mathit{{n}}_{{\\mathrm{{train}}}}=500$)",
        fontsize=SZ_SUPTITLE,
        fontweight="bold",
        y=1.02,
    )

    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    apply_style()
    for tag, cfg in [("phonon", DATASETS["phonon"])]:  # PHEASY-SI: only the big phonon dataset
        data_dir: Path = cfg["data_dir"]
        if not data_dir.exists():
            print(f"[skip] {tag}: missing data dir {data_dir}")
            continue

        best_df = pd.read_csv(data_dir / "best_pca_averaged.csv")
        plot_heatmap(best_df, cfg["title"], cfg["heatmap_out"])

        sens_df = pd.read_csv(data_dir / "pca_sensitivity_averaged.csv")
        plot_pca_sensitivity(sens_df, cfg["title"], cfg["pca_out"])


if __name__ == "__main__":
    main()
