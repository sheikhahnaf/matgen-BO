"""Refresh property-difficulty heatmaps + radar charts to global plot quality standards.

Sources (CSV from each dataset's analysis_v3* dir):
  property_difficulty_per_surrogate.csv  -> heatmap (3 surrogates x 4 descriptors x N props)
  radar_orb_pca_choices.csv              -> radar chart (ORB only, 3 surrogates x N props, R2 + best PCA)

Outputs (figures/<exact original filename>):
  fig_difficulty_dielectric_n500.pdf  (4 props)
  fig_difficulty_phonon_n500.pdf      (3 props)
  fig6_property_difficulty.pdf        (8 elastic props)
  fig_radar_dielectric_orb_R2_n500.pdf
  fig_radar_phonon_orb_R2_n500.pdf
  fig7_radar_orb_R2.pdf
"""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from plot_style import (
    apply_style, save_figure,
    SURROGATE_COLOURS, SURROGATE_LABEL,
    DESCRIPTOR_ORDER, DESCRIPTOR_LABEL,
    SZ_AXIS, SZ_TICK, SZ_SUPTITLE, SZ_SUBTITLE, SZ_LEGEND, SZ_ANNOT, SZ_VALUE_LABEL,
)

ANALYSIS = _ROOT / "archive" / "ASE-native-surrogates"
FIG_OUT = _ROOT / "01_static_benchmark" / "figures" / "regenerated"

DATASETS = {
    "dielectric": {
        "data_dir": ANALYSIS / "analysis_v3_dielectric_constant" / "n500" / "data",
        "props": ["band_gap", "n", "poly_electronic", "poly_total"],
        "prop_label": {"band_gap": "Band gap", "n": "n",
                       "poly_electronic": r"$\varepsilon_\infty$",
                       "poly_total": r"$\varepsilon_0$"},
        "diff_out": FIG_OUT / "fig_difficulty_dielectric_n500.pdf",
        "radar_out": FIG_OUT / "fig_radar_dielectric_orb_R2_n500.pdf",
        "label": "Dielectric",
    },
    "phonon": {
        "data_dir": ANALYSIS / "analysis_v3_phonon_dielectric_mp" / "n500" / "data",
        "props": ["eps_electronic", "eps_total", "last phdos peak"],
        "prop_label": {"eps_electronic": r"$\varepsilon_\infty$",
                       "eps_total": r"$\varepsilon_0$",
                       "last phdos peak": "PhDOS\npeak"},
        "diff_out": FIG_OUT / "fig_difficulty_phonon_n500.pdf",
        "radar_out": FIG_OUT / "fig_radar_phonon_orb_R2_n500.pdf",
        "label": "Phonon",
    },
    "elastic": {
        "data_dir": ANALYSIS / "analysis_v3" / "n500" / "data",
        "props": ["K_VRH", "K_Voigt", "K_Reuss", "G_VRH", "G_Voigt", "G_Reuss",
                  "poisson_ratio", "elastic_anisotropy"],
        "prop_label": {
            "K_VRH": "K$_{VRH}$", "K_Voigt": "K$_{Voigt}$", "K_Reuss": "K$_{Reuss}$",
            "G_VRH": "G$_{VRH}$", "G_Voigt": "G$_{Voigt}$", "G_Reuss": "G$_{Reuss}$",
            "poisson_ratio": "ν", "elastic_anisotropy": "A$_U$",
        },
        "diff_out": FIG_OUT / "fig6_property_difficulty.pdf",
        "radar_out": FIG_OUT / "fig7_radar_orb_R2.pdf",
        "label": "Elastic",
    },
}

MODEL_ORDER = ["gp", "mtgp_2", "dgp"]


def plot_difficulty(cfg: dict) -> None:
    csv = cfg["data_dir"] / "property_difficulty_per_surrogate.csv"
    df = pd.read_csv(csv)
    props = cfg["props"]
    desc_order = DESCRIPTOR_ORDER  # mace, orb, soap, uma

    n_panels = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels + 0.6, max(2.8, 0.55 * len(props) + 1.4)))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.get_cmap("RdYlGn")
    vmin, vmax = 0.0, 1.0

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        sub = df[df["model"] == model]
        # pivot: rows=property, cols=descriptor
        mat = (sub.pivot(index="property", columns="descriptor", values="R2")
                  .reindex(index=props, columns=desc_order))
        arr = mat.values

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        # cells: numeric annotations
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=SZ_VALUE_LABEL, color="0.4")
                    continue
                txt_color = "white" if (v < 0.30 or v > 0.78) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=SZ_VALUE_LABEL, color=txt_color)

        ax.set_xticks(range(len(desc_order)))
        ax.set_xticklabels([DESCRIPTOR_LABEL[d] for d in desc_order], fontsize=SZ_TICK)
        ax.set_yticks(range(len(props)))
        if idx == 0:
            ax.set_yticklabels([cfg["prop_label"][p] for p in props], fontsize=SZ_TICK)
            ax.set_ylabel("Property", fontsize=SZ_AXIS)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Featurizer", fontsize=SZ_AXIS)
        ax.set_title(SURROGATE_LABEL[model], fontsize=SZ_SUBTITLE, fontweight="bold")
        # clean spines: imshow -> all 4 spines visible by default
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle(f"{cfg['label']} — R² heatmap (n$_{{train}}$ = 500, best PCA per surrogate)",
                 fontsize=SZ_SUPTITLE, fontweight="bold", y=1.02)

    cbar = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02, shrink=0.85)
    cbar.set_label("R²", fontsize=SZ_AXIS)
    cbar.ax.tick_params(labelsize=SZ_TICK)

    save_figure(fig, cfg["diff_out"])


def plot_radar(cfg: dict) -> None:
    csv = cfg["data_dir"] / "radar_orb_pca_choices.csv"
    df = pd.read_csv(csv)
    props = cfg["props"]
    angles = np.linspace(0, 2 * np.pi, len(props), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, polar=True)

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].set_index("property").reindex(props)
        if sub["R2"].isna().all():
            continue
        vals = sub["R2"].fillna(0.0).tolist()
        vals_closed = vals + vals[:1]
        color = SURROGATE_COLOURS[model]
        ax.plot(angles_closed, vals_closed, "o-", lw=2.0, ms=6,
                color=color, label=SURROGATE_LABEL[model])
        ax.fill(angles_closed, vals_closed, alpha=0.12, color=color)

    # category labels — adjust radial distance to avoid overlap with axis ticks
    ax.set_xticks(angles)
    ax.set_xticklabels([cfg["prop_label"][p] for p in props], fontsize=SZ_TICK)
    # tick labels at 0.25, 0.50, 0.75, 1.00
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="0.3")
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(135)

    ax.grid(True, lw=0.6, alpha=0.45, color="0.65")
    ax.spines["polar"].set_color("0.5")
    ax.spines["polar"].set_linewidth(0.8)

    ax.set_title(f"{cfg['label']} — ORB · R² (n$_{{train}}$ = 500, best PCA per property)",
                 fontsize=SZ_SUBTITLE, fontweight="bold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.08), fontsize=SZ_LEGEND, frameon=False)

    # Slight padding so labels don't get clipped
    plt.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.08)

    save_figure(fig, cfg["radar_out"])


def main() -> None:
    apply_style()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    for key, cfg in DATASETS.items():
        print(f"[{key}] difficulty heatmap …")
        plot_difficulty(cfg)
        print(f"[{key}] radar chart …")
        plot_radar(cfg)
    print("done.")


if __name__ == "__main__":
    main()
