"""Per-paradigm bar plots for the closed-loop probe (Section 5.8).

Reuses the data layer already in matinvent-hcap-bo/analysis/top_structures/top_per_job.csv;
restyles the visualisations to match the FME paper's plot_style (Helvetica, etc.).

Produces:
  fig_bogen_spacegroup_dist_cp.png   — paradigm panel + BASE-vs-ACC panel (Cp)
  fig_bogen_spacegroup_dist_bm.png   — same for K_VRH
  fig_bogen_per_model_summary.png    — best/mean top-3/novelty per backbone, both targets
"""
from __future__ import annotations

from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_figure

apply_style()

ROOT = _ROOT / "archive" / "matinvent-hcap-bo" / "hcap_bo" / "analysis" / "top_structures"
OUT = _ROOT / "01_static_benchmark" / "figures" / "regenerated"
OUT.mkdir(parents=True, exist_ok=True)

PARADIGMS = ["mg", "cf", "adit"]
PARADIGM_LABEL = {"mg": "MatterGen", "cf": "CrystalFlow", "adit": "ADiT"}
PARADIGM_COLOR = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}

SETUPS = ["BASE", "ACC"]
SETUP_COLOR = {"BASE": "#888888", "ACC": "#cc4400"}

CRYSTAL_SYS = [
    ("Triclinic",   1,   2),
    ("Monoclinic",  3,  15),
    ("Orthorhombic",16,  74),
    ("Tetragonal", 75, 142),
    ("Trigonal",  143, 167),
    ("Hexagonal", 168, 194),
    ("Cubic",     195, 230),
]
SYS_ORDER = [name for name, _, _ in CRYSTAL_SYS]


def crystal_system(sg: int) -> str:
    for name, lo, hi in CRYSTAL_SYS:
        if lo <= sg <= hi:
            return name
    return "Unknown"


def load(prop: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / prop / "top_per_job.csv")
    df["crystal_system"] = df["sg_number"].apply(crystal_system)
    return df


def fig_spacegroup_dist(df: pd.DataFrame, prop_math: str, n_total: int, fname: Path) -> None:
    """Match the layout of top_structures/figures/spacegroup_dist_<prop>.png:
    1x2 horizontal bars, left = by paradigm, right = by setup (BASE vs ACC).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    y = np.arange(len(SYS_ORDER))
    height = 0.27

    # Compute per-panel xmax to leave room for legends inside the plot area
    paradigm_max = max(
        (df[df.paradigm == p].groupby("crystal_system").size().reindex(SYS_ORDER, fill_value=0).max())
        for p in PARADIGMS
    )
    setup_max = max(
        (df[df.setup == s].groupby("crystal_system").size().reindex(SYS_ORDER, fill_value=0).max())
        for s in SETUPS
    )

    # Left panel — by paradigm
    ax = axes[0]
    for i, p in enumerate(PARADIGMS):
        counts = (df[df.paradigm == p].groupby("crystal_system").size()
                                       .reindex(SYS_ORDER, fill_value=0).values)
        ax.barh(y + (i - 1) * height, counts, height,
                label=PARADIGM_LABEL[p], color=PARADIGM_COLOR[p],
                edgecolor="white", linewidth=0.6, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(SYS_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, paradigm_max * 1.30)  # extra headroom for legend
    ax.set_xlabel("# top-3 structures")
    ax.set_title("By backbone")
    ax.legend(loc="upper right", title="Backbone", title_fontsize=9,
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    ax.grid(True, axis="x", ls="--", lw=0.5, color="lightgray", alpha=0.6)
    ax.set_axisbelow(True)

    # Right panel — by setup (BASE vs ACC)
    ax = axes[1]
    height2 = 0.38
    for i, setup in enumerate(SETUPS):
        counts = (df[df.setup == setup].groupby("crystal_system").size()
                                        .reindex(SYS_ORDER, fill_value=0).values)
        ax.barh(y + (i - 0.5) * height2, counts, height2,
                label=setup, color=SETUP_COLOR[setup],
                edgecolor="white", linewidth=0.6, alpha=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(SYS_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, setup_max * 1.30)
    ax.set_xlabel("# top-3 structures")
    ax.set_title("By policy")
    ax.legend(loc="upper right", title="Policy", title_fontsize=9,
              frameon=True, framealpha=0.92, edgecolor="lightgray")
    ax.grid(True, axis="x", ls="--", lw=0.5, color="lightgray", alpha=0.6)
    ax.set_axisbelow(True)

    n_runs = n_total // 3 if n_total else 0
    fig.suptitle(f"Crystal-system distribution among top-3 generated {prop_math} structures "
                 f"({n_total} structures across {n_runs} runs)",
                 y=1.02, fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, fname)


def fig_per_model_summary(df_cp: pd.DataFrame, df_bm: pd.DataFrame) -> None:
    """2x2: best value (BASE vs ACC) + mean top-3 (±std) per backbone, for both targets.

    Novelty rates dropped — they are 100% (Cp) and 97% (K_VRH) across all backbones,
    so a uniform bar plot was uninformative; numbers are quoted verbally in §5.8.5/§5.8.6.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    paradigm_labels = [PARADIGM_LABEL[p] for p in PARADIGMS]
    paradigm_colors = [PARADIGM_COLOR[p] for p in PARADIGMS]
    x = np.arange(len(PARADIGMS))
    width = 0.36

    setup_alpha = {"BASE": 0.55, "ACC": 0.95}
    setup_hatch = {"BASE": "//", "ACC": ""}

    for row, (df, prop_math, prop_unit) in enumerate([
        (df_cp, r"$C_{p}$", "J/g/K"),
        (df_bm, r"$K_{\mathrm{VRH}}$", "GPa"),
    ]):
        # Column 1 — best value per (paradigm, setup)
        ax = axes[row, 0]
        for i, setup in enumerate(SETUPS):
            best = []
            for p in PARADIGMS:
                sub = df[(df.paradigm == p) & (df.setup == setup)]
                best.append(sub["value"].max() if not sub.empty else np.nan)
            offset = (i - 0.5) * width
            ax.bar(x + offset, best, width, label=setup,
                   color=paradigm_colors,
                   alpha=setup_alpha[setup], hatch=setup_hatch[setup],
                   edgecolor="white", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(paradigm_labels)
        ax.set_ylabel(f"Best {prop_math} ({prop_unit})")
        ax.set_title(f"Best {prop_math}", pad=6)
        ax.grid(axis="y", ls="--", lw=0.5, color="lightgray", alpha=0.6)
        ax.set_axisbelow(True)
        if row == 0:
            ax.legend(title="Policy", title_fontsize=9, loc="upper right")

        # Column 2 — mean top-3 value per paradigm
        ax = axes[row, 1]
        means = [df[df.paradigm == p]["value"].mean() for p in PARADIGMS]
        stds  = [df[df.paradigm == p]["value"].std()  for p in PARADIGMS]
        ax.bar(x, means, yerr=stds, capsize=4, color=paradigm_colors,
               alpha=0.9, edgecolor="white", linewidth=0.8,
               error_kw={"elinewidth": 0.9})
        ax.set_xticks(x); ax.set_xticklabels(paradigm_labels)
        ax.set_ylabel(f"Mean top-3 {prop_math} ({prop_unit})")
        ax.set_title(f"Mean top-3 {prop_math} ($\\pm$ std)", pad=6)
        ax.grid(axis="y", ls="--", lw=0.5, color="lightgray", alpha=0.6)
        ax.set_axisbelow(True)

    n_cp = df_cp["jobid"].nunique() if "jobid" in df_cp.columns else len(df_cp) // 3
    n_bm = df_bm["jobid"].nunique() if "jobid" in df_bm.columns else len(df_bm) // 3
    fig.suptitle(f"Per-backbone summary across the {n_cp}+{n_bm} closed-loop runs ($C_p$+$K_{{\\mathrm{{VRH}}}}$)",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUT / "fig_bogen_per_model_summary.png")


def main() -> None:
    df_cp = load("cp")
    df_bm = load("bm")
    fig_spacegroup_dist(df_cp, r"$C_{p}$", len(df_cp),
                        OUT / "fig_bogen_spacegroup_dist_cp.png")
    fig_spacegroup_dist(df_bm, r"$K_{\mathrm{VRH}}$", len(df_bm),
                        OUT / "fig_bogen_spacegroup_dist_bm.png")
    fig_per_model_summary(df_cp, df_bm)
    print("\n=== Per-backbone summary numbers (also quoted in §5.8.5 prose) ===")
    for prop_math, df in [("Cp", df_cp), ("K_VRH", df_bm)]:
        print(f"\n[{prop_math}]")
        for p in PARADIGMS:
            sub = df[df.paradigm == p]
            best = sub["value"].max() if not sub.empty else float("nan")
            mean = sub["value"].mean() if not sub.empty else float("nan")
            std  = sub["value"].std()  if not sub.empty else float("nan")
            novelty = (1 - sub["in_seed_pool"].mean()) * 100 if not sub.empty else float("nan")
            print(f"  {PARADIGM_LABEL[p]:<11}  best={best:7.3f}   mean±std={mean:6.3f} ± {std:5.3f}   novelty={novelty:5.1f}%")


if __name__ == "__main__":
    main()
