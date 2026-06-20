"""Comparison panel figure for A-PU vs CGNF synthesizability scores.

Produces a two-panel figure:
  Panel A – per-(backbone_name, policy) strip+box distributions of A-PU and
             CGNF scores, with a 0.5 reference line.
  Panel B – A-PU vs CGNF scatter (one point per structure), coloured by
             backbone; annotated with Spearman rho and fraction agreeing on
             the >0.5 synthesizability call; x=y and 0.5 guide lines drawn.

Usage
-----
CLI::

    python -m apu_synthesizability.panel \\
        --scored-csv results/scored.csv \\
        --out-png figures/panel.png

API::

    from apu_synthesizability.panel import make_panel
    make_panel("results/scored.csv", "figures/panel.png")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering; must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colorblind-friendly palette (Wong 2011) – one colour per backbone
_BACKBONE_COLORS: dict[str, str] = {
    "MatterGen":   "#E69F00",   # orange
    "CrystalFlow": "#56B4E9",   # sky blue
    "ADiT":        "#009E73",   # blue-green
}
_DEFAULT_COLOR = "#999999"  # grey for any unexpected backbone

# Policy marker styles
_POLICY_MARKERS: dict[str, str] = {
    "BASE": "o",
    "ACC":  "s",
}

_SCORE_TYPES = ("apu_score", "cgnf_score")
_SCORE_LABELS = {"apu_score": "A-PU", "cgnf_score": "CGNF"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str,
    title: str,
) -> None:
    """Draw per-(backbone_name, policy) strip + box on *ax* for one score column."""
    backbones = sorted(df["backbone_name"].unique())
    policies = sorted(df["policy"].unique())

    n_bb = len(backbones)
    n_pol = len(policies)
    group_width = 0.8
    slot_width = group_width / max(n_pol, 1)

    # Mapping backbone → x centre
    bb_x = {bb: i for i, bb in enumerate(backbones)}

    for pol_idx, policy in enumerate(policies):
        offset = (pol_idx - (n_pol - 1) / 2.0) * slot_width
        for bb in backbones:
            sub = df[(df["backbone_name"] == bb) & (df["policy"] == policy)][score_col]
            if sub.empty:
                continue
            x_centre = bb_x[bb] + offset
            values = sub.values

            # Box
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            iqr = q3 - q1
            whisker_lo = max(values.min(), q1 - 1.5 * iqr)
            whisker_hi = min(values.max(), q3 + 1.5 * iqr)
            color = _BACKBONE_COLORS.get(bb, _DEFAULT_COLOR)
            marker = _POLICY_MARKERS.get(policy, "^")
            lw = 1.2

            rect = mpatches.FancyBboxPatch(
                (x_centre - slot_width * 0.35, q1),
                slot_width * 0.7,
                q3 - q1,
                boxstyle="round,pad=0.01",
                linewidth=lw,
                edgecolor=color,
                facecolor=color,
                alpha=0.25,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.plot(
                [x_centre - slot_width * 0.35, x_centre + slot_width * 0.35],
                [med, med],
                color=color,
                linewidth=lw + 0.5,
                zorder=3,
            )
            ax.plot(
                [x_centre, x_centre],
                [whisker_lo, q1],
                color=color,
                linewidth=lw,
                linestyle="--",
                zorder=2,
            )
            ax.plot(
                [x_centre, x_centre],
                [q3, whisker_hi],
                color=color,
                linewidth=lw,
                linestyle="--",
                zorder=2,
            )

            # Jitter strip
            rng = np.random.default_rng(abs(hash(bb + policy)) % (2 ** 31))
            jitter = rng.uniform(-slot_width * 0.28, slot_width * 0.28, size=len(values))
            ax.scatter(
                x_centre + jitter,
                values,
                color=color,
                marker=marker,
                s=14,
                alpha=0.65,
                linewidths=0.3,
                edgecolors="white",
                zorder=4,
            )

    # Reference line at 0.5
    ax.axhline(0.5, color="0.5", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xticks(range(n_bb))
    ax.set_xticklabels(backbones, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Synthesizability score")
    ax.set_title(title)

    # Policy legend (marker shapes)
    pol_handles = [
        mpatches.Patch(facecolor="0.6", label=pol, linewidth=0)
        for pol in policies
    ]
    pol_marker_handles = [
        plt.Line2D(
            [0], [0],
            marker=_POLICY_MARKERS.get(pol, "^"),
            color="0.6",
            linestyle="none",
            markersize=5,
            label=pol,
        )
        for pol in policies
    ]
    ax.legend(
        handles=pol_marker_handles,
        title="Policy",
        title_fontsize=8,
        loc="upper right",
        framealpha=0.7,
    )


def _scatter_plot(
    ax: plt.Axes,
    df: pd.DataFrame,
) -> None:
    """Draw A-PU vs CGNF scatter on *ax*, coloured by backbone."""
    backbones = sorted(df["backbone_name"].unique())

    for bb in backbones:
        sub = df[df["backbone_name"] == bb]
        color = _BACKBONE_COLORS.get(bb, _DEFAULT_COLOR)
        ax.scatter(
            sub["apu_score"],
            sub["cgnf_score"],
            color=color,
            s=22,
            alpha=0.75,
            linewidths=0.3,
            edgecolors="white",
            label=bb,
            zorder=3,
        )

    # Guide lines
    ax.axhline(0.5, color="0.6", linewidth=0.7, linestyle=":", zorder=1)
    ax.axvline(0.5, color="0.6", linewidth=0.7, linestyle=":", zorder=1)
    ax.plot([0, 1], [0, 1], color="0.4", linewidth=0.8, linestyle="--", zorder=2)

    # Annotate
    rho, _ = spearmanr(df["apu_score"].values, df["cgnf_score"].values)
    agree = float(np.mean((df["apu_score"].values > 0.5) == (df["cgnf_score"].values > 0.5)))
    annotation = f"Spearman $\\rho$ = {rho:.2f}\nAgree (>0.5) = {agree:.0%}"
    ax.text(
        0.04, 0.96,
        annotation,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.85),
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("A-PU score")
    ax.set_ylabel("CGNF score")
    ax.set_title("A-PU vs CGNF (per structure)")
    ax.legend(
        title="Backbone",
        title_fontsize=8,
        loc="lower right",
        framealpha=0.7,
    )
    ax.set_aspect("equal", adjustable="box")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_panel(scored_csv: str, out_png: str) -> str:
    """Generate the comparison panel figure.

    Parameters
    ----------
    scored_csv:
        Path to the per-structure CSV produced by ``score_structures.main``.
        Required columns: ``backbone_name``, ``policy``, ``apu_score``,
        ``cgnf_score``.
    out_png:
        Output path for the PNG figure.

    Returns
    -------
    str
        The resolved *out_png* path (same value as input).
    """
    plt.rcParams.update(_RC)

    df = pd.read_csv(scored_csv)

    # Validate required columns
    required = {"backbone_name", "policy", "apu_score", "cgnf_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"scored_csv is missing columns: {missing}")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(13, 5))
    # Layout: [apu strip | cgnf strip | scatter]  width_ratios 2:2:3
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 3], wspace=0.38)

    ax_apu = fig.add_subplot(gs[0, 0])
    ax_cgnf = fig.add_subplot(gs[0, 1], sharey=ax_apu)
    ax_scatter = fig.add_subplot(gs[0, 2])

    _strip_plot(ax_apu, df, "apu_score", "(a) A-PU score")
    _strip_plot(ax_cgnf, df, "cgnf_score", "(b) CGNF score")
    _scatter_plot(ax_scatter, df)

    # Remove duplicate y-tick labels on shared axis
    plt.setp(ax_cgnf.get_yticklabels(), visible=False)
    ax_cgnf.set_ylabel("")

    fig.suptitle(
        "Synthesizability comparison: A-PU vs CGNF by backbone & policy",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig.tight_layout()

    out_png = str(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate the A-PU vs CGNF comparison panel figure."
    )
    p.add_argument(
        "--scored-csv",
        required=True,
        help="Path to the per-structure scored CSV (output of score_structures).",
    )
    p.add_argument(
        "--out-png",
        required=True,
        help="Output path for the panel PNG.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    out = make_panel(args.scored_csv, args.out_png)
    print(f"Panel saved to {out}")


if __name__ == "__main__":
    main()
