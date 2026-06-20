"""Discovery curves and surrogate-quality figure for the closed-loop section (§5.8).

Restyles the existing v4_metrics_plots/{cp,bm}/{best_running,gp_fit_quality}.png to use
the FME paper's plot_style (Helvetica, coordinated sizes, no-spine layout) and consolidates
both properties into single multi-panel figures.

Produces:
  fig_bogen_curves.png             — 2x3 grid: running-best Cp / K_VRH × 3 backbones
  fig_bogen_surrogate_quality.png  — 1x2 grid: GP CV5 RMSE per cycle for Cp / K_VRH
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "archive").is_dir())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_figure

apply_style()

ROOT = _ROOT / "archive" / "matinvent-hcap-bo"
OUT = _ROOT / "02_closed_loop_bo" / "figures" / "regenerated"
OUT.mkdir(parents=True, exist_ok=True)

PARADIGMS = ["mg", "cf", "adit"]
PARADIGM_LABEL = {"mg": "MatterGen", "cf": "CrystalFlow", "adit": "ADiT"}
PARADIGM_COLOR = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}
SETUP_LS = {"BASE": "--", "ACC": "-"}
SETUP_MARKER = {"BASE": "s", "ACC": "o"}


def discover(prop: str) -> pd.DataFrame:
    if prop == "Cp":
        roots = ROOT / "results"
        pat = re.compile(r"hcap_p3v4_(mg|cf|adit)_(baseline|accel)_seed(\d+)_(\d+)$")
        gp_subdir = ("rewards", "heat_capacity", "gp_routed_v4_log.csv")
    else:
        roots = ROOT / "results_bm"
        pat = re.compile(r"bm_p3v4_bm_(mg|cf|adit)_(baseline|accel)_seed(\d+)_(\d+)$")
        gp_subdir = ("rewards", "bulk_modulus", "bm_gp_routed_v4_log.csv")

    rows = []
    for d in sorted(roots.glob("*")):
        m = pat.match(d.name)
        if not m:
            continue
        paradigm, raw_setup, seed, jobid = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        setup = "BASE" if raw_setup == "baseline" else "ACC"
        rows.append({
            "paradigm": paradigm, "setup": setup, "seed": seed, "jobid": jobid,
            "dir": d, "metrics": d / "metrics.csv", "gp_log": d / Path(*gp_subdir),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("jobid").drop_duplicates(
        subset=["paradigm", "setup", "seed"], keep="last")


def best_running(jobs: pd.DataFrame, prop_label: str, minv: float, maxv: float,
                 max_cycles: int = 20) -> pd.DataFrame:
    """Per-seed running-best property value per cycle, forward-filled across cycles 0..max_cycles-1.

    Forward-fill is necessary because seeds finish different numbers of cycles and the
    `reward > 0` filter drops cycles where everything was NaN-gated. Without ffill, the
    cross-seed mean fluctuates as the membership of contributing seeds shifts per cycle.
    With ffill, each seed contributes its last-known running-best at every cycle, so the
    cross-seed mean is itself monotonically non-decreasing.
    """
    out = []
    for _, r in jobs.iterrows():
        ltm = r["dir"] / "samples" / "long_term_memory.csv"
        if not ltm.exists():
            continue
        try:
            df = pd.read_csv(ltm, usecols=["reward", "RL_step"])
        except Exception:
            continue
        df = df[df.reward > 0].copy()
        if df.empty:
            continue
        df["prop"] = df["reward"] * (maxv - minv) + minv
        df.loc[df.reward >= 1.0, "prop"] = maxv
        if prop_label == "Cp":
            df["score"] = -np.abs(df["prop"] - 1.5)
            per_cycle_max = df.groupby("RL_step")["score"].max()
            running_best = per_cycle_max.expanding().max()
            best_prop = pd.Series(1.5 - np.abs(running_best.values), index=running_best.index)
        else:
            per_cycle_max = df.groupby("RL_step")["prop"].max()
            running_best = per_cycle_max.expanding().max()
            best_prop = pd.Series(running_best.values, index=running_best.index)
        # Forward-fill across the full 0..max_cycles-1 range so dead/skipped cycles
        # carry the last known running-best instead of dropping out of the mean.
        full = best_prop.reindex(range(max_cycles)).ffill()
        for c, p in full.items():
            if pd.isna(p):
                continue  # leading cycles before this seed produced any positive reward
            out.append({"paradigm": r.paradigm, "setup": r.setup, "seed": r.seed,
                        "cycle": int(c), "best": float(p)})
    return pd.DataFrame(out)


def load_gp_metric(jobs: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for _, r in jobs.iterrows():
        if r.setup != "ACC" or not r.gp_log.exists():
            continue
        try:
            df = pd.read_csv(r.gp_log)
        except Exception:
            continue
        if col not in df.columns:
            continue
        for cyc, val in zip(df["cycle"], df[col]):
            out.append({"paradigm": r.paradigm, "seed": r.seed,
                        "cycle": int(cyc), "value": float(val)})
    return pd.DataFrame(out)


def fig_curves(df_cp: pd.DataFrame, df_bm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
    rows = [
        (df_cp, axes[0], r"running best $C_{p}$ (J/g/K)", 1.5, "target = 1.5", (0.2, 1.6)),
        (df_bm, axes[1], r"running best $K_{\mathrm{VRH}}$ (GPa)", None, None, None),
    ]
    for df, axrow, ylabel, target, target_lbl, ylim in rows:
        for ax, p in zip(axrow, PARADIGMS):
            sub = df[df.paradigm == p]
            for setup in ("BASE", "ACC"):
                s = sub[sub.setup == setup]
                if s.empty:
                    continue
                agg = s.groupby("cycle")["best"].agg(["mean", "std"])
                x = agg.index.values
                y = agg["mean"].values
                yerr = agg["std"].fillna(0).values
                color = PARADIGM_COLOR[p]
                ax.plot(x, y, ls=SETUP_LS[setup], color=color, lw=1.6,
                        marker=SETUP_MARKER[setup], ms=3.5, label=setup)
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12, lw=0)
            if target is not None:
                ax.axhline(target, color="gray", ls=":", lw=0.9, alpha=0.7,
                           label=target_lbl)
            ax.grid(True, ls="--", lw=0.5, color="lightgray", alpha=0.7)
            ax.set_axisbelow(True)
            if ylim is not None:
                ax.set_ylim(*ylim)
        axrow[0].set_ylabel(ylabel)

    # paradigm titles only on top row
    for ax, p in zip(axes[0], PARADIGMS):
        ax.set_title(PARADIGM_LABEL[p], color=PARADIGM_COLOR[p], fontweight="bold", pad=4)
    for ax in axes[1]:
        ax.set_xlabel("RL cycle")

    # single legend for setups (use mg axis to keep colour consistent)
    handles = [
        plt.Line2D([0], [0], color="black", ls="--", marker="s", ms=4, lw=1.4, label="BASE policy"),
        plt.Line2D([0], [0], color="black", ls="-",  marker="o", ms=4, lw=1.4, label="ACC policy"),
        plt.Line2D([0], [0], color="gray",  ls=":",  lw=0.9, label=r"target = 1.5 J/g/K"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Closed-loop discovery curves (5-seed mean ± std)",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, OUT / "fig_bogen_curves.png")


def fig_surrogate_quality(jobs_cp: pd.DataFrame, jobs_bm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=False)
    panels = [
        (jobs_cp, axes[0], r"GP CV5 RMSE (normalised $C_{p}$)", "Cp surrogate quality"),
        (jobs_bm, axes[1], r"GP CV5 RMSE (normalised $K_{\mathrm{VRH}}$)", "K$_{\\mathrm{VRH}}$ surrogate quality"),
    ]
    for jobs, ax, ylabel, title in panels:
        df = load_gp_metric(jobs, "gp_rmse_cv5")
        if df.empty:
            ax.text(0.5, 0.5, "no GP CV5 logs", ha="center", va="center", transform=ax.transAxes)
            continue
        for p in PARADIGMS:
            s = df[df.paradigm == p]
            if s.empty:
                continue
            agg = s.groupby("cycle")["value"].agg(["mean", "std"])
            x = agg.index.values
            y = agg["mean"].values
            yerr = agg["std"].fillna(0).values
            color = PARADIGM_COLOR[p]
            ax.plot(x, y, color=color, lw=1.6, marker="o", ms=3.5, label=PARADIGM_LABEL[p])
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12, lw=0)
        ax.set_xlabel("RL cycle")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=6)
        ax.grid(True, ls="--", lw=0.5, color="lightgray", alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(title="Backbone", title_fontsize=9, loc="upper right")
    fig.suptitle("Surrogate fit quality across BO cycles (ACC runs only)",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, OUT / "fig_bogen_surrogate_quality.png")


def main() -> None:
    jobs_cp = discover("Cp")
    jobs_bm = discover("BM")
    print(f"Cp jobs: {len(jobs_cp)}    BM jobs: {len(jobs_bm)}")

    df_cp = best_running(jobs_cp, "Cp", 0.25, 2.0)
    df_bm = best_running(jobs_bm, "BM", 20.0, 400.0)
    print(f"running-best rows  Cp: {len(df_cp)}   BM: {len(df_bm)}")

    fig_curves(df_cp, df_bm)
    fig_surrogate_quality(jobs_cp, jobs_bm)


if __name__ == "__main__":
    sys.exit(main())
