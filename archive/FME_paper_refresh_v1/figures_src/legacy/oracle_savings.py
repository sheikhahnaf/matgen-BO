"""Oracle-call savings: BASE vs ACC across cycles, both targets.

For BASE, every candidate that survives the SUN filter is sent to the oracle, so
the per-cycle BASE oracle count is the row count in long_term_memory.csv per RL_step.
For ACC, only the top-K (=4) GP-selected candidates per cycle hit the oracle; the
rest receive a GP-μ pseudo-evaluation. The per-cycle ACC oracle count is `n_oracle`
in `rewards/<prop>/{gp_routed,bm_gp_routed}_v4_log.csv`.

Produces:
  fig_bogen_oracle_savings.png    — 2x3 grid: Cp (top) and K_VRH (bottom) × 3 backbones
                                     showing cumulative oracle calls BASE (dashed) vs ACC (solid)

Headline numbers are also printed for the figure caption.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_figure

apply_style()

ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-hcap-bo")
OUT = Path("/Volumes/SSD1_SMAAA/matinvent-bo/FME_paper_refresh_v1/figures")

PARADIGMS = ["mg", "cf", "adit"]
PARADIGM_LABEL = {"mg": "MatterGen", "cf": "CrystalFlow", "adit": "ADiT"}
PARADIGM_COLOR = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}
MAX_CYCLES = 20


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
        rows.append({"paradigm": paradigm, "setup": setup, "seed": seed, "jobid": jobid,
                     "dir": d, "ltm": d / "samples" / "long_term_memory.csv",
                     "gp_log": d / Path(*gp_subdir)})
    df = pd.DataFrame(rows)
    return df.sort_values("jobid").drop_duplicates(
        subset=["paradigm", "setup", "seed"], keep="last")


def base_calls_per_cycle(ltm_path: Path) -> pd.Series:
    """BASE: every SUN-survivor goes to oracle => row count per RL_step."""
    if not ltm_path.exists():
        return pd.Series(dtype=int)
    df = pd.read_csv(ltm_path, usecols=["RL_step"])
    s = df.groupby("RL_step").size()
    return s.reindex(range(MAX_CYCLES), fill_value=0)


def acc_calls_per_cycle(gp_log: Path) -> pd.Series:
    """ACC: only top-K hit oracle => n_oracle from gp routing log."""
    if not gp_log.exists():
        return pd.Series(dtype=int)
    df = pd.read_csv(gp_log)
    s = df.set_index("cycle")["n_oracle"]
    return s.reindex(range(MAX_CYCLES), fill_value=0)


def per_seed_calls(jobs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in jobs.iterrows():
        per_cycle = (acc_calls_per_cycle(r.gp_log) if r.setup == "ACC"
                     else base_calls_per_cycle(r.ltm))
        for c, n in per_cycle.items():
            rows.append({"paradigm": r.paradigm, "setup": r.setup, "seed": r.seed,
                         "cycle": int(c), "n": int(n)})
    return pd.DataFrame(rows)


def fig_oracle_savings(df_cp: pd.DataFrame, df_bm: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
    rows = [
        (df_cp, axes[0], r"cumulative oracle calls ($C_{p}$)"),
        (df_bm, axes[1], r"cumulative oracle calls ($K_{\mathrm{VRH}}$)"),
    ]
    for df, axrow, ylabel in rows:
        for ax, p in zip(axrow, PARADIGMS):
            sub = df[df.paradigm == p]
            for setup, ls, marker in (("BASE", "--", "s"), ("ACC", "-", "o")):
                s = sub[sub.setup == setup]
                if s.empty:
                    continue
                # cumulative calls per seed, then mean ± std across seeds
                pivot = s.pivot_table(index="cycle", columns="seed", values="n",
                                      aggfunc="sum", fill_value=0).sort_index()
                cum = pivot.cumsum(axis=0)
                mean = cum.mean(axis=1)
                std = cum.std(axis=1)
                color = PARADIGM_COLOR[p]
                ax.plot(mean.index, mean.values, ls=ls, color=color, lw=1.6,
                        marker=marker, ms=3.5, label=setup)
                ax.fill_between(mean.index, (mean - std).values, (mean + std).values,
                                color=color, alpha=0.12, lw=0)
            ax.grid(True, ls="--", lw=0.5, color="lightgray", alpha=0.7)
            ax.set_axisbelow(True)
        axrow[0].set_ylabel(ylabel)

    # Top-row paradigm titles
    for ax, p in zip(axes[0], PARADIGMS):
        ax.set_title(PARADIGM_LABEL[p], color=PARADIGM_COLOR[p], fontweight="bold", pad=4)
    for ax in axes[1]:
        ax.set_xlabel("RL cycle")

    handles = [
        plt.Line2D([0], [0], color="black", ls="--", marker="s", ms=4, lw=1.4,
                   label="BASE (oracle on all SUN-survivors)"),
        plt.Line2D([0], [0], color="black", ls="-",  marker="o", ms=4, lw=1.4,
                   label="ACC (oracle on top-$K=4$ only)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Cumulative oracle calls per closed-loop run (5-seed mean ± std)",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, OUT / "fig_bogen_oracle_savings.png")


def headline_numbers(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} — total oracle calls per run (mean ± std across 5 seeds) ===")
    for p in PARADIGMS:
        for setup in ("BASE", "ACC"):
            s = df[(df.paradigm == p) & (df.setup == setup)]
            if s.empty:
                continue
            totals = s.groupby("seed")["n"].sum()
            print(f"  {PARADIGM_LABEL[p]:<12} {setup:<5}  total = {totals.mean():6.1f} ± {totals.std():5.1f}   "
                  f"(n_seeds = {len(totals)})")
        # speedup
        b = df[(df.paradigm == p) & (df.setup == "BASE")].groupby("seed")["n"].sum().mean()
        a = df[(df.paradigm == p) & (df.setup == "ACC")].groupby("seed")["n"].sum().mean()
        if a > 0:
            print(f"  {PARADIGM_LABEL[p]:<12} speedup BASE/ACC = {b/a:.2f}x")


def main() -> None:
    jobs_cp = discover("Cp")
    jobs_bm = discover("BM")
    print(f"Cp jobs: {len(jobs_cp)}    BM jobs: {len(jobs_bm)}")

    df_cp = per_seed_calls(jobs_cp)
    df_bm = per_seed_calls(jobs_bm)

    fig_oracle_savings(df_cp, df_bm)
    headline_numbers(df_cp, "Cp")
    headline_numbers(df_bm, "K_VRH")


if __name__ == "__main__":
    main()
