#!/usr/bin/env python
"""Corrected SI per-cycle discovery curves (replaces fig_bogen_curves.png).

The original generator (matinvent-hcap-bo/analysis/v4_metrics_plots/plot_metrics.py
plot_best_running) applied a target-distance REFLECTION for Cp:
    score = -|prop - 1.5|;  running_best = 1.5 - |running_best score|
which mirrors any ABOVE-target Cp below the target (a perfect prop=2.0 renders as
1.0), under-reporting the gated policy. This version uses the same running-best
mapping as the (verified) main-text dispatched figures: prop = reward*(mx-mn)+mn,
capped at mx when reward>=1, per-seed cummax over cycles, then seed-mean +/- std.
K_VRH was never reflected (no target), so those panels are unchanged in spirit.

Additive: writes a NEW filename; originals untouched. Read-only on data.
"""
import glob, os, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RES = ROOT / "02_closed_loop_bo" / "results" / "results-paper-v4"
OUT = ROOT / "02_closed_loop_bo" / "figures" / "regenerated" / "dispatched_regen"
OUT.mkdir(parents=True, exist_ok=True)
HORIZON = 20
LIM = {"Cp": (0.25, 2.0), "KVRH": (20.0, 400.0)}

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def dedup(dirs):
    best = {}
    for d in dirs:
        m = re.search(r"seed(\d+)(?:_patched)?_(\d+)$", d)
        if not m or not os.path.exists(f"{d}/samples/long_term_memory.csv"):
            continue
        seed, job = int(m.group(1)), int(m.group(2))
        if seed not in best or job > best[seed][0]:
            best[seed] = (job, d)
    return [v[1] for v in best.values()]


def running_best(d, prop):
    """Per-cycle running-best property (correct mapping; no reflection)."""
    mn, mx = LIM[prop]
    ltm = pd.read_csv(f"{d}/samples/long_term_memory.csv", usecols=["reward", "RL_step"])
    lv = ltm[ltm.reward > 0].copy()
    pv = lv.reward * (mx - mn) + mn
    pv[lv.reward >= 1.0] = mx
    return pv.groupby(lv.RL_step).max().reindex(range(HORIZON)).cummax().ffill().values


def cell(pattern, root, prop):
    ys = []
    for d in dedup(glob.glob(str(RES / root / pattern))):
        ys.append(running_best(d, prop))
    ys = np.array(ys)
    return ys.mean(0), ys.std(0), len(ys)


BB = {"mg": ("MatterGen", "#1f77b4"), "cf": ("CrystalFlow", "#2ca02c"),
      "adit": ("ADiT", "#d62728")}
POL = {"ACC": ("accel", "-", "o"), "BASE": ("baseline", "--", "s")}
YLAB = {"Cp": r"running best $C_p$ (J/g/K)", "KVRH": r"running best $K_{\mathrm{VRH}}$ (GPa)"}
CYC = np.arange(HORIZON)

fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4))
endpoints = {}
for i, prop in enumerate(["Cp", "KVRH"]):
    root = "results" if prop == "Cp" else "results_bm"
    pre = "hcap_p3v4" if prop == "Cp" else "bm_p3v4_bm"
    for j, (bb, (bl, color)) in enumerate(BB.items()):
        ax = axes[i, j]
        for pol, (suffix, ls, mk) in POL.items():
            my, sy, n = cell(f"{pre}_{bb}_{suffix}_seed*", root, prop)
            ax.plot(CYC, my, ls=ls, marker=mk, ms=3.0, lw=1.4, color=color,
                    alpha=1.0 if pol == "ACC" else 0.6, label=f"{pol} (n={n})")
            ax.fill_between(CYC, my - sy, my + sy, color=color, alpha=0.12)
            endpoints[(prop, bb, pol)] = (my[-1], sy[-1], n)
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(bl, color=color, fontweight="bold")
            ax.axhline(1.5, color="gray", ls=":", lw=0.9, alpha=0.7, label="target = 1.5")
            ax.set_ylim(0.2, 2.05)
        if i == 1:
            ax.set_xlabel("RL cycle")
        if j == 0:
            ax.set_ylabel(YLAB[prop])
        ax.legend(loc="upper left", handlelength=2.0)

fig.suptitle("Closed-loop discovery curves, corrected mapping  (5-seed mean $\\pm$ std)",
             y=0.998, fontsize=12.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_bogen_curves_corrected.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_bogen_curves_corrected")
print("\n=== Cp endpoints (cycle 19), corrected vs old-reflected SI figure ===")
for bb in BB:
    for pol in POL:
        v, s, n = endpoints[("Cp", bb, pol)]
        print(f"  Cp {bb:5s} {pol:4s}: {v:.3f} +/- {s:.3f}  (n={n})")
