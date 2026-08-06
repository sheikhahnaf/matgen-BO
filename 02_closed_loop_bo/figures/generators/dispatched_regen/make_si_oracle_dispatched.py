#!/usr/bin/env python
"""Dispatched-basis redraw of the three SI oracle-call figures, to match the
dispatched main-text figures (make_value_cost_dispatched.py).

  S2  fig_bogen_oracle_savings_dispatched      cumulative dispatched calls vs cycle, 3 backbones x BASE/ACC
  S3  fig_value_cost_comet_dispatched          endpoint value vs total dispatched calls, 3 backbones x BASE/ACC
  S5  fig_mg_ablation_oracle_cost_dispatched   cumulative dispatched calls vs cycle, MatterGen 4 arms

Dispatched count: BASE/cap-4/oracle-all = metrics.csv 'cost' (every SUN-survivor
sent to the oracle); ACC = gate-log 'n_oracle' (top-K dispatched). The earlier SI
figures counted the non-ACC arms by successful evaluations, mixing two conventions.
Additive: writes NEW filenames; originals untouched. Review-only.
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
YCALLS = {"Cp": r"cumulative oracle calls dispatched ($C_p$)",
          "KVRH": r"cumulative oracle calls dispatched ($K_{\mathrm{VRH}}$)"}
YVAL = {"Cp": r"running best $C_p$ (J/g/K)", "KVRH": r"running best $K_{\mathrm{VRH}}$ (GPa)"}

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


def run_arrays(d, prop, calls_from):
    mn, mx = LIM[prop]
    ltm = pd.read_csv(f"{d}/samples/long_term_memory.csv", usecols=["reward", "RL_step"])
    lv = ltm[ltm.reward > 0].copy()
    pv = lv.reward * (mx - mn) + mn
    pv[lv.reward >= 1.0] = mx
    rb = pv.groupby(lv.RL_step).max().reindex(range(HORIZON)).cummax().ffill().values
    if calls_from == "gate":
        sub = ("rewards/heat_capacity/gp_routed_v4_log.csv" if prop == "Cp"
               else "rewards/bulk_modulus/bm_gp_routed_v4_log.csv")
        g = pd.read_csv(f"{d}/{sub}")
        calls = (g.set_index("cycle")["n_oracle"].reindex(range(HORIZON), fill_value=0)
                 .cumsum().values.astype(float))
    else:
        m = pd.read_csv(f"{d}/metrics.csv")
        calls = m.set_index("step")["cost"].reindex(range(HORIZON)).ffill().values.astype(float)
    return calls, rb


def cell(pattern, root, prop, calls_from, excl=None):
    xs, ys = [], []
    for d in dedup(glob.glob(str(RES / root / pattern))):
        if excl and re.search(excl, d):
            continue
        c, r = run_arrays(d, prop, calls_from)
        xs.append(c); ys.append(r)
    xs, ys = np.array(xs), np.array(ys)
    return xs.mean(0), ys.mean(0), xs.std(0), ys.std(0), len(xs)


BB = {"mg": ("MatterGen", "#1f77b4"), "cf": ("CrystalFlow", "#2ca02c"),
      "adit": ("ADiT", "#d62728")}
POL = {"ACC": ("accel", "-", "o", "gate"), "BASE": ("baseline", "--", "s", "cost")}
CYC = np.arange(HORIZON)

# ============================================================== S2: calls/cycle
fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4))
for i, prop in enumerate(["Cp", "KVRH"]):
    root = "results" if prop == "Cp" else "results_bm"
    pre = "hcap_p3v4" if prop == "Cp" else "bm_p3v4_bm"
    for j, (bb, (bl, color)) in enumerate(BB.items()):
        ax = axes[i, j]
        for pol, (suffix, ls, mk, cf) in POL.items():
            mc, _, sc, _, n = cell(f"{pre}_{bb}_{suffix}_seed*", root, prop, cf)
            ax.plot(CYC, mc, ls=ls, marker=mk, ms=3.0, lw=1.4, color=color,
                    alpha=1.0 if pol == "ACC" else 0.6, label=f"{pol} (n={n})")
            ax.fill_between(CYC, mc - sc, mc + sc, color=color, alpha=0.12)
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(bl, color=color, fontweight="bold")
        if i == 1:
            ax.set_xlabel("RL cycle")
        if j == 0:
            ax.set_ylabel(YCALLS[prop])
        ax.legend(loc="upper left", handlelength=2.2)
fig.suptitle("Cumulative oracle calls per closed-loop run, dispatched basis  (5-seed mean $\\pm$ std)",
             y=0.997, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_bogen_oracle_savings_dispatched.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_bogen_oracle_savings_dispatched")

# ===================================================================== S3: comet
DATA = {}
for prop in ["Cp", "KVRH"]:
    root = "results" if prop == "Cp" else "results_bm"
    pre = "hcap_p3v4" if prop == "Cp" else "bm_p3v4_bm"
    for bb in BB:
        for pol, (suffix, _, _, cf) in POL.items():
            DATA[(prop, bb, pol)] = cell(f"{pre}_{bb}_{suffix}_seed*", root, prop, cf)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
for ax, prop in zip(axes, ["Cp", "KVRH"]):
    for bb, (bl, color) in BB.items():
        for pol, (_, ls, mk, _) in POL.items():
            mx, my, sx, sy, n = DATA[(prop, bb, pol)]
            ax.plot(mx, my, ls="-", lw=1.0, color=color, alpha=0.28, zorder=2)
            ax.errorbar(mx[-1], my[-1], xerr=sx[-1], yerr=sy[-1], fmt=mk, ms=9,
                        color=color, mfc=color if pol == "ACC" else "white",
                        mec=color, mew=1.4, elinewidth=1.1, capsize=3, zorder=4)
    ax.set_xlabel("total oracle calls dispatched per run")
    ax.set_ylabel(YVAL[prop])
    ax.grid(alpha=0.25, lw=0.5)
    if prop == "Cp":
        ax.axhline(1.5, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax.text(0.985, 1.512, "target = 1.5", color="gray", fontsize=8,
                ha="right", transform=ax.get_yaxis_transform())
    ax.set_title("heat capacity" if prop == "Cp" else "bulk modulus")
handles = ([Line2D([], [], color=c, lw=2.4, label=l) for l, c in BB.values()]
           + [Line2D([], [], color="#444", marker="o", ls="", ms=8, label="ACC (GP gate)"),
              Line2D([], [], color="#444", marker="s", ls="", ms=8, mfc="white", label="BASE")])
axes[1].legend(handles=handles, loc="lower right", ncol=1, handlelength=1.6)
fig.suptitle("Where each policy ends: best property vs oracle calls dispatched  (endpoints, 5-seed mean $\\pm$ std; tails = trajectories)",
             y=1.02, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_value_cost_comet_dispatched.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_value_cost_comet_dispatched")

# ============================================================ S5: ablation calls
ARMS = {
    "Cp": [("ACC", "hcap_p3v4_mg_accel_seed*", "results", "gate", "#1f77b4", "-", "o", None),
           ("BASE", "hcap_p3v4_mg_baseline_seed*", "results", "cost", "#ff7f0e", "--", "s", None),
           ("cap-4 (ours)", "hcap_mgabl_cap4_cp_seed*", "results", "cost", "#2ca02c", "-", "^", "_18636280|_18636282"),
           ("oracle-all (= BASE)", "hcap_p3v4_mg_baseline_seed*", "results", "cost", "#9467bd", ":", "D", None)],
    "KVRH": [("ACC", "bm_p3v4_bm_mg_accel_seed*", "results_bm", "gate", "#1f77b4", "-", "o", None),
             ("BASE", "bm_p3v4_bm_mg_baseline_seed*", "results_bm", "cost", "#ff7f0e", "--", "s", None),
             ("cap-4 (ours)", "hcap_mgabl_cap4_bm_seed*", "results_bm", "cost", "#2ca02c", "-", "^", None),
             ("oracle-all (ours)", "hcap_mgabl_oracleall_bm_seed*", "results_bm", "cost", "#9467bd", "-", "D", None)],
}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, prop in zip(axes, ["Cp", "KVRH"]):
    for label, pat, root, cf, color, ls, mk, excl in ARMS[prop]:
        mc, _, sc, _, n = cell(pat, root, prop, cf, excl)
        ax.plot(CYC, mc, ls=ls, marker=mk, ms=3.4, lw=1.5, color=color, label=f"{label} (n={n})")
        ax.fill_between(CYC, mc - sc, mc + sc, color=color, alpha=0.12)
    ax.set_xlabel("RL cycle")
    ax.set_ylabel(YCALLS[prop])
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("MatterGen / heat capacity" if prop == "Cp" else "MatterGen / bulk modulus")
    ax.legend(loc="upper left", handlelength=2.4)
fig.suptitle("MatterGen ablation: cumulative oracle calls per run, dispatched basis  (5-seed mean $\\pm$ std)",
             y=1.015, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_mg_ablation_oracle_cost_dispatched.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_mg_ablation_oracle_cost_dispatched")
