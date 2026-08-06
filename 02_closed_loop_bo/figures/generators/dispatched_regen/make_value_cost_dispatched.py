#!/usr/bin/env python
"""Dispatched-basis generators for the two main-text oracle cost-versus-value figures.

Paper artifacts (final manuscript versions):
  Fig 3  fig_value_cost_backbones_dispatched   discovery vs cumulative dispatched calls, 3 backbones
  Fig 5  fig_mg_ablation_value_cost_dispatched MatterGen budget ablation, 4 arms

The horizontal axis counts every oracle evaluation that is *dispatched*, on the
same footing for every policy. For BASE, cap-4, and oracle-all this is the
metrics.csv 'cost' column (each SUN-survivor sent to the oracle, whether or not
it returns a valid property); for ACC it is the gate log 'n_oracle' (the top-K
actually dispatched). Earlier figures counted BASE/cap-4/oracle-all by
successful evaluations (LTM rows) while counting ACC by dispatched calls, which
mixed two conventions; this generator removes that asymmetry.

Provenance: verbatim transform of fig_redesign_20260611/make_value_cost_dispatched.py
(2026-06-18) with two changes: (1) data root resolved in-tree instead of a local
mirror path; (2) the ACC legend label reads "ACC (ours)" to match the final
manuscript figure (reviewer remark, 2026-08-06). Outputs are written to
figures/regenerated/dispatched_regen/ and never overwrite the committed
figures/rendered/ copies.
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
XLAB = "cumulative oracle calls dispatched"

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
    """Per-cycle (cumulative dispatched calls, running best) for one run dir."""
    mn, mx = LIM[prop]
    ltm = pd.read_csv(f"{d}/samples/long_term_memory.csv", usecols=["reward", "RL_step"])
    lv = ltm[ltm.reward > 0].copy()
    pv = lv.reward * (mx - mn) + mn
    pv[lv.reward >= 1.0] = mx
    rb = pv.groupby(lv.RL_step).max().reindex(range(HORIZON)).cummax().ffill().values
    if calls_from == "gate":  # ACC: top-K actually dispatched to the oracle
        sub = ("rewards/heat_capacity/gp_routed_v4_log.csv" if prop == "Cp"
               else "rewards/bulk_modulus/bm_gp_routed_v4_log.csv")
        g = pd.read_csv(f"{d}/{sub}")
        calls = (g.set_index("cycle")["n_oracle"].reindex(range(HORIZON), fill_value=0)
                 .cumsum().values.astype(float))
    else:  # "cost": every SUN-survivor dispatched to the oracle (already cumulative)
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


def annotate_speedup(ax, amx, amy, bmx, bmy, star_color, i_row=0, corner="br"):
    """Dotted guide at BASE's final best; star where ACC first reaches it.

    corner='br' places the label bottom-right (Fig 3 grid, legend is upper-left);
    corner='ul' places it upper-left (Fig 5 ablation, legend is lower-right) so
    the label never sits on top of the legend.
    """
    target = bmy[-1]
    ax.axhline(target, color="gray", ls=":", lw=0.9, alpha=0.75, zorder=1)
    hit = np.argmax(amy >= target) if (amy >= target).any() else None
    if hit is not None and amy[hit] >= target:
        n_calls, m_calls = amx[hit], bmx[-1]
        ratio = m_calls / n_calls
        ax.plot(n_calls, target, marker="*", ms=13, color=star_color,
                mec="white", mew=0.8, zorder=6)
        txt = (f"gate matches BASE's best\nat {n_calls:.0f} calls "
               + (f"({ratio:.1f}x fewer)" if ratio >= 1.05 else f"(BASE: {m_calls:.0f})"))
        if corner == "ul":
            xytext, ha, va = (0.03, 0.97), "left", "top"
        else:
            xytext, ha, va = (0.97, 0.04 if i_row == 1 else 0.06), "right", "bottom"
        ax.annotate(txt, xy=(n_calls, target),
                    xytext=xytext, textcoords="axes fraction",
                    ha=ha, va=va, fontsize=7.8, color="#333",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.7, shrinkA=2, shrinkB=4))


# ============================================================ Fig 3: backbones
BB = {"mg": ("MatterGen", "#1f77b4"), "cf": ("CrystalFlow", "#2ca02c"),
      "adit": ("ADiT", "#d62728")}
YLAB = {"Cp": r"running best $C_p$ (J/g/K)", "KVRH": r"running best $K_{\mathrm{VRH}}$ (GPa)"}

fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6))
for i, prop in enumerate(["Cp", "KVRH"]):
    root = "results" if prop == "Cp" else "results_bm"
    pre = "hcap_p3v4" if prop == "Cp" else "bm_p3v4_bm"
    cells = {}
    for bb in BB:
        cells[(bb, "ACC")] = cell(f"{pre}_{bb}_accel_seed*", root, prop, "gate")
        cells[(bb, "BASE")] = cell(f"{pre}_{bb}_baseline_seed*", root, prop, "cost")
    row_xmax = max(v[0][-1] + v[2][-1] for v in cells.values()) * 1.06
    for j, (bb, (bl, color)) in enumerate(BB.items()):
        ax = axes[i, j]
        amx, amy, asx, asy, an = cells[(bb, "ACC")]
        bmx, bmy, bsx, bsy, bn = cells[(bb, "BASE")]
        ax.plot(bmx, bmy, "--", marker="s", ms=3.0, lw=1.3, color=color, alpha=0.55,
                label=f"BASE (n={bn})")
        ax.plot(amx, amy, "-", marker="o", ms=3.2, lw=1.6, color=color,
                label=f"ACC (n={an})")
        ax.errorbar(bmx[-1], bmy[-1], xerr=bsx[-1], yerr=bsy[-1], fmt="s", ms=7.5,
                    color=color, mfc="white", mec=color, mew=1.1, elinewidth=1.0,
                    capsize=2.5, alpha=0.85, zorder=4)
        ax.errorbar(amx[-1], amy[-1], xerr=asx[-1], yerr=asy[-1], fmt="o", ms=7.5,
                    color=color, mfc=color, mec="white", mew=0.9, elinewidth=1.0,
                    capsize=2.5, zorder=5)
        annotate_speedup(ax, amx, amy, bmx, bmy, color, i_row=i)
        ax.set_xlim(-row_xmax * 0.03, row_xmax)
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(bl, color=color, fontweight="bold")
        if i == 1:
            ax.set_xlabel(XLAB)
        if j == 0:
            ax.set_ylabel(YLAB[prop])
        ax.legend(loc="upper left", handlelength=2.2)
    ylims = [axes[i, j].get_ylim() for j in range(3)]
    lo, hi = min(l for l, _ in ylims), max(h for _, h in ylims)
    for j in range(3):
        axes[i, j].set_ylim(lo, hi)
        if j > 0:
            axes[i, j].set_yticklabels([])
fig.suptitle("Discovery vs oracle cost per backbone  (5-seed mean; endpoint $\\pm$ std; "
             "$\\star$ = gate reaches ungated policy's final best)", y=0.998, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_value_cost_backbones_dispatched.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_value_cost_backbones_dispatched")

# ============================================================ Fig 5: ablation
def draw(ax, mx, my, sx, sy, n, color, ls, marker, label):
    ax.plot(mx, my, ls=ls, marker=marker, ms=3.4, lw=1.5, color=color,
            label=f"{label} (n={n})", zorder=3)
    ax.errorbar(mx[-1], my[-1], xerr=sx[-1], yerr=sy[-1], color=color,
                fmt=marker, ms=8, mfc=color, mec="white", mew=0.9,
                elinewidth=1.1, capsize=3, zorder=4)

ARMS = {
    "Cp": [("ACC (ours)", "hcap_p3v4_mg_accel_seed*", "results", "gate", "#1f77b4", "-", "o", None),
           ("BASE", "hcap_p3v4_mg_baseline_seed*", "results", "cost", "#ff7f0e", "--", "s", None),
           ("cap-4 (ours)", "hcap_mgabl_cap4_cp_seed*", "results", "cost", "#2ca02c", "-", "^", "_18636280|_18636282"),
           ("oracle-all (= BASE)", "hcap_p3v4_mg_baseline_seed*", "results", "cost", "#9467bd", ":", "D", None)],
    "KVRH": [("ACC (ours)", "bm_p3v4_bm_mg_accel_seed*", "results_bm", "gate", "#1f77b4", "-", "o", None),
             ("BASE", "bm_p3v4_bm_mg_baseline_seed*", "results_bm", "cost", "#ff7f0e", "--", "s", None),
             ("cap-4 (ours)", "hcap_mgabl_cap4_bm_seed*", "results_bm", "cost", "#2ca02c", "-", "^", None),
             ("oracle-all (ours)", "hcap_mgabl_oracleall_bm_seed*", "results_bm", "cost", "#9467bd", "-", "D", None)],
}
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, prop in zip(axes, ["Cp", "KVRH"]):
    arm = {}
    for label, pat, root, cf, color, ls, mk, excl in ARMS[prop]:
        mx, my, sx, sy, n = cell(pat, root, prop, cf, excl)
        draw(ax, mx, my, sx, sy, n, color, ls, mk, label)
        arm[label.split()[0]] = (mx, my)
    annotate_speedup(ax, arm["ACC"][0], arm["ACC"][1], arm["BASE"][0], arm["BASE"][1], "#1f77b4", corner="ul")
    ax.set_xlabel(XLAB)
    if prop == "Cp":
        ax.set_ylabel(r"running best $C_p$ (J/g/K)")
    else:
        ax.set_ylabel(r"running best $K_{\mathrm{VRH}}$ (GPa)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("MatterGen / heat capacity" if prop == "Cp" else "MatterGen / bulk modulus")
    ax.legend(loc="lower right", handlelength=2.4)
fig.suptitle("Budget-matched ablation as cost vs discovery: what each oracle call buys  (5-seed mean; endpoint $\\pm$ std)",
             y=1.015, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_mg_ablation_value_cost_dispatched.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_mg_ablation_value_cost_dispatched")
