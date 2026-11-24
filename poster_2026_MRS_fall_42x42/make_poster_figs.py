#!/usr/bin/env python
"""Poster-scale figures for the 2026 MRS Fall poster (42 x 42 in variant).

Every figure is drawn at its printed size with printed font sizes (>= 20 pt), in the
poster palette (maroon = gated / ours, grey = ungated baseline). Data are read from the
repository and never modified:

  fig_benchmark   01_static_benchmark/data/*/aggregated_results.csv
  fig_discovery   02_closed_loop_bo/results/results-paper-v4   (same transform as
  fig_ablation      figures/generators/dispatched_regen/make_value_cost_dispatched.py)
  fig_dft_*       02_closed_loop_bo/validation/three_way_comparison/results/*.csv
  fig_structures  02_closed_loop_bo/validation/dft_validation/structures/*.cif
  fig_denoise     the MoN structure above with decreasing positional noise

Run:  python make_poster_figs.py [benchmark discovery ablation dft structures denoise]
"""
from __future__ import annotations
import glob, os, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

MAROON, GREY, DARK = "#500000", "#8a8a8a", "#3c3c3c"
AMBER, STEEL, LIGHT = "#c9892b", "#4f7cac", "#d9d9d9"

SZ = dict(label=24, tick=21, title=26, note=21, legend=21)
plt.rcParams.update({
    "font.family": "Arial", "font.size": SZ["tick"],
    "axes.labelsize": SZ["label"], "axes.titlesize": SZ["title"],
    "xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"],
    "legend.fontsize": SZ["legend"], "legend.frameon": False,
    "axes.linewidth": 1.6, "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
})


def save(fig, name, png=False):
    out = FIGS / f"{name}.{'png' if png else 'pdf'}"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300, transparent=png)
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


# ------------------------------------------------------------------ benchmark
def fig_benchmark():
    """Best mean held-out R^2 at n_train = 500 per featurizer and dataset."""
    data = ROOT / "01_static_benchmark" / "data"
    sets = {"Elastic\n(8 targets)": data / "analysis_v3" / "aggregated_results.csv",
            "Dielectric\n(4 targets)": data / "analysis_v3_dielectric_constant" / "aggregated_results.csv",
            "Phonon\n(4 targets)": data / "arm_a_dfpt" / "aggregated_results.csv"}
    order = ["soap", "mace", "uma", "orb"]
    label = {"soap": "SOAP", "mace": "MACE", "uma": "UMA", "orb": "ORB"}
    colour = {"soap": LIGHT, "mace": "#a9b8c9", "uma": "#7f8fa3", "orb": MAROON}
    best = {}
    for ds, path in sets.items():
        d = pd.read_csv(path)
        d = d[(d.metric == "R2") & (d.n_train == 500)]
        d = d[d.property != "kpoint_density"]  # calculation setting, not a target (as in the paper)
        avg = d.groupby(["descriptor", "model", "pca_components"])["mean"].mean()
        for desc in order:
            s = avg.loc[desc]
            (model, pca), val = s.idxmax(), s.max()
            best[(ds, desc)] = (val, model, pca)
            print(f"  {ds.split()[0]:10s} {desc:5s} best R2 = {val:.3f}  ({model}, PCA {pca})")
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    x = np.arange(len(sets)) * 1.08; w = 0.22
    for i, desc in enumerate(order):
        pos = x + (i - 1.5) * w
        vals = [best[(ds, desc)][0] for ds in sets]
        ax.bar(pos, vals, w * 0.92, color=colour[desc], edgecolor=DARK, lw=1.0, label=label[desc])
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=16,
                    color=MAROON if desc == "orb" else DARK,
                    fontweight="bold" if desc == "orb" else "normal")
    ax.set_xticks(x); ax.set_xticklabels(list(sets))
    ax.set_ylabel("best held-out $R^2$")
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, axis="y", alpha=0.3, lw=0.8); ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(-0.02, 1.16), columnspacing=1.2,
              handlelength=1.1, handletextpad=0.4)
    save(fig, "fig_benchmark")


# ------------------------------------------------------- closed-loop transforms
RES = ROOT / "02_closed_loop_bo" / "results" / "results-paper-v4"
HORIZON = 20
LIM = {"Cp": (0.25, 2.0), "KVRH": (20.0, 400.0)}


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
    """Per-cycle (cumulative dispatched oracle calls, running best) for one run."""
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


def first_match(amx, amy, bmx, bmy):
    """Calls at which the gated mean first reaches the ungated final best."""
    target = bmy[-1]
    if not (amy >= target).any():
        return None
    hit = int(np.argmax(amy >= target))
    return amx[hit], target, bmx[-1] / amx[hit]


YLAB = {"Cp": "running best $C_p$  (J g$^{-1}$ K$^{-1}$)",
        "KVRH": "running best $K_{\\mathrm{VRH}}$  (GPa)"}


# ------------------------------------------------------------------ discovery
def fig_discovery():
    BB = {"mg": "MatterGen", "cf": "CrystalFlow", "adit": "ADiT"}
    fig, axes = plt.subplots(2, 3, figsize=(19.6, 12.2), sharey="row")
    for i, prop in enumerate(["Cp", "KVRH"]):
        root = "results" if prop == "Cp" else "results_bm"
        pre = "hcap_p3v4" if prop == "Cp" else "bm_p3v4_bm"
        cells = {}
        for bb in BB:
            cells[(bb, "ACC")] = cell(f"{pre}_{bb}_accel_seed*", root, prop, "gate")
            cells[(bb, "BASE")] = cell(f"{pre}_{bb}_baseline_seed*", root, prop, "cost")
        xmax = max(v[0][-1] + v[2][-1] for v in cells.values()) * 1.05
        for j, (bb, name) in enumerate(BB.items()):
            ax = axes[i, j]
            amx, amy, asx, asy, an = cells[(bb, "ACC")]
            bmx, bmy, bsx, bsy, bn = cells[(bb, "BASE")]
            ax.plot(bmx, bmy, "--", color=GREY, lw=3.0, marker="s", ms=6, mfc="white", mew=1.6)
            ax.errorbar(bmx[-1], bmy[-1], xerr=bsx[-1], yerr=bsy[-1], fmt="s", ms=13, color=GREY,
                        mfc="white", mec=GREY, mew=2.2, elinewidth=2.0, capsize=5, zorder=4)
            ax.plot(amx, amy, "-", color=MAROON, lw=4.0, marker="o", ms=6.5)
            ax.errorbar(amx[-1], amy[-1], xerr=asx[-1], yerr=asy[-1], fmt="o", ms=13, color=MAROON,
                        mec="white", mew=1.6, elinewidth=2.2, capsize=5, zorder=5)
            ax.axhline(bmy[-1], color=GREY, ls=":", lw=1.8, zorder=1)
            fm = first_match(amx, amy, bmx, bmy)
            if fm:
                n_calls, target, ratio = fm
                ax.plot(n_calls, target, marker="*", ms=30, color=AMBER, mec=DARK, mew=1.2, zorder=7)
                ax.text(0.97, 0.06, f"{ratio:.1f}$\\times$ fewer calls",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=24,
                        fontweight="bold", color=MAROON,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MAROON, lw=1.6))
                print(f"  {prop:4s} {name:11s} gate matches at {n_calls:5.0f} calls; "
                      f"ungated {bmx[-1]:5.0f}; ratio {ratio:.1f}x  (n={an}/{bn})")
            ax.set_xlim(-xmax * 0.03, xmax)
            ax.grid(alpha=0.3, lw=0.8)
            if i == 0:
                ax.set_title(name, fontweight="bold", color=DARK, pad=12)
            if i == 1:
                ax.set_xlabel("cumulative oracle calls")
            if j == 0:
                ax.set_ylabel(YLAB[prop])
    handles = [Line2D([], [], color=MAROON, lw=4, marker="o", ms=10, label="gated: GP ranks, top 4 evaluated (ours)"),
               Line2D([], [], color=GREY, lw=3, ls="--", marker="s", ms=10, mfc="white", mew=2,
                      label="ungated: every survivor evaluated"),
               Line2D([], [], color=AMBER, lw=0, marker="*", ms=26, mec=DARK, mew=1.2,
                      label="gate reaches the ungated final best")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.045),
               columnspacing=1.6, handletextpad=0.5, fontsize=22)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    save(fig, "fig_discovery")


# ------------------------------------------------------------------- ablation
def fig_ablation():
    ARMS = {
        "Cp": [("gate: GP picks 4 (ours)", "hcap_p3v4_mg_accel_seed*", "results", "gate", MAROON, "-", "o", None),
               ("ungated", "hcap_p3v4_mg_baseline_seed*", "results", "cost", GREY, "--", "s", None),
               ("cap-4: first 4 survivors", "hcap_mgabl_cap4_cp_seed*", "results", "cost", AMBER, "-", "^",
                "_18636280|_18636282")],
        "KVRH": [("gate: GP picks 4 (ours)", "bm_p3v4_bm_mg_accel_seed*", "results_bm", "gate", MAROON, "-", "o", None),
                 ("ungated", "bm_p3v4_bm_mg_baseline_seed*", "results_bm", "cost", GREY, "--", "s", None),
                 ("cap-4: first 4 survivors", "hcap_mgabl_cap4_bm_seed*", "results_bm", "cost", AMBER, "-", "^", None),
                 ("oracle-all: no cap", "hcap_mgabl_oracleall_bm_seed*", "results_bm", "cost", STEEL, "-", "D", None)],
    }
    unit = {"Cp": "J g$^{-1}$ K$^{-1}$", "KVRH": "GPa"}
    fmt = {"Cp": "{:.2f}", "KVRH": "{:.0f}"}
    fig, axes = plt.subplots(1, 2, figsize=(19.6, 8.5))
    for ax, prop in zip(axes, ["Cp", "KVRH"]):
        for label, pat, root, cf, colour, ls, mk, excl in ARMS[prop]:
            mx, my, sx, sy, n = cell(pat, root, prop, cf, excl)
            ours = colour == MAROON
            ax.plot(mx, my, ls=ls, color=colour, lw=4.0 if ours else 3.0, marker=mk, ms=6.5,
                    mfc="white" if ls == "--" else colour, mew=1.6, zorder=5 if ours else 3)
            ax.errorbar(mx[-1], my[-1], xerr=sx[-1], yerr=sy[-1], fmt=mk, ms=13, color=colour,
                        mfc="white" if ls == "--" else colour, mec=colour if ls == "--" else "white",
                        mew=2.0, elinewidth=2.0, capsize=5, zorder=6 if ours else 4)
            print(f"  {prop:4s} {label:26s} end = {my[-1]:.2f} +/- {sy[-1]:.2f} at {mx[-1]:.0f} calls (n={n})")
        note = ("Same budget (4 calls per cycle):\ngate 1.51 vs cap-4 0.86" if prop == "Cp" else
                "Same budget (4 calls per cycle):\ngate 314 vs cap-4 233 GPa\n"
                "oracle-all: +9% for 4.8$\\times$ the calls")
        ax.text(*((0.40, 0.93, note) if prop == "Cp" else (0.97, 0.05, note)), transform=ax.transAxes,
                ha="left" if prop == "Cp" else "right", va="top" if prop == "Cp" else "bottom",
                fontsize=SZ["note"], color=DARK, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MAROON, lw=1.6))
        ax.set_xlabel("cumulative oracle calls")
        ax.set_ylabel(YLAB[prop])
        ax.grid(alpha=0.3, lw=0.8)
        ax.set_title("MatterGen · heat capacity" if prop == "Cp" else "MatterGen · bulk modulus",
                     fontweight="bold", color=DARK, pad=12)
    handles = [Line2D([], [], color=MAROON, lw=4, marker="o", ms=10, label="gate: GP picks 4 (ours)"),
               Line2D([], [], color=AMBER, lw=3, marker="^", ms=10, label="cap-4: first 4 survivors"),
               Line2D([], [], color=GREY, lw=3, ls="--", marker="s", ms=10, mfc="white", mew=2, label="ungated"),
               Line2D([], [], color=STEEL, lw=3, marker="D", ms=10, label="oracle-all: no budget cap")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.4, handletextpad=0.5, fontsize=22)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_ablation")


# ------------------------------------------------------------------------ DFT
VAL = ROOT / "02_closed_loop_bo" / "validation" / "three_way_comparison" / "results"


def fig_dft():
    from scipy.stats import spearmanr, pearsonr
    e = pd.read_csv(VAL / "esen_vs_dft_bm.csv")
    g = pd.read_csv(VAL / "gp_causal_allpreds_bm_seeded.csv")
    w = pd.read_csv(VAL / "three_way_causal_bm_seeded.csv")

    # (a) oracle against DFT
    d, s = e.dft_K0.values, e.esen_K0.values
    mae = np.mean(np.abs(s - d)); mape = np.mean(np.abs(s - d) / d) * 100
    rho = spearmanr(d, s)[0]
    print(f"  oracle vs DFT: n={len(d)}  MAE={mae:.1f} GPa  MAPE={mape:.2f}%  rho={rho:.2f}  bias={np.mean(s-d):+.1f}")
    lo, hi = 280, 385
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.fill_between([lo, hi], [lo - 10, hi - 10], [lo + 10, hi + 10], color=GREY, alpha=0.18, lw=0,
                    label="$\\pm$10 GPa")
    ax.plot([lo, hi], [lo, hi], "--", color=DARK, lw=2.0)
    ax.scatter(d, s, s=260, color=MAROON, edgecolor="white", linewidth=1.6, zorder=3)
    for name, dx, dy, ha, va in [("MoN", -4.5, 0, "right", "center"), ("TiVN2", 0, 4.5, "center", "bottom")]:
        r = e[e.formula == name].iloc[0]
        ax.text(r.dft_K0 + dx, r.esen_K0 + dy, name.replace("2", "$_2$"),
                fontsize=SZ["note"], ha=ha, va=va, color=DARK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", "box")
    ax.set_xticks([300, 325, 350, 375]); ax.set_yticks([300, 325, 350, 375])
    ax.set_xlabel("DFT $K_0$  (GPa)"); ax.set_ylabel("learned oracle $K_0$  (GPa)")
    ax.text(0.05, 0.95, f"MAE {mae:.1f} GPa  ({mape:.1f}%)\n$n$ = {len(d)} structures",
            transform=ax.transAxes, va="top", ha="left", fontsize=SZ["note"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MAROON, lw=1.6))
    ax.legend(loc="lower right", handlelength=1.2)
    ax.grid(alpha=0.3, lw=0.8)
    save(fig, "fig_dft_oracle")

    # (b) surrogate against oracle on unseen structures (causal replay)
    gp, es = g.gp_pred.values, g.esen.values
    rho = spearmanr(gp, es)[0]; r = pearsonr(gp, es)[0]
    rmse = np.sqrt(np.mean((gp - es) ** 2))
    print(f"  GP causal: n={len(gp)}  rho={rho:.3f}  r={r:.2f}  RMSE={rmse:.1f} GPa")
    top = 400
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.plot([0, top], [0, top], "--", color=DARK, lw=2.0)
    # coloured by the training points the loop had gathered when each prediction was made
    # (warm-start seed excluded), as in the manuscript figure
    sc = ax.scatter(es, gp, c=g.n_train_accum.values, cmap="viridis", s=46, alpha=0.6,
                    linewidths=0, zorder=2)
    ax.scatter(w.esen_K0, w.gp_causal_K0, s=330, facecolors="none", edgecolors=MAROON,
               linewidths=3.0, zorder=4, label="DFT-validated winners")
    ax.set_xlim(0, top); ax.set_ylim(0, top); ax.set_aspect("equal", "box")
    ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_xlabel("learned oracle $K_0$  (GPa)"); ax.set_ylabel("GP prediction before evaluation  (GPa)")
    ax.text(0.05, 0.95, f"Spearman $\\rho$ = {rho:.2f}\n$n$ = {len(gp):,} unseen structures",
            transform=ax.transAxes, va="top", ha="left", fontsize=SZ["note"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MAROON, lw=1.6))
    cax = ax.inset_axes([0.06, 0.655, 0.36, 0.042])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=sc.norm, cmap=sc.cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 30, 60, 90]); cb.ax.tick_params(labelsize=18, length=5, width=1.4)
    cb.outline.set_linewidth(1.2)
    cax.set_title("loop-accumulated $n_{\\mathrm{train}}$", fontsize=19, color=DARK, pad=8)
    ax.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.6)
    ax.grid(alpha=0.3, lw=0.8)
    save(fig, "fig_dft_surrogate")


# ----------------------------------------------------------- structure renders
STRUCT = ROOT / "02_closed_loop_bo" / "validation" / "dft_validation" / "structures"
ELEM = {  # distinct hues for elements that share a panel
    "Mo": "#2f9c95", "N": "#2f5fd0", "Os": "#6b8fb5", "W": "#d39b2a", "Re": "#8c2f39",
    "Co": "#e07a9a", "Ir": "#1d3557", "Ti": "#b8bcc4", "V": "#7d5ba6", "Ta": "#5fa8d3",
    "B": "#f2a48d",
}
NONMETAL = {"N", "B", "C", "P", "O"}


def _sprite(rgb, fade=0.0, n=200):
    """Shaded sphere as an RGBA image (light from the upper left)."""
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r2 = xx ** 2 + yy ** 2
    zz = np.sqrt(np.clip(1 - r2, 0, 1))
    L = np.array([-0.5, 0.6, 0.62]); L /= np.linalg.norm(L)
    diff = np.clip(xx * L[0] + yy * L[1] + zz * L[2], 0, 1)
    Hv = L + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
    spec = np.clip(xx * Hv[0] + yy * Hv[1] + zz * Hv[2], 0, 1) ** 45
    base = np.array(matplotlib.colors.to_rgb(rgb))
    base = base * (1 - fade) + fade * np.ones(3)           # far atoms fade toward white
    col = np.clip(base[None, None, :] * (0.40 + 0.70 * diff)[..., None] + 0.42 * spec[..., None], 0, 1)
    alpha = np.clip((1 - np.sqrt(r2)) * n / 3, 0, 1)
    return np.dstack([col, alpha])


def _block(atoms, min_len=5.0, reps=None):
    """Repeat to at least min_len per axis and add the periodic images on the faces."""
    reps = reps or [max(1, int(np.ceil(min_len / l))) for l in atoms.cell.lengths()]
    blk = atoms.repeat(reps)
    frac = blk.get_scaled_positions(wrap=True)
    P, S = [], []
    for f, s in zip(frac, blk.get_chemical_symbols()):
        opts = [[fi, fi + 1] if fi < 0.02 else ([fi, fi - 1] if fi > 0.98 else [fi]) for fi in f]
        for a in opts[0]:
            for b in opts[1]:
                for c in opts[2]:
                    P.append([a, b, c]); S.append(s)
    cell = np.array(blk.cell)
    return np.array(P) @ cell, S, cell


def _project(pos, cell, psi=-28.0, theta=17.0):
    """Longest block axis along screen x, then an oblique view. Returns X, Y, depth."""
    order = np.argsort(-np.linalg.norm(cell, axis=1))
    e1 = cell[order[0]] / np.linalg.norm(cell[order[0]])
    e2 = cell[order[1]] - (cell[order[1]] @ e1) * e1; e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    q = pos @ np.vstack([e1, e2, e3]).T
    ps, th = np.radians(psi), np.radians(theta)
    x1 = q[:, 0] * np.cos(ps) - q[:, 1] * np.sin(ps)
    y1 = q[:, 0] * np.sin(ps) + q[:, 1] * np.cos(ps)
    y2 = y1 * np.cos(th) - q[:, 2] * np.sin(th)
    z2 = y1 * np.sin(th) + q[:, 2] * np.cos(th)
    return x1, z2, -y2


def draw_structure(ax, pos, sym, cell, box=1.0, alpha=1.0, bonds=True, scale=1.0, lims=None,
                   blw=3.8, ratio=None, pad=0.9):
    from ase.data import atomic_numbers, covalent_radii
    rc = np.array([covalent_radii[atomic_numbers[s]] for s in sym])
    rad = np.where([s in NONMETAL for s in sym], 0.62, 0.55) * rc * scale
    X, Y, D = _project(pos, cell)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ cell
    cx, cy, cd = _project(corners, cell)
    items = []                                                  # (depth, kind, payload)
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(np.array([(a >> 2) & 1, (a >> 1) & 1, a & 1])
                             - np.array([(b >> 2) & 1, (b >> 1) & 1, b & 1]))) == 1:
                items.append(((cd[a] + cd[b]) / 2 - 0.4, "edge", (a, b)))
    if bonds:
        for i in range(len(sym)):
            for j in range(i + 1, len(sym)):
                if (sym[i] in NONMETAL) != (sym[j] in NONMETAL):
                    if np.linalg.norm(pos[i] - pos[j]) < 1.22 * (rc[i] + rc[j]):
                        items.append(((D[i] + D[j]) / 2, "bond", (i, j)))
    for i in range(len(sym)):
        items.append((D[i], "atom", i))
    dlo, dhi = D.min(), D.max()
    for z, (depth, kind, pl) in enumerate(sorted(items, key=lambda it: it[0])):
        if kind == "edge" and box > 0:
            a, b = pl
            ax.plot([cx[a], cx[b]], [cy[a], cy[b]], color="#8a8a8a", lw=1.5, alpha=box,
                    zorder=z, solid_capstyle="round")
        elif kind == "bond":
            i, j = pl
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#5a5a5a", lw=blw, alpha=alpha, zorder=z,
                    solid_capstyle="round")
            ax.plot([X[i], X[j]], [Y[i], Y[j]], color="#cfcfcf", lw=blw * 0.5, alpha=alpha, zorder=z,
                    solid_capstyle="round")
        elif kind == "atom":
            i = pl
            fade = 0.30 * (dhi - D[i]) / max(dhi - dlo, 1e-6)
            ax.imshow(_sprite(ELEM[sym[i]], fade), extent=(X[i] - rad[i], X[i] + rad[i],
                      Y[i] - rad[i], Y[i] + rad[i]), origin="lower", zorder=z, alpha=alpha,
                      interpolation="bilinear")
    if lims is None:
        lims = [min(X.min(), cx.min()) - pad, max(X.max(), cx.max()) + pad,
                min(Y.min(), cy.min()) - pad, max(Y.max(), cy.max()) + pad]
        if ratio:                                    # same window shape (w/h) for every panel
            hh = max((lims[1] - lims[0]) / ratio, lims[3] - lims[2]) / 2
            mx_, my_ = (lims[0] + lims[1]) / 2, (lims[2] + lims[3]) / 2
            lims = [mx_ - hh * ratio, mx_ + hh * ratio, my_ - hh, my_ + hh]
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal"); ax.axis("off")
    return lims


def fig_structures():
    """Six DFT-validated bulk-modulus structures found by gated runs."""
    from ase.io import read
    picks = [  # file stem, formula, space group, DFT K0 (GPa), backbone, repeats
        ("bm_top03_mg_ACC_seed113_Os5W3_sg65", "Os$_5$W$_3$", "Cmmm", 356, "MatterGen", None),
        ("bm_top01_adit_ACC_seed7_MoN_sg187", "MoN", "P$\\bar{6}$m2", 354, "ADiT", None),
        ("bm_top08_cf_ACC_seed99_Re5W_sg1", "Re$_5$W", "P1", 352, "CrystalFlow", None),
        ("bm_top05_mg_ACC_seed99_CoIrOs2_sg123", "CoIrOs$_2$", "P4/mmm", 346, "MatterGen", None),
        ("bm_top11_adit_ACC_seed17_TaB2W_sg38", "TaB$_2$W", "Amm2", 315, "ADiT", None),
        ("bm_top10_adit_ACC_seed99_TiVN2_sg160", "TiVN$_2$", "R3m", 296, "ADiT", (2, 1, 1)),
    ]
    dft = pd.read_csv(VAL / "esen_vs_dft_bm.csv")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.6))
    for ax, (stem, formula, sg, k, bb, reps) in zip(axes.ravel(), picks):
        rank = int(stem.split("_")[1].replace("top", ""))
        k_csv = dft.loc[dft["rank"] == rank, "dft_K0"].iloc[0]
        assert abs(k_csv - k) < 0.6, (stem, k_csv, k)
        pos, sym, cell = _block(read(STRUCT / f"{stem}.cif"), reps=reps)
        draw_structure(ax, pos, sym, cell, ratio=1.3, blw=3.4, pad=0.45)
        els = sorted(set(sym), key=lambda s: (s in NONMETAL, s))
        handles = [Line2D([], [], lw=0, marker="o", ms=14, mfc=ELEM[s], mec=DARK, mew=0.8, label=s) for s in els]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=len(els),
                  fontsize=19, handletextpad=0.05, columnspacing=0.8, borderpad=0, frameon=False)
        ax.text(0.5, -0.03, formula, transform=ax.transAxes, ha="center", va="top",
                fontsize=26, fontweight="bold", color=MAROON)
        ax.text(0.5, -0.24, f"{k} GPa  ·  {sg}", transform=ax.transAxes, ha="center", va="top",
                fontsize=21, color=DARK)
        print(f"  {stem:42s} atoms shown: {len(sym):3d}  K_DFT = {k_csv:.1f}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.105, wspace=0.03, hspace=0.62)
    save(fig, "fig_structures", png=True)


def fig_denoise():
    """Representative graphic: the generated MoN structure with decreasing positional noise
    (illustrates reverse diffusion; not a model trajectory)."""
    from ase.io import read
    atoms = read(STRUCT / "bm_top01_adit_ACC_seed7_MoN_sg187.cif")
    pos, sym, cell = _block(atoms)
    rng = np.random.default_rng(4)
    xi = rng.normal(size=pos.shape)
    lims = draw_structure(plt.figure().add_subplot(), pos, sym, cell); plt.close()
    lims = (lims[0] - 0.9, lims[1] + 0.9, lims[2] - 0.9, lims[3] + 0.9)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.75))
    for ax, sig, box, al in zip(axes, [1.25, 0.42, 0.0], [0.0, 0.35, 1.0], [0.60, 0.85, 1.0]):
        draw_structure(ax, pos + sig * xi, sym, cell, box=box, alpha=al, bonds=sig == 0.0, lims=lims, blw=3.0)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.16)
    for a, b in zip(axes[:-1], axes[1:]):
        xa = (a.get_position().x1 + b.get_position().x0) / 2
        fig.patches.append(matplotlib.patches.FancyArrowPatch((xa - 0.022, 0.5), (xa + 0.022, 0.5),
                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22,
                           color=MAROON, lw=2.4))
    save(fig, "fig_denoise", png=True)


TASKS = {"benchmark": fig_benchmark, "discovery": fig_discovery, "ablation": fig_ablation, "dft": fig_dft,
         "structures": fig_structures, "denoise": fig_denoise}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(TASKS)):
        print(f"[{name}]")
        TASKS[name]()
