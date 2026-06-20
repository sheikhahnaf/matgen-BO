"""Plot RL training + GP-fit metrics across cycles for all v4 5-seed runs.

Produces per-property plots:
  - reward_vs_cycle.png        (mean ± seed-band)
  - property_vs_cycle.png      (Cp or K_VRH best-so-far per cycle)
  - stability_vs_cycle.png     (frac_stable, frac_novel_unique_stable)
  - validity_vs_cycle.png
  - diversity_vs_cycle.png
  - gp_fit_quality.png         (cv5 RMSE/MAE over cycles, ACC only)
  - oracle_split.png           (n_oracle vs n_gp per cycle, ACC only)
  - filter_survival.png        (n_input over cycles, both BASE/ACC)

Output dirs:
  cp/   — Cp jobs
  bm/   — Bulk modulus jobs
  combined/ — cross-property summary panels
"""
import re
from pathlib import Path
import os as _os
from pathlib import Path as _Path
# Repo-relative defaults; override with MBO_REPO_ROOT / MBO_FIG_DIR / MBO_RESULTS_ROOT.
_REPO = _Path(_os.environ.get("MBO_REPO_ROOT", _Path(__file__).resolve().parent.parent))
_FIGS = _Path(_os.environ.get("MBO_FIG_DIR", _REPO / "figures"))
_RES = _Path(_os.environ.get("MBO_RESULTS_ROOT", _REPO / "hcap_bo" / "results-paper-v4"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = _RES
OUT = _Path(_os.environ.get("MBO_FIG_DIR", _REPO / "hcap_bo" / "analysis" / "v4_metrics_plots"))
SEEDS = [17, 99, 7, 23, 113]
PARADIGMS = ["mg", "cf", "adit"]
PARADIGM_COLORS = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}
SETUP_LS = {"BASE": "--", "ACC": "-"}

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def discover(prop: str):
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
    return pd.DataFrame(rows)


def load_metrics(jobs: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return tidy long DataFrame: paradigm, setup, seed, cycle, value."""
    out = []
    for _, r in jobs.iterrows():
        if not r.metrics.exists():
            continue
        try:
            df = pd.read_csv(r.metrics)
        except Exception:
            continue
        if col not in df.columns:
            continue
        for cyc, val in zip(df["step"], df[col]):
            out.append({"paradigm": r.paradigm, "setup": r.setup, "seed": r.seed,
                        "cycle": int(cyc), "value": val})
    return pd.DataFrame(out)


def load_gp_log(jobs: pd.DataFrame, col: str) -> pd.DataFrame:
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
                        "cycle": int(cyc), "value": val})
    return pd.DataFrame(out)


def plot_metric_panel(df: pd.DataFrame, title: str, ylabel: str, fname: Path,
                      target=None, smooth=False):
    """One axis per paradigm, two lines per axis (BASE vs ACC), shaded seed band."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, paradigm in zip(axes, PARADIGMS):
        sub = df[df.paradigm == paradigm]
        for setup in ["BASE", "ACC"]:
            s = sub[sub.setup == setup]
            if len(s) == 0:
                continue
            agg = s.groupby("cycle")["value"].agg(["mean", "std", "count"])
            x = agg.index.values
            y = agg["mean"].values
            yerr = agg["std"].fillna(0).values
            color = PARADIGM_COLORS[paradigm]
            ax.plot(x, y, ls=SETUP_LS[setup], color=color, lw=1.8,
                    label=setup, marker="o" if setup == "ACC" else "s", ms=4)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)
        if target is not None:
            ax.axhline(target, color="gray", ls=":", lw=1, alpha=0.7, label=f"target={target}")
        ax.set_title(f"{paradigm}", color=PARADIGM_COLORS[paradigm], fontweight="bold")
        ax.set_xlabel("RL cycle")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(fname)
    plt.close(fig)
    print(f"  → {fname.name}")


def plot_gp_fit(jobs: pd.DataFrame, prop_label: str, outdir: Path):
    """GP fit quality (CV5 RMSE/MAE) over cycles, ACC only, per paradigm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    metric_cols = [("gp_rmse_cv5", "GP CV5 RMSE"), ("gp_mae_cv5", "GP CV5 MAE")]
    for ax, (col, label) in zip(axes, metric_cols):
        df = load_gp_log(jobs, col)
        for paradigm in PARADIGMS:
            s = df[df.paradigm == paradigm]
            if len(s) == 0:
                continue
            agg = s.groupby("cycle")["value"].agg(["mean", "std"])
            x = agg.index.values
            y = agg["mean"].values
            yerr = agg["std"].fillna(0).values
            color = PARADIGM_COLORS[paradigm]
            ax.plot(x, y, color=color, lw=1.8, marker="o", ms=4, label=paradigm)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)
        ax.set_xlabel("RL cycle")
        ax.set_ylabel(f"{label} (normalized {prop_label})")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"GP-fit quality across cycles — ACC only ({prop_label})",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.savefig(outdir / "gp_fit_quality.png")
    plt.close(fig)
    print(f"  → gp_fit_quality.png")


def plot_oracle_split(jobs: pd.DataFrame, prop_label: str, outdir: Path):
    """n_oracle vs n_gp per cycle (ACC only)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, paradigm in zip(axes, PARADIGMS):
        sub_jobs = jobs[(jobs.paradigm == paradigm) & (jobs.setup == "ACC")]
        n_o, n_g = [], []
        for _, r in sub_jobs.iterrows():
            if not r.gp_log.exists():
                continue
            try:
                df = pd.read_csv(r.gp_log)
            except Exception:
                continue
            for col, store in (("n_oracle", n_o), ("n_gp", n_g)):
                if col in df.columns:
                    for cyc, v in zip(df["cycle"], df[col]):
                        store.append({"cycle": int(cyc), "v": v})
        if not n_o:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        do = pd.DataFrame(n_o).groupby("cycle")["v"].mean()
        dg = pd.DataFrame(n_g).groupby("cycle")["v"].mean()
        x = do.index.values
        ax.bar(x - 0.18, do.values, width=0.36, color=PARADIGM_COLORS[paradigm],
               label="n_oracle (top-K)", alpha=0.85)
        ax.bar(x + 0.18, dg.values, width=0.36, color=PARADIGM_COLORS[paradigm],
               alpha=0.35, hatch="//", label="n_gp (μ-only)")
        ax.set_title(paradigm, color=PARADIGM_COLORS[paradigm], fontweight="bold")
        ax.set_xlabel("RL cycle")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("# samples")
    fig.suptitle(f"Oracle vs GP split per cycle — ACC ({prop_label})",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(outdir / "oracle_split.png")
    plt.close(fig)
    print(f"  → oracle_split.png")


def plot_best_running(jobs: pd.DataFrame, prop_label: str, prop_col: str,
                      minv: float, maxv: float, target_str: str, outdir: Path):
    """Running best property value per cycle (denormalized from reward)."""
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
        if len(df) == 0:
            continue
        df["prop"] = df["reward"] * (maxv - minv) + minv
        df.loc[df.reward >= 1.0, "prop"] = maxv
        if prop_label == "Cp":
            df["score"] = -np.abs(df["prop"] - 1.5)  # higher = closer to target 1.5
            df["best_so_far"] = df.groupby("RL_step")["score"].transform("max")
            running_best = df.groupby("RL_step")["score"].max().expanding().max()
            running_best_prop = 1.5 - np.abs(running_best.values)
        else:
            df["best_so_far"] = df.groupby("RL_step")["prop"].transform("max")
            running_best_prop = df.groupby("RL_step")["prop"].max().expanding().max().values
        cycles = df.groupby("RL_step")["prop"].max().index.values
        for c, p in zip(cycles, running_best_prop):
            out.append({"paradigm": r.paradigm, "setup": r.setup, "seed": r.seed,
                        "cycle": int(c), "best": p})
    df = pd.DataFrame(out)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, paradigm in zip(axes, PARADIGMS):
        sub = df[df.paradigm == paradigm]
        for setup in ["BASE", "ACC"]:
            s = sub[sub.setup == setup]
            if len(s) == 0:
                continue
            agg = s.groupby("cycle")["best"].agg(["mean", "std", "count"])
            x = agg.index.values
            y = agg["mean"].values
            yerr = agg["std"].fillna(0).values
            color = PARADIGM_COLORS[paradigm]
            ax.plot(x, y, ls=SETUP_LS[setup], color=color, lw=1.8,
                    label=setup, marker="o" if setup == "ACC" else "s", ms=4)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)
        if prop_label == "Cp":
            ax.axhline(1.5, color="gray", ls=":", lw=1, alpha=0.7, label="target=1.5")
        ax.set_title(paradigm, color=PARADIGM_COLORS[paradigm], fontweight="bold")
        ax.set_xlabel("RL cycle")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    axes[0].set_ylabel(f"running best {prop_col} ({target_str})")
    fig.suptitle(f"Running best {prop_label} across cycles (5-seed mean ± std)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.savefig(outdir / "best_running.png")
    plt.close(fig)
    print(f"  → best_running.png")


def main():
    for prop, outdir in [("Cp", OUT / "cp"), ("BM", OUT / "bm")]:
        prop_label = "Cp" if prop == "Cp" else "K_VRH"
        prop_unit = "[J/g/K]" if prop == "Cp" else "[GPa]"
        target = 1.5 if prop == "Cp" else None
        minv, maxv = (0.25, 2.0) if prop == "Cp" else (20.0, 400.0)
        target_str = "→ 1.5 J/g/K" if prop == "Cp" else "max"

        print(f"\n=== {prop} ===")
        jobs = discover(prop)
        # dedupe: keep most recent jobid per (paradigm, setup, seed)
        jobs = jobs.sort_values("jobid").drop_duplicates(
            subset=["paradigm", "setup", "seed"], keep="last")
        print(f"  jobs: {len(jobs)}")

        plot_metric_panel(load_metrics(jobs, "reward mean"),
                          f"Reward (mean) per cycle — {prop}",
                          "reward [0..1]", outdir / "reward_vs_cycle.png")
        prop_col = "heat_capacity mean" if prop == "Cp" else None
        if prop_col and any(prop_col in pd.read_csv(j.metrics).columns
                            for _, j in jobs.iterrows() if j.metrics.exists()):
            plot_metric_panel(load_metrics(jobs, prop_col),
                              f"{prop_label} (cycle mean) — {prop}",
                              f"{prop_label} {prop_unit}",
                              outdir / "property_mean_vs_cycle.png", target=target)
        plot_metric_panel(load_metrics(jobs, "frac_stable_structures"),
                          f"frac stable per cycle — {prop}",
                          "frac stable", outdir / "frac_stable_vs_cycle.png")
        plot_metric_panel(load_metrics(jobs, "frac_novel_unique_stable_structures"),
                          f"frac NOVEL+UNIQUE+STABLE per cycle — {prop}",
                          "frac NUS", outdir / "frac_nus_vs_cycle.png")
        plot_metric_panel(load_metrics(jobs, "avg_structure_validity"),
                          f"avg structure validity per cycle — {prop}",
                          "validity", outdir / "validity_vs_cycle.png")
        plot_metric_panel(load_metrics(jobs, "div_ratio"),
                          f"diversity ratio per cycle — {prop}",
                          "div_ratio", outdir / "diversity_vs_cycle.png")
        plot_metric_panel(load_metrics(jobs, "avg_energy_above_hull_per_atom"),
                          f"E above hull per atom per cycle — {prop}",
                          "E_hull [eV/atom]", outdir / "ehull_vs_cycle.png")
        plot_metric_panel(load_metrics(jobs, "avg_rmsd_from_relaxation"),
                          f"RMSD-from-relaxation per cycle — {prop}",
                          "RMSD [Å]", outdir / "rmsd_vs_cycle.png")
        plot_gp_fit(jobs, prop_label, outdir)
        plot_oracle_split(jobs, prop_label, outdir)
        plot_best_running(jobs, prop, prop_label, minv, maxv, target_str, outdir)

    print("\nDone.")
    print(f"All plots in: {OUT}")


if __name__ == "__main__":
    main()
