"""Compare a baseline (LocalESEN) run against an accelerated (LocalESEN_GPRouted) run.

Produces two PNGs:
    1. speedup.png — best/mean Cp vs cumulative eSEN calls (the headline figure)
    2. cost.png    — per-cycle eSEN calls and wall-clock (the engineering claim)

Inputs (auto-discovered or passed as CLI args):
    --baseline-dir  upstream results dir of the baseline run
                    (contains rl.csv with per-cycle metrics + the LocalESEN log)
    --accel-dir     upstream results dir of the accelerated run
                    (also contains gp_routed_log.csv from LocalESEN_GPRouted)
    --output-dir    where to write the two PNGs

Run:
    python -m src.analyze.compare_runs \
        --baseline-dir results/hcap_p1_baseline_<JOB> \
        --accel-dir    results/hcap_p2_accel_<JOB> \
        --output-dir   results/compare_<TIMESTAMP>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_baseline_history(d: Path) -> pd.DataFrame:
    """Read upstream MatInvent's per-cycle CSV (logger=csv writes one).

    Tries common file names; falls back to scanning the dir.
    """
    candidates = [
        d / "rl.csv",
        d / "rl_history.csv",
        d / "csv" / "rl.csv",
        d / "metrics.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    csvs = list(d.rglob("*.csv"))
    if csvs:
        # Pick the one with most rows (likely the per-cycle log).
        best = max(csvs, key=lambda p: p.stat().st_size)
        return pd.read_csv(best)
    raise FileNotFoundError(f"No rl-history CSV under {d}")


def _load_gp_routing_log(root_dir: Path) -> Optional[pd.DataFrame]:
    """LocalESEN_GPRouted writes `gp_routed_log.csv` next to its calculator root_dir."""
    for p in (root_dir, *root_dir.rglob("gp_routed_log.csv")):
        if p.is_file() and p.name == "gp_routed_log.csv":
            return pd.read_csv(p)
    return None


def _running_max(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float64)
    cur = -np.inf
    for i, v in enumerate(arr):
        if np.isfinite(v) and v > cur:
            cur = v
        out[i] = cur
    return out


def _cumulative_cp(per_cycle_cp: np.ndarray, calls_per_cycle: np.ndarray):
    """Expand per-cycle (mean Cp, n_oracle_calls) into a step function in
    cumulative-call space."""
    cum_calls = np.cumsum(calls_per_cycle)
    return cum_calls, per_cycle_cp


def make_plots(
    base_df: pd.DataFrame,
    accel_df: pd.DataFrame,
    accel_log: Optional[pd.DataFrame],
    output_dir: Path,
    base_calls_per_cycle: int = 16,  # baseline = no skipping; eval_size
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-cycle Cp columns (upstream uses 'heat_capacity' or similar)
    cp_col = next(
        (c for c in ("heat_capacity_mean", "heat_capacity", "mean_heat_capacity",
                     "prop_heat_capacity")
         if c in base_df.columns),
        None,
    )
    if cp_col is None:
        # If we can't find Cp, fall back to 'reward'
        cp_col = next(
            (c for c in ("reward_mean", "reward") if c in base_df.columns), None,
        )
    if cp_col is None:
        raise ValueError(
            f"No Cp/reward column in baseline df; cols={list(base_df.columns)}"
        )
    print(f"[plot] using metric column: {cp_col}")

    # Per-cycle calls
    n_base = len(base_df)
    base_calls = np.full(n_base, base_calls_per_cycle, dtype=int)

    if accel_log is not None and "n_oracle" in accel_log.columns:
        accel_calls = accel_log["n_oracle"].to_numpy()
        accel_cp = (accel_log["mean_reward"]
                    if "mean_reward" in accel_log.columns
                    else accel_df[cp_col]).to_numpy()
    else:
        accel_calls = np.full(len(accel_df), base_calls_per_cycle, dtype=int)
        accel_cp = accel_df[cp_col].to_numpy()

    base_cp = base_df[cp_col].to_numpy()

    # ============================================================
    # Plot 1 — SPEEDUP: best Cp vs cumulative eSEN calls
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 5))
    base_cum, _ = _cumulative_cp(base_cp, base_calls)
    accel_cum, _ = _cumulative_cp(accel_cp, accel_calls)

    ax.plot(base_cum, _running_max(base_cp), "o-", color="C0",
            label=f"Baseline (LocalESEN) — {int(base_calls.sum())} calls",
            linewidth=2, markersize=6)
    ax.plot(accel_cum, _running_max(accel_cp), "s-", color="C1",
            label=f"Accelerated (GP-routed) — {int(accel_calls.sum())} calls",
            linewidth=2, markersize=6)
    ax.axhline(1.5, ls="--", color="grey", alpha=0.6, label="Target Cp = 1.5 J/g/K")
    ax.set_xlabel("Cumulative eSEN-30M-OAM calls", fontsize=12)
    ax.set_ylabel(f"Best {cp_col} so far  (J/g/K)", fontsize=12)
    ax.set_title("Heat-capacity acceleration: convergence vs oracle calls",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "speedup.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # Plot 2 — COST: per-cycle eSEN calls (bar)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    n_cycles = max(n_base, len(accel_calls))
    x = np.arange(n_cycles)
    w = 0.4
    base_pad = np.pad(base_calls, (0, n_cycles - n_base), constant_values=0)
    accel_pad = np.pad(accel_calls, (0, n_cycles - len(accel_calls)),
                       constant_values=0)
    ax.bar(x - w / 2, base_pad, width=w, color="C0", label="Baseline (LocalESEN)")
    ax.bar(x + w / 2, accel_pad, width=w, color="C1",
           label="Accelerated (GP-routed)")
    ax.set_xlabel("RL cycle", fontsize=12)
    ax.set_ylabel("eSEN-30M-OAM calls in this cycle", fontsize=12)
    ax.set_title("Per-cycle oracle cost", fontsize=13)
    ax.set_xticks(x)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "cost.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # Summary numbers
    # ============================================================
    n_base_calls = int(base_calls.sum())
    n_accel_calls = int(accel_calls.sum())
    speedup_calls = n_base_calls / max(1, n_accel_calls)
    summary = {
        "metric_column": cp_col,
        "baseline_total_calls": n_base_calls,
        "accelerated_total_calls": n_accel_calls,
        "speedup_factor": float(speedup_calls),
        "baseline_best_cp": float(np.nanmax(base_cp)),
        "accelerated_best_cp": float(np.nanmax(accel_cp)),
        "baseline_final_mean_cp": float(np.nanmean(base_cp[-3:])),
        "accelerated_final_mean_cp": float(np.nanmean(accel_cp[-3:])),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--accel-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--accel-calc-root", default=None,
        help="path to LocalESEN_GPRouted's root_dir (defaults to accel-dir/rewards/heat_capacity)"
    )
    args = p.parse_args()

    base = Path(args.baseline_dir)
    accel = Path(args.accel_dir)
    out = Path(args.output_dir)

    base_df = _load_baseline_history(base)
    accel_df = _load_baseline_history(accel)

    accel_log_root = (
        Path(args.accel_calc_root) if args.accel_calc_root else accel / "rewards"
    )
    accel_log = _load_gp_routing_log(accel_log_root)
    if accel_log is None:
        # fall back: search whole accel dir
        accel_log = _load_gp_routing_log(accel)
    print(
        f"[plot] base={len(base_df)} rows; accel={len(accel_df)} rows; "
        f"accel_log={'YES (' + str(len(accel_log)) + ' rows)' if accel_log is not None else 'NO'}"
    )

    make_plots(base_df, accel_df, accel_log, out)


if __name__ == "__main__":
    main()
