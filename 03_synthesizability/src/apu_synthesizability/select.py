"""Leaderboard aggregation and best-config selection for APU synthesizability sweeps.

Each result JSON produced by a sweep config is expected to contain at minimum:
    name, deployable, proxy_auprc, ece
Additional optional columns (arch, features, proxy_auroc, tpr_on_labeled,
proxy_precision, proxy_recall, pu_score, n_planted, n_test, n_train) are
included if present.

CLI usage:
    python -m apu_synthesizability.select \\
        --results-dir results/apu \\
        --leaderboard results/apu/leaderboard.csv \\
        --best results/apu/best.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# Preferred column order for CSV output (missing cols are appended at end).
_KEY_COLS = [
    "name",
    "arch",
    "features",
    "deployable",
    "proxy_auprc",
    "proxy_auroc",
    "ece",
    "tpr_on_labeled",
    "proxy_precision",
    "proxy_recall",
    "pu_score",
]


def load_results(results_dir: str) -> pd.DataFrame:
    """Read every ``*.json`` file in *results_dir* into a DataFrame.

    Each JSON must be a flat dict (one config per file).  An empty directory
    returns an empty DataFrame.

    Parameters
    ----------
    results_dir:
        Path to the directory containing result JSON files.

    Returns
    -------
    pd.DataFrame
        One row per config.  An empty DataFrame is returned when no JSON
        files are found.
    """
    paths = sorted(Path(results_dir).glob("*.json"))
    if not paths:
        return pd.DataFrame()

    records = []
    for p in paths:
        with p.open() as fh:
            records.append(json.load(fh))

    return pd.DataFrame(records)


def write_leaderboard(df: pd.DataFrame, csv_path: str) -> None:
    """Sort *df* by ``proxy_auprc`` desc (ties broken by ``ece`` asc) and
    write to *csv_path*.

    The ``features`` column (a list) is stringified so the CSV round-trips
    cleanly.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_results`.
    csv_path:
        Destination CSV path (parent directory must exist).
    """
    if df.empty:
        pd.DataFrame().to_csv(csv_path, index=False)
        return

    out = df.copy()

    # Stringify the features list if present so CSV is valid.
    if "features" in out.columns:
        out["features"] = out["features"].apply(
            lambda v: str(v) if isinstance(v, list) else v
        )

    # Sort: primary key AUPRC descending, secondary key ECE ascending.
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "proxy_auprc" in out.columns:
        sort_cols.append("proxy_auprc")
        ascending.append(False)
    if "ece" in out.columns:
        sort_cols.append("ece")
        ascending.append(True)

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    # Reorder columns: key cols first, then any remaining cols.
    present_key = [c for c in _KEY_COLS if c in out.columns]
    extra = [c for c in out.columns if c not in present_key]
    out = out[present_key + extra]

    out.to_csv(csv_path, index=False)


def select_best(df: pd.DataFrame) -> dict:
    """Return the best *deployable* config as a dict.

    Selection criterion: filter to ``deployable == True``, then sort by
    ``(-proxy_auprc, ece)`` ascending, return the first row.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_results`.

    Returns
    -------
    dict
        The winning config row.

    Raises
    ------
    ValueError
        If no deployable configs exist in *df*.
    """
    if df.empty or "deployable" not in df.columns:
        raise ValueError("No deployable configs found (DataFrame is empty or missing 'deployable' column).")

    deployable = df[df["deployable"] == True].copy()  # noqa: E712

    if deployable.empty:
        raise ValueError("No deployable configs found (all configs have deployable=False).")

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "proxy_auprc" in deployable.columns:
        sort_cols.append("proxy_auprc")
        ascending.append(False)
    if "ece" in deployable.columns:
        sort_cols.append("ece")
        ascending.append(True)

    if sort_cols:
        deployable = deployable.sort_values(sort_cols, ascending=ascending)

    return deployable.iloc[0].to_dict()


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry-point for aggregation and selection."""
    parser = argparse.ArgumentParser(
        description="Aggregate APU sweep results and select the best deployable config."
    )
    parser.add_argument(
        "--results-dir",
        default="results/apu",
        help="Directory containing per-config result JSON files.",
    )
    parser.add_argument(
        "--leaderboard",
        default="results/apu/leaderboard.csv",
        help="Output path for the sorted leaderboard CSV.",
    )
    parser.add_argument(
        "--best",
        default="results/apu/best.json",
        help="Output path for the best-config JSON.",
    )
    args = parser.parse_args(argv)

    df = load_results(args.results_dir)
    if df.empty:
        print("No result files found in", args.results_dir)
        return

    # Write leaderboard.
    write_leaderboard(df, args.leaderboard)
    print(f"Leaderboard written to {args.leaderboard} ({len(df)} configs).")

    # Select best.
    best = select_best(df)
    Path(args.best).write_text(json.dumps(best, indent=2, default=str))
    print(f"Best config written to {args.best}.")
    print(
        f"Winner: {best.get('name')} | "
        f"AUPRC={best.get('proxy_auprc')} | "
        f"ECE={best.get('ece')} | "
        f"arch={best.get('arch')} | "
        f"features={best.get('features')}"
    )


if __name__ == "__main__":
    main()
