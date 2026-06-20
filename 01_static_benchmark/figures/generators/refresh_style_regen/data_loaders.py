"""CSV loaders for the three FME-paper datasets.

The original analysis lives at:
    ASE_regression_test/analysis_v3/                       (elastic)
    ASE_regression_test/analysis_v3_dielectric_constant/   (dielectric)
    ASE_regression_test/analysis_v3_phonon_dielectric_mp/  (phonon)

Each has the same schema:
    aggregated_results.csv          (model, descriptor, pca_components, n_train, property, metric, mean, std, ...)
    n500/data/filtered_n500.csv     (same schema, only n_train=500)
    combined/data/learning_curves_orb.csv  (per-property R2 / RMSE / Spearman across n_train)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/Volumes/SSD1_SMAAA/matinvent-bo")
ANALYSIS_BASE = REPO / "ASE_regression_test"
# Phonon is now the new DFPT phonon-thermodynamics benchmark (replaces phonon-dielectric).
DFPT = REPO / "phonon_thermo_benchmark" / "paper_figures_new_phonon_2026-06-18" / "arm_a_dfpt"

DATASET_DIR = {
    "elastic": ANALYSIS_BASE / "analysis_v3",
    "dielectric": ANALYSIS_BASE / "analysis_v3_dielectric_constant",
    "phonon": DFPT,
}

DATASET_LABEL = {
    "elastic": "Elastic tensor (8 properties)",
    "dielectric": "Dielectric constant (4 properties)",
    "phonon": "Phonon thermodynamics (4 properties)",
}


def load_aggregated(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR[dataset] / "aggregated_results.csv")


def load_filtered_n500(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR[dataset] / "n500" / "data" / "filtered_n500.csv")


def load_learning_curves_orb(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR[dataset] / "combined" / "data" / "learning_curves_orb.csv")


def best_pca_per_surrogate(filtered_n500: pd.DataFrame, optimize_for: str = "R2") -> pd.DataFrame:
    """Replicates ASE_regression_test/analysis_v3/n500/scripts/bar_charts_averaged.py logic.

    For each (model, descriptor), pick the PCA dim that maximises mean metric across
    all properties at n_train=500. Returns columns:
      model, descriptor, best_pca, avg_R2, avg_RMSE, avg_Spearman
    """
    df = filtered_n500
    rows = []
    models = df["model"].unique()
    descriptors = df["descriptor"].unique()
    pcas = sorted(df["pca_components"].unique())
    metrics = ["R2", "RMSE", "Spearman"]

    for model in models:
        for desc in descriptors:
            sub = df[(df["model"] == model) & (df["descriptor"] == desc)]
            pca_perf: dict[int, dict[str, float]] = {}
            for pca in pcas:
                pca_data = sub[sub["pca_components"] == pca]
                pca_perf[pca] = {
                    m: pca_data[pca_data["metric"] == m]["mean"].mean()
                    for m in metrics
                }
            if optimize_for == "RMSE":
                best_pca = min(pca_perf, key=lambda p: pca_perf[p][optimize_for])
            else:
                best_pca = max(pca_perf, key=lambda p: pca_perf[p][optimize_for])
            row = {"model": model, "descriptor": desc, "best_pca": int(best_pca)}
            for m in metrics:
                row[f"avg_{m}"] = pca_perf[best_pca][m]
            rows.append(row)
    return pd.DataFrame(rows)
