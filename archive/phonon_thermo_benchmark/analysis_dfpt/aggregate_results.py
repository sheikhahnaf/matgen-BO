"""
Data Aggregation Script for Phonon-Thermo Benchmark -- Arm A (DFPT reference)

This script loads all *_holdout_summary.csv files from the GP, MTGP_2, and DGP
result directories tagged with the 'dfpt' dataset prefix and creates a unified
DataFrame for downstream analysis.

It is target-agnostic: property names are read dynamically from the CSV
'Property' column (Cv_300K, S_300K, F_300K, max_phonon_freq), and the config
(model / descriptor / PCA / n_train) is parsed from the directory/file naming
convention results/<model>/dfpt_pca<P>_n<N>/<model>_<descriptor>_holdout_summary.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Tuple

# Arm A dataset tag. Result directories are named e.g. 'dfpt_pca10_n500'.
DATASET_PREFIX = 'dfpt'

# Arm A surrogate models (DGP IS evaluated on Arm A).
MODELS = ['gp', 'mtgp_2', 'dgp']

# Coverage grid for the missing-combination report (Arm A).
# These only affect the coverage report, NOT the aggregated metrics.
DESCRIPTORS = ['soap', 'mace', 'orb', 'uma']
PCA_VALUES = [10, 25, 50]
# n_train grid is per-model (kept tolerant): if a model has no entry here it
# falls back to N_TRAIN_DEFAULT.
N_TRAIN_BY_MODEL = {
    'gp': [100, 250, 500],
    'mtgp_2': [100, 250, 500],
    'dgp': [100, 250, 500],
}
N_TRAIN_DEFAULT = [100, 250, 500]


def parse_config_from_path(csv_path: Path) -> Dict[str, str]:
    """
    Extract configuration parameters from CSV file path.

    Example paths:
        results/gp/dfpt_pca10_n100/gp_soap_holdout_summary.csv
        results/mtgp_2/dfpt_pca25_n250/mtgp_mace_holdout_summary.csv
        results/dgp/dfpt_pca50_n500/dgp_orb_holdout_summary.csv

    Returns:
        dict with keys: model, descriptor, pca_components, n_train
    """
    parts = csv_path.parts

    # Extract model type
    if 'mtgp_2' in parts:
        model = 'mtgp_2'
    elif 'mtgp' in parts:
        model = 'mtgp'
    elif 'dgp' in parts:
        model = 'dgp'
    elif 'gp' in parts:
        model = 'gp'
    else:
        raise ValueError(f"Unknown model in path: {csv_path}")

    # Extract descriptor from filename
    filename = csv_path.stem  # e.g., 'gp_soap_holdout_summary'
    desc_match = re.search(r'_(soap|mace|orb|uma)_', filename)
    if desc_match:
        descriptor = desc_match.group(1)
    else:
        raise ValueError(f"Cannot extract descriptor from: {filename}")

    # Extract PCA and n_train from directory name
    dir_name = csv_path.parent.name  # e.g., 'dfpt_pca10_n100'
    pca_match = re.search(r'pca(\d+)', dir_name)
    n_match = re.search(r'n(\d+)', dir_name)

    if not pca_match or not n_match:
        raise ValueError(f"Cannot extract PCA/n_train from: {dir_name}")

    return {
        'model': model,
        'descriptor': descriptor,
        'pca_components': int(pca_match.group(1)),
        'n_train': int(n_match.group(1))
    }


def load_single_csv(csv_path: Path) -> pd.DataFrame:
    """Load a single CSV and reshape from wide to long format."""
    df = pd.read_csv(csv_path)

    # Extract config from path
    config = parse_config_from_path(csv_path)

    # Reshape: each row is (property, metric, mean, std)
    metrics = ['R2', 'RMSE', 'MAE', 'SMAPE', 'Spearman']

    records = []
    for _, row in df.iterrows():
        property_name = row['Property']

        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'

            if mean_col in df.columns and std_col in df.columns:
                records.append({
                    **config,
                    'property': property_name,
                    'metric': metric,
                    'mean': row[mean_col],
                    'std': row[std_col],
                    'config': f"{config['model']}_{config['descriptor']}_pca{config['pca_components']}_n{config['n_train']}"
                })

    return pd.DataFrame(records)


def load_all_results(results_dir: Path, models: List[str] = None) -> pd.DataFrame:
    """
    Load all *_holdout_summary.csv files into a single DataFrame.

    Args:
        results_dir: Path to results directory
        models: List of model subdirectories to include (default: MODELS)

    Returns:
        Unified DataFrame with all results
    """
    if models is None:
        models = MODELS

    all_dfs = []

    for model in models:
        model_dir = results_dir / model
        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping...")
            continue

        csv_files = list(model_dir.glob('**/*_holdout_summary.csv'))
        # Only keep CSVs whose parent directory is tagged with the dataset
        # prefix (e.g. 'dfpt_pca10_n500'). This excludes smoke-test CSVs that
        # sit directly under the model dir with no pca/n tag.
        csv_files = [f for f in csv_files if f.parent.name.startswith(DATASET_PREFIX)]
        print(f"Found {len(csv_files)} CSV files for model '{model}' (dataset: {DATASET_PREFIX})")

        for csv_path in csv_files:
            try:
                df = load_single_csv(csv_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")

    if not all_dfs:
        raise ValueError("No CSV files loaded successfully!")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Add derived columns
    combined = compute_derived_metrics(combined)

    return combined


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analysis metrics."""
    df = df.copy()

    # Coefficient of variation (only for metrics with positive mean)
    df['cv'] = np.where(
        df['mean'] > 0,
        df['std'] / df['mean'],
        np.nan
    )

    # Relative error (std as percentage of mean, for RMSE/MAE)
    df['relative_std'] = df['std'] / (df['mean'].abs() + 1e-10) * 100

    return df


def filter_top_properties(df: pd.DataFrame, n_properties: int = 5,
                         metric: str = 'R2') -> pd.DataFrame:
    """
    Focus analysis on top N best-predicted properties.

    Args:
        df: Full dataframe
        n_properties: Number of properties to keep
        metric: Metric to rank by (default: R2)

    Returns:
        Filtered dataframe
    """
    # Find average performance per property
    property_performance = (
        df[df['metric'] == metric]
        .groupby('property')['mean']
        .mean()
        .sort_values(ascending=False)
    )

    top_properties = property_performance.head(n_properties).index.tolist()

    return df[df['property'].isin(top_properties)].copy()


def identify_missing_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report which model configurations are incomplete.

    The expected grid is per-model (N_TRAIN_BY_MODEL) so that arms with
    model-specific n_train ranges report coverage correctly. Only models that
    are actually present in the data are checked, so a partial run does not
    flag every model as missing.
    """
    expected_combinations = []

    models = df['model'].unique()

    for model in models:
        n_train_values = N_TRAIN_BY_MODEL.get(model, N_TRAIN_DEFAULT)
        for desc in DESCRIPTORS:
            for pca in PCA_VALUES:
                for n in n_train_values:
                    expected_combinations.append({
                        'model': model,
                        'descriptor': desc,
                        'pca_components': pca,
                        'n_train': n
                    })

    expected_df = pd.DataFrame(expected_combinations)

    # Find which configs actually exist
    actual_configs = df[['model', 'descriptor', 'pca_components', 'n_train']].drop_duplicates()

    # Merge to find missing
    merged = expected_df.merge(
        actual_configs,
        on=['model', 'descriptor', 'pca_components', 'n_train'],
        how='left',
        indicator=True
    )

    missing = merged[merged['_merge'] == 'left_only'].drop(columns='_merge')

    return missing


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generate a text summary of data coverage."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"PHONON-THERMO BENCHMARK ({DATASET_PREFIX.upper()}) - DATA SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Overall stats
    total_configs = df['config'].nunique()
    total_properties = df['property'].nunique()
    total_records = len(df)

    lines.append(f"Total unique configurations: {total_configs}")
    lines.append(f"Total properties: {total_properties}")
    lines.append(f"Total records: {total_records}")
    lines.append("")

    # Breakdown by model
    lines.append("Breakdown by model:")
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        n_configs = model_df['config'].nunique()
        lines.append(f"  {model:10s}: {n_configs:3d} configurations")
    lines.append("")

    # Breakdown by descriptor
    lines.append("Breakdown by descriptor:")
    for desc in sorted(df['descriptor'].unique()):
        desc_df = df[df['descriptor'] == desc]
        n_configs = desc_df['config'].nunique()
        lines.append(f"  {desc:10s}: {n_configs:3d} configurations")
    lines.append("")

    # Breakdown by PCA
    lines.append("Breakdown by PCA components:")
    for pca in sorted(df['pca_components'].unique()):
        pca_df = df[df['pca_components'] == pca]
        n_configs = pca_df['config'].nunique()
        lines.append(f"  PCA={pca:3d}: {n_configs:3d} configurations")
    lines.append("")

    # Breakdown by n_train
    lines.append("Breakdown by training size:")
    for n in sorted(df['n_train'].unique()):
        n_df = df[df['n_train'] == n]
        n_configs = n_df['config'].nunique()
        lines.append(f"  n={n:4d}: {n_configs:3d} configurations")
    lines.append("")

    # Properties list
    lines.append("Properties analyzed:")
    for prop in sorted(df['property'].unique()):
        lines.append(f"  - {prop}")
    lines.append("")

    # Missing configurations
    missing = identify_missing_combinations(df)
    if len(missing) > 0:
        lines.append(f"MISSING CONFIGURATIONS: {len(missing)}")
        lines.append("")
        for _, row in missing.iterrows():
            lines.append(f"  - {row['model']:10s} {row['descriptor']:4s} "
                        f"PCA={row['pca_components']:2d} n={row['n_train']:3d}")
    else:
        lines.append("All expected configurations are present!")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Write to file
    output_path.write_text(report)

    # Also print to console
    print(report)

    return report


def main():
    """Main aggregation pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    output_dir = Path(__file__).parent  # save in this script's directory

    print(f"Starting data aggregation for dataset: {DATASET_PREFIX}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load all results
    df = load_all_results(results_dir, models=MODELS)

    print(f"\nLoaded {len(df)} records from {df['config'].nunique()} configurations")
    print(f"DataFrame shape: {df.shape}")
    print()

    # Save aggregated results
    output_csv = output_dir / 'aggregated_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"Saved aggregated results to: {output_csv}")
    print()

    # Generate summary report
    summary_path = output_dir / 'data_summary.txt'
    generate_summary_report(df, summary_path)
    print(f"\nSaved summary report to: {summary_path}")

    # Show sample of data
    print("\nSample of aggregated data:")
    print(df.head(10).to_string())

    return df


if __name__ == '__main__':
    df = main()
