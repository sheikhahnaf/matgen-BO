"""
Phase 0: Data Preparation for Simplified ASE Regression Analysis (n=500 only)

This script:
1. Loads aggregated results from previous analysis
2. Filters to n=500 only (since DGP only has this dataset size)
3. Removes kpoint_density property
4. Keeps only R², RMSE, Spearman metrics
5. Saves filtered data for subsequent analyses
"""

import pandas as pd
import os

def get_property_list():
    """Return 3 properties for phonon_dielectric_mp dataset"""
    return [
        'eps_electronic', 'eps_total', 'last phdos peak'
    ]

def load_and_filter_data():
    """
    Load aggregated_results.csv and filter to:
    - n_train = 500 only
    - Remove kpoint_density
    - Keep R², RMSE, Spearman only

    Returns:
        DataFrame with filtered data
    """
    # Load data from previous analysis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', '..', 'aggregated_results.csv')

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find {input_path}. "
            "Make sure the previous analysis has been run."
        )

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique n_train values: {df['n_train'].unique()}")
    print(f"Unique properties: {df['property'].unique()}")
    print(f"Unique metrics: {df['metric'].unique()}")

    # Filter to n=500
    df = df[df['n_train'] == 500].copy()
    print(f"\nAfter filtering to n=500: {df.shape}")

    # Remove kpoint_density
    df = df[df['property'] != 'kpoint_density'].copy()
    print(f"After removing kpoint_density: {df.shape}")

    # Keep only R², RMSE, Spearman
    df = df[df['metric'].isin(['R2', 'RMSE', 'Spearman'])].copy()
    print(f"After filtering metrics: {df.shape}")

    # Verify we have all expected properties
    properties = get_property_list()
    actual_properties = sorted(df['property'].unique())
    expected_properties = sorted(properties)

    print(f"\nExpected properties: {expected_properties}")
    print(f"Actual properties: {actual_properties}")

    if actual_properties != expected_properties:
        print("WARNING: Property list mismatch!")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("FILTERED DATA SUMMARY")
    print('='*80)
    print(f"Total rows: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Descriptors: {sorted(df['descriptor'].unique())}")
    print(f"PCA components: {sorted(df['pca_components'].unique())}")
    print(f"Properties: {len(df['property'].unique())} properties")
    print(f"Metrics: {sorted(df['metric'].unique())}")

    # Check for missing combinations
    expected_rows = 3 * 4 * 3 * 3 * 3  # models × descriptors × PCA × 3 properties × metrics
    print(f"\nExpected rows: {expected_rows}")
    print(f"Actual rows: {len(df)}")

    if len(df) != expected_rows:
        print("WARNING: Missing data!")

        # Check which combinations are missing
        from itertools import product

        models = ['gp', 'mtgp_2', 'dgp']
        descriptors = ['mace', 'orb', 'soap', 'uma']
        pcas = [10, 25, 50]
        metrics = ['R2', 'RMSE', 'Spearman']

        all_combos = set(product(models, descriptors, pcas, properties, metrics))
        actual_combos = set(
            zip(df['model'], df['descriptor'], df['pca_components'],
                df['property'], df['metric'])
        )

        missing = all_combos - actual_combos
        if missing:
            print(f"\nMissing {len(missing)} combinations:")
            for combo in sorted(missing)[:10]:  # Show first 10
                print(f"  {combo}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")

    return df

def save_filtered_data(df):
    """Save filtered data to CSV"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'filtered_n500.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved filtered data to {output_path}")
    print(f"  Shape: {df.shape}")

def main():
    print("="*80)
    print("PHASE 0: DATA PREPARATION")
    print("="*80)

    # Load and filter data
    df = load_and_filter_data()

    # Save filtered data
    save_filtered_data(df)

    print("\n" + "="*80)
    print("PHASE 0 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
