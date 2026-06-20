"""
Prepare Learning Curves Data

This script:
1. Loads best_pca_per_property.csv from each n directory (n100, n250, n500)
2. Filters to ORB descriptor only (most consistent performer)
3. Concatenates data from all dataset sizes
4. Saves combined learning curves data for plotting

Output: learning_curves_orb.csv with columns:
    - model, property, n_train, best_pca, R2, RMSE, Spearman
"""

import pandas as pd
import os

def load_best_pca_data(n_train):
    """
    Load best PCA per property data for a specific n_train value

    Args:
        n_train: Training set size (100, 250, or 500)

    Returns:
        DataFrame with best PCA data, filtered to ORB only
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, '..', '..', 'analysis_v3_dielectric_constant', f'n{n_train}', 'data', 'best_pca_per_property.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            f"Make sure n{n_train} analysis has been run first."
        )

    print(f"Loading n={n_train} data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Original shape: {df.shape}")
    print(f"  Descriptors: {df['descriptor'].unique()}")

    # Filter to ORB only
    df = df[df['descriptor'] == 'orb'].copy()
    print(f"  After ORB filter: {df.shape}")

    # Add n_train column
    df['n_train'] = n_train

    return df

def main():
    print("="*80)
    print("PREPARING LEARNING CURVES DATA (ORB DESCRIPTOR ONLY)")
    print("="*80)

    # Load data from all three dataset sizes
    df_list = []
    for n in [100, 250, 500]:
        df = load_best_pca_data(n)
        df_list.append(df)

    # Concatenate
    learning_curves = pd.concat(df_list, ignore_index=True)

    print(f"\n{'='*80}")
    print("COMBINED DATA SUMMARY")
    print('='*80)
    print(f"Total rows: {len(learning_curves)}")
    print(f"Models: {sorted(learning_curves['model'].unique())}")
    print(f"Properties: {sorted(learning_curves['property'].unique())}")
    print(f"n_train values: {sorted(learning_curves['n_train'].unique())}")
    print(f"Descriptors: {learning_curves['descriptor'].unique()}")

    # Expected: 3 models × 4 properties × 3 n_train = 36 rows (note: dgp missing n500)
    expected_rows = 3 * 4 * 3
    print(f"\nExpected rows: {expected_rows}")
    print(f"Actual rows: {len(learning_curves)}")

    if len(learning_curves) != expected_rows:
        print("WARNING: Row count mismatch!")

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'learning_curves_orb.csv')
    learning_curves.to_csv(output_path, index=False)

    print(f"\n✓ Saved learning curves to {output_path}")
    print(f"  Shape: {learning_curves.shape}")
    print(f"  Columns: {learning_curves.columns.tolist()}")

    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
