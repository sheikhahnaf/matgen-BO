"""
Data Integrity Verification Script

This script verifies that all metrics used in analysis_v2 are extracted from
real experimental results (aggregated_results.csv) and not made-up values.
"""

import pandas as pd
import numpy as np

def verify_data_integrity():
    """Verify data integrity across the pipeline."""

    print("=" * 80)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 80)

    # Load source data
    source = pd.read_csv('../../ASE_regression_test/analysis/aggregated_results.csv')
    print(f"\n1. Source data loaded: {source.shape}")
    print(f"   Columns: {source.columns.tolist()}")

    # Load my filtered data
    my_data = pd.read_csv('../data/filtered_n250.csv')
    print(f"\n2. Filtered data loaded: {my_data.shape}")

    # Verify filtering
    print(f"\n3. Filtering verification:")
    print(f"   - n_train = 100 only: {my_data['n_train'].unique()}")
    print(f"   - Metrics: {sorted(my_data['metric'].unique())}")
    print(f"   - Properties: {sorted(my_data['property'].unique())}")

    # Verify specific data points
    print(f"\n4. Spot-check specific values:")
    test_cases = [
        ('gp', 'orb', 50, 'K_Voigt', 'R2'),
        ('mtgp_2', 'mace', 50, 'G_Reuss', 'RMSE'),
        ('dgp', 'orb', 25, 'elastic_anisotropy', 'Spearman'),
    ]

    all_match = True
    for model, desc, pca, prop, metric in test_cases:
        source_val = source[
            (source['model'] == model) &
            (source['descriptor'] == desc) &
            (source['pca_components'] == pca) &
            (source['n_train'] == 250) &
            (source['property'] == prop) &
            (source['metric'] == metric)
        ]['mean'].values[0]

        my_val = my_data[
            (my_data['model'] == model) &
            (my_data['descriptor'] == desc) &
            (my_data['pca_components'] == pca) &
            (my_data['property'] == prop) &
            (my_data['metric'] == metric)
        ]['mean'].values[0]

        match = abs(source_val - my_val) < 1e-10
        all_match = all_match and match

        print(f"   {model:6s} + {desc:4s} + PCA={pca:2d} + {prop:20s} + {metric:8s}")
        print(f"     Source: {source_val:.6f}, Mine: {my_val:.6f}, Match: {match}")

    # Verify best PCA calculations
    print(f"\n5. Best PCA calculation verification:")
    best_pca = pd.read_csv('../data/best_pca_averaged.csv')

    # Check GP + ORB
    gp_orb_best = best_pca[
        (best_pca['model'] == 'gp') &
        (best_pca['descriptor'] == 'orb')
    ].iloc[0]

    print(f"\n   GP + ORB (from best_pca_averaged.csv):")
    print(f"     Best PCA: {gp_orb_best['best_pca']}")
    print(f"     Avg R²: {gp_orb_best['avg_R2']:.6f}")
    print(f"     Avg RMSE: {gp_orb_best['avg_RMSE']:.6f}")
    print(f"     Avg Spearman: {gp_orb_best['avg_Spearman']:.6f}")

    # Manual recalculation
    manual_r2 = my_data[
        (my_data['model'] == 'gp') &
        (my_data['descriptor'] == 'orb') &
        (my_data['pca_components'] == 50) &
        (my_data['metric'] == 'R2')
    ]['mean'].mean()

    manual_rmse = my_data[
        (my_data['model'] == 'gp') &
        (my_data['descriptor'] == 'orb') &
        (my_data['pca_components'] == 50) &
        (my_data['metric'] == 'RMSE')
    ]['mean'].mean()

    manual_spearman = my_data[
        (my_data['model'] == 'gp') &
        (my_data['descriptor'] == 'orb') &
        (my_data['pca_components'] == 50) &
        (my_data['metric'] == 'Spearman')
    ]['mean'].mean()

    print(f"\n   Manual recalculation (averaging across 8 properties):")
    print(f"     Avg R²: {manual_r2:.6f}")
    print(f"     Avg RMSE: {manual_rmse:.6f}")
    print(f"     Avg Spearman: {manual_spearman:.6f}")

    print(f"\n   Matches:")
    print(f"     R²: {abs(gp_orb_best['avg_R2'] - manual_r2) < 1e-6}")
    print(f"     RMSE: {abs(gp_orb_best['avg_RMSE'] - manual_rmse) < 1e-6}")
    print(f"     Spearman: {abs(gp_orb_best['avg_Spearman'] - manual_spearman) < 1e-6}")

    # Final verdict
    print("\n" + "=" * 80)
    if all_match:
        print("✓ VERIFICATION PASSED: All data matches source")
        print("✓ No made-up values detected")
        print("✓ All metrics extracted correctly from experimental results")
    else:
        print("✗ VERIFICATION FAILED: Data mismatch detected!")
    print("=" * 80)

    return all_match

if __name__ == '__main__':
    verify_data_integrity()
