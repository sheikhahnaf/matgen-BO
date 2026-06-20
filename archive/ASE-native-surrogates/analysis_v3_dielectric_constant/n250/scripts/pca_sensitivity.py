"""
Phase 5: PCA Sensitivity Analysis

This script:
1. Shows how PCA choice affects performance (averaged across properties)
2. Shows how PCA choice affects performance per property
3. Identifies which properties are more/less sensitive to PCA choice
4. Creates line plots showing performance vs PCA components
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def get_property_list():
    """Return 4 properties for dielectric_constant dataset"""
    return [
        'band_gap', 'n', 'poly_electronic', 'poly_total'
    ]

def pca_sensitivity_averaged(df):
    """
    For each (model, descriptor) combo, show R² and Spearman at PCA=10, 25, 50
    averaged across all properties.

    Args:
        df: Filtered n=250 data

    Returns:
        DataFrame with columns: model, descriptor, pca, avg_R2, avg_Spearman
    """
    results = []

    models = ['gp', 'mtgp_2', 'dgp']
    descriptors = ['mace', 'orb', 'soap', 'uma']
    pcas = [10, 25, 50]

    for model in models:
        for descriptor in descriptors:
            for pca in pcas:
                subset = df[
                    (df['model'] == model) &
                    (df['descriptor'] == descriptor) &
                    (df['pca_components'] == pca)
                ]

                # Average across properties for each metric
                avg_r2 = subset[subset['metric'] == 'R2']['mean'].mean()
                avg_spearman = subset[subset['metric'] == 'Spearman']['mean'].mean()

                results.append({
                    'model': model,
                    'descriptor': descriptor,
                    'pca': pca,
                    'avg_R2': avg_r2,
                    'avg_Spearman': avg_spearman
                })

    return pd.DataFrame(results)

def pca_sensitivity_per_property(df):
    """
    Same as averaged, but for each property separately.

    Returns:
        DataFrame with columns: model, descriptor, property, pca, R2, Spearman
    """
    results = []

    models = ['gp', 'mtgp_2', 'dgp']
    descriptors = ['mace', 'orb', 'soap', 'uma']
    pcas = [10, 25, 50]
    properties = get_property_list()

    for model in models:
        for descriptor in descriptors:
            for prop in properties:
                for pca in pcas:
                    subset = df[
                        (df['model'] == model) &
                        (df['descriptor'] == descriptor) &
                        (df['property'] == prop) &
                        (df['pca_components'] == pca)
                    ]

                    # Skip if no data
                    if subset.empty:
                        continue

                    # Get R2 and Spearman
                    r2_vals = subset[subset['metric'] == 'R2']['mean'].values
                    spearman_vals = subset[subset['metric'] == 'Spearman']['mean'].values

                    # Skip if any metric is missing
                    if len(r2_vals) == 0 or len(spearman_vals) == 0:
                        continue

                    r2 = r2_vals[0]
                    spearman = spearman_vals[0]

                    results.append({
                        'model': model,
                        'descriptor': descriptor,
                        'property': prop,
                        'pca': pca,
                        'R2': r2,
                        'Spearman': spearman
                    })

    return pd.DataFrame(results)

def plot_pca_sensitivity_averaged(sensitivity_df, metric='R2',
                                 output_dir='../figures/pca_sensitivity'):
    """
    Line plot: x=PCA components, y=metric, separate lines for model-descriptor combos

    Args:
        sensitivity_df: DataFrame with PCA sensitivity data
        metric: Metric to plot ('R2' or 'Spearman')
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    models = ['gp', 'mtgp_2', 'dgp']
    descriptors = ['orb', 'mace', 'uma', 'soap']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}
    colors = {'gp': 'tab:blue', 'mtgp_2': 'tab:orange', 'dgp': 'tab:green'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metric_col = f'avg_{metric}' if f'avg_{metric}' in sensitivity_df.columns else metric

    for idx, descriptor in enumerate(descriptors):
        ax = axes[idx]
        desc_data = sensitivity_df[sensitivity_df['descriptor'] == descriptor]

        for model in models:
            model_data = desc_data[desc_data['model'] == model].sort_values('pca')

            ax.plot(model_data['pca'], model_data[metric_col],
                   marker='o', markersize=10, linewidth=2.5,
                   label=model_labels[model], color=colors[model])

        ax.set_xlabel('PCA Components', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{descriptor.upper()} Descriptor',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xticks([10, 25, 50])

        if metric in ['R2', 'Spearman']:
            ax.set_ylim(0, 1.05)

    plt.suptitle(f'PCA Sensitivity: Averaged {metric} Across Properties\n(n=250)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'averaged_{metric}_n250.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def plot_pca_sensitivity_per_property(sensitivity_df, metric='R2',
                                     output_dir='../figures/pca_sensitivity'):
    """
    8 subplots (one per property), showing PCA sensitivity for ORB descriptor only

    Args:
        sensitivity_df: DataFrame with per-property PCA sensitivity
        metric: Metric to plot ('R2' or 'Spearman')
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    properties = get_property_list()
    models = ['gp', 'mtgp_2', 'dgp']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}
    colors = {'gp': 'tab:blue', 'mtgp_2': 'tab:orange', 'dgp': 'tab:green'}

    # Use ORB only (best descriptor)
    orb_data = sensitivity_df[sensitivity_df['descriptor'] == 'orb']

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, prop in enumerate(properties):
        ax = axes[idx]
        prop_data = orb_data[orb_data['property'] == prop]

        for model in models:
            model_data = prop_data[prop_data['model'] == model].sort_values('pca')

            ax.plot(model_data['pca'], model_data[metric],
                   marker='o', markersize=8, linewidth=2,
                   label=model_labels[model], color=colors[model])

        ax.set_xlabel('PCA Components', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10, fontweight='bold')
        ax.set_title(prop, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks([10, 25, 50])

        if metric in ['R2', 'Spearman']:
            ax.set_ylim(0, 1.05)

    plt.suptitle(f'PCA Sensitivity Per Property: {metric} (ORB Descriptor, n=250)',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'per_property_{metric}_n250.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def analyze_sensitivity(sensitivity_avg, sensitivity_per_prop):
    """
    Analyze which configurations are most/least sensitive to PCA choice

    Args:
        sensitivity_avg: Averaged sensitivity data
        sensitivity_per_prop: Per-property sensitivity data
    """
    print("\n" + "="*80)
    print("PCA SENSITIVITY ANALYSIS")
    print("="*80)

    # Averaged sensitivity
    print("\nAveraged across properties (R² range across PCA 10→50):")
    print("-" * 60)

    for model in ['gp', 'mtgp_2', 'dgp']:
        model_label = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}[model]
        print(f"\n{model_label}:")

        for descriptor in ['orb', 'mace', 'uma', 'soap']:
            subset = sensitivity_avg[
                (sensitivity_avg['model'] == model) &
                (sensitivity_avg['descriptor'] == descriptor)
            ].sort_values('pca')

            r2_range = subset['avg_R2'].max() - subset['avg_R2'].min()

            print(f"  {descriptor.upper():6s}: R² range = {r2_range:.4f} "
                 f"(PCA10={subset.iloc[0]['avg_R2']:.3f}, "
                 f"PCA25={subset.iloc[1]['avg_R2']:.3f}, "
                 f"PCA50={subset.iloc[2]['avg_R2']:.3f})")

    # Property-level sensitivity (ORB only)
    print("\n" + "="*60)
    print("Per-property sensitivity (ORB descriptor, R² range):")
    print("-" * 60)

    orb_data = sensitivity_per_prop[sensitivity_per_prop['descriptor'] == 'orb']

    for model in ['gp', 'mtgp_2', 'dgp']:
        model_label = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}[model]
        print(f"\n{model_label}:")

        prop_sensitivities = []
        for prop in get_property_list():
            subset = orb_data[
                (orb_data['model'] == model) &
                (orb_data['property'] == prop)
            ].sort_values('pca')

            r2_range = subset['R2'].max() - subset['R2'].min()
            prop_sensitivities.append((prop, r2_range))

        # Sort by sensitivity (most to least)
        prop_sensitivities.sort(key=lambda x: x[1], reverse=True)

        for prop, r2_range in prop_sensitivities:
            print(f"  {prop:25s}: R² range = {r2_range:.4f}")

def main():
    print("="*80)
    print("PHASE 5: PCA SENSITIVITY ANALYSIS")
    print("="*80)

    # Load filtered data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'filtered_n250.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded data: {df.shape}")

    # Compute PCA sensitivity (averaged)
    print("\nComputing PCA sensitivity (averaged across properties)...")
    sensitivity_avg = pca_sensitivity_averaged(df)

    # Save averaged sensitivity
    output_path = os.path.join(script_dir, '..', 'data', 'pca_sensitivity_averaged.csv')
    sensitivity_avg.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Compute PCA sensitivity (per property)
    print("\nComputing PCA sensitivity (per property)...")
    sensitivity_per_prop = pca_sensitivity_per_property(df)

    # Save per-property sensitivity
    output_path = os.path.join(script_dir, '..', 'data', 'pca_sensitivity_per_property.csv')
    sensitivity_per_prop.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Plot averaged sensitivity
    print("\nGenerating averaged PCA sensitivity plots...")
    for metric in ['R2', 'Spearman']:
        print(f"  Creating {metric} plot...")
        plot_pca_sensitivity_averaged(
            sensitivity_avg, metric=metric,
            output_dir=os.path.join(script_dir, '..', 'figures', 'pca_sensitivity')
        )

    # Plot per-property sensitivity
    print("\nGenerating per-property PCA sensitivity plots...")
    for metric in ['R2', 'Spearman']:
        print(f"  Creating {metric} plot...")
        plot_pca_sensitivity_per_property(
            sensitivity_per_prop, metric=metric,
            output_dir=os.path.join(script_dir, '..', 'figures', 'pca_sensitivity')
        )

    # Analyze sensitivity
    analyze_sensitivity(sensitivity_avg, sensitivity_per_prop)

    print("\n" + "="*80)
    print("PHASE 5 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
