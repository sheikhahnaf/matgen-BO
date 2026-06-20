"""
Phase 2: Bar Charts (Per Property)

This script:
1. For each property, finds best PCA per surrogate (may differ from averaged case)
2. Creates bar charts for each property separately
3. Generates charts for all 3 metrics × 8 properties = 24 charts
4. Saves PCA choices to CSV for reference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

def get_property_list():
    """Return 8 properties (excluding kpoint_density)"""
    return [
        'K_Voigt', 'K_VRH', 'K_Reuss',
        'G_Voigt', 'G_VRH', 'G_Reuss',
        'elastic_anisotropy', 'poisson_ratio'
    ]

def get_best_pca_per_surrogate_per_property(df, property_name, metric='R2'):
    """
    For each (model, descriptor) combo, find PCA that gives best performance
    for THIS SPECIFIC property.

    Note: Best PCA may differ from averaged case!

    Args:
        df: DataFrame with filtered n=100 data
        property_name: Property to analyze
        metric: Metric to optimize ('R2', 'RMSE', or 'Spearman')

    Returns:
        DataFrame with best PCA configs for this property
        Columns: model, descriptor, best_pca, R2, RMSE, Spearman
    """
    results = []

    # Filter to this property
    prop_df = df[df['property'] == property_name].copy()

    models = ['gp', 'mtgp_2', 'dgp']
    descriptors = ['mace', 'orb', 'soap', 'uma']
    pcas = [10, 25, 50]

    for model in models:
        for descriptor in descriptors:
            subset = prop_df[(prop_df['model'] == model) &
                           (prop_df['descriptor'] == descriptor)].copy()

            # For each PCA, get metric values
            pca_performance = {}
            for pca in pcas:
                pca_data = subset[subset['pca_components'] == pca]

                # Get all metrics at this PCA
                metrics_dict = {}
                for m in ['R2', 'RMSE', 'Spearman']:
                    m_data = pca_data[pca_data['metric'] == m]
                    if len(m_data) > 0:
                        metrics_dict[m] = m_data['mean'].values[0]

                if len(metrics_dict) == 3:  # All metrics present
                    pca_performance[pca] = metrics_dict

            # Select best PCA based on specified metric
            if len(pca_performance) > 0:
                if metric == 'RMSE':
                    best_pca = min(pca_performance.keys(),
                                 key=lambda p: pca_performance[p][metric])
                else:  # R2 or Spearman (higher is better)
                    best_pca = max(pca_performance.keys(),
                                 key=lambda p: pca_performance[p][metric])

                # Get all metrics at best PCA
                best_metrics = pca_performance[best_pca]

                results.append({
                    'model': model,
                    'descriptor': descriptor,
                    'property': property_name,
                    'best_pca': best_pca,
                    'R2': best_metrics['R2'],
                    'RMSE': best_metrics['RMSE'],
                    'Spearman': best_metrics['Spearman']
                })

    return pd.DataFrame(results)

def plot_bar_chart_per_property(best_configs_df, property_name, metric='R2',
                                output_dir='../figures/bar_charts/per_property'):
    """
    Bar chart for a specific property.

    Args:
        best_configs_df: DataFrame with best PCA configs for this property
        property_name: Property being plotted
        metric: Metric to plot ('R2', 'RMSE', or 'Spearman')
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    descriptors = ['mace', 'orb', 'soap', 'uma']
    models = ['gp', 'mtgp_2', 'dgp']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}
    colors = {'gp': 'tab:blue', 'mtgp_2': 'tab:orange', 'dgp': 'tab:green'}

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(descriptors))
    width = 0.25

    for i, model in enumerate(models):
        model_data = best_configs_df[best_configs_df['model'] == model]

        # Skip if no valid data for this model (e.g., DGP for n=100, n=100)
        if model_data.empty or model_data[metric].isna().all():
            continue

        # Ensure correct order
        model_data = model_data.set_index('descriptor').loc[descriptors].reset_index()

        values = model_data[metric].values

        bars = ax.bar(x + i*width, values, width,
                     label=model_labels[model],
                     color=colors[model],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Descriptor', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=14, fontweight='bold')

    # Set title
    title = f'{property_name}: {metric}\n(n=100, Best PCA per Surrogate for This Property)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.upper() for d in descriptors], fontsize=12)
    ax.legend(fontsize=12, loc='best')
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits based on metric
    if metric in ['R2', 'Spearman']:
        ax.set_ylim(0, 1.05)
    else:  # RMSE
        # Auto-scale but start from 0
        ax.set_ylim(0, None)

    # Create PCA annotations text
    pca_annotations = []
    for descriptor in descriptors:
        desc_data = best_configs_df[best_configs_df['descriptor'] == descriptor]
        pca_str = []
        for model in models:
            model_pca = desc_data[desc_data['model'] == model]['best_pca'].values
            if len(model_pca) > 0:  # Only add if data exists
                pca_str.append(f"{model_labels[model]}:{int(model_pca[0])}")
        if pca_str:  # Only add descriptor if it has any data
            pca_annotations.append(f"{descriptor.upper()}({', '.join(pca_str)})")

    fig.text(0.5, 0.01, "Best PCA: " + " | ".join(pca_annotations),
            ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save figure
    output_path = os.path.join(output_dir, f'{property_name}_{metric}_n100.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {property_name}_{metric}_n100.png")

    plt.close()

def main():
    print("="*80)
    print("PHASE 2: BAR CHARTS (PER PROPERTY)")
    print("="*80)

    # Load filtered data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'filtered_n100.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded data: {df.shape}")

    properties = get_property_list()
    metrics = ['R2', 'RMSE', 'Spearman']

    # Collect all best PCA configs across properties
    all_best_configs = []

    # For each property
    for prop in properties:
        print(f"\nProcessing property: {prop}")

        # Get best PCA configs for this property (optimized for R2)
        best_configs = get_best_pca_per_surrogate_per_property(df, prop, metric='R2')
        all_best_configs.append(best_configs)

        # Create bar charts for each metric
        for metric in metrics:
            plot_bar_chart_per_property(
                best_configs, prop, metric=metric,
                output_dir=os.path.join(script_dir, '..', 'figures', 'bar_charts', 'per_property')
            )

    # Combine all configs and save
    combined_configs = pd.concat(all_best_configs, ignore_index=True)
    output_path = os.path.join(script_dir, '..', 'data', 'best_pca_per_property.csv')
    combined_configs.to_csv(output_path, index=False)
    print(f"\n✓ Saved all best PCA configs to {output_path}")

    print(f"\n✓ Generated {len(properties) * len(metrics)} charts")

    print("\n" + "="*80)
    print("PHASE 2 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
