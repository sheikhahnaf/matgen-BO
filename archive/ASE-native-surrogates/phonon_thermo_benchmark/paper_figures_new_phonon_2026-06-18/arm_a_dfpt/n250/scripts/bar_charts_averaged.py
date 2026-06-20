"""
Phase 1: Bar Charts (Averaged Across Properties)

This script:
1. For each (model, descriptor) combo, finds PCA that gives best AVERAGE performance
2. Creates bar charts comparing surrogates using best PCA configs
3. Generates charts for R², RMSE, and Spearman
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

def get_best_pca_per_surrogate_averaged(df, metric='R2'):
    """
    For each (model, descriptor) combo, find PCA that gives best AVERAGE performance
    across all 8 properties.

    Steps:
    1. For each (model, descriptor, pca) combo, compute average metric across 8 properties
    2. For each (model, descriptor), select PCA with best average
    3. Return filtered dataframe with only best PCA configs

    Args:
        df: DataFrame with filtered n=250 data
        metric: Metric to optimize ('R2', 'RMSE', or 'Spearman')

    Returns:
        DataFrame with best PCA configs and their average performance
        Columns: model, descriptor, best_pca, avg_R2, avg_RMSE, avg_Spearman
    """
    results = []

    models = ['gp', 'mtgp_2', 'dgp']
    descriptors = ['mace', 'orb', 'soap', 'uma']
    pcas = [10, 25, 50]

    for model in models:
        for descriptor in descriptors:
            # Get data for this model-descriptor combo
            subset = df[(df['model'] == model) & (df['descriptor'] == descriptor)].copy()

            # For each PCA, compute average across properties for each metric
            pca_performance = {}
            for pca in pcas:
                pca_data = subset[subset['pca_components'] == pca]

                # Average across properties for each metric
                avg_metrics = {}
                for m in ['R2', 'RMSE', 'Spearman']:
                    metric_data = pca_data[pca_data['metric'] == m]
                    avg_metrics[m] = metric_data['mean'].mean()

                pca_performance[pca] = avg_metrics

            # Select best PCA based on specified metric (only if data exists)
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
                    'best_pca': best_pca,
                    'avg_R2': best_metrics['R2'],
                    'avg_RMSE': best_metrics['RMSE'],
                    'avg_Spearman': best_metrics['Spearman']
                })

    return pd.DataFrame(results)

def plot_bar_chart_averaged(best_configs_df, metric='R2', output_dir='../figures/bar_charts'):
    """
    Bar chart: x-axis = descriptors, y-axis = metric value, bars grouped by model

    Title includes PCA choices.

    Args:
        best_configs_df: DataFrame with best PCA configs
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
        # Ensure correct order
        model_data = model_data.set_index('descriptor').loc[descriptors].reset_index()

        values = model_data[f'avg_{metric}'].values

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
    title = f'Averaged {metric} Across Properties\n(n=250, Best PCA per Surrogate)'
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
            pca = desc_data[desc_data['model'] == model]['best_pca'].values[0]
            pca_str.append(f"{model_labels[model]}:{pca}")
        pca_annotations.append(f"{descriptor.upper()}({', '.join(pca_str)})")

    fig.text(0.5, 0.01, "Best PCA per surrogate: " + " | ".join(pca_annotations),
            ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save figure
    output_path = os.path.join(output_dir, f'averaged_{metric}_n250.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def main():
    print("="*80)
    print("PHASE 1: BAR CHARTS (AVERAGED ACROSS PROPERTIES)")
    print("="*80)

    # Load filtered data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'filtered_n250.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded data: {df.shape}")

    # Get best PCA configs (optimized for R2, but return all metrics)
    print("\nFinding best PCA per surrogate (optimized for R²)...")
    best_configs = get_best_pca_per_surrogate_averaged(df, metric='R2')

    print("\nBest PCA configurations:")
    print(best_configs.to_string(index=False))

    # Save best PCA configs
    output_path = os.path.join(script_dir, '..', 'data', 'best_pca_averaged.csv')
    best_configs.to_csv(output_path, index=False)
    print(f"\n✓ Saved best PCA configs to {output_path}")

    # Create bar charts for each metric
    print("\nGenerating bar charts...")
    for metric in ['R2', 'RMSE', 'Spearman']:
        print(f"  Creating {metric} chart...")
        plot_bar_chart_averaged(best_configs, metric=metric,
                               output_dir=os.path.join(script_dir, '..', 'figures', 'bar_charts'))

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
