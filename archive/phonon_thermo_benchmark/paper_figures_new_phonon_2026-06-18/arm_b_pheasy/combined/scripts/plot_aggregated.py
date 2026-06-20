"""
Plot Aggregated Learning Curves

This script:
1. Loads learning curves data (ORB descriptor only)
2. Aggregates performance across all 8 properties (mean ± std)
3. Creates learning curve plots showing overall model scaling behavior

Output: 3 PDF files (R², RMSE, Spearman)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

# Color scheme (consistent with other analyses)
colors_models = {
    'gp': 'tab:blue',
    'mtgp_2': 'tab:orange',
    'dgp': 'tab:green'
}

# Model labels
model_labels = {
    'gp': 'GP',
    'mtgp_2': 'MTGP',
    'dgp': 'DGP'
}

def plot_aggregated_learning_curve(df, metric, output_dir):
    """
    Plot aggregated learning curve for a metric (averaged across properties)

    Args:
        df: DataFrame with learning curves data
        metric: Metric to plot ('R2', 'RMSE', or 'Spearman')
        output_dir: Directory to save figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for model in ['gp', 'mtgp_2']:
        df_model = df[df['model'] == model].copy()

        # Group by n_train and compute mean/std across properties
        grouped = df_model.groupby('n_train')[metric].agg(['mean', 'std']).reset_index()

        # Sort by n_train
        grouped = grouped.sort_values('n_train')

        # Plot with error bars
        ax.errorbar(grouped['n_train'], grouped['mean'],
                   yerr=grouped['std'],
                   marker='o', label=model_labels[model],
                   color=colors_models[model],
                   linewidth=2, markersize=8,
                   capsize=5, capthick=2)

    # Labels and title
    ax.set_xlabel('Training Set Size (n)', fontsize=14)
    ax.set_ylabel(f'{metric} (mean ± std)', fontsize=14)
    ax.set_title(f'Learning Curve - {metric} (ORB, averaged across properties)',
                fontsize=16, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # Set x-axis ticks to actual values
    ax.set_xticks([100, 250, 500, 1000, 2000])

    # Adjust y-axis limits for better visualization
    if metric == 'R2':
        ax.set_ylim(bottom=0, top=1.05)
    elif metric == 'Spearman':
        ax.set_ylim(bottom=0, top=1.05)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'averaged_{metric}_learning_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def print_summary_statistics(df, metric):
    """Print summary statistics for a metric"""
    print(f"\n{metric} Statistics (mean ± std across properties):")
    print("-" * 80)

    for n_train in [100, 250, 500, 1000, 2000]:
        print(f"\nn = {n_train}:")
        df_n = df[df['n_train'] == n_train]

        for model in ['gp', 'mtgp_2']:
            df_model = df_n[df_n['model'] == model]
            mean_val = df_model[metric].mean()
            std_val = df_model[metric].std()
            print(f"  {model_labels[model]:6s}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    print("="*80)
    print("PLOTTING AGGREGATED LEARNING CURVES")
    print("="*80)

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'learning_curves_orb.csv')

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find {data_path}. "
            "Run prepare_learning_curves.py first."
        )

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # Output directory
    output_dir = os.path.join(script_dir, '..', 'figures', 'aggregated')

    # Plot for each metric
    print("\nGenerating aggregated learning curve plots...")

    for metric in ['R2', 'RMSE', 'Spearman']:
        print(f"\n{metric}:")
        plot_aggregated_learning_curve(df, metric, output_dir)
        print_summary_statistics(df, metric)

    print(f"\n✓ Generated 3 plots in {output_dir}")

    print("\n" + "="*80)
    print("AGGREGATED PLOTTING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
