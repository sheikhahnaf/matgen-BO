"""
Plot Learning Curves Per Property

This script:
1. Loads learning curves data (ORB descriptor only)
2. For each property, creates 3 learning curve plots (R², RMSE, Spearman)
3. Shows how each model's performance scales with training set size

Output: 24 PDF files (8 properties × 3 metrics)
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

def get_property_list():
    """Return 8 properties"""
    return [
        'Cv_300K', 'S_300K', 'F_300K', 'max_phonon_freq'
    ]

def plot_property_learning_curve(df, property_name, metric, output_dir):
    """
    Plot learning curve for a specific property and metric

    Args:
        df: DataFrame with learning curves data
        property_name: Property to plot
        metric: Metric to plot ('R2', 'RMSE', or 'Spearman')
        output_dir: Directory to save figure
    """
    # Filter to this property
    df_prop = df[df['property'] == property_name].copy()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for model in ['gp', 'mtgp_2', 'dgp']:
        df_model = df_prop[df_prop['model'] == model].copy()

        # Sort by n_train for line plot
        df_model = df_model.sort_values('n_train')

        # Plot
        ax.plot(df_model['n_train'], df_model[metric],
               marker='o', label=model_labels[model],
               color=colors_models[model],
               linewidth=2, markersize=8)

    # Labels and title
    ax.set_xlabel('Training Set Size (n)', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f'{property_name} - {metric} Learning Curve (ORB)', fontsize=16, fontweight='bold')

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
    output_path = os.path.join(output_dir, f'{property_name}_{metric}_learning_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def main():
    print("="*80)
    print("PLOTTING LEARNING CURVES PER PROPERTY")
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
    output_dir = os.path.join(script_dir, '..', 'figures', 'per_property')

    # Get properties
    properties = get_property_list()

    # Plot for each property and metric
    total_plots = len(properties) * 3
    plot_count = 0

    print(f"\nGenerating {total_plots} learning curve plots...")

    for property_name in properties:
        print(f"\n{property_name}:")
        for metric in ['R2', 'RMSE', 'Spearman']:
            plot_property_learning_curve(df, property_name, metric, output_dir)
            plot_count += 1

    print(f"\n✓ Generated {plot_count} plots in {output_dir}")

    print("\n" + "="*80)
    print("PER-PROPERTY PLOTTING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
