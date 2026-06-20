"""
Phase 6: Radar Charts (ORB Featurizer Only)

This script:
1. Creates radar charts comparing surrogates across properties
2. Uses ONLY ORB descriptor (best performing)
3. Properties as vertices, surrogates as lines
4. Creates charts for R² and Spearman
5. Uses best PCA per surrogate per property
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import os

# Set style
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

def get_property_list():
    """Return 4 properties for dielectric_constant dataset"""
    return [
        'band_gap', 'n', 'poly_electronic', 'poly_total'
    ]

def prepare_radar_data_orb(df, metric='R2'):
    """
    For ORB descriptor only, get best PCA per surrogate per property.

    Args:
        df: Filtered n=500 data
        metric: Metric to optimize ('R2' or 'Spearman')

    Returns:
        DataFrame with columns: model, property, best_pca, value
    """
    results = []

    orb_df = df[df['descriptor'] == 'orb'].copy()
    models = ['gp', 'mtgp_2', 'dgp']
    properties = get_property_list()
    pcas = [10, 25, 50]

    for model in models:
        for prop in properties:
            # For this model-property, find best PCA
            subset = orb_df[
                (orb_df['model'] == model) &
                (orb_df['property'] == prop) &
                (orb_df['metric'] == metric)
            ]

            # Get metric value at each PCA
            pca_performance = {}
            for pca in pcas:
                pca_data = subset[subset['pca_components'] == pca]
                if len(pca_data) > 0:
                    pca_performance[pca] = pca_data['mean'].values[0]

            # Select best PCA
            if len(pca_performance) > 0:
                if metric == 'RMSE':
                    best_pca = min(pca_performance.keys(), key=lambda p: pca_performance[p])
                else:  # R2 or Spearman (higher is better)
                    best_pca = max(pca_performance.keys(), key=lambda p: pca_performance[p])

                best_value = pca_performance[best_pca]

                results.append({
                    'model': model,
                    'property': prop,
                    'best_pca': best_pca,
                    'value': best_value
                })

    return pd.DataFrame(results)

def plot_radar_chart_orb(radar_data, metric='R2', output_dir='../figures/radar_charts'):
    """
    Radar chart with properties as vertices, surrogates as lines.

    Args:
        radar_data: DataFrame with radar chart data
        metric: Metric being plotted ('R2' or 'Spearman')
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    properties = get_property_list()
    models = ['gp', 'mtgp_2', 'dgp']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}
    colors = {'gp': 'tab:blue', 'mtgp_2': 'tab:orange', 'dgp': 'tab:green'}

    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(properties), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw=dict(projection='polar'))

    # Plot each model
    for model in models:
        model_data = radar_data[radar_data['model'] == model]
        if model_data.empty:
            continue
        # Ensure correct order
        model_data = model_data.set_index('property').reindex(properties).reset_index()

        values = model_data['value'].tolist()
        values += values[:1]  # Close the circle

        ax.plot(angles, values, 'o-', linewidth=3, markersize=10,
               label=model_labels[model], color=colors[model], alpha=0.9)
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(properties, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

    title = f'Surrogate Comparison Across Properties - ORB Descriptor\n' \
            f'{metric} (n=500, Best PCA per Surrogate per Property)'
    ax.set_title(title, size=16, fontweight='bold', pad=30)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=13)
    ax.grid(True, alpha=0.3)

    # Add PCA annotations table below
    print_pca_table = True
    if print_pca_table:
        # Create PCA table
        pca_table_data = []
        for prop in properties:
            row = [prop]
            for model in models:
                pca_rows = radar_data[
                    (radar_data['model'] == model) &
                    (radar_data['property'] == prop)
                ]['best_pca'].values
                row.append(f"PCA={pca_rows[0]}" if len(pca_rows) > 0 else "N/A")
            pca_table_data.append(row)

        # Format as text
        pca_text_lines = ["Best PCA per Surrogate per Property:"]
        pca_text_lines.append("-" * 70)
        header = f"{'Property':25s} {'GP':15s} {'MTGP':15s} {'DGP':15s}"
        pca_text_lines.append(header)
        pca_text_lines.append("-" * 70)
        for row in pca_table_data:
            pca_text_lines.append(f"{row[0]:25s} {row[1]:15s} {row[2]:15s} {row[3]:15s}")

        fig.text(0.5, -0.05, "\n".join(pca_text_lines),
                ha='center', fontsize=8, style='italic',
                family='monospace', wrap=True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save figure
    output_path = os.path.join(output_dir, f'orb_{metric}_n500.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def analyze_radar_data(radar_data_r2, radar_data_spearman):
    """
    Analyze radar chart data to identify patterns

    Args:
        radar_data_r2: Radar data for R²
        radar_data_spearman: Radar data for Spearman
    """
    print("\n" + "="*80)
    print("RADAR CHART ANALYSIS (ORB Descriptor)")
    print("="*80)

    properties = get_property_list()
    models = ['gp', 'mtgp_2', 'dgp']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}

    # Average performance per model
    print("\nAverage R² per model (across 8 properties):")
    print("-" * 60)
    for model in models:
        avg_r2 = radar_data_r2[radar_data_r2['model'] == model]['value'].mean()
        print(f"  {model_labels[model]:6s}: {avg_r2:.3f}")

    print("\nAverage Spearman per model (across 8 properties):")
    print("-" * 60)
    for model in models:
        avg_spearman = radar_data_spearman[radar_data_spearman['model'] == model]['value'].mean()
        print(f"  {model_labels[model]:6s}: {avg_spearman:.3f}")

    # Best/worst properties per model
    print("\n" + "="*60)
    print("Best and Worst Properties per Model (R²):")
    print("-" * 60)

    for model in models:
        model_data = radar_data_r2[radar_data_r2['model'] == model].set_index('property')
        if model_data.empty:
            continue
        best_prop = model_data['value'].idxmax()
        best_r2 = model_data['value'].max()
        worst_prop = model_data['value'].idxmin()
        worst_r2 = model_data['value'].min()

        print(f"\n{model_labels[model]}:")
        print(f"  Best:  {best_prop:25s} R²={best_r2:.3f}")
        print(f"  Worst: {worst_prop:25s} R²={worst_r2:.3f}")
        print(f"  Range: {best_r2 - worst_r2:.3f}")

    # PCA usage patterns
    print("\n" + "="*60)
    print("PCA Usage Patterns (ORB Descriptor):")
    print("-" * 60)

    for model in models:
        model_data = radar_data_r2[radar_data_r2['model'] == model]
        pca_counts = model_data['best_pca'].value_counts().sort_index()

        print(f"\n{model_labels[model]}:")
        for pca, count in pca_counts.items():
            print(f"  PCA={pca}: {count}/8 properties ({count/8*100:.0f}%)")

def main():
    print("="*80)
    print("PHASE 6: RADAR CHARTS (ORB FEATURIZER ONLY)")
    print("="*80)

    # Load filtered data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'filtered_n500.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded data: {df.shape}")

    # Prepare radar data for R²
    print("\nPreparing radar data for R²...")
    radar_data_r2 = prepare_radar_data_orb(df, metric='R2')

    # Prepare radar data for Spearman
    print("Preparing radar data for Spearman...")
    radar_data_spearman = prepare_radar_data_orb(df, metric='Spearman')

    # Save PCA choices
    radar_data_combined = radar_data_r2.copy()
    radar_data_combined = radar_data_combined.rename(columns={'value': 'R2'})
    radar_data_combined['Spearman'] = radar_data_spearman['value'].values

    output_path = os.path.join(script_dir, '..', 'data', 'radar_orb_pca_choices.csv')
    radar_data_combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved radar PCA choices to {output_path}")

    # Create radar charts
    print("\nGenerating radar charts...")
    print("  Creating R² radar chart...")
    plot_radar_chart_orb(
        radar_data_r2, metric='R2',
        output_dir=os.path.join(script_dir, '..', 'figures', 'radar_charts')
    )

    print("  Creating Spearman radar chart...")
    plot_radar_chart_orb(
        radar_data_spearman, metric='Spearman',
        output_dir=os.path.join(script_dir, '..', 'figures', 'radar_charts')
    )

    # Analyze radar data
    analyze_radar_data(radar_data_r2, radar_data_spearman)

    print("\n" + "="*80)
    print("PHASE 6 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
