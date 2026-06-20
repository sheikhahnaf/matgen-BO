"""
Phase 4: Property Difficulty Matrix

This script:
1. Creates separate difficulty matrices for each surrogate (GP, MTGP, DGP)
2. Uses best PCA on average from Phase 1
3. Shows R² for each (descriptor, property) pair
4. Helps identify which properties are hardest to predict
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('white')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

def get_property_list():
    """Return 8 properties (excluding kpoint_density)"""
    return [
        'K_Voigt', 'K_VRH', 'K_Reuss',
        'G_Voigt', 'G_VRH', 'G_Reuss',
        'elastic_anisotropy', 'poisson_ratio'
    ]

def create_property_difficulty_matrix_per_surrogate(df, best_pca_avg):
    """
    Create 3 separate difficulty matrices (one per surrogate: GP, MTGP, DGP)

    For each surrogate:
    - Use best PCA on average (from Phase 1)
    - Show R² for each (descriptor, property) pair
    - Rank properties by difficulty

    Args:
        df: Filtered n=500 data
        best_pca_avg: DataFrame with best PCA choices from Phase 1

    Returns:
        Dict of DataFrames: {'gp': df_gp, 'mtgp_2': df_mtgp, 'dgp': df_dgp}
    """
    results = {}
    properties = get_property_list()
    descriptors = ['mace', 'orb', 'soap', 'uma']

    for model in ['gp', 'mtgp_2', 'dgp']:
        property_difficulty = []

        for descriptor in descriptors:
            # Get best PCA for this model-descriptor
            best_pca = best_pca_avg[
                (best_pca_avg['model'] == model) &
                (best_pca_avg['descriptor'] == descriptor)
            ]['best_pca'].values[0]

            # Get R² for each property at best PCA
            for prop in properties:
                r2 = df[
                    (df['model'] == model) &
                    (df['descriptor'] == descriptor) &
                    (df['pca_components'] == best_pca) &
                    (df['property'] == prop) &
                    (df['metric'] == 'R2')
                ]['mean'].values[0]

                property_difficulty.append({
                    'descriptor': descriptor,
                    'property': prop,
                    'R2': r2,
                    'best_pca': best_pca
                })

        results[model] = pd.DataFrame(property_difficulty)

    return results

def plot_property_difficulty_matrix(difficulty_dicts, output_dir='../figures/property_difficulty'):
    """
    Create 3 heatmaps (one per surrogate)

    Each heatmap: rows=properties, columns=descriptors, color=R²

    Args:
        difficulty_dicts: Dict of DataFrames for each model
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    properties = get_property_list()
    descriptors = ['orb', 'mace', 'uma', 'soap']  # ORB first (best)
    models = ['gp', 'mtgp_2', 'dgp']
    model_labels = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}

    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    for idx, model in enumerate(models):
        ax = axes[idx]
        df = difficulty_dicts[model]

        # Pivot
        pivot = df.pivot(index='property', columns='descriptor', values='R2')

        # Reorder
        pivot = pivot.loc[properties, descriptors]

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax,
                   cbar_kws={'label': 'R²'},
                   linewidths=2, linecolor='white')

        ax.set_title(f'{model_labels[model]} Property Difficulty\n(n=500, Best PCA on Average)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Descriptor', fontsize=12, fontweight='bold')
        ax.set_ylabel('Property' if idx == 0 else '', fontsize=12, fontweight='bold')

        # Uppercase descriptor labels
        ax.set_xticklabels([d.upper() for d in descriptors], fontsize=10)

        if idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(properties, fontsize=10, rotation=0)

    plt.suptitle('Property Prediction Difficulty by Surrogate and Descriptor',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'difficulty_matrix_per_surrogate_n500.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def analyze_difficulty(difficulty_dicts):
    """
    Analyze and print property difficulty rankings

    Args:
        difficulty_dicts: Dict of DataFrames for each model
    """
    print("\n" + "="*80)
    print("PROPERTY DIFFICULTY ANALYSIS")
    print("="*80)

    properties = get_property_list()

    for model in ['gp', 'mtgp_2', 'dgp']:
        df = difficulty_dicts[model]
        model_label = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}[model]

        print(f"\n{model_label}:")
        print("-" * 40)

        # Average R² across descriptors for each property
        property_avg = df.groupby('property')['R2'].mean().sort_values(ascending=False)

        print("Properties ranked by average R² (easiest → hardest):")
        for i, (prop, r2) in enumerate(property_avg.items(), 1):
            print(f"  {i}. {prop:25s} R²={r2:.3f}")

def main():
    print("="*80)
    print("PHASE 4: PROPERTY DIFFICULTY MATRIX")
    print("="*80)

    # Load filtered data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'filtered_n500.csv')
    df = pd.read_csv(data_path)

    # Load best PCA configs from Phase 1
    configs_path = os.path.join(script_dir, '..', 'data', 'best_pca_averaged.csv')
    best_pca_avg = pd.read_csv(configs_path)

    print(f"Loaded data: {df.shape}")
    print(f"Loaded best PCA configs: {best_pca_avg.shape}")

    # Create difficulty matrices
    print("\nCreating property difficulty matrices...")
    difficulty_dicts = create_property_difficulty_matrix_per_surrogate(df, best_pca_avg)

    # Plot matrices
    print("\nGenerating plot...")
    plot_property_difficulty_matrix(
        difficulty_dicts,
        output_dir=os.path.join(script_dir, '..', 'figures', 'property_difficulty')
    )

    # Analyze difficulty
    analyze_difficulty(difficulty_dicts)

    # Save data
    combined_df = []
    for model, df_diff in difficulty_dicts.items():
        df_diff['model'] = model
        combined_df.append(df_diff)

    combined_df = pd.concat(combined_df, ignore_index=True)
    output_path = os.path.join(script_dir, '..', 'data', 'property_difficulty_per_surrogate.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved property difficulty data to {output_path}")

    print("\n" + "="*80)
    print("PHASE 4 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
