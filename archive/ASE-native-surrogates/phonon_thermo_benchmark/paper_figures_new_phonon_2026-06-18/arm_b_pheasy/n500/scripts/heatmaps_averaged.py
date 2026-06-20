"""
Phase 3: Heatmaps (Averaged with Spearman)

This script:
1. Uses best PCA configs from Phase 1 (averaged across properties)
2. Creates heatmaps for R², RMSE, and Spearman
3. Annotates with PCA choices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style('white')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

def plot_heatmap_averaged(best_configs_df, metric='R2', output_dir='../figures/heatmaps'):
    """
    Heatmap: rows=descriptors, columns=models, color=metric value

    Uses best PCA configs from Phase 1 (averaged across properties)

    Args:
        best_configs_df: DataFrame with best PCA configs from Phase 1
        metric: Metric to plot ('R2', 'RMSE', or 'Spearman')
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pivot data
    pivot = best_configs_df.pivot(
        index='descriptor',
        columns='model',
        values=f'avg_{metric}'
    )

    # Reorder: descriptors (ORB first as it's best), models (GP, MTGP, DGP)
    pivot = pivot.loc[['orb', 'mace', 'uma', 'soap'], ['gp', 'mtgp_2']]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Choose colormap based on metric
    if metric == 'RMSE':
        cmap = 'RdYlGn_r'  # Reversed: red=bad (high), green=good (low)
        fmt = '.2f'
    else:  # R2 or Spearman
        cmap = 'RdYlGn'  # Normal: green=good (high), red=bad (low)
        fmt = '.3f'

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
               ax=ax, cbar_kws={'label': metric},
               linewidths=2, linecolor='white',
               vmin=0 if metric in ['R2', 'Spearman'] else None,
               vmax=1 if metric in ['R2', 'Spearman'] else None)

    # Customize
    ax.set_title(f'Averaged {metric} Across Properties\n(n=500, Best PCA per Surrogate)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Descriptor', fontsize=14, fontweight='bold')

    # Uppercase labels
    ax.set_xticklabels(['GP', 'MTGP'], fontsize=12)
    ax.set_yticklabels(['ORB', 'MACE', 'UMA', 'SOAP'], fontsize=12, rotation=0)

    # Add PCA annotations below heatmap
    pca_annotations = []
    for descriptor in ['orb', 'mace', 'uma', 'soap']:
        for model in ['gp', 'mtgp_2']:
            pca = best_configs_df[
                (best_configs_df['descriptor'] == descriptor) &
                (best_configs_df['model'] == model)
            ]['best_pca'].values[0]

            model_label = {'gp': 'GP', 'mtgp_2': 'MTGP', 'dgp': 'DGP'}[model]
            pca_annotations.append(f"{model_label}-{descriptor.upper()}:{pca}")

    # Split annotations into lines for readability
    n_per_line = 4
    annotation_lines = []
    for i in range(0, len(pca_annotations), n_per_line):
        annotation_lines.append(", ".join(pca_annotations[i:i+n_per_line]))

    fig.text(0.5, 0.01, "Best PCA: " + "\n".join(annotation_lines),
            ha='center', fontsize=8, style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save figure
    output_path = os.path.join(output_dir, f'averaged_{metric}_n500.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()

def main():
    print("="*80)
    print("PHASE 3: HEATMAPS (AVERAGED WITH SPEARMAN)")
    print("="*80)

    # Load best PCA configs from Phase 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_path = os.path.join(script_dir, '..', 'data', 'best_pca_averaged.csv')
    best_configs = pd.read_csv(configs_path)

    print(f"Loaded best PCA configs: {best_configs.shape}")
    print("\nBest PCA configurations:")
    print(best_configs.to_string(index=False))

    # Create heatmaps for each metric
    print("\nGenerating heatmaps...")
    for metric in ['R2', 'RMSE', 'Spearman']:
        print(f"  Creating {metric} heatmap...")
        plot_heatmap_averaged(
            best_configs, metric=metric,
            output_dir=os.path.join(script_dir, '..', 'figures', 'heatmaps')
        )

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
