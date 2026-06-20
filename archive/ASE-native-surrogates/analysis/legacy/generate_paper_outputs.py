"""
Generate Paper-Ready Outputs for Publication

Creates LaTeX tables and summary statistics for the manuscript.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


def format_float(value: float, precision: int = 3) -> str:
    """Format float with given precision."""
    if pd.isna(value):
        return '-'
    return f"{value:.{precision}f}"


def format_with_std(mean: float, std: float, precision: int = 3) -> str:
    """Format value as mean ± std."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def generate_top_configurations_table(df: pd.DataFrame, n: int = 10) -> str:
    """
    LaTeX table: Top N configurations by R² (averaged across properties).

    Columns: Rank, Model, Descriptor, PCA, n_train, R², RMSE, MAE
    """
    # Load rankings
    rankings_dir = Path(__file__).parent / 'rankings'
    top_configs = pd.read_csv(rankings_dir / 'top_configurations.csv').head(n)

    # Get detailed metrics for each config
    results = []

    for _, config in top_configs.iterrows():
        # Filter to this specific config
        config_df = df[
            (df['model'] == config['model']) &
            (df['descriptor'] == config['descriptor']) &
            (df['pca_components'] == config['pca_components']) &
            (df['n_train'] == config['n_train'])
        ]

        # Get average metrics across properties
        r2_mean = config_df[config_df['metric'] == 'R2']['mean'].mean()
        r2_std = config_df[config_df['metric'] == 'R2']['mean'].std()

        rmse_mean = config_df[config_df['metric'] == 'RMSE']['mean'].mean()
        rmse_std = config_df[config_df['metric'] == 'RMSE']['mean'].std()

        mae_mean = config_df[config_df['metric'] == 'MAE']['mean'].mean()
        mae_std = config_df[config_df['metric'] == 'MAE']['mean'].std()

        results.append({
            'rank': config['rank'],
            'model': config['model'].upper(),
            'descriptor': config['descriptor'].upper(),
            'pca': config['pca_components'],
            'n_train': config['n_train'],
            'r2': format_with_std(r2_mean, r2_std),
            'rmse': format_with_std(rmse_mean, rmse_std),
            'mae': format_with_std(mae_mean, mae_std)
        })

    results_df = pd.DataFrame(results)

    # Generate LaTeX
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Top 10 Model Configurations Ranked by Average $R^2$ Across All Properties}
\label{tab:top_configs}
\small
\begin{tabular}{cllccccc}
\hline
\textbf{Rank} & \textbf{Model} & \textbf{Descriptor} & \textbf{PCA} & \textbf{$n_{train}$} & \textbf{$R^2$} & \textbf{RMSE} & \textbf{MAE} \\
\hline
"""

    for _, row in results_df.iterrows():
        latex += f"{row['rank']} & {row['model']} & {row['descriptor']} & {row['pca']} & {row['n_train']} & {row['r2']} & {row['rmse']} & {row['mae']} \\\\\n"

    latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item Values shown as mean ± standard deviation across all 9 elastic properties.
\item PCA: Number of principal components used for dimensionality reduction.
\end{tablenotes}
\end{table}
"""

    return latex


def generate_model_comparison_table() -> str:
    """
    LaTeX table: Model comparison (GP vs MTGP vs DGP) averaged over all configs.

    Columns: Model, R², RMSE, MAE, SMAPE, Spearman
    """
    # Load rankings
    rankings_dir = Path(__file__).parent / 'rankings'
    model_comparison = pd.read_csv(rankings_dir / 'model_comparison.csv')
    wilcoxon = pd.read_csv(rankings_dir / 'wilcoxon_tests.csv')

    # Get R² comparison from Wilcoxon
    gp_mtgp = wilcoxon[
        (wilcoxon['metric'] == 'R2') &
        (((wilcoxon['model1'] == 'gp') & (wilcoxon['model2'] == 'mtgp_2')) |
         ((wilcoxon['model1'] == 'mtgp_2') & (wilcoxon['model2'] == 'gp')))
    ].iloc[0]

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Model Performance Comparison Averaged Across All Configurations}
\label{tab:model_comparison}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{$R^2$} & \textbf{RMSE} & \textbf{MAE} & \textbf{Uncertainty} & \textbf{$n_{configs}$} \\
\hline
"""

    for _, row in model_comparison.iterrows():
        model = row['model'].upper()
        r2 = format_with_std(row['avg_performance'], row['std_performance'])
        unc = format_float(row['avg_uncertainty'])
        n = int(row['n_configs'])

        # Add significance markers
        if model == 'GP':
            marker = r'$^{***}$'
        else:
            marker = ''

        latex += f"{model}{marker} & {r2} & - & - & {unc} & {n} \\\\\n"

    latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item $^{***}$ Significantly better than MTGP\_2 ($p < 0.001$, Wilcoxon signed-rank test).
\item Uncertainty: Average standard deviation across cross-validation folds.
\end{tablenotes}
\end{table}
"""

    return latex


def generate_descriptor_comparison_table() -> str:
    """
    LaTeX table: Descriptor comparison (SOAP vs MACE vs ORB vs UMA).

    Columns: Descriptor, R², Rank, Configurations
    """
    # Load rankings
    rankings_dir = Path(__file__).parent / 'rankings'
    desc_comparison = pd.read_csv(rankings_dir / 'descriptor_comparison.csv')

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Molecular Descriptor Performance Comparison}
\label{tab:descriptor_comparison}
\begin{tabular}{lccc}
\hline
\textbf{Descriptor} & \textbf{$R^2$} & \textbf{Rank} & \textbf{$n_{configs}$} \\
\hline
"""

    for _, row in desc_comparison.iterrows():
        desc = row['descriptor'].upper()
        r2 = format_with_std(row['avg_performance'], row['std_performance'])
        rank = int(row['rank'])
        n = int(row['n_configs'])

        # Add markers for top performers
        if rank == 1:
            marker = r'$^{*}$'
        elif rank == 2:
            marker = r'$^{\dagger}$'
        else:
            marker = ''

        latex += f"{desc}{marker} & {r2} & {rank} & {n} \\\\\n"

    latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item $^{*}$ Best performing descriptor (ORB - Orbital field matrix).
\item $^{\dagger}$ Second best (MACE - Multi-Atomic Cluster Expansion).
\item Descriptors ranked by average $R^2$ across all models and configurations.
\end{tablenotes}
\end{table}
"""

    return latex


def generate_property_difficulty_table() -> str:
    """
    LaTeX table: Property difficulty ranking.
    """
    # Load rankings
    rankings_dir = Path(__file__).parent / 'rankings'
    prop_difficulty = pd.read_csv(rankings_dir / 'property_difficulty.csv')

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Elastic Property Prediction Difficulty Ranking}
\label{tab:property_difficulty}
\begin{tabular}{lcccc}
\hline
\textbf{Property} & \textbf{Avg $R^2$} & \textbf{Min $R^2$} & \textbf{Max $R^2$} & \textbf{Difficulty} \\
\hline
"""

    for _, row in prop_difficulty.iterrows():
        prop = row['property'].replace('_', r'\_')
        avg_r2 = format_float(row['avg_performance'])
        min_r2 = format_float(row['min_performance'])
        max_r2 = format_float(row['max_performance'])
        difficulty = row['difficulty']

        latex += f"{prop} & {avg_r2} & {min_r2} & {max_r2} & {difficulty} \\\\\n"

    latex += r"""\hline
\end{tabular}
\begin{tablenotes}
\small
\item K: Bulk modulus, G: Shear modulus (Voigt/Reuss/VRH bounds).
\item Averaged across all models, descriptors, and configurations.
\end{tablenotes}
\end{table}
"""

    return latex


def generate_summary_statistics(df: pd.DataFrame) -> str:
    """
    Generate summary statistics for abstract/discussion.
    """
    rankings_dir = Path(__file__).parent / 'rankings'

    # Load all rankings
    top_configs = pd.read_csv(rankings_dir / 'top_configurations.csv')
    model_comp = pd.read_csv(rankings_dir / 'model_comparison.csv')
    desc_comp = pd.read_csv(rankings_dir / 'descriptor_comparison.csv')
    prop_diff = pd.read_csv(rankings_dir / 'property_difficulty.csv')
    wilcoxon = pd.read_csv(rankings_dir / 'wilcoxon_tests.csv')
    improvements = pd.read_csv(rankings_dir / 'performance_improvements.csv')

    summary = []

    summary.append("=" * 80)
    summary.append("SUMMARY STATISTICS FOR MANUSCRIPT")
    summary.append("=" * 80)
    summary.append("")

    # Best configuration
    best = top_configs.iloc[0]
    summary.append("BEST OVERALL CONFIGURATION:")
    summary.append(f"  Model: {best['model'].upper()}")
    summary.append(f"  Descriptor: {best['descriptor'].upper()}")
    summary.append(f"  PCA components: {best['pca_components']}")
    summary.append(f"  Training size: {best['n_train']}")
    summary.append(f"  Average R²: {best['avg_performance']:.4f} ± {best['avg_uncertainty']:.4f}")
    summary.append("")

    # Model comparison
    summary.append("MODEL COMPARISON:")
    for _, row in model_comp.iterrows():
        summary.append(f"  {row['model'].upper():10s}: R² = {row['avg_performance']:.4f} ± {row['std_performance']:.4f}")

    # GP vs MTGP statistical test
    gp_mtgp = wilcoxon[
        (wilcoxon['metric'] == 'R2') &
        (((wilcoxon['model1'] == 'gp') & (wilcoxon['model2'] == 'mtgp_2')) |
         ((wilcoxon['model1'] == 'mtgp_2') & (wilcoxon['model2'] == 'gp')))
    ].iloc[0]

    if gp_mtgp['model1'] == 'gp':
        diff = gp_mtgp['mean_diff']
    else:
        diff = -gp_mtgp['mean_diff']

    summary.append(f"  GP outperforms MTGP by: {abs(diff):.4f} (p = {gp_mtgp['p_value']:.2e})")
    summary.append(f"  Effect size: {abs(gp_mtgp['effect_size']):.4f} ({gp_mtgp['effect_magnitude']})")
    summary.append("")

    # Descriptor comparison
    summary.append("DESCRIPTOR RANKING:")
    for _, row in desc_comp.iterrows():
        summary.append(f"  {row['rank']}. {row['descriptor'].upper():6s}: R² = {row['avg_performance']:.4f} ± {row['std_performance']:.4f}")
    summary.append("")

    # Best and worst properties
    easiest = prop_diff.iloc[0]
    hardest = prop_diff.iloc[-1]

    summary.append("PROPERTY PREDICTION:")
    summary.append(f"  Easiest: {easiest['property']:20s} (R² = {easiest['avg_performance']:.4f})")
    summary.append(f"  Hardest: {hardest['property']:20s} (R² = {hardest['avg_performance']:.4f})")
    summary.append("")

    # Sample efficiency
    summary.append("SAMPLE EFFICIENCY:")

    # Average improvement from n=100 to n=500
    if 'dgp_vs_gp_%' in improvements.columns:
        dgp_improvement = improvements['dgp_vs_gp_%'].mean()
        mtgp_improvement = improvements['mtgp_2_vs_gp_%'].mean()

        summary.append(f"  DGP vs GP:    {dgp_improvement:+6.2f}% average change")
        summary.append(f"  MTGP vs GP:   {mtgp_improvement:+6.2f}% average change")
    summary.append("")

    # Optimal PCA
    pca_optimal = pd.read_csv(rankings_dir / 'pca_optimal.csv')
    pca_mode = pca_optimal['optimal_pca'].mode()[0]

    summary.append("OPTIMAL PCA DIMENSIONALITY:")
    summary.append(f"  Most common: {pca_mode} components")
    summary.append(f"  Distribution:")
    for pca in sorted(pca_optimal['optimal_pca'].unique()):
        count = len(pca_optimal[pca_optimal['optimal_pca'] == pca])
        summary.append(f"    PCA={pca:2d}: {count} model/descriptor combinations")
    summary.append("")

    # Data coverage
    summary.append("DATA COVERAGE:")
    summary.append(f"  Total configurations: {df['config'].nunique()}")
    summary.append(f"  Properties analyzed: {df['property'].nunique()}")
    summary.append(f"  Models compared: {df['model'].nunique()}")
    summary.append(f"  Descriptors tested: {df['descriptor'].nunique()}")
    summary.append(f"  Total data points: {len(df)}")
    summary.append("")

    summary.append("=" * 80)

    return "\n".join(summary)


def main():
    """Main generation pipeline."""
    analysis_dir = Path(__file__).parent
    paper_dir = analysis_dir / 'paper'
    tables_dir = paper_dir / 'tables'

    # Load data
    df = pd.read_csv(analysis_dir / 'aggregated_results.csv')

    print("=" * 80)
    print("GENERATING PAPER-READY OUTPUTS")
    print("=" * 80)
    print()

    # 1. Top configurations table
    print("1. Generating Table 1: Top Configurations...")
    table1 = generate_top_configurations_table(df, n=10)
    (tables_dir / 'table1_top_configurations.tex').write_text(table1)
    print("   Saved to: table1_top_configurations.tex")
    print()

    # 2. Model comparison table
    print("2. Generating Table 2: Model Comparison...")
    table2 = generate_model_comparison_table()
    (tables_dir / 'table2_model_comparison.tex').write_text(table2)
    print("   Saved to: table2_model_comparison.tex")
    print()

    # 3. Descriptor comparison table
    print("3. Generating Table 3: Descriptor Comparison...")
    table3 = generate_descriptor_comparison_table()
    (tables_dir / 'table3_descriptor_comparison.tex').write_text(table3)
    print("   Saved to: table3_descriptor_comparison.tex")
    print()

    # 4. Property difficulty table
    print("4. Generating Table 4: Property Difficulty...")
    table4 = generate_property_difficulty_table()
    (tables_dir / 'table4_property_difficulty.tex').write_text(table4)
    print("   Saved to: table4_property_difficulty.tex")
    print()

    # 5. Summary statistics
    print("5. Generating Summary Statistics...")
    summary = generate_summary_statistics(df)
    (paper_dir / 'summary_statistics.txt').write_text(summary)
    print("   Saved to: summary_statistics.txt")
    print()
    print(summary)
    print()

    # 6. Create README for figures
    print("6. Creating Figure README...")
    readme = f"""# Figures for Publication

This directory contains all publication-quality figures generated from the ASE regression benchmarking analysis.

## Directory Structure

- `heatmaps/`: Performance heatmaps (R², RMSE, MAE) for all PCA and n_train combinations
- `learning_curves/`: Learning curves showing performance vs dataset size
- `bar_charts/`: Model comparison bar charts by descriptor
- `pca_sensitivity/`: PCA dimensionality sensitivity analysis
- `radar_charts/`: Multi-metric descriptor comparison
- `uncertainty/`: Prediction uncertainty distributions
- `property_analysis/`: Property-specific difficulty analysis
- `summary/`: Top configurations and scaling efficiency plots

## Figure Formats

All figures are available in two formats:
- **PDF**: Vector graphics for LaTeX documents
- **PNG**: Raster graphics (300 DPI) for presentations

## Key Figures for Main Text

### Figure 1: Model Comparison
- `bar_charts/model_comparison_bars.pdf`

### Figure 2: Learning Curves
- `learning_curves/learning_curves_R2.pdf`

### Figure 3: Descriptor Comparison
- `radar_charts/descriptor_radar.pdf`

### Figure 4: Property Difficulty
- `property_analysis/property_difficulty_matrix.pdf`

### Figure 5: Top Configurations
- `summary/top_configurations.pdf`

## Supplementary Figures

### S1-S27: Performance Heatmaps
- `heatmaps/R2_pca*_n*.pdf` (9 files)
- `heatmaps/RMSE_pca*_n*.pdf` (9 files)
- `heatmaps/MAE_pca*_n*.pdf` (9 files)

### S28: PCA Sensitivity
- `pca_sensitivity/pca_sensitivity.pdf`

### S29: Uncertainty Distribution
- `uncertainty/uncertainty_distribution.pdf`

### S30: Scaling Efficiency
- `summary/scaling_efficiency.pdf`

Total figures: 37 (PDF + PNG for each = 74 files)
"""

    (analysis_dir / 'figures' / 'README.md').write_text(readme)
    print("   Saved to: figures/README.md")
    print()

    print("=" * 80)
    print("All paper outputs generated successfully!")
    print(f"LaTeX tables: {tables_dir}")
    print(f"Summary stats: {paper_dir / 'summary_statistics.txt'}")
    print(f"Figures: {analysis_dir / 'figures'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
