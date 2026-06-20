"""
gp_regression.py — Single-task GP per property, per descriptor, per split.

Matches notebook cell 5 (ASE_GP_MTGP_DGP_Botorch_multisplit3.ipynb) exactly:
  - np.random.seed(split_idx * 42) for reproducible splits
  - torch.float64 default dtype
  - plot_parity_with_metrics style (figsize=(10,8), metrics box bottom-right)
  - train_property_multi_split: same model used for internal test AND holdout
  - Saves {descriptor}_holdout_summary.csv per descriptor

Usage:
    python gp_regression.py --dataset elastic_tensor_2015 \
        --pca-components 50 --n-train 500 --n-splits 5 --device cuda
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    ALL_DESCRIPTORS, auto_detect_targets, get_featurizer,
    load_matminer_dataset, make_parser, prepare_dataset,
)

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)


# =============================================================================
# Metrics (matching notebook calculate_metrics)
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    denom = np.abs(y_pred) + np.abs(y_true)
    denom[denom == 0] = 1e-8
    smape = np.mean(2 * np.abs(y_pred - y_true) / denom) * 100
    spearman, _ = spearmanr(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'SMAPE': smape, 'Spearman': spearman}


def aggregate_metrics(metrics_list):
    """Compute mean and std of metrics across splits."""
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = {'mean': np.mean(values), 'std': np.std(values)}
    return aggregated


# =============================================================================
# GP Training (matching notebook train_gp)
# =============================================================================

def train_gp(train_x, train_y, device, verbose=False):
    """Train a BoTorch SingleTaskGP model."""
    train_y_2d = train_y.unsqueeze(-1) if train_y.dim() == 1 else train_y
    model = SingleTaskGP(train_x, train_y_2d).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    if verbose:
        print(f"    Model fitted successfully")
    return model


# =============================================================================
# Parity Plot (matching notebook plot_parity_with_metrics exactly)
# =============================================================================

def plot_parity_with_metrics(y_train_true, y_train_pred, y_test_true, y_test_pred,
                              y_train_std=None, y_test_std=None,
                              title="Train/Test Parity Plot", save_path=None):
    """Create parity plot with metrics inset and optional error bars."""
    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    if y_train_std is not None:
        ax.errorbar(y_train_true, y_train_pred, yerr=y_train_std,
                    fmt='o', alpha=0.6, markersize=6, color='blue',
                    ecolor='lightblue', elinewidth=1, capsize=2,
                    label='Train', markeredgecolor='darkblue', markeredgewidth=0.5)
    else:
        ax.scatter(y_train_true, y_train_pred, alpha=0.6, s=50,
                   color='blue', label='Train', edgecolors='darkblue', linewidth=0.5)

    if y_test_std is not None:
        ax.errorbar(y_test_true, y_test_pred, yerr=y_test_std,
                    fmt='o', alpha=0.6, markersize=6, color='red',
                    ecolor='lightcoral', elinewidth=1, capsize=2,
                    label='Test', markeredgecolor='darkred', markeredgewidth=0.5)
    else:
        ax.scatter(y_test_true, y_test_pred, alpha=0.6, s=50,
                   color='red', label='Test', edgecolors='darkred', linewidth=0.5)

    all_y = np.concatenate([y_train_true, y_test_true, y_train_pred, y_test_pred])
    min_val, max_val = all_y.min(), all_y.max()
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin],
            'k--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    metrics_text = "Train Metrics:\n"
    metrics_text += f"R\u00b2 = {train_metrics['R2']:.4f}\n"
    metrics_text += f"RMSE = {train_metrics['RMSE']:.4f}\n"
    metrics_text += f"MAE = {train_metrics['MAE']:.4f}\n"
    metrics_text += f"SMAPE = {train_metrics['SMAPE']:.2f}%\n"
    metrics_text += f"Spearman = {train_metrics['Spearman']:.4f}\n\n"
    metrics_text += "Test Metrics:\n"
    metrics_text += f"R\u00b2 = {test_metrics['R2']:.4f}\n"
    metrics_text += f"RMSE = {test_metrics['RMSE']:.4f}\n"
    metrics_text += f"MAE = {test_metrics['MAE']:.4f}\n"
    metrics_text += f"SMAPE = {test_metrics['SMAPE']:.2f}%\n"
    metrics_text += f"Spearman = {test_metrics['Spearman']:.4f}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    plt.close()
    return train_metrics, test_metrics


# =============================================================================
# Multi-Split Training (matching notebook train_property_multi_split exactly)
# =============================================================================

def train_property_multi_split(X_train_all, y_train_all, X_holdout, y_holdout,
                                device, property_name, n_splits=5, train_ratio=0.8,
                                output_dir="plots"):
    """
    Train GP for a single property across multiple random splits.
    For each split:
      - Plot 1: Train (80%) vs Test (20%) internal split
      - Plot 2: All training samples vs TRUE HOLDOUT (same split model)
    Matches notebook cell 5 exactly: np.random.seed(split_idx * 42).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_test_metrics = []
    all_holdout_metrics = []
    n_total = len(y_train_all)
    n_train = int(train_ratio * n_total)

    print(f"\n  Training samples: {n_total}, Holdout samples: {len(y_holdout)}")

    for split_idx in range(n_splits):
        print(f"\n  --- Split {split_idx + 1}/{n_splits} ---")

        np.random.seed(split_idx * 42)
        perm = np.random.permutation(n_total)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        # Scale features and target
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train_all[train_idx])
        X_test_scaled = feature_scaler.transform(X_train_all[test_idx])

        target_scaler = StandardScaler()
        y_train_scaled = target_scaler.fit_transform(
            y_train_all[train_idx].reshape(-1, 1)).flatten()

        X_train_t = torch.tensor(X_train_scaled, device=device, dtype=torch.float64)
        X_test_t = torch.tensor(X_test_scaled, device=device, dtype=torch.float64)
        y_train_t = torch.tensor(y_train_scaled, device=device)

        # Train
        model = train_gp(X_train_t, y_train_t, device, verbose=True)
        model.eval()

        # === INTERNAL TEST EVALUATION ===
        with torch.no_grad():
            train_posterior = model.posterior(X_train_t)
            train_pred = target_scaler.inverse_transform(
                train_posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)).flatten()
            train_true = y_train_all[train_idx]
            train_std = (train_posterior.variance.squeeze(-1).sqrt().cpu().numpy()
                         * target_scaler.scale_[0])

            test_posterior = model.posterior(X_test_t)
            test_pred = target_scaler.inverse_transform(
                test_posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)).flatten()
            test_true = y_train_all[test_idx]
            test_std = (test_posterior.variance.squeeze(-1).sqrt().cpu().numpy()
                        * target_scaler.scale_[0])

        test_metrics = calculate_metrics(test_true, test_pred)
        all_test_metrics.append(test_metrics)

        plot_parity_with_metrics(
            train_true, train_pred, test_true, test_pred,
            y_train_std=train_std, y_test_std=test_std,
            title=f"{property_name} - Train/Test Split {split_idx+1}/{n_splits}",
            save_path=os.path.join(
                output_dir, f"parity_{property_name}_traintest_split{split_idx+1}.png"))

        print(f"    Internal Test R\u00b2: {test_metrics['R2']:.4f}, "
              f"RMSE: {test_metrics['RMSE']:.4f}")

        # === TRUE HOLDOUT EVALUATION ===
        if X_holdout is not None and len(X_holdout) > 0:
            X_all_train_scaled = feature_scaler.transform(X_train_all)
            X_holdout_scaled = feature_scaler.transform(X_holdout)
            X_all_train_t = torch.tensor(
                X_all_train_scaled, device=device, dtype=torch.float64)
            X_holdout_t = torch.tensor(
                X_holdout_scaled, device=device, dtype=torch.float64)

            with torch.no_grad():
                all_train_posterior = model.posterior(X_all_train_t)
                all_train_pred = target_scaler.inverse_transform(
                    all_train_posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)
                ).flatten()
                all_train_std = (
                    all_train_posterior.variance.squeeze(-1).sqrt().cpu().numpy()
                    * target_scaler.scale_[0])

                holdout_posterior = model.posterior(X_holdout_t)
                holdout_pred = target_scaler.inverse_transform(
                    holdout_posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)
                ).flatten()
                holdout_std = (
                    holdout_posterior.variance.squeeze(-1).sqrt().cpu().numpy()
                    * target_scaler.scale_[0])

            holdout_metrics = calculate_metrics(y_holdout, holdout_pred)
            all_holdout_metrics.append(holdout_metrics)

            plot_parity_with_metrics(
                y_train_all, all_train_pred, y_holdout, holdout_pred,
                y_train_std=all_train_std, y_test_std=holdout_std,
                title=f"RBF Kernel - Holdout Prediction on {property_name} (Split {split_idx+1})",
                save_path=os.path.join(
                    output_dir, f"parity_{property_name}_holdout_split{split_idx+1}.png"))

            print(f"    HOLDOUT R\u00b2: {holdout_metrics['R2']:.4f}, "
                  f"RMSE: {holdout_metrics['RMSE']:.4f}")

    return {
        'test': aggregate_metrics(all_test_metrics),
        'holdout': aggregate_metrics(all_holdout_metrics) if all_holdout_metrics else None,
        'all_holdout_metrics': all_holdout_metrics,
    }


# =============================================================================
# Run one descriptor
# =============================================================================

def run_descriptor(descriptor, df_train, df_holdout, target_cols, args, output_dir):
    """Featurize and run all properties for one descriptor."""
    print(f"\n{'='*70}")
    print(f"DESCRIPTOR: {descriptor.upper()}")
    print(f"{'='*70}")

    featurizer = get_featurizer(descriptor, args.pca_components, args.device)
    X_train_all = featurizer.fit_transform(df_train)
    X_holdout_all = featurizer.transform(df_holdout)
    print(f"  Features: train={X_train_all.shape}, holdout={X_holdout_all.shape}")

    device = torch.device(args.device)
    plot_dir = os.path.join(output_dir, 'plots', descriptor)
    os.makedirs(plot_dir, exist_ok=True)

    all_property_results = {}

    for prop in target_cols:
        if prop not in df_train.columns:
            print(f"\n  Skipping {prop} — not in dataset")
            continue

        valid_train = df_train[prop].notna().values
        valid_holdout = df_holdout[prop].notna().values

        if valid_train.sum() < 50:
            print(f"\n  Skipping {prop} — only {valid_train.sum()} valid training samples")
            continue

        print(f"\n  {'#'*60}")
        print(f"  PROPERTY: {prop}")

        y_train_all = df_train.loc[valid_train, prop].values.astype(float)
        y_holdout = df_holdout.loc[valid_holdout, prop].values.astype(float)
        X_train_valid = X_train_all[valid_train]
        X_holdout_valid = X_holdout_all[valid_holdout]

        prop_dir = os.path.join(plot_dir, prop.replace('/', '_'))

        results = train_property_multi_split(
            X_train_valid, y_train_all,
            X_holdout_valid, y_holdout,
            device, prop,
            n_splits=args.n_splits,
            train_ratio=0.8,
            output_dir=prop_dir,
        )

        all_property_results[prop] = results

    return all_property_results


# =============================================================================
# Save results
# =============================================================================

def save_results(results, output_dir, descriptor, dataset_name):
    """Save per-property holdout metrics summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY: GP-{descriptor.upper()} HOLDOUT Metrics (mean +/- std)")
    print(f"{'='*120}")
    header = (f"{'Property':<22} | {'R2':<18} | {'RMSE':<18} | "
              f"{'MAE':<18} | {'SMAPE (%)':<18} | {'Spearman':<18}")
    print(header)
    print("-" * 120)

    for prop, res in results.items():
        if res['holdout'] is None:
            continue
        agg = res['holdout']
        r2_str = f"{agg['R2']['mean']:.4f} +/- {agg['R2']['std']:.4f}"
        rmse_str = f"{agg['RMSE']['mean']:.4f} +/- {agg['RMSE']['std']:.4f}"
        mae_str = f"{agg['MAE']['mean']:.4f} +/- {agg['MAE']['std']:.4f}"
        smape_str = f"{agg['SMAPE']['mean']:.2f} +/- {agg['SMAPE']['std']:.2f}"
        spearman_str = (f"{agg['Spearman']['mean']:.4f} +/- "
                        f"{agg['Spearman']['std']:.4f}")
        print(f"{prop:<22} | {r2_str:<18} | {rmse_str:<18} | "
              f"{mae_str:<18} | {smape_str:<18} | {spearman_str:<18}")

        summary_rows.append({
            'Property': prop,
            'R2_mean': agg['R2']['mean'], 'R2_std': agg['R2']['std'],
            'RMSE_mean': agg['RMSE']['mean'], 'RMSE_std': agg['RMSE']['std'],
            'MAE_mean': agg['MAE']['mean'], 'MAE_std': agg['MAE']['std'],
            'SMAPE_mean': agg['SMAPE']['mean'], 'SMAPE_std': agg['SMAPE']['std'],
            'Spearman_mean': agg['Spearman']['mean'],
            'Spearman_std': agg['Spearman']['std'],
        })

    print("-" * 120)

    if summary_rows:
        csv_path = os.path.join(output_dir, f"gp_{descriptor}_holdout_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"\nSaved HOLDOUT summary to: {csv_path}")

    return summary_rows


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser('gp')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = args.output_dir or os.path.join('results', 'gp', args.dataset)
    os.makedirs(out_root, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"n_train: {args.n_train}, n_splits: {args.n_splits}, PCA: {args.pca_components}")

    df = load_matminer_dataset(args.dataset)
    print(f"Dataset loaded: {len(df)} samples")

    df_train, df_holdout = prepare_dataset(df, args.n_train, args.seed)
    target_cols = auto_detect_targets(df_train)
    print(f"Auto-detected target columns ({len(target_cols)}): {target_cols}")
    print(f"Training set: {len(df_train)}, Holdout set: {len(df_holdout)}")

    descriptors = ALL_DESCRIPTORS if args.descriptor == 'all' else [args.descriptor]

    for descriptor in descriptors:
        results = run_descriptor(
            descriptor, df_train, df_holdout, target_cols, args, out_root
        )
        save_results(results, out_root, descriptor, args.dataset)

    print(f"\nAll results saved to: {out_root}")
    print("\nGP regression — COMPLETE!")


if __name__ == '__main__':
    main()
