"""
mtgp_regression.py — Multi-task GP across all properties jointly, per descriptor.

Matches notebook MTGP SOAP cell (ASE_GP_MTGP_DGP_Botorch_multisplit3.ipynb) exactly:
  - torch.float64 default dtype
  - np.random.seed(split_idx * 42) on raw samples (not pairs)
  - prepare_mtgp_data_from_arrays: iterates samples x tasks, skips NaN
  - Per-task StandardScaler fit on training portion of Y_train per split
  - train_mtgp -> MultiTaskGP + fit_gpytorch_mll
  - predict_mtgp appends task_idx column (raw, not inverse-transformed)
  - analyze_mtgp_hyperparameters on split 1 model
  - plot_parity_with_metrics style (figsize=(10,8), metrics text bottom-right)
  - Saves mtgp_{descriptor}_holdout_summary.csv per descriptor

Usage:
    python mtgp_regression.py --dataset elastic_tensor_2015 \
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
import gpytorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
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
# MTGP Data Preparation (matching notebook prepare_mtgp_data_from_arrays)
# =============================================================================

def prepare_mtgp_data_from_arrays(X, Y, task_names):
    """
    Prepare data in MTGP format: X with task index appended, Y flattened.

    BoTorch MultiTaskGP expects:
      - train_X: [n_total_obs, n_features + 1] with task index as last column
      - train_Y: [n_total_obs, 1]

    We iterate over samples x tasks, skipping NaN entries.
    Matches notebook exactly.
    """
    n_samples, n_tasks = Y.shape
    n_features = X.shape[1]
    total_obs = int(np.sum(~np.isnan(Y)))

    train_X = np.zeros((total_obs, n_features + 1))
    train_Y = np.zeros(total_obs)

    idx = 0
    for i in range(n_samples):
        for task in range(n_tasks):
            if not np.isnan(Y[i, task]):
                train_X[idx, :-1] = X[i]
                train_X[idx, -1] = task
                train_Y[idx] = Y[i, task]
                idx += 1

    return train_X, train_Y


# =============================================================================
# MTGP Training (matching notebook train_mtgp)
# =============================================================================

def train_mtgp(train_X, train_Y, num_tasks, device, verbose=False):
    """Train a BoTorch MultiTaskGP model."""
    train_X_t = torch.tensor(train_X, device=device, dtype=torch.float64)
    train_Y_t = torch.tensor(train_Y, device=device, dtype=torch.float64).unsqueeze(-1)

    model = MultiTaskGP(
        train_X=train_X_t,
        train_Y=train_Y_t,
        task_feature=-1,
    ).to(device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    if verbose:
        print(f"    MTGP fitted successfully with {num_tasks} tasks")

    return model


# =============================================================================
# MTGP Prediction (matching notebook predict_mtgp)
# =============================================================================

def predict_mtgp(model, X, task_idx, device):
    """Get predictions for a specific task (returns scaled mean, std)."""
    n_samples = X.shape[0]
    X_with_task = np.zeros((n_samples, X.shape[1] + 1))
    X_with_task[:, :-1] = X
    X_with_task[:, -1] = task_idx

    X_t = torch.tensor(X_with_task, device=device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X_t)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()

    return mean, std


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
# MTGP Hyperparameter Analysis (matching notebook analyze_mtgp_hyperparameters)
# =============================================================================

def analyze_mtgp_hyperparameters(model, task_names, save_dir="mtgp_analysis"):
    """
    Comprehensive analysis of learned MTGP hyperparameters.
    Matches notebook analyze_mtgp_hyperparameters exactly.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    analysis = {}

    print("\n" + "=" * 80)
    print("1. TASK COVARIANCE ANALYSIS (ICM)")
    print("=" * 80)

    try:
        # BoTorch MultiTaskGP uses ProductKernel: kernels[0]=data kernel, kernels[1]=IndexKernel (task correlations)
        # IndexKernel provides the ICM structure with covar_factor and var
        task_covar_kernel = model.covar_module.kernels[1]
        covar_factor = task_covar_kernel.covar_factor.detach().cpu().numpy()
        var = task_covar_kernel.var.detach().cpu().numpy()

        print(f"\nICM covar_factor shape: {covar_factor.shape} (num_tasks x rank)")
        print(f"ICM var (diagonal noise per task): shape={var.shape}")
        print(f"  Values: {var}")

        task_covariance = covar_factor @ covar_factor.T + np.diag(var)
        diag = np.sqrt(np.diag(task_covariance))
        diag[diag == 0] = 1e-8
        task_correlation = task_covariance / np.outer(diag, diag)

        analysis['covar_factor'] = covar_factor
        analysis['task_var'] = var
        analysis['task_covariance'] = task_covariance
        analysis['task_correlation'] = task_correlation

        print("\nLearned Task Correlation Matrix:")
        short_names = [name[:15] for name in task_names]
        header = f"{'':>16}" + "".join([f"{n:>16}" for n in short_names])
        print(header)
        for i, name in enumerate(short_names):
            row = (f"{name:>16}" +
                   "".join([f"{task_correlation[i,j]:>16.3f}"
                             for j in range(len(task_names))]))
            print(row)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        im = axes[0].imshow(task_correlation, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(len(task_names)))
        axes[0].set_yticks(range(len(task_names)))
        axes[0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticklabels(short_names, fontsize=8)
        axes[0].set_title('Learned Task Correlations (ICM)\nRed=positive, Blue=negative')
        plt.colorbar(im, ax=axes[0])
        for i in range(len(task_names)):
            for j in range(len(task_names)):
                axes[0].text(j, i, f'{task_correlation[i,j]:.2f}',
                             ha='center', va='center', fontsize=6,
                             color='white' if abs(task_correlation[i,j]) > 0.5 else 'black')

        im2 = axes[1].imshow(covar_factor, aspect='auto', cmap='coolwarm')
        axes[1].set_xlabel('Latent Factor Index')
        axes[1].set_ylabel('Task')
        axes[1].set_yticks(range(len(task_names)))
        axes[1].set_yticklabels(short_names, fontsize=8)
        axes[1].set_title('ICM Covariance Factor\n(How each task loads on latent factors)')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'task_correlations_icm.png'),
                    dpi=200, bbox_inches='tight')
        print(f"\nSaved: {save_dir}/task_correlations_icm.png")
        plt.close()

    except Exception as e:
        print(f"  Could not extract task covariance: {e}")
        try:
            print("\n  Trying alternative: model.covar_module...")
            for name, param in model.named_parameters():
                if 'task' in name.lower() or 'covar_factor' in name.lower():
                    print(f"    Found: {name}, shape={param.shape}")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("2. DATA KERNEL ANALYSIS (Feature Importance)")
    print("=" * 80)

    try:
        # BoTorch MultiTaskGP: model.covar_module.kernels[0] is the data kernel (RBF/Matern over features)
        data_kernel = model.covar_module.kernels[0]
        print(f"  Found data kernel at: model.covar_module.kernels[0]")
        print(f"  Data kernel type: {type(data_kernel).__name__}")

        if data_kernel is not None:
            try:
                # ScaleKernel wraps base_kernel, otherwise access lengthscale directly
                if hasattr(data_kernel, 'base_kernel'):
                    kernel_with_ls = data_kernel.base_kernel
                else:
                    kernel_with_ls = data_kernel

                ls = kernel_with_ls.lengthscale.detach().cpu().numpy().squeeze()

                print(f"\nData kernel lengthscales shape: {ls.shape}")
                analysis['data_lengthscales'] = ls

                if ls.ndim == 0:
                    print(f"  Single lengthscale (isotropic): {ls:.4f}")
                else:
                    print(f"  Top 10 most important features (shortest lengthscale):")
                    sorted_idx = np.argsort(ls)
                    for rank, idx in enumerate(sorted_idx[:10]):
                        print(f"    Rank {rank+1}: Feature {idx}, "
                              f"lengthscale = {ls[idx]:.4f}")

                    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                    axes[0].bar(range(len(ls)), ls)
                    axes[0].set_xlabel('Input Feature Index')
                    axes[0].set_ylabel('Lengthscale')
                    axes[0].set_title('Data Kernel ARD Lengthscales\n(shorter = more important)')

                    importance = 1.0 / (ls + 1e-6)
                    importance = importance / importance.sum()
                    axes[1].bar(range(len(importance)), importance)
                    axes[1].set_xlabel('Input Feature Index')
                    axes[1].set_ylabel('Relative Importance (1/lengthscale)')
                    axes[1].set_title('Feature Importance')

                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'data_kernel_lengthscales.png'),
                                dpi=200, bbox_inches='tight')
                    print(f"\nSaved: {save_dir}/data_kernel_lengthscales.png")
                    plt.close()

            except Exception as e:
                print(f"  Could not extract lengthscales: {e}")

            try:
                if hasattr(data_kernel, 'outputscale'):
                    os_val = data_kernel.outputscale.detach().cpu().item()
                    print(f"\nData kernel output scale: {os_val:.4f}")
                    analysis['data_outputscale'] = os_val
            except Exception:
                pass
        else:
            print("  Could not locate data kernel.")

    except Exception as e:
        print(f"  Data kernel analysis failed: {e}")

    print("\n" + "=" * 80)
    print("3. LIKELIHOOD NOISE ANALYSIS")
    print("=" * 80)

    try:
        noise = model.likelihood.noise.detach().cpu().numpy()
        print(f"\nLikelihood noise: {noise}")
        if noise.ndim == 0 or noise.size == 1:
            print(f"  Shared noise variance: {float(noise):.6f} "
                  f"(std = {np.sqrt(float(noise)):.6f})")
        else:
            print(f"  Per-observation noise shape: {noise.shape}")
        analysis['likelihood_noise'] = noise

        if hasattr(model.likelihood, 'task_noises'):
            task_noises = model.likelihood.task_noises.detach().cpu().numpy()
            print(f"\nPer-task noise variances:")
            for i, name in enumerate(task_names):
                if i < len(task_noises):
                    print(f"  {name:>30}: {task_noises[i]:.6f} "
                          f"(std = {np.sqrt(task_noises[i]):.6f})")
            analysis['task_noises'] = task_noises

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(task_noises)), np.sqrt(task_noises))
            ax.set_xticks(range(len(task_noises)))
            ax.set_xticklabels([n[:15] for n in task_names],
                               rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Noise Std Dev')
            ax.set_title('Learned Noise Per Task')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'task_noise_levels.png'),
                        dpi=200, bbox_inches='tight')
            print(f"\nSaved: {save_dir}/task_noise_levels.png")
            plt.close()

    except Exception as e:
        print(f"  Could not extract noise: {e}")

    print("\n" + "=" * 80)
    print("4. ALL LEARNED PARAMETERS")
    print("=" * 80)

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        val = param.detach().cpu()
        if val.numel() <= 10:
            print(f"  {name:>60}: {val.numpy().flatten()}")
        else:
            print(f"  {name:>60}: shape={list(val.shape)}, "
                  f"mean={val.mean():.4f}, std={val.std():.4f}")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    analysis['total_params'] = total_params

    return analysis


# =============================================================================
# Multi-Split Training (matching notebook train_mtgp_multi_split exactly)
# =============================================================================

def train_mtgp_multi_split(X_train_all, Y_train_all, X_holdout, Y_holdout,
                            device, task_names, n_splits=5, train_ratio=0.8,
                            output_dir="mtgp_plots", holdout_feat_valid=None,
                            descriptor=""):
    """
    Train MTGP across multiple random splits with holdout evaluation.
    Matches notebook train_mtgp_multi_split exactly:
      - np.random.seed(split_idx * 42) on raw samples
      - prepare_mtgp_data_from_arrays: iterates samples x tasks, skips NaN
      - Per-task StandardScaler fit on training portion
      - Returns per-task aggregated metrics and split 1 model for analysis

    DATA-1 audit fix: ``holdout_feat_valid`` is a boolean mask aligned to the
    holdout rows (True == genuinely featurized). Holdout rows that fell back to a
    zero/constant feature vector are AND-ed out of the per-task holdout metric
    mask so they do not pessimize HOLDOUT metrics. Training is unchanged.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_total = X_train_all.shape[0]
    n_train = int(train_ratio * n_total)
    num_tasks = len(task_names)

    if holdout_feat_valid is None:
        holdout_feat_valid = np.ones(X_holdout.shape[0], dtype=bool)

    all_task_metrics = {task: {'test': [], 'holdout': []} for task in task_names}
    analysis_model = None

    print(f"\n  Training samples: {n_total}, Holdout samples: {X_holdout.shape[0]}")
    print(f"  Number of tasks: {num_tasks}")

    for split_idx in range(n_splits):
        print(f"\n  --- Split {split_idx + 1}/{n_splits} ---")

        np.random.seed(split_idx * 42)
        perm = np.random.permutation(n_total)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        X_train = X_train_all[train_idx]
        X_test = X_train_all[test_idx]
        Y_train = Y_train_all[train_idx]
        Y_test = Y_train_all[test_idx]

        # Scale features
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        X_all_scaled = feature_scaler.transform(X_train_all)
        X_holdout_scaled = feature_scaler.transform(X_holdout)

        # Scale each task's output separately
        target_scalers = {}
        Y_train_scaled = np.full_like(Y_train, np.nan)
        Y_test_scaled = np.full_like(Y_test, np.nan)

        for task_idx, task_name in enumerate(task_names):
            train_mask = ~np.isnan(Y_train[:, task_idx])
            if train_mask.sum() > 0:
                scaler = StandardScaler()
                Y_train_scaled[train_mask, task_idx] = scaler.fit_transform(
                    Y_train[train_mask, task_idx].reshape(-1, 1)).ravel()
                target_scalers[task_name] = scaler

                test_mask = ~np.isnan(Y_test[:, task_idx])
                if test_mask.sum() > 0:
                    Y_test_scaled[test_mask, task_idx] = scaler.transform(
                        Y_test[test_mask, task_idx].reshape(-1, 1)).ravel()

        # Prepare MTGP format
        mtgp_train_X, mtgp_train_Y = prepare_mtgp_data_from_arrays(
            X_train_scaled, Y_train_scaled, task_names)
        print(f"    Training pairs: {len(mtgp_train_Y)}")

        # Train
        model = train_mtgp(mtgp_train_X, mtgp_train_Y, num_tasks, device, verbose=True)

        if split_idx == 0:
            analysis_model = model

        # Evaluate per task
        for task_idx, task_name in enumerate(task_names):
            if task_name not in target_scalers:
                continue
            scaler = target_scalers[task_name]

            train_valid = ~np.isnan(Y_train[:, task_idx])
            test_valid = ~np.isnan(Y_test[:, task_idx])
            all_valid = ~np.isnan(Y_train_all[:, task_idx])
            # DATA-1: holdout = target-not-NaN AND genuinely featurized.
            holdout_target_valid = ~np.isnan(Y_holdout[:, task_idx])
            holdout_valid = holdout_target_valid & holdout_feat_valid
            if split_idx == 0:
                n_excluded = int(holdout_target_valid.sum() - holdout_valid.sum())
                _tag = f"{descriptor}/{task_name}" if descriptor else task_name
                print(f"[holdout][{_tag}] excluded {n_excluded}/"
                      f"{int(holdout_target_valid.sum())} rows "
                      f"(invalid/unknown-species features)")

            if train_valid.sum() == 0 or test_valid.sum() == 0:
                continue

            # Internal test
            train_pred_s, train_std_s = predict_mtgp(
                model, X_train_scaled[train_valid], task_idx, device)
            train_pred = scaler.inverse_transform(train_pred_s.reshape(-1, 1)).ravel()
            train_true = Y_train[train_valid, task_idx]
            train_std = train_std_s * scaler.scale_[0]

            test_pred_s, test_std_s = predict_mtgp(
                model, X_test_scaled[test_valid], task_idx, device)
            test_pred = scaler.inverse_transform(test_pred_s.reshape(-1, 1)).ravel()
            test_true = Y_test[test_valid, task_idx]
            test_std = test_std_s * scaler.scale_[0]

            test_metrics = calculate_metrics(test_true, test_pred)
            all_task_metrics[task_name]['test'].append(test_metrics)

            plot_parity_with_metrics(
                train_true, train_pred, test_true, test_pred,
                y_train_std=train_std, y_test_std=test_std,
                title=(f"MTGP - {task_name} - "
                       f"Train/Test Split {split_idx+1}/{n_splits}"),
                save_path=os.path.join(
                    output_dir,
                    f"mtgp_parity_{task_name}_traintest_split{split_idx+1}.png"))

            print(f"    {task_name} Test R\u00b2: {test_metrics['R2']:.4f}, "
                  f"RMSE: {test_metrics['RMSE']:.4f}")

            # Holdout
            if holdout_valid.sum() > 0:
                all_pred_s, all_std_s = predict_mtgp(
                    model, X_all_scaled[all_valid], task_idx, device)
                all_pred = scaler.inverse_transform(all_pred_s.reshape(-1, 1)).ravel()
                all_true = Y_train_all[all_valid, task_idx]
                all_std = all_std_s * scaler.scale_[0]

                ho_pred_s, ho_std_s = predict_mtgp(
                    model, X_holdout_scaled[holdout_valid], task_idx, device)
                ho_pred = scaler.inverse_transform(ho_pred_s.reshape(-1, 1)).ravel()
                ho_true = Y_holdout[holdout_valid, task_idx]
                ho_std = ho_std_s * scaler.scale_[0]

                holdout_metrics = calculate_metrics(ho_true, ho_pred)
                all_task_metrics[task_name]['holdout'].append(holdout_metrics)

                plot_parity_with_metrics(
                    all_true, all_pred, ho_true, ho_pred,
                    y_train_std=all_std, y_test_std=ho_std,
                    title=(f"MTGP - Holdout on {task_name} "
                           f"(Split {split_idx+1})"),
                    save_path=os.path.join(
                        output_dir,
                        f"mtgp_parity_{task_name}_holdout_split{split_idx+1}.png"))

                print(f"    {task_name} HOLDOUT R\u00b2: {holdout_metrics['R2']:.4f}, "
                      f"RMSE: {holdout_metrics['RMSE']:.4f}")

    # Aggregate
    results = {}
    for task_name in task_names:
        if all_task_metrics[task_name]['test']:
            results[task_name] = {
                'test': aggregate_metrics(all_task_metrics[task_name]['test']),
                'holdout': (aggregate_metrics(all_task_metrics[task_name]['holdout'])
                            if all_task_metrics[task_name]['holdout'] else None),
                'all_holdout_metrics': all_task_metrics[task_name]['holdout'],
            }

    return results, analysis_model


# =============================================================================
# Run one descriptor
# =============================================================================

def run_descriptor(descriptor, df_train, df_holdout, target_cols, args, output_dir):
    """Featurize and run MTGP for all properties jointly for one descriptor."""
    print(f"\n{'='*70}")
    print(f"DESCRIPTOR: {descriptor.upper()}")
    print(f"{'='*70}")

    featurizer = get_featurizer(descriptor, args.pca_components, args.device)
    X_train_all = featurizer.fit_transform(df_train)
    X_holdout_all = featurizer.transform(df_holdout)
    print(f"  Features: train={X_train_all.shape}, holdout={X_holdout_all.shape}")

    # DATA-1 audit fix: holdout rows that could not be genuinely featurized carry
    # a zero/constant feature vector; this mask (aligned to df_holdout rows) lets
    # train_mtgp_multi_split drop them from HOLDOUT metrics. Training unchanged.
    holdout_feat_valid = featurizer.last_valid_mask_
    if holdout_feat_valid is None:
        holdout_feat_valid = np.ones(len(df_holdout), dtype=bool)

    # Filter to valid properties
    valid_props = []
    for prop in target_cols:
        if prop in df_train.columns and df_train[prop].notna().sum() >= 50:
            valid_props.append(prop)

    if not valid_props:
        print("  No valid properties found.")
        return {}, None

    print(f"  Valid properties ({len(valid_props)}): {valid_props}")

    Y_train_all = np.stack(
        [df_train[p].values.astype(float) for p in valid_props], axis=1
    )
    Y_holdout_all = np.stack(
        [df_holdout[p].values.astype(float) for p in valid_props], axis=1
    )

    device = torch.device(args.device)
    plot_dir = os.path.join(output_dir, 'plots', descriptor)
    os.makedirs(plot_dir, exist_ok=True)

    results, analysis_model = train_mtgp_multi_split(
        X_train_all, Y_train_all,
        X_holdout_all, Y_holdout_all,
        device, valid_props,
        n_splits=args.n_splits,
        train_ratio=0.8,
        output_dir=plot_dir,
        holdout_feat_valid=holdout_feat_valid,
        descriptor=descriptor,
    )

    return results, analysis_model, valid_props


# =============================================================================
# Save results
# =============================================================================

def save_results(results, output_dir, descriptor, task_names):
    """Save per-property holdout metrics summary CSV."""
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY: MTGP-{descriptor.upper()} HOLDOUT Metrics (mean +/- std)")
    print(f"{'='*120}")
    header = (f"{'Property':<22} | {'R2':<18} | {'RMSE':<18} | "
              f"{'MAE':<18} | {'SMAPE (%)':<18} | {'Spearman':<18}")
    print(header)
    print("-" * 120)

    for prop in task_names:
        if prop not in results or results[prop]['holdout'] is None:
            continue
        agg = results[prop]['holdout']
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
        csv_path = os.path.join(output_dir, f"mtgp_{descriptor}_holdout_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"\nSaved HOLDOUT summary to: {csv_path}")

    return summary_rows


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser('mtgp')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = args.output_dir or os.path.join('results', 'mtgp', args.dataset)
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
        result_tuple = run_descriptor(
            descriptor, df_train, df_holdout, target_cols, args, out_root
        )
        if len(result_tuple) == 3:
            results, analysis_model, valid_props = result_tuple
        else:
            results, analysis_model = result_tuple
            valid_props = list(results.keys())

        save_results(results, out_root, descriptor, valid_props)

        # Hyperparameter analysis on split 1 model
        if analysis_model is not None:
            analysis_dir = os.path.join(out_root, f"mtgp_{descriptor}_analysis")
            print(f"\n{'='*70}")
            print(f"Running Hyperparameter Analysis (split 1 model)...")
            print(f"{'='*70}")
            analyze_mtgp_hyperparameters(
                analysis_model, valid_props, save_dir=analysis_dir
            )
        else:
            print("WARNING: No model available for analysis.")

    print(f"\nAll results saved to: {out_root}")
    print("\nMTGP regression — COMPLETE!")


if __name__ == '__main__':
    main()
