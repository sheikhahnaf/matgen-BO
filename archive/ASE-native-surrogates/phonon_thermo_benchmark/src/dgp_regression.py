"""
dgp_regression.py — Multi-task Deep GP (variational) per descriptor.

Matches notebook DGP multi-split cells (ASE_GP_MTGP_DGP_Botorch_multisplit3.ipynb) exactly:
  - torch.float32 default dtype (model uses .double() internally)
  - prepare_and_standardize_data: per-column StandardScaler on ALL training data
  - prepare_training_pairs: creates (X|task_idx, y) long-format pairs
  - Reduction search: compare_models_with_split_{descriptor} with REDUCTIONS_TO_TEST
    using train_model_with_validation (5000 epochs, patience=500), 80/20 via
    torch.Generator().manual_seed(42)
  - Best reduction by min avg test_rmse across tasks
  - Holdout scaled ONCE before split loop using training scalers
  - 5-split loop on PAIRS: np.random.seed(split_idx * 42) on all_train_x pairs
  - Per split: train_model_with_validation (5000 epochs) for internal test
  - Per split: separate full_model = train_model (3000 epochs) for holdout
  - evaluate_model_on_holdout for holdout parity plots
  - evaluate_model_train_test for internal test metrics
  - analyze_dgp_hyperparameters on split 1 model
  - Saves dgp_{descriptor}_holdout_summary.csv per descriptor

Per-descriptor REDUCTIONS_TO_TEST (matching notebook main functions):
  - SOAP:  [-45]
  - MACE:  [-45, -40, -35, -30, -20]
  - ORB:   [-20, -10, 0]
  - eSEN:  [-20, -10, 0]

Usage:
    python dgp_regression.py --dataset elastic_tensor_2015 \\
        --pca-components 50 --n-train 500 --n-splits 5 --device cuda
"""

import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood, GaussianLikelihood,
)
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from linear_operator.operators import to_linear_operator
from torch import Tensor
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    ALL_DESCRIPTORS, auto_detect_targets, get_featurizer,
    load_matminer_dataset, make_parser, prepare_dataset,
)

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float32)

# Per-descriptor reduction search space. These match the DOCUMENTED per-descriptor
# protocol in the module docstring above and the source notebook main functions
# (compare_models_with_split_{soap,mace,orb,uma}); each descriptor searches its own
# reduction set rather than a single shared list. (DGP-1 audit fix.)
DESCRIPTOR_REDUCTIONS = {
    'soap':  [-45],
    'mace':  [-45, -40, -35, -30, -20],
    'orb':   [-20, -10, 0],
    'uma':   [-20, -10, 0],
}


# =============================================================================
# DGP Architecture (exact copy from notebook cell 7)
# =============================================================================

def get_task_value_remapping(task_values: Tensor, dtype: torch.dtype) -> Optional[Tensor]:
    task_range = torch.arange(
        len(task_values), dtype=task_values.dtype, device=task_values.device)
    mapper = None
    if not torch.equal(task_values, task_range):
        mapper = torch.full(
            (task_values.max().item() + 1,), float("nan"),
            dtype=dtype, device=task_values.device)
        mapper[task_values] = task_range.to(dtype=dtype)
    return mapper


class DGPLastLayer3(gpytorch.models.ApproximateGP):
    def __init__(self, input_dims: int, output_dims: int, num_inducing: int = 128,
                 linear_mean: bool = True):
        num_latents = 10
        inducing_points = torch.randn(num_latents, num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([num_latents]),
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            VariationalStrategy(self, inducing_points, variational_distribution,
                                learn_inducing_locations=True),
            num_tasks=output_dims,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)
        self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims: int, output_dims: int, num_inducing: int = 32,
                 linear_mean: bool = True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultiTaskDeepGP(DeepGP, MultiTaskGPyTorchModel):
    """Multi-task Deep GP from notebook cell 7 (HEA code format)."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        reduction: int = 8,
        train_Yvar: Optional[Tensor] = None,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        output_tasks: Optional[List[int]] = None,
        all_tasks: Optional[List[int]] = None,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        super().__init__()
        self.reduction = reduction

        with torch.no_grad():
            transformed_X = self.transform_inputs(X=train_X, input_transform=input_transform)
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        (
            all_tasks_inferred,
            task_feature,
            self.num_non_task_features,
        ) = self.get_all_tasks(transformed_X, task_feature, output_tasks)

        if all_tasks is not None and not set(all_tasks_inferred).issubset(all_tasks):
            raise UnsupportedError(
                f"Provided {all_tasks=} does not contain all inferred "
                f"tasks {all_tasks_inferred=}."
            )
        all_tasks = all_tasks or all_tasks_inferred
        self.num_tasks = len(all_tasks)

        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar)

        train_Y = train_Y.squeeze(-1)
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if set(output_tasks) - set(all_tasks):
                raise RuntimeError("All output tasks must be present in input data.")
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        if likelihood is None:
            if train_Yvar is None:
                likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
            else:
                likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar.squeeze(-1))

        self._task_feature = task_feature
        self._base_idxr = torch.arange(self.num_non_task_features)
        self._base_idxr[task_feature:] += 1

        hidden_layer = DGPHiddenLayer(
            input_dims=train_X.shape[-1] - 1,
            output_dims=max(1, self.num_tasks + self.reduction),
            linear_mean=True,
        )
        last_layer = DGPLastLayer3(
            input_dims=max(1, self.num_tasks + self.reduction),
            output_dims=self.num_tasks,
            linear_mean=True,
        )
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            num_tasks=self.num_tasks)

        self._rank = self.num_tasks
        task_mapper = get_task_value_remapping(
            task_values=torch.tensor(
                all_tasks, dtype=torch.long, device=train_X.device),
            dtype=train_X.dtype,
        )
        self.register_buffer("_task_mapper", task_mapper)
        self._expected_task_values = set(all_tasks)

        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def _split_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_shape, d = x.shape[:-2], x.shape[-1]
        x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
        task_idcs = (
            x[..., self._task_feature]
            .view(batch_shape + torch.Size([-1, 1]))
            .to(dtype=torch.long)
        )
        task_idcs = self._map_tasks(task_values=task_idcs)
        return x_basic, task_idcs

    def _map_tasks(self, task_values: Tensor) -> Tensor:
        if self._task_mapper is None:
            return task_values
        return self._task_mapper[task_values.long()]

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        hidden = self.hidden_layer(x_basic)

        h_sample = torch.distributions.Normal(
            loc=hidden.mean, scale=hidden.variance.sqrt()
        ).rsample()

        if h_sample.dim() == 3:
            task_id = torch.broadcast_to(
                task_idcs.squeeze(-1),
                (h_sample.shape[-3], task_idcs.squeeze(-1).shape[-1]),
            )
        else:
            task_id = torch.broadcast_to(
                task_idcs.squeeze(-1),
                (h_sample.shape[-4], h_sample.shape[-3],
                 task_idcs.squeeze(-1).shape[-1]),
            )

        return self.last_layer(h_sample, task_indices=task_id)

    @classmethod
    def get_all_tasks(
        cls,
        train_X: Tensor,
        task_feature: int,
        output_tasks: Optional[List[int]] = None,
    ) -> Tuple[List[int], int, int]:
        if train_X.ndim != 2:
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")
        d = train_X.shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have -{d} <= task_feature <= {d}")
        task_feature = task_feature % (d + 1)
        all_tasks = (
            train_X[..., task_feature].unique(sorted=True).to(dtype=torch.long).tolist()
        )
        return all_tasks, task_feature, d


# =============================================================================
# Data Preparation (matching notebook prepare_and_standardize_data exactly)
# =============================================================================

def prepare_and_standardize_data(df, input_vars, output_vars, verbose=False):
    """
    Standardize data column-wise. Matches notebook prepare_and_standardize_data.
    Returns (input_scaled, output_scaled, input_scalers, output_scalers).
    """
    input_scaled = np.full_like(df[input_vars].values, np.nan, dtype=np.float64)
    input_scalers = {}

    for j, col in enumerate(input_vars):
        mask = ~np.isnan(df[input_vars].values[:, j])
        if mask.any():
            scaler = StandardScaler()
            input_scaled[mask, j] = scaler.fit_transform(
                df[input_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            input_scalers[col] = scaler
            if verbose:
                print(f"Input {col}: {np.sum(mask)}/{len(mask)} valid observations")

    output_scaled = np.full_like(df[output_vars].values, np.nan, dtype=np.float64)
    output_scalers = {}

    for j, col in enumerate(output_vars):
        mask = ~np.isnan(df[output_vars].values[:, j])
        if mask.any():
            scaler = StandardScaler()
            output_scaled[mask, j] = scaler.fit_transform(
                df[output_vars].values[mask, j].reshape(-1, 1)
            ).ravel()
            output_scalers[col] = scaler
            if verbose:
                print(f"Output {col}: {np.sum(mask)}/{len(mask)} valid observations")

    return input_scaled, output_scaled, input_scalers, output_scalers


def prepare_training_pairs(input_scaled, output_scaled):
    """Create training pairs for the DGP model (matching notebook exactly)."""
    n_samples, n_tasks = output_scaled.shape
    n_features = input_scaled.shape[1]

    total_pairs = int(np.sum(~np.isnan(output_scaled)))
    train_x = np.zeros((total_pairs, n_features + 1))
    train_y = np.zeros(total_pairs)

    idx = 0
    for i in range(n_samples):
        for task in range(n_tasks):
            if not np.isnan(output_scaled[i, task]):
                train_x[idx, :-1] = input_scaled[i]
                train_x[idx, -1] = task
                train_y[idx] = output_scaled[i, task]
                idx += 1

    return train_x, train_y


# =============================================================================
# Training functions (matching notebook train_model and train_model_with_validation)
# =============================================================================

def train_model(train_x, train_y, num_tasks, num_epochs=500, reduction=8):
    """Train DGP model (no validation set). Matching notebook train_model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_t = torch.tensor(train_x, dtype=torch.float64).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.float64).to(device)

    model = MultiTaskDeepGP(
        train_X=train_x_t,
        train_Y=train_y_t.unsqueeze(-1),
        task_feature=-1,
        reduction=reduction
    ).to(device)

    model = model.double()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)

    mll = DeepApproximateMLL(
        VariationalELBO(
            model.likelihood,
            model,
            num_data=train_x_t.shape[0],
            beta=0.5
        )
    )

    model.train()
    model.likelihood.train()

    best_loss = float('inf')
    patience = 50
    patience_counter = 0

    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {i+1}")
            break

        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Epoch {i+1}/{num_epochs} - Loss: {loss.item():.3f}')

    # Save loss curve
    return model


def train_model_with_validation(train_x, train_y, val_x, val_y,
                                 num_tasks, num_epochs=5000, reduction=8):
    """
    Train DGP with validation set monitoring.
    Matches notebook train_model_with_validation exactly:
      - patience=500 (on validation loss)
      - Adam lr=0.01
      - VariationalELBO beta=0.5
      - Logs every 50 epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_t = torch.tensor(train_x, dtype=torch.float64).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.float64).to(device)
    val_x_t = torch.tensor(val_x, dtype=torch.float64).to(device)
    val_y_t = torch.tensor(val_y, dtype=torch.float64).to(device)

    model = MultiTaskDeepGP(
        train_X=train_x_t,
        train_Y=train_y_t.unsqueeze(-1),
        task_feature=-1,
        reduction=reduction
    ).to(device)

    model = model.double()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.01)

    mll = DeepApproximateMLL(
        VariationalELBO(
            model.likelihood,
            model,
            num_data=train_x_t.shape[0],
            beta=0.5
        )
    )

    best_loss = float('inf')
    patience = 500
    patience_counter = 0

    for i in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_output = model(val_x_t)
            val_loss = -mll(val_output, val_y_t)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {i+1}")
            break

        if i % 50 == 0:
            print(f'Epoch {i+1}/{num_epochs} - Train Loss: {loss.item():.3f} '
                  f'- Test Loss: {val_loss.item():.3f}')

    return model


# =============================================================================
# Evaluation functions (matching notebook evaluate_model_train_test exactly)
# =============================================================================

def evaluate_model_train_test(model, train_x, train_y, test_x, test_y,
                               task_names, output_scalers):
    """
    Evaluate model on both train and test sets. Returns list of metric dicts.
    Matches notebook evaluate_model_train_test exactly.
    Keys: task, test_mae, test_rmse, test_r2, test_kendall, test_spearman, smape,
          n_train_samples, n_test_samples
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        train_posterior = model.posterior(torch.tensor(train_x).to(device))
        train_mean = train_posterior.mean.squeeze().cpu().numpy().T
        train_std = train_posterior.variance.sqrt().squeeze().cpu().numpy().T

        test_posterior = model.posterior(torch.tensor(test_x).to(device))
        test_mean = test_posterior.mean.squeeze().cpu().numpy().T
        test_std = test_posterior.variance.sqrt().squeeze().cpu().numpy().T

    metrics = []
    for task_idx, task_name in enumerate(task_names):
        train_mask = (train_x[:, -1] == task_idx)
        test_mask = (test_x[:, -1] == task_idx)

        if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
            train_preds_scaled = train_mean[train_mask]
            train_true_scaled = train_y[train_mask]
            test_preds_scaled = test_mean[test_mask]
            test_true_scaled = test_y[test_mask]

            scaler = output_scalers[task_name]
            train_preds = scaler.inverse_transform(
                train_preds_scaled.reshape(-1, 1)).ravel()
            train_true = scaler.inverse_transform(
                train_true_scaled.reshape(-1, 1)).ravel()

            test_preds = scaler.inverse_transform(
                test_preds_scaled.reshape(-1, 1)).ravel()
            test_true = scaler.inverse_transform(
                test_true_scaled.reshape(-1, 1)).ravel()

            test_mae = mean_absolute_error(test_true, test_preds)
            test_rmse = np.sqrt(mean_squared_error(test_true, test_preds))
            test_r2 = r2_score(test_true, test_preds)
            test_kendall = kendalltau(test_true, test_preds)[0]
            test_spearman = spearmanr(test_true, test_preds)[0]

            denom = np.abs(test_preds) + np.abs(test_true)
            denom[denom == 0] = 1e-8
            smape = np.mean(2 * np.abs(test_preds - test_true) / denom) * 100

            metrics.append({
                'task': task_name,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_kendall': test_kendall,
                'test_spearman': test_spearman,
                'smape': smape,
                'n_train_samples': int(np.sum(train_mask)),
                'n_test_samples': int(np.sum(test_mask)),
            })

    return metrics


def evaluate_model_on_holdout(model, train_x, train_y, holdout_x, holdout_y,
                               task_names, output_scalers,
                               output_dir="plots", split_idx=0):
    """
    Evaluate model on holdout set with parity plots.
    Matches notebook evaluate_model_on_holdout exactly.
    """
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)

    # Clear GPU cache before evaluation to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        train_posterior = model.posterior(torch.tensor(train_x).to(device))
        train_mean = train_posterior.mean.squeeze().cpu().numpy().T
        train_std = train_posterior.variance.sqrt().squeeze().cpu().numpy().T

        # Clear cache again before holdout evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate holdout set in batches to avoid OOM
        batch_size = 50  # Process 50 samples at a time
        holdout_x_tensor = torch.tensor(holdout_x).to(device)
        n_holdout = len(holdout_x)

        holdout_means = []
        holdout_stds = []

        for i in range(0, n_holdout, batch_size):
            batch_end = min(i + batch_size, n_holdout)
            batch_x = holdout_x_tensor[i:batch_end]

            batch_posterior = model.posterior(batch_x)

            # Get numpy arrays WITHOUT squeeze to preserve all dimensions
            batch_mean = batch_posterior.mean.cpu().numpy()
            batch_std = batch_posterior.variance.sqrt().cpu().numpy()

            # Debug: print shapes
            if i == 0:
                print(f"  DEBUG First batch raw shape: {batch_mean.shape}")

            # Handle different possible posterior shapes:
            # - (1, batch_size, n_tasks) → remove first dim → (batch_size, n_tasks)
            # - (batch_size, n_tasks) → keep as-is
            if batch_mean.ndim == 3 and batch_mean.shape[0] == 1:
                batch_mean = batch_mean[0]  # Now (batch_size, n_tasks)
                batch_std = batch_std[0]

            # Ensure 2D even if batch_size=1: (1, n_tasks) not (n_tasks,)
            if batch_mean.ndim == 1:
                batch_mean = batch_mean.reshape(1, -1)
                batch_std = batch_std.reshape(1, -1)

            if i == 0:
                print(f"  DEBUG First batch after fix: {batch_mean.shape}")

            holdout_means.append(batch_mean)
            holdout_stds.append(batch_std)

            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate batch results - all batches are (batch_size, 1)
        # Result: (n_holdout, 1)
        holdout_mean_2d = np.concatenate(holdout_means, axis=0)
        holdout_std_2d = np.concatenate(holdout_stds, axis=0)

        # Squeeze to match train format: (n_holdout,)
        holdout_mean = holdout_mean_2d.squeeze()
        holdout_std = holdout_std_2d.squeeze()

        print(f"  Holdout mean shape: {holdout_mean.shape}")
        print(f"  Holdout std shape: {holdout_std.shape}")

        # Validation
        assert len(holdout_mean) == n_holdout, \
            f"Shape mismatch: holdout_mean {holdout_mean.shape} vs n_holdout {n_holdout}"

    metrics = {}
    for task_idx, task_name in enumerate(task_names):
        train_mask = (train_x[:, -1] == task_idx)
        holdout_mask = (holdout_x[:, -1] == task_idx)

        if np.sum(train_mask) > 0 and np.sum(holdout_mask) > 0:
            train_preds_scaled = train_mean[train_mask]
            train_std_scaled = train_std[train_mask]
            train_true_scaled = train_y[train_mask]

            # Extract predictions for current task (both are 1D arrays)
            holdout_preds_scaled = holdout_mean[holdout_mask]
            holdout_std_scaled = holdout_std[holdout_mask]
            holdout_true_scaled = holdout_y[holdout_mask]

            scaler = output_scalers[task_name]
            train_preds = scaler.inverse_transform(
                train_preds_scaled.reshape(-1, 1)).ravel()
            train_true = scaler.inverse_transform(
                train_true_scaled.reshape(-1, 1)).ravel()
            train_uncertainties = train_std_scaled * np.sqrt(scaler.var_)[0]

            holdout_preds = scaler.inverse_transform(
                holdout_preds_scaled.reshape(-1, 1)).ravel()
            holdout_true = scaler.inverse_transform(
                holdout_true_scaled.reshape(-1, 1)).ravel()
            holdout_uncertainties = holdout_std_scaled * np.sqrt(scaler.var_)[0]

            rmse = np.sqrt(mean_squared_error(holdout_true, holdout_preds))
            mae = np.mean(np.abs(holdout_true - holdout_preds))
            r2 = r2_score(holdout_true, holdout_preds)
            denom = np.abs(holdout_preds) + np.abs(holdout_true)
            denom[denom == 0] = 1e-8
            smape = np.mean(2 * np.abs(holdout_preds - holdout_true) / denom) * 100
            spearman_val, _ = spearmanr(holdout_true, holdout_preds)

            metrics[task_name] = {
                'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'SMAPE': smape, 'Spearman': spearman_val
            }

            fig, ax = plt.subplots(figsize=(10, 8))

            ax.errorbar(train_true, train_preds, yerr=train_uncertainties,
                        fmt='o', alpha=0.6, markersize=6, color='blue',
                        ecolor='lightblue', elinewidth=1, capsize=2,
                        label='Train', markeredgecolor='darkblue', markeredgewidth=0.5)

            ax.errorbar(holdout_true, holdout_preds, yerr=holdout_uncertainties,
                        fmt='o', alpha=0.6, markersize=6, color='red',
                        ecolor='lightcoral', elinewidth=1, capsize=2,
                        label='Holdout', markeredgecolor='darkred', markeredgewidth=0.5)

            all_y = np.concatenate([train_true, holdout_true, train_preds, holdout_preds])
            min_val, max_val = all_y.min(), all_y.max()
            margin = (max_val - min_val) * 0.05
            ax.plot([min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin],
                    'k--', lw=2, label='Perfect Prediction')

            ax.set_xlabel('True Values', fontsize=14, fontweight='bold')
            ax.set_ylabel('Predicted Values', fontsize=14, fontweight='bold')
            ax.set_title(
                f"DGP - Holdout Prediction on {task_name} (Split {split_idx+1})",
                fontsize=16, fontweight='bold')
            ax.legend(loc='upper left', fontsize=11)
            ax.grid(True, alpha=0.3)

            train_metrics = {
                'R2': r2_score(train_true, train_preds),
                'RMSE': np.sqrt(mean_squared_error(train_true, train_preds)),
                'MAE': np.mean(np.abs(train_true - train_preds)),
                'Spearman': spearmanr(train_true, train_preds)[0],
            }

            metrics_text = "Train Metrics:\n"
            metrics_text += f"R\u00b2 = {train_metrics['R2']:.4f}\n"
            metrics_text += f"RMSE = {train_metrics['RMSE']:.4f}\n"
            metrics_text += f"MAE = {train_metrics['MAE']:.4f}\n"
            metrics_text += f"Spearman = {train_metrics['Spearman']:.4f}\n\n"
            metrics_text += "Holdout Metrics:\n"
            metrics_text += f"R\u00b2 = {r2:.4f}\n"
            metrics_text += f"RMSE = {rmse:.4f}\n"
            metrics_text += f"MAE = {mae:.4f}\n"
            metrics_text += f"SMAPE = {smape:.2f}%\n"
            metrics_text += f"Spearman = {spearman_val:.4f}"

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            plt.tight_layout()
            save_path = os.path.join(
                output_dir, f"dgp_parity_{task_name}_holdout_split{split_idx+1}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {save_path}")
            plt.close()

            print(f"    {task_name} HOLDOUT R\u00b2: {r2:.4f}, "
                  f"RMSE: {rmse:.4f}, Spearman: {spearman_val:.4f}")

    return metrics


# =============================================================================
# Reduction Search (matching notebook compare_models_with_split_soap exactly)
# =============================================================================

def compare_models_with_split(input_scaled, output_scaled, output_vars, output_scalers,
                               reductions, descriptor_name=""):
    """
    Compare DGP models with different reduction parameters.
    Uses 80/20 split via torch.Generator().manual_seed(42) on training pairs.
    Matches notebook compare_models_with_split_soap/mace/orb/uma exactly.
    """
    model_results = {}

    for reduction in reductions:
        print(f"\nEvaluating model with reduction={reduction}")

        train_x, train_y = prepare_training_pairs(input_scaled, output_scaled)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_x),
            torch.tensor(train_y)
        )

        train_size = int(0.80 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_x_split = train_dataset.dataset.tensors[0][train_dataset.indices].numpy()
        train_y_split = train_dataset.dataset.tensors[1][train_dataset.indices].numpy()
        test_x_split = test_dataset.dataset.tensors[0][test_dataset.indices].numpy()
        test_y_split = test_dataset.dataset.tensors[1][test_dataset.indices].numpy()

        model = train_model_with_validation(
            train_x_split,
            train_y_split,
            test_x_split,
            test_y_split,
            num_tasks=output_scaled.shape[1],
            num_epochs=5000,
            reduction=reduction
        )

        metrics = evaluate_model_train_test(
            model,
            train_x_split,
            train_y_split,
            test_x_split,
            test_y_split,
            output_vars,
            output_scalers
        )

        model_results[reduction] = metrics

    return model_results


def find_best_reduction(model_results):
    """Find reduction with minimum average test RMSE across all tasks."""
    best_reduction = None
    best_rmse = float('inf')

    for reduction, metrics in model_results.items():
        avg_rmse = np.mean([m['test_rmse'] for m in metrics])
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_reduction = reduction

    return best_reduction, best_rmse


# =============================================================================
# DGP Hyperparameter Analysis (matching notebook analyze_dgp_hyperparameters)
# =============================================================================

def analyze_dgp_hyperparameters(model, output_vars, save_dir="dgp_analysis"):
    """
    Comprehensive analysis of learned DGP hyperparameters.
    Matches notebook analyze_dgp_hyperparameters exactly.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    analysis = {}
    hidden = model.hidden_layer
    last = model.last_layer

    print("\n" + "=" * 80)
    print("1. HIDDEN LAYER KERNEL ANALYSIS")
    print("=" * 80)

    try:
        lengthscales = hidden.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        print(f"\nHidden layer lengthscales shape: {lengthscales.shape}")
        print(f"  (Each row = one output dim, each col = one input feature's lengthscale)")

        avg_lengthscales = np.mean(lengthscales, axis=0).squeeze()
        print(f"\nAverage lengthscale per input feature (lower = more important):")
        if len(avg_lengthscales.shape) > 0:
            sorted_idx = np.argsort(avg_lengthscales)
            for rank, idx in enumerate(sorted_idx[:10]):
                print(f"  Rank {rank+1}: Feature {idx}, "
                      f"lengthscale = {avg_lengthscales[idx]:.4f}")

        analysis['hidden_lengthscales'] = lengthscales
        analysis['avg_lengthscales'] = avg_lengthscales

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        im = axes[0].imshow(lengthscales.squeeze(), aspect='auto', cmap='viridis')
        axes[0].set_xlabel('Input Feature Index')
        axes[0].set_ylabel('Hidden Output Dimension')
        axes[0].set_title('Hidden Layer ARD Lengthscales\n(darker = shorter = more important)')
        plt.colorbar(im, ax=axes[0])

        importance = 1.0 / (avg_lengthscales + 1e-6)
        importance = importance / importance.sum()
        axes[1].bar(range(len(importance)), importance)
        axes[1].set_xlabel('Input Feature Index')
        axes[1].set_ylabel('Relative Importance (1/lengthscale)')
        axes[1].set_title('Feature Importance from Hidden Layer')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'hidden_layer_lengthscales.png'),
                    dpi=200, bbox_inches='tight')
        print(f"\nSaved: {save_dir}/hidden_layer_lengthscales.png")
        plt.close()

    except Exception as e:
        print(f"  Could not extract hidden lengthscales: {e}")

    try:
        outputscales = hidden.covar_module.outputscale.detach().cpu().numpy()
        print(f"\nHidden layer output scales: {outputscales}")
        analysis['hidden_outputscales'] = outputscales
    except Exception as e:
        print(f"  Could not extract output scales: {e}")

    print("\n" + "=" * 80)
    print("2. LAST LAYER LMC ANALYSIS (Task Correlations)")
    print("=" * 80)

    try:
        lmc_coefficients = last.variational_strategy.lmc_coefficients.detach().cpu().numpy()
        print(f"\nLMC coefficients shape: {lmc_coefficients.shape}")
        print(f"  (num_latents x num_tasks)")

        W = lmc_coefficients.T  # [num_tasks, num_latents]
        task_covariance = W @ W.T
        diag = np.sqrt(np.diag(task_covariance))
        diag[diag == 0] = 1e-8
        task_correlation = task_covariance / np.outer(diag, diag)

        analysis['lmc_coefficients'] = lmc_coefficients
        analysis['task_correlation'] = task_correlation

        print("\nLearned Task Correlation Matrix:")
        short_names = [name[:15] for name in output_vars]
        header = f"{'':>16}" + "".join([f"{n:>16}" for n in short_names])
        print(header)
        for i, name in enumerate(short_names):
            row = (f"{name:>16}" +
                   "".join([f"{task_correlation[i,j]:>16.3f}"
                             for j in range(len(output_vars))]))
            print(row)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        im = axes[0].imshow(task_correlation, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(len(output_vars)))
        axes[0].set_yticks(range(len(output_vars)))
        axes[0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_yticklabels(short_names, fontsize=8)
        axes[0].set_title('Learned Task Correlations (from LMC)\nRed=positive, Blue=negative')
        plt.colorbar(im, ax=axes[0])
        for i in range(len(output_vars)):
            for j in range(len(output_vars)):
                axes[0].text(j, i, f'{task_correlation[i,j]:.2f}',
                             ha='center', va='center', fontsize=6,
                             color='white' if abs(task_correlation[i,j]) > 0.5 else 'black')

        im2 = axes[1].imshow(W, aspect='auto', cmap='coolwarm')
        axes[1].set_xlabel('Latent GP Index')
        axes[1].set_ylabel('Task')
        axes[1].set_yticks(range(len(output_vars)))
        axes[1].set_yticklabels(short_names, fontsize=8)
        axes[1].set_title('LMC Mixing Weights\n(How each task combines latent GPs)')
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'task_correlations_lmc.png'),
                    dpi=200, bbox_inches='tight')
        print(f"\nSaved: {save_dir}/task_correlations_lmc.png")
        plt.close()

    except Exception as e:
        print(f"  Could not extract LMC weights: {e}")

    print("\n" + "=" * 80)
    print("3. LIKELIHOOD NOISE ANALYSIS")
    print("=" * 80)

    try:
        if hasattr(model.likelihood, 'task_noises'):
            task_noises = model.likelihood.task_noises.detach().cpu().numpy()
            print(f"\nPer-task noise variances:")
            for i, name in enumerate(output_vars):
                print(f"  {name:>30}: {task_noises[i]:.6f} "
                      f"(std = {np.sqrt(task_noises[i]):.6f})")
            analysis['task_noises'] = task_noises

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(output_vars)), np.sqrt(task_noises))
            ax.set_xticks(range(len(output_vars)))
            ax.set_xticklabels([n[:15] for n in output_vars],
                               rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Noise Std Dev (standardized scale)')
            ax.set_title('Learned Noise Per Task\n(Higher = more irreducible noise)')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'task_noise_levels.png'),
                        dpi=200, bbox_inches='tight')
            print(f"\nSaved: {save_dir}/task_noise_levels.png")
            plt.close()

        elif hasattr(model.likelihood, 'noise'):
            noise = model.likelihood.noise.detach().cpu().item()
            print(f"\nShared noise variance: {noise:.6f} (std = {np.sqrt(noise):.6f})")
            analysis['shared_noise'] = noise

    except Exception as e:
        print(f"  Could not extract noise params: {e}")

    print("\n" + "=" * 80)
    print("4. INDUCING POINTS ANALYSIS")
    print("=" * 80)

    try:
        hidden_ip = hidden.variational_strategy.inducing_points.detach().cpu().numpy()
        print(f"\nHidden layer inducing points shape: {hidden_ip.shape}")
        print(f"  Mean: {hidden_ip.mean():.4f}, Std: {hidden_ip.std():.4f}")
        analysis['hidden_inducing_points_shape'] = hidden_ip.shape
    except Exception as e:
        print(f"  Could not extract hidden inducing points: {e}")

    try:
        last_ip = last.variational_strategy.base_variational_strategy.inducing_points.detach().cpu().numpy()
        print(f"\nLast layer inducing points shape: {last_ip.shape}")
        print(f"  Mean: {last_ip.mean():.4f}, Std: {last_ip.std():.4f}")
        analysis['last_inducing_points_shape'] = last_ip.shape
    except Exception as e:
        print(f"  Could not extract last layer inducing points: {e}")

    print("\n" + "=" * 80)
    print("5. LAST LAYER KERNEL ANALYSIS")
    print("=" * 80)

    try:
        last_lengthscales = last.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        last_outputscales = last.covar_module.outputscale.detach().cpu().numpy()
        print(f"\nLast layer RBF lengthscales shape: {last_lengthscales.shape}")
        print(f"  Values: {last_lengthscales.squeeze()}")
        print(f"\nLast layer output scales shape: {last_outputscales.shape}")
        print(f"  Values: {last_outputscales.squeeze()}")

        analysis['last_lengthscales'] = last_lengthscales
        analysis['last_outputscales'] = last_outputscales

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(range(len(last_lengthscales.squeeze())), last_lengthscales.squeeze())
        axes[0].set_xlabel('Latent GP Index')
        axes[0].set_ylabel('Lengthscale')
        axes[0].set_title('Last Layer RBF Lengthscales\n(per latent GP)')

        axes[1].bar(range(len(last_outputscales.squeeze())), last_outputscales.squeeze())
        axes[1].set_xlabel('Latent GP Index')
        axes[1].set_ylabel('Output Scale')
        axes[1].set_title('Last Layer Output Scales\n(signal variance per latent GP)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'last_layer_kernel_params.png'),
                    dpi=200, bbox_inches='tight')
        print(f"\nSaved: {save_dir}/last_layer_kernel_params.png")
        plt.close()

    except Exception as e:
        print(f"  Could not extract last layer kernel params: {e}")

    print("\n" + "=" * 80)
    print("6. MODEL PARAMETER SUMMARY")
    print("=" * 80)

    total_params = 0
    trainable_params = 0
    module_params = {}
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        module = name.split('.')[0]
        module_params[module] = module_params.get(module, 0) + param.numel()

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    for module, count in sorted(module_params.items(), key=lambda x: -x[1]):
        print(f"  {module:>20}: {count:>10,} ({100*count/total_params:.1f}%)")

    analysis['total_params'] = total_params
    analysis['trainable_params'] = trainable_params

    return analysis


# =============================================================================
# Main descriptor loop
# =============================================================================

def run_descriptor(descriptor, df_train, df_holdout, target_cols, args, output_dir):
    """Run DGP for all properties jointly for one descriptor, matching notebook."""
    print(f"\n{'='*70}")
    print(f"DESCRIPTOR: {descriptor.upper()}")
    print(f"{'='*70}")

    featurizer = get_featurizer(descriptor, args.pca_components, args.device)
    X_train_all = featurizer.fit_transform(df_train)
    X_holdout_all = featurizer.transform(df_holdout)
    print(f"  Features: train={X_train_all.shape}, holdout={X_holdout_all.shape}")

    # DATA-1 audit fix: holdout rows that could not be genuinely featurized carry
    # a zero/constant feature vector; this mask (aligned to df_holdout rows) lets
    # us drop them from the HOLDOUT pair set below. Training is unchanged.
    holdout_feat_valid = featurizer.last_valid_mask_
    if holdout_feat_valid is None:
        holdout_feat_valid = np.ones(len(df_holdout), dtype=bool)

    # Valid properties
    valid_props = [
        p for p in target_cols
        if p in df_train.columns and df_train[p].notna().sum() >= 50
    ]
    if not valid_props:
        print("  No valid properties found.")
        return {}, []

    print(f"  Valid properties ({len(valid_props)}): {valid_props}")

    # Setup HEA-format DataFrames (matching notebook exactly)
    INPUT_VARS = [f'FEAT_{i}' for i in range(X_train_all.shape[1])]
    OUTPUT_VARS = valid_props

    df_hea = pd.DataFrame(X_train_all, columns=INPUT_VARS)
    for col in OUTPUT_VARS:
        df_hea[col] = df_train[col].values

    df_holdout_hea = pd.DataFrame(X_holdout_all, columns=INPUT_VARS)
    for col in OUTPUT_VARS:
        df_holdout_hea[col] = df_holdout[col].values

    print(f"\nTraining DataFrame shape: {df_hea.shape}")
    print(f"Holdout DataFrame shape: {df_holdout_hea.shape}")

    # Prepare and standardize ALL training data
    print(f"\n{'='*70}")
    print("Preparing and standardizing data...")
    print(f"{'='*70}")

    input_scaled, output_scaled, input_scalers, output_scalers = \
        prepare_and_standardize_data(df_hea, INPUT_VARS, OUTPUT_VARS)

    # Prepare holdout scaled data ONCE using training scalers
    print(f"\nScaling holdout data using training scalers...")
    holdout_input_scaled = np.zeros_like(X_holdout_all, dtype=np.float64)
    for j, col in enumerate(INPUT_VARS):
        if col in input_scalers:
            holdout_input_scaled[:, j] = input_scalers[col].transform(
                X_holdout_all[:, j].reshape(-1, 1)
            ).ravel()

    holdout_output_scaled = np.full(
        (len(df_holdout_hea), len(OUTPUT_VARS)), np.nan, dtype=np.float64)
    for j, col in enumerate(OUTPUT_VARS):
        if col in output_scalers:
            mask = df_holdout_hea[col].notna().values
            if mask.any():
                holdout_output_scaled[mask, j] = output_scalers[col].transform(
                    df_holdout_hea[col].values[mask].reshape(-1, 1)
                ).ravel()

    # DATA-1: exclude invalid-feature holdout rows from the HOLDOUT metric set.
    # The DGP works in long-format (structure x task) pairs, and
    # prepare_training_pairs skips NaN targets; so NaN-ing the targets of invalid
    # rows here drops exactly those rows from the holdout pairs without touching
    # training (which uses input_scaled/output_scaled, built independently above).
    for j, col in enumerate(OUTPUT_VARS):
        target_valid_j = df_holdout_hea[col].notna().values
        kept_j = target_valid_j & holdout_feat_valid
        n_excluded = int(target_valid_j.sum() - kept_j.sum())
        print(f"[holdout][{descriptor}/{col}] excluded {n_excluded}/"
              f"{int(target_valid_j.sum())} rows (invalid/unknown-species features)")
    holdout_output_scaled[~holdout_feat_valid, :] = np.nan

    holdout_x, holdout_y = prepare_training_pairs(
        holdout_input_scaled, holdout_output_scaled)
    print(f"Holdout pairs: {len(holdout_x)}")

    # Get reduction search space for this descriptor
    desc_key = descriptor.lower()
    reductions = DESCRIPTOR_REDUCTIONS.get(desc_key, [-20, -10, 0])
    print(f"\nREDUCTIONS_TO_TEST for {descriptor}: {reductions}")

    # Reduction search
    print(f"\n{'='*70}")
    print("Comparing models with different reduction parameters...")
    print(f"{'='*70}")

    model_results = compare_models_with_split(
        input_scaled, output_scaled, OUTPUT_VARS, output_scalers,
        reductions=reductions, descriptor_name=descriptor
    )

    best_reduction, best_rmse = find_best_reduction(model_results)
    print(f"\nBest reduction parameter: {best_reduction}")
    print(f"Best average test RMSE: {best_rmse:.4f}")

    # 5-split evaluation on PAIRS (matching notebook exactly)
    N_SPLITS = args.n_splits
    TRAIN_RATIO = 0.8

    print(f"\n{'='*70}")
    print(f"Running {N_SPLITS}-split holdout evaluation with best reduction={best_reduction}")
    print(f"{'='*70}")

    all_train_x, all_train_y = prepare_training_pairs(input_scaled, output_scaled)
    n_total = len(all_train_x)
    n_train_pairs = int(TRAIN_RATIO * n_total)

    all_splits_holdout_metrics = {prop: [] for prop in OUTPUT_VARS}
    all_splits_test_metrics = {prop: [] for prop in OUTPUT_VARS}
    analysis_model = None

    plot_dir = os.path.join(output_dir, f"dgp_{descriptor}_parity_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for split_idx in range(N_SPLITS):
        print(f"\n{'='*70}")
        print(f"SPLIT {split_idx + 1}/{N_SPLITS}")
        print(f"{'='*70}")

        np.random.seed(split_idx * 42)
        perm = np.random.permutation(n_total)
        train_idx, test_idx = perm[:n_train_pairs], perm[n_train_pairs:]

        split_train_x = all_train_x[train_idx]
        split_train_y = all_train_y[train_idx]
        split_test_x = all_train_x[test_idx]
        split_test_y = all_train_y[test_idx]

        model = train_model_with_validation(
            split_train_x, split_train_y,
            split_test_x, split_test_y,
            num_tasks=len(OUTPUT_VARS),
            num_epochs=5000,
            reduction=best_reduction
        )

        if split_idx == 0:
            analysis_model = model

        # Internal test evaluation
        print(f"\n  Evaluating on internal test set...")
        test_metrics = evaluate_model_train_test(
            model,
            split_train_x, split_train_y,
            split_test_x, split_test_y,
            OUTPUT_VARS, output_scalers
        )
        for metric in test_metrics:
            all_splits_test_metrics[metric['task']].append({
                'R2': metric['test_r2'],
                'RMSE': metric['test_rmse'],
                'MAE': metric['test_mae'],
                'SMAPE': metric['smape'],
                'Spearman': metric['test_spearman'],
            })

        # Holdout evaluation (train on FULL training pairs)
        print(f"\n  Training on full training set for holdout evaluation...")
        full_model = train_model(
            all_train_x, all_train_y,
            num_tasks=len(OUTPUT_VARS),
            num_epochs=3000,
            reduction=best_reduction
        )

        print(f"\n  Evaluating on HOLDOUT set...")
        holdout_metrics = evaluate_model_on_holdout(
            full_model,
            all_train_x, all_train_y,
            holdout_x, holdout_y,
            OUTPUT_VARS, output_scalers,
            output_dir=plot_dir,
            split_idx=split_idx
        )

        for prop, m in holdout_metrics.items():
            all_splits_holdout_metrics[prop].append(m)

        print(f"\n  Split {split_idx+1} HOLDOUT summary:")
        for prop, m in holdout_metrics.items():
            print(f"    {prop:<22} R\u00b2={m['R2']:.4f}, "
                  f"RMSE={m['RMSE']:.4f}, Spearman={m['Spearman']:.4f}")

    # Aggregate results
    results = {}
    for prop in OUTPUT_VARS:
        test_list = all_splits_test_metrics[prop]
        holdout_list = all_splits_holdout_metrics[prop]

        if test_list:
            # Aggregate test metrics
            test_agg = {}
            for key in test_list[0]:
                vals = [m[key] for m in test_list]
                test_agg[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

            # Aggregate holdout metrics
            holdout_agg = None
            if holdout_list:
                holdout_agg = {}
                for key in holdout_list[0]:
                    vals = [m[key] for m in holdout_list]
                    holdout_agg[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

            results[prop] = {
                'test': test_agg,
                'holdout': holdout_agg,
                'all_holdout_metrics': holdout_list,
            }

    return results, valid_props, analysis_model


# =============================================================================
# Save results
# =============================================================================

def save_results(results, output_dir, descriptor, valid_props):
    """Save holdout summary tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Internal test summary
    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY: DGP-{descriptor.upper()} Internal Test Metrics (mean +/- std)")
    print(f"{'='*120}")
    header = (f"{'Property':<22} | {'R2':<18} | {'RMSE':<18} | "
              f"{'MAE':<18} | {'SMAPE (%)':<18} | {'Spearman':<18}")
    print(header)
    print("-" * 120)

    for prop in valid_props:
        if prop not in results or not results[prop]['test']:
            continue
        agg = results[prop]['test']
        print(f"{prop:<22} | "
              f"{agg['R2']['mean']:<10.4f} +/- {agg['R2']['std']:<10.4f} | "
              f"{agg['RMSE']['mean']:<10.4f} +/- {agg['RMSE']['std']:<10.4f} | "
              f"{agg['MAE']['mean']:<10.4f} +/- {agg['MAE']['std']:<10.4f} | "
              f"{agg['SMAPE']['mean']:<10.2f} +/- {agg['SMAPE']['std']:<10.2f} | "
              f"{agg['Spearman']['mean']:<10.4f} +/- {agg['Spearman']['std']:.4f}")

    print("-" * 120)

    # Holdout summary
    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY: DGP-{descriptor.upper()} HOLDOUT Metrics (mean +/- std)")
    print(f"{'='*120}")
    print(header)
    print("-" * 120)

    summary_rows = []
    for prop in valid_props:
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
        csv_path = os.path.join(output_dir, f"dgp_{descriptor}_holdout_summary.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"\nSaved HOLDOUT summary to: {csv_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = make_parser('dgp')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = args.output_dir or os.path.join('results', 'dgp', args.dataset)
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
            results, valid_props, analysis_model = result_tuple
        else:
            results, valid_props = result_tuple
            analysis_model = None

        save_results(results, out_root, descriptor, valid_props)

        # Hyperparameter analysis on split 1 model
        if analysis_model is not None:
            analysis_dir = os.path.join(out_root, f"dgp_{descriptor}_analysis")
            print(f"\n{'='*70}")
            print(f"Running Hyperparameter Analysis (split 1 model)...")
            print(f"{'='*70}")
            analyze_dgp_hyperparameters(
                analysis_model, valid_props, save_dir=analysis_dir
            )
        else:
            print("WARNING: No model available for analysis.")

    print(f"\nAll results saved to: {out_root}")
    print("\nDGP regression — COMPLETE!")


if __name__ == '__main__':
    main()
