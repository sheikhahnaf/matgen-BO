"""GP Surrogate model using BoTorch.

This implementation exactly matches the user's working BoTorch code for
GP-based property prediction with uncertainty quantification.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

# Don't set default dtype globally - it affects other parts of the code
# torch.set_default_dtype(torch.float64)  # DISABLED - causes dtype conflicts with mattergen


class GPSurrogate:
    """
    Gaussian Process surrogate model using BoTorch SingleTaskGP.

    Exactly matches user's implementation with:
    - BoTorch SingleTaskGP model
    - fit_gpytorch_mll for training
    - StandardScaler for features and targets
    - Proper uncertainty quantification

    Attributes:
        model: BoTorch SingleTaskGP instance
        feature_scaler: StandardScaler for input features
        target_scaler: StandardScaler for target values
        device: PyTorch device ('cpu' or 'cuda')
        is_trained: Whether model has been fitted
    """

    def __init__(
        self,
        input_dim: int,
        task: str,
        device: str = 'cpu'
    ):
        """
        Initialize GP Surrogate.

        Args:
            input_dim: Dimension of input features (e.g., 50 after PCA)
            task: Property name being modeled
            device: Device to run on ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.task = task
        self.device = torch.device(device)

        self.model = None  # SingleTaskGP instance
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.X_train = []
        self.y_train = []
        self.is_trained = False
        self.n_fitted_samples = 0  # Track number of samples actually fitted

    def add_data(self, X: np.ndarray, y: np.ndarray):
        """
        Add training data to buffer.

        Args:
            X: Feature matrix (n_samples, input_dim)
            y: Target values (n_samples,)
        """
        self.X_train.append(X)
        self.y_train.append(y)

    def fit(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        noise_var: Optional[np.ndarray] = None
    ):
        """
        Fit GP model on training data.

        Supports both homoscedastic (SingleTaskGP) and heteroscedastic (FixedNoiseGP):
        1. Standardize features and targets
        2. Convert to torch tensors
        3. Initialize SingleTaskGP (if noise_var=None) or FixedNoiseGP (if noise_var provided)
        4. Fit using fit_gpytorch_mll

        Args:
            X: Feature matrix (n_samples, input_dim). If None, use accumulated data.
            y: Target values (n_samples,). If None, use accumulated data.
            noise_var: Noise variance per sample (n_samples,). If None, use SingleTaskGP.
                      If provided, use FixedNoiseGP for heteroscedastic noise modeling.
        """
        # Use provided data or accumulated data
        if X is not None and y is not None:
            X_all = X
            y_all = y
        else:
            if not self.X_train or not self.y_train:
                raise ValueError("No training data available!")
            X_all = np.vstack(self.X_train) if len(self.X_train) > 1 else self.X_train[0]
            y_all = np.concatenate(self.y_train) if len(self.y_train) > 1 else self.y_train[0]

        # Standardize features
        X_scaled = self.feature_scaler.fit_transform(X_all)

        # Standardize targets
        y_scaled = self.target_scaler.fit_transform(y_all.reshape(-1, 1)).flatten()

        # Convert to torch tensors
        train_X = torch.tensor(X_scaled, device=self.device, dtype=torch.float64)
        train_Y = torch.tensor(y_scaled, device=self.device, dtype=torch.float64)

        # Reshape train_Y to (n, 1) for GP models
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)

        # Initialize GP model (heteroscedastic or homoscedastic)
        if noise_var is not None:
            # Heteroscedastic GP with FixedNoiseGP
            # Note: noise_var is in ORIGINAL scale, need to transform to scaled space
            train_Yvar = torch.tensor(noise_var, device=self.device, dtype=torch.float64)

            # Scale noise variance to match scaled targets
            # σ²_scaled = σ²_original / scale²
            scale_factor = self.target_scaler.scale_[0] ** 2
            train_Yvar_scaled = train_Yvar / scale_factor

            if train_Yvar_scaled.dim() == 1:
                train_Yvar_scaled = train_Yvar_scaled.unsqueeze(-1)

            # In BoTorch 0.16+, use SingleTaskGP with train_Yvar for heteroscedastic noise
            self.model = SingleTaskGP(train_X, train_Y, train_Yvar=train_Yvar_scaled).to(self.device)
            print(f"GP model (heteroscedastic) fitted on {len(X_all)} samples for {self.task}")
        else:
            # Homoscedastic GP with SingleTaskGP (exact match to user's original code)
            self.model = SingleTaskGP(train_X, train_Y).to(self.device)
            print(f"GP model (homoscedastic) fitted on {len(X_all)} samples for {self.task}")

        # Fit using fit_gpytorch_mll
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

        self.is_trained = True
        self.n_fitted_samples = len(X_all)  # Track fitted sample count
        self._y_train_fitted = y_all.copy()  # Store original y for best_observed

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with GP model.

        Args:
            X: Feature matrix (n_samples, input_dim)
            return_std: Whether to return standard deviation

        Returns:
            mean: Predicted mean values (n_samples,)
            std: Predicted standard deviation (n_samples,) if return_std=True
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call fit() first.")

        # Standardize features
        X_scaled = self.feature_scaler.transform(X)
        test_X = torch.tensor(X_scaled, device=self.device, dtype=torch.float64)

        # GP prediction
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
            mean_scaled = posterior.mean.squeeze(-1).cpu().numpy()
            variance = posterior.variance.squeeze(-1).cpu().numpy()

        # Inverse transform predictions to original scale
        mean_original = self.target_scaler.inverse_transform(
            mean_scaled.reshape(-1, 1)
        ).flatten()

        if return_std:
            # Transform std to original scale
            std_original = np.sqrt(variance) * self.target_scaler.scale_[0]
            return mean_original, std_original
        else:
            return mean_original, None

    def save(self, path: str):
        """
        Save GP model and scalers to disk.

        Args:
            path: Path to save model
        """
        import pickle

        save_dict = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'input_dim': self.input_dim,
            'task': self.task,
            'is_trained': self.is_trained,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"GP model saved to {path}")

    def load(self, path: str):
        """
        Load GP model and scalers from disk.

        Args:
            path: Path to load model from
        """
        import pickle

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.input_dim = save_dict['input_dim']
        self.task = save_dict['task']
        self.feature_scaler = save_dict['feature_scaler']
        self.target_scaler = save_dict['target_scaler']
        self.is_trained = save_dict['is_trained']

        if save_dict['model_state_dict'] and self.is_trained:
            # Reconstruct model (need dummy data for initialization)
            # This is a limitation - better to save full model
            print("Warning: Model state dict loaded but model reconstruction not implemented")
            print("Please retrain model after loading scalers")

        print(f"GP model loaded from {path}")

    def get_training_data_size(self) -> int:
        """Get number of training samples (from buffers OR fitted data)."""
        # Return max of buffer size and fitted sample count
        # This handles both add_data() workflow and direct fit(X,y) workflow
        buffer_size = sum(len(X) for X in self.X_train) if self.X_train else 0
        return max(buffer_size, self.n_fitted_samples)
