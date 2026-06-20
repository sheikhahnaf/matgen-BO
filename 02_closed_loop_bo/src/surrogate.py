"""BoTorch SingleTaskGP surrogate for Cp regression.

Adapted from (read-only upstream):
    /Volumes/SSD1_SMAAA/matinvent-bo/ASE_regression_test/gp_regression.py

Differences vs upstream:
    - Wrapped as a class with input/output standardization persisted on the
      object (upstream re-scaled per-call).
    - Adds heteroscedastic noise via FixedNoiseGP when y_var is supplied.
    - Adds save/load (torch state_dict + scalers).
    - Predicts in the *original* y-scale (upstream returned scaled).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize  # imported only for type ref
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler


class HCapSurrogate:
    """SingleTaskGP over PCA-reduced ORB embeddings, target = Cp (J/g/K).

    Standardizes X and y on fit; predicts in original y-scale with std rescaled
    by the y-scaler.
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float64):
        self.device = device
        self.dtype = dtype
        self.x_scaler: Optional[StandardScaler] = None
        self.y_scaler: Optional[StandardScaler] = None
        self.model = None  # SingleTaskGP | FixedNoiseGP

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_var: Optional[np.ndarray] = None,
    ) -> None:
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.shape}")
        if y.ndim != 1:
            y = y.reshape(-1)

        self.x_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(y.reshape(-1, 1))

        Xs = self.x_scaler.transform(X)
        ys = self.y_scaler.transform(y.reshape(-1, 1)).ravel()

        Xt = self._to_tensor(Xs)
        yt = self._to_tensor(ys).unsqueeze(-1)

        # Disable BoTorch's default outcome_transform (Standardize) — we already
        # standardize y via self.y_scaler, and BoTorch's transform makes
        # save/load awkward because Standardize re-fits during __init__ and
        # clobbers the loaded state_dict's mean/stdev buffers.
        if y_var is not None:
            sigma_y = float(self.y_scaler.scale_[0])
            y_var_scaled = (y_var.astype(np.float64) / (sigma_y ** 2)).reshape(-1, 1)
            yvar_t = self._to_tensor(y_var_scaled)
            self.model = SingleTaskGP(
                Xt, yt, train_Yvar=yvar_t, outcome_transform=None,
            ).to(self.device)
            self._is_fixed_noise = True
        else:
            self.model = SingleTaskGP(Xt, yt, outcome_transform=None).to(self.device)
            self._is_fixed_noise = False

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.model.eval()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma) in the original y-scale."""
        if self.model is None:
            raise ValueError("Surrogate not fitted.")
        Xs = self.x_scaler.transform(X)
        Xt = self._to_tensor(Xs)

        with torch.no_grad():
            posterior = self.model.posterior(Xt)
            mu_s = posterior.mean.detach().cpu().numpy().ravel()
            sigma_s = posterior.variance.sqrt().detach().cpu().numpy().ravel()

        sigma_y = float(self.y_scaler.scale_[0])
        mu = self.y_scaler.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        sigma = sigma_s * sigma_y
        return mu, sigma

    # ----- persistence --------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # For fixed-noise GP, the train_Yvar is stored in the likelihood's
        # `noise_covar.noise` buffer (a (N,) tensor in the *standardized* y-scale).
        # We extract it explicitly so reload reconstructs a proper noise vector.
        train_yvar = None
        is_fixed_noise = getattr(self, "_is_fixed_noise", False)
        if self.model is not None and is_fixed_noise:
            try:
                # FixedNoiseGaussianLikelihood (used internally by SingleTaskGP
                # when train_Yvar is supplied) stores noise on .noise_covar.noise.
                noise = self.model.likelihood.noise_covar.noise
                train_yvar = noise.detach().cpu().reshape(-1, 1).clone()
            except AttributeError:
                # Fallback: try common alternative location
                train_yvar = getattr(self.model, "train_Yvar", None)
                if train_yvar is not None:
                    train_yvar = train_yvar.detach().cpu().clone()

        state = {
            "device": self.device,
            "dtype": str(self.dtype),
            "x_scaler": self.x_scaler,
            "y_scaler": self.y_scaler,
            "model_state": self.model.state_dict() if self.model is not None else None,
            "model_train_inputs": [t.cpu() for t in self.model.train_inputs] if self.model is not None else None,
            "model_train_targets": self.model.train_targets.cpu() if self.model is not None else None,
            "model_train_yvar": train_yvar,    # (N,1) in standardized y-scale, or None
            "is_fixed_noise": is_fixed_noise,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "HCapSurrogate":
        with open(path, "rb") as f:
            state = pickle.load(f)
        dev = device or state["device"]
        dtype = torch.float64 if "float64" in state["dtype"] else torch.float32

        obj = cls(device=dev, dtype=dtype)
        obj.x_scaler = state["x_scaler"]
        obj.y_scaler = state["y_scaler"]

        Xt = state["model_train_inputs"][0].to(device=dev, dtype=dtype)
        yt = state["model_train_targets"].to(device=dev, dtype=dtype)
        if yt.dim() == 1:
            yt = yt.unsqueeze(-1)

        if state["is_fixed_noise"]:
            train_yvar = state.get("model_train_yvar")
            if train_yvar is None:
                raise RuntimeError(
                    "Saved surrogate is_fixed_noise=True but no train_yvar persisted; "
                    "save() may pre-date the fix in this method."
                )
            yvar_t = train_yvar.to(device=dev, dtype=dtype)
            obj.model = SingleTaskGP(
                Xt, yt, train_Yvar=yvar_t, outcome_transform=None,
            ).to(dev)
            obj._is_fixed_noise = True
        else:
            obj.model = SingleTaskGP(Xt, yt, outcome_transform=None).to(dev)
            obj._is_fixed_noise = False
        obj.model.load_state_dict(state["model_state"])
        obj.model.eval()
        return obj
