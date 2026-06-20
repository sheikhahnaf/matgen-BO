"""Non-negative PU risk classifier (Kiryo et al., 2017).

Risk formulation (sigmoid surrogate, l(z) = sigmoid(-z)):
    R_p_plus  = E_{x in P}[ l(+g(x)) ]
    R_p_minus = E_{x in P}[ l(-g(x)) ]
    R_u_minus = E_{x in U}[ l(-g(x)) ]
    R = pi * R_p_plus + max(0, R_u_minus - pi * R_p_minus)

When (R_u_minus - pi * R_p_minus) < 0, we clamp to zero (non-negative correction).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _build_mlp(n_in: int, hidden: tuple) -> nn.Sequential:
    layers = []
    prev = n_in
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))   # logit output; no activation
    return nn.Sequential(*layers)


class NNPUClassifier:
    """Non-negative PU risk classifier (Kiryo et al., 2017).

    Parameters
    ----------
    pi : float
        Class prior P(Y=1). Default 0.3.
    hidden : tuple of int
        Hidden layer sizes for the MLP. Default (128, 64).
    epochs : int
        Training epochs (full-batch). Default 100.
    lr : float
        Adam learning rate. Default 1e-3.
    seed : int
        Random seed for torch and numpy. Default 0.
    """

    def __init__(self, pi: float = 0.3, hidden: tuple = (128, 64),
                 epochs: int = 100, lr: float = 1e-3, seed: int = 0, **kw):
        self.pi = pi
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def fit(self, X: np.ndarray, s: np.ndarray) -> "NNPUClassifier":
        """Train on PU data.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Feature matrix.
        s : array-like, shape (n,)
            Label indicator: 1 = labeled positive, 0 = unlabeled.

        Returns
        -------
        self
        """
        # --- reproducibility ---
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)

        # --- standardize ---
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X_norm = (X - self.mean_) / self.std_

        # --- split positives vs unlabeled ---
        pos_idx = np.where(s == 1)[0]
        unl_idx = np.where(s == 0)[0]

        X_pos = torch.tensor(X_norm[pos_idx])      # (n_p, d)
        X_unl = torch.tensor(X_norm[unl_idx])      # (n_u, d)

        n_features = X_norm.shape[1]
        self.net_ = _build_mlp(n_features, tuple(self.hidden))

        optimizer = optim.Adam(self.net_.parameters(), lr=self.lr)

        pi = torch.tensor(self.pi, dtype=torch.float32)

        # sigmoid(-z) = probability of wrong-sign prediction
        # l(+g(x)) = sigmoid(-g(x))   <- positive loss for +1 label
        # l(-g(x)) = sigmoid(+g(x))   <- positive loss for -1 label

        self.net_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()

            g_pos = self.net_(X_pos).squeeze(1)   # (n_p,)
            g_unl = self.net_(X_unl).squeeze(1)   # (n_u,)

            R_p_plus  = torch.sigmoid(-g_pos).mean()
            R_p_minus = torch.sigmoid( g_pos).mean()
            R_u_minus = torch.sigmoid( g_unl).mean()

            correction = R_u_minus - pi * R_p_minus

            # Non-negative correction: if clamped, the gradient still flows
            # through the positive term (pi * R_p_plus) only.
            loss = pi * R_p_plus + torch.clamp(correction, min=0.0)

            loss.backward()
            optimizer.step()

        self.net_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(positive) for each sample.

        Parameters
        ----------
        X : array-like, shape (n, d)

        Returns
        -------
        p : np.ndarray, shape (n,)
        """
        X = np.asarray(X, dtype=np.float32)
        X_norm = (X - self.mean_) / self.std_
        X_t = torch.tensor(X_norm)

        with torch.no_grad():
            logits = self.net_(X_t).squeeze(1)
            proba = torch.sigmoid(logits)

        return proba.numpy()
