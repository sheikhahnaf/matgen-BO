import numpy as np
import pytest

@pytest.fixture
def toy_pu():
    """Separable toy PU problem: 200 positives, 600 unlabeled (30% hidden positives)."""
    rng = np.random.default_rng(0)
    n_pos, n_unl = 200, 600
    Xp = rng.normal(2.0, 1.0, size=(n_pos, 8))
    n_hidden = int(0.3 * n_unl)
    Xu_pos = rng.normal(2.0, 1.0, size=(n_hidden, 8))      # hidden positives
    Xu_neg = rng.normal(-2.0, 1.0, size=(n_unl - n_hidden, 8))
    X = np.vstack([Xp, Xu_pos, Xu_neg])
    s = np.concatenate([np.ones(n_pos), np.zeros(n_unl)])           # labeled indicator
    y_true = np.concatenate([np.ones(n_pos + n_hidden), np.zeros(n_unl - n_hidden)])  # oracle (eval only)
    return X, s, y_true
