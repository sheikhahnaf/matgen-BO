import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from .pu import mv_bag_indices


def make_model(kind, seed=0, n_jobs=1, **kw):
    # kw overrides the defaults below (used by Optuna tuning to inject hyperparameters);
    # with no kw this reproduces the fixed-HP defaults used by the frozen sweep.
    if kind == "rf":
        params = dict(n_estimators=200, n_jobs=n_jobs, random_state=seed)
        params.update(kw)
        return RandomForestClassifier(**params)
    if kind == "mlp":
        params = dict(hidden_layer_sizes=(128, 64), max_iter=300, random_state=seed)
        params.update(kw)
        return MLPClassifier(**params)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      tree_method="hist", n_jobs=n_jobs, random_state=seed)
        params.update(kw)
        return XGBClassifier(**params)
    if kind == "nnpu":
        from .nnpu import NNPUClassifier        # implemented in Task 3b
        return NNPUClassifier(seed=seed, **kw)
    raise ValueError(f"unknown model kind {kind}")


class _ProbaAdapter:
    def __init__(self, m): self.m = m
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p


def _fit_one_bag(base, base_kw, seed, X, pos, unl, b):
    """Train a single Mordelet-Vert bag (module-level so joblib pickles it cleanly).

    Each bag keeps ALL positives and draws an equal-size random unlabeled sample
    as pseudo-negatives.  The base estimator runs single-threaded (n_jobs=1):
    parallelism lives at the bag level, so cores are not oversubscribed by
    nesting estimator threads inside already-parallel bags.
    """
    bag = mv_bag_indices(pos, unl, seed=seed + b)
    idx = np.concatenate([bag.pos_idx, bag.neg_idx])
    y = np.concatenate([np.ones(len(bag.pos_idx)), np.zeros(len(bag.neg_idx))])
    m = _ProbaAdapter(make_model(base, seed=seed + b, n_jobs=1, **base_kw))
    m.fit(X[idx], y)
    return m


class PUBaggingClassifier:
    """Mordelet-Vert bagging over any base classifier; averages per-bag P(positive).

    Bags train in parallel across CPU cores (``n_jobs``).  Following
    Mordelet & Vert (2014) every bag uses *all* labeled positives plus a random
    equal-size draw from the unlabeled pool as pseudo-negatives — ensemble
    diversity comes from the unlabeled draw, NOT from subsampling positives.
    Positives are never discarded (that would throw away signal and depress
    recall/AUPRC); speed at scale comes from bag-level parallelism instead.
    """
    def __init__(self, base="xgboost", n_bags=20, seed=0, n_jobs=-1, **base_kw):
        self.base = base
        self.n_bags = n_bags
        self.seed = seed
        self.n_jobs = n_jobs
        self.base_kw = base_kw
        self.models_ = []

    def fit(self, X, s):
        s = np.asarray(s)
        pos = np.where(s == 1)[0]
        unl = np.where(s == 0)[0]
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_bag)(self.base, self.base_kw, self.seed, X, pos, unl, b)
            for b in range(self.n_bags)
        )
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)
