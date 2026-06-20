"""Abstaining PU bagging classifier (feature-based port of the notebook's
AbstainingPUClassifier).

The notebook's class operated on SMILES + Morgan fingerprints; this version takes a
precomputed feature matrix X (our ORB+Magpie(+stability) bank), so no molecular
substrate is involved.  The abstention/OOD/decision policy is preserved:

  * confidence  : abstain if |mean_proba - decision_threshold| < confidence_threshold
  * disagreement: abstain if bag-std > disagreement_threshold
  * OOD         : abstain if normalized distance-to-positive-training > ood_threshold

Deliberate deviation from the notebook for OOD: the notebook used cosine distance on
binary Morgan fingerprints.  Our features are continuous ORB/Magpie descriptors on
very different scales, so we standardize (z-score, fit on positive training rows) and
use Euclidean kNN — cosine on raw mixed-scale features would be dominated by the
largest-magnitude Magpie columns.  Distances are normalized by the 95th percentile of
positive-training distances, so ood_threshold≈1 means "beyond the bulk of positives".

Bagging follows Mordelet & Vert: every bag keeps ALL positives and draws
neg_sample_ratio * n_pos unlabeled as pseudo-negatives (no positive subsampling).
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from .models import make_model, _ProbaAdapter


def _fit_bag(base, base_kw, seed, X, pos, unl, neg_ratio, b):
    rng = np.random.default_rng(seed + b)
    n_neg = min(int(round(len(pos) * neg_ratio)), len(unl))
    n_neg = max(n_neg, 1)
    neg = rng.choice(unl, size=n_neg, replace=False)
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    m = _ProbaAdapter(make_model(base, seed=seed + b, n_jobs=1, **base_kw))
    m.fit(X[idx], y)
    return m


class AbstainingPUClassifier:
    def __init__(self, base="xgboost", n_bags=10, neg_sample_ratio=1.0, seed=0,
                 confidence_threshold=0.15, disagreement_threshold=0.25,
                 ood_threshold=1.0, decision_threshold=0.5, n_jobs=-1, **base_kw):
        self.base = base
        self.n_bags = int(n_bags)
        self.neg_sample_ratio = float(neg_sample_ratio)
        self.seed = int(seed)
        self.confidence_threshold = float(confidence_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.ood_threshold = float(ood_threshold)
        self.decision_threshold = float(decision_threshold)
        self.n_jobs = n_jobs
        self.base_kw = base_kw
        self.models_ = []

    # -- fit --
    def fit(self, X, s):
        X = np.asarray(X, dtype=np.float32)
        s = np.asarray(s)
        pos = np.where(s == 1)[0]
        unl = np.where(s == 0)[0]
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_bag)(self.base, self.base_kw, self.seed, X, pos, unl,
                              self.neg_sample_ratio, b)
            for b in range(self.n_bags)
        )
        # OOD detector on standardized positive features
        Xp = X[pos]
        self.mu_ = Xp.mean(axis=0)
        self.sd_ = Xp.std(axis=0) + 1e-8
        Xp_n = (Xp - self.mu_) / self.sd_
        k = int(min(5, max(1, len(pos) - 1)))
        self.nn_ = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=self.n_jobs)
        self.nn_.fit(Xp_n)
        d, _ = self.nn_.kneighbors(Xp_n)
        self.dist95_ = float(np.percentile(d.mean(axis=1), 95)) + 1e-6
        return self

    # -- scoring --
    def predict_proba_all_bags(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.array([m.predict_proba(X) for m in self.models_])  # (n_bags, n)

    def predict_proba(self, X):
        return self.predict_proba_all_bags(X).mean(axis=0)

    def bag_disagreement(self, X):
        return self.predict_proba_all_bags(X).std(axis=0)

    def ood_scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mu_) / self.sd_
        d, _ = self.nn_.kneighbors(Xn)
        return d.mean(axis=1) / self.dist95_

    # -- decision with abstention --
    def predict(self, X):
        all_probs = self.predict_proba_all_bags(X)
        mean_p = all_probs.mean(axis=0)
        std_p = all_probs.std(axis=0)
        ood = self.ood_scores(X)
        t = self.decision_threshold
        abst_conf = np.abs(mean_p - t) < self.confidence_threshold
        abst_dis = std_p > self.disagreement_threshold
        abst_ood = ood > self.ood_threshold
        abstain = abst_conf | abst_dis | abst_ood
        pred = np.where(mean_p > t, 1, 0).astype(float)
        pred[abstain] = -1
        return {
            "predictions": pred,
            "probabilities": mean_p,
            "disagreement": std_p,
            "ood_scores": ood,
            "abstain": abstain.astype(float),
            "abstain_confidence": abst_conf,
            "abstain_disagreement": abst_dis,
            "abstain_ood": abst_ood,
        }
