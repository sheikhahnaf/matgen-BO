from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

@dataclass
class PUMetrics:
    tpr_on_labeled: float
    proxy_precision: float
    proxy_recall: float
    proxy_f1: float
    proxy_auroc: float
    proxy_auprc: float
    pu_score: float          # Lee-Liu: recall^2 / P(predict positive)
    ece: float
    n_planted: int
    n_test: int
    def to_dict(self):
        return asdict(self)

def expected_calibration_error(scores, labels, n_bins=10):
    scores = np.asarray(scores, float); labels = np.asarray(labels, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    ece = 0.0; n = len(scores)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = scores[m].mean(); acc = labels[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)

def pu_metrics(scores, planted, labeled_pos_scores, threshold=0.5, n_ece_bins=10) -> PUMetrics:
    """
    scores: model scores on the test pool (planted positives + unlabeled rest), in [0,1].
    planted: 1 if the test point is a planted (known) positive, else 0. Same length as scores.
    labeled_pos_scores: model scores on held-out KNOWN positives (the labeled test set).
    """
    scores = np.asarray(scores, float); planted = np.asarray(planted, int)
    lps = np.asarray(labeled_pos_scores, float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (planted == 1)).sum())
    fp = int(((pred == 1) & (planted == 0)).sum())
    fn = int(((pred == 0) & (planted == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    auroc = float(roc_auc_score(planted, scores)) if planted.min() != planted.max() else float("nan")
    auprc = float(average_precision_score(planted, scores)) if planted.max() > 0 else float("nan")
    p_pred_pos = float((scores >= threshold).mean()) or 1e-12
    pu_score = (recall ** 2) / p_pred_pos
    tpr_on_labeled = float((lps >= threshold).mean()) if len(lps) else float("nan")
    ece = expected_calibration_error(scores, planted, n_ece_bins)
    return PUMetrics(tpr_on_labeled, precision, recall, f1, auroc, auprc, pu_score, ece,
                     int(planted.sum()), int(len(scores)))
