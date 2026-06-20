import numpy as np
from apu_synthesizability.metrics import pu_metrics, PUMetrics

def _perfect_scores(y):           # scores that perfectly separate planted positives
    return y.astype(float) * 0.9 + 0.05

def test_returns_pumetrics_dataclass():
    y = np.array([1,1,0,0,1,0]); s = _perfect_scores(y)
    m = pu_metrics(scores=s, planted=y, labeled_pos_scores=np.array([0.95,0.96]), threshold=0.5)
    assert isinstance(m, PUMetrics)

def test_perfect_separation_gives_auroc_one():
    y = np.array([1,1,1,0,0,0]); s = _perfect_scores(y)
    m = pu_metrics(scores=s, planted=y, labeled_pos_scores=np.array([0.9,0.9]), threshold=0.5)
    assert m.proxy_auroc == 1.0
    assert m.proxy_auprc > 0.99

def test_tpr_on_labeled_counts_fraction_above_threshold():
    y = np.array([1,0]); s = np.array([0.9,0.1])
    m = pu_metrics(scores=s, planted=y, labeled_pos_scores=np.array([0.8,0.2,0.9,0.6]), threshold=0.5)
    assert m.tpr_on_labeled == 0.75   # 3 of 4 above 0.5

def test_lee_liu_pu_score_matches_formula():
    y = np.array([1,1,0,0]); s = np.array([0.9,0.4,0.6,0.1])  # recall on planted, P(pred+)
    m = pu_metrics(scores=s, planted=y, labeled_pos_scores=np.array([0.9]), threshold=0.5)
    recall = 1/2                       # planted positives scored >0.5: only the 0.9
    p_pred_pos = 2/4                   # scores>0.5: 0.9, 0.6
    assert abs(m.pu_score - recall**2 / p_pred_pos) < 1e-9

def test_ece_zero_for_calibrated_constant():
    # all planted with score 1.0, all non-planted with score 0.0 -> perfectly calibrated
    y = np.array([1,1,0,0]); s = np.array([1.0,1.0,0.0,0.0])
    m = pu_metrics(scores=s, planted=y, labeled_pos_scores=np.array([1.0]), threshold=0.5, n_ece_bins=10)
    assert m.ece < 1e-9
