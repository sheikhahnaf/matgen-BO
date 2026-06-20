import numpy as np
from apu_synthesizability.models import make_model, PUBaggingClassifier

def test_make_model_known_kinds():
    for kind in ["xgboost", "rf", "mlp"]:
        m = make_model(kind, seed=0)
        assert hasattr(m, "fit") and hasattr(m, "predict_proba")

def test_pubagging_learns_toy(toy_pu):
    X, s, y = toy_pu
    clf = PUBaggingClassifier(base="rf", n_bags=10, seed=0)   # rf = no GPU, fast
    clf.fit(X, s)
    p = clf.predict_proba(X)
    # mean score for true positives should exceed that for true negatives
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2
