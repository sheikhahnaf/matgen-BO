import numpy as np
from apu_synthesizability.nnpu import NNPUClassifier

def test_nnpu_separates_toy(toy_pu):
    X, s, y = toy_pu
    clf = NNPUClassifier(pi=0.3, epochs=150, lr=1e-2, seed=0).fit(X, s)
    p = clf.predict_proba(X)
    assert p.shape == (X.shape[0],)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.15
