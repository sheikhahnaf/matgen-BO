import numpy as np
from apu_synthesizability.pu import planted_test_split, mv_bag_indices

def test_planted_split_hides_positives_in_test_unlabeled(toy_pu):
    X, s, _ = toy_pu
    sp = planted_test_split(s, test_frac=0.2, plant_frac=0.5, seed=0)
    # planted points are known positives moved into the test pool, marked planted=1 and s=0 in test
    assert sp.planted.sum() > 0
    # no train positive appears in the test planted set
    assert set(np.where(sp.test_mask & (sp.planted == 1))[0]).isdisjoint(np.where(sp.train_mask & (s == 1))[0])

def test_mv_bag_balances_positives_and_sampled_unlabeled(toy_pu):
    X, s, _ = toy_pu
    pos = np.where(s == 1)[0]; unl = np.where(s == 0)[0]
    bag = mv_bag_indices(pos, unl, seed=1)
    assert len(bag.pos_idx) == len(pos)
    assert len(bag.neg_idx) == len(pos)          # equal-size pseudo-negative draw
    assert set(bag.neg_idx).issubset(set(unl))
