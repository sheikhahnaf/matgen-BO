from dataclasses import dataclass
import numpy as np

@dataclass
class PlantedSplit:
    train_mask: np.ndarray   # bool over all rows
    test_mask: np.ndarray
    planted: np.ndarray      # int over all rows: 1 if a planted positive in the test pool

@dataclass
class Bag:
    pos_idx: np.ndarray
    neg_idx: np.ndarray

def planted_test_split(s, test_frac=0.2, plant_frac=0.5, seed=0) -> PlantedSplit:
    """Hold out a test pool; plant a fraction of held-out KNOWN positives into the test unlabeled
    pool (marked planted=1) so AUROC/recall on them estimates recovery of hidden positives."""
    s = np.asarray(s, int); n = len(s); rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test = idx[: int(test_frac * n)]
    train_mask = np.ones(n, bool); train_mask[test] = False
    test_mask = ~train_mask
    planted = np.zeros(n, int)
    test_pos = test[s[test] == 1]
    n_plant = int(plant_frac * len(test_pos))
    planted[test_pos[:n_plant]] = 1
    return PlantedSplit(train_mask, test_mask, planted)

def mv_bag_indices(pos_idx, unl_idx, seed=0) -> Bag:
    """Mordelet-Vert bag: all positives + an equal-size random draw from unlabeled as pseudo-neg."""
    rng = np.random.default_rng(seed)
    neg = rng.choice(unl_idx, size=len(pos_idx), replace=len(unl_idx) < len(pos_idx))
    return Bag(np.asarray(pos_idx), np.asarray(neg))
