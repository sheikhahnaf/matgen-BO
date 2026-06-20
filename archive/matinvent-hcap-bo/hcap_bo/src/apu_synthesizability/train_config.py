"""train_config — run one APU synthesizability sweep config end to end.

Loads a pre-built feature bank (.npz), assembles the feature matrix from
the blocks listed in ``cfg["features"]``, trains the requested PU classifier,
evaluates on the test split with the planted-positive protocol, computes
PUMetrics, writes a JSON result, and returns the result dict.

CLI usage::

    python -m apu_synthesizability.train_config \\
        --config configs/sweep/rf_orb.yaml \\
        --bank   data/feature_bank.npz \\
        --out    results/rf_orb.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .metrics import pu_metrics
from .models import PUBaggingClassifier
from .nnpu import NNPUClassifier

# ---------------------------------------------------------------------------
# Feature blocks that are produced by external prep and are NOT in a basic
# bank created by build_feature_bank (magpie, orb_pca, stability, label, split).
# Requesting one of these when the key is absent produces a clear error.
# ---------------------------------------------------------------------------
_PREP_ONLY_BLOCKS = {"cgnf_score", "mp_props"}


def _load_bank(bank_path: str) -> Dict[str, np.ndarray]:
    """Load a .npz bank and return a dict of arrays."""
    d = np.load(bank_path, allow_pickle=True)
    return dict(d)


def _assemble_X(bank: Dict[str, np.ndarray], feature_names: list) -> np.ndarray:
    """Horizontally concatenate feature blocks by name.

    Parameters
    ----------
    bank:
        Dict of arrays loaded from the .npz.
    feature_names:
        Ordered list of block names to concatenate.

    Returns
    -------
    X : np.ndarray, shape (N, sum_of_dims)

    Raises
    ------
    ValueError
        If a requested block is absent and is one of the prep-only blocks
        (cgnf_score, mp_props), or for any missing block.
    """
    blocks = []
    for name in feature_names:
        if name not in bank:
            if name in _PREP_ONLY_BLOCKS:
                raise ValueError(
                    f"Feature block '{name}' is not present in the bank. "
                    f"This block is produced by the prep pipeline "
                    f"(apu_synthesizability.prep) and must be added before "
                    f"running the sweep. Available blocks: {list(bank.keys())}"
                )
            raise ValueError(
                f"Feature block '{name}' is not present in the bank. "
                f"Available blocks: {list(bank.keys())}"
            )
        arr = bank[name]
        # Ensure 2-D: (N,) → (N, 1), (N, 1) stays as-is
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        blocks.append(arr.astype(np.float32))
    return np.concatenate(blocks, axis=1)


def run_config(cfg: dict, bank_path: str, out_json: str) -> dict:
    """Run one config end to end: load bank → assemble X → PU train → eval → JSON.

    Parameters
    ----------
    cfg:
        Config dict.  Expected keys:

        - ``features``   : list of feature-block names to concatenate.
        - ``arch``       : ``"nnpu"`` or a base-model name for PUBaggingClassifier
                           (``"rf"``, ``"mlp"``, ``"xgboost"``).
        - ``n_bags``     : int, number of bags for PUBaggingClassifier (default 20).
        - ``seed``       : int, random seed (default 0).

    bank_path:
        Path to a .npz feature bank produced by ``build_feature_bank``
        (must contain ``label`` and ``split`` arrays).
    out_json:
        Path to write the JSON result.

    Returns
    -------
    result : dict
        ``{**cfg, **pu_metrics.to_dict(), "n_train": int, "n_test": int}``
    """
    # ------------------------------------------------------------------
    # 1. Load bank
    # ------------------------------------------------------------------
    bank = _load_bank(bank_path)

    label = bank["label"].astype(int)          # (N,)  1=positive, 0=unlabeled
    split = bank["split"].astype(str)           # (N,)  "train"/"val"/"test"

    # ------------------------------------------------------------------
    # 2. Assemble feature matrix
    # ------------------------------------------------------------------
    X = _assemble_X(bank, cfg["features"])     # (N, D)

    # ------------------------------------------------------------------
    # 3. Train/test masks
    # ------------------------------------------------------------------
    train_mask = np.isin(split, ["train", "val"])
    test_mask  = split == "test"

    X_train = X[train_mask]
    s_train  = label[train_mask]

    X_test  = X[test_mask]
    s_test  = label[test_mask]    # 1 = positive, 0 = unlabeled

    # ------------------------------------------------------------------
    # 4. Fit model
    # ------------------------------------------------------------------
    seed = int(cfg.get("seed", 0))

    if cfg.get("arch") == "nnpu":
        model = NNPUClassifier(seed=seed)
    else:
        model = PUBaggingClassifier(
            base=cfg["arch"],
            n_bags=int(cfg.get("n_bags", 20)),
            seed=seed,
        )

    model.fit(X_train, s_train)

    # ------------------------------------------------------------------
    # 5. Evaluation: clean held-out protocol on the TEST split only
    #
    # Both classes are held out from training.  The eval pool is the test
    # positives (planted=1) versus the test unlabeled (planted=0).  We never
    # touch the train-unlabeled rows here — they were used as pseudo-negatives
    # during PU training, so scoring positives against them would leak and
    # inflate the metric (and would not be comparable to CGNF's held-out
    # numbers).  ``tpr_on_labeled`` becomes CGNF-style recall on the held-out
    # test positives; AUROC/AUPRC are test_pos vs test_unl.
    # ------------------------------------------------------------------
    test_pos_idx = np.where(s_test == 1)[0]    # held-out positives
    test_unl_idx = np.where(s_test == 0)[0]    # held-out unlabeled

    # Eval pool = test positives ∪ test unlabeled
    pool_idx     = np.concatenate([test_pos_idx, test_unl_idx])
    pool_planted = np.concatenate([
        np.ones(len(test_pos_idx), dtype=int),
        np.zeros(len(test_unl_idx), dtype=int),
    ])

    pool_scores        = model.predict_proba(X_test[pool_idx])
    labeled_pos_scores = model.predict_proba(X_test[test_pos_idx])

    # ------------------------------------------------------------------
    # 6. Metrics
    # ------------------------------------------------------------------
    m = pu_metrics(
        scores=pool_scores,
        planted=pool_planted,
        labeled_pos_scores=labeled_pos_scores,
    )

    # ------------------------------------------------------------------
    # 7. Build result dict and write JSON
    # ------------------------------------------------------------------
    result: dict = {
        **cfg,
        **m.to_dict(),
        "n_train": int(train_mask.sum()),
        "n_test":  int(test_mask.sum()),
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one APU synthesizability config end to end."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--bank", required=True,
                        help="Path to .npz feature bank.")
    parser.add_argument("--out", required=True,
                        help="Path to write JSON result.")
    args = parser.parse_args()

    import yaml  # noqa: lazy — optional dep for CLI only
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    result = run_config(cfg, args.bank, args.out)

    # Print key metrics to stdout
    print(f"Config: {cfg.get('name', args.config)}")
    print(f"  arch          : {cfg.get('arch')}")
    print(f"  features      : {cfg.get('features')}")
    print(f"  n_train       : {result['n_train']}")
    print(f"  n_test        : {result['n_test']}")
    print(f"  proxy_auroc   : {result.get('proxy_auroc', float('nan')):.4f}")
    print(f"  proxy_auprc   : {result.get('proxy_auprc', float('nan')):.4f}")
    print(f"  tpr_on_labeled: {result.get('tpr_on_labeled', float('nan')):.4f}")
    print(f"  ece           : {result.get('ece', float('nan')):.4f}")
    print(f"  pu_score      : {result.get('pu_score', float('nan')):.4f}")
    print(f"Result written → {args.out}")


if __name__ == "__main__":
    main()
