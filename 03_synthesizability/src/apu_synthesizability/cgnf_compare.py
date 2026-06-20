"""CGNF-matched head-to-head comparison on our held-out test split.

CGNF (Jang et al., Matter 2024) evaluates with a *pure PU held-out protocol* — NOT
an e_hull/SAScore proxy.  It holds out ~20% of known-synthesized compositions as the
positive test set, treats the unlabeled pool as pseudo-negatives, and reports:
  * TPR  = recall on the held-out KNOWN positives, and
  * estimated precision = a class-prior PU estimate (Elkan & Noto, 2008, SCAR).
It also reports a prior-adjusted decision threshold (its 0.741) alongside 0.5.

Our bank's ``test`` split already implements that protocol (held-out positives +
held-out unlabeled).  This module scores BOTH pretrained CGNF and each A-PU config
on the *same* test split and computes one identical metric panel, so the comparison
is apples-to-apples on identical data rather than against CGNF's paper numbers
(which are on CGNF's own dataset).

Estimated precision (class-prior PU, SCAR assumption):
    c    = mean score over held-out positives                 # P(s=1 | y=1) label freq
    beta = mean score over unlabeled / c   (clipped to [0,1]) # prior P(y=1) in unlabeled
    r    = recall on held-out positives at threshold t
    q_u  = predicted-positive rate on unlabeled at threshold t
    est_precision = clip( r * beta / q_u , 0, 1 )
Intuition: of the unlabeled flagged positive, the expected truly-positive fraction is
r*beta/q_u.  Perfect model -> q_u == beta -> precision 1; predict-all-positive ->
q_u == 1 -> precision == beta (the prior).  Identical estimator applied to CGNF and
A-PU, so the *comparison* is fair regardless of any residual estimator bias.

Nothing here writes to results/apu/ or cache/bank.npz — outputs go to a new dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .metrics import expected_calibration_error
from .train_config import _assemble_X, _load_bank
from .models import PUBaggingClassifier
from .nnpu import NNPUClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Matched PU metric panel
# ---------------------------------------------------------------------------

def _panel_at_threshold(pos_scores, unl_scores, beta, t) -> Dict[str, float]:
    """Threshold-dependent metrics at decision threshold ``t``."""
    r = float((pos_scores >= t).mean())              # recall / TPR on held-out positives
    q_u = float((unl_scores >= t).mean())            # predicted-positive rate on unlabeled
    tp = int((pos_scores >= t).sum())
    fp = int((unl_scores >= t).sum())
    prec_pess = float(tp / (tp + fp)) if (tp + fp) else 0.0   # unlabeled hidden-pos as FP
    # est_precision is over the representative (unlabeled) pool; undefined if nothing
    # is predicted positive there (q_u == 0), so report NaN rather than a spurious 0.
    est_prec = float(np.clip(r * beta / q_u, 0.0, 1.0)) if q_u > 0 else float("nan")
    f1_est = (float(2 * est_prec * r / (est_prec + r))
              if (est_prec == est_prec and (est_prec + r) > 0) else float("nan"))
    return {
        "threshold": float(t),
        "tpr": r,
        "predpos_rate_unl": q_u,
        "precision_pessimistic": prec_pess,
        "precision_estimated": est_prec,
        "f1_estimated": f1_est,
    }


def pu_panel(pos_scores, unl_scores) -> Dict[str, float]:
    """Full CGNF-matched panel from held-out positive scores and unlabeled scores.

    Ranking metrics (AUROC/AUPRC) and ECE treat held-out positives as label 1 and
    unlabeled as label 0 — identical to ``metrics.pu_metrics`` so re-scored configs
    reproduce the frozen numbers (a determinism check).
    """
    pos_scores = np.asarray(pos_scores, float)
    unl_scores = np.asarray(unl_scores, float)

    # Elkan-Noto label frequency + prior among unlabeled
    c = float(pos_scores.mean())
    beta = float(np.clip(unl_scores.mean() / c, 0.0, 1.0)) if c > 0 else 0.0

    # prior-adjusted threshold: choose t so predicted-positive rate on unlabeled == beta
    if beta <= 0.0:
        t_prior = 1.0
    elif beta >= 1.0:
        t_prior = 0.0
    else:
        t_prior = float(np.quantile(unl_scores, 1.0 - beta))

    # ranking metrics (prior-free), identical labelling to metrics.pu_metrics
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(unl_scores))]).astype(int)
    s = np.concatenate([pos_scores, unl_scores])
    auroc = float(roc_auc_score(y, s)) if y.min() != y.max() else float("nan")
    auprc = float(average_precision_score(y, s)) if y.max() > 0 else float("nan")
    ece = expected_calibration_error(s, y)

    out: Dict[str, float] = {
        "n_pos": int(len(pos_scores)),
        "n_unl": int(len(unl_scores)),
        "label_freq_c": c,
        "prior_beta": beta,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
    }
    for tag, t in (("at0.5", 0.5), ("atprior", t_prior)):
        for k, v in _panel_at_threshold(pos_scores, unl_scores, beta, t).items():
            out[f"{k}_{tag}"] = v
    return out


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

def score_apu_config(cfg: dict, bank_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Re-train one A-PU config on train+val and score the held-out test split.

    Returns (test_positive_scores, test_unlabeled_scores).  Deterministic given the
    config seed, so AUROC/AUPRC/TPR reproduce the frozen results/apu numbers.
    """
    bank = _load_bank(bank_path)
    label = bank["label"].astype(int)
    split = bank["split"].astype(str)
    X = _assemble_X(bank, cfg["features"])

    tr = np.isin(split, ["train", "val"])
    te = split == "test"
    seed = int(cfg.get("seed", 0))

    if cfg.get("arch") == "nnpu":
        model = NNPUClassifier(seed=seed)
    else:
        model = PUBaggingClassifier(base=cfg["arch"], n_bags=int(cfg.get("n_bags", 20)), seed=seed)
    model.fit(X[tr], label[tr])

    X_te, lab_te = X[te], label[te]
    pos_scores = model.predict_proba(X_te[lab_te == 1])
    unl_scores = model.predict_proba(X_te[lab_te == 0])
    return np.asarray(pos_scores, float), np.asarray(unl_scores, float)


class _CompositionHolder:
    """Minimal stand-in exposing ``.composition`` for CGNF's get_dataset()."""
    __slots__ = ("composition",)

    def __init__(self, composition):
        self.composition = composition


def score_cgnf(formulas: List[str]) -> np.ndarray:
    """Score formulas with pretrained CGNF (composition-only).

    Requires the syn_score package importable as ``rewards.calculators.syn_score``
    (set PYTHONPATH to the matinvent dir on Grace).  Elements absent from the CGNF
    element embedding yield NaN (caller filters).
    """
    import json as _json
    from pymatgen.core import Composition
    from rewards.calculators.syn_score import EMB_PATH
    from rewards.calculators.syn_score.predict import predict

    with open(EMB_PATH) as fh:
        emb = _json.load(fh)

    holders, ok_mask = [], []
    for f in formulas:
        try:
            comp = Composition(f)
            els = comp.reduced_composition.get_el_amt_dict()
            ok = all(str(e) in emb for e in els)
        except Exception:
            ok = False
        ok_mask.append(ok)
        if ok:
            holders.append(_CompositionHolder(comp))

    scores_ok = np.asarray(predict(holders), float) if holders else np.array([])
    out = np.full(len(formulas), np.nan, float)
    j = 0
    for i, ok in enumerate(ok_mask):
        if ok:
            out[i] = scores_ok[j]
            j += 1
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(configs_dir: str, bank_path: str, out_dir: str,
        runnable_features=("orb_pca", "magpie", "stability")) -> dict:
    import glob, os, yaml

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bank = _load_bank(bank_path)
    split = bank["split"].astype(str)
    label = bank["label"].astype(int)
    formula = bank["formula"].astype(str)
    te = split == "test"

    runnable = set(runnable_features)
    panels: Dict[str, dict] = {}

    # --- CGNF on the same test split (positives + unlabeled) ---
    test_pos_formulas = formula[te][label[te] == 1].tolist()
    test_unl_formulas = formula[te][label[te] == 0].tolist()
    print(f"[cgnf] scoring {len(test_pos_formulas)} test-pos + {len(test_unl_formulas)} test-unl formulas")
    cg_pos = score_cgnf(test_pos_formulas)
    cg_unl = score_cgnf(test_unl_formulas)
    cg_pos_v = cg_pos[~np.isnan(cg_pos)]
    cg_unl_v = cg_unl[~np.isnan(cg_unl)]
    panels["CGNF_pretrained"] = {
        **pu_panel(cg_pos_v, cg_unl_v),
        "n_pos_dropped": int(np.isnan(cg_pos).sum()),
        "n_unl_dropped": int(np.isnan(cg_unl).sum()),
    }
    with open(out_path / "CGNF_pretrained.json", "w") as fh:
        json.dump(panels["CGNF_pretrained"], fh, indent=2)
    print("[cgnf] panel:", json.dumps(panels["CGNF_pretrained"]))

    # --- each A-PU config on the same test split ---
    yamls = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    for p in yamls:
        cfg = yaml.safe_load(open(p))
        if not cfg.get("deployable", False):
            continue
        if not set(cfg.get("features", [])) <= runnable:
            continue
        name = cfg.get("name", os.path.basename(p)[:-5])
        print(f"[apu] scoring {name}")
        pos, unl = score_apu_config(cfg, bank_path)
        panels[name] = pu_panel(pos, unl)
        with open(out_path / f"{name}.json", "w") as fh:
            json.dump(panels[name], fh, indent=2)

    with open(out_path / "_all_panels.json", "w") as fh:
        json.dump(panels, fh, indent=2)
    print(f"[done] wrote {len(panels)} panels -> {out_dir}")
    return panels


def main():
    ap = argparse.ArgumentParser(description="CGNF-matched head-to-head on the test split.")
    ap.add_argument("--configs-dir", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.configs_dir, args.bank, args.out)


if __name__ == "__main__":
    main()
