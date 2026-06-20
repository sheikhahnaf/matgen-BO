"""Tests for the CGNF-matched PU metric panel (pu_panel).

The panel applies one identical estimator to CGNF and every A-PU config on the
same held-out test split.  We test the estimator's *properties* (it is the
fairness contract), not magic numbers:

* perfect ranking  -> AUROC == AUPRC == 1
* random model     -> AUROC ~ 0.5 and estimated precision ~ prior (beta)
* informative model-> estimated precision > prior (better than guessing)
* prior_beta is a valid probability and the prior threshold matches it
"""
import numpy as np

from apu_synthesizability.cgnf_compare import pu_panel


def _clip01(a):
    return np.clip(a, 1e-4, 1 - 1e-4)


def test_panel_keys_present():
    rng = np.random.default_rng(0)
    p = pu_panel(_clip01(rng.normal(0.7, 0.1, 200)), _clip01(rng.normal(0.3, 0.1, 600)))
    for k in ["auroc", "auprc", "ece", "prior_beta", "label_freq_c",
              "tpr_at0.5", "precision_estimated_at0.5", "precision_pessimistic_at0.5",
              "tpr_atprior", "precision_estimated_atprior"]:
        assert k in p, f"missing key {k}"


def test_perfect_ranking_auroc_auprc_one():
    # every positive scores strictly above every unlabeled
    pos = np.full(100, 0.9)
    unl = np.full(300, 0.1)
    p = pu_panel(pos, unl)
    assert abs(p["auroc"] - 1.0) < 1e-9
    assert abs(p["auprc"] - 1.0) < 1e-9
    assert p["tpr_at0.5"] == 1.0          # all positives recovered at 0.5


def test_random_model_precision_near_prior():
    # pos and unl from the SAME distribution -> uninformative
    rng = np.random.default_rng(1)
    pos = _clip01(rng.normal(0.5, 0.15, 4000))
    unl = _clip01(rng.normal(0.5, 0.15, 4000))
    p = pu_panel(pos, unl)
    assert abs(p["auroc"] - 0.5) < 0.05, p["auroc"]
    # an uninformative model's precision should collapse toward the prior
    assert abs(p["precision_estimated_at0.5"] - p["prior_beta"]) < 0.08, p


def test_informative_model_beats_prior():
    # unlabeled is a mixture: 30% look like positives (hidden positives), 70% clearly negative
    rng = np.random.default_rng(2)
    pos = _clip01(rng.normal(0.75, 0.1, 3000))
    n_hidden = 900
    unl = _clip01(np.concatenate([
        rng.normal(0.75, 0.1, n_hidden),       # hidden positives
        rng.normal(0.2, 0.1, 2100),            # true negatives
    ]))
    p = pu_panel(pos, unl)
    assert 0.0 < p["prior_beta"] < 1.0, p["prior_beta"]
    # informative ranking -> estimated precision strictly exceeds the prior
    assert p["precision_estimated_at0.5"] > p["prior_beta"] + 0.05, p
    # AUROC is inherently capped (~0.85 here) because ~30% of the unlabeled pool are
    # hidden positives that the ranker correctly scores high — so it is not a bug that
    # AUROC < 0.9; the estimated-precision-beats-prior property above is the real check.
    assert p["auroc"] > 0.8, p["auroc"]


def test_prior_threshold_matches_prior_rate():
    # at the prior-adjusted threshold, the predicted-positive rate on unlabeled
    # should be approximately beta (that is how the threshold is defined)
    rng = np.random.default_rng(3)
    pos = _clip01(rng.normal(0.7, 0.12, 2000))
    unl = _clip01(rng.normal(0.35, 0.18, 4000))
    p = pu_panel(pos, unl)
    assert abs(p["predpos_rate_unl_atprior"] - p["prior_beta"]) < 0.03, p
