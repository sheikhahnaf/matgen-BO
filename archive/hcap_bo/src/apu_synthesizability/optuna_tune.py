"""Optuna-tuned A-PU synthesizability scorer (one feature_set x base_model study).

Faithful to the notebook's tuning *methodology* (TPESampler seed 42, MedianPruner,
StratifiedKFold inner CV, the same XGBoost search space), with two deliberate,
documented departures using our own judgement:

  1. Objective = AUPRC on the held-out PU split (positives vs unlabeled), NOT the
     notebook's e_hull-based proxy_f1.  e_hull measures thermodynamic stability, not
     synthesizability, and is partly circular here (ORB-energy stability is already a
     feature).  AUPRC is the prior-free ranking metric the PU task actually targets.

  2. Abstention / OOD / decision thresholds are set POST-HOC, not Optuna-tuned.  AUPRC
     is invariant to the abstain decision (it ranks scores), so a ranking objective
     cannot tune abstention thresholds.  We tune the scorer for ranking, then attach
     the abstention/OOD policy (notebook defaults; OOD threshold is intrinsically
     calibrated to the 95th percentile of positive-training distances) and REPORT the
     resulting coverage.  Tuning abstention would require a selective objective + a
     trustworthy proxy, which we deliberately avoided.

Writes only to the given --out dir.  Never touches results/apu or bank.npz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from .train_config import _assemble_X, _load_bank
from .abstain import AbstainingPUClassifier
from .cgnf_compare import pu_panel

SEED = 42
N_FOLDS = 5

# feature_set name -> bank blocks
FEATURE_SETS = {
    "mag": ["magpie"],
    "orb": ["orb_pca"],
    "orb_mag": ["orb_pca", "magpie"],
    "orb_mag_stab": ["orb_pca", "magpie", "stability"],
}


def _suggest(trial, base):
    """Search space (notebook XGBoost space + a compact RF space)."""
    common = {
        "n_bags": trial.suggest_int("n_bags", 5, 20),
        "neg_sample_ratio": trial.suggest_float("neg_sample_ratio", 0.5, 2.0),
    }
    if base == "xgboost":
        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        }
    elif base == "rf":
        hp = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    else:
        raise ValueError(f"unsupported base {base}")
    return common, hp


def _fit_scorer(X, s, base, n_bags, neg_ratio, hp, seed=SEED, n_jobs=-1):
    return AbstainingPUClassifier(
        base=base, n_bags=n_bags, neg_sample_ratio=neg_ratio, seed=seed,
        n_jobs=n_jobs, **hp,
    ).fit(X, s)


def make_objective(X_tv, s_tv, base):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(len(s_tv)), s_tv))

    def objective(trial):
        common, hp = _suggest(trial, base)
        aps = []
        for k, (tr, va) in enumerate(folds):
            model = _fit_scorer(X_tv[tr], s_tv[tr], base,
                                common["n_bags"], common["neg_sample_ratio"], hp)
            p = model.predict_proba(X_tv[va])
            ap = average_precision_score(s_tv[va], p) if s_tv[va].max() > 0 else 0.0
            aps.append(float(ap))
            trial.report(float(np.mean(aps)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(aps))

    return objective


def run_study(feature_set, base, bank_path, out_dir,
              n_trials=40, timeout=6000) -> dict:
    out = Path(out_dir) / f"{feature_set}__{base}"
    out.mkdir(parents=True, exist_ok=True)

    bank = _load_bank(bank_path)
    label = bank["label"].astype(int)
    split = bank["split"].astype(str)
    X = _assemble_X(bank, FEATURE_SETS[feature_set]).astype(np.float32)

    tv = np.isin(split, ["train", "val"])
    te = split == "test"
    X_tv, s_tv = X[tv], label[tv]
    X_te, s_te = X[te], label[te]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        study_name=f"apu_{feature_set}_{base}",
    )
    study.optimize(make_objective(X_tv, s_tv, base),
                   n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    best = dict(study.best_params)
    with open(out / "best_params.json", "w") as fh:
        json.dump({"best_value_auprc_cv": float(study.best_value),
                   "best_params": best, "n_trials": len(study.trials)}, fh, indent=2)

    # --- refit best on full train+val, attach post-hoc abstention policy ---
    n_bags = best.pop("n_bags")
    neg_ratio = best.pop("neg_sample_ratio")
    final = AbstainingPUClassifier(
        base=base, n_bags=n_bags, neg_sample_ratio=neg_ratio, seed=SEED,
        confidence_threshold=0.15, disagreement_threshold=0.25, ood_threshold=1.0,
        decision_threshold=0.5, n_jobs=-1, **best,
    ).fit(X_tv, s_tv)

    # --- evaluate on the held-out test split (same panel as cgnf_compare) ---
    pos_scores = final.predict_proba(X_te[s_te == 1])
    unl_scores = final.predict_proba(X_te[s_te == 0])
    panel = pu_panel(pos_scores, unl_scores)

    dec = final.predict(X_te)
    abst = dec["abstain"].astype(bool)
    panel["coverage"] = float(1.0 - abst.mean())
    panel["abstain_rate"] = float(abst.mean())
    panel["abstain_rate_pos"] = float(abst[s_te == 1].mean())
    panel["abstain_breakdown"] = {
        "confidence": float(dec["abstain_confidence"].mean()),
        "disagreement": float(dec["abstain_disagreement"].mean()),
        "ood": float(dec["abstain_ood"].mean()),
    }
    panel["feature_set"] = feature_set
    panel["base"] = base
    panel["best_params"] = study.best_params
    with open(out / "test_panel.json", "w") as fh:
        json.dump(panel, fh, indent=2)

    import joblib
    joblib.dump(final, out / "model.joblib")

    print(f"[{feature_set}__{base}] cv_auprc={study.best_value:.4f} "
          f"test_auprc={panel['auprc']:.4f} test_auroc={panel['auroc']:.4f} "
          f"ece={panel['ece']:.4f} coverage={panel['coverage']:.3f}")
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--base", required=True, choices=["xgboost", "rf"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=6000)
    args = ap.parse_args()
    run_study(args.feature_set, args.base, args.bank, args.out,
              n_trials=args.n_trials, timeout=args.timeout)


if __name__ == "__main__":
    main()
