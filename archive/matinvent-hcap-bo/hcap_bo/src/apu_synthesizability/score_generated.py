"""Score the generated diffusion structures with the tuned A-PU model + CGNF.

The fair head-to-head: both scorers are out-of-distribution on these generated
structures (unlike the in-distribution MP test split). We report, per structure
and per (backbone, policy):

  * A-PU synthesizability probability (tuned orb_mag model) and its abstain/OOD
    decision (the OOD layer is exactly what flags generated structures that fall
    outside the MP training manifold),
  * pretrained CGNF synthesizability score,
  * concordance (Spearman + >0.5 agreement), overall and on non-abstained.

Features are built to MATCH the bank: raw 256-d ORB embedding (single ORB pass) →
the bank's saved PCA (bank.npz.pca.pkl) → orb_pca(50), concatenated with Magpie,
in the same order as FEATURE_SETS['orb_mag'] = ['orb_pca','magpie'].

Writes only to the given --out-csv (+ a _summary.json sibling).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .score_structures import load_structures, concordance
from .cgnf_compare import score_cgnf  # formula-based, element-coverage guarded
from .features import orb_features, magpie_features


def build_features_orb_mag(structures, pca, device="cpu"):
    """raw 256-d ORB -> bank PCA -> orb_pca(50), concat Magpie(132)."""
    raw, _, _ = orb_features(structures, n_pca=None, device=device)   # (N,256)
    orb_pca = pca.transform(raw)                                       # (N,50)
    formulas = [s.composition.reduced_formula for s in structures]
    magpie = magpie_features(formulas)                                # (N,132)
    return np.concatenate([orb_pca, magpie], axis=1).astype(np.float32), formulas


def run(cif_dir, model_path, pca_path, out_csv, syn_score_parent=None, device="cpu"):
    import joblib
    import pandas as pd

    items = load_structures(cif_dir)
    tags, structs = zip(*items)
    structs = list(structs)
    print(f"loaded {len(structs)} structures")

    with open(pca_path, "rb") as fh:
        pca = pickle.load(fh)
    X, formulas = build_features_orb_mag(structs, pca, device=device)

    model = joblib.load(model_path)               # AbstainingPUClassifier
    apu_p = np.asarray(model.predict_proba(X), float)
    dec = model.predict(X)

    cgnf = score_cgnf(formulas)                    # NaN if elements not in CGNF embedding

    # concordance (drop NaN CGNF); overall and on non-abstained
    abst = dec["abstain"].astype(bool)
    valid = ~np.isnan(cgnf)
    conc_all = concordance(apu_p[valid], cgnf[valid]) if valid.sum() > 2 else None
    keep = valid & ~abst
    conc_keep = concordance(apu_p[keep], cgnf[keep]) if keep.sum() > 2 else None

    rows = []
    for i, t in enumerate(tags):
        rows.append({
            "file_target": t.target, "rank": t.rank,
            "backbone": t.backbone, "backbone_name": t.backbone_name,
            "policy": t.policy, "seed": t.seed,
            "formula": t.formula, "spacegroup": t.spacegroup,
            "apu_score": float(apu_p[i]),
            "apu_decision": int(dec["predictions"][i]),     # 1 / 0 / -1 (abstain)
            "abstain": bool(abst[i]),
            "abstain_ood": bool(dec["abstain_ood"][i]),
            "abstain_conf": bool(dec["abstain_confidence"][i]),
            "abstain_disagree": bool(dec["abstain_disagreement"][i]),
            "ood_score": float(dec["ood_scores"][i]),
            "disagreement": float(dec["disagreement"][i]),
            "cgnf_score": float(cgnf[i]) if valid[i] else float("nan"),
        })
    df = pd.DataFrame(rows)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    panel = (df.groupby(["backbone_name", "policy"])
               .agg(n=("apu_score", "size"),
                    apu_mean=("apu_score", "mean"),
                    cgnf_mean=("cgnf_score", "mean"),
                    abstain_rate=("abstain", "mean"),
                    ood_rate=("abstain_ood", "mean"))
               .round(3))

    summary = {
        "n": int(len(df)),
        "n_cgnf_uncovered": int((~valid).sum()),
        "apu_mean": float(np.nanmean(apu_p)),
        "cgnf_mean": float(np.nanmean(cgnf)),
        "abstain_rate": float(abst.mean()),
        "ood_rate": float(dec["abstain_ood"].astype(bool).mean()),
        "concordance_all": conc_all,
        "concordance_nonabstained": conc_keep,
    }
    with open(str(out).replace(".csv", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== per (backbone, policy) ===")
    print(panel.to_string())
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return df, panel, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif-dir", required=True)
    ap.add_argument("--apu-model", required=True)
    ap.add_argument("--pca", required=True)
    ap.add_argument("--syn-score-parent", default=None)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    run(a.cif_dir, a.apu_model, a.pca, a.out_csv, a.syn_score_parent, a.device)


if __name__ == "__main__":
    main()
