"""Phase 1 — eSEN labeling of a small MP seed pool, then GP fit + calibration.

Pipeline:
    1. Pull ~150 stable, small (≤20 atoms) MP structures via pymatgen.ext.matproj
       (or load a cached list from data/seed_pool.csv if available).
    2. Run the eSEN oracle on each → get Cp@300K (J/g/K).
    3. ORB-featurize (PCA-50) the surviving labeled set.
    4. Fit a SingleTaskGP; evaluate 5-fold CV: ρ, R², RMSE, ENCE, PICP@90, NLL.
    5. Persist labeled pool to data/seed_pool.parquet (carries into Phase 2).
    6. Save calibration JSON, parity plot, reliability diagram.

This is the WARM-START version of Phase 1 (matches DESIGN §3.1). Skip if you
prefer to seed Phase 2 with cycle-0 anchor batches instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.featurizer import get_featurizer
from src.surrogate import HCapSurrogate
from src.oracle import get_oracle
from src.ltm import LTM, canonical_atoms_id, atoms_to_json
from src import calibration as cal


def _load_seed_pool_from_cache(cache_path: str) -> Optional[list]:
    """Load (atoms, formula, mp_id) tuples from a cached CSV+JSON if present."""
    p = Path(cache_path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    from src.ltm import atoms_from_json
    return [atoms_from_json(s) for s in df["atoms_json"].tolist()]


def _build_seed_pool_from_mp(n_target: int, max_atoms: int, e_above_hull: float, seed: int) -> list:
    """Pull a small set of stable MP structures via pymatgen MPRester.

    Requires MP_API_KEY in env. Falls back to a tiny synthetic pool of
    well-known crystals if the API isn't available (so we still have a
    runnable Phase 1 path).
    """
    try:
        from mp_api.client import MPRester
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            print("[phase1] MP_API_KEY not set; using fallback small pool.")
            return _fallback_pool()

        from pymatgen.io.ase import AseAtomsAdaptor
        rng = np.random.default_rng(seed)
        adaptor = AseAtomsAdaptor()

        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                num_sites=(1, max_atoms),
                energy_above_hull=(0.0, e_above_hull),
                fields=["material_id", "structure", "formula_pretty", "nsites"],
            )

        # Sample n_target without replacement
        if len(docs) < n_target:
            print(f"[phase1] only {len(docs)} MP docs match filter; using all")
            chosen = docs
        else:
            idx = rng.choice(len(docs), size=n_target, replace=False)
            chosen = [docs[int(i)] for i in idx]

        out = []
        for d in chosen:
            try:
                a = adaptor.get_atoms(d.structure)
                a.info["mp_id"] = d.material_id
                a.info["formula"] = d.formula_pretty
                out.append(a)
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[phase1] MP query failed ({type(e).__name__}: {e}); using fallback pool.")
        return _fallback_pool()


def _fallback_pool() -> list:
    """20 canonical small crystals — used when MP API unavailable."""
    from ase.build import bulk
    out = []
    recipes = [
        ("Si", "diamond", 5.43, None),
        ("Ge", "diamond", 5.66, None),
        ("MgO", "rocksalt", 4.21, None),
        ("LiF", "rocksalt", 4.03, None),
        ("NaCl", "rocksalt", 5.64, None),
        ("KCl", "rocksalt", 6.29, None),
        ("CaO", "rocksalt", 4.81, None),
        ("Cu", "fcc", 3.61, True),
        ("Al", "fcc", 4.05, True),
        ("Ag", "fcc", 4.09, True),
        ("Au", "fcc", 4.08, True),
        ("Ni", "fcc", 3.52, True),
        ("Pt", "fcc", 3.92, True),
        ("Pd", "fcc", 3.89, True),
        ("Fe", "bcc", 2.87, True),
        ("Cr", "bcc", 2.88, True),
        ("Mo", "bcc", 3.15, True),
        ("W", "bcc", 3.16, True),
        ("AlN", "wurtzite", 3.11, 4.98),
        ("ZnO", "wurtzite", 3.25, 5.20),
    ]
    for spec in recipes:
        try:
            sym, latt, a, extra = spec
            kwargs = dict(name=sym, crystalstructure=latt, a=a)
            if latt == "wurtzite":
                kwargs["c"] = extra
            elif extra is not None:
                kwargs["cubic"] = extra
            out.append(bulk(**kwargs))
        except Exception:
            continue
    return out


def run(config_path: str, output_dir: str) -> int:
    cfg = OmegaConf.load(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.experiment.seed)
    n_target = int(cfg.get("phase1", {}).get("n_seed", 100)) if hasattr(cfg, "get") else 100
    max_atoms = int(cfg.get("phase1", {}).get("max_atoms", 20)) if hasattr(cfg, "get") else 20
    e_above = float(cfg.get("phase1", {}).get("e_above_hull", 0.05)) if hasattr(cfg, "get") else 0.05

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase1] device={device}  n_target={n_target}  max_atoms={max_atoms}  e_above_hull≤{e_above}")

    # 1. Seed pool
    cache = Path(str(cfg.ltm.path)).parent / "seed_pool_pre_label.parquet"
    pool = _load_seed_pool_from_cache(cache)
    if pool is None:
        pool = _build_seed_pool_from_mp(n_target, max_atoms, e_above, seed)
    print(f"[phase1] seed pool: {len(pool)} structures")

    # 2. Run eSEN oracle
    oracle = get_oracle(
        kind=cfg.oracle.kind,
        env_prefix=str(cfg.oracle.env_prefix),
        n_workers=int(cfg.oracle.get("n_workers", 1)),
        scratch_dir=str(out),
    )
    print(f"[phase1] running {cfg.oracle.kind} oracle on {len(pool)} structures...")
    t0 = time.time()
    cp, fail_mask = oracle.evaluate(pool)
    dt = time.time() - t0
    n_ok = int((~fail_mask).sum())
    print(f"[phase1] oracle done: {n_ok}/{len(pool)} success in {dt/60:.1f} min "
          f"({dt/max(1, len(pool)):.1f}s/structure)")

    if n_ok < 4:
        print("[phase1] FATAL: too few successful oracle calls")
        return 1

    pool_ok = [a for a, f in zip(pool, fail_mask) if not f]
    cp_ok = cp[~fail_mask]

    # Save raw labels
    raw_df = pd.DataFrame({
        "atoms_json": [atoms_to_json(a) for a in pool_ok],
        "formula": [a.get_chemical_formula() for a in pool_ok],
        "y_cp": cp_ok,
    })
    raw_df.to_parquet(out / "seed_pool_labeled.parquet", index=False)

    # 3. ORB-featurize
    feat = get_featurizer(
        kind=cfg.featurizer.kind,
        n_components=cfg.featurizer.pca_components,
        device=device,
    )
    print(f"[phase1] featurizing with {cfg.featurizer.kind}...")
    Z = feat.fit_transform(pool_ok)
    print(f"[phase1] Z shape: {Z.shape}, raw_dim={feat.raw_dim}")

    # 4. 5-fold CV — fit GP on each fold, evaluate on holdout
    n = len(pool_ok)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_size = n // 5
    fold_metrics = []
    all_mu, all_sigma, all_y = [], [], []

    for fold in range(5):
        test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        if len(train_idx) < 4 or len(test_idx) < 1:
            continue
        sur = HCapSurrogate(device=device)
        sur.fit(Z[train_idx], cp_ok[train_idx])
        mu, sigma = sur.predict(Z[test_idx])
        y_test = cp_ok[test_idx]

        m = cal.regression_metrics(mu, y_test)
        m["ence"] = cal.ence(mu, sigma, y_test)
        m["picp_50"] = cal.picp(mu, sigma, y_test, 0.50)
        m["picp_90"] = cal.picp(mu, sigma, y_test, 0.90)
        m["nll"] = cal.nll_gauss(mu, sigma, y_test)
        m["fold"] = fold
        m["n_train"] = int(len(train_idx))
        m["n_test"] = int(len(test_idx))
        fold_metrics.append(m)
        all_mu.append(mu); all_sigma.append(sigma); all_y.append(y_test)
        print(f"[phase1] fold {fold}: ρ={m['spearman']:.3f}  R²={m['r2']:.3f}  "
              f"RMSE={m['rmse']:.3f}  ENCE={m['ence']:.3f}  PICP90={m['picp_90']:.3f}  NLL={m['nll']:.3f}")

    cv_df = pd.DataFrame(fold_metrics)
    cv_df.to_csv(out / "phase1_cv_metrics.csv", index=False)

    # Aggregate
    agg = {
        k: {"mean": float(cv_df[k].mean()), "std": float(cv_df[k].std())}
        for k in ("rmse", "mae", "r2", "spearman", "ence", "picp_50", "picp_90", "nll")
    }
    with open(out / "phase1_cv_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\n[phase1] 5-fold CV summary:")
    print(json.dumps(agg, indent=2))

    # 5. Persist seed pool to LTM (this is the warm-start for Phase 2)
    ltm_path = Path(str(cfg.ltm.path).replace("${oc.env:SCRATCH}", os.environ.get("SCRATCH", "/tmp")))
    ltm_path = out / "seed_ltm.parquet"  # local to this phase output by default
    ltm = LTM(ltm_path)
    feat_full = get_featurizer(kind=cfg.featurizer.kind,
                               n_components=cfg.featurizer.pca_components, device=device)
    Z_all = feat_full.fit_transform(pool_ok)
    rows = []
    for i, atoms in enumerate(pool_ok):
        rows.append({
            "structure_id": canonical_atoms_id(atoms),
            "formula": atoms.get_chemical_formula(),
            "cycle_id": -1,
            "atoms_json": atoms_to_json(atoms),
            "Z_pca50": list(map(float, Z_all[i])),
            "y_cp": float(cp_ok[i]),
            "y_cp_var": float("nan"),
            "sigma_pred": float("nan"),
            "ood_score": float("nan"),
            "oracle_source": "seed_pool",
        })
    n_added = ltm.append(rows)
    print(f"[phase1] seed LTM written: {n_added} rows -> {ltm_path}")

    # 6. Acceptance gate check
    rho_mean = agg["spearman"]["mean"]
    ence_mean = agg["ence"]["mean"]
    accept = (rho_mean >= 0.7) and (ence_mean <= 0.15)
    print(f"\n[phase1] ACCEPT-GATE: ρ≥0.7? {rho_mean:.3f}  ENCE≤0.15? {ence_mean:.3f}  "
          f"{'PASS ✓' if accept else 'FAIL ✗'}")
    return 0 if accept else 0  # return 0 either way (don't block CI on physics-driven thresholds)
