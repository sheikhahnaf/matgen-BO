#!/usr/bin/env python
"""Example 2: what the acquisition gate does with one batch of candidates.

Takes one real cycle from a published run: the candidates that passed the SUN
filter, and what the heat-capacity oracle later returned for each of them.
Fits the gate's GP on the labeled seed pool, scores the candidates, and applies
the gate's selection rule. Runs on a laptop CPU in a few seconds, on data that
ships with the repository.

    python examples/02_gate_selection.py
    python examples/02_gate_selection.py --run hcap_p3v4_cf_baseline_seed7_3006075 --cycle 3
    python examples/02_gate_selection.py --diversity-weight 0      # top four by Expected Improvement

The selection rule (02_closed_loop_bo/src/calculators/local_esen_gp_routed_v4.py):

    first pick   the candidate with the largest Expected Improvement (EI)
    next picks   the candidate with the largest  EI - w * s,
                 where s is its largest cosine similarity (ORB PCA-50 features)
                 to the candidates already picked, and w is the diversity weight

EI is in the units of the property (J/g/K here). When no candidate is predicted
to come close to the best value in the training set, EI is many orders of
magnitude smaller than w * s, and every pick after the first is decided by the
similarity term. The printed EI column shows which regime a batch is in.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "04_revision_round1" / "candidate_table" / "out" / "published"
sys.path.insert(0, str(REPO / "02_closed_loop_bo"))

from src.calculators.local_esen_gp_routed_v4 import _expected_improvement, _greedy_dpp_topk  # noqa: E402
from src.surrogate import HCapSurrogate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hcap_p3v4_mg_baseline_seed7_3006073")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--k", type=int, default=4, help="oracle calls per cycle")
    ap.add_argument("--diversity-weight", type=float, default=0.5)
    args = ap.parse_args()

    # Labeled seed pool of the heat-capacity gate, and the ORB embeddings of its structures.
    seed = pd.read_parquet(DATA / "seed_pool_cp.parquet")
    X_seed = np.load(DATA / "orb_raw_seed_cp.npy")

    # Candidates of one cycle. Runs without the gate sent every candidate to the oracle,
    # so the outcome of each one is known.
    table = pd.read_parquet(DATA / "candidates_cp.parquet")
    X_all = np.load(DATA / "orb_raw_cp.npy")
    pool = table[(table["run"] == args.run) & (table["step"] == args.cycle)]
    if pool.empty:
        raise SystemExit(f"no candidates for run {args.run}, cycle {args.cycle}")
    X_pool = X_all[pool.index.to_numpy()]

    # The gate's surrogate: PCA-50 of the embedding, then a GP with a small fixed noise.
    pca = PCA(n_components=50, random_state=0).fit(X_seed)
    gp = HCapSurrogate(device="cpu")
    y_seed = seed["y"].to_numpy(dtype=float)
    gp.fit(pca.transform(X_seed), y_seed, y_var=np.full(len(y_seed), 1e-6 / 0.9))

    Z = pca.transform(X_pool)
    mu, sigma = gp.predict(Z)
    best = float(y_seed.max())                         # incumbent: best value in the training set
    ei = _expected_improvement(mu, sigma, best)
    picks = [int(i) for i in _greedy_dpp_topk(ei, Z, args.k, lambda_div=args.diversity_weight)]

    out = pd.DataFrame({
        "formula": pool["formula"].to_numpy(),
        "GP mean": mu.round(3),
        "GP sd": sigma.round(3),
        "sd below best": ((best - mu) / sigma).round(1),
        "EI": [f"{v:.1e}" for v in ei],
        "pick": ["" if i not in picks else f"#{picks.index(i) + 1}" for i in range(len(pool))],
        "oracle": [f"{v:.3f}" if ok else "no value" for v, ok in zip(pool["y"], pool["success"])],
    })
    print(f"run {args.run}, cycle {args.cycle}: {len(pool)} candidates, incumbent {best:.3f} J/g/K, "
          f"diversity weight {args.diversity_weight}")
    print(out.to_string(index=False))
    print("\noracle values are in J/g/K and are clipped to the reward window 0.25-2.0 for runs without the gate")


if __name__ == "__main__":
    main()
