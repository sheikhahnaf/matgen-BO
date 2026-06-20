"""Phase 0 smoke test: import everything, featurize 3 random crystals,
fit a tiny GP, run acquisition, exercise calibration metrics.

Does NOT call the FairChem oracle (no GPU/heavy compute required).
Useful as a CI-style integration probe before launching full SLURM jobs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from omegaconf import OmegaConf

from src.featurizer import ORBFeaturizer
from src.surrogate import HCapSurrogate
from src.acquisition import select_topk
from src import calibration as cal


def _toy_atoms_pool(n: int):
    rng = np.random.default_rng(0)
    elements = ["Si", "Al", "Mg", "Cu", "Fe", "Ti", "Zn", "Na"]
    structs = []
    for _ in range(n):
        e = rng.choice(elements)
        a = float(rng.uniform(2.8, 4.5))
        try:
            structs.append(bulk(e, "fcc", a=a, cubic=True))
        except Exception:
            structs.append(bulk(e, "bcc", a=a, cubic=True))
    return structs


def run(config_path: str, output_dir: str) -> int:
    cfg = OmegaConf.load(config_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device}")

    pool = _toy_atoms_pool(80)
    n_train = 60
    n_query = len(pool) - n_train
    print(f"[smoke] built {len(pool)} toy bulk crystals (train={n_train}, query={n_query})")

    feat = ORBFeaturizer(n_components=cfg.featurizer.pca_components, device=device)
    t0 = time.time()
    Z = feat.fit_transform(pool[:n_train])
    print(f"[smoke] featurized {n_train} in {time.time()-t0:.1f}s, Z={Z.shape}, raw_dim={feat.raw_dim}")

    # Fit a toy surrogate on synthetic targets so calibration code exercises.
    rng = np.random.default_rng(0)
    y = (Z[:, 0] * 0.7 + Z[:, 1] * 0.3) + rng.normal(0, 0.05, size=len(Z))
    sur = HCapSurrogate(device=device)
    sur.fit(Z, y)
    print(f"[smoke] surrogate fitted on {len(y)} points")

    Z_query = feat.transform(pool[n_train:])
    mu, sigma = sur.predict(Z_query)
    print(f"[smoke] predicted on {len(Z_query)}: mu mean={mu.mean():.3f}  sigma mean={sigma.mean():.3f}")

    idx = select_topk(sur, Z_query, k=min(8, len(Z_query)), diversity="kdpp", seed=0)
    print(f"[smoke] qLogNEI top-{len(idx)} (with k-DPP) = {idx.tolist()}")

    # Calibration on a synthetic held-out (re-use mu/sigma; create fake y_true)
    y_true = mu + rng.normal(0, sigma)
    print(
        f"[smoke] ENCE={cal.ence(mu, sigma, y_true):.3f}  "
        f"PICP90={cal.picp(mu, sigma, y_true, level=0.90):.2f}  "
        f"NLL={cal.nll_gauss(mu, sigma, y_true):.3f}"
    )

    feat.save(out / "featurizer.pkl")
    sur.save(out / "surrogate.pkl")
    print(f"[smoke] saved featurizer + surrogate to {out}")

    return 0
