"""Phase 0+: side-by-side ORB vs UMA featurizer sanity test.

Featurizes the same 12 toy crystals with both ORB and UMA, prints raw dims +
runtimes + sample feature norms. No GP, no oracle — just verifies both
featurizer pipelines work in this env on this device.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

from src.featurizer import get_featurizer


def run(config_path: str, output_dir: str) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[feat-compare] device={device}")

    rng = np.random.default_rng(0)
    elements = ["Si", "Al", "Mg", "Cu", "Fe", "Ti"]
    atoms = []
    for _ in range(12):
        e = rng.choice(elements)
        a = float(rng.uniform(2.8, 4.5))
        try:
            atoms.append(bulk(e, "fcc", a=a, cubic=True))
        except Exception:
            atoms.append(bulk(e, "bcc", a=a, cubic=True))

    results = {}
    for kind in ("orb", "uma"):
        try:
            print(f"\n[feat-compare] {kind.upper()} loading...")
            f = get_featurizer(kind=kind, n_components=10, device=device)
            t0 = time.time()
            Z = f.fit_transform(atoms)
            dt = time.time() - t0
            results[kind] = {
                "raw_dim": int(f.raw_dim),
                "shape": tuple(Z.shape),
                "elapsed_s": dt,
                "feat_norm_mean": float(np.linalg.norm(Z, axis=1).mean()),
            }
            print(f"  {kind}: raw_dim={f.raw_dim}, Z.shape={Z.shape}, {dt:.1f}s, |Z|={np.linalg.norm(Z, axis=1).mean():.3f}")
        except Exception as e:
            print(f"  {kind}: FAILED — {type(e).__name__}: {e}")
            results[kind] = {"error": f"{type(e).__name__}: {e}"}

    import json
    with open(out / "feat_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[feat-compare] saved to {out / 'feat_compare.json'}")
    return 0
