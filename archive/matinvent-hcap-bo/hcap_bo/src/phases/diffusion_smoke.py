"""Smoke-test entry point for the 4 viable diffusion backends.

Runs `RemoteGeneratorAdapter.sample(n=4, seed=0)` for each model, reports
PASS/FAIL plus a brief signature (number of atoms, formulas) so we can spot
trivially-broken outputs (empty cells, all-same-element, NaN coords etc.)
before launching the full Phase-2 SLURM jobs.

Usage:
    python -m src.cli diffusion_smoke \\
        --config configs/hcap_bo.yaml \\
        --output-dir /scratch/.../results/diffusion_smoke_$JOBID
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from omegaconf import OmegaConf


_BACKENDS = ["adit", "crysbfn", "crystalflow"]
# Note: crystalformer requires JAX which has a cudnn conflict with torch in
# mat-zoo-modern (torch wants cudnn 8.9, JAX wants 9.21). Deferred until we
# stand up a separate `mat-zoo-jax` env for it.


def _load_adapter_block(model: str):
    repo = Path(__file__).resolve().parents[2]
    frag = repo / "configs" / "diffusion" / f"{model}.yaml"
    if not frag.exists():
        raise FileNotFoundError(f"missing fragment {frag}")
    cfg = OmegaConf.load(str(frag))
    return cfg.generation.adapter


def _instantiate(cfg_block):
    import importlib
    d = OmegaConf.to_container(cfg_block, resolve=True)
    target = d.pop("_target_")
    mod_path, _, name = target.rpartition(".")
    cls = getattr(importlib.import_module(mod_path), name)
    return cls(**d)


def _smoke_one(model: str, n: int = 4) -> dict:
    rec: dict = {"model": model, "ok": False}
    t0 = time.time()
    try:
        cfg = _load_adapter_block(model)
        adapter = _instantiate(cfg)
        atoms_list = adapter.sample(n=n, seed=0)
        rec.update({
            "ok": True,
            "n_returned": len(atoms_list),
            "formulas": [a.get_chemical_formula() for a in atoms_list[:n]],
            "natoms": [int(len(a)) for a in atoms_list[:n]],
            "elapsed_s": round(time.time() - t0, 1),
        })
    except Exception as e:
        rec.update({
            "error_type": type(e).__name__,
            "error_msg": str(e)[:500],
            "traceback": traceback.format_exc()[-2000:],
            "elapsed_s": round(time.time() - t0, 1),
        })
    return rec


def run(config_path: str, output_dir: str) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] testing {len(_BACKENDS)} backends → {out}")
    results = []
    for model in _BACKENDS:
        print(f"\n[smoke] === {model} ===")
        rec = _smoke_one(model, n=4)
        results.append(rec)
        with open(out / f"{model}.json", "w") as f:
            json.dump(rec, f, indent=2)
        if rec["ok"]:
            print(f"[smoke] {model:14s} PASS  "
                  f"n={rec['n_returned']}  formulas={rec['formulas']}  "
                  f"({rec['elapsed_s']}s)")
        else:
            print(f"[smoke] {model:14s} FAIL  "
                  f"{rec['error_type']}: {rec['error_msg']}")

    summary = {"results": results,
               "n_pass": sum(1 for r in results if r["ok"]),
               "n_fail": sum(1 for r in results if not r["ok"])}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[smoke] DONE — pass: {summary['n_pass']} / {len(results)}")
    return 0
