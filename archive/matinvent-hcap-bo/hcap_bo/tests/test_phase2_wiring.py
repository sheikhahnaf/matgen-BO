"""Unit tests for `phase2_open._make_pool_provider`.

Exercises three branches:
  1. `_target_` adapter block → instantiates a fake adapter and routes calls.
  2. `pool_path` set       → slices a pre-saved extxyz file.
  3. neither set           → synthetic-bulk fallback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write as ase_write
from omegaconf import OmegaConf

from src.phases.phase2_open import _make_pool_provider, _instantiate


class _FakeAdapter:
    """Generator stand-in. Records every sample() call so we can assert on it."""

    name = "fake_adapter"
    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeAdapter.calls = []

    def sample(self, n=64, chemical_system=None, property_conditions=None, seed=None):
        _FakeAdapter.calls.append({"n": n, "seed": seed})
        return [bulk("Si", "fcc", a=3.0 + 0.001 * i, cubic=True) for i in range(n)]


def test_instantiate_resolves_target_attribute():
    cfg = OmegaConf.create({
        "_target_": "tests.test_phase2_wiring._FakeAdapter",
        "checkpoint": "/dummy/path.ckpt",
        "device": "cpu",
    })
    obj = _instantiate(cfg)
    assert isinstance(obj, _FakeAdapter)
    assert obj.kwargs["checkpoint"] == "/dummy/path.ckpt"


def test_pool_provider_uses_adapter_when_target_present(tmp_path):
    cfg = OmegaConf.create({
        "experiment": {"seed": 7},
        "generation": {
            "adapter": {
                "_target_": "tests.test_phase2_wiring._FakeAdapter",
                "checkpoint": "/p.ckpt",
            },
            "batch_size": 4,
            "cycles": 3,
        },
    })
    provider, kind = _make_pool_provider(cfg, n_total=12, output_dir=tmp_path)
    assert kind == "adapter"

    out0 = provider(0, 4)
    out1 = provider(1, 4)
    assert len(out0) == 4 and len(out1) == 4
    # Per-cycle seed bumps
    assert _FakeAdapter.calls[0]["seed"] == 7
    assert _FakeAdapter.calls[1]["seed"] == 8
    # Each cycle's atoms saved to disk for reproducibility
    assert (tmp_path / "cycle000_samples.xyz").exists()
    assert (tmp_path / "cycle001_samples.xyz").exists()


def test_pool_provider_uses_pool_path_when_no_adapter(tmp_path):
    pool_atoms = [bulk("Cu", "fcc", a=3.6 + 0.01 * i, cubic=True) for i in range(8)]
    pool_path = tmp_path / "pool.xyz"
    ase_write(pool_path, pool_atoms, format="extxyz")

    cfg = OmegaConf.create({
        "experiment": {"seed": 0},
        "generation": {"pool_path": str(pool_path), "batch_size": 4},
    })
    provider, kind = _make_pool_provider(cfg, n_total=8, output_dir=tmp_path)
    assert kind == "pool_path"
    out0 = provider(0, 4)
    out1 = provider(1, 4)
    assert len(out0) == 4 and len(out1) == 4


def test_pool_provider_falls_back_to_synthetic(tmp_path):
    cfg = OmegaConf.create({
        "experiment": {"seed": 0},
        "generation": {"pool_path": None, "batch_size": 4},
    })
    provider, kind = _make_pool_provider(cfg, n_total=4, output_dir=tmp_path)
    assert kind == "synthetic"
    batch = provider(0, 4)
    assert len(batch) == 4
    assert all(hasattr(a, "get_chemical_formula") for a in batch)


def test_adapter_block_with_null_target_falls_through(tmp_path):
    """If `adapter` is present but missing `_target_`, fall through cleanly."""
    cfg = OmegaConf.create({
        "experiment": {"seed": 0},
        "generation": {
            "adapter": None,        # explicit null
            "pool_path": None,
            "batch_size": 4,
        },
    })
    provider, kind = _make_pool_provider(cfg, n_total=4, output_dir=tmp_path)
    assert kind == "synthetic"
