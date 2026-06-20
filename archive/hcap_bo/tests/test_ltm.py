"""Unit tests for src/ltm.py — append-only parquet store + dedup."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from src.ltm import LTM, canonical_atoms_id, atoms_to_json, atoms_from_json


def _row(sid: str, cycle: int, cp: float):
    return {
        "structure_id": sid,
        "formula": "Si2",
        "cycle_id": cycle,
        "atoms_json": "{}",
        "Z_pca50": [0.0] * 50,
        "y_cp": cp,
        "y_cp_var": float("nan"),
        "sigma_pred": 0.1,
        "ood_score": float("nan"),
        "oracle_source": "oracle",
    }


def test_canonical_id_stable():
    a = bulk("Si", "diamond", a=5.43)
    b = bulk("Si", "diamond", a=5.43)
    assert canonical_atoms_id(a) == canonical_atoms_id(b)


def test_canonical_id_different_for_different_lattice():
    a = bulk("Si", "diamond", a=5.43)
    b = bulk("Si", "diamond", a=5.50)
    assert canonical_atoms_id(a) != canonical_atoms_id(b)


def test_atoms_roundtrip_json():
    a = bulk("Si", "diamond", a=5.43)
    s = atoms_to_json(a)
    a2 = atoms_from_json(s)
    np.testing.assert_array_almost_equal(a.get_positions(), a2.get_positions())
    np.testing.assert_array_almost_equal(a.get_cell().array, a2.get_cell().array)


def test_append_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        ltm = LTM(Path(tmp) / "ltm.parquet")
        added = ltm.append([_row("a", 0, 1.0), _row("b", 0, 2.0), _row("a", 1, 99.0)])
        assert added == 2  # third is dup
        assert ltm.size() == 2
        df = ltm.load()
        # The first entry for "a" survives (cp=1.0), not the second (99.0)
        assert df.set_index("structure_id").loc["a", "y_cp"] == 1.0


def test_persistence_across_reload():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ltm.parquet"
        ltm = LTM(path)
        ltm.append([_row("x", 0, 0.5), _row("y", 0, 1.5)])
        ltm2 = LTM(path)  # reopen
        assert ltm2.size() == 2
        # And the dedup set is rehydrated
        assert ltm2.has("x") and ltm2.has("y")


def test_features_and_targets_returns_aligned_arrays():
    with tempfile.TemporaryDirectory() as tmp:
        ltm = LTM(Path(tmp) / "ltm.parquet")
        ltm.append([_row("a", 0, 1.0), _row("b", 0, 2.0), _row("c", 0, 3.0)])
        Z, y = ltm.features_and_targets()
        assert Z.shape == (3, 50)
        assert y.shape == (3,)
        np.testing.assert_array_equal(np.sort(y), [1.0, 2.0, 3.0])


def test_y_cp_required():
    with tempfile.TemporaryDirectory() as tmp:
        ltm = LTM(Path(tmp) / "ltm.parquet")
        bad = {"structure_id": "z", "formula": "X", "cycle_id": 0,
               "atoms_json": "{}", "Z_pca50": [0.0] * 50}
        with pytest.raises(AssertionError):
            ltm.append([bad])
