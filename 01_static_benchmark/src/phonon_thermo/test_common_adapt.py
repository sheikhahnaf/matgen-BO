"""
test_common_adapt.py — Tests for the THREE adaptations made to common.py for the
phonon thermodynamic benchmark:

1. load_matminer_dataset now reads a locally pickled DataFrame.
2. prepare_dataset gained a holdout_cap (arg + HOLDOUT_CAP env fallback).
3. featurizers memoize RAW (pre-PCA) embeddings to <FEAT_CACHE>/<descriptor>/.

Run with the matinvent conda env:
  eval "$(/Users/alvi/miniconda3/bin/conda shell.zsh hook)" && conda activate matinvent \
    && cd /Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark \
    && python -m pytest src/test_common_adapt.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import common  # noqa: E402
from common import (  # noqa: E402
    load_matminer_dataset,
    make_parser,
    prepare_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _trivial_structure(a: float = 3.0):
    """Single-atom cubic pymatgen Structure (cheap, always valid)."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(a)
    return Structure(lattice, ["Cu"], [[0.0, 0.0, 0.0]])


def _two_element_structure(elements, a: float = 3.5):
    """Two-site cubic pymatgen Structure with the given element pair."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(a)
    return Structure(lattice, list(elements),
                     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def _df_from_structures(structures) -> pd.DataFrame:
    """Build a featurizer-ready df (ase_atoms + material_id) from pmg Structures."""
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    return pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(len(structures))],
        "ase_atoms": [adaptor.get_atoms(s) for s in structures],
    })


def _make_df(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "material_id": [f"mp-{i}" for i in range(n_rows)],
        "structure": [_trivial_structure(3.0 + 0.01 * i) for i in range(n_rows)],
        "target": rng.normal(size=n_rows),
    })


# ---------------------------------------------------------------------------
# 1. Loader: pickle round-trip
# ---------------------------------------------------------------------------

def test_loader_pickle_roundtrip(tmp_path):
    df = pd.DataFrame({
        "material_id": ["mp-1", "mp-2", "mp-3"],
        "target": [1.0, 2.0, 3.0],
    })
    pkl = tmp_path / "tiny.pkl"
    df.to_pickle(pkl)

    loaded = load_matminer_dataset(str(pkl))

    assert list(loaded.columns) == ["material_id", "target"]
    assert len(loaded) == 3
    assert loaded["target"].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 2. Holdout cap
# ---------------------------------------------------------------------------

def test_holdout_cap_via_arg():
    """2000 rows, n_train=100 -> 1900 holdout, capped at 300."""
    df = _make_df(2000)
    df_train, df_holdout = prepare_dataset(df, n_train=100, seed=42, holdout_cap=300)
    assert len(df_train) == 100
    assert len(df_holdout) == 300


def test_holdout_cap_none_keeps_all():
    """No cap -> full 1900-row holdout."""
    df = _make_df(2000)
    df_train, df_holdout = prepare_dataset(df, n_train=100, seed=42, holdout_cap=None)
    assert len(df_train) == 100
    assert len(df_holdout) == 1900


def test_holdout_cap_via_env(monkeypatch):
    """HOLDOUT_CAP env var enables the cap when the arg is omitted (driver path)."""
    monkeypatch.setenv("HOLDOUT_CAP", "250")
    df = _make_df(2000)
    # Called positionally exactly like the unmodified drivers do.
    df_train, df_holdout = prepare_dataset(df, 100, 42)
    assert len(df_train) == 100
    assert len(df_holdout) == 250


def test_holdout_cap_larger_than_holdout_is_noop():
    """Cap above holdout size leaves the holdout untouched."""
    df = _make_df(200)
    df_train, df_holdout = prepare_dataset(df, n_train=100, seed=42, holdout_cap=500)
    assert len(df_train) == 100
    assert len(df_holdout) == 100  # only 100 remain after train split


def test_parser_has_holdout_cap():
    parser = make_parser("gp")
    args = parser.parse_args(["--holdout-cap", "300"])
    assert args.holdout_cap == 300
    # default is None
    args2 = parser.parse_args([])
    assert args2.holdout_cap is None


# ---------------------------------------------------------------------------
# 3. Feature cache (RAW embedding round-trip via the helpers)
# ---------------------------------------------------------------------------

def test_feature_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FEAT_CACHE", str(tmp_path))
    vec = np.arange(8, dtype=float)
    assert common._cache_load("soap", "mp-42") is None  # miss
    common._cache_save("soap", "mp-42", vec)
    assert os.path.exists(str(tmp_path / "soap" / "mp-42.npy"))
    got = common._cache_load("soap", "mp-42")
    assert got is not None
    np.testing.assert_array_equal(got, vec)


def test_feature_cache_disabled_without_material_id():
    """material_id=None must be a transparent no-op (caching skipped)."""
    assert common._cache_load("mace", None) is None
    common._cache_save("mace", None, np.ones(3))  # must not raise
    df = pd.DataFrame({"structure": [None]})
    assert common._get_material_ids(df) is None


def test_get_material_ids_present():
    df = _make_df(3)
    mids = common._get_material_ids(df)
    assert mids == ["mp-0", "mp-1", "mp-2"]


def test_soap_cache_namespaced_by_species_and_width_guarded(tmp_path, monkeypatch):
    """A SOAP vector cached under one species basis must NOT be served to a run
    with a different species set.

    Two defences are exercised:
      (1) Order-sensitive species signature => different cache namespace
          (different directory), so the run-B path never even looks at run-A's
          file.
      (2) Width guard => even a same-namespace stale vector of the wrong length
          is treated as a miss.
    """
    import hashlib
    monkeypatch.setenv("FEAT_CACHE", str(tmp_path))

    species_a = ["Cu", "O"]
    species_b = ["Cu", "O", "Si"]  # different/larger basis -> different width
    sig_a = hashlib.md5(",".join(species_a).encode()).hexdigest()[:8]
    sig_b = hashlib.md5(",".join(species_b).encode()).hexdigest()[:8]
    assert sig_a != sig_b, "different species sets must yield different signatures"

    desc_a = f"soap_{sig_a}"
    desc_b = f"soap_{sig_b}"

    # Run A caches a width-10 vector under namespace A.
    vec_a = np.arange(10, dtype=float)
    common._cache_save(desc_a, "mp-7", vec_a)

    # (1) Namespacing: run B (namespace B) gets a clean miss; A's file is untouched.
    assert common._cache_load(desc_b, "mp-7") is None
    assert os.path.exists(str(tmp_path / desc_a / "mp-7.npy"))
    assert not os.path.exists(str(tmp_path / desc_b / "mp-7.npy"))

    # (2) Width guard: even querying namespace A with a different expected width
    # (e.g. a basis change that happened to collide) is rejected as a miss.
    assert common._cache_load(desc_a, "mp-7", expected_len=12) is None
    # ...but the correct width is still served.
    got = common._cache_load(desc_a, "mp-7", expected_len=10)
    np.testing.assert_array_equal(got, vec_a)


def test_holdout_subsample_deterministic_same_seed():
    """Two same-seed prepare_dataset calls subsample the IDENTICAL holdout rows."""
    df = _make_df(2000)
    _, ho1 = prepare_dataset(df, n_train=100, seed=7, holdout_cap=300)
    _, ho2 = prepare_dataset(df, n_train=100, seed=7, holdout_cap=300)
    assert len(ho1) == len(ho2) == 300
    # Same material_ids, in the same order, both times.
    assert ho1["material_id"].tolist() == ho2["material_id"].tolist()


# ---------------------------------------------------------------------------
# 4. DATA-1: SOAP featurizer last_valid_mask_ (unknown-species exclusion)
# ---------------------------------------------------------------------------

def test_soap_last_valid_mask_unknown_species(tmp_path, monkeypatch):
    """SOAPFeaturizer.last_valid_mask_ must be False for a holdout structure whose
    element is absent from the fitted training-split species basis, and True for an
    in-basis structure.

    The fitted species set is fixed by the TRAINING split (Cu + O). A holdout row
    containing Si (unknown to that basis) falls back to a zero vector and so must be
    flagged invalid; an in-basis Cu/O holdout row must be flagged valid.
    """
    from common import SOAPFeaturizer

    # Isolate the feature cache so this test never reads/writes the shared dir.
    monkeypatch.setenv("FEAT_CACHE", str(tmp_path))

    # Training split fixes the species basis to {Cu, O}.
    train_df = _df_from_structures([
        _two_element_structure(["Cu", "O"], a=3.3),
        _two_element_structure(["Cu", "O"], a=3.5),
        _two_element_structure(["Cu", "O"], a=3.7),
        _two_element_structure(["Cu", "Cu"], a=3.5),
    ])

    # Holdout: row 0 in-basis (Cu/O -> valid), row 1 has Si (unknown -> invalid).
    holdout_df = _df_from_structures([
        _two_element_structure(["Cu", "O"], a=3.5),
        _two_element_structure(["Cu", "Si"], a=3.5),
    ])

    feat = SOAPFeaturizer(n_components=2)
    X_train = feat.fit_transform(train_df)
    assert feat.last_valid_mask_ is not None
    assert feat.last_valid_mask_.shape == (len(train_df),)
    # All training rows are in-basis by construction -> all valid.
    assert feat.last_valid_mask_.all()

    X_holdout = feat.transform(holdout_df)
    mask = feat.last_valid_mask_
    assert mask is not None
    assert mask.shape == (len(holdout_df),)
    assert bool(mask[0]) is True, "in-basis Cu/O holdout row should be valid"
    assert bool(mask[1]) is False, "unknown-species (Si) holdout row should be invalid"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
