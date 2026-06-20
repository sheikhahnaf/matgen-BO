"""Tests for apu_synthesizability.features — feature bank.

Matminer-dependent tests are skipped when matminer is not installed.
ORB/eSEN/fairchem tests are skipped when those packages are absent.
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

# Skip the whole matminer block if matminer isn't importable
matminer = pytest.importorskip("matminer")  # skip if not installed locally

from apu_synthesizability.features import magpie_features  # noqa: E402


# Module-level picklable stand-in for a PCA object
class _FakePCA:
    """Picklable stand-in for a fitted sklearn PCA used in tests."""
    pass


# ---------------------------------------------------------------------------
# magpie_features
# ---------------------------------------------------------------------------

def test_magpie_features_shape():
    """Magpie ElementProperty preset → 132-d per formula."""
    X = magpie_features(["Fe2O3", "LiMg"])
    assert X.shape[0] == 2 and X.shape[1] > 50, (
        f"Expected (2, >50), got {X.shape}"
    )


def test_magpie_features_exact_dim():
    """Verify the known Magpie dimension is exactly 132."""
    X = magpie_features(["Fe2O3"])
    assert X.shape == (1, 132), f"Expected (1, 132), got {X.shape}"


def test_magpie_features_finite():
    """All values should be finite for valid formulas."""
    X = magpie_features(["Al2O3", "NaCl", "TiO2"])
    assert X.shape[0] == 3
    assert np.all(np.isfinite(X)), "Expected all-finite Magpie features for common oxides."


def test_magpie_features_different_formulas_differ():
    """Different compositions produce different feature vectors."""
    X = magpie_features(["Fe", "Cu"])
    assert not np.allclose(X[0], X[1]), "Fe and Cu should have different Magpie features."


def test_magpie_features_reproducible():
    """Same formula called twice gives identical features."""
    X1 = magpie_features(["SiO2"])
    X2 = magpie_features(["SiO2"])
    np.testing.assert_array_equal(X1, X2)


# ---------------------------------------------------------------------------
# build_feature_bank (Magpie only, mocking ORB)
# ---------------------------------------------------------------------------

def test_build_feature_bank_no_overwrite(tmp_path):
    """build_feature_bank raises FileExistsError if the .npz already exists."""
    import pandas as pd
    from unittest.mock import patch
    from pymatgen.core import Structure, Lattice

    from apu_synthesizability.features import build_feature_bank

    out = tmp_path / "bank.npz"
    out.touch()  # pre-create to trigger the guard

    manifest = pd.DataFrame({
        "material_id": ["mp-1"],
        "formula": ["Fe2O3"],
        "label": [1],
        "split": ["train"],
    })
    struct = Structure(Lattice.cubic(4.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    with pytest.raises(FileExistsError, match="already exists"):
        # We need to mock orb_features so it doesn't actually load ORB
        with patch("apu_synthesizability.features.orb_features",
                   return_value=(np.zeros((1, 50)), None, np.zeros((1, 1)))):
            build_feature_bank(manifest, [struct], str(out))


def test_build_feature_bank_with_stability(tmp_path):
    """with_stability=True adds an ORB-energy 'stability' block (orb_stability mocked)."""
    import pandas as pd
    from unittest.mock import patch
    from pymatgen.core import Structure, Lattice
    from apu_synthesizability.features import build_feature_bank

    out = tmp_path / "bank.npz"
    manifest = pd.DataFrame({
        "material_id": ["mp-1", "mp-2"],
        "formula": ["Fe2O3", "Al2O3"],
        "label": [1, 0],
        "split": ["train", "test"],
    })
    structs = [
        Structure(Lattice.cubic(4.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.2), ["Al", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]
    fake_pca_feats = np.zeros((2, 50), dtype=np.float32)
    fake_energy = np.array([[-5.0], [-4.0]])

    with patch("apu_synthesizability.features.orb_features",
               return_value=(fake_pca_feats, _FakePCA(), fake_energy)):
        build_feature_bank(manifest, structs, str(out), n_pca=50, with_stability=True)

    bank = np.load(out, allow_pickle=True)
    assert "stability" in bank, "stability block missing when with_stability=True"
    assert bank["stability"].shape == (2, 1)
    assert bank["stability"].tolist() == [[-5.0], [-4.0]]


def test_build_feature_bank_saves_npz(tmp_path):
    """build_feature_bank saves an .npz with the expected named blocks."""
    import pandas as pd
    from unittest.mock import patch
    from pymatgen.core import Structure, Lattice

    from apu_synthesizability.features import build_feature_bank

    out = tmp_path / "bank.npz"

    manifest = pd.DataFrame({
        "material_id": ["mp-1", "mp-2"],
        "formula": ["Fe2O3", "Al2O3"],
        "label": [1, 0],
        "split": ["train", "test"],
    })
    structs = [
        Structure(Lattice.cubic(4.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure(Lattice.cubic(4.2), ["Al", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]

    fake_pca_feats = np.zeros((2, 50), dtype=np.float32)

    with patch("apu_synthesizability.features.orb_features",
               return_value=(fake_pca_feats, _FakePCA(), np.zeros((2, 1)))):
        build_feature_bank(manifest, structs, str(out), n_pca=50,
                           with_stability=False)

    assert out.exists(), ".npz was not created"

    bank = np.load(out, allow_pickle=True)
    required_keys = {"material_id", "formula", "magpie", "orb_pca", "label", "split"}
    for k in required_keys:
        assert k in bank, f"Missing key '{k}' in .npz bank"

    assert bank["magpie"].shape == (2, 132)
    assert bank["orb_pca"].shape == (2, 50)
    assert bank["label"].tolist() == [1, 0], "label array mismatch"
    assert bank["split"].tolist() == ["train", "test"], "split array mismatch"
    assert "stability" not in bank, "stability should be absent when with_stability=False"

    # PCA pickle should also exist
    pca_pkl = str(out) + ".pca.pkl"
    assert os.path.isfile(pca_pkl), "PCA pickle was not saved"
    with open(pca_pkl, "rb") as f:
        pca_obj = pickle.load(f)
    # _FakePCA is a module-level class so it round-trips through pickle
    assert isinstance(pca_obj, _FakePCA), (
        f"Expected _FakePCA, got {type(pca_obj)}"
    )
