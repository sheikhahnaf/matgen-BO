"""Unit tests for ORB featurizer."""

import pytest
import numpy as np
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.orb.featurizer import ORBFeaturizer


@pytest.fixture
def featurizer(device):
    """Create ORB featurizer instance."""
    return ORBFeaturizer(n_components=10, device=device)


def test_featurizer_initialization(featurizer):
    """Test featurizer initializes correctly."""
    assert featurizer.n_components == 10
    assert featurizer.device == 'cpu'
    assert not featurizer.is_fitted


def test_featurize_single_structure(featurizer, fcc_structure):
    """Test featurization of a single structure."""
    adaptor = AseAtomsAdaptor()
    ase_atoms = adaptor.get_atoms(fcc_structure)
    df = pd.DataFrame({'ase_atoms': [ase_atoms]})

    features = featurizer.fit_transform(df)

    assert features.shape == (1, 10)  # 1 structure, 10 components after PCA
    assert featurizer.is_fitted
    assert not np.any(np.isnan(features))


def test_featurize_multiple_structures(featurizer, structure_list):
    """Test featurization of multiple structures."""
    adaptor = AseAtomsAdaptor()
    ase_atoms_list = [adaptor.get_atoms(s) for s in structure_list]
    df = pd.DataFrame({'ase_atoms': ase_atoms_list})

    features = featurizer.fit_transform(df)

    assert features.shape == (len(structure_list), 10)
    assert featurizer.is_fitted
    assert not np.any(np.isnan(features))


def test_transform_after_fit(featurizer, fcc_structure, bcc_structure):
    """Test transform on new data after fitting."""
    adaptor = AseAtomsAdaptor()

    # Fit on one structure
    ase_atoms_fit = adaptor.get_atoms(fcc_structure)
    df_fit = pd.DataFrame({'ase_atoms': [ase_atoms_fit]})
    featurizer.fit_transform(df_fit)

    # Transform on different structure
    ase_atoms_test = adaptor.get_atoms(bcc_structure)
    df_test = pd.DataFrame({'ase_atoms': [ase_atoms_test]})
    features_test = featurizer.transform(df_test)

    assert features_test.shape == (1, 10)
    assert not np.any(np.isnan(features_test))


def test_pca_reduction(device, fcc_structure):
    """Test that PCA correctly reduces dimensions."""
    adaptor = AseAtomsAdaptor()
    ase_atoms = adaptor.get_atoms(fcc_structure)
    df = pd.DataFrame({'ase_atoms': [ase_atoms]})

    # Test different n_components
    for n_comp in [5, 10, 20]:
        featurizer = ORBFeaturizer(n_components=n_comp, device=device)
        features = featurizer.fit_transform(df)
        assert features.shape[1] == n_comp


def test_featurize_direct_method(featurizer, structure_list):
    """Test direct featurize method."""
    features = featurizer.featurize(structure_list)

    assert features.shape == (len(structure_list), 10)
    assert featurizer.is_fitted
    assert not np.any(np.isnan(features))


def test_consistency_across_calls(featurizer, fcc_structure):
    """Test that featurization is consistent across multiple calls."""
    features1 = featurizer.featurize([fcc_structure])
    features2 = featurizer.transform_structures([fcc_structure])

    np.testing.assert_array_almost_equal(features1, features2, decimal=5)


def test_batch_consistency(featurizer, structure_list):
    """Test that batch and sequential featurization give same results."""
    # Batch featurization
    features_batch = featurizer.featurize(structure_list)

    # Sequential featurization
    features_seq = []
    for structure in structure_list:
        feat = featurizer.transform_structures([structure])
        features_seq.append(feat[0])
    features_seq = np.array(features_seq)

    np.testing.assert_array_almost_equal(features_batch, features_seq, decimal=5)


@pytest.mark.parametrize("n_structures", [1, 3, 5, 10])
def test_different_batch_sizes(featurizer, fcc_structure, n_structures):
    """Test featurization with different batch sizes."""
    structures = [fcc_structure] * n_structures
    features = featurizer.featurize(structures)

    assert features.shape == (n_structures, 10)
    assert not np.any(np.isnan(features))
