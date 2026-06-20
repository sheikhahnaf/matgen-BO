"""ORB Featurizer for extracting embeddings from crystal structures.

This module implements the exact ORBFeaturizer from the user's working code.
Uses ORB (Orbital Materials) model to extract structure embeddings for GP input.
"""

import functools
import numpy as np
from typing import List, Union
import pandas as pd
from pymatgen.core.structure import Structure
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


# Global state for ORB model
_ORB_CALC = None
_ORB_PCA = None


class ORBFeaturizer:
    """
    Handles ORB embedding extraction with PCA.

    Extracts intermediate representations from ORB model's decoder layer,
    then reduces dimensionality using PCA.

    Attributes:
        n_components (int): Number of PCA components (default: 50)
        is_fitted (bool): Whether PCA has been fitted
    """

    def __init__(self, n_components=50, device='cpu'):
        """
        Initialize ORB Featurizer.

        Args:
            n_components (int): Number of PCA components for dimensionality reduction
            device (str): Device to run ORB model on ('cpu' or 'cuda')
        """
        self.n_components = n_components
        self.is_fitted = False
        self.device = device

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit PCA and transform structures to features.

        Args:
            df: DataFrame with 'ase_atoms' column containing ASE Atoms objects

        Returns:
            np.ndarray: Feature matrix (n_structures, n_components)
        """
        return self._featurize(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform structures to features using fitted PCA.

        Args:
            df: DataFrame with 'ase_atoms' column containing ASE Atoms objects

        Returns:
            np.ndarray: Feature matrix (n_structures, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Featurizer not fitted!")
        return self._featurize(df, fit=False)

    def featurize(self, structures: List[Structure]) -> np.ndarray:
        """
        Featurize a list of pymatgen Structure objects.

        Convenience method that converts structures to DataFrame format.

        Args:
            structures: List of pymatgen Structure objects

        Returns:
            np.ndarray: Feature matrix (n_structures, n_components)
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()
        ase_atoms_list = [adaptor.get_atoms(s) for s in structures]

        df = pd.DataFrame({'ase_atoms': ase_atoms_list})

        if self.is_fitted:
            return self.transform(df)
        else:
            return self.fit_transform(df)

    def _featurize(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Internal featurization method.

        Args:
            df: DataFrame with 'ase_atoms' column
            fit: Whether to fit PCA (True) or use existing fit (False)

        Returns:
            np.ndarray: Feature matrix
        """
        global _ORB_CALC, _ORB_PCA

        # Lazy-load ORB calculator
        if _ORB_CALC is None:
            print("Loading ORB model...")
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator

            orbff = pretrained.orb_v3_conservative_inf_omat(
                device=self.device,
                precision="float32-high",
            )
            _ORB_CALC = ORBCalculator(orbff, device=self.device)
            print("ORB loaded!")

        atoms_list = df['ase_atoms'].tolist()
        print(f"Computing ORB embeddings for {len(atoms_list)} structures...")

        embeddings = []
        captured = {}

        # Get the inner model's decoder
        inner_model = _ORB_CALC.model.model
        decoder = inner_model._decoder

        # Store original forward
        original_forward = decoder.forward

        # Wrap forward to capture input
        @functools.wraps(original_forward)
        def wrapped_forward(x, *args, **kwargs):
            captured['node_feats'] = x.clone()
            return original_forward(x, *args, **kwargs)

        # Monkey-patch the forward method
        decoder.forward = wrapped_forward
        print("  Wrapped decoder.forward to capture embeddings")

        for i, atoms in enumerate(atoms_list):
            if atoms is None:
                embeddings.append(None)
                continue

            try:
                captured.clear()

                atoms_copy = atoms.copy()
                atoms_copy.calc = _ORB_CALC

                _ = atoms_copy.get_potential_energy()

                if 'node_feats' in captured and captured['node_feats'] is not None:
                    node_feats = captured['node_feats'].detach().cpu().numpy()

                    if i == 0:
                        print(f"  Captured tensor shape: {node_feats.shape}")

                    # Mean pool over atoms
                    embedding = node_feats.mean(axis=0)

                    if i == 0:
                        print(f"  Embedding dimension: {embedding.shape[0]}")

                    embeddings.append(embedding)
                else:
                    if i < 5:
                        print(f"  Structure {i}: No features captured")
                    embeddings.append(None)

            except Exception as e:
                if i < 5:
                    print(f"  Warning {i}: {str(e)[:60]}")
                embeddings.append(None)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(atoms_list)}")

        # Restore original forward
        decoder.forward = original_forward

        # Handle valid/invalid
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            raise ValueError("No valid ORB embeddings extracted!")

        embed_dim = valid_embeddings[0].shape[0]
        print(f"  Valid: {len(valid_embeddings)}/{len(embeddings)}, dim: {embed_dim}")

        features = np.array([
            e if e is not None else np.zeros(embed_dim)
            for e in embeddings
        ])
        print(f"Raw embeddings shape: {features.shape}")

        # PCA reduction
        if fit:
            if self.n_components and features.shape[1] > self.n_components:
                print(f"Fitting PCA: {features.shape[1]} -> {self.n_components}")
                _ORB_PCA = PCA(n_components=self.n_components)
                features = _ORB_PCA.fit_transform(features)
                print(f"PCA variance retained: {np.sum(_ORB_PCA.explained_variance_ratio_)*100:.1f}%")
            self.is_fitted = True
        else:
            if _ORB_PCA is not None:
                features = _ORB_PCA.transform(features)

        print(f"Final feature shape: {features.shape}")
        return features
