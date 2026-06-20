"""
common.py — Shared utilities for ASE regression test suite.

Provides:
- CLI argument parsing (shared by gp/mtgp/dgp scripts)
- Data loading from matminer datasets
- Structure standardization
- Descriptor featurizers: SOAP, ORB, MACE, UMA/eSEN
- Metrics computation
- Parity plotting (paper-quality, 300 DPI)
- Results saving (CSV, JSON, LaTeX)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
})
import matplotlib.pyplot as plt
import spglib
import torch
from ase import Atoms
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METADATA_COLS = {
    'structure', 'composition', 'formula', 'spacegroup', 'material_id',
    'cif', 'task_id', 'nsites', 'volume', 'density', 'space_group',
}


def make_parser(model_name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{model_name} regression on matminer datasets")
    p.add_argument('--dataset', default='elastic_tensor_2015',
                   help='matminer dataset name')
    p.add_argument('--pca-components', type=int, default=50,
                   dest='pca_components')
    p.add_argument('--n-train', type=int, default=500, dest='n_train',
                   help='Number of training samples (rest become holdout)')
    p.add_argument('--holdout-cap', type=int, default=None, dest='holdout_cap',
                   help='Cap on holdout set size (randomly subsampled if exceeded; default: no cap)')
    p.add_argument('--n-splits', type=int, default=5, dest='n_splits')
    p.add_argument('--output-dir', default=None, dest='output_dir',
                   help='Output directory (default: results/<model>/<dataset>)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--descriptor', default='all',
                   choices=['all', 'soap', 'orb', 'mace', 'uma'],
                   help='Which descriptor(s) to run')
    p.add_argument('--dry-run', action='store_true', dest='dry_run',
                   help='Skip heavy descriptor computation, use random features for smoke test')
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matminer_dataset(dataset_name: str):
    """ADAPTED: dataset_name is now a path to a locally prepared pickled DataFrame."""
    import pandas as pd
    return pd.read_pickle(dataset_name)


def auto_detect_targets(df: pd.DataFrame) -> list:
    """Return numeric target columns excluding metadata and >20% NaN cols."""
    candidates = [
        c for c in df.columns
        if c not in METADATA_COLS
        and df[c].dtype in (float, 'float64', 'float32', int, 'int64')
        and df[c].isna().mean() <= 0.20
    ]
    return candidates


def prepare_dataset(df: pd.DataFrame, n_train: int, seed: int, holdout_cap=None):
    """Standardize structures, split into train/holdout, return (df_train, df_holdout).

    ADAPTED: ``holdout_cap`` (keyword-only in practice; the drivers call this
    positionally with the original three args) optionally bounds the holdout set
    size. If ``holdout_cap`` is None, the ``HOLDOUT_CAP`` env var is consulted so
    SLURM can enable the cap without editing the unmodified driver scripts.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()

    df = df.copy()
    df['ase_atoms'] = df['structure'].apply(
        lambda s: adaptor.get_atoms(s) if s is not None else None
    )
    df = df[df['ase_atoms'].notna()].reset_index(drop=True)
    print(f"Samples with valid structures: {len(df)}")

    print("Standardizing structures...")
    df['ase_atoms'] = df['ase_atoms'].apply(standardize_structure)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_train = df.iloc[:n_train].copy()
    df_holdout = df.iloc[n_train:].copy()

    if holdout_cap is None:
        _env = os.environ.get("HOLDOUT_CAP")
        holdout_cap = int(_env) if _env else None
    if holdout_cap is not None and len(df_holdout) > holdout_cap:
        df_holdout = df_holdout.sample(n=holdout_cap, random_state=seed).reset_index(drop=True)

    print(f"Training set: {len(df_train)}, Holdout set: {len(df_holdout)}")
    return df_train, df_holdout


def standardize_structure(atoms):
    """Standardize cell and orientation using spglib (to primitive)."""
    try:
        cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        res = spglib.standardize_cell(cell, to_primitive=True, symprec=1e-3)
        if res is not None:
            new_cell, new_pos, new_numbers = res
            return Atoms(numbers=new_numbers, scaled_positions=new_pos, cell=new_cell, pbc=True)
    except Exception:
        pass
    return atoms


# ---------------------------------------------------------------------------
# Featurizers
# ---------------------------------------------------------------------------

# ADAPTED: on-disk cache for RAW (pre-PCA) per-structure embeddings, keyed on
# material_id. Lets repeated runs (different PCA dims, models, seeds) skip the
# expensive featurizer forward pass. Disabled transparently when material_id is
# absent or a cached vector cannot be read.
#
# Namespacing: mace/orb/uma produce a fixed-width, split-independent vector, so
# their descriptor namespace is just the model name. SOAP is the exception — its
# raw width and element->position basis depend on the per-run TRAINING-split
# species set, so the SOAP path namespaces by an order-sensitive species
# signature (soap_<md5(species)[:8]>) and additionally width-guards on load.

def _feat_cache_root() -> str:
    return os.environ.get("FEAT_CACHE", "data/feat_cache")


def _get_material_ids(df: pd.DataFrame):
    """Return list of material_id strings aligned with df rows, or None if absent."""
    if 'material_id' not in df.columns:
        return None
    return [str(m) for m in df['material_id'].tolist()]


def _cache_path(descriptor: str, material_id: str) -> str:
    return os.path.join(_feat_cache_root(), descriptor, f"{material_id}.npy")


def _cache_load(descriptor: str, material_id, expected_len=None):
    """Load a cached RAW embedding; return None on miss/absent id/read error.

    If ``expected_len`` is given and the cached vector's length differs, the entry
    is treated as a MISS and recomputed. This is a safety net for descriptors
    whose raw width depends on run-specific state (notably SOAP, whose width is set
    by the per-run training-split species basis); it is a no-op for fixed-width
    descriptors (mace/orb/uma).
    """
    if material_id is None:
        return None
    path = _cache_path(descriptor, material_id)
    if os.path.exists(path):
        try:
            vec = np.load(path)
        except Exception:
            return None
        if expected_len is not None and vec.shape[-1] != expected_len:
            return None
        return vec
    return None


def _cache_save(descriptor: str, material_id, vec) -> None:
    """Persist a RAW embedding to the cache (best-effort; silent on failure).

    Write is atomic (write to a pid-tagged temp file then ``os.replace``) so that
    concurrent SLURM array tasks never observe a truncated .npy.
    """
    if material_id is None or vec is None:
        return
    path = _cache_path(descriptor, material_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{os.getpid()}.tmp"
        # Write through an open handle so np.save does NOT append a ".npy" suffix
        # to our temp name; then atomically swap into place.
        with open(tmp, 'wb') as fh:
            np.save(fh, np.asarray(vec))
        os.replace(tmp, path)
    except Exception:
        pass


class SOAPFeaturizer:
    """SOAP + compositional features with PCA reduction."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.soap = None
        self.pca = None
        self.species = None
        self.is_fitted = False
        # DATA-1 audit fix: per-row validity mask for the most recent
        # featurize call (len == number of input df rows). True == the row was
        # genuinely featurized; False == it fell back to a zero vector (unknown
        # species relative to the fitted training-split basis, or a featurizer
        # exception). Lets HOLDOUT metrics drop invalid-feature rows.
        self.last_valid_mask_ = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._featurize(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("SOAPFeaturizer not fitted!")
        return self._featurize(df, fit=False)

    def _featurize(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        from dscribe.descriptors import SOAP as DscribeSOAP

        atoms_list = df['ase_atoms'].tolist()
        valid_atoms = [a for a in atoms_list if a is not None]
        valid_indices = [i for i, a in enumerate(atoms_list) if a is not None]
        material_ids = _get_material_ids(df)
        valid_mids = (
            [material_ids[i] for i in valid_indices] if material_ids is not None
            else [None] * len(valid_atoms)
        )

        if fit:
            all_species: set = set()
            for atoms in valid_atoms:
                all_species.update(atoms.get_chemical_symbols())
            self.species = sorted(list(all_species))
            self.soap = DscribeSOAP(
                species=self.species, r_cut=5.0, n_max=4, l_max=3,
                average="outer", sparse=False
            )

        n_soap = self.soap.get_number_of_features()
        # SOAP's raw vector layout depends on self.species (the union over the
        # current TRAINING split only), so a cached vector is only valid for the
        # SAME species basis. Namespace the cache by an ORDER-SENSITIVE signature
        # of self.species: a different set/order => different namespace => safe
        # recompute, never a silent element->position mislabel across seed/n_train
        # sweeps. The width guard (expected_len) is a second line of defence.
        import hashlib
        sig = hashlib.md5(",".join(self.species).encode()).hexdigest()[:8]
        soap_desc = f'soap_{sig}'
        species_set = set(self.species)
        soap_feats = []
        # DATA-1: per-(valid_atoms) validity, aligned to the soap_feats order.
        # A row is INVALID if it contains an element absent from the fitted
        # training-split species basis (unknown-species zero fallback) or if
        # featurization raised (exception zero fallback). Determined from the
        # species check directly (not the cache path) so a previously-cached
        # zero vector is still correctly flagged invalid.
        soap_valid = []
        for atoms, mid in zip(valid_atoms, valid_mids):
            unknown = set(atoms.get_chemical_symbols()) - species_set
            cached = _cache_load(soap_desc, mid, expected_len=n_soap)
            if cached is not None:
                soap_feats.append(cached)
                soap_valid.append(not unknown)
                continue
            try:
                if unknown:
                    feat = np.zeros(n_soap)
                    is_valid = False
                else:
                    feat = self.soap.create(atoms).flatten()
                    is_valid = True
            except Exception:
                feat = np.zeros(n_soap)
                is_valid = False
            _cache_save(soap_desc, mid, feat)
            soap_feats.append(feat)
            soap_valid.append(is_valid)
        soap_feats = np.array(soap_feats)

        comp_feats = []
        for atoms in valid_atoms:
            nums = atoms.get_atomic_numbers()
            comp_feats.append([
                np.mean(nums),
                np.std(nums) if len(nums) > 1 else 0.0,
                len(set(nums)),
                len(nums),
            ])
        comp_feats = np.array(comp_feats)

        combined = np.hstack([soap_feats, comp_feats])

        if fit:
            if combined.shape[1] > self.n_components:
                self.pca = PCA(n_components=self.n_components)
                combined = self.pca.fit_transform(combined)
            self.is_fitted = True
        else:
            if self.pca is not None:
                combined = self.pca.transform(combined)

        features = np.zeros((len(df), combined.shape[1]))
        # DATA-1: scatter the per-(valid_atoms) validity through the SAME
        # valid_indices map the features use, so index alignment matches.
        # Rows with atoms is None (absent from valid_indices) stay False.
        valid_mask = np.zeros(len(df), dtype=bool)
        for i, idx in enumerate(valid_indices):
            features[idx] = combined[i]
            valid_mask[idx] = soap_valid[i]
        self.last_valid_mask_ = valid_mask
        return features


class MACEFeaturizer:
    """MACE node-level mean-pool embeddings with PCA reduction."""

    def __init__(self, n_components: int = 50, device: str = 'cpu'):
        self.n_components = n_components
        self.device = device
        self.pca = None
        self.is_fitted = False
        self._calc = None
        # DATA-1 audit fix: per-row validity mask for the most recent featurize
        # call (len == number of input df rows). False where the embedding could
        # not be extracted and fell back to a zero vector. Lets HOLDOUT metrics
        # drop invalid-feature rows.
        self.last_valid_mask_ = None

    def _load_calc(self):
        if self._calc is None:
            from mace.calculators import mace_mp
            print("Loading MACE model...")
            self._calc = mace_mp(model="medium", default_dtype="float64", device=self.device)
            print("MACE loaded!")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._featurize(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("MACEFeaturizer not fitted!")
        return self._featurize(df, fit=False)

    def _featurize(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        atoms_list = df['ase_atoms'].tolist()
        material_ids = _get_material_ids(df)
        print(f"Computing MACE descriptors for {len(atoms_list)} structures...")

        embeddings = []
        for i, atoms in enumerate(atoms_list):
            if atoms is None:
                embeddings.append(None)
                continue
            mid = material_ids[i] if material_ids is not None else None
            cached = _cache_load('mace', mid)
            if cached is not None:
                embeddings.append(cached)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(atoms_list)}")
                continue
            self._load_calc()
            try:
                atoms_copy = atoms.copy()
                atoms_copy.calc = self._calc
                descriptors = self._calc.get_descriptors(atoms_copy)
                if isinstance(descriptors, np.ndarray):
                    embedding = descriptors.mean(axis=0)
                elif isinstance(descriptors, dict):
                    for key in ['node_feats', 'descriptors', 'features']:
                        if key in descriptors:
                            desc = descriptors[key]
                            if hasattr(desc, 'numpy'):
                                desc = desc.numpy()
                            embedding = desc.mean(axis=0)
                            break
                    else:
                        desc = list(descriptors.values())[0]
                        if hasattr(desc, 'numpy'):
                            desc = desc.numpy()
                        embedding = np.array(desc).flatten()
                else:
                    if hasattr(descriptors, 'numpy'):
                        descriptors = descriptors.numpy()
                    embedding = np.array(descriptors).flatten()
                _cache_save('mace', mid, embedding)
                embeddings.append(embedding)
            except Exception as e:
                if i < 5:
                    print(f"  Warning {i}: {str(e)[:80]}")
                embeddings.append(None)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(atoms_list)}")

        return self._apply_pca(embeddings, len(df), fit)

    def _apply_pca(self, embeddings, n_total, fit):
        valid = [e for e in embeddings if e is not None]
        if not valid:
            raise ValueError("No valid embeddings extracted!")
        dim = valid[0].shape[0]
        # DATA-1: validity is "embedding was extracted" (not the None->zeros
        # fallback). `embeddings` is already aligned to the input df rows.
        self.last_valid_mask_ = np.array([e is not None for e in embeddings], dtype=bool)
        features = np.array([e if e is not None else np.zeros(dim) for e in embeddings])
        if fit:
            if self.n_components and features.shape[1] > self.n_components:
                self.pca = PCA(n_components=self.n_components)
                features = self.pca.fit_transform(features)
                print(f"  PCA variance retained: {np.sum(self.pca.explained_variance_ratio_)*100:.1f}%")
            self.is_fitted = True
        else:
            if self.pca is not None:
                features = self.pca.transform(features)
        return features


class ORBFeaturizer:
    """ORB node embedding capture (decoder hook) with PCA reduction."""

    def __init__(self, n_components: int = 50, device: str = 'cpu'):
        self.n_components = n_components
        self.device = device
        self.pca = None
        self.is_fitted = False
        self._calc = None
        # DATA-1 audit fix: per-row validity mask for the most recent featurize
        # call (len == number of input df rows). False where the embedding could
        # not be extracted and fell back to a zero vector. Lets HOLDOUT metrics
        # drop invalid-feature rows.
        self.last_valid_mask_ = None

    def _load_calc(self):
        if self._calc is None:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            print("Loading ORB model...")
            orbff = pretrained.orb_v3_conservative_inf_omat(
                device=self.device, precision="float32-high"
            )
            self._calc = ORBCalculator(orbff, device=self.device)
            print("ORB loaded!")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._featurize(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ORBFeaturizer not fitted!")
        return self._featurize(df, fit=False)

    def _featurize(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        import functools
        atoms_list = df['ase_atoms'].tolist()
        material_ids = _get_material_ids(df)
        print(f"Computing ORB embeddings for {len(atoms_list)} structures...")

        captured: dict = {}
        decoder = None
        original_forward = None

        def _ensure_hook():
            nonlocal decoder, original_forward
            if decoder is not None:
                return
            self._load_calc()
            inner_model = self._calc.model.model
            decoder = inner_model._decoder
            original_forward = decoder.forward

            @functools.wraps(original_forward)
            def wrapped_forward(x, *args, **kwargs):
                captured['node_feats'] = x.clone()
                return original_forward(x, *args, **kwargs)

            decoder.forward = wrapped_forward

        embeddings = []
        for i, atoms in enumerate(atoms_list):
            if atoms is None:
                embeddings.append(None)
                continue
            mid = material_ids[i] if material_ids is not None else None
            cached = _cache_load('orb', mid)
            if cached is not None:
                embeddings.append(cached)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(atoms_list)}")
                continue
            _ensure_hook()
            try:
                captured.clear()
                atoms_copy = atoms.copy()
                atoms_copy.calc = self._calc
                atoms_copy.get_potential_energy()
                if 'node_feats' in captured and captured['node_feats'] is not None:
                    node_feats = captured['node_feats'].detach().cpu().numpy()
                    embedding = node_feats.mean(axis=0)
                    _cache_save('orb', mid, embedding)
                    embeddings.append(embedding)
                else:
                    embeddings.append(None)
            except Exception as e:
                if i < 5:
                    print(f"  Warning {i}: {str(e)[:80]}")
                embeddings.append(None)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(atoms_list)}")

        if decoder is not None:
            decoder.forward = original_forward

        return self._apply_pca(embeddings, len(df), fit)

    def _apply_pca(self, embeddings, n_total, fit):
        valid = [e for e in embeddings if e is not None]
        if not valid:
            raise ValueError("No valid ORB embeddings extracted!")
        dim = valid[0].shape[0]
        # DATA-1: validity is "embedding was extracted" (not the None->zeros
        # fallback). `embeddings` is already aligned to the input df rows.
        self.last_valid_mask_ = np.array([e is not None for e in embeddings], dtype=bool)
        features = np.array([e if e is not None else np.zeros(dim) for e in embeddings])
        if fit:
            if self.n_components and features.shape[1] > self.n_components:
                self.pca = PCA(n_components=self.n_components)
                features = self.pca.fit_transform(features)
                print(f"  PCA variance retained: {np.sum(self.pca.explained_variance_ratio_)*100:.1f}%")
            self.is_fitted = True
        else:
            if self.pca is not None:
                features = self.pca.transform(features)
        return features


class UMAFeaturizer:
    """UMA/eSEN (FAIRChem) embedding capture with forward hook and PCA."""

    def __init__(self, n_components: int = 50, device: str = 'cpu'):
        self.n_components = n_components
        self.device = device
        self.pca = None
        self.is_fitted = False
        self._calc = None
        self._predictor = None
        # DATA-1 audit fix: per-row validity mask for the most recent featurize
        # call (len == number of input df rows). False where the embedding could
        # not be extracted and fell back to a zero vector. Lets HOLDOUT metrics
        # drop invalid-feature rows.
        self.last_valid_mask_ = None

    def _load_calc(self):
        if self._calc is None:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            print("Loading UMA/eSEN model...")
            torch.set_default_dtype(torch.float32)
            self._predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=self.device)
            self._calc = FAIRChemCalculator(self._predictor, task_name="omat")
            torch.set_default_dtype(torch.float64)
            print("UMA loaded!")

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self._featurize(df, fit=True)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("UMAFeaturizer not fitted!")
        return self._featurize(df, fit=False)

    def _featurize(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        atoms_list = df['ase_atoms'].tolist()
        material_ids = _get_material_ids(df)
        print(f"Computing UMA embeddings for {len(atoms_list)} structures...")

        captured: dict = {}

        def hook_fn(module, inp, out):
            captured['output'] = out

        handle = None
        original_dtype = torch.get_default_dtype()

        def _ensure_hook():
            nonlocal handle
            if handle is not None:
                return
            self._load_calc()
            model = self._predictor.model
            handle = model.module.register_forward_hook(hook_fn)
            # NOTE: this global default-dtype switch intentionally fires only on a
            # cache MISS (lazily, here). A fully-cached run never touches the
            # global dtype, and `original_dtype` is still restored unconditionally
            # at the end. Do not lift this back to an unconditional set at method
            # entry — that would defeat the model-free fully-cached fast path.
            torch.set_default_dtype(torch.float32)

        embeddings = []
        for i, atoms in enumerate(atoms_list):
            if atoms is None:
                embeddings.append(None)
                continue
            mid = material_ids[i] if material_ids is not None else None
            cached = _cache_load('uma', mid)
            if cached is not None:
                embeddings.append(cached)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(atoms_list)}")
                continue
            _ensure_hook()
            try:
                captured.clear()
                atoms_copy = atoms.copy()
                atoms_copy.set_positions(atoms_copy.get_positions().astype(np.float32))
                atoms_copy.set_cell(atoms_copy.get_cell().array.astype(np.float32))
                atoms_copy.calc = self._calc
                atoms_copy.get_potential_energy()

                if ('output' in captured
                        and 'omat_embeddings' in captured['output']
                        and 'embeddings' in captured['output']['omat_embeddings']):
                    emb = captured['output']['omat_embeddings']['embeddings']
                    node_feats = emb.detach().cpu().numpy().reshape(
                        emb.shape[0] if emb.ndim >= 2 else 1, -1
                    )
                    embedding = node_feats.mean(axis=0)
                    _cache_save('uma', mid, embedding)
                    embeddings.append(embedding)
                else:
                    embeddings.append(None)
            except Exception as e:
                if i < 5:
                    print(f"  Warning {i}: {str(e)[:80]}")
                embeddings.append(None)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(atoms_list)}")

        if handle is not None:
            handle.remove()
        torch.set_default_dtype(original_dtype)

        return self._apply_pca(embeddings, len(df), fit)

    def _apply_pca(self, embeddings, n_total, fit):
        valid = [e for e in embeddings if e is not None]
        if not valid:
            raise ValueError("No valid UMA embeddings extracted!")
        dim = valid[0].shape[0]
        # DATA-1: validity is "embedding was extracted" (not the None->zeros
        # fallback). `embeddings` is already aligned to the input df rows.
        self.last_valid_mask_ = np.array([e is not None for e in embeddings], dtype=bool)
        features = np.array([e if e is not None else np.zeros(dim) for e in embeddings])
        if fit:
            if self.n_components and features.shape[1] > self.n_components:
                self.pca = PCA(n_components=self.n_components)
                features = self.pca.fit_transform(features)
                print(f"  PCA variance retained: {np.sum(self.pca.explained_variance_ratio_)*100:.1f}%")
            self.is_fitted = True
        else:
            if self.pca is not None:
                features = self.pca.transform(features)
        return features


def get_featurizer(descriptor: str, n_components: int, device: str):
    """Factory returning the featurizer for the given descriptor name."""
    descriptor = descriptor.lower()
    if descriptor == 'soap':
        return SOAPFeaturizer(n_components=n_components)
    elif descriptor == 'mace':
        return MACEFeaturizer(n_components=n_components, device=device)
    elif descriptor == 'orb':
        return ORBFeaturizer(n_components=n_components, device=device)
    elif descriptor == 'uma':
        return UMAFeaturizer(n_components=n_components, device=device)
    else:
        raise ValueError(f"Unknown descriptor: {descriptor}")


ALL_DESCRIPTORS = ['soap', 'mace', 'orb', 'uma']


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.abs(y_pred) + np.abs(y_true)
    denom[denom == 0] = 1e-8
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / denom) * 100)
    spearman, _ = spearmanr(y_true, y_pred)
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'SMAPE': smape, 'Spearman': float(spearman)}


def aggregate_metrics(metrics_list: list) -> dict:
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        aggregated[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    return aggregated


# ---------------------------------------------------------------------------
# Plotting (paper-quality)
# ---------------------------------------------------------------------------

def parity_plot(
    y_train_true, y_train_pred, y_test_true, y_test_pred,
    y_train_std=None, y_test_std=None,
    title="Parity Plot", xlabel="True Values", ylabel="Predicted Values",
    split_label="Test", output_path=None
):
    """
    Paper-quality parity plot with error bars (±2σ), diagonal, and metrics inset.
    Returns (train_metrics, test_metrics).
    """
    train_metrics = compute_metrics(y_train_true, y_train_pred)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Train points
    if y_train_std is not None:
        ax.errorbar(y_train_true, y_train_pred, yerr=2 * y_train_std,
                    fmt='o', alpha=0.5, markersize=5, color='steelblue',
                    ecolor='lightsteelblue', elinewidth=0.8, capsize=2,
                    label='Train', markeredgecolor='navy', markeredgewidth=0.4)
    else:
        ax.scatter(y_train_true, y_train_pred, alpha=0.5, s=30,
                   color='steelblue', label='Train',
                   edgecolors='navy', linewidth=0.4)

    # Test/Holdout points
    if y_test_std is not None:
        ax.errorbar(y_test_true, y_test_pred, yerr=2 * y_test_std,
                    fmt='o', alpha=0.6, markersize=5, color='firebrick',
                    ecolor='lightcoral', elinewidth=0.8, capsize=2,
                    label=split_label, markeredgecolor='darkred', markeredgewidth=0.4)
    else:
        ax.scatter(y_test_true, y_test_pred, alpha=0.6, s=30,
                   color='firebrick', label=split_label,
                   edgecolors='darkred', linewidth=0.4)

    all_y = np.concatenate([y_train_true, y_test_true, y_train_pred, y_test_pred])
    mn, mx = all_y.min(), all_y.max()
    margin = (mx - mn) * 0.05
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            'k--', lw=1.5, label='Perfect')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    metrics_text = (
        f"Train: R²={train_metrics['R2']:.3f}  RMSE={train_metrics['RMSE']:.3f}\n"
        f"{split_label}: R²={test_metrics['R2']:.3f}  RMSE={test_metrics['RMSE']:.3f}\n"
        f"{split_label}: MAE={test_metrics['MAE']:.3f}  ρ={test_metrics['Spearman']:.3f}"
    )
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.85))

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Results saving
# ---------------------------------------------------------------------------

def save_results(
    results_dict: dict,
    output_dir: str,
    model_name: str,
    descriptor_name: str,
    dataset_name: str,
):
    """
    Save per-property metrics as CSV, JSON and a LaTeX table.

    results_dict format:
        {property_name: {'holdout': aggregate_metrics(...), 'test': ..., 'all_holdout_metrics': [...]}}
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- CSV: per-property mean±std ---
    rows = []
    for prop, res in results_dict.items():
        agg = res.get('holdout') or res.get('test')
        if agg is None:
            continue
        row = {'Property': prop, 'Descriptor': descriptor_name, 'Model': model_name}
        for metric, vals in agg.items():
            row[f'{metric}_mean'] = vals['mean']
            row[f'{metric}_std'] = vals['std']
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'{descriptor_name}_metrics.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # --- JSON: all raw per-split metrics ---
    json_path = os.path.join(output_dir, f'{descriptor_name}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=float)
    print(f"  Saved JSON: {json_path}")

    # --- LaTeX table ---
    latex_path = os.path.join(output_dir, f'{model_name}_metrics_table.tex')
    _write_latex_table(summary_df, latex_path, model_name, descriptor_name, dataset_name)
    print(f"  Saved LaTeX: {latex_path}")

    return summary_df


def _write_latex_table(df: pd.DataFrame, path: str, model_name, descriptor_name, dataset_name):
    """Write a paper-ready LaTeX table for holdout metrics."""
    metric_cols = [c for c in df.columns if c.endswith('_mean')]
    metrics = [c.replace('_mean', '') for c in metric_cols]

    header_cols = ' & '.join(['Property'] + [f'{m} (mean±std)' for m in metrics])
    rows = []
    for _, row in df.iterrows():
        cells = [row['Property']]
        for m in metrics:
            cells.append(f"${row[f'{m}_mean']:.3f} \\pm {row[f'{m}_std']:.3f}$")
        rows.append(' & '.join(cells) + r' \\')

    content = '\n'.join(rows)
    caption = (
        f"{model_name.upper()} holdout metrics ({descriptor_name} descriptor) "
        f"on \\texttt{{{dataset_name}}} dataset."
    )
    label = f"tab:{model_name}_{descriptor_name}_{dataset_name}"

    latex = (
        r"\begin{table}[ht]" + '\n'
        r"\centering" + '\n'
        r"\caption{" + caption + r"}" + '\n'
        r"\label{" + label + r"}" + '\n'
        r"\begin{tabular}{l" + "r" * len(metrics) + r"}" + '\n'
        r"\toprule" + '\n'
        + header_cols + r' \\' + '\n'
        + r"\midrule" + '\n'
        + content + '\n'
        + r"\bottomrule" + '\n'
        + r"\end{tabular}" + '\n'
        + r"\end{table}"
    )
    with open(path, 'w') as f:
        f.write(latex)


def save_cross_descriptor_summary(all_results: dict, output_dir: str, model_name: str):
    """
    Merge per-descriptor DataFrames into one cross-descriptor summary CSV.

    all_results: {descriptor: summary_df}
    """
    frames = list(all_results.values())
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    path = os.path.join(output_dir, f'{model_name}_all_descriptors_summary.csv')
    combined.to_csv(path, index=False)
    print(f"  Saved cross-descriptor summary: {path}")
    return combined
