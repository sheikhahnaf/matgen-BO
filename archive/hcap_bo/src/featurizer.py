"""ORB-v3 and UMA/eSEN featurizers (PCA-reduced, mean-pooled node embeddings).

Adapted from (read-only upstream):
    /Volumes/SSD1_SMAAA/matinvent-bo/ASE_regression_test/common.py
    class ORBFeaturizer (lines ~292-380), class UMAFeaturizer (lines ~383-478).

Differences vs upstream:
    - Both featurizers accept list[ase.Atoms] directly (upstream took a DataFrame).
    - Shared abstract base for fit_transform/transform/save/load.
    - Adds ood_score(): PCA reconstruction error per sample.
    - Auto-cap PCA n_components at min(n_samples, n_features) to avoid sklearn
      blow-ups on small calibration batches.
    - Module-level singletons for both heavy models.
    - Factory get_featurizer(kind) for config-driven selection.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

# Module-level singletons — model loads are heavy (>2 GB GPU each).
_ORB_CALC = None
_ORB_DEVICE = None
_UMA_PREDICTOR = None
_UMA_CALC = None
_UMA_DEVICE = None


def _load_orb(device: str):
    global _ORB_CALC, _ORB_DEVICE
    if _ORB_CALC is None or _ORB_DEVICE != device:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device, precision="float32-high"
        )
        _ORB_CALC = ORBCalculator(orbff, device=device)
        _ORB_DEVICE = device
    return _ORB_CALC


def _load_uma(device: str):
    """Load FAIRChem UMA-S-1p1 predictor + calculator (singleton)."""
    global _UMA_PREDICTOR, _UMA_CALC, _UMA_DEVICE
    import torch
    if _UMA_CALC is None or _UMA_DEVICE != device:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        # UMA needs float32 default during model build; restore after.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            _UMA_PREDICTOR = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
            _UMA_CALC = FAIRChemCalculator(_UMA_PREDICTOR, task_name="omat")
        finally:
            torch.set_default_dtype(prev)
        _UMA_DEVICE = device
    return _UMA_PREDICTOR, _UMA_CALC


class ORBFeaturizer:
    """Mean-pooled ORB-v3 node embeddings, optionally PCA-reduced.

    Usage:
        feat = ORBFeaturizer(n_components=50, device="cuda")
        Z_train = feat.fit_transform(atoms_train)   # fits PCA
        Z_query = feat.transform(atoms_query)
        feat.save("data/featurizer_state.pkl")
    """

    def __init__(self, n_components: int = 50, device: str = "cpu"):
        self.n_components = n_components
        self.device = device
        self.pca: Optional[PCA] = None
        self.is_fitted: bool = False
        self.raw_dim: Optional[int] = None  # native ORB embedding dim before PCA

    # ----- featurization -------------------------------------------------

    def _featurize_raw(self, atoms_list) -> np.ndarray:
        """Run ORB and return a (N, raw_dim) array of mean-pooled node feats.

        Uses register_forward_pre_hook on the decoder (matches UMA's pattern,
        avoids the non-thread-safe monkey-patch). The hook captures the input
        tensor to the decoder (which is the node-feature tensor) into a local
        dict, then is removed in the finally block.
        """
        calc = _load_orb(self.device)
        decoder = calc.model.model._decoder
        captured: dict = {}

        def pre_hook(module, args):
            # Decoder forward takes (x, ...); capture x = node features.
            if args:
                captured["node_feats"] = args[0].detach()

        handle = decoder.register_forward_pre_hook(pre_hook)

        embeddings = []
        failure_mask = np.zeros(len(atoms_list), dtype=bool)

        try:
            for i, atoms in enumerate(atoms_list):
                if atoms is None:
                    embeddings.append(None)
                    failure_mask[i] = True
                    continue
                try:
                    captured.clear()
                    a = atoms.copy()
                    a.calc = calc
                    a.get_potential_energy()
                    nf = captured.get("node_feats")
                    if nf is not None:
                        node = nf.cpu().numpy()
                        embeddings.append(node.mean(axis=0))
                    else:
                        embeddings.append(None)
                        failure_mask[i] = True
                except Exception:
                    embeddings.append(None)
                    failure_mask[i] = True
        finally:
            handle.remove()

        valid = [e for e in embeddings if e is not None]
        if not valid:
            raise ValueError("No valid ORB embeddings extracted — all structures failed.")
        dim = valid[0].shape[0]
        self.raw_dim = dim
        feats = np.array([e if e is not None else np.zeros(dim) for e in embeddings])
        return feats, failure_mask

    def fit_transform(self, atoms_list) -> np.ndarray:
        feats, mask = self._featurize_raw(atoms_list)
        self._failure_mask_last = mask
        # PCA needs n_components <= min(n_samples, n_features); cap defensively.
        max_pc = max(1, min(feats.shape[0], feats.shape[1]) - 1)
        n_pc = min(self.n_components, max_pc) if self.n_components else 0
        if n_pc and feats.shape[1] > n_pc:
            self.pca = PCA(n_components=n_pc)
            feats = self.pca.fit_transform(feats)
        self.is_fitted = True
        return feats

    def transform(self, atoms_list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("ORBFeaturizer not fitted — call fit_transform first.")
        feats, mask = self._featurize_raw(atoms_list)
        self._failure_mask_last = mask
        if self.pca is not None:
            feats = self.pca.transform(feats)
        return feats

    def transform_with_mask(self, atoms_list):
        """Returns (Z, failure_mask). Failed rows in Z are zeros; mask is True there."""
        Z = self.transform(atoms_list)
        return Z, self._failure_mask_last

    # ----- OOD / diagnostics --------------------------------------------

    def ood_score(self, atoms_list) -> np.ndarray:
        """Per-sample PCA reconstruction error in the *raw* embedding space.

        High score = sample lies far off the training PCA manifold (likely OOD).
        Useful for triggering PCA refit when MatterGen samples drift away from
        the LTM-trained basis.
        """
        if not self.is_fitted or self.pca is None:
            raise ValueError("ORBFeaturizer must be fitted with PCA to compute OOD scores.")
        feats_raw, _ = self._featurize_raw(atoms_list)
        # Project + reconstruct
        z = self.pca.transform(feats_raw)
        reco = self.pca.inverse_transform(z)
        return np.linalg.norm(feats_raw - reco, axis=1)

    # ----- persistence --------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "n_components": self.n_components,
            "device": self.device,
            "raw_dim": self.raw_dim,
            "is_fitted": self.is_fitted,
            "pca": self.pca,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ORBFeaturizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_components=state["n_components"], device=device or state["device"])
        obj.pca = state["pca"]
        obj.raw_dim = state["raw_dim"]
        obj.is_fitted = state["is_fitted"]
        return obj


class UMAFeaturizer:
    """Mean-pooled UMA/eSEN embeddings (FAIRChem `uma-s-1p1`, task=omat).

    Captured via forward hook on the predictor's `module`; reads
    `output['omat_embeddings']['embeddings']`. Mean-pools along atoms.
    """

    def __init__(self, n_components: int = 50, device: str = "cpu"):
        self.n_components = n_components
        self.device = device
        self.pca: Optional[PCA] = None
        self.is_fitted: bool = False
        self.raw_dim: Optional[int] = None
        self._failure_mask_last: Optional[np.ndarray] = None

    def _featurize_raw(self, atoms_list):
        import torch
        predictor, calc = _load_uma(self.device)
        captured: dict = {}

        def hook(module, inp, out):
            captured["output"] = out

        handle = predictor.model.module.register_forward_hook(hook)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        embeddings = []
        failure_mask = np.zeros(len(atoms_list), dtype=bool)

        try:
            for i, atoms in enumerate(atoms_list):
                if atoms is None:
                    embeddings.append(None)
                    failure_mask[i] = True
                    continue
                try:
                    captured.clear()
                    a = atoms.copy()
                    a.set_positions(a.get_positions().astype(np.float32))
                    a.set_cell(a.get_cell().array.astype(np.float32))
                    a.calc = calc
                    a.get_potential_energy()

                    out = captured.get("output")
                    if (
                        out is not None
                        and "omat_embeddings" in out
                        and "embeddings" in out["omat_embeddings"]
                    ):
                        emb = out["omat_embeddings"]["embeddings"]
                        node = emb.detach().cpu().numpy()
                        if node.ndim == 1:
                            node = node.reshape(1, -1)
                        embeddings.append(node.mean(axis=0))
                    else:
                        embeddings.append(None)
                        failure_mask[i] = True
                except Exception:
                    embeddings.append(None)
                    failure_mask[i] = True
        finally:
            handle.remove()
            torch.set_default_dtype(prev)

        valid = [e for e in embeddings if e is not None]
        if not valid:
            raise ValueError("No valid UMA embeddings extracted — all structures failed.")
        dim = valid[0].shape[0]
        self.raw_dim = dim
        feats = np.array([e if e is not None else np.zeros(dim) for e in embeddings])
        return feats, failure_mask

    def fit_transform(self, atoms_list) -> np.ndarray:
        feats, mask = self._featurize_raw(atoms_list)
        self._failure_mask_last = mask
        max_pc = max(1, min(feats.shape[0], feats.shape[1]) - 1)
        n_pc = min(self.n_components, max_pc) if self.n_components else 0
        if n_pc and feats.shape[1] > n_pc:
            self.pca = PCA(n_components=n_pc)
            feats = self.pca.fit_transform(feats)
        self.is_fitted = True
        return feats

    def transform(self, atoms_list) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("UMAFeaturizer not fitted — call fit_transform first.")
        feats, mask = self._featurize_raw(atoms_list)
        self._failure_mask_last = mask
        if self.pca is not None:
            feats = self.pca.transform(feats)
        return feats

    def transform_with_mask(self, atoms_list):
        Z = self.transform(atoms_list)
        return Z, self._failure_mask_last

    def ood_score(self, atoms_list) -> np.ndarray:
        if not self.is_fitted or self.pca is None:
            raise ValueError("UMAFeaturizer must be fitted with PCA to compute OOD scores.")
        feats_raw, _ = self._featurize_raw(atoms_list)
        z = self.pca.transform(feats_raw)
        reco = self.pca.inverse_transform(z)
        return np.linalg.norm(feats_raw - reco, axis=1)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "kind": "uma",
            "n_components": self.n_components,
            "device": self.device,
            "raw_dim": self.raw_dim,
            "is_fitted": self.is_fitted,
            "pca": self.pca,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "UMAFeaturizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(n_components=state["n_components"], device=device or state["device"])
        obj.pca = state["pca"]
        obj.raw_dim = state["raw_dim"]
        obj.is_fitted = state["is_fitted"]
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_featurizer(kind: str, n_components: int = 50, device: str = "cpu"):
    """Build a featurizer by name. kind in {"orb", "uma"}."""
    kind = kind.lower()
    if kind == "orb":
        return ORBFeaturizer(n_components=n_components, device=device)
    if kind == "uma":
        return UMAFeaturizer(n_components=n_components, device=device)
    raise ValueError(f"Unknown featurizer kind: {kind!r} (expected one of: orb, uma)")
