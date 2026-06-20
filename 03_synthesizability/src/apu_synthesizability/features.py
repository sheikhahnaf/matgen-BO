"""Feature bank for APU synthesizability sweep.

Provides three named feature blocks:
    - magpie:    132-d composition features (ElementProperty "magpie" preset via matminer)
    - orb_pca:   n_pca-d mean-pooled ORB-v3 node embeddings (PCA-reduced)
    - stability: 1-d ORB-predicted total energy per atom (stability proxy)

IMPORT-SAFETY RULE
------------------
This module MUST import cleanly even when orb_models, fairchem, and matminer are
NOT installed (they are GPU-node-only dependencies).  All heavy imports are LAZY —
deferred to the body of each function that uses them.  Only numpy and stdlib are
imported at module level.

Design notes
------------
* orb_features() ports the ORBFeaturizer from
  /Volumes/SSD1_SMAAA/matinvent-hcap-bo/src/featurizer.py
  which uses register_forward_pre_hook on the decoder (preferred over the
  monkey-patch approach in the original notebook A_PU_RandomForest_alloy.ipynb).
  The hook captures the first positional arg to _decoder.forward, which is the
  per-atom node-feature tensor.  Mean-pooling over the atom axis gives a 256-d
  (ORB-v3 native dim) structure embedding.

* esen_stability() calls fairchem OCPCalculator with eSEN-30M-OAM for a
  single-point energy evaluation (no relaxation, no phonons).  It returns
  total_energy / n_atoms — a structural stability proxy, NOT a formation
  energy or e_above_hull.  Use with caution; label the axis clearly.

* build_feature_bank() orchestrates all three blocks and writes a named .npz
  bank.  The fitted PCA object is pickled alongside as <out_npz>.pca.pkl
  because np.savez cannot store arbitrary Python objects.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level ORB singleton (expensive to load; re-used across calls)
# ---------------------------------------------------------------------------
_ORB_CALC = None
_ORB_DEVICE = None


def _load_orb(device: str = "cpu"):
    """Lazy-load ORB-v3-conservative-inf-omat calculator (singleton per device)."""
    global _ORB_CALC, _ORB_DEVICE
    if _ORB_CALC is None or _ORB_DEVICE != device:
        from orb_models.forcefield import pretrained  # noqa: lazy
        from orb_models.forcefield.calculator import ORBCalculator  # noqa: lazy

        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
        )
        _ORB_CALC = ORBCalculator(orbff, device=device)
        _ORB_DEVICE = device
    return _ORB_CALC


# ---------------------------------------------------------------------------
# 1. Magpie features (matminer + pymatgen)
# ---------------------------------------------------------------------------

def magpie_features(formulas: List[str]) -> np.ndarray:
    """Compute Magpie ElementProperty composition features.

    Uses matminer's ``ElementProperty.from_preset("magpie")`` featurizer.
    Missing/invalid formulas receive a row of NaNs.

    Parameters
    ----------
    formulas:
        List of composition strings, e.g. ["Fe2O3", "LiMg"].

    Returns
    -------
    np.ndarray, shape (len(formulas), 132)
        132-d Magpie feature vector per formula.
    """
    # --- lazy imports ---
    from matminer.featurizers.composition import ElementProperty  # noqa
    from pymatgen.core import Composition  # noqa

    ep = ElementProperty.from_preset("magpie")
    results: List[np.ndarray] = []
    dim: Optional[int] = None

    for formula in formulas:
        try:
            comp = Composition(formula)
            feats = np.array(ep.featurize(comp), dtype=np.float64)
            if dim is None:
                dim = len(feats)
            results.append(feats)
        except Exception:
            # Placeholder until we know the dimension
            results.append(None)  # type: ignore[arg-type]

    if dim is None:
        raise ValueError("magpie_features: no formula could be featurized.")

    # Fill failed rows with NaN vectors of the right dimension
    final = []
    for r in results:
        if r is None:
            final.append(np.full(dim, np.nan))
        else:
            final.append(r)

    return np.array(final, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. ORB features (mean-pooled node embeddings + PCA)
# ---------------------------------------------------------------------------

def orb_features(
    structures,
    n_pca: Optional[int] = 50,
    pca=None,
    device: str = "cpu",
) -> Tuple[np.ndarray, object, np.ndarray]:
    """Compute mean-pooled ORB-v3 node embeddings, optionally PCA-reduced.

    For each structure:
      1. Convert pymatgen Structure → ASE Atoms.
      2. Run ORB-v3-conservative-inf-omat forward pass (single pass).
      3. Capture pre-decoder node features via ``register_forward_pre_hook``.
      4. Mean-pool over atoms → 256-d embedding.
      5. Capture the energy returned by ``get_potential_energy()`` as
         ``e_per_atom = E_total / N_atoms`` (no second ORB pass needed).
    If ``n_pca`` is an int, apply PCA to ``n_pca`` components (fit if
    ``pca is None``).  If ``n_pca is None``, skip PCA entirely and return
    the raw 256-d embeddings with ``pca=None`` — useful for sharded runs
    where PCA must be fit globally at merge time.

    Structures that fail produce zero-vectors and NaN energies.

    Parameters
    ----------
    structures:
        Iterable of ``pymatgen.core.Structure`` objects.
    n_pca:
        Target PCA dimension.  Capped at min(n_samples, raw_dim) − 1.
        Pass ``None`` to skip PCA and return raw 256-d embeddings.
    pca:
        If not None, an already-fitted ``sklearn.decomposition.PCA`` object.
        If None and ``n_pca`` is an int, PCA is fitted on the current batch.
        Ignored when ``n_pca is None``.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    features : np.ndarray, shape (N, n_pca) or (N, 256) when n_pca is None
        Mean-pooled ORB node embeddings, PCA-reduced if ``n_pca`` is set.
    pca_object : fitted sklearn.decomposition.PCA, or None
        The fitted PCA object (when ``n_pca`` is an int and PCA was applied),
        or None (when ``n_pca is None`` or PCA was not needed).
    energies : np.ndarray, shape (N, 1), dtype float64
        ORB-predicted energy per atom in eV/atom.  NaN for failed structures.
        Captured from the SAME forward pass as the embeddings (no second pass).

    Notes
    -----
    The hook approach (``register_forward_pre_hook``) is safer than the
    monkey-patch approach in the original notebook because it is re-entrant and
    can be removed atomically via ``handle.remove()`` in a ``finally`` block.

    When ``n_pca is None`` (raw mode), the returned ``features`` array holds
    the full 256-d ORB mean-pooled embeddings.  PCA should be fit once on the
    concatenation of all shards at merge time.
    """
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: lazy

    calc = _load_orb(device)
    adaptor = AseAtomsAdaptor()
    decoder = calc.model.model._decoder

    captured: dict = {}

    def _pre_hook(module, args):
        if args:
            captured["node_feats"] = args[0].detach()

    handle = decoder.register_forward_pre_hook(_pre_hook)

    raw_embeddings: List[Optional[np.ndarray]] = []
    raw_energies: List[float] = []

    try:
        for i, struct in enumerate(structures):
            if struct is None:
                raw_embeddings.append(None)
                raw_energies.append(float("nan"))
                continue
            try:
                captured.clear()
                atoms = adaptor.get_atoms(struct)
                atoms_cp = atoms.copy()
                atoms_cp.calc = calc
                e_total = atoms_cp.get_potential_energy()
                raw_energies.append(e_total / len(atoms_cp))
                nf = captured.get("node_feats")
                if nf is not None:
                    raw_embeddings.append(nf.cpu().numpy().mean(axis=0))
                else:
                    raw_embeddings.append(None)
            except Exception as exc:
                print(f"[orb_features] structure {i} failed: {exc}")
                raw_embeddings.append(None)
                raw_energies.append(float("nan"))
    finally:
        handle.remove()

    valid = [e for e in raw_embeddings if e is not None]
    if not valid:
        raise ValueError("orb_features: no valid embeddings extracted — all structures failed.")

    raw_dim = valid[0].shape[0]
    features = np.array(
        [e if e is not None else np.zeros(raw_dim) for e in raw_embeddings],
        dtype=np.float32,
    )

    # PCA — skipped entirely when n_pca is None (raw / shard mode)
    if n_pca is None:
        fitted_pca = None
    elif features.shape[1] > n_pca:
        from sklearn.decomposition import PCA  # noqa: lazy

        if pca is None:
            max_pc = min(features.shape[0], features.shape[1]) - 1
            n_pc = min(n_pca, max_pc)
            fitted_pca = PCA(n_components=n_pc)
            features = fitted_pca.fit_transform(features)
        else:
            fitted_pca = pca
            features = fitted_pca.transform(features)
    else:
        fitted_pca = None  # no PCA was applied (features already <= n_pca dim)

    energies = np.array(raw_energies, dtype=np.float64).reshape(-1, 1)

    return features, fitted_pca, energies


# ---------------------------------------------------------------------------
# 3. eSEN stability (total energy per atom)
# ---------------------------------------------------------------------------

def esen_stability(structures, device: Optional[str] = None) -> np.ndarray:
    """Compute eSEN-30M-OAM total energy per atom for each structure.

    This is a single-point (no relaxation) energy evaluation used as a
    structural stability proxy.  The returned value is ``E_total / N_atoms``
    in eV/atom as reported by the model.

    WARNING: This is NOT a formation energy, NOT an e_above_hull.
    It is a raw DFT-surrogate total energy per atom.  It correlates with
    stability rank within a dataset but cannot be compared across element
    compositions without a reference-energy correction.  Label any axis or
    feature column clearly as ``esen_E_per_atom_eV``.

    Parameters
    ----------
    structures:
        Iterable of ``pymatgen.core.Structure`` objects.
    device:
        Torch device.  If None, uses ``"cuda"`` when available else ``"cpu"``.

    Returns
    -------
    np.ndarray, shape (N, 1)
        Energy per atom in eV/atom.  NaN for structures that failed.
    """
    # --- lazy imports ---
    import torch  # noqa: lazy
    from fairchem.core import OCPCalculator  # noqa: lazy
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    local_cache = os.environ.get(
        "FAIRCHEM_LOCAL_CACHE",
        os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~")),
                     ".cache", "huggingface"),
    )
    calc = OCPCalculator(
        model_name="eSEN-30M-OAM",
        local_cache=local_cache,
        cpu=(device == "cpu"),
    )

    adaptor = AseAtomsAdaptor()
    results: List[float] = []

    for i, struct in enumerate(structures):
        if struct is None:
            results.append(float("nan"))
            continue
        try:
            atoms = adaptor.get_atoms(struct)
            atoms_cp = atoms.copy()
            atoms_cp.calc = calc
            e_total = atoms_cp.get_potential_energy()
            results.append(e_total / len(atoms_cp))
        except Exception as exc:
            print(f"[esen_stability] structure {i} failed: {exc}")
            results.append(float("nan"))

    return np.array(results, dtype=np.float64).reshape(-1, 1)


def orb_stability(structures, device: Optional[str] = None) -> np.ndarray:
    """ORB-predicted total energy per atom (eV/atom) as a fast stability feature.

    Standalone; the default bank path now gets ORB energy from orb_features (single pass) instead of calling this.

    Reuses the same ORB calculator already used for the embeddings (a single
    fast MLIP) rather than a separate, much slower model, so it is the
    practical stability signal at dataset scale.

    WARNING: like ``esen_stability`` this is a raw per-atom total energy, NOT a
    formation energy or e_above_hull. It also comes from the SAME model as the
    ORB embeddings, so as a feature it is correlated with the ``orb_pca`` block
    (it is a stability signal, not an independent one). Label any column as
    ``orb_E_per_atom_eV``.

    Parameters
    ----------
    structures:
        Iterable of ``pymatgen.core.Structure`` objects.
    device:
        Torch device. If None, uses ``"cuda"`` when available else ``"cpu"``.

    Returns
    -------
    np.ndarray, shape (N, 1)
        ORB energy per atom in eV/atom. NaN for structures that failed.
    """
    import torch  # noqa: lazy
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: lazy

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    calc = _load_orb(device)
    adaptor = AseAtomsAdaptor()
    results: List[float] = []
    for i, struct in enumerate(structures):
        if struct is None:
            results.append(float("nan"))
            continue
        try:
            atoms = adaptor.get_atoms(struct)
            atoms.calc = calc
            e_total = atoms.get_potential_energy()
            results.append(e_total / len(atoms))
        except Exception as exc:
            print(f"[orb_stability] structure {i} failed: {exc}")
            results.append(float("nan"))
    return np.array(results, dtype=np.float64).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 4. build_feature_bank — orchestrator
# ---------------------------------------------------------------------------

def build_feature_bank(
    manifest,
    structures,
    out_npz: str,
    n_pca: int = 50,
    with_stability: bool = False,
    device: str = "cpu",
) -> None:
    """Build and save a named feature bank (.npz) for the APU sweep.

    Computes:
        - magpie block   : (N, 132)  — composition features
        - orb_pca block  : (N, n_pca) — ORB mean-pooled embeddings, PCA-reduced
        - stability block: (N, 1)    — ORB-predicted energy/atom (only if with_stability=True)

    The fitted PCA object is pickled to ``<out_npz>.pca.pkl`` because
    ``np.savez`` cannot serialise arbitrary Python objects.

    Parameters
    ----------
    manifest:
        pandas DataFrame with at least columns ``material_id`` and ``formula``.
    structures:
        Aligned list/array of ``pymatgen.core.Structure`` objects,
        same length and order as ``manifest``.
    out_npz:
        Output path for the .npz bank.  Must NOT already exist
        (no-overwrite rule).
    n_pca:
        PCA dimension for ORB features.
    with_stability:
        If True, compute ORB-predicted energy/atom and include a ``stability``
        block (fast; reuses the ORB calculator).
    device:
        Torch device string for ORB (used for both embeddings and stability).

    Raises
    ------
    FileExistsError
        If ``out_npz`` already exists.
    """
    out_path = Path(out_npz)
    if out_path.exists():
        raise FileExistsError(
            f"build_feature_bank: output file already exists: {out_npz}\n"
            "Delete or rename the existing file to avoid overwriting data."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    material_ids = np.array(manifest["material_id"].tolist())
    formulas = manifest["formula"].tolist()
    structs = list(structures)
    labels = manifest["label"].to_numpy()
    splits = manifest["split"].to_numpy()

    # --- Magpie ---
    print(f"[build_feature_bank] computing Magpie for {len(formulas)} formulas …")
    magpie = magpie_features(formulas)
    print(f"  magpie shape: {magpie.shape}")

    # --- ORB + PCA (single pass: embeddings + energy captured together) ---
    print(f"[build_feature_bank] computing ORB embeddings …")
    orb_pca, fitted_pca, orb_energy = orb_features(structs, n_pca=n_pca, device=device)
    print(f"  orb_pca shape: {orb_pca.shape}")

    # --- Persist PCA object ---
    pca_pkl_path = str(out_npz) + ".pca.pkl"
    with open(pca_pkl_path, "wb") as f:
        pickle.dump(fitted_pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  PCA saved → {pca_pkl_path}")

    # --- ORB-energy stability (optional) ---
    save_kwargs: dict = {
        "material_id": material_ids,
        "formula": np.array(formulas),
        "magpie": magpie,
        "orb_pca": orb_pca,
        "label": labels,
        "split": splits,
    }

    if with_stability:
        stab = orb_energy
        print(f"[build_feature_bank] using ORB energy from embedding pass (single-pass)")
        print(f"  stability shape: {stab.shape}")
        save_kwargs["stability"] = stab

    # --- Save bank ---
    np.savez(out_path, **save_kwargs)
    print(f"[build_feature_bank] saved → {out_npz}")
    print(f"  blocks: {list(save_kwargs.keys())}")
