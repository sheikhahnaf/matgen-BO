"""Tests for sharded featurization helpers (no ORB, no GPU required)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# shard_indices — partition correctness
# ---------------------------------------------------------------------------

def test_shard_indices_partition():
    """Union of shard_indices over all shards == range(n_rows), disjoint, sizes differ by <=1."""
    from apu_synthesizability.prep import shard_indices

    n_rows = 103
    n_shards = 20

    all_indices: list[np.ndarray] = []
    for s in range(n_shards):
        idx = shard_indices(n_rows, s, n_shards)
        all_indices.append(idx)

    # Union equals full range
    union = np.concatenate(all_indices)
    union_sorted = np.sort(union)
    np.testing.assert_array_equal(union_sorted, np.arange(n_rows))

    # Disjoint: no index appears twice
    assert len(union) == len(set(union.tolist())), "shard indices are not disjoint"

    # Sizes differ by at most 1
    sizes = [len(idx) for idx in all_indices]
    assert max(sizes) - min(sizes) <= 1, (
        f"shard sizes differ by more than 1: min={min(sizes)}, max={max(sizes)}"
    )


def test_shard_indices_single_shard():
    """With n_shards=1, shard_indices returns all rows."""
    from apu_synthesizability.prep import shard_indices

    idx = shard_indices(50, 0, 1)
    np.testing.assert_array_equal(idx, np.arange(50))


def test_shard_indices_out_of_range():
    from apu_synthesizability.prep import shard_indices

    with pytest.raises(ValueError, match="out of range"):
        shard_indices(100, 5, 5)  # valid range is 0-4


def test_shard_indices_zero_shards():
    from apu_synthesizability.prep import shard_indices

    with pytest.raises(ValueError, match="n_shards"):
        shard_indices(100, 0, 0)


# ---------------------------------------------------------------------------
# merge — concatenation + PCA + persistence
# ---------------------------------------------------------------------------

def _make_shard_npz(path: Path, n: int, shard_id: int, with_stability: bool = True) -> list[str]:
    """Write a synthetic shard .npz and return material_id list in order."""
    rng = np.random.default_rng(seed=shard_id)
    orb_raw = rng.random((n, 256), dtype=np.float32)
    magpie = rng.random((n, 132))
    labels = rng.integers(0, 2, size=n)
    splits = rng.choice(["train", "val", "test"], size=n)
    material_ids = np.array([f"mp-{shard_id * 1000 + i}" for i in range(n)])
    formulas = np.array([f"Fe{i}O{i+1}" for i in range(n)])

    save_kwargs = dict(
        orb_raw=orb_raw,
        magpie=magpie,
        label=labels,
        split=splits,
        material_id=material_ids,
        formula=formulas,
    )
    if with_stability:
        save_kwargs["stability"] = rng.random((n, 1))

    np.savez(path, **save_kwargs)
    return material_ids.tolist()


def test_merge_concats_and_pcas(tmp_path: Path):
    """merge() produces correct shapes and preserves shard order."""
    from apu_synthesizability.merge_shards import merge

    K = 30
    n_shards = 3

    expected_order: list[str] = []
    for s in range(n_shards):
        shard_path = tmp_path / f"shard_{s}.npz"
        ids = _make_shard_npz(shard_path, K, shard_id=s, with_stability=True)
        expected_order.extend(ids)

    out_npz = tmp_path / "bank.npz"
    merge(str(tmp_path), str(out_npz), n_pca=50)

    # --- Load and verify ---
    bank = np.load(out_npz, allow_pickle=False)

    # orb_pca: (90, 50)
    assert bank["orb_pca"].shape == (90, 50), (
        f"orb_pca shape: {bank['orb_pca'].shape}"
    )

    # magpie: (90, 132)
    assert bank["magpie"].shape == (90, 132), (
        f"magpie shape: {bank['magpie'].shape}"
    )

    # stability: (90, 1)
    assert bank["stability"].shape == (90, 1), (
        f"stability shape: {bank['stability'].shape}"
    )

    # label and split length 90
    assert len(bank["label"]) == 90
    assert len(bank["split"]) == 90

    # material_id length 90 and in shard order
    assert len(bank["material_id"]) == 90
    assert list(bank["material_id"]) == expected_order, (
        "material_id order does not match expected shard-concatenation order"
    )

    # pca.pkl exists and is a PCA object
    pca_pkl = Path(str(out_npz) + ".pca.pkl")
    assert pca_pkl.exists(), "pca.pkl not found"
    with open(pca_pkl, "rb") as f:
        pca_obj = pickle.load(f)
    from sklearn.decomposition import PCA
    assert isinstance(pca_obj, PCA), f"expected PCA, got {type(pca_obj)}"
    assert pca_obj.n_components_ == 50


def test_merge_no_overwrite(tmp_path: Path):
    """merge() raises FileExistsError if out_npz already exists."""
    from apu_synthesizability.merge_shards import merge

    # Create one shard so merge has something to read
    _make_shard_npz(tmp_path / "shard_0.npz", 10, shard_id=0)

    out_npz = tmp_path / "bank.npz"
    out_npz.touch()  # pre-create to trigger guard

    with pytest.raises(FileExistsError):
        merge(str(tmp_path), str(out_npz), n_pca=10)


def test_merge_missing_stability_omitted(tmp_path: Path):
    """If any shard lacks stability, the merged bank omits the stability block."""
    from apu_synthesizability.merge_shards import merge

    _make_shard_npz(tmp_path / "shard_0.npz", 20, shard_id=0, with_stability=True)
    _make_shard_npz(tmp_path / "shard_1.npz", 20, shard_id=1, with_stability=False)

    out_npz = tmp_path / "bank_nostab.npz"
    merge(str(tmp_path), str(out_npz), n_pca=10)

    bank = np.load(out_npz, allow_pickle=False)
    assert "stability" not in bank.files, (
        "stability block should be omitted when some shards lack it"
    )


def test_merge_no_shards_raises(tmp_path: Path):
    """merge() raises ValueError if no shard files found."""
    from apu_synthesizability.merge_shards import merge

    out_npz = tmp_path / "bank.npz"
    with pytest.raises(ValueError, match="no shard"):
        merge(str(tmp_path), str(out_npz), n_pca=10)
