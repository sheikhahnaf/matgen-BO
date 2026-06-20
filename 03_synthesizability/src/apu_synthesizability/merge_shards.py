"""Merge sharded featurization outputs into a single feature bank.

After all SLURM array tasks finish (each writing ``shard_<i>.npz``), run::

    PYTHONPATH=$WORKDIR/src python -m apu_synthesizability.merge_shards \\
        --shard-dir $SCRATCH/apu-synth-sweep/cache_shards \\
        --out       $SCRATCH/apu-synth-sweep/cache/bank.npz \\
        --n-pca     50

This fits PCA globally on the concatenated raw ORB embeddings from all shards
(PCA must be global, NOT per-shard) and writes the final bank.npz + pca.pkl.

Design notes
------------
* Shards are sorted numerically by their index (shard_0, shard_1, …) so the
  global row order is deterministic and identical to the original single-pass
  order.
* The ``stability`` block is optional: if ALL shards contain it, it is merged;
  if ANY shard lacks it, it is omitted and a warning is printed.
* No-overwrite guard on the output npz.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core merge function
# ---------------------------------------------------------------------------

def merge(
    shard_dir: str,
    out_npz: str,
    n_pca: int = 50,
) -> None:
    """Concatenate shard npz files, fit global PCA, and write bank.npz.

    Parameters
    ----------
    shard_dir:
        Directory containing ``shard_*.npz`` files produced by shard mode.
    out_npz:
        Output path for the merged feature bank.  Must NOT already exist.
    n_pca:
        Number of PCA components to retain from the 256-d ORB raw embeddings.
        Capped at ``min(n_pca, n_samples, n_features)``.

    Raises
    ------
    FileExistsError
        If ``out_npz`` already exists.
    ValueError
        If no shard files are found.
    """
    from sklearn.decomposition import PCA  # noqa: lazy

    out_path = Path(out_npz)
    if out_path.exists():
        raise FileExistsError(
            f"merge_shards: output file already exists: {out_npz}\n"
            "Delete or rename the existing file to avoid overwriting data."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Discover shard files, sort numerically ---
    shard_dir_path = Path(shard_dir)
    shard_files = sorted(
        shard_dir_path.glob("shard_*.npz"),
        key=lambda p: int(re.search(r"shard_(\d+)\.npz", p.name).group(1)),  # type: ignore[union-attr]
    )

    if not shard_files:
        raise ValueError(
            f"merge_shards: no shard_*.npz files found in {shard_dir}"
        )

    print(f"[merge_shards] found {len(shard_files)} shard files in {shard_dir}")

    # --- Load and concatenate all shards in order ---
    all_material_id: list = []
    all_formula: list = []
    all_orb_raw: list = []
    all_magpie: list = []
    all_stability: list = []
    all_label: list = []
    all_split: list = []
    has_stability_flags: list = []

    for sf in shard_files:
        data = np.load(sf, allow_pickle=False)
        n = len(data["material_id"])
        print(f"  {sf.name}: {n} rows")

        all_material_id.append(data["material_id"])
        all_formula.append(data["formula"])
        all_orb_raw.append(data["orb_raw"])
        all_magpie.append(data["magpie"])
        all_label.append(data["label"])
        all_split.append(data["split"])

        if "stability" in data:
            all_stability.append(data["stability"])
            has_stability_flags.append(True)
        else:
            has_stability_flags.append(False)

    # Concatenate arrays
    material_id = np.concatenate(all_material_id, axis=0)
    formula = np.concatenate(all_formula, axis=0)
    orb_raw = np.concatenate(all_orb_raw, axis=0)
    magpie = np.concatenate(all_magpie, axis=0)
    label = np.concatenate(all_label, axis=0)
    split = np.concatenate(all_split, axis=0)

    n_total = len(material_id)
    print(f"[merge_shards] total rows after concat: {n_total}")

    # --- Fit global PCA on full concatenated orb_raw ---
    n_components = min(n_pca, n_total, orb_raw.shape[1])
    print(
        f"[merge_shards] fitting PCA(n_components={n_components}) "
        f"on orb_raw shape {orb_raw.shape} …"
    )
    pca = PCA(n_components=n_components)
    orb_pca = pca.fit_transform(orb_raw)
    print(f"  orb_pca shape: {orb_pca.shape}")

    # --- Pickle PCA object ---
    pca_pkl_path = str(out_npz) + ".pca.pkl"
    with open(pca_pkl_path, "wb") as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[merge_shards] PCA saved → {pca_pkl_path}")

    # --- Stability block (only if ALL shards have it) ---
    include_stability = all(has_stability_flags)
    if not include_stability and any(has_stability_flags):
        n_missing_stab = sum(1 for x in has_stability_flags if not x)
        print(
            f"[merge_shards] WARNING: {n_missing_stab}/{len(shard_files)} shards "
            "lack stability; omitting stability block from merged bank."
        )

    save_kwargs: dict = {
        "material_id": material_id,
        "formula": formula,
        "orb_pca": orb_pca,
        "magpie": magpie,
        "label": label,
        "split": split,
    }

    if include_stability:
        stability = np.concatenate(all_stability, axis=0)
        save_kwargs["stability"] = stability
        print(f"[merge_shards] stability block included, shape: {stability.shape}")

    # --- Save merged bank ---
    np.savez(out_path, **save_kwargs)
    print(f"[merge_shards] saved → {out_npz}")
    print(
        f"\n=== Merge Summary ===\n"
        f"  Shards merged:   {len(shard_files)}\n"
        f"  Total rows:      {n_total}\n"
        f"  orb_pca shape:   {orb_pca.shape}\n"
        f"  magpie shape:    {magpie.shape}\n"
        f"  stability:       {'included' if include_stability else 'omitted'}\n"
        f"  PCA pkl →        {pca_pkl_path}\n"
        f"  Bank npz →       {out_npz}\n"
        f"=====================\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apu_synthesizability.merge_shards",
        description=(
            "Merge shard_*.npz files into a single feature bank, "
            "fitting global PCA on the concatenated ORB raw embeddings."
        ),
    )
    p.add_argument(
        "--shard-dir",
        required=True,
        help="Directory containing shard_*.npz files.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output path for the merged bank.npz.",
    )
    p.add_argument(
        "--n-pca",
        type=int,
        default=50,
        help="Number of PCA components (default: 50).",
    )
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    merge(
        shard_dir=args.shard_dir,
        out_npz=args.out,
        n_pca=args.n_pca,
    )


if __name__ == "__main__":
    main()
