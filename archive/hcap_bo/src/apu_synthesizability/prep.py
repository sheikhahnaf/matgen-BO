"""Grace prep driver for A-PU synthesizability sweep.

Runs on Grace (GPU node) — NOT locally.  Heavy imports (mp_api, pymatgen
Structure fetch, orb_models, fairchem) are all lazy so ``import
apu_synthesizability.prep`` succeeds without those packages installed.

Usage (on Grace)::

    # Full bank (original behaviour, unchanged)
    PYTHONPATH=$WORKDIR/src python -m apu_synthesizability.prep \\
        --notebook  /scratch/.../A_PU_RandomForest_alloy.ipynb \\
        --out-dir   $SCRATCH/apu-synth-sweep/cache \\
        --with-stability

    # Shard mode — one SLURM array task
    PYTHONPATH=$WORKDIR/src python -m apu_synthesizability.prep \\
        --notebook  /scratch/.../A_PU_RandomForest_alloy.ipynb \\
        --out-dir   $SCRATCH/apu-synth-sweep/cache_shards \\
        --shard-idx 3 --n-shards 20 \\
        --with-stability

In shard mode the script writes ``<out-dir>/shard_<shard_idx>.npz`` with
raw (256-d, un-PCA'd) ORB embeddings.  PCA is fit ONCE across all shards by
``merge_shards.py``.

The script exits early (non-zero) if ``<out-dir>/bank.npz`` already exists to
prevent accidental clobber (full mode).  In shard mode the no-overwrite guard
is on the individual shard file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m apu_synthesizability.prep",
        description="Fetch MP structures, build feature bank for APU sweep.",
    )
    p.add_argument(
        "--notebook",
        default="/scratch/user/ahnafalvi/notebooks/A_PU_RandomForest_alloy.ipynb",
        help="Path to the A-PU notebook that contains the MP API key.",
    )
    p.add_argument(
        "--out-dir",
        default="${SCRATCH}/apu-synth-sweep/cache",
        help="Output directory for manifest.parquet and bank.npz.",
    )
    p.add_argument(
        "--e-hull-max",
        type=float,
        default=0.5,
        help="Max energy above hull (eV) for unlabeled materials.",
    )
    p.add_argument(
        "--max-pos",
        type=int,
        default=60000,
        help="Maximum number of positive (synthesized) samples to query.",
    )
    p.add_argument(
        "--max-unl",
        type=int,
        default=120000,
        help="Maximum number of unlabeled samples to query.",
    )
    p.add_argument(
        "--n-pca",
        type=int,
        default=50,
        help="PCA dimensionality for ORB node embeddings (full-bank mode only).",
    )
    p.add_argument(
        "--with-stability",
        action="store_true",
        default=False,
        help="Include ORB-predicted energy/atom stability feature (fast; reuses ORB).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for train/val/test split assignment.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of material IDs per MPRester structure-fetch request.",
    )
    # --- shard mode ---
    p.add_argument(
        "--shard-idx",
        type=int,
        default=None,
        help=(
            "Shard index (0-based) for this SLURM array task.  "
            "When given, only the corresponding contiguous chunk of the "
            "dataset is featurized and saved as shard_<shard_idx>.npz "
            "with raw (un-PCA'd) 256-d ORB embeddings."
        ),
    )
    p.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help=(
            "Total number of shards.  Must be consistent across all array "
            "tasks so that the union of shards covers the full dataset."
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Shard index helper
# ---------------------------------------------------------------------------

def shard_indices(n_rows: int, shard_idx: int, n_shards: int) -> np.ndarray:
    """Return contiguous row indices for a given shard.

    Splits ``range(n_rows)`` into ``n_shards`` roughly equal contiguous chunks
    using ``np.array_split`` and returns the indices for chunk ``shard_idx``.

    Parameters
    ----------
    n_rows:
        Total number of rows in the full dataset.
    shard_idx:
        0-based index of the requested shard (0 <= shard_idx < n_shards).
    n_shards:
        Total number of shards.

    Returns
    -------
    np.ndarray of int
        1-D array of row indices belonging to this shard (contiguous).

    Raises
    ------
    ValueError
        If ``shard_idx`` is out of range or ``n_shards`` < 1.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not (0 <= shard_idx < n_shards):
        raise ValueError(
            f"shard_idx={shard_idx} out of range for n_shards={n_shards}"
        )
    chunks = np.array_split(np.arange(n_rows), n_shards)
    return chunks[shard_idx]


# ---------------------------------------------------------------------------
# Structure fetch
# ---------------------------------------------------------------------------

def _fetch_structures(api_key: str, material_ids: list, batch_size: int = 1000) -> dict:
    """Fetch pymatgen Structure objects from MP by material_id in batches.

    Parameters
    ----------
    api_key:
        Materials Project API key.
    material_ids:
        List of material_id strings (e.g. ["mp-1", "mp-2", ...]).
    batch_size:
        Number of IDs per MPRester call.  Default 1000 (MP rate-limit safe).

    Returns
    -------
    dict mapping material_id (str) -> pymatgen Structure
        Only IDs for which a structure was successfully retrieved are present.
    """
    # Lazy import — mp_api is only available on Grace
    from mp_api.client import MPRester  # noqa: lazy

    id_to_struct: dict = {}
    total = len(material_ids)
    n_batches = (total + batch_size - 1) // batch_size

    print(f"[prep] fetching structures for {total} materials in {n_batches} batches …")

    with MPRester(api_key) as mpr:
        for i in range(n_batches):
            batch = material_ids[i * batch_size: (i + 1) * batch_size]
            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch,
                    fields=["material_id", "structure"],
                )
                for doc in docs:
                    mid = str(doc.material_id)
                    if doc.structure is not None:
                        id_to_struct[mid] = doc.structure
            except Exception as exc:
                print(
                    f"[prep] WARNING: batch {i+1}/{n_batches} failed — "
                    f"{exc}; skipping batch."
                )
            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                print(
                    f"[prep]   batch {i+1}/{n_batches} done; "
                    f"{len(id_to_struct)} structures fetched so far."
                )

    return id_to_struct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Expand shell variables in out-dir (e.g. ${SCRATCH})
    import os
    out_dir = Path(os.path.expandvars(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_mode = args.shard_idx is not None

    if not shard_mode:
        # ------------------------------------------------------------------ #
        # FULL-BANK MODE (original behaviour — unchanged)
        # ------------------------------------------------------------------ #
        bank_path = out_dir / "bank.npz"
        manifest_path = out_dir / "manifest.parquet"

        # --- No-overwrite guard ---
        if bank_path.exists():
            print(
                f"[prep] ERROR: feature bank already exists at {bank_path}\n"
                "  Delete or rename the existing file to avoid overwriting data.",
                file=sys.stderr,
            )
            sys.exit(1)

        # --- Lazy imports for MP / dataset / feature modules ---
        from apu_synthesizability.dataset import (  # noqa: lazy
            read_mp_api_key,
            query_mp,
            assign_splits,
        )
        from apu_synthesizability.features import build_feature_bank  # noqa: lazy

        # --- Step 1: read API key ---
        print(f"[prep] reading MP API key from {args.notebook} …")
        api_key = read_mp_api_key(args.notebook)
        print("[prep] API key found.")

        # --- Step 2: query MP for material IDs + labels ---
        print(
            f"[prep] querying MP: e_hull_max={args.e_hull_max}, "
            f"max_pos={args.max_pos}, max_unl={args.max_unl} …"
        )
        df = query_mp(
            api_key,
            e_hull_max=args.e_hull_max,
            max_pos=args.max_pos,
            max_unl=args.max_unl,
        )

        n_pos_raw = int((df["label"] == 1).sum())
        n_unl_raw = int((df["label"] == 0).sum())
        print(
            f"[prep] MP query complete:"
            f"  positives={n_pos_raw}, unlabeled={n_unl_raw}, total={len(df)}"
        )

        # --- Step 3: fetch structures ---
        all_ids = df["material_id"].tolist()
        id_to_struct = _fetch_structures(api_key, all_ids, batch_size=args.batch_size)

        n_fetched = len(id_to_struct)
        n_missing = len(all_ids) - n_fetched
        print(
            f"[prep] structures fetched: {n_fetched}/{len(all_ids)} "
            f"({n_missing} dropped — no structure returned)."
        )

        # --- Step 4: align manifest to retrieved structures ---
        mask = df["material_id"].isin(id_to_struct)
        n_dropped = int((~mask).sum())
        if n_dropped:
            print(f"[prep] dropping {n_dropped} manifest rows with no structure.")
        df = df[mask].reset_index(drop=True)

        structures = [id_to_struct[mid] for mid in df["material_id"]]

        n_pos_final = int((df["label"] == 1).sum())
        n_unl_final = int((df["label"] == 0).sum())
        n_total_final = len(df)

        # --- Step 5: assign splits ---
        df = assign_splits(df, seed=args.seed)

        # --- Step 6: write manifest ---
        df.to_parquet(manifest_path, index=False)
        print(f"[prep] manifest written → {manifest_path}")

        # --- Print dataset counts (spec §12) ---
        print("\n=== Dataset Counts ===")
        print(f"  Positives (synthesized, label=1):  {n_pos_raw:>8d}  (raw query)")
        print(f"  Unlabeled (theoretical, label=0):  {n_unl_raw:>8d}  (raw query)")
        print(f"  Total queried:                     {n_pos_raw + n_unl_raw:>8d}")
        print(f"  Structures fetched:                {n_fetched:>8d}")
        print(f"  Dropped (no structure):            {n_dropped:>8d}")
        print(f"  Final positives:                   {n_pos_final:>8d}")
        print(f"  Final unlabeled:                   {n_unl_final:>8d}")
        print(f"  Final total:                       {n_total_final:>8d}")
        print("======================\n")

        # --- Step 7: build feature bank ---
        print(
            f"[prep] building feature bank → {bank_path}  "
            f"(n_pca={args.n_pca}, with_stability={args.with_stability}) …"
        )
        build_feature_bank(
            manifest=df,
            structures=structures,
            out_npz=str(bank_path),
            n_pca=args.n_pca,
            with_stability=args.with_stability,
        )

        print("[prep] done.")

    else:
        # ------------------------------------------------------------------ #
        # SHARD MODE — featurize a disjoint chunk, save raw ORB embeddings
        # ------------------------------------------------------------------ #
        shard_idx = args.shard_idx
        n_shards = args.n_shards

        shard_path = out_dir / f"shard_{shard_idx}.npz"

        # --- No-overwrite guard on this shard ---
        if shard_path.exists():
            print(
                f"[prep/shard] ERROR: shard file already exists at {shard_path}\n"
                "  Delete or rename the existing file to avoid overwriting data.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"[prep/shard] shard {shard_idx}/{n_shards}  "
            f"out={shard_path}"
        )

        # --- Lazy imports ---
        from apu_synthesizability.dataset import (  # noqa: lazy
            read_mp_api_key,
            query_mp,
            assign_splits,
        )
        from apu_synthesizability.features import (  # noqa: lazy
            orb_features,
            magpie_features,
        )

        # --- Step 1: read API key ---
        print(f"[prep/shard] reading MP API key from {args.notebook} …")
        api_key = read_mp_api_key(args.notebook)
        print("[prep/shard] API key found.")

        # --- Step 2: query MP (deterministic — same as full mode) ---
        print(
            f"[prep/shard] querying MP: e_hull_max={args.e_hull_max}, "
            f"max_pos={args.max_pos}, max_unl={args.max_unl} …"
        )
        df = query_mp(
            api_key,
            e_hull_max=args.e_hull_max,
            max_pos=args.max_pos,
            max_unl=args.max_unl,
        )

        # Assign splits on the FULL df first (deterministic order preserved)
        df = assign_splits(df, seed=args.seed)

        # --- Step 3: select this shard's rows (contiguous, no reshuffle) ---
        idx = shard_indices(len(df), shard_idx, n_shards)
        df_shard = df.iloc[idx].reset_index(drop=True)
        n_shard = len(df_shard)
        print(
            f"[prep/shard] shard rows: {n_shard} "
            f"(rows {int(idx[0])}–{int(idx[-1])} of {len(df)} total)"
        )

        # --- Step 4: fetch structures for this shard only ---
        shard_ids = df_shard["material_id"].tolist()
        id_to_struct = _fetch_structures(api_key, shard_ids, batch_size=args.batch_size)

        n_fetched = len(id_to_struct)
        n_missing = len(shard_ids) - n_fetched
        print(
            f"[prep/shard] structures fetched: {n_fetched}/{len(shard_ids)} "
            f"({n_missing} dropped)."
        )

        # Align shard manifest to fetched structures
        mask = df_shard["material_id"].isin(id_to_struct)
        n_dropped = int((~mask).sum())
        if n_dropped:
            print(f"[prep/shard] dropping {n_dropped} rows with no structure.")
        df_shard = df_shard[mask].reset_index(drop=True)

        structures = [id_to_struct[mid] for mid in df_shard["material_id"]]

        import torch  # noqa: lazy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[prep/shard] device={device}")

        # --- Step 5: Magpie ---
        formulas = df_shard["formula"].tolist()
        print(f"[prep/shard] computing Magpie for {len(formulas)} formulas …")
        magpie = magpie_features(formulas)
        print(f"  magpie shape: {magpie.shape}")

        # --- Step 6: ORB raw embeddings (n_pca=None → skip PCA, get 256-d) ---
        print(f"[prep/shard] computing ORB raw embeddings (n_pca=None) …")
        orb_raw, _pca_none, orb_energy = orb_features(
            structures, n_pca=None, device=device
        )
        print(f"  orb_raw shape: {orb_raw.shape}")

        # --- Step 7: assemble and save shard npz ---
        save_kwargs: dict = {
            "material_id": np.array(df_shard["material_id"].tolist()),
            "formula": np.array(formulas),
            "orb_raw": orb_raw,
            "magpie": magpie,
            "label": df_shard["label"].to_numpy(),
            "split": df_shard["split"].to_numpy(),
        }

        if args.with_stability:
            save_kwargs["stability"] = orb_energy
            print(f"  stability shape: {orb_energy.shape}")

        np.savez(shard_path, **save_kwargs)

        n_pos_shard = int((df_shard["label"] == 1).sum())
        n_unl_shard = int((df_shard["label"] == 0).sum())
        print(
            f"\n=== Shard {shard_idx}/{n_shards} Counts ===\n"
            f"  Total rows:   {len(df_shard):>8d}\n"
            f"  Positives:    {n_pos_shard:>8d}\n"
            f"  Unlabeled:    {n_unl_shard:>8d}\n"
            f"  orb_raw dim:  {orb_raw.shape[1]}\n"
            f"  Saved →       {shard_path}\n"
            f"=========================\n"
        )

        print("[prep/shard] done.")


if __name__ == "__main__":
    main()
