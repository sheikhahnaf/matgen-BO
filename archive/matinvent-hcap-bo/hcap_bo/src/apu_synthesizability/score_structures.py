"""Score generated diffusion structures with A-PU model and/or CGNF.

This module provides:
- ``parse_tag``      – decode filename → Tag dataclass
- ``load_structures`` – read a directory of *.cif files via pymatgen
- ``score_apu``      – run a trained A-PU sklearn estimator over structures
- ``score_cgnf``     – call the CGNF predict() from matinvent-bo (Grace-side)
- ``concordance``    – Spearman rho + threshold-agreement between two score arrays
- ``main``           – CLI entry point
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backbone code → human-readable name
# ---------------------------------------------------------------------------

_BACKBONE_NAMES: dict[str, str] = {
    "mg": "MatterGen",
    "cf": "CrystalFlow",
    "adit": "ADiT",
}

# ---------------------------------------------------------------------------
# Tag dataclass
# ---------------------------------------------------------------------------


@dataclass
class Tag:
    """Structured metadata decoded from a generated-structure CIF filename."""

    target: str       # "bm" | "cp"
    rank: int         # numeric rank (1-based)
    backbone: str     # raw code: "mg" | "cf" | "adit"
    backbone_name: str  # human name: "MatterGen" | "CrystalFlow" | "ADiT"
    policy: str       # "BASE" | "ACC"
    seed: int
    formula: str      # chemical formula string (may contain parentheses)
    spacegroup: int


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

# Pattern explanation:
#   ^(bm|cp)_top(\d+)_(mg|cf|adit)_(BASE|ACC)_seed(\d+)_(.+?)_sg(\d+)\.cif$
#
# The formula field (.+?) uses a *non-greedy* match so that the last _sg\d+
# group is consumed greedily from the right, correctly handling formulas that
# contain underscores (none exist today, but the regex is safe regardless).
_FILENAME_RE = re.compile(
    r"^(?P<target>bm|cp)"
    r"_top(?P<rank>\d+)"
    r"_(?P<backbone>mg|cf|adit)"
    r"_(?P<policy>BASE|ACC)"
    r"_seed(?P<seed>\d+)"
    r"_(?P<formula>.+?)"
    r"_sg(?P<spacegroup>\d+)\.cif$"
)


def parse_tag(filename: str) -> Tag:
    """Parse a CIF filename into a :class:`Tag`.

    Parameters
    ----------
    filename:
        Basename only, e.g. ``"bm_top11_adit_ACC_seed17_TaB2W_sg38.cif"``.

    Returns
    -------
    Tag

    Raises
    ------
    ValueError
        If the filename does not match the expected convention.
    """
    name = os.path.basename(filename)
    m = _FILENAME_RE.match(name)
    if m is None:
        raise ValueError(
            f"Filename '{name}' does not match the expected CIF naming convention.\n"
            "Expected: <target>_top<NN>_<backbone>_<policy>_seed<N>_<formula>_sg<N>.cif"
        )
    backbone = m.group("backbone")
    return Tag(
        target=m.group("target"),
        rank=int(m.group("rank")),
        backbone=backbone,
        backbone_name=_BACKBONE_NAMES[backbone],
        policy=m.group("policy"),
        seed=int(m.group("seed")),
        formula=m.group("formula"),
        spacegroup=int(m.group("spacegroup")),
    )


# ---------------------------------------------------------------------------
# Structure loader
# ---------------------------------------------------------------------------


def load_structures(cif_dir: str | Path) -> List[Tuple[Tag, "pymatgen.core.Structure"]]:
    """Load all ``*.cif`` files in *cif_dir* using pymatgen.

    pymatgen is imported lazily so that the rest of the module (parse_tag,
    concordance, …) can be used in environments without pymatgen installed.

    Parameters
    ----------
    cif_dir:
        Path to the directory containing generated CIF files.

    Returns
    -------
    list of (Tag, pymatgen.core.Structure) pairs, sorted by filename.
    """
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise ImportError(
            "pymatgen is required for load_structures(). "
            "Install it with: pip install pymatgen"
        ) from exc

    cif_dir = Path(cif_dir)
    cif_files = sorted(cif_dir.glob("*.cif"))
    if not cif_files:
        raise FileNotFoundError(f"No *.cif files found in {cif_dir}")

    result: List[Tuple[Tag, "pymatgen.core.Structure"]] = []
    for path in cif_files:
        tag = parse_tag(path.name)
        struct = Structure.from_file(str(path))
        result.append((tag, struct))
    return result


# ---------------------------------------------------------------------------
# A-PU scorer
# ---------------------------------------------------------------------------


def score_apu(
    structures: List,
    model,
    feature_fn: Callable,
) -> np.ndarray:
    """Score structures with a trained A-PU model.

    Parameters
    ----------
    structures:
        List of pymatgen Structure objects (same order as ``feature_fn`` expects).
    model:
        A trained sklearn-compatible estimator with ``predict_proba``.
    feature_fn:
        Callable that takes *structures* and returns a 2-D feature matrix
        (n_structures, n_features).

    Returns
    -------
    np.ndarray of shape (n_structures,) with positive-class probabilities.
    """
    X = feature_fn(structures)
    proba = model.predict_proba(X)
    # predict_proba returns (n, 2); column 1 is P(positive)
    if proba.ndim == 2:
        return proba[:, 1]
    return np.asarray(proba)


# ---------------------------------------------------------------------------
# CGNF scorer
# ---------------------------------------------------------------------------


def score_cgnf(
    structures: List,
    syn_score_parent: Optional[str] = None,
) -> np.ndarray:
    """Score structures with the pretrained CGNF synthesizability scorer.

    This is intended to run on Grace (or wherever the matinvent-bo package and
    CGNF checkpoints reside).  The function is **not** unit-tested live; it is
    wired for completeness.

    Parameters
    ----------
    structures:
        List of pymatgen Structure objects.
    syn_score_parent:
        If provided, prepend this directory to ``sys.path`` before importing,
        allowing ``from rewards.calculators.syn_score.predict import predict``
        to resolve.  Typically the path to the ``matinvent`` package root, e.g.
        ``/Volumes/SSD1_SMAAA/matinvent-bo``.

    Returns
    -------
    np.ndarray of shape (n_structures,) with synthesizability scores in [0, 1].
    """
    if syn_score_parent is not None and syn_score_parent not in sys.path:
        sys.path.insert(0, syn_score_parent)
    try:
        from rewards.calculators.syn_score.predict import predict  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Cannot import CGNF predict(). "
            "Pass syn_score_parent pointing to the matinvent-bo bundle root, "
            "or run on a machine where the matinvent package is installed.\n"
            f"Original error: {exc}"
        ) from exc
    return np.asarray(predict(structures))


# ---------------------------------------------------------------------------
# Concordance metric
# ---------------------------------------------------------------------------


def concordance(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute concordance metrics between two score arrays.

    Parameters
    ----------
    a, b:
        1-D arrays of scores in [0, 1].  Must have the same length.

    Returns
    -------
    dict with keys:
        ``"spearman"``      – Spearman rank correlation (float)
        ``"agree_gt_half"`` – fraction of samples where (a>0.5)==(b>0.5)
        ``"n"``             – number of samples
    """
    from scipy.stats import spearmanr  # lazy import — scipy is always present

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length.")

    n = len(a)
    rho, _ = spearmanr(a, b)
    agree = float(np.mean((a > 0.5) == (b > 0.5)))
    return {"spearman": float(rho), "agree_gt_half": agree, "n": n}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score generated diffusion structures with A-PU and CGNF models."
    )
    p.add_argument("--cif-dir", required=True, help="Directory containing *.cif files.")
    p.add_argument(
        "--apu-model",
        required=True,
        help="Path to a pickled A-PU model (sklearn estimator with predict_proba).",
    )
    p.add_argument(
        "--syn-score-parent",
        default=None,
        help="Parent directory for CGNF syn_score package (inserted on sys.path).",
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for per-structure results.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI: load structures, score with A-PU + CGNF, write CSV, print summary."""
    import pickle

    import pandas as pd

    args = _build_parser().parse_args(argv)

    # ---- load model ----
    with open(args.apu_model, "rb") as fh:
        apu_bundle = pickle.load(fh)
    # Bundle may be a dict {"model": ..., "feature_fn": ...} or a plain model.
    if isinstance(apu_bundle, dict):
        model = apu_bundle["model"]
        feature_fn = apu_bundle["feature_fn"]
    else:
        raise ValueError(
            "APU model pickle must be a dict with keys 'model' and 'feature_fn'."
        )

    # ---- load structures ----
    print(f"Loading CIFs from {args.cif_dir} …")
    items = load_structures(args.cif_dir)
    tags, structs = zip(*items)
    print(f"  Loaded {len(structs)} structures.")

    # ---- score A-PU ----
    print("Scoring with A-PU model …")
    apu_scores = score_apu(list(structs), model, feature_fn)

    # ---- score CGNF ----
    print("Scoring with CGNF …")
    cgnf_scores = score_cgnf(list(structs), syn_score_parent=args.syn_score_parent)

    # ---- concordance ----
    conc = concordance(apu_scores, cgnf_scores)
    print(
        f"\nOverall concordance (n={conc['n']}):\n"
        f"  Spearman rho      = {conc['spearman']:.4f}\n"
        f"  Agree (>0.5 gate) = {conc['agree_gt_half']:.3f}"
    )

    # ---- build DataFrame ----
    rows = []
    for tag, apu_s, cgnf_s in zip(tags, apu_scores, cgnf_scores):
        rows.append(
            {
                "target": tag.target,
                "rank": tag.rank,
                "backbone": tag.backbone,
                "backbone_name": tag.backbone_name,
                "policy": tag.policy,
                "seed": tag.seed,
                "formula": tag.formula,
                "spacegroup": tag.spacegroup,
                "apu_score": float(apu_s),
                "cgnf_score": float(cgnf_s),
            }
        )
    df = pd.DataFrame(rows)

    # ---- per-(backbone, policy) summary ----
    print("\nPer-(backbone, policy) mean scores:")
    summary = (
        df.groupby(["backbone_name", "policy"])[["apu_score", "cgnf_score"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())

    # ---- write CSV ----
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
