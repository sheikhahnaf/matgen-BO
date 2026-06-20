"""Offline tests for apu_synthesizability.prep (no network, no GPU)."""

import sys
import pytest


def test_import_without_mp_api():
    """prep module must import cleanly even when mp_api is absent."""
    import apu_synthesizability.prep  # noqa: F401


def test_argparse_defaults():
    """Parser defaults match the spec values."""
    from apu_synthesizability.prep import _build_parser
    args = _build_parser().parse_args([])
    assert args.e_hull_max == 0.5
    assert args.max_pos == 60000
    assert args.max_unl == 120000
    assert args.n_pca == 50
    assert args.with_stability is False
    assert args.seed == 0
    assert args.batch_size == 1000


def test_argparse_with_stability_flag():
    """--with-stability sets the flag to True."""
    from apu_synthesizability.prep import _build_parser
    args = _build_parser().parse_args(["--with-stability"])
    assert args.with_stability is True


def test_argparse_custom_values():
    """All numeric CLI args are parsed correctly."""
    from apu_synthesizability.prep import _build_parser
    args = _build_parser().parse_args([
        "--e-hull-max", "0.3",
        "--max-pos", "10000",
        "--max-unl", "20000",
        "--n-pca", "32",
        "--seed", "42",
        "--batch-size", "500",
    ])
    assert args.e_hull_max == pytest.approx(0.3)
    assert args.max_pos == 10000
    assert args.max_unl == 20000
    assert args.n_pca == 32
    assert args.seed == 42
    assert args.batch_size == 500


def test_main_exits_on_existing_bank(tmp_path):
    """main() exits with sys.exit(1) when bank.npz already exists."""
    # Create a fake notebook with an API key so read_mp_api_key doesn't fail
    import json
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "source": ['api_key = "AAAAAAAAAAAAAAAA"\n'],
            }
        ]
    }
    nb_path = tmp_path / "fake_notebook.ipynb"
    nb_path.write_text(json.dumps(nb))

    # Pre-create bank.npz to trigger the no-overwrite guard
    bank_path = tmp_path / "bank.npz"
    bank_path.touch()

    from apu_synthesizability.prep import _build_parser, main
    import os

    # Patch sys.exit to catch it
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--notebook", str(nb_path),
            "--out-dir", str(tmp_path),
        ])
    assert exc_info.value.code == 1
