"""Tests for the comparison panel figure (panel.py)."""
import os

import numpy as np
import pandas as pd
import pytest


def test_make_panel_creates_png(tmp_path):
    """make_panel must create a non-trivial PNG file without errors."""
    from apu_synthesizability.panel import make_panel

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "target": rng.choice(["cp", "bm"], n),
            "backbone_name": rng.choice(["MatterGen", "CrystalFlow", "ADiT"], n),
            "policy": rng.choice(["BASE", "ACC"], n),
            "seed": rng.integers(0, 120, n),
            "apu_score": rng.random(n),
            "cgnf_score": rng.random(n),
        }
    )
    csv = tmp_path / "scored.csv"
    df.to_csv(csv, index=False)

    out = tmp_path / "panel.png"
    p = make_panel(str(csv), str(out))

    assert os.path.isfile(p), f"Expected PNG at {p}, but file not found."
    assert os.path.getsize(p) > 5000, (
        f"PNG at {p} is suspiciously small ({os.path.getsize(p)} bytes). "
        "Was the figure actually rendered?"
    )


def test_make_panel_returns_out_path(tmp_path):
    """Return value must equal the out_png argument."""
    from apu_synthesizability.panel import make_panel

    rng = np.random.default_rng(1)
    n = 20
    df = pd.DataFrame(
        {
            "target": rng.choice(["cp", "bm"], n),
            "backbone_name": rng.choice(["MatterGen", "CrystalFlow", "ADiT"], n),
            "policy": rng.choice(["BASE", "ACC"], n),
            "seed": rng.integers(0, 120, n),
            "apu_score": rng.random(n),
            "cgnf_score": rng.random(n),
        }
    )
    csv = tmp_path / "scored2.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "panel2.png"

    result = make_panel(str(csv), str(out))
    assert result == str(out)


def test_make_panel_single_backbone(tmp_path):
    """Should not crash when only one backbone is present."""
    from apu_synthesizability.panel import make_panel

    rng = np.random.default_rng(2)
    n = 15
    df = pd.DataFrame(
        {
            "target": ["bm"] * n,
            "backbone_name": ["MatterGen"] * n,
            "policy": rng.choice(["BASE", "ACC"], n),
            "seed": rng.integers(0, 60, n),
            "apu_score": rng.random(n),
            "cgnf_score": rng.random(n),
        }
    )
    csv = tmp_path / "single.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "single_panel.png"

    p = make_panel(str(csv), str(out))
    assert os.path.isfile(p) and os.path.getsize(p) > 5000
