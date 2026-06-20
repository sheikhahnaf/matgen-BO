"""Unit tests for src/oracle.py — Path.with_name fix + class hierarchy."""

import tempfile
from pathlib import Path

import pytest

from src.oracle import (
    HCapOracle_eSEN, HCapOracle_ORB, HCapOracle_UMA, get_oracle,
)


def test_runner_relpath_resolves_correctly():
    """Committee fix: the slash in 'runners/run_esen.py' must NOT raise ValueError
    via Path.with_name() — we now use parent / runner_relpath."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    for cls, name in [
        (HCapOracle_eSEN, "run_esen.py"),
        (HCapOracle_ORB,  "run_orb_phonon.py"),
        (HCapOracle_UMA,  "run_uma_phonon.py"),
    ]:
        # Use a real env_prefix path; we just want to verify the runner path build.
        with tempfile.TemporaryDirectory() as env_tmp:
            try:
                obj = cls(env_prefix=env_tmp)
                expected = src_dir / "runners" / name
                assert Path(obj.runner_script).resolve() == expected.resolve(), \
                    f"Wrong runner for {cls.__name__}: got {obj.runner_script}"
            except FileNotFoundError as e:
                msg = str(e)
                # Acceptable: the env exists but maybe runners/ missing in test layout.
                if "Runner script missing" in msg:
                    expected = src_dir / "runners" / name
                    assert str(expected) in msg, f"Wrong path in error: {msg}"


def test_get_oracle_factory():
    with tempfile.TemporaryDirectory() as tmp:
        for kind, cls in [("esen", HCapOracle_eSEN), ("orb", HCapOracle_ORB),
                          ("uma", HCapOracle_UMA), ("fairchem", HCapOracle_eSEN)]:
            obj = get_oracle(kind=kind, env_prefix=tmp)
            assert isinstance(obj, cls)


def test_get_oracle_unknown_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="Unknown oracle kind"):
            get_oracle(kind="banana", env_prefix=tmp)


def test_missing_env_prefix_raises():
    with pytest.raises(FileNotFoundError, match="Env prefix missing"):
        HCapOracle_eSEN(env_prefix="/this/path/definitely/does/not/exist")
