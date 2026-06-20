"""Smoke tests: imports, package structure. Real tests added in Phase 0."""
import importlib


def test_package_importable():
    pkg = importlib.import_module("src")
    assert pkg.__version__ == "0.0.1"


def test_stubs_present():
    for mod in [
        "src.featurizer",
        "src.surrogate",
        "src.acquisition",
        "src.oracle_fairchem",
        "src.ltm",
        "src.calibration",
        "src.cli",
    ]:
        # importing each stub raises NotImplementedError at top level — that
        # is the intentional "not yet implemented" signal. Wrap in try/except.
        try:
            importlib.import_module(mod)
        except NotImplementedError:
            pass
        except Exception as e:
            raise AssertionError(f"{mod} failed unexpectedly: {e}") from e
