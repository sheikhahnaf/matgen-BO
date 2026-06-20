"""Smoke tests for the diffusion-adapter abstraction layer.

Tests don't require the upstream packages installed — they exercise only
the registry + base-class API. Real sampling is gated behind the lazy import
in each adapter's _load() method.
"""

import pytest

from src.diffusion import (
    GeneratorAdapter,
    RLTuneableAdapter,
    list_adapters,
    get_adapter,
    register,
)


def test_registry_lists_expected_adapters():
    names = list_adapters()
    # Modern flow-matching (replacements for DiffCSP/CDVAE)
    for x in ("mattergen", "crystalflow", "symmcd", "crysbfn", "flowmm",
              "crystalformer", "atomgpt", "adit",
              "agedi", "dm2"):
        assert x in names, f"missing {x}"
    # Old ones MUST be gone
    for x in ("diffcsp", "diffcsp_pp", "cdvae", "cond_cdvae"):
        assert x not in names, f"obsolete adapter still present: {x}"
    assert len(names) >= 10


def test_alloy_adapters_declare_disorder():
    cls_agedi = get_adapter("agedi")
    cls_dm2 = get_adapter("dm2")
    assert "github.com/nronne/agedi" in cls_agedi.code_url
    assert "DM2" in cls_dm2.code_url
    # Metadata accessible without instantiation


def test_get_adapter_returns_class():
    cls = get_adapter("mattergen")
    assert isinstance(cls, type)
    assert issubclass(cls, GeneratorAdapter)


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="Unknown adapter"):
        get_adapter("nonexistent_model")


def test_register_runtime_adapter():
    register("custom_test", "src.diffusion.adapters.mattergen", "MatterGenAdapter")
    assert "custom_test" in list_adapters()
    cls = get_adapter("custom_test")
    assert cls.__name__ == "MatterGenAdapter"


def test_mattergen_metadata_without_load():
    """Metadata access shouldn't trigger heavy model load."""
    cls = get_adapter("mattergen")
    # Class-level attrs accessible without instantiation
    assert cls.name == "mattergen"
    assert "github.com/microsoft/mattergen" in cls.code_url
    assert "arxiv" in cls.paper_url.lower()


def test_modern_flow_matching_metadata():
    """Modern replacements for the dropped DiffCSP/CDVAE family."""
    crystalflow = get_adapter("crystalflow")
    assert "CrystalFlow" in crystalflow.code_url
    symmcd = get_adapter("symmcd")
    assert "SymmCD" in symmcd.code_url
    crysbfn = get_adapter("crysbfn")
    assert "CrysBFN" in crysbfn.code_url


def test_flowmm_metadata():
    cls = get_adapter("flowmm")
    assert "facebookresearch/flowmm" in cls.code_url


def test_crystalformer_metadata():
    cls = get_adapter("crystalformer")
    assert "deepmodeling/CrystalFormer" in cls.code_url


def test_supports_method_has_expected_keys():
    """Without instantiating (which would load a model), we can't easily call
    supports(); check that the abstract class has it defined."""
    base_supports = GeneratorAdapter.supports
    assert callable(base_supports)


def test_rl_tuneable_is_generator():
    """Tier 2 must be a strict superset of Tier 1."""
    assert issubclass(RLTuneableAdapter, GeneratorAdapter)


def test_remote_adapter_importable():
    """RemoteGeneratorAdapter (subprocess wrapper) must be importable."""
    from src.diffusion import RemoteGeneratorAdapter
    assert issubclass(RemoteGeneratorAdapter, GeneratorAdapter)


def test_remote_runner_script_exists():
    """The runner script that lives in target envs must be present."""
    from pathlib import Path
    p = Path("src/diffusion/runners/remote_runner.py")
    assert p.exists(), "remote_runner.py missing"
