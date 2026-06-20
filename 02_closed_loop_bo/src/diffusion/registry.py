"""Name → adapter-class registry. Lazy-imports each adapter so the registry
listing doesn't pull heavy upstream packages at import time.

Usage:
    from src.diffusion import get_adapter
    Adapter = get_adapter("mattergen")
    gen = Adapter(checkpoint="pretrained-uncond", device="cuda")
    atoms = gen.sample(n=64)

To use an adapter that lives in a different conda env, wrap with
RemoteGeneratorAdapter (subprocess + extxyz IPC):
    from src.diffusion import RemoteGeneratorAdapter
    gen = RemoteGeneratorAdapter(
        env_prefix=f"{os.environ['SCRATCH']}/envs/mat-zoo-modern",
        model_name="crystalflow",
        adapter_kwargs={"checkpoint": "...", "device": "cuda"},
    )
"""

from __future__ import annotations

import importlib

# name -> (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    # ----- Tier 1 + Tier 2 (RL-ready) ---------------------------------
    "mattergen":      ("src.diffusion.adapters.mattergen",      "MatterGenAdapter"),

    # ----- Modern flow-matching crystal generators (recommended set) --
    "crystalflow":    ("src.diffusion.adapters.crystalflow",    "CrystalFlowAdapter"),
    "symmcd":         ("src.diffusion.adapters.symmcd",         "SymmCDAdapter"),
    "crysbfn":        ("src.diffusion.adapters.crysbfn",        "CrysBFNAdapter"),
    "flowmm":         ("src.diffusion.adapters.flowmm",         "FlowMMAdapter"),

    # ----- Modern Transformer / autoregressive generators -------------
    "crystalformer":  ("src.diffusion.adapters.crystalformer",  "CrystalFormerAdapter"),
    "atomgpt":        ("src.diffusion.adapters.atomgpt",        "AtomGPTAdapter"),
    "adit":           ("src.diffusion.adapters.adit",           "ADiTAdapter"),

    # ----- Alloy / disorder-capable -----------------------------------
    "agedi":          ("src.diffusion.adapters.agedi",          "AGeDiAdapter"),
    "dm2":            ("src.diffusion.adapters.dm2",            "DM2Adapter"),

    # NB: The older DiffCSP / DiffCSP++ / CDVAE / Cond-CDVAE adapters were
    # removed because their upstream stacks (Python 3.8, torch 1.9-1.10) no
    # longer coexist with modern PyG / e3nn / fairchem and offer no improvement
    # over their successors above (CrystalFlow / SymmCD / CrysBFN).
}


def list_adapters() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_adapter(name: str):
    """Return the adapter class (not instance). Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Known: {list_adapters()}")
    module_path, class_name = _REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def register(name: str, module_path: str, class_name: str) -> None:
    """Register a new adapter at runtime (for user-defined plugins)."""
    _REGISTRY[name] = (module_path, class_name)
