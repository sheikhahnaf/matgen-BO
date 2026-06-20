"""matinvent_diffusers — pluggable diffusion-generator adapters for matinvent-bo.

See docs/diffusion_adapters_design.md for the full design.
"""

from src.diffusion.base import GeneratorAdapter, RLTuneableAdapter
from src.diffusion.registry import register, get_adapter, list_adapters
from src.diffusion.remote import RemoteGeneratorAdapter

__all__ = [
    "GeneratorAdapter",
    "RLTuneableAdapter",
    "RemoteGeneratorAdapter",
    "register",
    "get_adapter",
    "list_adapters",
]
