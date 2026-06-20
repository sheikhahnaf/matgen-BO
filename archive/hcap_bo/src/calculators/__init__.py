"""Drop-in calculator overrides for the upstream MatInvent RL pipeline.

Each class here mimics the interface of `rewards.calculators.<X>` from the
pristine matinvent codebase, with the same constructor signature so it can be
swapped via a Hydra `_target_` override:

    python main.py reward=heat_capacity \\
        reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN

Pristine matinvent code is NEVER edited; the only behaviour change is which
class Hydra instantiates when it builds the reward graph.
"""

from src.calculators.local_esen import LocalESEN
from src.calculators.local_esen_gp_routed import LocalESEN_GPRouted
from src.calculators.local_esen_gp_routed_v3 import LocalESEN_GPRoutedV3
from src.calculators.local_esen_gp_routed_v4 import LocalESEN_GPRoutedV4
from src.calculators.local_esen_bm import LocalESEN_BM
from src.calculators.local_esen_bm_gp_routed_v4 import LocalESEN_BM_GPRoutedV4

__all__ = [
    "LocalESEN", "LocalESEN_GPRouted", "LocalESEN_GPRoutedV3", "LocalESEN_GPRoutedV4",
    "LocalESEN_BM", "LocalESEN_BM_GPRoutedV4",
]
