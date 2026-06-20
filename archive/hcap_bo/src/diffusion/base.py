"""Abstract base classes for diffusion-generator adapters.

Tier 1 (GeneratorAdapter) — sampling-only; works for ANY diffusion model.
    Used by Phase 0 / Phase 1 / Phase 2 (open-loop BO).

Tier 2 (RLTuneableAdapter) — adds score-network gradient access for REINFORCE
    / DPO-style RL fine-tuning. Required only by Phase 3 (RL coupling).

Each adapter is a thin shim over an upstream model's sample API; conversion
to ASE Atoms is the only required output format. See `adapters/` for concrete
implementations and `docs/diffusion_adapters_design.md` for the full design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ase import Atoms


class GeneratorAdapter(ABC):
    """Tier-1 adapter: sampling only.

    Implementations MUST set:
        name      str — short identifier (used by Hydra `_target_`)
        code_url  str — canonical github
        paper_url str — arXiv / DOI

    Implementations MUST implement:
        sample(n, ...) -> list[Atoms]

    Implementations SHOULD override:
        supports() -> dict — declare which conditioning is available
        info()     -> dict — metadata for logging
    """

    # Class-level metadata; subclasses override.
    name: str = "abstract"
    code_url: str = ""
    paper_url: str = ""

    @abstractmethod
    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        """Generate `n` candidate ASE Atoms.

        Args:
            n: batch size (= eval_size × 4 in MatInvent's default).
            chemical_system: restrict to elements in this list (if model supports).
                e.g. ["V", "Cr", "Mo", "Nb", "Ta", "W"] for refractory pool.
            property_conditions: e.g. {"band_gap": 3.0} (if model has adapter).
            seed: RNG seed for reproducibility (if model supports).

        Returns:
            List of N ASE Atoms with PBC + lattice + species + positions set.
            Implementations should NOT pre-filter for validity — the upstream
            OptFilter / opt_filter pipeline handles validity / novel / unique
            / stable post-hoc.
        """

    def supports(self) -> dict:
        """Conditioning capabilities of this adapter.

        Returns dict with keys:
            unconditional      bool
            chemical_system    bool
            properties         list[str] — supported property names
            disordered         bool — native fractional occupancy
        """
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "disordered": False,
        }

    def info(self) -> dict:
        return {
            "name": self.name,
            "code_url": self.code_url,
            "paper_url": self.paper_url,
            "supports": self.supports(),
        }

    def __repr__(self) -> str:
        s = self.supports()
        flags = ",".join(k for k, v in s.items() if v and k != "properties")
        return f"<{self.__class__.__name__}({self.name}) supports={flags}>"


class RLTuneableAdapter(GeneratorAdapter):
    """Tier-2 adapter: encapsulates a paradigm-specific RL fine-tune step.

    Used by Phase-3 (RL coupling). The KEY abstraction is `rl_finetune_step`:
    the entire paradigm-specific RL update (noise schedule, loss formulation,
    KL primitive, backprop) lives inside the adapter so MatInvent's pipeline
    stays paradigm-agnostic.

    See `docs/RL_HOOK_DESIGN.md` for the full design rationale and
    paradigm-by-paradigm pseudo-code.
    """

    @abstractmethod
    def get_dataloader(self, samples, rewards, batch_size: int):
        """Return a DataLoader yielding reward-labeled batches for RL."""

    @abstractmethod
    def rl_finetune_step(
        self,
        prior_adapter: "RLTuneableAdapter",
        batch,
        cfg,
        optimizer,
    ) -> dict:
        """One paradigm-specific RL fine-tune update on `batch`.

        Required behavior:
            1. Compute generative-policy loss weighted by batch.reward.
            2. Compute KL regularizer vs `prior_adapter`.
            3. Combine, backward, optimizer.step().
            4. Return loss dict (must include 'loss', 'loss_diff', 'loss_kl').
        """

    @abstractmethod
    def parameters(self):
        """Iterable of trainable parameters (for optimizer construction)."""

    @abstractmethod
    def to(self, device):
        """Move underlying network to device. Returns self."""

    @abstractmethod
    def save(self, ckpt_dir: str):
        """Save model weights to `ckpt_dir`."""

    @classmethod
    @abstractmethod
    def load(cls, ckpt_path: str, **kwargs) -> "RLTuneableAdapter":
        """Load model weights from `ckpt_path`."""
