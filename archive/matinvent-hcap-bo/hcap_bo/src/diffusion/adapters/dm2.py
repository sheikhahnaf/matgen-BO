"""DM2 adapter — Diffusion Models for Disordered Materials (Yang & Schwalbe-Koda).

Upstream: https://github.com/digital-synthesis-lab/DM2
Project:  https://kaiyang1010.github.io/DM2/
Paper:    npj ComMatSci 2025 / arXiv 2507.05024 — "A Generative Diffusion
          Model for Amorphous Materials" (Yang, Schwalbe-Koda)

Why this is in the alloy-relevant set:
    DM2 is the most mature disordered-materials generator. It generates
    AMORPHOUS atomistic structures up to 3 orders of magnitude faster than
    classical MD (cooling-rate-conditioned). Demonstrated on a-SiO₂, polymer
    melts, and **metallic glasses** — which are the extreme-disorder limit
    of alloys (no crystalline order). For RHEA work where the alloy is a
    glass-forming system or where you specifically want amorphous configs,
    DM2 is the SOTA today.

Conditioning:
    - unconditional ✓ (per system trained)
    - cooling_rate ✓ (key conditional in the paper)
    - composition: fixed per-model — re-training required per system (e.g.
      a-SiO₂ model ≠ metallic-glass model). NOT a flexible composition
      adapter; for HEA glass discovery you'd need to train DM2 on a HEA
      training set first.
    - disordered: ✓ (it's the whole point)

For RL coupling (Tier 2): The denoising score net is exposed in the codebase;
REINFORCE adaptation is straightforward but model-specific.
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class DM2Adapter(GeneratorAdapter):
    name = "dm2"
    code_url = "https://github.com/digital-synthesis-lab/DM2"
    paper_url = "https://arxiv.org/abs/2507.05024"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        cooling_rate_K_per_ps: Optional[float] = None,
        n_steps: int = 1000,
    ):
        """
        Args:
            checkpoint: per-system checkpoint (e.g. "silica", "metallic_glass").
            cooling_rate_K_per_ps: condition signal — DM2's headline conditional
                is cooling rate; lower rates produce more relaxed / lower-energy
                amorphous structures.
            n_steps: number of reverse-diffusion steps.
        """
        self.checkpoint = checkpoint
        self.device = device
        self.cooling_rate_K_per_ps = cooling_rate_K_per_ps
        self.n_steps = int(n_steps)
        self._model = None

    def _load(self):
        if self._model is None:
            from dm2.model import load_model   # type: ignore
            self._model = load_model(self.checkpoint, device=self.device)
        return self._model

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        if chemical_system is not None:
            raise NotImplementedError(
                "DM2 is per-system; use a checkpoint trained on the chemistry "
                "you want. For HEA metallic glass, train DM2 on a HEA-glass "
                "MD trajectory dataset first."
            )
        model = self._load()
        kwargs = {"num_samples": n, "n_steps": self.n_steps}
        if self.cooling_rate_K_per_ps is not None:
            kwargs["cooling_rate"] = float(self.cooling_rate_K_per_ps)
        if property_conditions:
            # DM2 supports a few extra conditions per checkpoint
            kwargs.update(property_conditions)
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        return list(model.sample(**kwargs))   # ASE Atoms

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": ["cooling_rate"],
            "disordered": True,   # amorphous output
        }
