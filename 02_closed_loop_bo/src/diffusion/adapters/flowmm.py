"""FlowMM adapter — Riemannian flow matching for materials (Meta FAIR).

Upstream: https://github.com/facebookresearch/flowmm  (ICML 2024)
Paper:    Miller, Sun, Sriram et al. — "FlowMM: Generating Materials with
          Riemannian Flow Matching" — arXiv 2406.04713
Install:  pip install -e git+https://github.com/facebookresearch/flowmm.git#egg=flowmm

FlowMM is a flow-matching alternative to score-based diffusion. Same level of
quality as MatterGen on MP-20 metrics, faster sampling (Euler-method ODE solve
in 50-1000 steps vs MatterGen's 1000-step SDE).

Conditioning:
    - unconditional ✓
    - chemical_system: ✗ (no per-system adapter trained)
    - properties: ✗ (FlowLLM extends with LLM-text conditioning — separate adapter)

For RL coupling (Tier 2): flow-matching uses a vector-field gradient instead
of a score-net gradient. The REINFORCE update is structurally similar to
MatterGen's; needs porting.
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class FlowMMAdapter(GeneratorAdapter):
    name = "flowmm"
    code_url = "https://github.com/facebookresearch/flowmm"
    paper_url = "https://arxiv.org/abs/2406.04713"

    def __init__(
        self,
        checkpoint: str,
        n_inference_steps: int = 250,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.n_inference_steps = int(n_inference_steps)
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            # FlowMM is now pip-installed (--no-deps) from the local clone,
            # but the actual model class lives in flowmm.cspnet or
            # flowmm.rfm_pl. We import the high-level loader.
            from flowmm.rfm.manifold_pretrained_pl import (  # type: ignore
                ManifoldPretrainedPL,
            )
            self._model = ManifoldPretrainedPL.load_from_checkpoint(
                self.checkpoint, map_location=self.device,
            )
        return self._model

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        if chemical_system is not None or property_conditions:
            raise NotImplementedError(
                "FlowMM doesn't support conditioning out-of-the-box. "
                "Use FlowLLM (separate adapter, TBD) for text-conditional generation."
            )
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        model = self._load()
        out = model.sample(
            num_samples=n,
            num_steps=self.n_inference_steps,
        )
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        return [adaptor.get_atoms(s) for s in out]

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "disordered": False,
        }
