"""AGeDi adapter — Atomistic Generative Diffusion (Hammer group, Aarhus / DTU).

Upstream: https://github.com/nronne/agedi
Docs:     https://agedi.readthedocs.io
Paper:    arXiv 2507.18314 — "Atomistic Generative Diffusion for Materials
          Modeling" (Rønne, Hammer)
License:  GPL-3.0
Install:  pip install -e git+https://github.com/nronne/agedi.git#egg=agedi

Why this is in the alloy-relevant set:
    AGeDi combines score-based diffusion for atomic POSITIONS with a
    novel CONTINUOUS-TIME DISCRETE DIFFUSION for atomic TYPES. The atomic-type
    interpolation lets you generate bimetallic clusters and 2D materials
    *beyond the training distribution*. The paper explicitly demonstrates atom-
    type interpolation that is the closest existing diffusion-native handle
    on substitutional disorder — extensible (with retraining) to HEAs.

Conditioning (per the paper):
    - unconditional ✓
    - chemical_system: ✗ (must train a per-system adapter)
    - properties: classifier-free guidance toward symmetries (e.g. specific
        space groups in 2D materials)
    - disordered: ✓ via atom-type interpolation across the training set's
        chemical systems

Domain demonstrated in the paper:
    - QCD dataset (metallic clusters): bimetallic Au-Cu, Pt-Au generation
    - C2DB dataset (2D materials): symmetry-controlled 2D crystals

For RL coupling (Tier 2): AGeDi exposes its score model directly; a REINFORCE
update is straightforward but the integration with MatInvent's RL loop has not
been ported.
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class AGeDiAdapter(GeneratorAdapter):
    name = "agedi"
    code_url = "https://github.com/nronne/agedi"
    paper_url = "https://arxiv.org/abs/2507.18314"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        guidance_scale: float = 1.0,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.guidance_scale = float(guidance_scale)
        self._model = None

    def _load(self):
        if self._model is None:
            # Lazy import — agedi has its own torch/torch_geometric stack
            from agedi.models import load_model   # type: ignore
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
            # AGeDi can interpolate types within the training distribution but
            # not enforce a specific chemical system without a custom adapter.
            raise NotImplementedError(
                "AGeDi requires per-system fine-tuning for chemical_system "
                "conditioning. Use unconditional sampling + post-hoc filter."
            )
        model = self._load()
        kwargs = {"num_samples": n, "guidance_scale": self.guidance_scale}
        if property_conditions:
            # Classifier-free guidance — passes property condition tensor
            kwargs["conditions"] = property_conditions
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        out = model.sample(**kwargs)  # returns ASE Atoms list directly
        return list(out)

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,   # not natively
            "properties": ["space_group"],   # symmetry guidance
            "disordered": True,   # atom-type interpolation = soft disorder
        }
