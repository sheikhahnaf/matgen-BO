"""SymmCD adapter — Symmetry-Preserving Crystal Generation with Diffusion.

Upstream: https://github.com/sibasmarak/SymmCD  (Levy, Panigrahi et al., ICLR 2025)
Paper:    arXiv:2502.03638
Install:  pip install -e git+https://github.com/sibasmarak/SymmCD.git#egg=symmcd

Replaces DiffCSP. Decomposes crystals into asymmetric unit + symmetry
transformations and learns their joint distribution via diffusion. Strong
symmetry-aware generation across all 230 space groups.

Conditioning:
    - unconditional ✓
    - chemical_system: ✗
    - properties: ✗
    - space_group: ✓ (intrinsic to model)
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class SymmCDAdapter(GeneratorAdapter):
    name = "symmcd"
    code_url = "https://github.com/sibasmarak/SymmCD"
    paper_url = "https://arxiv.org/abs/2502.03638"

    def __init__(
        self,
        checkpoint: str,
        space_group: Optional[int] = None,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint
        self.space_group = space_group
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            # SymmCD upstream has no pyproject build-system; we use it via
            # PYTHONPATH-style sys.path injection. The adapter expects the
            # repo to be cloned at $SCRATCH/diffusion-zoo-repos/SymmCD.
            import sys, os
            repo = os.environ.get(
                "SYMMCD_REPO",
                f"{os.environ.get('SCRATCH', '/tmp')}/diffusion-zoo-repos/SymmCD",
            )
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from symmcd.sample import load_model  # type: ignore
            self._model = load_model(self.checkpoint, device=self.device)
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
                "SymmCD conditioning beyond space_group not implemented; "
                "use unconditional + post-hoc filter."
            )
        model = self._load()
        kwargs = {"num_samples": n}
        if self.space_group is not None:
            kwargs["space_group"] = int(self.space_group)
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        out = model.generate(**kwargs)
        from pymatgen.io.ase import AseAtomsAdaptor
        return [AseAtomsAdaptor().get_atoms(s) for s in out]

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "space_group": True,
            "disordered": False,
        }
