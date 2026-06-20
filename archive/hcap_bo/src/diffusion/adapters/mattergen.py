"""MatterGen adapter — reference Tier-1 + Tier-2 implementation.

Upstream: https://github.com/microsoft/mattergen  (Microsoft Research, Nature 2025)
Paper:    https://arxiv.org/abs/2312.03687
Install:  pip install mattergen   (already present in matinvent-hcap-bo env)

Conditioning available out-of-the-box:
    - unconditional ✓
    - chemical_system (via per-system adapter, downloaded from HF)
    - properties: band_gap, ml_bulk_modulus, dft_mag_density, hhi_score,
                  energy_above_hull, space_group, chemical_system
                  (each via a separately-trained adapter)

For RL fine-tuning (Tier 2), MatInvent's existing reward-weighted policy-
gradient pipeline (`pipeline.mat_invent.MatInvent.ft_step`) is the reference
implementation. We expose the same hooks here so other adapters can mirror
the API.
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import RLTuneableAdapter


class MatterGenAdapter(RLTuneableAdapter):
    name = "mattergen"
    code_url = "https://github.com/microsoft/mattergen"
    paper_url = "https://arxiv.org/abs/2312.03687"

    def __init__(
        self,
        checkpoint: str = "pretrained-uncond",
        device: str = "cuda",
        adapter: Optional[str] = None,   # e.g. "chemical_system"
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.adapter = adapter
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # Tier 1 — sampling
    # ------------------------------------------------------------------

    def _load(self):
        if self._model is None:
            from mattergen.suite import MatterGenSuite
            self._model = MatterGenSuite(
                model_name=self.checkpoint,
                device=self.device,
            )
        return self._model

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        model = self._load()
        kwargs = {}
        if chemical_system is not None:
            kwargs["chemical_system"] = "-".join(sorted(chemical_system))
        if property_conditions:
            kwargs.update(property_conditions)
        if seed is not None:
            kwargs["seed"] = int(seed)
        out = model.sample(batch_size=n, num_batches=1, **kwargs)
        # MatterGenSuite returns pymatgen Structure → convert to ASE Atoms
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        return [adaptor.get_atoms(s) for s in out]

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": True,
            "properties": [
                "band_gap", "ml_bulk_modulus", "dft_mag_density",
                "hhi_score", "energy_above_hull", "space_group",
            ],
            "disordered": False,
        }

    # ------------------------------------------------------------------
    # Tier 2 — RL hooks (delegates to MatInvent's pipeline)
    # ------------------------------------------------------------------

    def score_log_prob(self, atoms_list, t: float):
        raise NotImplementedError(
            "MatterGen RL is wired through matinvent.pipeline.mat_invent — "
            "use the upstream RL loop with this adapter as the generator."
        )

    def update_weights(self, advantages, atoms_list, **opt_kwargs) -> dict:
        raise NotImplementedError(
            "Use matinvent.pipeline.mat_invent.MatInvent.ft_step for RL update."
        )
