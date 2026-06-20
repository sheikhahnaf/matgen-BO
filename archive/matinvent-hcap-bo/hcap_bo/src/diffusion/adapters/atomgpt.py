"""AtomGPT adapter — Atomistic Generative Pretrained Transformer (NIST).

Upstream: https://github.com/atomgptlab/atomgpt
Paper:    JPCL 2024 [10.1021/acs.jpclett.4c01126] — "AtomGPT: Atomistic
          Generative Pretrained Transformer for Forward and Inverse Materials Design"
Install:  pip install atomgpt   (or pip install -e git+https://github.com/atomgptlab/atomgpt.git)

Decoder-only transformer over SLICES (Simplified Line-Input Crystal-Encoding
System) — generates crystal structures as token strings. Strong for natural-
language-style property prompting; trivially supports REINFORCE since the
log-prob is the standard token-level autoregressive log-prob.

Conditioning:
    - unconditional ✓
    - text-style property prompts ✓ (e.g. "formation_energy < -2 eV/atom")
"""

from __future__ import annotations

from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class AtomGPTAdapter(GeneratorAdapter):
    name = "atomgpt"
    code_url = "https://github.com/atomgptlab/atomgpt"
    paper_url = "https://pubs.acs.org/doi/10.1021/acs.jpclett.4c01126"

    def __init__(
        self,
        checkpoint: str = "atomgpt-mp20",
        device: str = "cuda",
        temperature: float = 0.8,
        max_tokens: int = 256,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._model = None

    def _load(self):
        if self._model is None:
            from atomgpt.inverse_models.inverse_models import AtomGPT  # type: ignore
            self._model = AtomGPT.from_pretrained(self.checkpoint, device=self.device)
        return self._model

    def sample(self, n=64, chemical_system=None, property_conditions=None, seed=None):
        if seed is not None:
            import torch; torch.manual_seed(int(seed))
        prompt = ""
        if property_conditions:
            prompt = ",".join(f"{k}={v}" for k, v in property_conditions.items())
        out = self._load().generate(
            num_samples=n,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        from pymatgen.io.ase import AseAtomsAdaptor
        return [AseAtomsAdaptor().get_atoms(s) for s in out]

    def supports(self):
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": ["formation_energy", "band_gap"],   # via text prompts
            "disordered": False,
        }
