"""ADiT adapter — All-Atom Diffusion Transformer (Meta FAIR, ICML 2025).

Upstream: https://github.com/facebookresearch/all-atom-diffusion-transformer
Paper:    arXiv:2503.03965

Two modes (pick via constructor kwargs):
    1. `pregenerated_zip_path` (recommended) — uses the upstream-shipped
       10k pre-generated MP-20 crystals (HF: chaitjo/all-atom-diffusion-transformer
       /resolve/main/ADiT_crystals_mp20.zip). Extracts CIFs once and serves them
       as a static pool — no inference, no model load.
    2. `checkpoint` — full inference path via the upstream Hydra Lightning
       evaluator. Significantly heavier; usually unnecessary because the
       pre-generated 10k pool covers Phase-2's 64×10=640 cycle budget.

Returns:
    list[ase.Atoms]
"""

from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path
from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class ADiTAdapter(GeneratorAdapter):
    name = "adit"
    code_url = "https://github.com/facebookresearch/all-atom-diffusion-transformer"
    paper_url = "https://arxiv.org/abs/2503.03965"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        pregenerated_zip_path: Optional[str] = None,
        extract_dir: Optional[str] = None,
        device: str = "cuda",
        n_steps: int = 250,
    ):
        self.checkpoint = checkpoint
        self.pregenerated_zip_path = pregenerated_zip_path
        self.device = device
        self.n_steps = int(n_steps)
        self._cif_paths: Optional[list[Path]] = None

        if pregenerated_zip_path is None and checkpoint is not None:
            zip_default = Path(checkpoint).parent / "ADiT_crystals_mp20.zip"
            if zip_default.exists():
                self.pregenerated_zip_path = str(zip_default)

        if extract_dir is None and self.pregenerated_zip_path:
            self.extract_dir = str(Path(self.pregenerated_zip_path).with_suffix(""))
        else:
            self.extract_dir = extract_dir

    def _ensure_pool_extracted(self) -> list[Path]:
        if self._cif_paths is not None:
            return self._cif_paths
        if not self.pregenerated_zip_path:
            raise RuntimeError(
                "ADiT pre-generated mode requires `pregenerated_zip_path`; "
                "full-inference path not yet implemented (TODO: hydra-lightning)."
            )
        zip_path = Path(self.pregenerated_zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"Pre-generated zip missing: {zip_path}")
        out = Path(self.extract_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _real_cifs(root: Path) -> list[Path]:
            # Filter out AppleDouble / macOS metadata sidecar files that share
            # the .cif extension but aren't real CIFs (would cause ase.io.read
            # to drop those samples silently).
            return sorted(
                p for p in root.rglob("*.cif")
                if "__MACOSX" not in p.parts and not p.name.startswith("._")
            )

        cifs = _real_cifs(out)
        if len(cifs) == 0:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out)
            cifs = _real_cifs(out)
        if len(cifs) == 0:
            raise RuntimeError(f"No CIFs found after unzip into {out}")
        self._cif_paths = cifs
        return cifs

    def sample(self, n=64, chemical_system=None, property_conditions=None, seed=None):
        if chemical_system or property_conditions:
            raise NotImplementedError("ADiT pre-generated pool is unconditional only.")
        cifs = self._ensure_pool_extracted()
        rng = random.Random(seed if seed is not None else 0)
        chosen = rng.sample(cifs, k=min(int(n), len(cifs))) if n < len(cifs) else cifs[:n]

        from ase.io import read as ase_read
        out: list[Atoms] = []
        for p in chosen:
            try:
                a = ase_read(str(p))
                if isinstance(a, list):
                    out.extend(a)
                else:
                    out.append(a)
            except Exception as e:
                print(f"[adit] warning: failed to read {p.name}: {e}")
        return out[:n]

    def supports(self):
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "disordered": False,
            "non_periodic": True,
            "pregenerated_pool": self.pregenerated_zip_path is not None,
        }
