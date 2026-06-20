"""CrystalFormer adapter — autoregressive JAX Transformer (deepmodeling).

Upstream: https://github.com/deepmodeling/CrystalFormer
Paper:    arXiv 2403.15734

Sample API (per audit): subprocess into upstream `main.py`:
    python main.py --optimizer none --restore_path <ckpt.pkl> \\
        --num_samples N --top_p 1.0 --temperature 1.0 --K 30 \\
        --output_filename <out.csv>
Then parse the CSV via the repo's `scripts/awl2struct.py`.

Why subprocess: CrystalFormer is JAX-based and tightly coupled to
optax+haiku training scaffolding. Calling its functions in-process from a
PyTorch env risks XLA/CUDA driver clashes. Subprocess gives a clean boundary.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class CrystalFormerAdapter(GeneratorAdapter):
    name = "crystalformer"
    code_url = "https://github.com/deepmodeling/CrystalFormer"
    paper_url = "https://arxiv.org/abs/2403.15734"

    def __init__(
        self,
        checkpoint: str,
        space_group: Optional[int] = None,
        device: str = "cuda",
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 30,
    ):
        self.checkpoint = checkpoint
        self.space_group = space_group
        self.device = device
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def _repo_root(self) -> Path:
        return Path(os.environ.get(
            "CRYSTALFORMER_REPO",
            f"{os.environ.get('SCRATCH', '/tmp')}/diffusion-zoo-repos/CrystalFormer",
        ))

    def _csv_to_atoms(self, csv_path: Path) -> list[Atoms]:
        """Use upstream's awl2struct.py to convert CSV → pymatgen → ASE."""
        import sys
        repo = self._repo_root()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from scripts.awl2struct import awl2struct  # type: ignore
        except Exception:
            from awl2struct import awl2struct  # type: ignore

        structs = awl2struct(str(csv_path))
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        out: list[Atoms] = []
        for s in structs:
            try:
                out.append(adaptor.get_atoms(s))
            except Exception as e:
                print(f"[crystalformer] skip bad struct: {e}")
        return out

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        if property_conditions:
            raise NotImplementedError("CrystalFormer has no property adapter.")

        repo = self._repo_root()
        main_py = repo / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(f"CrystalFormer main.py missing at {main_py}")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "samples.csv"
            cmd = [
                "python", str(main_py),
                "--optimizer", "none",
                "--restore_path", self.checkpoint,
                "--num_samples", str(int(n)),
                "--top_p", str(self.top_p),
                "--temperature", str(self.temperature),
                "--K", str(self.top_k),
                "--output_filename", str(out_csv),
            ]
            if self.space_group is not None:
                cmd += ["--spacegroup", str(int(self.space_group))]
            if chemical_system:
                formula = "".join(chemical_system)
                cmd += ["--formula", formula]
            if seed is not None:
                cmd += ["--seed", str(int(seed))]

            env = {**os.environ}
            env.setdefault("JAX_PLATFORMS", "cuda" if self.device == "cuda" else "cpu")

            print(f"[crystalformer] running: {' '.join(cmd[:6])} ...")
            proc = subprocess.run(
                cmd, cwd=str(repo), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900,
            )
            if proc.returncode != 0 or not out_csv.exists():
                raise RuntimeError(
                    f"CrystalFormer subprocess failed (code {proc.returncode}). "
                    f"stderr (tail):\n{proc.stderr.decode()[-1500:]}"
                )
            return self._csv_to_atoms(out_csv)

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": True,
            "properties": [],
            "space_group": True,
            "disordered": False,
        }
