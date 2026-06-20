"""CrysBFN adapter — Bayesian Flow Network for crystal generation.

Upstream: https://github.com/wu-han-lin/CrysBFN  (Wu, Han et al., ICLR 2025 Spotlight)
Paper:    arXiv:2502.02016 — "A Periodic Bayesian Flow for Material Generation"

Sample API (per audit):
    `crysbfn.pl_modules.crysbfn_plmodel.CrysBFN_PL_Model.sample(
        num_samples=N, sample_steps=100, show_bar=True, **kwargs)`
returns dict-of-tensors:
    {frac_coords, num_atoms, atom_types, lengths, angles}

We convert to ASE Atoms here. Headline: 200× speedup vs diffusion (NFE=10 in paper).

Environment requirements (set in `_load`):
    PROJECT_ROOT  → repo root
    HYDRA_JOBS    → repo/hydra
    WABDB_MODE    → 'offline'  (suppresses W&B init)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class CrysBFNAdapter(GeneratorAdapter):
    name = "crysbfn"
    code_url = "https://github.com/wu-han-lin/CrysBFN"
    paper_url = "https://arxiv.org/abs/2502.02016"

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        sample_steps: int = 100,
    ):
        # Back-compat: older configs ship `checkpoint:` pointing at the
        # .ema_state_dict file; upstream's load_model expects the *dir* containing
        # both the file and `hparams.yaml`. Accept both, normalize to model_path.
        if model_path is None:
            if checkpoint is None:
                raise ValueError("CrysBFN adapter needs either `model_path` (dir) or `checkpoint` (file)")
            ck = Path(checkpoint)
            model_path = str(ck.parent if ck.suffix or ck.is_file() else ck)
        self.model_path = model_path
        self.device = device
        self.sample_steps = int(sample_steps)
        self._model = None

    def _load(self):
        if self._model is None:
            import sys
            repo = os.environ.get(
                "CRYSBFN_REPO",
                f"{os.environ.get('SCRATCH', '/tmp')}/diffusion-zoo-repos/CrysBFN",
            )
            # Force-set required env vars BEFORE importing crysbfn — its
            # __init__ calls load_dotenv() and asserts PROJECT_ROOT.
            os.environ["PROJECT_ROOT"] = repo
            os.environ["HYDRA_JOBS"] = str(Path(repo) / "hydra")
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WABDB"] = "offline"
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(Path(repo) / ".env", override=False)
            except ImportError:
                pass
            (Path(repo) / "hydra").mkdir(parents=True, exist_ok=True)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            scripts_dir = str(Path(repo) / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            # Use upstream's load_model which handles `.ema_state_dict`
            # files via Hydra-instantiate + torch.load + load_state_dict.
            from eval_utils import load_model  # type: ignore
            model, _test_loader, _cfg = load_model(
                Path(self.model_path), load_data=False, testing=False,
            )
            self._model = model
            self._model.eval()
            try:
                self._model.to(self.device)
            except Exception:
                pass
        return self._model

    @staticmethod
    def _dict_to_atoms(out: dict) -> list[Atoms]:
        """Convert CrysBFN sample-output dict → list of ASE Atoms."""
        import numpy as np
        from ase.cell import Cell

        def _np(t):
            if hasattr(t, "detach"):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        frac = _np(out["frac_coords"])
        atypes = _np(out["atom_types"])
        natoms = _np(out["num_atoms"]).astype(int)
        lengths = _np(out["lengths"])
        angles = _np(out["angles"])

        atoms_list: list[Atoms] = []
        cur = 0
        for i, n in enumerate(natoms):
            n = int(n)
            cell = Cell.fromcellpar([
                float(lengths[i, 0]), float(lengths[i, 1]), float(lengths[i, 2]),
                float(angles[i, 0]),  float(angles[i, 1]),  float(angles[i, 2]),
            ])
            zs = atypes[cur : cur + n]
            zs = [int(z) if hasattr(z, "__int__") else int(z.item()) for z in zs]
            atoms_list.append(Atoms(
                numbers=zs,
                scaled_positions=frac[cur : cur + n],
                cell=cell,
                pbc=True,
            ))
            cur += n
        return atoms_list

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        if chemical_system is not None or property_conditions:
            raise NotImplementedError(
                "CrysBFN public DNG checkpoint is unconditional only. "
                "For composition-conditioned CSP use a CSP-tagged checkpoint."
            )
        if seed is not None:
            import torch
            torch.manual_seed(int(seed))
        model = self._load()
        out = model.sample(
            num_samples=int(n),
            sample_steps=int(self.sample_steps),
            show_bar=False,
        )
        return self._dict_to_atoms(out)

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "composition_csp": True,
            "disordered": False,
            "tunable_NFE": True,
        }
