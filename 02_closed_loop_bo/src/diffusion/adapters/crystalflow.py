"""CrystalFlow adapter — flow-based generative model for crystals.

Upstream: https://github.com/ixsluo/CrystalFlow  (Luo et al., Nat. Comm. Oct 2025)
Paper:    arXiv:2412.11693

Sample API (per audit):
    `eval_utils.load_model(model_path, load_data=False)` returns (model, _, cfg);
    feed `SampleDataset(dataset='mp_20', total_num=N, conditions={})` through
    a `torch_geometric.loader.DataLoader`, then run `model.sample(batch, ...)`
    which returns dict-of-tensors `{frac_coords, num_atoms, atom_types, lattices}`.

Upstream packages itself as `diffcsp` (legacy from forking DiffCSP). The adapter
imports from that namespace; if upstream renames, the import falls back to
`crystalflow.*`.

Checkpoint layout: `model_path` must be a hydra-style directory containing both
the `.ckpt` file and the `hparams.yaml`. The release zip
(github.com/ixsluo/CrystalFlow/releases) extracts to such a layout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class CrystalFlowAdapter(GeneratorAdapter):
    name = "crystalflow"
    code_url = "https://github.com/ixsluo/CrystalFlow"
    paper_url = "https://arxiv.org/abs/2412.11693"

    DEFAULT_STEP_LR = 1e-5

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        n_steps: int = 100,
        train_dataset: str = "mp_20",
        batch_size: int = 64,
        step_lr: Optional[float] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.n_steps = int(n_steps)
        self.train_dataset = train_dataset
        self.batch_size = int(batch_size)
        self.step_lr = float(step_lr) if step_lr is not None else self.DEFAULT_STEP_LR
        self._model = None
        self._cfg = None

    # Back-compat: hcap_bo.yaml ships `checkpoint:`; some callers may still
    # pass it. We treat `checkpoint` as alias for `model_path`.
    @classmethod
    def _from_legacy_kwargs(cls, **kwargs):
        if "checkpoint" in kwargs and "model_path" not in kwargs:
            kwargs["model_path"] = kwargs.pop("checkpoint")
        return cls(**kwargs)

    def _ensure_repo_on_path(self):
        repo = os.environ.get(
            "CRYSTALFLOW_REPO",
            f"{os.environ.get('SCRATCH', '/tmp')}/diffusion-zoo-repos/CrystalFlow",
        )
        # Force PROJECT_ROOT — upstream hparams.yaml interpolates ${PROJECT_ROOT}
        # via OmegaConf and KeyErrors out if missing.
        os.environ["PROJECT_ROOT"] = repo
        os.environ["HYDRA_JOBS"] = str(Path(repo) / "hydra")
        os.environ.setdefault("WANDB_MODE", "offline")
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

    def _load(self):
        if self._model is None:
            self._ensure_repo_on_path()
            try:
                from eval_utils import load_model  # type: ignore
            except ImportError:
                from diffcsp.eval_utils import load_model  # type: ignore
            self._model, _, self._cfg = load_model(
                Path(self.model_path), load_data=False,
            )
            try:
                self._model = self._model.to(self.device)
            except Exception:
                pass
            self._model.eval()
        return self._model

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        if chemical_system or property_conditions:
            raise NotImplementedError(
                "CrystalFlow conditioning supported by the upstream CLI is not "
                "exposed here yet; use the unconditional DNG checkpoint."
            )
        import torch
        if seed is not None:
            torch.manual_seed(int(seed))

        self._ensure_repo_on_path()
        from generation import SampleDataset, diffusion  # type: ignore
        from torch_geometric.loader import DataLoader  # type: ignore

        model = self._load()
        ds = SampleDataset(
            dataset=self.train_dataset,
            total_num=int(n),
            conditions={},
        )
        loader = DataLoader(ds, batch_size=self.batch_size)

        sample_kwargs = {
            "step_lr": self.step_lr,
            "N": self.n_steps,
        }
        with torch.no_grad():
            frac, atypes, lattices, lengths, angles, natoms = diffusion(
                loader, model, **sample_kwargs,
            )

        return self._tensors_to_atoms(frac, atypes, lengths, angles, natoms)

    @staticmethod
    def _tensors_to_atoms(frac, atypes, lengths, angles, natoms) -> list[Atoms]:
        import numpy as np
        from ase.cell import Cell

        def _np(t):
            if hasattr(t, "detach"):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        frac = _np(frac)
        atypes = _np(atypes)
        natoms = _np(natoms).astype(int)
        lengths = _np(lengths)
        angles = _np(angles)

        atoms_list: list[Atoms] = []
        cur = 0
        for i, ni in enumerate(natoms):
            ni = int(ni)
            cell = Cell.fromcellpar([
                float(lengths[i, 0]), float(lengths[i, 1]), float(lengths[i, 2]),
                float(angles[i, 0]),  float(angles[i, 1]),  float(angles[i, 2]),
            ])
            zs = atypes[cur : cur + ni]
            zs = [int(z) if hasattr(z, "__int__") else int(z.item()) for z in zs]
            atoms_list.append(Atoms(
                numbers=zs,
                scaled_positions=frac[cur : cur + ni],
                cell=cell,
                pbc=True,
            ))
            cur += ni
        return atoms_list

    def supports(self) -> dict:
        return {
            "unconditional": True,
            "chemical_system": False,
            "properties": [],
            "disordered": False,
        }
