"""CrystalFlow ModelSuite for MatInvent — Phase-3 RL coupling.

Loads the upstream CrystalFlow pretrained DNG-mp-20 checkpoint into our
`CrystalFlowModule` (which subclasses MatInvent's DiffCSPModule), and provides
sampling + dataloader integration so MatInvent's pipeline can swap MatterGen
for CrystalFlow with a single Hydra `_target_` override.

Wire via:
    configs/model/crystalflow.yaml:
        _target_: src.p3_models.crystalflow_suite.CrystalFlowSuite
        model_name: crystalflow_mp20
        model_path: ${oc.env:SCRATCH}/checkpoints/crystalflow/ckpt-v1.0.0/DNG-mp-20
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import hydra
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.suite.base import ModelSuite
from models.diffcsp.finetune import DiffCSPDataset


class CrystalFlowSampler:
    """Sample from CrystalFlow's CSPFlow.sample(); convert to MatInvent format."""

    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)

    def generate(self, model, batch_size=None, num_batches=None, **kwargs):
        """Returns (data_list, struc_list) — matches DiffCSPSampler interface.

        For CSPFlow, sample() expects a batch with a `num_atoms` distribution.
        We use the same SampleDataset upstream uses (mp_20 atom-count prior).
        """
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure
        import sys, os
        repo = os.environ.get(
            "CRYSTALFLOW_REPO",
            os.path.expandvars("$SCRATCH/diffusion-zoo-repos/CrystalFlow"),
        )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        scripts_dir = str(Path(repo) / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from generation import SampleDataset, diffusion  # type: ignore
        from torch_geometric.loader import DataLoader as PyGLoader

        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        # 4x oversample to survive MatInvent's OptFilter (validity + novel
        # + unique + stable). With CrystalFlow's DNG-mp-20, ~25% of samples
        # survive — generating 64 leaves us ~16 stable+novel+unique candidates.
        oversample_factor = int(kwargs.get("oversample_factor", 4))
        total = batch_size * num_batches * oversample_factor
        model.eval()

        ds = SampleDataset(dataset="mp_20", total_num=total, conditions={})
        loader = PyGLoader(ds, batch_size=batch_size)

        # Step LR for CFM ODE solver — paper-recommended for mp-20
        step_lr = float(kwargs.get("step_lr", 1e-5))
        n_steps = int(kwargs.get("n_steps", 100))

        all_outputs = []
        for batch in loader:
            try:
                batch = batch.to(model.device)
            except Exception:
                pass
            with torch.no_grad():
                # CSPFlow.sample(batch, step_lr, N) — flow-ODE integration
                try:
                    outputs, _ = model.sample(batch, step_lr=step_lr, N=n_steps)
                except TypeError:
                    outputs, _ = model.sample(batch, step_lr=step_lr)
            all_outputs.append({
                k: v.detach().cpu() if hasattr(v, "detach") else v
                for k, v in outputs.items()
            })

        # Concatenate batches
        frac_coords = torch.cat([o["frac_coords"] for o in all_outputs], dim=0)
        num_atoms = torch.cat([o["num_atoms"] for o in all_outputs], dim=0)
        atom_types = torch.cat([o["atom_types"] for o in all_outputs], dim=0)
        lattices = torch.cat([o["lattices"] for o in all_outputs], dim=0)

        # If atom_types is one-hot/probs (B*N, vocab), take argmax → atomic numbers
        if atom_types.ndim == 2:
            atom_types = atom_types.argmax(dim=-1) + 1

        # Lattice mat → (lengths, angles). The function lives in upstream
        # CrystalFlow's `diffcsp.common.data_utils`, NOT MatInvent's
        # `models.diffcsp.utils`. Repo already on sys.path from earlier.
        try:
            from diffcsp.common.data_utils import lattices_to_params_shape  # type: ignore
            lengths, angles = lattices_to_params_shape(lattices)
        except ImportError:
            # Fallback: compute via pymatgen (lattice mat → abc + angles)
            from pymatgen.core.lattice import Lattice as _Lat
            lengths_list, angles_list = [], []
            for i in range(lattices.shape[0]):
                lat = _Lat(lattices[i].numpy())
                lengths_list.append(list(lat.abc))
                angles_list.append(list(lat.angles))
            lengths = torch.tensor(lengths_list, dtype=torch.float32)
            angles = torch.tensor(angles_list, dtype=torch.float32)

        data_list, struc_list = [], []
        offset = torch.cumsum(num_atoms, dim=0).tolist()
        offset = [0] + offset
        for i in range(len(num_atoms)):
            _at = atom_types[offset[i]: offset[i + 1]]
            _fc = frac_coords[offset[i]: offset[i + 1]]
            _ln = lengths[i].view(1, -1)
            _ag = angles[i].view(1, -1)
            # Build Data + Structure together — APPEND BOTH OR NEITHER so the
            # two lists stay aligned. Reward-RL-update pairing depends on this.
            try:
                d = Data(
                    frac_coords=_fc,
                    atom_types=_at.long(),
                    lengths=_ln,
                    angles=_ag,
                    num_atoms=num_atoms[i],
                    num_nodes=int(num_atoms[i].item()),
                )
                lat = Lattice.from_parameters(
                    *(_ln.flatten().tolist() + _ag.flatten().tolist())
                )
                struct = Structure(
                    lattice=lat,
                    species=_at.long().numpy().tolist(),
                    coords=_fc.numpy(),
                    coords_are_cartesian=False,
                )
            except Exception:
                # Skip both — lengths stay aligned
                continue
            data_list.append(d)
            struc_list.append(struct)
        return data_list, struc_list

from src.p3_models.crystalflow_module import CrystalFlowModule


class CrystalFlowSuite(ModelSuite):
    """ModelSuite wrapping the upstream CrystalFlow checkpoint via CFM math.

    `model_path` must be a directory containing both:
        - a *.ckpt file (Lightning weights, decoder state)
        - hparams.yaml  (Hydra config used at training time)

    The DNG-mp-20 checkpoint extracted from CrystalFlow's GitHub release zip
    has this layout: $SCRATCH/checkpoints/crystalflow/ckpt-v1.0.0/DNG-mp-20/.
    """

    def __init__(
        self,
        model_name: str,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        model_path: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=model_path,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def load_model(self):
        """Instantiate CrystalFlowModule and load upstream checkpoint."""
        if self.model_path is None:
            raise ValueError(
                "CrystalFlowSuite requires `model_path` pointing at a "
                "directory containing both .ckpt and hparams.yaml "
                "(typically $SCRATCH/checkpoints/crystalflow/ckpt-v1.0.0/DNG-mp-20)"
            )

        # Make CrystalFlow's upstream `diffcsp` namespace importable so the
        # decoder hparam (`diffcsp.pl_modules.cspnet.CSPNet`) resolves to its
        # actual evolved CSPNet (which has `lattice_dim` etc that MatInvent's
        # older fork lacks). Repo is at $SCRATCH/diffusion-zoo-repos/CrystalFlow.
        import sys
        repo = os.environ.get(
            "CRYSTALFLOW_REPO",
            os.path.expandvars("$SCRATCH/diffusion-zoo-repos/CrystalFlow"),
        )
        if repo not in sys.path:
            sys.path.insert(0, repo)

        model_path = Path(os.path.abspath(self.model_path))
        with initialize_config_dir(str(model_path), version_base="1.1"):
            cfg = compose(config_name="hparams")
            # Replace the upstream Lightning-module target with ours so the
            # add_noise/calc_sample_loss/calc_kl_reg overrides take effect.
            cfg.model._target_ = "src.p3_models.crystalflow_module.CrystalFlowModule"
            # CrystalFlow's hparams reference `diffcsp.*` (their packaging) —
            # remap to MatInvent's mirror so we don't need a separate
            # `diffcsp` namespace install. Architectures are bit-identical.
            self._remap_diffcsp_targets(cfg)
            model = hydra.utils.instantiate(
                cfg.model,
                optim=cfg.optim,
                _recursive_=False,
            )

            # Pick the most-recent checkpoint
            ckpts = list(model_path.glob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No *.ckpt under {model_path}")

            # CrystalFlow release uses 'epoch=NNNN-step=...' filenames
            def _epoch(p: Path) -> int:
                try:
                    return int(p.name.split("-")[0].split("=")[1])
                except Exception:
                    return -1

            ckpt = str(sorted(ckpts, key=_epoch)[-1])
            # map_location handles CPU-only login nodes; on GPU compute nodes
            # the device cast still happens via .to() afterwards.
            map_location = "cpu" if not torch.cuda.is_available() else None
            # New Lightning requires load_from_checkpoint on the CLASS, not
            # an instance. Use the already-instantiated class for loading.
            from src.p3_models.crystalflow_module import CrystalFlowModule
            model = CrystalFlowModule.load_from_checkpoint(
                ckpt,
                hparams_file=str(model_path / "hparams.yaml"),
                strict=False,
                map_location=map_location,
            )

            # Auxiliary scalers (lattice, prop) — optional; load if present
            for fname, attr in [
                ("lattice_scaler.pt", "lattice_scaler"),
                ("prop_scaler.pt", "scaler"),
            ]:
                p = model_path / fname
                if p.exists():
                    try:
                        setattr(model, attr, torch.load(p, weights_only=False))
                    except Exception:
                        pass

        model.config = cfg
        return model

    # ------------------------------------------------------------------
    @staticmethod
    def _remap_diffcsp_targets(cfg):
        """Walk the config and rewrite `diffcsp.*` _target_ strings to
        MatInvent's mirror at `models.diffcsp.*`. Architectures are bit-
        identical because CrystalFlow upstream IS forked from DiffCSP.

        Targets we DON'T need to load (datamodule, datasets, optim) are
        skipped — only decoder/scheduler matter for inference + RL.
        """
        from omegaconf import OmegaConf as _OC, ListConfig, DictConfig as _DC

        # Only remap the top-level Lightning-module target; leave decoder
        # + scheduler resolving to upstream CrystalFlow code (we add its repo
        # to sys.path in load_model). This way the trained weights load into
        # the exact architecture they were trained on.
        REMAP = {
            "diffcsp.pl_modules.diffusion.CSPDiffusion": "src.p3_models.crystalflow_module.CrystalFlowModule",
            "diffcsp.pl_modules.diffcsppp_diffusion_w_type.CSPDiffusion": "src.p3_models.crystalflow_module.CrystalFlowModule",
        }

        def _walk(node):
            if isinstance(node, _DC):
                if "_target_" in node and node["_target_"] in REMAP:
                    node["_target_"] = REMAP[node["_target_"]]
                for k in list(node.keys()):
                    _walk(node[k])
            elif isinstance(node, ListConfig):
                for item in node:
                    _walk(item)

        _walk(cfg)

    def get_sampler(self):
        """CrystalFlow sampler — wraps CSPFlow.sample()."""
        return CrystalFlowSampler(
            batch_size=self.sample_cfg.batch_size,
            num_batches=self.sample_cfg.num_batches,
        )

    def get_dataloader(
        self,
        samples: List[Data],
        rewards: NDArray | None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ):
        """Reward-labeled dataloader for ft_step. CrystalFlow's data layout
        matches DiffCSP's PyG batch format, so we reuse the same dataset."""
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size
        dataset = DiffCSPDataset(samples, rewards)
        return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

    # ------------------------------------------------------------------
    def save_model(self, model, save_dir: str):
        """Persist current weights for resumed RL fine-tuning."""
        os.makedirs(save_dir, exist_ok=True)
        cfg = getattr(model, "config", None)
        ckpt_dict = {"state_dict": model.state_dict()}
        if cfg is not None:
            OmegaConf.save(cfg, os.path.join(save_dir, "hparams.yaml"))
        torch.save(ckpt_dict, os.path.join(save_dir, "last.ckpt"))
