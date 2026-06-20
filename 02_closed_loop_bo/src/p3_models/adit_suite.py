"""ADiT ModelSuite for MatInvent — Phase-3 RL coupling.

Loads two checkpoints:
  - vae.ckpt  → frozen autoencoder (VAE)
  - ldm.ckpt  → trainable denoiser (DiT in latent space) — fine-tuned via REINFORCE

Wire via:
    configs/model/adit.yaml:
        _target_: src.p3_models.adit_suite.ADiTSuite
        vae_checkpoint: ${oc.env:SCRATCH}/checkpoints/adit/vae.ckpt
        dit_checkpoint: ${oc.env:SCRATCH}/checkpoints/adit/ldm.ckpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.suite.base import ModelSuite


class ADiTSampler:
    """Sample from ADiT's sample_and_decode; convert to MatInvent format.

    For RL fine-tuning we need the agent to produce ASE-style structures
    that we then convert into PyG batches matching ADiT's training schema.
    """

    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)

    # MP-20 atom-count empirical distribution (index = num_atoms; from DiffCSP).
    # Used as `num_nodes_bincount` for ADiT's sample_and_decode.
    MP20_NUM_NODES_DIST = [
        0.0, 0.0021742334905660377, 0.021079009433962265, 0.019826061320754717,
        0.15271226415094338, 0.047132959905660375, 0.08464770047169812,
        0.021079009433962265, 0.07808814858490566, 0.03434551886792453,
        0.0972877358490566, 0.013303360849056603, 0.09669811320754718,
        0.02155807783018868, 0.06522700471698113, 0.014372051886792452,
        0.06703272405660378, 0.00972877358490566, 0.053176591981132074,
        0.010576356132075472, 0.08995430424528301,
    ]

    def generate(self, model, batch_size=None, num_batches=None, **kwargs):
        from pymatgen.core.lattice import Lattice
        from pymatgen.core.structure import Structure
        from torch_geometric.data import Data

        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        # 4x oversample so survivors of OptFilter (validity+novel+unique+stable) > 0.
        oversample_factor = int(kwargs.get("oversample_factor", 4))
        total = batch_size * num_batches * oversample_factor
        model.eval()

        # Upstream signature (per ldm_module.py:587):
        #   sample_and_decode(num_nodes_bincount, spacegroups_bincount,
        #                     batch_size, cfg_scale=4.0, dataset_idx=0)
        # RETURNS: (out, batch, samples) — a 3-tuple, NOT a list of structures.
        #   - out: Dict[atom_types(logits), pos(nm), frac_coords, lengths(scaled), angles(rad)]
        #   - batch["num_atoms"]: per-crystal atom counts used to slice `out`
        num_nodes = torch.tensor(self.MP20_NUM_NODES_DIST, dtype=torch.float32)
        cfg_scale = float(kwargs.get("cfg_scale", 4.0))
        with torch.no_grad():
            out, decoded_batch, _samples = model.sample_and_decode(
                num_nodes_bincount=num_nodes,
                spacegroups_bincount=None,
                batch_size=total,
                cfg_scale=cfg_scale,
                dataset_idx=0,
            )

        # Per-crystal slicing follows upstream ldm_module.py:531.
        struc_list, data_list = [], []
        start_idx = 0
        for idx_in_batch, num_atom in enumerate(decoded_batch["num_atoms"].tolist()):
            try:
                _atom_types = out["atom_types"].narrow(0, start_idx, num_atom).argmax(dim=1)
                _atom_types = torch.where(_atom_types == 0, torch.ones_like(_atom_types), _atom_types)
                _pos_A = out["pos"].narrow(0, start_idx, num_atom) * 10.0  # nm → Å
                _frac = out["frac_coords"].narrow(0, start_idx, num_atom)
                _lengths_unscaled = out["lengths"][idx_in_batch] * float(num_atom) ** (1.0 / 3.0)
                _angles_deg = torch.rad2deg(out["angles"][idx_in_batch])

                lengths_np = _lengths_unscaled.detach().cpu().numpy().astype(float)
                angles_np = _angles_deg.detach().cpu().numpy().astype(float)
                lat = Lattice.from_parameters(*(list(lengths_np) + list(angles_np)))
                struct = Structure(
                    lattice=lat,
                    species=_atom_types.detach().cpu().numpy().tolist(),
                    coords=_frac.detach().cpu().numpy(),
                    coords_are_cartesian=False,
                )

                cell = torch.tensor(lat.matrix, dtype=torch.float32).unsqueeze(0)
                lengths_t = torch.tensor(lengths_np, dtype=torch.float32).view(1, -1)
                angles_t = torch.tensor(angles_np, dtype=torch.float32).view(1, -1)
                atoms_t = _atom_types.detach().cpu().long()
                frac_t = _frac.detach().cpu().float()
                d_pyg = Data(
                    atom_types=atoms_t,
                    frac_coords=frac_t,
                    pos=_pos_A.detach().cpu().float(),
                    cell=cell,
                    lattices=cell.clone(),
                    lattices_scaled=cell.clone() / max(num_atom, 1) ** (1/3),
                    lengths=lengths_t,
                    lengths_scaled=lengths_t / max(num_atom, 1) ** (1/3),
                    angles=angles_t,
                    angles_radians=angles_t * (np.pi / 180.0),
                    num_atoms=torch.LongTensor([num_atom]),
                    num_nodes=int(num_atom),
                    token_idx=torch.arange(num_atom),
                    dataset_idx=torch.LongTensor([0]),
                    spacegroup=torch.LongTensor([0]),
                )
            except Exception:
                start_idx += num_atom
                continue
            data_list.append(d_pyg)
            struc_list.append(struct)
            start_idx += num_atom
        return data_list, struc_list

# Add ADiT upstream to sys.path before importing
_REPO = os.environ.get(
    "ADIT_REPO",
    os.path.expandvars("$SCRATCH/diffusion-zoo-repos/all-atom-diffusion-transformer"),
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class ADiTSuite(ModelSuite):
    """ModelSuite wrapping ADiT (VAE + LatentDiffusion DiT).

    For Phase-3 RL the VAE is loaded once and frozen; only the DiT denoiser
    receives gradient updates from MatInvent's ft_step.
    """

    def __init__(
        self,
        model_name: str,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        vae_checkpoint: str | None = None,
        dit_checkpoint: str | None = None,
        config_overrides: list[str] = [],
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            model_path=None,
            config_overrides=config_overrides,
            device=device,
            **kwargs,
        )
        self.vae_checkpoint = vae_checkpoint
        self.dit_checkpoint = dit_checkpoint

    # ------------------------------------------------------------------
    def load_model(self):
        """Instantiate ADiTModule and load both VAE + DiT checkpoints."""
        if self.dit_checkpoint is None or not Path(self.dit_checkpoint).exists():
            raise FileNotFoundError(
                f"ADiT DiT checkpoint missing: {self.dit_checkpoint}. "
                "Download via `huggingface-cli download chaitjo/all-atom-diffusion-transformer ldm.ckpt`."
            )
        if self.vae_checkpoint is None or not Path(self.vae_checkpoint).exists():
            raise FileNotFoundError(
                f"ADiT VAE checkpoint missing: {self.vae_checkpoint}. "
                "Download via `huggingface-cli download chaitjo/all-atom-diffusion-transformer vae.ckpt`."
            )

        # Lightning's load_from_checkpoint will auto-instantiate the
        # autoencoder + denoiser nested modules from the saved hparams.
        from src.p3_models.adit_module import ADiTModule

        # The released ADiT checkpoint was pickled when upstream's package
        # was named `src.*` (we renamed it to `adit.*` to avoid colliding
        # with our project). Alias the legacy module paths so unpickle
        # finds classes correctly without renaming the upstream tree back.
        import adit
        import sys as _sys
        _sys.modules.setdefault("src", adit)
        for sub in ("src.models", "src.data", "src.utils", "src.tools", "src.eval"):
            tail = sub[len("src."):]
            try:
                mod = __import__(f"adit.{tail}", fromlist=[tail])
                _sys.modules.setdefault(sub, mod)
            except ImportError:
                pass

        map_location = "cpu" if not torch.cuda.is_available() else None
        # Override the embedded autoencoder_ckpt path (it points to the
        # original training-time path on chaitjo's filesystem, which
        # doesn't exist here). Lightning's load_from_checkpoint accepts
        # **kwargs that override saved hparams.
        model = ADiTModule.load_from_checkpoint(
            self.dit_checkpoint,
            strict=False,
            map_location=map_location,
            autoencoder_ckpt=self.vae_checkpoint,
        )

        # The autoencoder was already loaded inside ADiTModule.__init__
        # via autoencoder_ckpt=vae_checkpoint, so no further VAE-load step.

        # Freeze VAE parameters — REINFORCE only updates the DiT denoiser
        for p in model.autoencoder.parameters():
            p.requires_grad = False
        model.autoencoder.eval()

        return model

    # ------------------------------------------------------------------
    def get_sampler(self):
        """Wrap ADiT's sample_and_decode into MatInvent's (data_list, struc_list) format."""
        return ADiTSampler(
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
        """Reward-labeled DataLoader for ft_step. ADiT uses standard PyG."""
        if batch_size is None:
            batch_size = self.finetune_cfg.batch_size

        # Augment each Data with reward attribute
        for i, d in enumerate(samples):
            if rewards is not None:
                d.reward = torch.tensor([float(rewards[i])])

        from torch_geometric.data import Batch as _Batch
        return DataLoader(samples, batch_size=batch_size, shuffle=shuffle)

    def save_model(self, model, save_dir: str):
        """Persist DiT weights only (VAE is frozen, no need to save)."""
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {"state_dict": {
            k: v for k, v in model.state_dict().items() if not k.startswith("autoencoder.")
        }}
        torch.save(ckpt, os.path.join(save_dir, "ldm_finetuned.ckpt"))
