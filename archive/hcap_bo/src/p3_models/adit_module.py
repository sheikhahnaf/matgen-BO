"""ADiT Phase-3 RL module — latent-space DiT REINFORCE.

Subclasses upstream `LatentDiffusionLitModule` (the trained DiT in latent
space) and exposes MatInvent-style `(add_noise, calc_sample_loss,
calc_kl_reg)` triad. The frozen VAE encoder/decoder lives inside as
`self.autoencoder` (subclass of `VariationalAutoencoderLitModule`).

Training-loop semantics:
    add_noise(batch, t):
        1. VAE.encode(batch) → latent x_1 (gradients DO NOT flow back into VAE)
        2. interpolant.corrupt_batch({x_1, mask, ...}) → noisy x_t
    calc_sample_loss(input_all):
        denoiser(x_t, t, dataset_idx, spacegroup, mask, x_sc)  → pred_x
        MSE(gt_x_1, pred_x) per-graph (with mask handling)
    calc_kl_reg(agent_pred, prior_pred, batch):
        MSE between current and frozen reference predictions

The VAE decoder is also frozen — we steer the latent distribution via REINFORCE
on the DiT parameters. Reward signal flows through the deterministic decode
chain because the decoder is differentiable (we just don't backprop through
its parameters; a stop_grad on the decoder side would also work).
"""

from __future__ import annotations

import os
import sys

# Add ADiT upstream to sys.path before subclassing
_REPO = os.environ.get(
    "ADIT_REPO",
    os.path.expandvars("$SCRATCH/diffusion-zoo-repos/all-atom-diffusion-transformer"),
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from torch_geometric.utils import to_dense_batch

from adit.models.ldm_module import LatentDiffusionLitModule  # type: ignore[import-not-found]


class ADiTModule(LatentDiffusionLitModule):
    """MatInvent-shaped wrapper around ADiT's latent diffusion module.

    Adds add_noise / calc_sample_loss / calc_kl_reg, thin proxies that map
    upstream's forward() and criterion() to MatInvent's RL update interface.

    Overrides upstream `__init__` to skip eval-evaluator construction (which
    requires the original training-time data CSVs). For RL training we only
    need autoencoder + denoiser + interpolant.
    """

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser,
        interpolant,
        augmentations,
        sampling,
        conditioning,
        optimizer,
        scheduler,
        scheduler_frequency: str,
        compile: bool,
    ) -> None:
        # Skip LatentDiffusionLitModule.__init__ (which builds heavy
        # evaluators and reads CSV data files). Call grandparent
        # (LightningModule) __init__ instead.
        from lightning import LightningModule
        LightningModule.__init__(self)

        self.save_hyperparameters(logger=False)

        # Defensive normalization: upstream's `sample_and_decode` (line 609 of
        # ldm_module.py) does `self.hparams.conditioning.spacegroup`. After
        # load_from_checkpoint, hparams may come back as a plain AttributeDict
        # missing this nested attr. Coerce to a DictConfig with the field.
        from omegaconf import OmegaConf as _OC, DictConfig as _DC
        cond = self.hparams.get("conditioning", None) if hasattr(self.hparams, "get") else None
        if cond is None or not isinstance(cond, _DC):
            cond_dict = dict(cond) if cond is not None else {}
            cond_dict.setdefault("spacegroup", False)
            cond_dict.setdefault("dataset_idx", True)
            self.hparams["conditioning"] = _OC.create(cond_dict)

        # Autoencoder (frozen)
        from adit.models.vae_module import VariationalAutoencoderLitModule
        self.autoencoder_ckpt = autoencoder_ckpt
        map_location = "cpu" if not torch.cuda.is_available() else None
        self.autoencoder = VariationalAutoencoderLitModule.load_from_checkpoint(
            autoencoder_ckpt, map_location=map_location, strict=False,
        )
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

        # Denoiser (the trainable DiT)
        self.denoiser = denoiser

        # Interpolant for rectified-flow corruption
        self.interpolant = interpolant

        # Eval evaluators stubbed — RL training doesn't need them.
        self.val_generation_evaluators = {}
        self.test_generation_evaluators = {}
        self.train_metrics = {}
        self.val_metrics = {}

    # ------------------------------------------------------------------
    # add_noise: VAE encode + interpolant corrupt → noisy dense latent
    # ------------------------------------------------------------------
    def add_noise(self, batch, time=None):
        """Encode the batch into latent space + corrupt with the rectified-flow
        interpolant. VAE forward is no_grad (frozen)."""
        # Encode (no grad — frozen VAE)
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            encoded_batch["x"] = encoded_batch["posterior"].sample()
            x_1 = encoded_batch["x"]
            x_1, mask = to_dense_batch(x_1, encoded_batch["batch"])

        dense_encoded_batch = {
            "x_1": x_1,
            "token_mask": mask,
            "diffuse_mask": mask,
        }

        # Optional time injection: ADiT samples t ~ U(0, 1) inside corrupt_batch.
        # If MatInvent passes a discrete `time`, override.
        if time is not None:
            self.interpolant.device = x_1.device
            n_bins = max(int(getattr(self.interpolant, "num_steps", 100)), 1)
            t_lo = (int(time) % n_bins) / n_bins
            t_hi = ((int(time) % n_bins) + 1) / n_bins
            t = torch.empty(x_1.size(0), device=x_1.device).uniform_(t_lo, t_hi)
            # Manually corrupt with our t (bypassing interpolant's internal sampler)
            x_t = (1.0 - t.view(-1, 1, 1)) * 0 + t.view(-1, 1, 1) * x_1  # placeholder
            # Use upstream's corrupt_batch but inject our t via attribute hint
            try:
                noisy = self.interpolant.corrupt_batch(dense_encoded_batch, t=t)
            except TypeError:
                # Older interpolant API may not accept `t` kwarg — fall back to default sampling
                self.interpolant.device = x_1.device
                noisy = self.interpolant.corrupt_batch(dense_encoded_batch)
        else:
            self.interpolant.device = x_1.device
            noisy = self.interpolant.corrupt_batch(dense_encoded_batch)

        # Conditioning inputs: dataset_idx (mp20=0+1=1), spacegroup (0=null)
        dataset_idx = (
            (batch.dataset_idx if hasattr(batch, "dataset_idx") else torch.zeros(x_1.size(0), dtype=torch.long, device=x_1.device))
            + 1
        )
        # `hparams.conditioning` may not exist on minimal-init checkpoints —
        # default to no spacegroup conditioning (null class 0).
        try:
            use_sg = bool(self.hparams.conditioning.spacegroup)
        except (AttributeError, KeyError):
            use_sg = False
        spacegroup = (
            batch.spacegroup if hasattr(batch, "spacegroup") and use_sg
            else torch.zeros(x_1.size(0), dtype=torch.long, device=x_1.device)
        )

        return {
            "noisy": noisy,
            "mask": mask,
            "dataset_idx": dataset_idx,
            "spacegroup": spacegroup,
        }, None, batch.batch

    # ------------------------------------------------------------------
    # calc_sample_loss: DiT forward + MSE
    # ------------------------------------------------------------------
    def calc_sample_loss(self, input_all):
        noised, _, batch_idx = input_all
        if not isinstance(noised, dict):
            raise RuntimeError("ADiTModule.calc_sample_loss expects dict from add_noise")

        noisy = noised["noisy"]
        mask = noised["mask"]
        dataset_idx = noised["dataset_idx"]
        spacegroup = noised["spacegroup"]

        # DiT forward — predict x_1 from x_t
        pred_x = self.denoiser(
            x=noisy["x_t"],
            t=noisy["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=mask,
            x_sc=None,
        )

        # Per-graph MSE (matching upstream criterion's masking logic)
        gt_x_1 = noisy["x_1"]
        norm_scale = 1 - torch.min(noisy["t"].unsqueeze(-1), torch.tensor(0.9, device=gt_x_1.device))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = mask * noisy["diffuse_mask"]  # token_mask * diffuse_mask
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1) + 1e-8
        x_loss_per_graph = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom

        return x_loss_per_graph, pred_x  # per-graph loss + agent prediction

    # ------------------------------------------------------------------
    # calc_kl_reg: per-graph squared diff of denoiser predictions
    # ------------------------------------------------------------------
    def calc_kl_reg(self, agent_pred, prior_pred, batch):
        # agent_pred / prior_pred: (B, max_tokens, latent_dim)
        # Mean over (token, latent) dims → per-graph scalar
        return (agent_pred - prior_pred.detach()).pow(2).mean(dim=(-1, -2))
