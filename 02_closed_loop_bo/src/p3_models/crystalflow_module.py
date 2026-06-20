"""CrystalFlow Phase-3 RL module.

Subclasses upstream `CSPFlow` (the actual Lightning module the trained
DNG-mp-20 checkpoint corresponds to) and adds three thin wrappers that adapt
its CFM training-step API to MatInvent's `(add_noise, calc_sample_loss,
calc_kl_reg)` triad.

CSPFlow already implements:
    - the CSPNet decoder + type_encoding (vocabulary-restricted to mp-20's 28 elements)
    - the CFM time scheduler + interpolant + velocity-field target
    - all symmetry-aware machinery (lattice_polar, polar decomposition, etc.)

MatInvent's `pipeline/mat_invent.py:ft_step` calls our 3 methods inside its
`for t in range(timesteps)` loop. CFM's training is single-step (no discrete
timestep iteration), so we simply randomize t internally on each call — this
amounts to running CSPFlow's normal training loop over `timesteps` random
batches, weighted by reward (REINFORCE).
"""

from __future__ import annotations

import sys
import os

# Ensure CrystalFlow upstream is importable BEFORE subclassing CSPFlow.
_REPO = os.environ.get(
    "CRYSTALFLOW_REPO",
    os.path.expandvars("$SCRATCH/diffusion-zoo-repos/CrystalFlow"),
)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.nn.functional as F  # noqa: F401  (used by CSPFlow internals)
from torch_scatter import scatter

# Subclass upstream's actual Lightning module — the one that owns the
# trained weights via `load_from_checkpoint`.
from diffcsp.pl_modules.flow import CSPFlow  # type: ignore[import-not-found]


class CrystalFlowModule(CSPFlow):
    """MatInvent-shaped wrapper around CrystalFlow's CSPFlow.

    Adds 3 RL-friendly methods:
        add_noise(batch, time)            → input_all
        calc_sample_loss(input_all)       → (loss_per_sample, agent_pred)
        calc_kl_reg(agent_pred, prior_pred, batch) → kl_per_sample

    Inherits everything else (decoder, schedulers, symmetry handling,
    sampling) from CSPFlow so the trained checkpoint loads unchanged.
    """

    # ------------------------------------------------------------------
    # add_noise: prepare a noised batch (CFM linear interpolation, single t)
    #            matches upstream CSPFlow.forward shape (lattices_rep + lattices_mat)
    # ------------------------------------------------------------------
    def add_noise(self, batch, time=None):
        """Sample a noised batch for one CFM training step.

        Compute lattice in BOTH 6-vector polar form (`lattices_rep`, what the
        CFM operates on) AND 3x3 matrix form (`lattices_mat`, what CSPNet's
        edge model needs for distance computation). Mirrors CSPFlow.forward
        but exposes the per-step state as a dict for calc_sample_loss.
        """
        from models.diffcsp.utils import lattice_params_to_matrix_torch
        from diffcsp.pl_modules.lattice_utils import lattice_polar_build_torch  # type: ignore
        import torch.nn.functional as F

        bsz = batch.num_graphs
        device = self.device

        # CFM continuous time t ∈ [0, 1)
        if time is None:
            t = torch.rand(bsz, device=device)
        else:
            timesteps = max(int(getattr(self.beta_scheduler, "timesteps", 1000)), 1)
            time_idx = int(time) % timesteps
            t_lo = time_idx / timesteps
            t_hi = (time_idx + 1) / timesteps
            t = torch.empty(bsz, device=device).uniform_(t_lo, t_hi)

        # Time embedding — upstream CSPFlow uses raw t (no 1000x scaling)
        time_emb = self.time_embedding(t)

        # ------ Lattice in polar (6-vec) form ------
        lattices_mat_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles).to(device)
        # 3x3 matrix → 6-vector polar log via upstream's decompose()
        lattices_rep_T = self.latticedecompnn.decompose(lattices_mat_T)  # (B, 6)
        # Sample prior in 6-vec polar space (matches sample_lattice_polar)
        lattices_rep_0 = (
            torch.randn(bsz, 6, device=device) * self.lattice_polar_sigma
        )
        lattices_rep_0[:, -1] += 1
        # CFM linear interp on 6-vec rep + velocity target
        tar_l = lattices_rep_T - lattices_rep_0
        input_lattice_rep = lattices_rep_0 + t[:, None] * tar_l
        # 3x3 mat form (for distance computation in CSPNet)
        input_lattice_mat = lattice_polar_build_torch(input_lattice_rep)

        # ------ Frac coords ------
        frac = batch.frac_coords.to(device)
        f0 = torch.rand_like(frac)
        # Wrapped-torus velocity target: ((x_T - x_0 - 0.5) mod 1) - 0.5
        tar_f = (frac - f0 - 0.5) % 1.0 - 0.5
        t_per_atom = t.repeat_interleave(batch.num_atoms)[:, None]
        input_frac = (f0 + t_per_atom * tar_f) % 1.0

        # ------ Atom types ------
        if self.type_encoding is None:
            from diffcsp.pl_modules.cspnet import MAX_ATOMIC_NUM as UP_MAX  # type: ignore
            gt_oh = F.one_hot(batch.atom_types - 1, num_classes=UP_MAX).float()
            rd_oh = torch.randn_like(gt_oh)
        else:
            gt_oh = self.type_encoding(batch.atom_types).to(device)
            if gt_oh.ndim == 1:
                gt_oh = F.one_hot(gt_oh.long(), num_classes=self.type_encoding.out_dim).float()
            else:
                gt_oh = gt_oh.float()
            # Use upstream's get_rd_encoded_types for sampling matched prior
            try:
                rd_oh = self.type_encoding.get_rd_encoded_types(
                    batch.atom_types.shape[0], device=device,
                ).float()
            except Exception:
                rd_oh = torch.randn_like(gt_oh)
        tar_t = gt_oh - rd_oh
        input_atom_types = rd_oh + t_per_atom * tar_t

        return {
            "time_emb": time_emb,
            "input_atom_types": input_atom_types,
            "input_frac": input_frac,
            "input_lattice_rep": input_lattice_rep,
            "input_lattice_mat": input_lattice_mat,
            "num_atoms": batch.num_atoms,
            "batch_idx": batch.batch,
            "targets": (tar_l, tar_f, tar_t),
        }, None, batch.batch

    # ------------------------------------------------------------------
    # calc_sample_loss: decoder forward + CFM MSE
    # ------------------------------------------------------------------
    def calc_sample_loss(self, input_all):
        """Per-sample CFM regression loss + decoder predictions for KL."""
        noised, _, batch_idx = input_all
        if not isinstance(noised, dict):
            # Defensive: legacy/old format
            raise RuntimeError("CrystalFlowModule.calc_sample_loss expects dict from add_noise")

        tar_l, tar_f, tar_t = noised["targets"]

        # Decoder forward — upstream CSPNet API uses kwargs `lattices_rep`
        # (6-vec polar) + `lattices_mat` (3x3 matrix), plus frac_coords,
        # atom_types, num_atoms, node2graph, time_emb (as `t`).
        pred_l, pred_f, pred_t = self.decoder(
            t=noised["time_emb"],
            atom_types=noised["input_atom_types"],
            frac_coords=noised["input_frac"],
            lattices_rep=noised["input_lattice_rep"],
            num_atoms=noised["num_atoms"],
            node2graph=noised["batch_idx"],
            lattices_mat=noised["input_lattice_mat"],
            cemb=None,
            guide_indicator=None,
        )

        # Per-graph CFM MSE losses
        loss_l = (pred_l - tar_l).pow(2).mean(dim=-1)               # (B,)
        loss_f_per_atom = (pred_f - tar_f).pow(2).mean(dim=-1)
        loss_f = scatter(loss_f_per_atom, noised["batch_idx"], dim=0, reduce="mean")
        loss_t_per_atom = (pred_t - tar_t).pow(2).mean(dim=-1)
        loss_t = scatter(loss_t_per_atom, noised["batch_idx"], dim=0, reduce="mean")

        cost_l = float(getattr(self.hparams, "cost_lattice", 1.0))
        cost_c = float(getattr(self.hparams, "cost_coord", 1.0))
        cost_y = float(getattr(self.hparams, "cost_type", 1.0))
        loss_per_sample = cost_l * loss_l + cost_c * loss_f + cost_y * loss_t

        agent_pred = (pred_l, pred_f, pred_t)
        return loss_per_sample, agent_pred

    # ------------------------------------------------------------------
    # calc_kl_reg: per-graph squared diff of decoder outputs (agent vs prior)
    # ------------------------------------------------------------------
    def calc_kl_reg(self, agent_pred, prior_pred, batch):
        """KL surrogate: squared diff of decoder outputs aggregated per-graph.

        agent_pred / prior_pred shapes (from calc_sample_loss):
            pred_l: (B, 6)         lattice 6-vector velocity
            pred_f: (n_atoms, 3)   frac-coord velocity per atom
            pred_t: (n_atoms, T)   atom-type velocity per atom
        """
        pred_l, pred_f, pred_t = agent_pred
        prior_l, prior_f, prior_t = prior_pred

        kl_l = (pred_l - prior_l.detach()).pow(2).mean(dim=-1)
        kl_f_per = (pred_f - prior_f.detach()).pow(2).mean(dim=-1)
        kl_f = scatter(kl_f_per, batch.batch, dim=0, reduce="mean")
        kl_t_per = (pred_t - prior_t.detach()).pow(2).mean(dim=-1)
        kl_t = scatter(kl_t_per, batch.batch, dim=0, reduce="mean")
        return kl_l + kl_f + kl_t
