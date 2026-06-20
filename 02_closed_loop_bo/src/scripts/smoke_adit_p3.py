"""Smoke test for ADiT Phase-3 RL module."""

from __future__ import annotations

import os
import sys
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data


def fake_batch(n: int = 4):
    """Build PyG batch matching ADiT's MP20Dataset Data() schema."""
    datas = []
    import numpy as np
    for i in range(n):
        nat = 4
        frac = torch.rand(nat, 3)
        L = torch.diag(torch.tensor([3.5, 3.6, 3.7]))   # 3x3 cell
        cell = L.unsqueeze(0)                            # (1, 3, 3)
        pos = torch.einsum("bi,bij->bj", frac, torch.repeat_interleave(cell, nat, dim=0))
        lengths = torch.tensor([[3.5, 3.6, 3.7]])
        angles = torch.tensor([[90.0, 90.0, 90.0]])
        d = Data(
            atom_types=torch.randint(1, 20, (nat,)),
            frac_coords=frac,
            pos=pos,
            cell=cell,
            lattices=cell.clone(),                       # alias
            lattices_scaled=cell.clone() / nat ** (1/3),
            lengths=lengths,
            lengths_scaled=lengths / nat ** (1/3),
            angles=angles,
            angles_radians=angles * (np.pi / 180.0),
            num_atoms=torch.LongTensor([nat]),
            num_nodes=nat,
            token_idx=torch.arange(nat),
            dataset_idx=torch.LongTensor([0]),
            spacegroup=torch.LongTensor([0]),
            reward=torch.tensor([0.5]),
        )
        datas.append(d)
    return Batch.from_data_list(datas)


def main():
    from src.p3_models.adit_suite import ADiTSuite

    sample_cfg = OmegaConf.create({"batch_size": 4, "num_batches": 1})
    finetune_cfg = OmegaConf.create({"lr": 1e-5, "epochs": 1, "batch_size": 4, "timesteps": 4, "accum_steps": 1, "sigma": 1e-3})

    suite = ADiTSuite(
        model_name="adit_mp20",
        sample_cfg=sample_cfg,
        finetune_cfg=finetune_cfg,
        vae_checkpoint=os.path.expandvars("$SCRATCH/checkpoints/adit/vae.ckpt"),
        dit_checkpoint=os.path.expandvars("$SCRATCH/checkpoints/adit/ldm.ckpt"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[smoke] suite instantiated")

    model = suite.load_model()
    print(f"[smoke] model loaded: {type(model).__name__}, "
          f"n_params total={sum(p.numel() for p in model.parameters()):,}, "
          f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.eval()

    batch = fake_batch(n=4)
    print(f"[smoke] batch num_graphs={batch.num_graphs}")

    # add_noise
    print("[smoke] add_noise...")
    out = model.add_noise(batch, time=2)
    print(f"  add_noise returned: {type(out).__name__} of length {len(out)}")
    print(f"  item types: {[type(x).__name__ for x in out]}")

    # calc_sample_loss
    print("[smoke] calc_sample_loss...")
    loss, agent_pred = model.calc_sample_loss(out)
    print(f"  loss shape: {loss.shape}, mean: {loss.mean().item():.4f}")
    print(f"  agent_pred shape: {agent_pred.shape}")

    # calc_kl_reg
    print("[smoke] calc_kl_reg...")
    with torch.no_grad():
        _, prior_pred = model.calc_sample_loss(out)
    kl = model.calc_kl_reg(agent_pred, prior_pred, batch)
    print(f"  kl shape: {kl.shape}, mean: {kl.mean().item():.6f}")

    # Backward
    total = (loss + 1e-3 * kl).mean()
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[smoke] backward OK — params with non-zero grad: {n_grad}")

    print("[smoke] ALL CHECKS PASS — ADiT RL hooks are live")


if __name__ == "__main__":
    sys.exit(main() or 0)
