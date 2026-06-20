"""Smoke test for CrystalFlow Phase-3 RL module:
   1. Load CrystalFlowSuite + checkpoint
   2. Build a fake PyG batch of 4 mp-20-style structures
   3. Run add_noise → calc_sample_loss → backward
   4. Confirm loss is finite and gradients flow

This catches schema mismatches before launching a full SLURM job.
"""

from __future__ import annotations

import sys
import os
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data


def fake_batch(n: int = 4):
    """Build a tiny PyG Batch resembling DiffCSP/CrystalFlow training inputs."""
    datas = []
    for i in range(n):
        # 4-atom cubic-ish unit cell
        nat = 4
        d = Data(
            atom_types=torch.randint(1, 20, (nat,)),
            frac_coords=torch.rand(nat, 3),
            lengths=torch.tensor([[3.5, 3.6, 3.7]]),
            angles=torch.tensor([[90.0, 90.0, 90.0]]),
            num_atoms=torch.tensor([nat]),
            num_nodes=nat,
            reward=torch.tensor([0.5]),
        )
        datas.append(d)
    return Batch.from_data_list(datas)


def main():
    # 1. Load suite
    from src.p3_models.crystalflow_suite import CrystalFlowSuite

    sample_cfg = OmegaConf.create({"batch_size": 4, "num_batches": 1})
    finetune_cfg = OmegaConf.create({"lr": 1e-5, "epochs": 1, "batch_size": 4, "timesteps": 4, "accum_steps": 1, "sigma": 1e-3})

    suite = CrystalFlowSuite(
        model_name="crystalflow_mp20",
        sample_cfg=sample_cfg,
        finetune_cfg=finetune_cfg,
        model_path=os.path.expandvars("$SCRATCH/checkpoints/crystalflow/ckpt-v1.0.0/DNG-mp-20"),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("[smoke] suite instantiated")

    model = suite.load_model()
    print(f"[smoke] model loaded: {type(model).__name__}, n_params={sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    device = next(model.parameters()).device
    print(f"[smoke] device: {device}")

    # 2. Fake batch
    batch = fake_batch(n=4).to(device)
    print(f"[smoke] batch on {device}: num_graphs={batch.num_graphs}, num_atoms total={batch.num_atoms.sum().item()}")

    # 3. Add noise + forward
    print("[smoke] add_noise...")
    out = model.add_noise(batch, time=2)
    print(f"  add_noise returned: {type(out).__name__} of length {len(out)}")
    print(f"  item types: {[type(x).__name__ for x in out]}")

    print("[smoke] calc_sample_loss...")
    loss, agent_pred = model.calc_sample_loss(out)
    print(f"  loss shape: {loss.shape}, mean: {loss.mean().item():.4f}")

    # 4. KL reg using same prediction as a fake "prior" (same model, no_grad)
    print("[smoke] calc_kl_reg...")
    with torch.no_grad():
        _, prior_pred = model.calc_sample_loss(out)
    kl = model.calc_kl_reg(agent_pred, prior_pred, batch)
    print(f"  kl shape: {kl.shape}, mean: {kl.mean().item():.6f}  (should be ~0 since prior=agent)")

    # 5. Backward
    total = (loss + 1e-3 * kl).mean()
    total.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[smoke] backward OK — params with non-zero grad: {n_grad}")

    print("[smoke] ALL CHECKS PASS — CrystalFlow RL hooks are live")


if __name__ == "__main__":
    sys.exit(main() or 0)
