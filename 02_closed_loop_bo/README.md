# 02 — Closed-loop BO discovery
MatterGen/CrystalFlow/ADiT + GP-routed Expected-Improvement & DPP gate; heat-capacity and bulk-modulus oracles.

- `src/` loop code, configs, SLURM
- `data/hcap_data/` LTM parquets
- `results/results-paper-v4/` per-trajectory long_term_memory.csv + metrics + generated structures (.extxyz)
- `figures/`

Full reruns need a GPU + the generative checkpoints (HuggingFace, see ../docs/EXTERNAL.md); figures regenerate from in-tree CSVs on CPU.
