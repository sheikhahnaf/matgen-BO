# External artifacts (not vendored in this repo)

Everything needed to **reproduce the paper figures** is in-tree (code + figure-driving CSVs + prepared dataset pickles + rendered figures). The items below are large raw inputs / model checkpoints that are hosted externally; you only need them for a *full from-scratch rerun*, not to regenerate figures.

## Raw phonon deposit (≈79 GB)
- **Materials Project Phonon database v1.1 (26,413 compounds), processed with Pheasy** — Zenodo record `20196565`
  - https://doi.org/10.5281/zenodo.20196565
  - Consumed by `01_static_benchmark/src/phonon_thermo/zenodo_prep.py` to build the pheasy arm (26,413 → 11,818 after dropping dynamically-unstable / non-finite-thermo materials). The prepared output is vendored: `01_static_benchmark/data/pheasy_phonon_thermo.pkl`.

## ACES results pull (≈6.7 GB)
- The full per-run training-output pull from the HPC runs is **not vendored** (regenerable; aggregated CSVs that drive every figure are in-tree under `0*/data/` and `0*/results/`). Available on request / archive on Zenodo if needed.

## Raw static-benchmark per-run results (≈9 GB)
- The raw per-run regression outputs (`results/{gp,mtgp,mtgp_2,dgp}/…` from the ASE static benchmark) are **not vendored here** — they are regenerable, and every paper figure reads the aggregated CSVs (`archive/ASE-native-surrogates/analysis_v3*/…`, ≈34 MB) which **are** in-tree.
- The complete raw results are public in the companion repo **`github.com/sheikhahnaf/ASE-native-surrogates`** (under `results/`). Only `01_static_benchmark/figures/generators/metric_disagreement.py` (a supplementary diagnostic) needs `results/gp/`; clone that repo and point its `_ROOT` at it to regenerate that one figure.

## Model checkpoints (HuggingFace)
- **Synthesizability classifier — 8 Optuna-tuned ORB-PU checkpoints:** https://huggingface.co/SheikhAhnaf/apu-synthesizability-checkpoints
- **Generative-model checkpoints (MatterGen / CrystalFlow / ADiT RL cycles):** same HuggingFace org.
- The repo vendors the *code, metrics, leaderboards, and scores* for these; only the trained weights (`*.joblib`, `*.ckpt`) are external.

## Fetching
Run `bash docs/fetch_external.sh` to download the above into `external/` (requires `curl` and, for HuggingFace, `huggingface-cli`).
