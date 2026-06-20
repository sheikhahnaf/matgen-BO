#!/bin/bash
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_%x_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --exclude=fc023

# Per-backend Phase-2 open-loop BO ablation.
# Submit as:
#     sbatch --job-name=p2_<MODEL> --export=ALL,MODEL=<MODEL> scripts/run_phase2_per_backend.slurm
#
# Required env vars:
#   MODEL — one of: crystalflow, crysbfn, symmcd, flowmm, crystalformer, atomgpt, adit
#
# Optional:
#   CYCLES=10  BATCH=64  K=8  ANCHOR_EVERY=5  RUN_TAG=

set -euo pipefail

if [ -z "${MODEL:-}" ]; then
  echo "ERROR: MODEL env var required (e.g. MODEL=crystalflow)" >&2
  exit 2
fi

CYCLES="${CYCLES:-10}"
BATCH="${BATCH:-64}"
K="${K:-8}"
ANCHOR_EVERY="${ANCHOR_EVERY:-5}"
RUN_TAG="${RUN_TAG:-}"

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"
cd "$PROJ"

# HF token for any gated checkpoints the target adapter may pull
if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
  set -a; source "$SCRATCH/.secrets/hf_token.env"; set +a
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

OUT_BASE="$PROJ/results/p2_${MODEL}${RUN_TAG:+_${RUN_TAG}}_${SLURM_JOB_ID}"
mkdir -p "$OUT_BASE"

echo "================================================================"
echo "Phase-2 ablation: backend=$MODEL  cycles=$CYCLES  batch=$BATCH  K=$K"
echo "  output_dir = $OUT_BASE"
echo "  node       = $(hostname)"
echo "  start      = $(date)"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# 1) Merge base hcap_bo.yaml with the per-model override fragment.
$PY - <<PYEOF
from omegaconf import OmegaConf
import os, sys
base = OmegaConf.load("configs/hcap_bo.yaml")
override = OmegaConf.load(f"configs/diffusion/${MODEL}.yaml")
merged = OmegaConf.merge(base, override)
# Phase-2 ablation tweaks (the things we swept on the MatterGen pair)
merged.generation.cycles = ${CYCLES}
merged.generation.batch_size = ${BATCH}
merged.acquisition.q = ${K}
merged.oracle.anchor_every = ${ANCHOR_EVERY}
merged.experiment.name = f"p2_${MODEL}_${SLURM_JOB_ID}"
out = "${OUT_BASE}/config_merged.yaml"
OmegaConf.save(merged, out)
print(f"merged config saved to {out}")
PYEOF

# 2) Run Phase-2 open-loop BO with the merged config.
$PY -m src.cli phase2_open \
    --config "$OUT_BASE/config_merged.yaml" \
    --output-dir "$OUT_BASE"

echo "Phase-2 [$MODEL] finished at $(date)"
