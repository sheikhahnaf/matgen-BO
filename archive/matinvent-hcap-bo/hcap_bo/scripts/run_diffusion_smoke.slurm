#!/bin/bash
#SBATCH --job-name=diff_smoke
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/diff_smoke_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/diff_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --exclude=fc023

# Smoke-test all 4 viable diffusion backends (n=4 samples each).
# Each backend runs in mat-zoo-modern env via RemoteGeneratorAdapter.

set -euo pipefail

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"
cd "$PROJ"

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
  set -a; source "$SCRATCH/.secrets/hf_token.env"; set +a
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

OUT="$PROJ/results/diffusion_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "================================================================"
echo "Diffusion smoke test on $(hostname) at $(date)"
echo "  output_dir = $OUT"
echo "================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -m src.cli diffusion_smoke \
    --config configs/hcap_bo.yaml \
    --output-dir "$OUT"

echo "Smoke finished at $(date)"
echo "Summary:"
cat "$OUT/summary.json" || true
