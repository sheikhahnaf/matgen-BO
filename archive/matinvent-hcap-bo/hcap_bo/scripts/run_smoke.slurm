#!/bin/bash
#SBATCH --job-name=hcap_smoke
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/smoke_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=16G

set -euo pipefail

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"

PY="$ENV_PREFIX/bin/python"
cd "$PROJ"

export PYTHONNOUSERSITE=1
echo "Smoke test starting at $(date) on $(hostname)"
echo "Python: $($PY -c 'import sys; print(sys.executable, sys.version)')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

$PY -m src.cli smoke \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/smoke_${SLURM_JOB_ID}"

echo "Smoke test finished at $(date)"
