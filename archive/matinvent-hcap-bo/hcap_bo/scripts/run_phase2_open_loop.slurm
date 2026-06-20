#!/bin/bash
#SBATCH --job-name=hcap_p2_open
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_open_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_open_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G

set -euo pipefail

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"
cd "$PROJ"

export PYTHONNOUSERSITE=1
echo "Phase 2 starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Smoke-quick: short run (5 cycles, B'=8, K=4) to validate plumbing end-to-end
$PY -m src.cli phase2_open \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/p2_open_${SLURM_JOB_ID}"

echo "Phase 2 finished at $(date)"
