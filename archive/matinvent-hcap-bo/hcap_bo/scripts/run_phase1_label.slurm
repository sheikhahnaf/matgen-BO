#!/bin/bash
#SBATCH --job-name=hcap_p1_label
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_label_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_label_%j.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=24G

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

# Phase 1 needs the fairchem env (eSEN + ORB + phonopy + quacc)
ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-fairchem"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"

cd "$PROJ"
export PYTHONNOUSERSITE=1
echo "Phase 1 labeling starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -m src.cli phase1_label \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/p1_label_${SLURM_JOB_ID}"

echo "Phase 1 finished at $(date)"
