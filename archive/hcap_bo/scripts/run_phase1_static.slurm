#!/bin/bash
#SBATCH --job-name=hcap_p1_static
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_static_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_static_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G

set -euo pipefail

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"

source activate "$ENV_PREFIX"

cd "$PROJ"

echo "Phase 1 static-regression baseline starting at $(date) on $(hostname)"
nvidia-smi

python -m src.cli phase1_static \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/p1_static_${SLURM_JOB_ID}"

echo "Phase 1 finished at $(date)"
