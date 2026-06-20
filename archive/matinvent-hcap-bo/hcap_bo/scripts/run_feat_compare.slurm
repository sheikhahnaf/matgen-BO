#!/bin/bash
#SBATCH --job-name=hcap_feat_compare
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/feat_compare_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/feat_compare_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=20G

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"

cd "$PROJ"
export PYTHONNOUSERSITE=1
echo "Featurizer compare starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -m src.cli feat_compare \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/feat_compare_${SLURM_JOB_ID}"

echo "Featurizer compare finished at $(date)"
