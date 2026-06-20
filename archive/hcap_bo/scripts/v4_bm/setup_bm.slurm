#!/bin/bash
#SBATCH --job-name=bm_setup
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs_bm/bm_setup_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs_bm/bm_setup_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --exclude=fc023

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PRISTINE="$SCRATCH/matinvent-pristine"
PY="$ENV_PREFIX/bin/python"

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:$PRISTINE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GP_DEVICE=cuda

echo "===== Step 1: build BM seed pool from elastic_tensor_2015 ====="
$PY -u $PROJ/scripts/build_bm_seed_pool.py

echo
echo "===== Step 2: validate BM oracle on 7 known materials ====="
$PY -u $PROJ/scripts/validate_bm_oracle.py

echo
echo "===== Setup finished at $(date) ====="
