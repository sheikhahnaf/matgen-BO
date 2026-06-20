#!/bin/bash
#SBATCH --job-name=dgp_mace_soap_n250_diel
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000

# HF_TOKEN is loaded from a user-provided .env file (never hardcoded; see .env.example).
# Put a .env next to this script or in the directory you run sbatch from:  HF_TOKEN=hf_xxx
set -a; [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ] && . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
[ -z "${HF_TOKEN:-}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2

PYTHON=/scratch/user/ahnafalvi/envs/ase-test-mace/bin/python
SCRIPT=/scratch/user/ahnafalvi/ASE_regression_test/dgp_regression.py
DATASET=dielectric_constant
NTRAIN=250

for PCA in 10 25 50; do
    OUTDIR=/scratch/user/ahnafalvi/results/dgp/${DATASET}_pca${PCA}_n${NTRAIN}

    echo "=== DGP MACE | pca=${PCA} n_train=${NTRAIN} ==="
    $PYTHON $SCRIPT \
        --dataset $DATASET \
        --pca-components $PCA \
        --n-train $NTRAIN \
        --n-splits 5 \
        --descriptor mace \
        --output-dir $OUTDIR \
        --device cuda

    echo "=== DGP SOAP | pca=${PCA} n_train=${NTRAIN} ==="
    $PYTHON $SCRIPT \
        --dataset $DATASET \
        --pca-components $PCA \
        --n-train $NTRAIN \
        --n-splits 5 \
        --descriptor soap \
        --output-dir $OUTDIR \
        --device cuda
done

echo "=== DGP MACE/SOAP DIELECTRIC n_train=250 SWEEP COMPLETE ==="
