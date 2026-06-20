#!/bin/bash
#SBATCH --job-name=gp_soap_orb_uma_sweep
#SBATCH --time=16:00:00
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

# SOAP uses ase-test-mace; ORB and UMA use ase-test
PYTHON_MACE=/scratch/user/ahnafalvi/envs/ase-test-mace/bin/python
PYTHON_FAIRCHEM=/scratch/user/ahnafalvi/envs/ase-test/bin/python
SCRIPT=/scratch/user/ahnafalvi/ASE_regression_test/gp_regression.py
DATASET=elastic_tensor_2015

for PCA in 10 25 50; do
    for NTRAIN in 100 250 500; do
        OUTDIR=/scratch/user/ahnafalvi/results/gp/${DATASET}_pca${PCA}_n${NTRAIN}

        echo "=== GP SOAP | pca=${PCA} n_train=${NTRAIN} ==="
        $PYTHON_MACE $SCRIPT \
            --dataset $DATASET \
            --pca-components $PCA \
            --n-train $NTRAIN \
            --n-splits 5 \
            --descriptor soap \
            --output-dir $OUTDIR \
            --device cuda

        echo "=== GP ORB | pca=${PCA} n_train=${NTRAIN} ==="
        $PYTHON_FAIRCHEM $SCRIPT \
            --dataset $DATASET \
            --pca-components $PCA \
            --n-train $NTRAIN \
            --n-splits 5 \
            --descriptor orb \
            --output-dir $OUTDIR \
            --device cuda

        echo "=== GP UMA | pca=${PCA} n_train=${NTRAIN} ==="
        $PYTHON_FAIRCHEM $SCRIPT \
            --dataset $DATASET \
            --pca-components $PCA \
            --n-train $NTRAIN \
            --n-splits 5 \
            --descriptor uma \
            --output-dir $OUTDIR \
            --device cuda
    done
done

echo "=== GP SOAP/ORB/UMA SWEEP COMPLETE ==="
