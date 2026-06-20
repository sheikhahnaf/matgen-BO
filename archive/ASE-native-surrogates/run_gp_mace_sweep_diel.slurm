#!/bin/bash
#SBATCH --job-name=gp_mace_sweep_diel
#SBATCH --time=08:00:00
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
SCRIPT=/scratch/user/ahnafalvi/ASE_regression_test/gp_regression.py
DATASET=dielectric_constant

for PCA in 10 25 50; do
    for NTRAIN in 100 250 500; do
        echo "=== GP MACE | pca=${PCA} n_train=${NTRAIN} ==="
        $PYTHON $SCRIPT \
            --dataset $DATASET \
            --pca-components $PCA \
            --n-train $NTRAIN \
            --n-splits 5 \
            --descriptor mace \
            --output-dir /scratch/user/ahnafalvi/results/gp/${DATASET}_pca${PCA}_n${NTRAIN} \
            --device cuda
    done
done

echo "=== GP MACE DIELECTRIC SWEEP COMPLETE ==="
