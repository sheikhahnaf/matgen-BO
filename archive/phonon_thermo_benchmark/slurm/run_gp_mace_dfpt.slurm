#!/bin/bash
#SBATCH --job-name=gp_mace_dfpt
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=156192594849
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module purge
module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000

# HF_TOKEN is loaded from a user-provided .env file (never hardcoded; see .env.example).
# Put a .env next to this script or in the directory you run sbatch from:  HF_TOKEN=hf_xxx
set -a; [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ] && . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
[ -z "${HF_TOKEN:-}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2

export FEAT_CACHE=/scratch/user/u.sa119259/phonon_thermo_benchmark/data/feat_cache

# Two python envs by absolute path: SOAP/MACE -> ase-test-mace; ORB/UMA -> ase-test.
PY_MACE=/scratch/user/u.sa119259/envs/ase-test-mace/bin/python
PY_FAIR=/scratch/user/u.sa119259/envs/ase-test/bin/python

for PCA in 10 25 50; do
    for NTRAIN in 100 250 500; do
    echo "=== GP MACE | pca=${PCA} n_train=${NTRAIN} ==="
    $PY_MACE /scratch/user/u.sa119259/phonon_thermo_benchmark/src/gp_regression.py \
        --dataset /scratch/user/u.sa119259/phonon_thermo_benchmark/data/dfpt_phonon_thermo.pkl \
        --pca-components ${PCA} \
        --n-train ${NTRAIN} \
        --n-splits 5 \
        --descriptor mace \
        --output-dir /scratch/user/u.sa119259/phonon_thermo_benchmark/results/gp/dfpt_pca${PCA}_n${NTRAIN} \
        --device cuda
    done
done

echo "=== gp_mace_dfpt SWEEP COMPLETE ==="
