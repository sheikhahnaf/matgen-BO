#!/bin/bash
#SBATCH --job-name=dgp_mace_soap_pheasy_n1000
#SBATCH --time=40:00:00
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

# HF_TOKEN from user-provided .env (never hardcoded).
set -a; [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ] && . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
[ -z "${HF_TOKEN:-}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2

# Node-local feature cache: pheasy has ~11.8k structures; a persistent per-structure $SCRATCH
# cache would risk the 250k-inode quota. /tmp is node-local, huge inode budget, auto-cleaned;
# features are bit-identical, reused across this job's PCA loop. (Same policy as pheasy GP/MTGP.)
export FEAT_CACHE=/tmp/feat_cache_${SLURM_JOB_ID}
export HOLDOUT_CAP=3000

PY_MACE=/scratch/user/u.sa119259/envs/ase-test-mace/bin/python
PY_FAIR=/scratch/user/u.sa119259/envs/ase-test/bin/python

NTRAIN=1000

for PCA in 10 25 50; do
    echo "=== DGP MACE | pca=${PCA} n_train=${NTRAIN} ==="
    $PY_MACE /scratch/user/u.sa119259/phonon_thermo_benchmark/src/dgp_regression.py \
        --dataset /scratch/user/u.sa119259/phonon_thermo_benchmark/data/pheasy_phonon_thermo.pkl \
        --pca-components ${PCA} \
        --n-train ${NTRAIN} \
        --n-splits 10 \
        --descriptor mace \
        --output-dir /scratch/user/u.sa119259/phonon_thermo_benchmark/results/dgp/pheasy_pca${PCA}_n${NTRAIN} \
        --device cuda
    echo "=== DGP SOAP | pca=${PCA} n_train=${NTRAIN} ==="
    $PY_MACE /scratch/user/u.sa119259/phonon_thermo_benchmark/src/dgp_regression.py \
        --dataset /scratch/user/u.sa119259/phonon_thermo_benchmark/data/pheasy_phonon_thermo.pkl \
        --pca-components ${PCA} \
        --n-train ${NTRAIN} \
        --n-splits 10 \
        --descriptor soap \
        --output-dir /scratch/user/u.sa119259/phonon_thermo_benchmark/results/dgp/pheasy_pca${PCA}_n${NTRAIN} \
        --device cuda
done

echo "=== dgp_mace_soap_pheasy_n1000 SWEEP COMPLETE ==="
