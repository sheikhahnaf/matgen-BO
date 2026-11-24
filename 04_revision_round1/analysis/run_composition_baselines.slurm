#!/bin/bash
#SBATCH --job-name=compbase
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cpu
#SBATCH --mem=96G
#
# Composition-baseline analysis on a CPU node (exact GPs on ~2,000-3,000 points).
# Required (sbatch --export): REPO = root of a clone of this repository, TARGET = cp | bm.
set -euo pipefail
module purge
module load GCC/13.2.0 Anaconda3/2024.02-1 WebProxy
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH/envs/matinvent-hcap-bo}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "${REPO:?set REPO}/04_revision_round1/analysis"
echo "=== composition baselines target=${TARGET:?} $(date) $(hostname) ==="
"$ENV_PREFIX/bin/python" -u analyze_composition_baselines.py "$TARGET"
echo "=== finished $(date) ==="
