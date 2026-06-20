#!/bin/bash
# =============================================================================
# apu_optuna.slurm — Optuna-tuned A-PU studies (one per array task).
# 8 configs = {mag, orb, orb_mag, orb_mag_stab} x {xgboost, rf}.
# Each study: TPESampler(42) + MedianPruner + 5-fold inner CV, AUPRC objective;
# best model refit on train+val, post-hoc abstention/OOD attached, evaluated on
# the held-out test split.  Writes ONLY to results/apu_optuna/.  nnPU excluded
# (weak in the fixed-HP sweep).
# =============================================================================
#SBATCH --array=0-7%8
#SBATCH --job-name=apu_optuna
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.err

set -uo pipefail

ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
PY="$ENV_PREFIX/bin/python"

mkdir -p "$WORKDIR/logs" "$WORKDIR/results/apu_optuna"

module purge
module load GCC/13.2.0 Anaconda3/2024.02-1 WebProxy

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

FEATS=(mag mag orb orb orb_mag orb_mag orb_mag_stab orb_mag_stab)
BASES=(xgboost rf xgboost rf xgboost rf xgboost rf)
FS="${FEATS[$SLURM_ARRAY_TASK_ID]}"
BASE="${BASES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Optuna study task $SLURM_ARRAY_TASK_ID : ${FS} x ${BASE}"
echo "Node $(hostname)  $(date)   out=results/apu_optuna/${FS}__${BASE}"
echo "========================================================"
echo "Python: $("$PY" --version 2>&1)"

cd "$WORKDIR"
"$PY" -u -m apu_synthesizability.optuna_tune \
    --feature-set "$FS" \
    --base        "$BASE" \
    --bank        "$WORKDIR/cache/bank.npz" \
    --out         "$WORKDIR/results/apu_optuna" \
    --n-trials    40 \
    --timeout     6000

echo "--- finished, exit $? , $(date) ---"
