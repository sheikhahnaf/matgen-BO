#!/bin/bash
# =============================================================================
# apu_score_generated.slurm — score the 40 generated structures with the tuned
# A-PU model (orb_mag xgboost) + pretrained CGNF.  Writes results/generated_scores/.
# CPU; ORB runs eager (TORCHDYNAMO_DISABLE) since structure sizes vary.
# =============================================================================
#SBATCH --job-name=apu_score_gen
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=24G
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.err

set -uo pipefail

ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
MATINVENT=/scratch/user/ahnafalvi/matinvent-bo/matinvent   # CGNF import root
PY="$ENV_PREFIX/bin/python"

mkdir -p "$WORKDIR/logs" "$WORKDIR/results/generated_scores"

module purge
module load GCC/13.2.0 Anaconda3/2024.02-1 WebProxy

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$MATINVENT"
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1

echo "Node $(hostname)  $(date)"; echo "Python: $("$PY" --version 2>&1)"

cd "$WORKDIR"
"$PY" -u -m apu_synthesizability.score_generated \
    --cif-dir   "$WORKDIR/generated_structures" \
    --apu-model "$WORKDIR/results/apu_optuna/orb_mag__xgboost/model.joblib" \
    --pca       "$WORKDIR/cache/bank.npz.pca.pkl" \
    --out-csv   "$WORKDIR/results/generated_scores/scores.csv" \
    --device    cpu

echo "--- finished, exit $? , $(date) ---"
