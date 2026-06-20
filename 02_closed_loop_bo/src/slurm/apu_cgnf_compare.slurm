#!/bin/bash
# =============================================================================
# apu_cgnf_compare.slurm — CGNF-matched head-to-head on the held-out test split.
#
# Scores pretrained CGNF AND every deployable A-PU config on the SAME test split
# (held-out positives + held-out unlabeled) and computes one identical PU metric
# panel (TPR, estimated precision [class-prior], pessimistic precision, AUROC,
# AUPRC, ECE) at threshold 0.5 and at the prior-adjusted threshold.
#
# WRITES ONLY to results/cgnf_compare/.  Does NOT touch results/apu/ or bank.npz.
# CPU-only (CGNF is a small composition MLP; A-PU is tabular). No GPU.
# =============================================================================
#SBATCH --job-name=apu_cgnf_cmp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.err

set -uo pipefail

ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
MATINVENT=/scratch/user/ahnafalvi/matinvent-bo/matinvent   # makes `rewards.calculators.syn_score` importable
PY="$ENV_PREFIX/bin/python"

mkdir -p "$WORKDIR/logs"

module purge
module load GCC/13.2.0 Anaconda3/2024.02-1 WebProxy

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src:$MATINVENT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

OUT="$WORKDIR/results/cgnf_compare"

echo "========================================================"
echo "CGNF-matched comparison   Node: $(hostname)   $(date)"
echo "Out: $OUT  (results/apu and bank.npz are untouched)"
echo "========================================================"
echo "Python: $("$PY" --version 2>&1)"

cd "$WORKDIR"
"$PY" -u -m apu_synthesizability.cgnf_compare \
    --configs-dir "$WORKDIR/configs/apu" \
    --bank        "$WORKDIR/cache/bank.npz" \
    --out         "$OUT"

echo "--- finished, exit $? , $(date) ---"
