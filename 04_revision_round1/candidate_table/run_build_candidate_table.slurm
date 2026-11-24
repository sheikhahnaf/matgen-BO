#!/bin/bash
#SBATCH --job-name=candtab
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#
# Build the candidate tables (heat capacity and bulk modulus) from the raw run
# folders. Required (sbatch --export): REV_ROOT. Optional: EXTRA_RESULTS,
# EXTRA_LTM, EXTRA_LOGS (space-separated directories of additional runs, e.g.
# the revision arms), TAG (output subfolder, default "published").

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="${ENV_PREFIX:-$SCRATCH/envs/matinvent-hcap-bo}"
PROJ="${PROJ:-$SCRATCH/matinvent-hcap-bo}"
REV_ROOT="${REV_ROOT:?set REV_ROOT}"
TAG="${TAG:-published}"
PY="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ"

OUT="$REV_ROOT/candidate_table/out/$TAG"
mkdir -p "$OUT"
SCRIPT="$REV_ROOT/candidate_table/build_candidate_table.py"
echo "=== candidate table TAG=$TAG $(date) $(hostname) ==="

RUNLIST_CP=(); RUNLIST_BM=()
[ -f "$REV_ROOT/candidate_table/runs_cp_${TAG}.txt" ] && RUNLIST_CP=(--run-list "$REV_ROOT/candidate_table/runs_cp_${TAG}.txt")
[ -f "$REV_ROOT/candidate_table/runs_bm_${TAG}.txt" ] && RUNLIST_BM=(--run-list "$REV_ROOT/candidate_table/runs_bm_${TAG}.txt")

$PY -u "$SCRIPT" --target cp --proj "$PROJ" \
    --results ${RESULTS_CP:-$PROJ/results} \
    --logs ${LOGS_CP:-$PROJ/logs} \
    --ltm ${LTM_CP:-$PROJ/data} \
    --seed-pool "$PROJ/data/ltm_seed_pool_v3.parquet" \
    "${RUNLIST_CP[@]}" --out "$OUT"

$PY -u "$SCRIPT" --target bm --proj "$PROJ" \
    --results ${RESULTS_BM:-$PROJ/results_bm} \
    --logs ${LOGS_BM:-$PROJ/logs_bm $PROJ/logs} \
    --ltm ${LTM_BM:-$PROJ/data/bm} \
    --seed-pool "$PROJ/data/bm/ltm_bm_seed_pool.parquet" \
    "${RUNLIST_BM[@]}" --out "$OUT"

echo "=== candidate table finished $(date) ==="
