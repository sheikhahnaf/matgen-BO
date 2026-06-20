#!/bin/bash
# =============================================================================
# apu_prep_shard.slurm — SLURM array: sharded parallel APU featurization
#
# Each array task featurizes a disjoint contiguous chunk of the MP dataset
# (raw 256-d ORB embeddings, no PCA). After all tasks finish, run
# merge_shards.py once to fit global PCA and produce bank.npz.
#
# Array range: 0 to (N_SHARDS-1). Adjust --array as needed:
#   --array=0-19%10  means 20 shards, at most 10 running concurrently.
#
# Usage:
#   default (20 shards):
#     sbatch slurm/apu_prep_shard.slurm
#
#   custom shard count (e.g. 8 shards, all concurrent):
#     N_SHARDS=8 sbatch --array=0-7 slurm/apu_prep_shard.slurm
#
#   smoke test (2 shards, small data):
#     sbatch --array=0-1 --export=ALL,N_SHARDS=2,MP_MAX_POS=300,MP_MAX_UNL=600 \
#            slurm/apu_prep_shard.slurm
#
# After all tasks complete, merge with:
#   PYTHONPATH=$WORKDIR/src "$PY" -u -m apu_synthesizability.merge_shards \
#       --shard-dir "$WORKDIR/cache_shards" \
#       --out       "$WORKDIR/cache/bank.npz" \
#       --n-pca     50
# =============================================================================

# --- Array directive: 0-19 = 20 shards, at most 10 running at once ---
# Adjust the range (0-N) to match N_SHARDS-1 below.
#SBATCH --array=0-19%10
#SBATCH --job-name=apu_prep_shard
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.err

set -uo pipefail

# --- fixed paths (CONFIRMED) ---
ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
NOTEBOOK="$WORKDIR/A_PU_RandomForest_alloy.ipynb"   # holds the MP API key (not committed)

# --- overridable knobs ---
N_SHARDS=${N_SHARDS:-20}          # must match --array upper bound + 1
MP_MAX_POS=${MP_MAX_POS:-50000}
MP_MAX_UNL=${MP_MAX_UNL:-60000}
WITH_STABILITY=${WITH_STABILITY:-1}
E_HULL_MAX=${E_HULL_MAX:-0.5}

echo "========================================"
echo "Job:      $SLURM_JOB_ID   ArrayTask: $SLURM_ARRAY_TASK_ID/$((N_SHARDS-1))"
echo "Node:     $(hostname)   Started: $(date)"
echo "WORKDIR:  $WORKDIR"
echo "N_SHARDS: $N_SHARDS   MAX_POS: $MP_MAX_POS   MAX_UNL: $MP_MAX_UNL"
echo "========================================"

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy
PY="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src"
# ORB uses torch.compile; varying structure sizes cause dynamo recompilation
# thrashing (cache_size_limit) that stalls featurization. Disable compile ->
# eager mode = identical numbers, no recompiles.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# persistent model-weight caches so ORB is not re-downloaded each run
export HF_HOME=/scratch/user/ahnafalvi/.cache/huggingface
export FAIRCHEM_LOCAL_CACHE=/scratch/user/ahnafalvi/.cache/fairchem

# ensure output dirs exist
mkdir -p "$WORKDIR/cache_shards" "$WORKDIR/logs"
mkdir -p "$HF_HOME" "$FAIRCHEM_LOCAL_CACHE"

echo "Python:   $("$PY" --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

STAB_FLAG=""
[ "$WITH_STABILITY" = "1" ] && STAB_FLAG="--with-stability"

cd "$WORKDIR"
echo "--- starting apu_synthesizability.prep shard $SLURM_ARRAY_TASK_ID ---"
"$PY" -u -m apu_synthesizability.prep \
    --notebook   "$NOTEBOOK" \
    --out-dir    "$WORKDIR/cache_shards" \
    --shard-idx  "$SLURM_ARRAY_TASK_ID" \
    --n-shards   "$N_SHARDS" \
    --e-hull-max "$E_HULL_MAX" \
    --max-pos    "$MP_MAX_POS" \
    --max-unl    "$MP_MAX_UNL" \
    $STAB_FLAG \
    --seed       0
EXIT_CODE=$?
echo "--- shard $SLURM_ARRAY_TASK_ID finished, exit $EXIT_CODE, $(date) ---"
exit $EXIT_CODE
