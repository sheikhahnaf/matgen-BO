#!/bin/bash
# =============================================================================
# apu_prep.slurm — Grace batch job: MP query + structure fetch + feature bank
# Pattern matches the working closed-loop Phase-3 scripts (T4 GPU + WebProxy).
# Parameterized via env vars so a small smoke run can precede the full job:
#   smoke: sbatch --export=ALL,MP_MAX_POS=300,MP_MAX_UNL=600,OUT_SUBDIR=cache_smoke slurm/apu_prep.slurm
#   full : sbatch slurm/apu_prep.slurm
# =============================================================================
#SBATCH --job-name=apu_prep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%j.err

set -uo pipefail

# --- fixed paths (CONFIRMED) ---
ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
NOTEBOOK="$WORKDIR/A_PU_RandomForest_alloy.ipynb"   # holds the MP API key (not committed)

# --- overridable knobs (defaults = full run) ---
MP_MAX_POS=${MP_MAX_POS:-60000}
MP_MAX_UNL=${MP_MAX_UNL:-120000}
OUT_SUBDIR=${OUT_SUBDIR:-cache}
WITH_STABILITY=${WITH_STABILITY:-1}
E_HULL_MAX=${E_HULL_MAX:-0.5}
N_PCA=${N_PCA:-50}

echo "======================================== "
echo "Job:      $SLURM_JOB_ID   Node: $(hostname)   Started: $(date)"
echo "WORKDIR:  $WORKDIR   OUT_SUBDIR: $OUT_SUBDIR"
echo "MAX_POS:  $MP_MAX_POS   MAX_UNL: $MP_MAX_UNL   WITH_STABILITY: $WITH_STABILITY"
echo "========================================"

module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy
PY="$ENV_PREFIX/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src"
# ORB uses torch.compile; varying structure sizes cause dynamo recompilation
# thrashing (cache_size_limit) that stalls featurization. Disable compile ->
# eager mode = identical numbers, ~30ms/structure, no recompiles.
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
# persistent model-weight caches so ORB/eSEN are not re-downloaded each run
export HF_HOME=/scratch/user/ahnafalvi/.cache/huggingface
export FAIRCHEM_LOCAL_CACHE=/scratch/user/ahnafalvi/.cache/fairchem
mkdir -p "$HF_HOME" "$FAIRCHEM_LOCAL_CACHE" "$WORKDIR/$OUT_SUBDIR"

echo "Python:   $("$PY" --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

STAB_FLAG=""
[ "$WITH_STABILITY" = "1" ] && STAB_FLAG="--with-stability"

cd "$WORKDIR"
echo "--- starting apu_synthesizability.prep ---"
"$PY" -u -m apu_synthesizability.prep \
    --notebook   "$NOTEBOOK" \
    --out-dir    "$WORKDIR/$OUT_SUBDIR" \
    --e-hull-max "$E_HULL_MAX" \
    --max-pos    "$MP_MAX_POS" \
    --max-unl    "$MP_MAX_UNL" \
    --n-pca      "$N_PCA" \
    $STAB_FLAG \
    --seed       0
EXIT_CODE=$?
echo "--- prep finished, exit $EXIT_CODE, $(date) ---"
exit $EXIT_CODE
