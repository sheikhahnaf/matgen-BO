#!/bin/bash
# =============================================================================
# apu_sweep.slurm — Grace SLURM array: train one A-PU config per task.
# Each task trains the i-th DEPLOYABLE config (deployable:true, features
# subset of {orb_pca, magpie, stability, cgnf_score}) against the cached
# feature bank and writes a per-config metrics JSON.
#
# Usage:
#   sbatch slurm/apu_sweep.slurm
#
# Prerequisite: apu_prep.slurm must have completed and written
#   $WORKDIR/cache/bank.npz before submitting this array.
# =============================================================================

# --- array size MUST match the number of deployable configs ----------------
# Current count: 20 (26 total YAMLs minus 2 deployable:false mp_props refs
# minus 4 cgnf-feature configs — CGNF is a post-hoc comparator, not a training
# feature, so cgnf_score is no longer in the RUNNABLE set below).
# If configs/apu/*.yaml changes, re-run the filter one-liner below and update
# the upper bound accordingly (the in-script sanity check will catch mismatch).
#SBATCH --array=0-19%10

#SBATCH --job-name=apu_sweep
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --mem=32G
# NOTE: no GPU. Featurization (ORB) is already cached in bank.npz; the sweep is
# pure tabular ML (sklearn / xgboost-hist), so it runs on CPU cores. Bag-level
# parallelism (PUBaggingClassifier n_jobs=-1) uses cpus-per-task.
#SBATCH --output=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.out
#SBATCH --error=/scratch/user/ahnafalvi/apu-synth-sweep/logs/%x_%A_%a.err

set -uo pipefail

# --- fixed paths (mirrors apu_prep.slurm) ------------------------------------
ENV_PREFIX=/scratch/user/ahnafalvi/envs/matinvent-hcap-fairchem
WORKDIR=/scratch/user/ahnafalvi/apu-synth-sweep
PY="$ENV_PREFIX/bin/python"

# --- ensure output directories exist -----------------------------------------
mkdir -p "$WORKDIR/results/apu"
mkdir -p "$WORKDIR/logs"

# --- Grace module + env setup (CPU-only; no CUDA needed) ---------------------
module purge
module load GCC/13.2.0 Anaconda3/2024.02-1 WebProxy

export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR/src"
# Single-threaded BLAS/estimators: parallelism lives at the bag level (joblib),
# so this prevents BLAS thread oversubscription inside each parallel worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- build the sorted, filtered list of runnable configs ---------------------
# A config is runnable iff: deployable:true AND features ⊆ {orb_pca, magpie,
# stability, cgnf_score}.  The two mp_props reference configs are excluded
# because they carry deployable:false.
#
# The one-liner emits one absolute path per line (sorted), which mapfile reads
# into the CONFIGS bash array.

mapfile -t CONFIGS < <(
  "$PY" -u - "$WORKDIR/configs/apu" <<'EOF'
import sys, yaml, glob, os

RUNNABLE = {'orb_pca', 'magpie', 'stability'}  # cgnf_score dropped: CGNF is a post-hoc comparator, not a feature
configs_dir = sys.argv[1]
yamls = sorted(glob.glob(os.path.join(configs_dir, '*.yaml')))
for p in yamls:
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get('deployable', False):
        continue
    if set(cfg.get('features', [])) <= RUNNABLE:
        print(p)
EOF
)

TOTAL=${#CONFIGS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no deployable configs found in $WORKDIR/configs/apu — aborting"
    exit 1
fi

# Sanity-check that the array bound still matches reality
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID >= TOTAL=$TOTAL" \
         "(update --array bound in the script)"
    exit 1
fi

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
CFG_NAME="$(basename "$CFG" .yaml)"
OUT_JSON="$WORKDIR/results/apu/${CFG_NAME}.json"

echo "========================================================"
echo "Array job:  ${SLURM_ARRAY_JOB_ID:-<local>}  task: $SLURM_ARRAY_TASK_ID / $((TOTAL-1))"
echo "Node:       $(hostname)   Started: $(date)"
echo "Config:     $CFG_NAME"
echo "Config path:$CFG"
echo "Output:     $OUT_JSON"
echo "========================================================"

# --- skip if already done (rerunnable) ---------------------------------------
if [ -f "$OUT_JSON" ]; then
    echo "Output JSON already exists — skipping (delete to re-run):"
    echo "  $OUT_JSON"
    exit 0
fi

echo "Python: $("$PY" --version 2>&1)"
echo "CPUs available to task: ${SLURM_CPUS_PER_TASK:-?}"

cd "$WORKDIR"
echo "--- starting apu_synthesizability.train_config ---"
"$PY" -u -m apu_synthesizability.train_config \
    --config "$CFG" \
    --bank   "$WORKDIR/cache/bank.npz" \
    --out    "$OUT_JSON"

EXIT_CODE=$?
echo "--- train_config finished, exit $EXIT_CODE, $(date) ---"
exit $EXIT_CODE
