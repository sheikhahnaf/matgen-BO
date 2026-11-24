#!/bin/bash
#SBATCH --job-name=rev1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --exclude=fc023
#
# Closed-loop arms added in the revision. Each arm is one of the original v4
# command lines with a single extra Hydra override.
#
#   ARM=cap4         budget-matched arbitrary selection.
#                    Original BASE command line + `sample_cfg.max_num=4`:
#                    at most the first four SUN survivors reach the oracle each
#                    cycle (head order is generation order). Same definition as
#                    the MatterGen cap-4 arm (scripts/v4_ablation).
#
#   ARM=frozen_gate  frozen-generator control.
#                    Original ACC (GP-gated) command line +
#                    `pipeline._target_=fme_rev.frozen_pipeline.MatInventFrozen`:
#                    the gate selects and refits as in ACC, the generator stays
#                    at its pretrained weights for the whole run.
#
# Required (sbatch --export):
#   ARM       cap4 | frozen_gate
#   BACKBONE  mg | cf | adit
#   TARGET    cp | bm
#   SEED      7 | 17 | 23 | 99 | 113
#   REV_ROOT  directory holding this folder's files; outputs go to $REV_ROOT/runs
# Optional: CYCLES (default 20)
#
# Code locations (override to run from a clone of this repository):
#   PROJ      02_closed_loop_bo sources and configs
#   PRISTINE  MatInvent with the two zero-success-cycle guards (see patches/)
#   ENV_PREFIX  python environment (02_closed_loop_bo/src/configs/env_hcapbo.frozen.txt)

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="${ENV_PREFIX:-$SCRATCH/envs/matinvent-hcap-bo}"
PROJ="${PROJ:-$SCRATCH/matinvent-hcap-bo}"
PRISTINE="${PRISTINE:-$SCRATCH/matinvent-pristine-patched}"
REV_ROOT="${REV_ROOT:?set REV_ROOT}"
PY="$ENV_PREFIX/bin/python"

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REV_ROOT:$PROJ:$PRISTINE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CYCLES="${CYCLES:-20}"
ARM="${ARM:?set ARM=cap4|frozen_gate}"
BACKBONE="${BACKBONE:?set BACKBONE=mg|cf|adit}"
TARGET="${TARGET:?set TARGET=cp|bm}"
SEED="${SEED:?set SEED}"
export PYTHONHASHSEED="$SEED"

case "$BACKBONE" in
    mg)   MODEL=mattergen ;;
    cf)   MODEL=crystalflow ;;
    adit) MODEL=adit ;;
    *) echo "unknown BACKBONE=$BACKBONE"; exit 2 ;;
esac

EXPNAME="rev1_${ARM}_${BACKBONE}_${TARGET}_seed${SEED}_${SLURM_JOB_ID}"
RESULTS_DIR="$REV_ROOT/runs"
mkdir -p "$RESULTS_DIR" "$REV_ROOT/ltm"

# --- reward block: identical to the original v4 / v4_bm scripts -------------
if [ "$TARGET" = "cp" ]; then
    REWARD_NAME=heat_capacity
    ORACLE=src.calculators.LocalESEN
    GATED=src.calculators.LocalESEN_GPRoutedV4
    EXTRA_REWARD=()
    GP_TARGET=1.5
    GP_SEED_POOL="$PROJ/data/ltm_seed_pool_v3.parquet"
elif [ "$TARGET" = "bm" ]; then
    REWARD_NAME=bulk_modulus_local
    ORACLE=src.calculators.LocalESEN_BM
    GATED=src.calculators.LocalESEN_BM_GPRoutedV4
    EXTRA_REWARD=(reward.prop_cfg.0.target=ascending reward.prop_cfg.0.maxv=400.0)
    GP_TARGET=300
    GP_SEED_POOL="$PROJ/data/bm/ltm_bm_seed_pool.parquet"
else
    echo "unknown TARGET=$TARGET"; exit 2
fi

# --- arm block: the single override that defines each arm --------------------
if [ "$ARM" = "cap4" ]; then
    CALC="$ORACLE"
    ARM_ARGS=(sample_cfg.max_num=4)
elif [ "$ARM" = "frozen_gate" ]; then
    CALC="$GATED"
    ARM_ARGS=(pipeline._target_=fme_rev.frozen_pipeline.MatInventFrozen)
    # GP gate settings: identical to the original ACC runs
    export GP_PCA_COMPONENTS=50
    export GP_TOP_K=4
    export GP_K_RATIO=0.5
    export GP_TARGET_CP="$GP_TARGET"
    export GP_ANCHOR_EVERY=999
    export GP_COLD_START_MIN=0
    export GP_ACQUISITION=ei
    export GP_DPP_LAMBDA=0.5
    export GP_WARMSTART_DECAY=0.9
    export GP_WARMSTART_NOISE_MIN=1e-6
    export GP_LTM_SEED="$GP_SEED_POOL"
    export GP_LTM_PATH="$REV_ROOT/ltm/ltm_${EXPNAME}.parquet"
else
    echo "unknown ARM=$ARM"; exit 2
fi

cd "$PRISTINE"
echo "=== REV1 arm=$ARM backbone=$BACKBONE target=$TARGET seed=$SEED cycles=$CYCLES ==="
echo "    expname=$EXPNAME   $(date)  $(hostname)"
echo "    PROJ=$PROJ  PRISTINE=$PRISTINE  REV_ROOT=$REV_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u main.py \
    --config-dir="$PROJ/configs" \
    expname=${EXPNAME} \
    +seed=${SEED} \
    model=${MODEL} \
    reward=${REWARD_NAME} \
    reward.prop_cfg.0.calculator._target_=${CALC} \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=${CYCLES} \
    "${EXTRA_REWARD[@]}" \
    eval_size=16 \
    "${ARM_ARGS[@]}" \
    logger=csv \
    results_dir=${RESULTS_DIR}

echo "=== REV1 $ARM/$BACKBONE/$TARGET seed=$SEED finished at $(date) ==="
