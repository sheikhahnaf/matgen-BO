#!/bin/bash
#SBATCH --job-name=p3i_accel
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3i_accel_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3i_accel_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --exclude=fc023

# Phase-3 IMPROVED ACCEL: 20 cycles + warm-start LTM + EI acquisition.
# Tests fixes #2 (Phase-1/prior warm-start) and #3 (EI vs PI) on the
# accel arm where those fixes apply.

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PRISTINE="$SCRATCH/matinvent-pristine"
PY="$ENV_PREFIX/bin/python"

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PRISTINE:$PROJ"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GP-routing knobs
export GP_PCA_COMPONENTS=50
export GP_TOP_K=8
export GP_TARGET_CP=1.5
export GP_ANCHOR_EVERY=5
export GP_COLD_START_MIN=16
# Fix #3: Expected Improvement (smoother than PI when target is far)
export GP_ACQUISITION=ei
# Fix #2: warm-start the GP from prior accel runs' labels (seeds 42+17+99
# combined, 160 unique labels, includes Al2Li11Mg7=1.56 and Li12Mg7Zn=1.47).
export GP_LTM_SEED="$PROJ/data/ltm_seed_pool.parquet"

SEED="${SEED:-17}"
CYCLES="${CYCLES:-20}"
export PYTHONHASHSEED="$SEED"
EXPNAME="hcap_p3i_accel_seed${SEED}_${SLURM_JOB_ID}"
# Each run gets its own LTM file (warm-started from seed pool at init).
export GP_LTM_PATH="$PROJ/data/ltm_${EXPNAME}.parquet"

cd "$PRISTINE"
echo "Phase-3 IMPROVED ACCEL seed=$SEED rl_epoch=$CYCLES acq=ei warm-start at $(date) on $(hostname)"
echo "  GP_LTM_SEED=$GP_LTM_SEED"
echo "  GP_LTM_PATH=$GP_LTM_PATH"
echo "  GP_ACQUISITION=$GP_ACQUISITION"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u main.py \
    expname=${EXPNAME} \
    +seed=${SEED} \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRouted \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=${CYCLES} \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase-3 improved accel (seed=$SEED, $CYCLES cycles) finished at $(date)"
