#!/bin/bash
#SBATCH --job-name=p3v3_mg_a
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3v3_mg_a_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3v3_mg_a_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --exclude=fc023

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
export PYTHONPATH="$PROJ:$PRISTINE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CYCLES="${CYCLES:-20}"
SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"

# v3 GP-routed knobs
export GP_PCA_COMPONENTS=50
export GP_TOP_K=8
export GP_TARGET_CP=1.5
export GP_ANCHOR_EVERY=5
export GP_COLD_START_MIN=50           # was 16 — user's hunch: more data first
export GP_ACQUISITION=ei
export GP_DPP_LAMBDA=0.5              # diversity weight in DPP top-K
export GP_WARMSTART_DECAY=0.9         # exp decay of warm-start influence
export GP_WARMSTART_NOISE_MIN=1e-6
export GP_LTM_SEED="$PROJ/data/ltm_seed_pool_v3.parquet"

EXPNAME="hcap_p3v3_mg_accel_seed${SEED}_${SLURM_JOB_ID}"
export GP_LTM_PATH="$PROJ/data/ltm_${EXPNAME}.parquet"

cd "$PRISTINE"
echo "Phase-3 v3 MG ACCEL seed=$SEED cycles=$CYCLES at $(date) on $(hostname)"
echo "  GP_LTM_SEED=$GP_LTM_SEED"
echo "  GP_LTM_PATH=$GP_LTM_PATH"
echo "  GP_COLD_START_MIN=$GP_COLD_START_MIN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u main.py \
    --config-dir="$PROJ/configs" \
    expname=${EXPNAME} \
    +seed=${SEED} \
    model=mattergen \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRoutedV3 \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=${CYCLES} \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase-3 v3 mg accel (seed=$SEED) finished at $(date)"
