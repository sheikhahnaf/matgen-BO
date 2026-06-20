#!/bin/bash
#SBATCH --job-name=p3_adit_accel
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3_adit_accel_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3_adit_accel_%j.err
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

# GP-routing knobs
export GP_PCA_COMPONENTS=50
export GP_TOP_K=8
export GP_TARGET_CP=1.5
export GP_ANCHOR_EVERY=5
export GP_COLD_START_MIN=16

SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
EXPNAME="hcap_p3_adit_accel_seed${SEED}_${SLURM_JOB_ID}"
export GP_LTM_PATH="$PROJ/data/ltm_${EXPNAME}.parquet"

cd "$PRISTINE"
echo "Phase-3 ADiT ACCEL seed=$SEED at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u main.py \
    --config-dir="$PROJ/configs" \
    expname=${EXPNAME} \
    +seed=${SEED} \
    model=adit \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRouted \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=10 \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase-3 ADiT accel finished at $(date)"
