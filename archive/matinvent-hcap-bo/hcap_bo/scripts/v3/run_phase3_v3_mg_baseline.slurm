#!/bin/bash
#SBATCH --job-name=p3v3_mg_b
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3v3_mg_b_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p3v3_mg_b_%j.err
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
EXPNAME="hcap_p3v3_mg_baseline_seed${SEED}_${SLURM_JOB_ID}"

cd "$PRISTINE"
echo "Phase-3 v3 MG BASELINE seed=$SEED cycles=$CYCLES at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Pure RL with full oracle — same as the v2 mg-improved baseline.
# v3 only changes the accel side; baselines stay identical for fair comparison.
$PY -u main.py \
    --config-dir="$PROJ/configs" \
    expname=${EXPNAME} \
    +seed=${SEED} \
    model=mattergen \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=${CYCLES} \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase-3 v3 mg baseline (seed=$SEED) finished at $(date)"
