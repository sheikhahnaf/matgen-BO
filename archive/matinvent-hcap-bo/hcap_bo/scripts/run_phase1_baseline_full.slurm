#!/bin/bash
#SBATCH --job-name=hcap_p1_full
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_full_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --exclude=fc023      # avoid contention with sibling job

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

# Seed support — pass SEED via --export=ALL,SEED=N. Defaults to 42.
SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
EXPNAME="hcap_p1_baseline_seed${SEED}_${SLURM_JOB_ID}"

cd "$PRISTINE"
echo "Phase 1 BASELINE (rl_epoch=10) seed=$SEED at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Identical config to Phase 2 — ONLY calculator differs.
# `+seed=$SEED` adds the field to the cfg so any seed-aware code in
# MatInvent honors it; without seed-aware code, the variation comes from
# PYTHONHASHSEED + non-deterministic CUDA — sufficient to demonstrate
# robustness across stochastic noise (paper-standard for RL+GP).
$PY -u main.py \
    expname=${EXPNAME} \
    +seed=${SEED} \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=10 \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase 1 baseline (seed=$SEED) finished at $(date)"
