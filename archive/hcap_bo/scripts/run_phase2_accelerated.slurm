#!/bin/bash
#SBATCH --job-name=hcap_p2_accel
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_accel_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p2_accel_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --exclude=fc023      # avoid contention with sibling job (force a different T4 node)

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

# GP-routing knobs (consumed by LocalESEN_GPRouted via env vars).
# This run uses ACQUISITION-TOP-K routing (PI for Cp > target), NOT σ-threshold.
export GP_PCA_COMPONENTS=50
export GP_TOP_K=8                     # Oracle top 8 of 16 per cycle (= 2× speedup deterministic)
export GP_TARGET_CP=1.5               # PI target (J/g/K) — matches MatInvent Cp threshold
export GP_ANCHOR_EVERY=5              # Every 5th cycle, force-oracle all 16 (recalibration)
export GP_COLD_START_MIN=16           # Force-oracle all until LTM has 16 true labels
SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
EXPNAME="hcap_p2_accel_seed${SEED}_${SLURM_JOB_ID}"
export GP_LTM_PATH="$PROJ/data/ltm_${EXPNAME}.parquet"

cd "$PRISTINE"
echo "Phase 2 ACCELERATED (LocalESEN_GPRouted) seed=$SEED at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Same upstream config as Phase 1 baseline; only the calculator changes.
$PY -u main.py \
    expname=${EXPNAME} \
    +seed=${SEED} \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_GPRouted \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=10 \
    eval_size=16 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase 2 accelerated (seed=$SEED) finished at $(date)"
