#!/bin/bash
#SBATCH --job-name=hcap_p1_baseline
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_baseline_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/p1_baseline_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=32G

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PRISTINE="$SCRATCH/matinvent-pristine"
PY="$ENV_PREFIX/bin/python"

# Source HF token for eSEN-30M-OAM checkpoint download
if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"
export PYTHONNOUSERSITE=1
# pristine main.py imports from cwd-relative `rewards.*` etc., AND we add our project for LocalESEN
export PYTHONPATH="$PRISTINE:$PROJ"

cd "$PRISTINE"
echo "Phase 1 baseline (exact MatInvent Cp config + LocalESEN override) starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# SMOKE CONFIG: 2 RL epochs × eval_size=4 (=8 oracle calls). Once plumbing works,
# scale to upstream rl_epoch=120, eval_size=16 (=1920 oracle calls) for the full run.
$PY -u main.py \
    expname=hcap_p1_baseline_${SLURM_JOB_ID} \
    reward=heat_capacity \
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN \
    reward.prop_cfg.0.calculator.env_name=null \
    reward.prop_cfg.0.calculator.worker=1 \
    rl_epoch=2 \
    eval_size=4 \
    logger=csv \
    results_dir=$PROJ/results

echo "Phase 1 baseline finished at $(date)"
