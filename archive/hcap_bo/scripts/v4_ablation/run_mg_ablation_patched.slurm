#!/bin/bash
#SBATCH --job-name=mgabl
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/mgabl_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/mgabl_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#
# MatterGen ablation arms for the closed-loop oracle-budget study.
# Provenance: byte-identical to scripts/v4{,_bm}/run_phase3_v4_mg_baseline.slurm
# EXCEPT the single Hydra override `sample_cfg.max_num` (the post-SUN cap).
#   - eval_size stays 16  -> generation batch = eval_size*4 = 64 (pristine MatterGen),
#                            finetune batch = eval_size = 16 (pristine). UNCHANGED.
#   - sample_cfg.max_num is the ONLY knob varied:
#       CAP=4       -> gen-64-cap-4   (oracle <=4 head-of-SUN survivors; head order is
#                                      generation order = arbitrary, so ~ random-4)
#       CAP=100000  -> gen-64-oracle-all (never truncates; oracle every SUN survivor)
#
# Required env vars (set via sbatch --export):
#   TARGET = cp | bm        (heat capacity or bulk modulus)
#   CAP    = 4 | 100000     (post-SUN cap)
#   SEED   = 7|17|23|99|113
# Optional: CYCLES (default 20)
#
# Writes to a DISTINCT expname/results dir; never overwrites baseline/accel runs.

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PRISTINE="$SCRATCH/matinvent-pristine-patched"
PY="$ENV_PREFIX/bin/python"

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJ:$PRISTINE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CYCLES="${CYCLES:-20}"
SEED="${SEED:?set SEED}"
TARGET="${TARGET:?set TARGET=cp|bm}"
CAP="${CAP:?set CAP=4|100000}"
export PYTHONHASHSEED="$SEED"

# Human-readable arm tag for the expname (drives the output dir, guarantees no clash)
if [ "$CAP" = "4" ]; then ARM="cap4"; else ARM="oracleall"; fi
EXPNAME="hcap_mgabl_${ARM}_${TARGET}_seed${SEED}_patched_${SLURM_JOB_ID}"

cd "$PRISTINE"
echo "=== MG ABLATION  arm=$ARM target=$TARGET cap(max_num)=$CAP seed=$SEED cycles=$CYCLES ==="
echo "    expname=$EXPNAME   $(date)  $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "$TARGET" = "cp" ]; then
    REWARD_ARGS=(
        reward=heat_capacity
        reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN
        reward.prop_cfg.0.calculator.env_name=null
        reward.prop_cfg.0.calculator.worker=1
    )
    RESULTS_DIR="$PROJ/results"
else
    REWARD_ARGS=(
        reward=bulk_modulus_local
        reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_BM
        reward.prop_cfg.0.calculator.env_name=null
        reward.prop_cfg.0.calculator.worker=1
        reward.prop_cfg.0.target=ascending
        reward.prop_cfg.0.maxv=400.0
    )
    RESULTS_DIR="$PROJ/results_bm"
fi

# NOTE the single ablation override:  sample_cfg.max_num=${CAP}
#      eval_size stays 16 so generation (64) and finetune (16) are pristine.
$PY -u main.py \
    --config-dir="$PROJ/configs" \
    expname=${EXPNAME} \
    +seed=${SEED} \
    model=mattergen \
    "${REWARD_ARGS[@]}" \
    rl_epoch=${CYCLES} \
    eval_size=16 \
    sample_cfg.max_num=${CAP} \
    logger=csv \
    results_dir=${RESULTS_DIR}

echo "=== MG ABLATION $ARM/$TARGET seed=$SEED finished at $(date) ==="
