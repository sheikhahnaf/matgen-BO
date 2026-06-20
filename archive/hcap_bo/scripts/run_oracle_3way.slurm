#!/bin/bash
#SBATCH --job-name=hcap_oracle_3way
#SBATCH --output=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/oracle_3way_%j.out
#SBATCH --error=/scratch/user/ahnafalvi/matinvent-hcap-bo/logs/oracle_3way_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:t4:1
#SBATCH --partition=gpu
#SBATCH --mem=24G

set -euo pipefail
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

# matinvent-hcap-bo env now has fairchem 1.10 + phonopy + ORB; the
# matinvent-hcap-fairchem env still has UMA (fairchem 2.14). For 3-way:
#   - eSEN runner: needs fairchem 1.10 → matinvent-hcap-bo env
#   - ORB runner: needs orb_models → either env (use matinvent-hcap-bo)
#   - UMA runner: needs fairchem 2.14 → matinvent-hcap-fairchem env
# We default to matinvent-hcap-bo here; UMA will fail until we point it.
# A later refactor will route per-oracle.
ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
PROJ="$SCRATCH/matinvent-hcap-bo"
PY="$ENV_PREFIX/bin/python"

# Source HF token so eSEN-30M-OAM can download from facebook/OMAT24
if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export FAIRCHEM_LOCAL_CACHE="$SCRATCH/.cache/huggingface"

cd "$PROJ"
# Skip user site-packages — ~/.local/.../numexpr is incompatible with our env's numpy 2.x
export PYTHONNOUSERSITE=1
echo "Oracle 3-way starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -m src.cli oracle_3way \
    --config configs/hcap_bo.yaml \
    --output-dir "$PROJ/results/oracle_3way_${SLURM_JOB_ID}"

echo "Oracle 3-way finished at $(date)"
