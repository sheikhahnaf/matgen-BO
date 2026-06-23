#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=142705333487
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Causal GP three-way for HEAT CAPACITY (Cv): GP-vs-eSEN + ranking only (no phonon Cv DFT
# yet). Runs the dedicated cp script (three_way_causal_cp.py); the bm script is untouched.
module purge
module load WebProxy
export HCAP_ROOT=$SCRATCH/matinvent-hcap-bo/hcap_bo
export DATA_ROOT=$SCRATCH/dft_validation
export OUT_ROOT=$SCRATCH/three_way_comparison
export DEVICE=cuda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
mkdir -p $OUT_ROOT/results
cd $OUT_ROOT/drivers
echo "start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$SCRATCH/envs/matinvent-hcap-bo/bin/python three_way_causal_cp.py; rc=$?
echo "end $(date) rc=$rc"
exit $rc
