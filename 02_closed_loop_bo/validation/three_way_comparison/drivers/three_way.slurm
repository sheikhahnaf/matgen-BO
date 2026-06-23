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
# Causal (per-cycle) GP three-way. GPU node so the ~4200 ORB forwards take minutes,
# not hours. WebProxy: compute nodes have no internet; ORB fetches its ckpt from S3 at
# featurize time (weights are also pre-cached on the login node). GP refits stay on CPU
# (many tiny fits). Runs on FASTER so it survives laptop sleep.
module purge
module load WebProxy
export HCAP_ROOT=$SCRATCH/matinvent-hcap-bo/hcap_bo
export DATA_ROOT=$SCRATCH/dft_validation        # inputs: structures + eos_<stem>/K0.json
export OUT_ROOT=$SCRATCH/three_way_comparison    # all three-way outputs
export K0_DIR=$SCRATCH/dft_validation
export DEVICE=cuda
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
mkdir -p $OUT_ROOT/results
cd $OUT_ROOT/drivers
echo "start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$SCRATCH/envs/matinvent-hcap-bo/bin/python three_way_causal.py; rc=$?
echo "end $(date) rc=$rc"
exit $rc
