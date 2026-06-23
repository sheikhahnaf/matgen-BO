#!/bin/bash
#SBATCH --job-name=tw_seeded_full
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --account=142705333487
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# FAITHFUL causal three-way, FULL seed (no subsample), BM + Cp, on FASTER (Linux: no macOS
# OMP deadlock, no laptop sleep). Each run is an isolated process; all runs parallel via xargs.
PY=$SCRATCH/envs/matinvent-hcap-bo/bin/python
export HCAP_ROOT=$SCRATCH/matinvent-hcap-bo/hcap_bo
export DATA_ROOT=$SCRATCH/dft_validation
export OUT_ROOT=$SCRATCH/three_way_comparison
export K0_DIR=$SCRATCH/dft_validation
export N_SEED=100000            # >= seed size -> use the FULL seed pool (no subsample)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd $OUT_ROOT/drivers
mkdir -p $OUT_ROOT/results/seeded_parts
echo "start $(date)  N_SEED=full  cpus=$SLURM_CPUS_PER_TASK"
ls $HCAP_ROOT/data/bm/ltm_bm_p3v4_bm_*.parquet | xargs -P $SLURM_CPUS_PER_TASK -I{} $PY three_way_causal_seeded.py --one {} bm
$PY three_way_causal_seeded.py --agg bm
ls $HCAP_ROOT/data/cp/ltm_hcap_p3v4_*.parquet | xargs -P $SLURM_CPUS_PER_TASK -I{} $PY three_way_causal_seeded.py --one {} cp
$PY three_way_causal_seeded.py --agg cp
echo "end $(date) rc=$?"
