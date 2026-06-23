#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=cpu
#SBATCH --account=142705333487
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
module purge
module load intel/2022a
module load vasp/6.3.2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export PMG_VASP_PSP_DIR=$SCRATCH/dft_validation/psp
export PMG_DEFAULT_FUNCTIONAL=PBE_54
export OMP_NUM_THREADS=1
cd $SCRATCH/dft_validation
echo "start $(date) CIF=$CIF"
$SCRATCH/envs/matinvent-hcap-bo/bin/python campaign_conv.py "$CIF"
echo "end $(date)"
