#!/bin/bash
# Per-displacement phonon worker for SLURM job arrays (one displacement per array task).
# Submit with --array=0-(N-1) and --export=ALL,POSCAR=...,L=12,MAG=... (see submit_cp_phonon.sh).
#SBATCH --job-name=pha
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --partition=cpu
#SBATCH --account=142705333487
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
module purge
module load intel/2022a
module load vasp/6.3.2
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export PMG_VASP_PSP_DIR=$SCRATCH/dft_validation/psp
export PMG_DEFAULT_FUNCTIONAL=PBE_54
export OMP_NUM_THREADS=1
cd $SCRATCH/dft_validation
echo "start $(date) POSCAR=$POSCAR L=$L MAG=${MAG:-none} disp=$SLURM_ARRAY_TASK_ID"
$SCRATCH/envs/matinvent-hcap-bo/bin/python phonon_disp_array.py "$POSCAR" "$L" "${MAG:-none}" "$SLURM_ARRAY_TASK_ID"; rc=$?
echo "end $(date) rc=$rc"
exit $rc
