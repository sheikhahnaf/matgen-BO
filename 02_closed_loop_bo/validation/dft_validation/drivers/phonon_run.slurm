#!/bin/bash
# Run the full phonon driver for one structure at 64c/24h. Serves two roles:
#   (1) MID-cell RESUME: resubmit a timed-out structure; the idempotent cust() skips banked
#       displacements and finishes the rest, then writes phonon.json.
#   (2) BIG-cell GATHER: submitted with --dependency=afterany on the per-disp array; every disp
#       is already converged, so it skips compute, reads forces, and writes phonon.json (and mops
#       up any displacement the array left unconverged, serially).
#SBATCH --job-name=phrun
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240G
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
echo "start $(date) POSCAR=$POSCAR L=$L MAG=${MAG:-none}"
$SCRATCH/envs/matinvent-hcap-bo/bin/python campaign_phonon.py "$POSCAR" "$L" ${MAG:+$MAG}; rc=$?
echo "end $(date) rc=$rc"
exit $rc
