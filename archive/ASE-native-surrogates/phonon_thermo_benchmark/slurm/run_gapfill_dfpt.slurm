#!/bin/bash
#SBATCH --job-name=gapfill_dfpt
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=156192594849
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Re-run the 9 Arm-A (DFPT) cells that died with "OSError: [Errno 122] Disk quota
# exceeded" when $SCRATCH was pinned at its 250k-inode limit by the Zenodo extraction.
# Those failures were a transient infrastructure fault, not a code/science problem; the
# inode quota is now clear (~85k free), so these simply need to be re-run to complete the
# already-approved sweep. The DFPT feature cache is already fully populated on $SCRATCH
# (1253 .npy per descriptor), so this reuses it and adds essentially no new inodes.
#
# Missing cells (model / descriptor / pca / n_train), all n_splits=5:
#   MTGP: mace@(10,250) mace@(10,500) mace@(25,100) mace@(25,500)
#         soap@(25,100) orb@(25,100) uma@(25,100)
#   DGP:  mace@(10,250) mace@(25,100)

module purge
module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000

# HF_TOKEN is loaded from a user-provided .env file (never hardcoded; see .env.example).
set -a; [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ] && . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
[ -z "${HF_TOKEN:-}" ] && echo "[warn] HF_TOKEN not set and no .env found; gated HF downloads may fail" >&2

export FEAT_CACHE=/scratch/user/u.sa119259/phonon_thermo_benchmark/data/feat_cache

ROOT=/scratch/user/u.sa119259/phonon_thermo_benchmark
DS=$ROOT/data/dfpt_phonon_thermo.pkl
SRC=$ROOT/src
PY_MACE=/scratch/user/u.sa119259/envs/ase-test-mace/bin/python
PY_FAIR=/scratch/user/u.sa119259/envs/ase-test/bin/python

# Each entry: <model> <descriptor> <pca> <ntrain>   (model in {mtgp,dgp})
CELLS=(
  "mtgp mace 10 250"
  "mtgp mace 10 500"
  "mtgp mace 25 100"
  "mtgp mace 25 500"
  "mtgp soap 25 100"
  "mtgp orb  25 100"
  "mtgp uma  25 100"
  "dgp  mace 10 250"
  "dgp  mace 25 100"
)

for entry in "${CELLS[@]}"; do
  set -- $entry
  MODEL=$1; FEAT=$2; PCA=$3; N=$4
  case $FEAT in
    mace|soap) PY=$PY_MACE ;;
    orb|uma)   PY=$PY_FAIR ;;
  esac
  case $MODEL in
    mtgp) SCRIPT=mtgp_regression.py; OUT=mtgp_2 ;;
    dgp)  SCRIPT=dgp_regression.py;  OUT=dgp ;;
  esac
  echo "=== GAPFILL ${MODEL} ${FEAT} | pca=${PCA} n_train=${N} ==="
  $PY $SRC/$SCRIPT \
      --dataset $DS \
      --pca-components ${PCA} \
      --n-train ${N} \
      --n-splits 5 \
      --descriptor ${FEAT} \
      --output-dir $ROOT/results/${OUT}/dfpt_pca${PCA}_n${N} \
      --device cuda
done

echo "=== gapfill_dfpt SWEEP COMPLETE ==="
