#!/bin/bash
#SBATCH --job-name=gpcal
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# Static-benchmark GP calibration rerun for one dataset.
# Same grid as the published GP sweep: PCA in {10,25,50}, n_train in {100,250,500},
# five splits, descriptors SOAP / MACE / ORB / UMA.
#
# Required (sbatch --export):
#   KIND      elastic | dielectric | dfpt
#   REV_ROOT  directory holding calibration/ and static_src/ ; outputs in $REV_ROOT/calibration/out
# Optional:
#   DESCRIPTORS  space-separated subset (default "orb soap mace uma")
#   NTRAINS      default "500 250 100"

set -uo pipefail
module purge
module load CUDA/12.4.0 GCC/13.2.0 WebProxy/0000

if [ -f "$SCRATCH/.secrets/hf_token.env" ]; then
    set -a; . "$SCRATCH/.secrets/hf_token.env"; set +a
fi
export MATMINER_DATA="${MATMINER_DATA:-$SCRATCH/matminer_data}"
export PYTHONNOUSERSITE=1
export FEAT_CACHE="${TMPDIR:-/tmp}/feat_cache_$SLURM_JOB_ID"

REV_ROOT="${REV_ROOT:?set REV_ROOT}"
KIND="${KIND:?set KIND=elastic|dielectric|dfpt}"
DESCRIPTORS="${DESCRIPTORS:-orb soap mace uma}"
NTRAINS="${NTRAINS:-500 250 100}"

# SOAP and MACE run in the MACE environment, ORB and UMA in the fairchem one
# (same assignment as the published sweeps).
PY_MACE="${PY_MACE:-$SCRATCH/envs/ase-test-mace/bin/python}"
PY_FAIRCHEM="${PY_FAIRCHEM:-$SCRATCH/envs/ase-test/bin/python}"
DRIVER="$REV_ROOT/calibration/gp_calibration_rerun.py"

case "$KIND" in
    elastic)    SRC="$REV_ROOT/static_src";               DATASET=elastic_tensor_2015 ;;
    dielectric) SRC="$REV_ROOT/static_src";               DATASET=dielectric_constant ;;
    dfpt)       SRC="$REV_ROOT/static_src/phonon_thermo"; DATASET="$REV_ROOT/static_data/dfpt_phonon_thermo.pkl" ;;
    *) echo "unknown KIND=$KIND"; exit 2 ;;
esac

OUT="$REV_ROOT/calibration/out/$KIND"
mkdir -p "$OUT"
cd "$OUT"
echo "=== GP calibration rerun kind=$KIND descriptors=[$DESCRIPTORS] n=[$NTRAINS] $(date) $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for N in $NTRAINS; do
  for PCA in 50 25 10; do
    for D in $DESCRIPTORS; do
        case "$D" in soap|mace) PY="$PY_MACE" ;; *) PY="$PY_FAIRCHEM" ;; esac
        CELL="$OUT/pca${PCA}_n${N}"
        if [ -f "$CELL/calib_preds_${D}.csv.gz" ]; then
            echo "--- skip (done): $KIND $D pca=$PCA n=$N"; continue
        fi
        echo "--- $KIND $D pca=$PCA n=$N  $(date +%T)"
        $PY -u "$DRIVER" "$SRC" \
            --dataset "$DATASET" \
            --pca-components $PCA \
            --n-train $N \
            --n-splits 5 \
            --descriptor $D \
            --output-dir "$CELL" \
            --device cuda \
            || echo "!!! FAILED: $KIND $D pca=$PCA n=$N"
    done
  done
done
echo "=== GP calibration rerun $KIND finished $(date) ==="
