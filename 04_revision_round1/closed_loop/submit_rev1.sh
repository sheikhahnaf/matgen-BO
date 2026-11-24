#!/bin/bash
# Submit the revision closed-loop arms.
#
#   bash submit_rev1.sh smoke     four short test jobs (both arms, all three backbones)
#   bash submit_rev1.sh cap4      cap-4 for CrystalFlow and ADiT, Cp and K_VRH, 5 seeds (20 jobs)
#   bash submit_rev1.sh frozen    frozen-generator control, 3 backbones, 2 targets, 5 seeds (30 jobs)
#
# Environment: REV_ROOT (required), ACCOUNT (SLURM account, required).

set -euo pipefail
MODE="${1:?usage: submit_rev1.sh smoke|cap4|frozen}"
REV_ROOT="${REV_ROOT:?set REV_ROOT}"
ACCOUNT="${ACCOUNT:?set ACCOUNT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/run_rev1_arm.slurm"
SEEDS=(7 17 23 99 113)
mkdir -p "$REV_ROOT/logs" "$REV_ROOT/runs" "$REV_ROOT/ltm"

submit () {  # arm backbone target seed cycles walltime
    local arm=$1 bb=$2 tgt=$3 seed=$4 cycles=$5 wall=$6
    local name="rev1_${arm}_${bb}_${tgt}_s${seed}"
    sbatch --account="$ACCOUNT" --job-name="$name" --time="$wall" \
        --output="$REV_ROOT/logs/${name}_%j.out" --error="$REV_ROOT/logs/${name}_%j.err" \
        --export=ALL,ARM=$arm,BACKBONE=$bb,TARGET=$tgt,SEED=$seed,CYCLES=$cycles,REV_ROOT=$REV_ROOT \
        "$SCRIPT" | awk -v n="$name" '/Submitted batch job/{print n, $NF}'
}

case "$MODE" in
    smoke)
        submit cap4        cf   cp 7 2 02:00:00
        submit frozen_gate cf   cp 7 2 02:00:00
        submit frozen_gate adit bm 7 2 02:00:00
        submit frozen_gate mg   cp 7 1 03:00:00
        ;;
    cap4)
        for bb in cf adit; do for tgt in cp bm; do for s in "${SEEDS[@]}"; do
            submit cap4 $bb $tgt $s 20 24:00:00
        done; done; done
        ;;
    frozen)
        for bb in cf adit; do for tgt in cp bm; do for s in "${SEEDS[@]}"; do
            submit frozen_gate $bb $tgt $s 20 24:00:00
        done; done; done
        for tgt in cp bm; do for s in "${SEEDS[@]}"; do
            submit frozen_gate mg $tgt $s 20 48:00:00
        done; done
        ;;
    *) echo "unknown mode $MODE"; exit 2 ;;
esac
