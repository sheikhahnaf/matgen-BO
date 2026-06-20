#!/usr/bin/env bash
# Submit the MatterGen oracle-budget ablation sweep.
#
# Arms (all pristine MatterGen, only sample_cfg.max_num varies):
#   cap4 / cp   : gen-64-cap-4   on heat capacity   (cap binds; 5 seeds)
#   cap4 / bm   : gen-64-cap-4   on bulk modulus    (cap binds; 5 seeds)
#   oracleall/bm: gen-64-oracle-all on bulk modulus (cap binds; 5 seeds)
#
# NOT submitted: oracleall/cp — the cap never binds for Cp (survival ~9 < 16),
# so it is bit-identical to the existing BASE-Cp runs. Reuse BASE-Cp for that cell.
#
# Seeds match the existing BASE/ACC runs for paired comparison.
# Each job uses a distinct expname (hcap_mgabl_<arm>_<target>_seed<seed>_<jobid>)
# and writes to results/ or results_bm/ — never overwrites baseline/accel dirs.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM="$HERE/run_mg_ablation.slurm"
SEEDS=(7 17 23 99 113)
DRY="${DRY_RUN:-0}"   # DRY_RUN=1 prints sbatch lines without submitting

submit () {
    local target="$1" cap="$2"
    for s in "${SEEDS[@]}"; do
        local cmd=(sbatch --export=ALL,TARGET="$target",CAP="$cap",SEED="$s",CYCLES=20 "$SLURM")
        if [ "$DRY" = "1" ]; then
            echo "[dry-run] ${cmd[*]}"
        else
            echo "submitting: target=$target cap=$cap seed=$s"
            "${cmd[@]}"
            sleep 1
        fi
    done
}

echo "=== cap-4 on Cp (5 seeds) ==="
submit cp 4
echo "=== cap-4 on K_VRH (5 seeds) ==="
submit bm 4
echo "=== oracle-all on K_VRH (5 seeds) ==="
submit bm 100000

echo "Done. 15 jobs total (5 cap4-cp + 5 cap4-bm + 5 oracleall-bm)."
echo "Verify with: squeue -u \$USER -o '%.18i %.20j %.8T %.10M %R'"
