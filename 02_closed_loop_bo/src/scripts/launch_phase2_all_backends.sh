#!/bin/bash
# Submit one Phase-2 ablation job per diffusion backend.
#
# Usage:
#   bash scripts/launch_phase2_all_backends.sh                 # all 7
#   bash scripts/launch_phase2_all_backends.sh crystalflow symmcd  # subset
#   CYCLES=5 K=4 bash scripts/launch_phase2_all_backends.sh    # cheaper sweep
#
# Each job is independent and bills T4 hours. Default sweep = 7 × ~3-4 hr each.

set -euo pipefail

# Per audit (docs/DIFFUSION_API_AUDIT.md): only 4 of 7 alternatives have public
# pretrained MP-20 checkpoints + workable APIs. SymmCD/FlowMM/AGeDi shipped no
# public ckpt; AtomGPT only has a Tc-supercon ckpt (property-mismatched for Cp).
DEFAULT_MODELS=(crystalflow crysbfn adit)
# crystalformer deferred — JAX/torch cudnn conflict; needs separate mat-zoo-jax env.

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

CYCLES="${CYCLES:-10}"
BATCH="${BATCH:-64}"
K="${K:-8}"
ANCHOR_EVERY="${ANCHOR_EVERY:-5}"
RUN_TAG="${RUN_TAG:-}"

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
SLURM="$PROJ/scripts/run_phase2_per_backend.slurm"

if [ ! -f "$SLURM" ]; then
  echo "ERROR: $SLURM missing" >&2
  exit 1
fi

echo "Submitting Phase-2 ablation: ${#MODELS[@]} backend(s)"
echo "  CYCLES=$CYCLES  BATCH=$BATCH  K=$K  ANCHOR_EVERY=$ANCHOR_EVERY  RUN_TAG='$RUN_TAG'"
echo "----------------------------------------------------------------"

for MODEL in "${MODELS[@]}"; do
  CKPT_FRAG="$PROJ/configs/diffusion/${MODEL}.yaml"
  if [ ! -f "$CKPT_FRAG" ]; then
    echo "  [SKIP] $MODEL — no fragment at $CKPT_FRAG"
    continue
  fi
  JOB_NAME="p2_${MODEL}${RUN_TAG:+_${RUN_TAG}}"
  EXPORTS="ALL,MODEL=${MODEL},CYCLES=${CYCLES},BATCH=${BATCH},K=${K},ANCHOR_EVERY=${ANCHOR_EVERY}"
  if [ -n "$RUN_TAG" ]; then EXPORTS="${EXPORTS},RUN_TAG=${RUN_TAG}"; fi
  JID="$(sbatch --job-name="$JOB_NAME" --export="$EXPORTS" "$SLURM" | awk '{print $4}')"
  echo "  [SUBMIT] $MODEL  job=$JID  name=$JOB_NAME"
done

echo "----------------------------------------------------------------"
echo "Track with:  squeue -u \$USER --format='%.10i %.20j %.8T %.10M %.6D %R'"
