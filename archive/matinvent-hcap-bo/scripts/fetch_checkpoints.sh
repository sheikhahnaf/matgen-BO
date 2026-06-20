#!/usr/bin/env bash
# Fetch all third-party checkpoints needed to run the MatInvent + diffusion pipelines.
# Idempotent: re-runs skip files that already exist.
#
# Sources (canonical upstream):
#   1. syn_score        : kaist-amsg/Synthesizability-stoi-CGNF (GitHub)        — 100 .pth.tar files, ~1.0 GB
#   2. MatterGen base   : jwchen25/MatInvent/mattergen_base       (HF Hub)      — diffusion prior
#   3. DiffCSP MP20     : jwchen25/MatInvent/diffcsp_mp20         (HF Hub)      — diffusion prior
#   4. ALIGNN per-prop  : jwchen25/MatInvent/prop_pred/alignn/*   (HF Hub)      — property predictors
#   5. eSEN-30M-OAM     : facebook/OMAT24                         (HF Hub)      — Cp oracle (optional)
#
# Layouts expected by MatInvent: see matinvent/rewards/calculators/*/README.md
#
# Usage:
#   bash scripts/fetch_checkpoints.sh              # everything
#   bash scripts/fetch_checkpoints.sh syn_score    # just syn_score
#   bash scripts/fetch_checkpoints.sh mattergen    # just MatterGen
#   bash scripts/fetch_checkpoints.sh esen         # just FairChem oracle

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MATINVENT_ROOT="$REPO_ROOT/matinvent"
TARGET="${1:-all}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing required command '$1'"; exit 1; }; }

# ---------- syn_score (100-bag CGNF ensemble) ----------
fetch_syn_score() {
  local dst="$MATINVENT_ROOT/rewards/calculators/syn_score/model_pt"
  mkdir -p "$dst"
  echo "==> syn_score: 100 .pth.tar files into $dst"
  local base="https://raw.githubusercontent.com/kaist-amsg/Synthesizability-stoi-CGNF/main/models"
  local count=0
  for i in $(seq 1 100); do
    local f="checkpoint_bag_${i}.pth.tar"
    if [[ -s "$dst/$f" ]]; then
      count=$((count+1))
      continue
    fi
    curl -fsSL --retry 3 -o "$dst/$f" "$base/$f" || { echo "FAIL: $f"; rm -f "$dst/$f"; exit 1; }
    count=$((count+1))
    [[ $((count % 10)) -eq 0 ]] && echo "    $count/100"
  done
  echo "    done: $count/100 files"
}

# ---------- HF Hub checkpoints (MatterGen, DiffCSP, ALIGNN) ----------
fetch_hf_matinvent() {
  need_cmd huggingface-cli
  local cache="$MATINVENT_ROOT/.hf_cache/jwchen25_MatInvent"
  mkdir -p "$cache"
  echo "==> jwchen25/MatInvent → $cache"
  huggingface-cli download jwchen25/MatInvent \
    --local-dir "$cache" --local-dir-use-symlinks False --resume-download
  echo "    Symlinking into MatInvent's expected paths:"
  # MatterGen base
  ln -sfn "$cache/mattergen_base" "$MATINVENT_ROOT/src/mattergen/checkpoints/mattergen_base"
  # DiffCSP
  mkdir -p "$MATINVENT_ROOT/models/diffcsp"
  ln -sfn "$cache/diffcsp_mp20" "$MATINVENT_ROOT/models/diffcsp/checkpoints"
  # ALIGNN property predictors
  mkdir -p "$MATINVENT_ROOT/rewards/calculators/alignn"
  ln -sfn "$cache/prop_pred/alignn" "$MATINVENT_ROOT/rewards/calculators/alignn/checkpoints"
  echo "    done."
}

# ---------- eSEN-30M-OAM (FairChem Cp oracle) ----------
fetch_esen() {
  need_cmd huggingface-cli
  local cache="$MATINVENT_ROOT/.hf_cache/facebook_OMAT24"
  mkdir -p "$cache"
  echo "==> facebook/OMAT24 (eSEN-30M-OAM) → $cache"
  huggingface-cli download facebook/OMAT24 \
    --local-dir "$cache" --local-dir-use-symlinks False --resume-download \
    --include "eSEN-30M-OAM*"
  echo "    done. Point FairChem configs at $cache/eSEN-30M-OAM"
}

verify() {
  echo ""
  echo "==> Verification:"
  local synscore_count
  synscore_count=$(find "$MATINVENT_ROOT/rewards/calculators/syn_score/model_pt" -name "checkpoint_bag_*.pth.tar" 2>/dev/null | wc -l | tr -d ' ')
  echo "    syn_score: $synscore_count / 100 .pth.tar files"
  for path in "$MATINVENT_ROOT/src/mattergen/checkpoints/mattergen_base" \
              "$MATINVENT_ROOT/models/diffcsp/checkpoints" \
              "$MATINVENT_ROOT/rewards/calculators/alignn/checkpoints"; do
    if [[ -e "$path" ]]; then echo "    OK: $path"; else echo "    MISSING: $path"; fi
  done
}

case "$TARGET" in
  all)        fetch_syn_score; fetch_hf_matinvent; verify ;;
  syn_score)  fetch_syn_score; verify ;;
  mattergen|hf|matinvent)  fetch_hf_matinvent; verify ;;
  esen|fairchem)           fetch_esen ;;
  verify)     verify ;;
  *) echo "Unknown target: $TARGET (use: all | syn_score | mattergen | esen | verify)"; exit 1 ;;
esac
