#!/usr/bin/env bash
# Fetch the external artifacts (raw data + checkpoints) needed only for a full from-scratch rerun.
# Figure regeneration does NOT need these — the figure-driving CSVs + prepared pkls are in-tree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$HERE/external"
mkdir -p "$EXT"

echo "[1/2] HuggingFace checkpoints (requires: pip install huggingface_hub)"
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download SheikhAhnaf/apu-synthesizability-checkpoints --local-dir "$EXT/apu-synthesizability-checkpoints" || \
    echo "  (skip) could not download apu checkpoints"
else
  echo "  huggingface-cli not found — install with: pip install huggingface_hub"
fi

echo "[2/2] Zenodo raw phonon deposit (record 20196565, ~79 GB — large!)"
echo "  Browse/download: https://doi.org/10.5281/zenodo.20196565"
echo "  (Not auto-downloaded by default due to size. To fetch the manifest:)"
echo "    curl -L -o '$EXT/zenodo_files.json' 'https://zenodo.org/api/records/20196565'"

echo "Done. External artifacts (when fetched) live under: $EXT"
