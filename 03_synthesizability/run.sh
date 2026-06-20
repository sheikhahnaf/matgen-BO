#!/usr/bin/env bash
# Regenerate synthesizability (A-PU vs CGNF) figures from in-tree results/ (no external downloads).
# Model checkpoints live on HuggingFace (see ../docs/EXTERNAL.md) — not needed for these figures.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/figures/generators"
echo "== Synthesizability: regenerating A-PU vs CGNF figures (figures/regenerated/) =="
( cd "$G" && echo "-- plot_synth_compare.py" && python plot_synth_compare.py ) \
  && echo "   ok" || echo "   (plot_synth_compare.py reported errors — see README.md)"
echo "Done. Rendered figures under figures/regenerated/."
