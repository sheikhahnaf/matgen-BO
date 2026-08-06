#!/usr/bin/env bash
# Regenerate synthesizability (A-PU vs CGNF) figures from in-tree results/ (no external downloads).
# Model checkpoints live on HuggingFace (see ../docs/EXTERNAL.md) — not needed for these figures.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/figures/generators"
echo "== Synthesizability: regenerating A-PU vs CGNF figures (figures/regenerated/) =="
( cd "$G" && echo "-- plot_synth_compare.py" && python plot_synth_compare.py ) \
  && echo "   ok" || echo "   (plot_synth_compare.py reported errors — see README.md)"
# Final manuscript versions (corrected OOD flag; main-text Fig 10, SI Fig S18).
# Needs the trained ORB-PU model: in-tree copy, $MATGEN_BO_APU_MODEL, or auto-download
# from HuggingFace (SheikhAhnaf/apu-synthesizability-checkpoints, ~139 MB).
( cd "$G/oodfix_regen" && echo "-- make_synth_oodfix.py" && python make_synth_oodfix.py ) \
  && echo "   ok" || echo "   (make_synth_oodfix.py reported errors — model download may require network; see header)"
echo "Done. Rendered figures under figures/regenerated/."
