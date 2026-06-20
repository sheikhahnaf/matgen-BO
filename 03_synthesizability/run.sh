#!/usr/bin/env bash
set -uo pipefail; HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "Synthesizability: generators in figures/generators/, metrics/leaderboard/scores in results/."
echo "Checkpoints are on HuggingFace (see ../docs/EXTERNAL.md)."
echo "Run e.g.:  python figures/generators/plot_synth_compare.py"
