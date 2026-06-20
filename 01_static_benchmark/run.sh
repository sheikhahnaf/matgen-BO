#!/usr/bin/env bash
# Regenerate static-benchmark figures from in-tree data/ (no external downloads).
set -uo pipefail; HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "Static benchmark: generators are in figures/generators/, data in data/."
echo "Run e.g.:  python figures/generators/s13_s14_regen/regen_s13_s14.py"
echo "If a generator errors on a data path, point its data-path constant at \$HERE/data/ (see ../README.md)."
