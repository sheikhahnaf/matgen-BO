#!/usr/bin/env bash
set -uo pipefail; HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
echo "Closed-loop: generators in figures/generators/, per-trajectory CSVs in results/results-paper-v4/, LTM parquets in data/hcap_data/."
echo "Run e.g.:  python figures/generators/closed_loop_curves.py"
