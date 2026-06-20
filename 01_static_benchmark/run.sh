#!/usr/bin/env bash
# Regenerate static-benchmark figures from in-tree archive/ data (no external downloads).
# Each generator resolves the repo root via Path(__file__) and reads archive/<source-tree>/...,
# writing into figures/regenerated/ (main) or figures/generators/*/out/ (refresh/s13_s14).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/figures/generators"
echo "== Static benchmark: regenerating figures (figures/regenerated/, generators/*/out/) =="
run() { ( cd "$1" && echo "-- $(basename "$1")/$2" && python "$2" >/dev/null ) && echo "   ok" || echo "   ($2 reported errors — see README.md)"; }
# main elastic / dielectric panels (+ legacy phonon-dielectric)
run "$G" fig2_bar_charts.py
run "$G" heatmap_pca_replotter.py
run "$G" difficulty_radar.py
run "$G" learning_curves.py
# phonon-refresh style (DFPT) — the paper's FINAL phonon figures
run "$G/refresh_style_regen" fig2_bar_charts.py
run "$G/refresh_style_regen" heatmap_pca_replotter.py
run "$G/refresh_style_regen" difficulty_radar.py
run "$G/refresh_style_regen" learning_curves.py
# cross-dataset / cross-surrogate supplementary (S-figures), with published-value self-checks
run "$G/s13_s14_regen" regen_s13_s14.py
echo "Done. Rendered figures under figures/regenerated/ and figures/generators/*/out/."
