#!/usr/bin/env bash
# End-to-end reproduction driver for the static-benchmark paper figures.
#
# Three modes:
#   bash reproduce.sh figures     # regenerate paper figures from CSVs already in this repo
#   bash reproduce.sh sweep       # re-run the full sweep (108 configs × 5 splits × 3 datasets)
#   bash reproduce.sh full        # sweep + aggregate + figures (≈100s of GPU-hours)
#
# The default 'figures' target uses the analysis_v3*/n500/data/*.csv files committed
# to the repository, so the paper figures can be regenerated in seconds without
# re-running the GPU sweep.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-figures}"

case "$TARGET" in
  figures)
    echo "==> Regenerating paper figures from committed analysis_v3*/n500/data/*.csv"
    # Figure scripts use repo-relative paths (override with ASE_REPO_ROOT).
    cd "$REPO_ROOT/figures_src"
    for script in fig2_bar_charts.py heatmap_pca_replotter.py learning_curves.py difficulty_radar.py metric_disagreement.py; do
      echo "    --> $script"
      python "$script"
    done
    echo "==> Figures written to figures_src/.../  (see each script for exact output paths)"
    ;;
  sweep)
    echo "==> Submitting the full SLURM sweep (3 surrogates × 4 featurizers × 3 datasets × 3 n_train)"
    echo "    This requires a SLURM cluster + conda env 'matinvent'."
    cd "$REPO_ROOT"
    for s in run_gp_*.slurm run_mtgp_*.slurm run_dgp_*.slurm; do
      [[ -f "$s" ]] || continue
      echo "    sbatch $s"
      sbatch "$s"
    done
    echo "==> Submitted. Monitor with 'squeue -u \$USER'. Re-run with 'reproduce.sh full' after all jobs complete."
    ;;
  full)
    bash "$0" sweep
    echo "==> Wait for SLURM jobs to complete, then re-run 'reproduce.sh aggregate' and 'reproduce.sh figures'"
    ;;
  aggregate)
    cd "$REPO_ROOT"
    for d in analysis_v3 analysis_v3_dielectric_constant analysis_v3_phonon_dielectric_mp; do
      [[ -d "$d/n500/scripts" ]] || continue
      echo "==> $d/n500/scripts/aggregate_results.py"
      (cd "$d" && python n500/scripts/aggregate_results.py)
    done
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash reproduce.sh {figures|sweep|aggregate|full}"
    exit 1
    ;;
esac
