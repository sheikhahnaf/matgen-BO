#!/usr/bin/env bash
# End-to-end reproduction driver for the closed-loop diffusion + BO paper figures.
#
# Targets:
#   bash reproduce.sh figures        # regenerate closed-loop figures from hcap_bo/analysis/ CSVs
#   bash reproduce.sh checkpoints    # bash scripts/fetch_checkpoints.sh
#   bash reproduce.sh sweep          # submit Phase-3 SLURM batches (3 backbones × 2 policies × 2 targets × 5 seeds)
#   bash reproduce.sh aggregate      # rebuild top_per_job.csv + global_top20.csv + v4_metrics_plots/
#   bash reproduce.sh full           # checkpoints + sweep (run aggregate + figures after SLURM jobs finish)
#
# Default 'figures' regenerates paper plots from the analysis CSVs committed under
# hcap_bo/analysis/, so a fresh clone can reproduce the closed-loop figures in
# seconds without re-running any SLURM jobs.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-figures}"

case "$TARGET" in
  figures)
    echo "==> Regenerating closed-loop figures from committed hcap_bo/results-paper-v4 + hcap_bo/analysis data"
    # Figure scripts use repo-relative paths (override with MBO_REPO_ROOT / MBO_RESULTS_ROOT / MBO_FIG_DIR).
    cd "$REPO_ROOT/figures_src"
    for script in closed_loop_curves.py closed_loop_extras.py oracle_savings.py mg_ablation_plots.py synth_compare.py; do
      echo "    --> $script"
      python "$script"
    done
    echo "==> Figures written to figures/"
    ;;
  checkpoints)
    bash "$REPO_ROOT/scripts/fetch_checkpoints.sh" all
    ;;
  sweep)
    echo "==> Submitting Phase-3 SLURM batches (Cp + K_VRH)"
    echo "    Requires SLURM + conda envs 'matinvent' and 'matinvent-hcap-fairchem'."
    cd "$REPO_ROOT/hcap_bo"
    for s in scripts/v4/run_phase3_v4_*.slurm scripts/v4_bm/run_phase3_v4_bm_*.slurm; do
      [[ -f "$s" ]] || continue
      echo "    sbatch $s"
      sbatch "$s"
    done
    echo "==> Submitted. Monitor with 'squeue -u \$USER'."
    ;;
  aggregate)
    cd "$REPO_ROOT/hcap_bo"
    echo "==> Rebuilding analysis/top_structures/top_per_job.csv and global_top20.csv"
    python analysis/top_structures/analyze_top.py
    echo "==> Rebuilding analysis/v4_metrics_plots/"
    python analysis/v4_metrics_plots/plot_metrics.py
    ;;
  full)
    bash "$0" checkpoints
    bash "$0" sweep
    echo "==> Wait for SLURM jobs to complete, then re-run 'reproduce.sh aggregate' and 'reproduce.sh figures'"
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash reproduce.sh {figures|checkpoints|sweep|aggregate|full}"
    exit 1
    ;;
esac
