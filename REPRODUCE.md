# Reproduce — paper artifact → command

All inputs are in-tree (no external downloads). Activate the env first:
`conda activate matinvent`. Generators live in `0*/figures/generators/`, read the
CSVs in `0*/data/`, and write to `0*/figures/rendered/`.

## Static benchmark (§3.2–3.4)
| Paper artifact | Generator | Inputs (in-tree) |
|---|---|---|
| Fig 12 (bar R², 3 datasets) | `01_static_benchmark/figures/generators/fig2_bar_charts.py` | `01_static_benchmark/data/{analysis_v3*,arm_*}` |
| Fig 13 / S6 (R² heatmaps) | `.../generators/heatmap_pca_replotter.py` | same |
| Fig 14 (property difficulty) | `.../generators/difficulty_radar.py` | same |
| S7 (PCA sensitivity), S8 (radar) | `.../generators/{heatmap_pca_replotter,difficulty_radar}.py` | same |
| S9/S10 (learning curves) | `.../generators/learning_curves.py` | `.../data/*/combined/data/learning_curves_orb.csv` |
| S12/S13 (cross-dataset / cross-surrogate) | `.../generators/s13_s14_regen/regen_s13_s14.py` | `.../data/{analysis_v3*,arm_a_dfpt}` |
| Tables 3–5 (best configs) | derive from `.../data/*/aggregated_results.csv` (n_train=500) | in-tree CSVs |
| Phonon-style refresh figs | `.../generators/refresh_style_regen/*.py` | `.../data/arm_a_dfpt` |

## Closed-loop BO (§3.1)
| Paper artifact | Generator | Inputs |
|---|---|---|
| Closed-loop discovery curves | `02_closed_loop_bo/figures/generators/closed_loop_curves.py` | `02_closed_loop_bo/results/results-paper-v4/.../long_term_memory.csv` |
| Oracle-savings / per-cycle | `.../generators/oracle_savings.py`, `closed_loop_extras.py` | same + `02_closed_loop_bo/data/hcap_data/bm/*.parquet` |

## Synthesizability (§2.7)
| Paper artifact | Generator | Inputs |
|---|---|---|
| Fig 11 (per-backbone/policy synth + A-PU vs CGNF) | `03_synthesizability/figures/generators/plot_synth_compare.py` | `03_synthesizability/results/` (metrics/leaderboard/scores) |
| Table S4/S5 (Optuna leaderboard, CGNF vs ORB-PU) | derive from `03_synthesizability/results/` CSVs | in-tree |
