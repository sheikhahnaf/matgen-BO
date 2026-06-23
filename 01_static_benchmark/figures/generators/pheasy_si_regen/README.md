# Pheasy (11,818) — "fourth dataset" SI figure set

The same per-dataset benchmark plots the paper shows for elastic / dielectric / phonon
(DFPT), reproduced here for the **large Pheasy phonon-thermodynamics dataset** (11,818
materials, 4 targets: Cv₃₀₀, S₃₀₀, F₃₀₀, ω_max), now including **DGP** alongside GP and
MTGP. Intended as an SI "big dataset" example. **Not yet wired into the paper SI.**

## Figures (`out_pheasy/`)
| File | Paper analogue |
|---|---|
| `fig2_bar_pheasy_R2_grouped.png` | Fig 12 — grouped bar, avg R² @ n_train=500, best PCA per surrogate |
| `fig_heatmap_pheasy_R2_n500.pdf` | Fig 13 — featurizer × surrogate R² heatmap |
| `fig_pca_pheasy_R2_n500.pdf` | S7 — PCA sensitivity |
| `fig_difficulty_pheasy_n500.pdf` | Fig 14 — per-property difficulty (3 surrogate panels) |
| `fig_radar_pheasy_orb_R2_n500.pdf` | S8 — ORB radar across the 4 targets |
| `fig_combined_pheasy_{R2,Spearman}_learning_curve.png` | S9/S10 — learning curves |

## Data source
Generators read `../../../data/arm_b_pheasy_full/` (repo-relative via `_ROOT`), a
DGP-inclusive rebuild of the pheasy analysis dir:
- raw per-cell results staged under `../../../data/_pheasy_full_build/results/{gp,mtgp_2,dgp}/`
  (gp & mtgp_2 from the local pull; **dgp pulled from ACES** after the n100–n1000 DGP runs);
- re-aggregated with all three models (`aggregate_pheasy_full.py`) → `aggregated_results.csv`;
- per-n derived tables + combined learning curves rebuilt via the source pipeline
  (`build_pheasy_full_pern.py`, `build_pheasy_full_combined.py`).

n_train coverage: GP 100–2000, MTGP/DGP 100–1000 (DGP pheasy was not run at n2000).

## Caveats (for the SI text)
- **DGP collapses at PCA=50** (R²≈0 for every descriptor) — see the PCA-sensitivity panel.
  DGP's best PCA is 25, so the best-PCA figures (bar/heatmap/radar) are unaffected; the
  sensitivity plot honestly shows the pca50 failure (likely variational posterior collapse).
- **RMSE learning curve dropped**: averaging RMSE across the 4 heterogeneous-magnitude
  targets (F ≫ others) is uninterpretable. R² and Spearman are retained.

## Regenerate
```bash
conda activate matinvent
cd 01_static_benchmark/figures/generators/pheasy_si_regen
python fig2_bar_charts.py && python heatmap_pca_replotter.py \
  && python difficulty_radar.py && python learning_curves.py
```
Outputs land in `out_pheasy/` (reference figures never overwritten; re-runs archive to `out_pheasy/legacy/`).
