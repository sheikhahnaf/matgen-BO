# Arm A (DFPT) — results-analysis pipeline

Adapted from `ASE_regression_test/analysis_v3_phonon_dielectric_mp/` (read-only original).

## aggregate_results.py (validated, runs clean)
- `DATASET_PREFIX = 'dfpt'`
- `MODELS = ['gp', 'mtgp_2', 'dgp']` (DGP IS evaluated on Arm A)
- Walks `../results/<model>/dfpt_pca<P>_n<N>/<model>_<descriptor>_holdout_summary.csv`
- Coverage grid (missing-combo report only): descriptors `[soap,mace,orb,uma]`,
  PCA `[10,25,50]`, n_train per-model = `[100,250,500]`.
- `identify_missing_combinations` only iterates over models actually present in
  the data, so a partial run does not flag absent models.
- Smoke-test CSVs (`results/smoke/`, `smoke_patch/`, `aces_smoke/`) are excluded
  because their parent dir is `gp`/`aces_smoke`, not a `dfpt_*` tag dir.

Run:
```
eval "$(/Users/alvi/miniconda3/bin/conda shell.zsh hook)" && conda activate matinvent \
  && cd /Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark \
  && python analysis_dfpt/aggregate_results.py
```
Outputs `analysis_dfpt/aggregated_results.csv` + `data_summary.txt`.

## figure_scripts_staged/ (NOT yet adapted — figures deferred until full results)
Verbatim copies of the original per-n and combined learning-curve scripts. They
are hardcoded to the OLD dataset and must be adapted before use:
- `get_property_list()` must return `['Cv_300K','S_300K','F_300K','max_phonon_freq']`
  (old: `['eps_electronic','eps_total','last phdos peak']`).
- The hardcoded `n=100` filter / `n100` filename suffix must be parameterized per
  n-value (Arm A: 100/250/500).
- Model lists `['gp','mtgp_2','dgp']` are already correct for Arm A.
- `prepare_data.py` expected-row counts assume 3 properties × 3 metrics — update to
  4 properties.
