# Arm B (Pheasy) — results-analysis pipeline

Adapted from `ASE_regression_test/analysis_v3_phonon_dielectric_mp/` (read-only original).

## aggregate_results.py (validated, runs clean — currently 0 rows, expected)
- `DATASET_PREFIX = 'pheasy'`
- `MODELS = ['gp', 'mtgp_2']` (NO DGP on Arm B)
- Walks `../results/<model>/pheasy_pca<P>_n<N>/<model>_<descriptor>_holdout_summary.csv`
- Coverage grid (missing-combo report only): descriptors `[soap,mace,orb,uma]`,
  PCA `[10,25,50]`, n_train per-model:
  - GP   : `[100,250,500,1000,2000]`
  - MTGP : `[100,250,500,1000]` (no n=2000)
- `identify_missing_combinations` is tolerant: per-model n_train grid, and it only
  iterates over models actually present, so neither DGP nor n=2000 MTGP is ever
  expected.
- When no pheasy results exist yet, `load_all_results` returns an empty (correctly
  typed) DataFrame instead of raising, writes an empty `aggregated_results.csv`, and
  the summary reads "No results aggregated yet". This is the current state.

Run:
```
eval "$(/Users/alvi/miniconda3/bin/conda shell.zsh hook)" && conda activate matinvent \
  && cd /Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark \
  && python analysis_pheasy/aggregate_results.py
```

## figure_scripts_staged/ (NOT yet adapted — figures deferred until full results)
Verbatim copies of the original per-n and combined learning-curve scripts. They
are hardcoded to the OLD dataset and 3 models, and must be adapted before use:
- `get_property_list()` must return `['Cv_300K','S_300K','F_300K','max_phonon_freq']`.
- Model lists `['gp','mtgp_2','dgp']` must drop `dgp` for Arm B.
- The hardcoded `n=100` filter / `n100` filename suffix must be parameterized per
  n-value (Arm B: 100/250/500/1000 + GP-only 2000).
- The radar/learning-curve "ORB only" assumption from the original may not hold;
  revisit once full results land.
