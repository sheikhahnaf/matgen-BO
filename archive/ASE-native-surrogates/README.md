# ASE-native-surrogates

Static-benchmark code for the foundation-model-embeddings × probabilistic-surrogates benchmark in materials property prediction. Paper: [FoundationalEmbeddings_2026](https://github.com/sheikhahnaf/FoundationalEmbeddings_2026).

## Quick start

Regenerate every figure and table in the paper's static-benchmark sections (Figs 2–7, Tables 1–9, parity plots) from the committed `analysis_v3*/n500/data/*.csv` files:

```bash
git clone https://github.com/sheikhahnaf/ASE-native-surrogates && cd ASE-native-surrogates
conda env create -f env.yml && conda activate matinvent
bash reproduce.sh figures        # ≈30 s; rewrites figures_src/ paths on first run
```

`reproduce.sh` has four targets:

| Target | What it does |
| --- | --- |
| `figures` (default) | Regenerate paper figures from CSVs already in the repo. |
| `sweep` | `sbatch` the full per-(surrogate, featurizer, dataset, n_train) SLURM matrix. |
| `aggregate` | Re-run `analysis_v3*/n500/scripts/aggregate_results.py` after a sweep. |
| `full` | Sweep + aggregate; you re-run `figures` after SLURM completes. |

Full end-to-end (re-run the entire 8,100-fit sweep on a SLURM cluster):

```bash
bash reproduce.sh full           # submits ~30 SLURM jobs; needs ≈100 GPU-hours
bash reproduce.sh aggregate      # after SLURM jobs finish
bash reproduce.sh figures
```

## Structure

```
.
├── common.py                     # featurizer + dataset utilities
├── gp_regression.py              # BoTorch SingleTaskGP runs
├── mtgp_regression.py            # BoTorch MultiTaskGP runs
├── dgp_regression.py             # Multi-task variational DGP runs
├── run_*.slurm                   # per-(surrogate, featurizer, dataset, n_train) batches
├── results/<surrogate>/<dataset>_pca{10,25,50}_n{100,250,500}/
│   └── *_holdout_summary.csv     # per-property R^2, RMSE, Spearman (mean ± std over 5 splits)
├── analysis_v3/                  # elastic-tensor aggregated tables and figures (n100/n250/n500)
├── analysis_v3_dielectric_constant/    # dielectric aggregated tables
├── analysis_v3_phonon_dielectric_mp/   # phonon-dielectric aggregated tables
├── combined_*/                   # cross-n_train learning-curve data
├── _pptx_tmp/                    # presentation-ready summary plots
└── figures_src/                  # paper-figure plot scripts (read analysis_v3*/n500/data/*.csv)
```

## Reproducing the static benchmark

1. Install: `conda env create -f env.yml && conda activate matinvent`
2. Submit a sweep: `sbatch run_gp_soap_orb_uma_sweep.slurm` (similar for mtgp, dgp; per-dataset variants suffixed `_diel`, `_phonon`).
3. Aggregate: scripts under `analysis_v3*/n500/scripts/`.
4. Paper figures: scripts under `figures_src/` consume `analysis_v3*/n500/data/*.csv`.

### Note on hardcoded paths

The four scripts under `figures_src/` (`data_loaders.py`, `learning_curves.py`, `difficulty_radar.py`, `metric_disagreement.py`) carry absolute paths from the development machine (`/Volumes/SSD1_SMAAA/matinvent-bo/...`). Before running them elsewhere, edit the `ROOT`/`REPO`/`REPO_ROOT` constants at the top of each file or:

```bash
sed -i.bak 's|/Volumes/SSD1_SMAAA/matinvent-bo|.|g' figures_src/*.py
```

The regression drivers (`gp/mtgp/dgp_regression.py`, `common.py`) take all paths via CLI flags and do not need editing.

## Cross-factorial design

- 4 featurizers × 3 surrogates × 3 PCA × 3 n_train × 3 datasets × 5 random splits × (8+4+3) properties = 8,100 fits.
- Datasets: matminer's `elastic_tensor_2015`, `dielectric_constant`, `phonon_dielectric_mp`.
- Featurizers: SOAP (DScribe), MACE-MP-0, ORB v3 (`orb_v3_conservative_inf_omat`), UMA-S 1.1 (OMat task).
- Surrogates: BoTorch defaults — GP (RBF + ARD), MTGP (RBF + full-rank ICM), DGP (two-layer Matérn-5/2 + RBF, LMC variational).

See the paper (FoundationalEmbeddings_2026) for the full protocol and results.

## Figure regeneration notes

All benchmark figures regenerate from the committed CSVs via `figures_src/`
(paths are repo-relative; set `ASE_REPO_ROOT` / `ASE_FIG_DIR` to override;
outputs land in `figures/`).

**Known gap (disclosed):** the SI parity panels (per-split holdout scatter for
$K_{VRH}$, band gap, and the phonon DOS peak) cannot be regenerated from the
committed data alone — per-split predictions were not persisted, only the
per-split summary metrics in `results/`. Reproducing those three panels
requires re-running the corresponding `gp_regression.py` configuration
(ORB + GP, PCA=50, n_train=500) to refit and re-predict, after fetching
checkpoints with `./fetch_checkpoints.sh`. All quoted parity $R^2$ values are
the 5-split means recorded in the committed summary CSVs.
