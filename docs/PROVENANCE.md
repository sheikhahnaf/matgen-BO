# Provenance — where each part came from

This repo was assembled (additively, `mkdir`+`cp`) from several working repos/dirs. The curated `01–03/` dirs hold the reproduction-facing subset; the verbatim full source trees live under `archive/` (completeness guarantee).

## Curated subsystems (`01–03/`)
| Destination | Source |
|---|---|
| `01_static_benchmark/src/{gp,mtgp,dgp}_regression.py, common.py` | `ASE-native-surrogates` (ASE_regression_test) |
| `01_static_benchmark/src/phonon_thermo/` | `phonon_thermo_benchmark/src/` |
| `01_static_benchmark/data/analysis_v3*, arm_a_dfpt, arm_b_pheasy` (CSVs) | ASE `analysis_v3*` + `phonon_thermo_benchmark/.../arm_*` |
| `01_static_benchmark/data/*.pkl` | ASE `phonon_thermo_benchmark/data/{dfpt,pheasy}_phonon_thermo.pkl` |
| `01_static_benchmark/figures/generators/` | `FME_paper_refresh_v1/figures_src/` + phonon `s13_s14_regen`, `refresh_style_regen` |
| `01_static_benchmark/figures/rendered/` | `FoundationalEmbeddings_2026/figures/` |
| `02_closed_loop_bo/{src,data,results}` | `matinvent-BO/hcap_bo/` (mbo-publish) — closed-loop code, LTM parquets, `results-paper-v4/` |
| `02_closed_loop_bo/figures/generators/` | `FME_paper_refresh_v1/figures_src/closed_loop_*.py, oracle_savings.py` |
| `03_synthesizability/src/apu_synthesizability/` | `hcap_bo/src/apu_synthesizability/` |
| `03_synthesizability/results/` | `hcap_bo/results/apu_grace/` (metrics/leaderboard; checkpoints external) |
| `shared/` | `figures_src/plot_style.py, data_loaders.py` + ASE `common.py` |

## `archive/` (verbatim, complete)
- `ASE-native-surrogates/`, `phonon_thermo_benchmark/`, `FME_paper_refresh_v1/`, `hcap_bo/`, `matinvent-hcap-bo/`

## Intentional exclusions
- **Copyrighted PDFs** — never published: the `…/data/diffusion_reviews/` review papers, plus a third-party Nature Machine Intelligence article (`s42256-…pdf`) and the working paper draft (`main.pdf`), both purged from history. All cited in the paper.
- **Raw static-benchmark per-run results** (`archive/ASE-native-surrogates/results/`, ≈9 GB) — regenerable; the aggregated CSVs every figure reads are in-tree. Full raw set is public in the `sheikhahnaf/ASE-native-surrogates` repo. See `EXTERNAL.md`.
- **ACES results pull** (≈6.7 GB) — regenerable; aggregated CSVs are in-tree. See `EXTERNAL.md`.
- **Model checkpoints** (`*.joblib`, `*.ckpt`) — on HuggingFace. See `EXTERNAL.md`.
