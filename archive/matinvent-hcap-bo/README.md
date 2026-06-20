# matinvent-BO

Closed-loop, surrogate-gated diffusion for Bayesian materials discovery. Paper: [FoundationalEmbeddings_2026](https://github.com/sheikhahnaf/FoundationalEmbeddings_2026).

The repository couples a pretrained diffusion prior (MatterGen, CrystalFlow, ADiT), an online REINFORCE-style policy-gradient update, and a Gaussian-process Expected-Improvement gate that caps oracle calls at top-K per cycle.

## Quick start

Regenerate every figure and table in the paper's closed-loop sections (Figs 8–12, Tables 10–11) from the committed `hcap_bo/analysis/` CSVs:

```bash
git clone https://github.com/sheikhahnaf/matinvent-BO && cd matinvent-BO
conda env create -f matinvent/env.yml && conda activate matinvent
bash reproduce.sh figures        # ≈30 s; rewrites hardcoded paths on first run
```

`reproduce.sh` has five targets:

| Target | What it does |
| --- | --- |
| `figures` (default) | Regenerate closed-loop figures from CSVs already in the repo. |
| `checkpoints` | `bash scripts/fetch_checkpoints.sh all` — pulls syn_score + MatterGen + ALIGNN + eSEN from upstream. |
| `sweep` | `sbatch` the Phase-3 SLURM batches (3 backbones × 2 policies × 2 targets × 5 seeds). |
| `aggregate` | Rebuild `hcap_bo/analysis/top_structures/top_per_job.csv`, `global_top20.csv`, and `v4_metrics_plots/`. |
| `full` | `checkpoints` + `sweep`; re-run `aggregate` and `figures` after SLURM finishes. |

Full end-to-end (re-run the closed-loop probe on a SLURM cluster with GPUs):

```bash
bash reproduce.sh checkpoints    # pulls ~1.3 GB of third-party weights
bash reproduce.sh sweep          # submit Phase-3 SLURM batches
bash reproduce.sh aggregate      # after SLURM jobs finish
bash reproduce.sh figures
```

## Structure

```
.
├── matinvent/               # modified MatInvent codebase (Cao et al. lineage, our edits)
│   ├── pipeline/            # SUN filter, RL loop, replay buffer, diversity filter
│   ├── rewards/             # property oracles (eSEN+phonopy for Cp, Birch–Murnaghan EOS for K_VRH)
│   │   ├── calculators/{orb,fairchem,dft,alignn,pymatgen}/
│   │   └── gp/              # GP-routed acquisition (EI + DPP, top-K)
│   ├── models/              # diffusion-prior adapters (MatterGen, CrystalFlow, ADiT, …)
│   ├── configs/             # Hydra YAMLs per (target, backbone, policy)
│   ├── src/mattergen/       # MatterGen source (checkpoints excluded; download separately)
│   ├── scripts/             # auxiliary tooling
│   └── tests/               # adapter and pipeline unit tests
│
├── hcap_bo/                 # closed-loop heat-capacity + bulk-modulus pipeline
│   ├── src/                 # ORB+PCA50+GP+EI+DPP gate; LocalESEN_GPRoutedV4 calculator
│   ├── scripts/             # SLURM batches for v3 and v4 sweeps
│   ├── configs/             # reward (Cp, K_VRH), pipeline, GP routing
│   ├── analysis/            # top_structures, v4_metrics_plots, top_per_job aggregation
│   ├── results-paper/       # curated final results used in the paper
│   ├── docs/                # integration decks and figures
│   ├── tests/               # pytest suite
│   └── COMMITTEE_REVIEW.md  # design review log
│
├── figures_src/             # closed-loop plot scripts (read hcap_bo/analysis/*)
│   ├── closed_loop_curves.py        # discovery curves + CV5 RMSE per cycle
│   ├── closed_loop_extras.py        # crystal-system distributions + per-backbone summary
│   ├── oracle_savings.py            # cumulative oracle-call curves (BASE vs ACC)
│   ├── regen_bogen.py               # one-shot regen helper
│   ├── plot_style.py                # shared rcParams
│   └── data_loaders.py              # shared CSV loaders
│
└── pristine_matinvent/      # reference unmodified MatInvent (diff against `matinvent/`)
```

## What's in the paper

The diffusion / closed-loop probe (§3.7, §4.8, §5.7–§5.8 of the paper) reports:
- Three pretrained backbones × two policies (BASE = vanilla REINFORCE, ACC = GP-EI-DPP-gated) × two targets (Cp, K_VRH) × five seeds = **59 completed trajectories** (one ADiT/ACC Cp seed did not complete).
- ACC matches or exceeds BASE on MatterGen and ADiT for both targets; CrystalFlow is noisier.
- The gate caps oracle calls at K=4 per cycle (92 calls per 20-cycle run including warm-start), reducing oracle cost by **1.8–3.5×** for backbones whose unfiltered SUN-survival rate exceeds K, and bounding rather than monotonically reducing it for the rest.

## Synthesizability scoring (ORB-PU / A-PU)

`hcap_bo/src/apu_synthesizability/` adds a positive-unlabeled (PU) synthesizability
scorer for inorganic compositions, trained on 109,283 Materials Project entries
(49,283 labeled positives, 60,000 unlabeled). Features are ORB-v3 embeddings
(PCA-reduced) and Magpie composition descriptors; the estimator is a Mordelet–Vert
PU bagging classifier with an optional abstention / out-of-distribution layer.
Hyperparameters are tuned with Optuna (5-fold cross-validation, AUPRC objective). On
a held-out MP split the selected model (ORB+Magpie XGBoost) reaches AUPRC 0.961,
AUROC 0.967, ECE 0.024. It is evaluated against the pretrained CGNF stoichiometry
model (Jang et al., *Matter* 2024) on the same split and on generated structures.
Tables, figures, and the analysis are in `hcap_bo/syn_finding.md` and
`hcap_bo/analysis/synth_figures/`.

Checkpoints: all eight tuned models (3.6 GB), including the selected
`apu_optuna/orb_mag__xgboost/model.joblib`, are archived in a private Hugging Face
repository
[SheikhAhnaf/apu-synthesizability-checkpoints](https://huggingface.co/SheikhAhnaf/apu-synthesizability-checkpoints)
(access required), together with the training-bank PCA (`cache/bank.npz.pca.pkl`)
needed at inference. Each checkpoint is reproducible from
`hcap_bo/slurm/apu_optuna.slurm` (seed 42).

## Fetching third-party checkpoints

The repository excludes ~1 GB of upstream model weights to keep the clone fast and to avoid re-hosting third-party artifacts. A single script pulls everything from canonical sources:

```bash
bash scripts/fetch_checkpoints.sh        # everything (~1.3 GB)
bash scripts/fetch_checkpoints.sh syn_score    # just the CGNF synthesizability ensemble
bash scripts/fetch_checkpoints.sh mattergen    # just MatterGen + DiffCSP + ALIGNN
bash scripts/fetch_checkpoints.sh esen         # just FairChem eSEN-30M-OAM (Cp oracle)
```

Sources:
- syn_score (100-bag CGNF ensemble, ~1.0 GB): `kaist-amsg/Synthesizability-stoi-CGNF/models/` (Jang et al., *Matter* 2024).
- MatterGen / DiffCSP / ALIGNN property predictors: `huggingface.co/jwchen25/MatInvent` (Chen et al. 2025, MIT-licensed MatInvent).
- eSEN-30M-OAM: `huggingface.co/facebook/OMAT24` (Meta FAIR).

## What is intentionally excluded

- Model checkpoints (`*.ckpt`, `*.pt`, `*.pth.tar`): use `scripts/fetch_checkpoints.sh`.
- `hcap_bo/results/`, `results_bm/`, `logs/`, `logs_bm/`: raw run outputs (already in `.gitignore`); rerun the SLURM scripts to regenerate.
- `hcap_bo/data/diffusion_reviews/`: 107 MB of literature PDFs; not code.

## Environments

Two conda environments:
- `matinvent` — the modified MatInvent pipeline (see `matinvent/env.yml`).
- `matinvent-hcap-fairchem` — the FAIRChem-1.10 stack for the closed-loop heat-capacity oracle (see `matinvent/fairchem.env.yml`).

## Note on hardcoded paths

The closed-loop plot scripts (`figures_src/closed_loop_*.py`, `oracle_savings.py`, `regen_bogen.py`, `data_loaders.py`) and a handful of `hcap_bo/` analysis/utility scripts (`hcap_bo/src/{featurizer,surrogate}.py`, `hcap_bo/analysis/top_structures/*.py`, `hcap_bo/analysis/v4_metrics_plots/plot_metrics.py`, `hcap_bo/docs/build_{deck,excalidraw}.py`, `hcap_bo/scripts/download_diffusion_reviews.py`) carry absolute paths from the development machine (`/Volumes/SSD1_SMAAA/matinvent-bo/...` and `/Volumes/SSD1_SMAAA/matinvent-hcap-bo/...`). Before running, point them at this repo by editing the `ROOT`/`REPO`/`OUT`/`SRC` constants at the top of each file, or sed-replace:

```bash
# from this repo's root
find . -name "*.py" -exec sed -i.bak \
  -e 's|/Volumes/SSD1_SMAAA/matinvent-bo/FME_paper_refresh_v1|figures_src|g' \
  -e 's|/Volumes/SSD1_SMAAA/matinvent-hcap-bo|hcap_bo|g' \
  -e 's|/Volumes/SSD1_SMAAA/matinvent-bo|.|g' {} +
```

The core MatInvent pipeline (`matinvent/main.py`, `matinvent/pipeline/`, `matinvent/rewards/`) takes all paths via Hydra configs and does not need editing.

## Citing

If you use this code, please cite the paper (BibTeX in `FoundationalEmbeddings_2026/refs.bib`).
