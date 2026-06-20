# matgen-BO

Reproduction code, data, and figures for the paper's three subsystems:
probabilistic **surrogates on pretrained atomistic embeddings** (GP / MTGP / DGP × MACE/ORB/UMA/SOAP), a **closed-loop generative discovery loop** (MatterGen / CrystalFlow / ADiT, GP-routed Expected-Improvement + DPP gate), and an **ORB-PU synthesizability** classifier.

## Layout
```
01_static_benchmark/   surrogate benchmark — elastic, dielectric, phonon-thermodynamics (DFPT + Pheasy)
02_closed_loop_bo/     discovery loop + oracles (heat capacity, bulk modulus)
03_synthesizability/   ORB-PU classifier (checkpoints on HuggingFace — see docs/EXTERNAL.md)
shared/                common.py, plot_style.py, data_loaders.py
docs/                  EXTERNAL.md (Zenodo/HF), PROVENANCE.md, fetch_external.sh
archive/               verbatim copies of every source tree (completeness; not needed to reproduce)
REPRODUCE.md           which command regenerates each paper Figure/Table
```

## Install
```bash
conda env create -f environment.yml   # creates the `matinvent` env
conda activate matinvent
# (or: pip install -r requirements.txt)
```

## Reproduce
Everything needed to regenerate the figures is **in-tree**. Each generator resolves the
repo root from its own location and reads the complete verbatim data mirror under
`archive/<source-tree>/` (e.g. `archive/ASE-native-surrogates/analysis_v3/…`), writing
freshly rendered figures to `0*/figures/regenerated/`. The paper-reference figures under
`0*/figures/rendered/` are never overwritten. No external downloads required for figures.
```bash
conda activate matinvent
bash reproduce.sh          # regenerates figures for all three subsystems from in-tree data
# or per subsystem:
bash 01_static_benchmark/run.sh
```
See **REPRODUCE.md** for the Figure/Table → command map.

> **Self-contained:** the generators are path-portable — no machine-specific paths, no
> manual edits. They discover the repo root via `Path(__file__)` and read in-tree
> `archive/` data, so a fresh clone reproduces the figures as-is. `s13_s14_regen` also
> self-checks its reproduced R² values against the published numbers.

## External (full from-scratch rerun only)
Raw 79 GB phonon deposit (Zenodo `20196565`) and model checkpoints (HuggingFace `SheikhAhnaf/...`) are **not** needed to reproduce figures — see `docs/EXTERNAL.md` / `docs/fetch_external.sh`.

## License
MIT — see `LICENSE`.
