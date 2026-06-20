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
Everything needed to regenerate the figures is **in-tree** (figure-driving CSVs under `0*/data/`, prepared datasets `0*/data/*.pkl`, rendered references under `0*/figures/rendered/`). No external downloads required for figures.
```bash
bash reproduce.sh          # regenerates figures for all three subsystems from in-tree data
# or per subsystem:
bash 01_static_benchmark/run.sh
```
See **REPRODUCE.md** for the Figure/Table → command map.

> **Note:** the vendored figure generators under `0*/figures/generators/` were written against the original working tree and may carry absolute data paths; point them at the in-tree `0*/data/` directories (each generator's data-path constant near the top) if a path error appears. The figure-driving CSVs they need are all present in-tree.

## External (full from-scratch rerun only)
Raw 79 GB phonon deposit (Zenodo `20196565`) and model checkpoints (HuggingFace `SheikhAhnaf/...`) are **not** needed to reproduce figures — see `docs/EXTERNAL.md` / `docs/fetch_external.sh`.

## License
MIT — see `LICENSE`.
