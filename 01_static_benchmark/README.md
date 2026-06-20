# 01 — Static surrogate benchmark
GP/MTGP/DGP x {MACE,ORB,UMA,SOAP} on elastic, dielectric, phonon-thermo (DFPT n=1,253 + Pheasy n=11,818).

- `src/` regression + phonon data-prep code
- `data/` figure-driving CSVs + prepared pkls (dfpt/pheasy_phonon_thermo.pkl)
- `results/` per-run holdout summaries
- `figures/generators` + `figures/rendered`

Reproduce: see ../REPRODUCE.md. CPU-only; figures regenerate in minutes from in-tree CSVs.
