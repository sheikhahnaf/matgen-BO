# matinvent-hcap-bo

ORB-embedding Bayesian-optimization acceleration of MatInvent for the **heat-capacity** RL task.

This is a **dedicated, isolated workspace**. It does not modify or depend on the layout of either:
- `/Volumes/SSD1_SMAAA/matinvent-bo/matinvent/` (upstream MatInvent codebase)
- `/Volumes/SSD1_SMAAA/matinvent-bo/ASE_regression_test/` (upstream ORB+GP regression study)

Both upstream repos are referenced read-only during development. Any code reused from them is copied into `src/` here, never edited in place.

## Goal

Replace MatInvent's per-RL-step FairChem `eSEN-30M-OAM` heat-capacity oracle with an ORB-v3 + BoTorch-GP surrogate that calls the FairChem oracle only when the GP is uncertain, plus a pre-oracle BO-screening filter that ranks a larger generated batch and oracles only the top-k.

**Headline target:** ≥5× reduction in FairChem heat-capacity calls at matched-or-better SUN ratio and convergence to `Cp > 1.5 J/g/K`, vs. the upstream MatInvent baseline.

## Compute layout

| Resource | Location | Purpose |
|---|---|---|
| Local working tree | `/Volumes/SSD1_SMAAA/matinvent-hcap-bo/` | edit, design docs, notebooks |
| FASTER scratch | `/scratch/user/ahnafalvi/matinvent-hcap-bo/` | training, BO loop, results |
| FASTER conda env | `/scratch/user/ahnafalvi/envs/matinvent-hcap-bo/` | dedicated env (NOT shared) |
| HF / model cache | `/scratch/user/ahnafalvi/hf_cache/` (existing, read/write) | shared with other projects |

All RL / BO loops run on FASTER A100 nodes via SLURM. The local tree is for editing and result analysis only.

## Read-only references

- MatInvent paper: arXiv:2511.03112
- Foundation Model Embeddings paper: `/Volumes/SSD1_SMAAA/matinvent-bo/Foundation_Model_Embeddings_as_Plug-in_Featurizers_for_Probabilistic_Surrogates_in_Materials_Design/main.pdf`
- Dodds et al., Chem. Sci. 2024 (RL+AL surrogate-screened oracle): D3SC04653B
- ORB-v3: arXiv:2504.06231
- BoTorch acquisition docs: https://botorch.readthedocs.io/

## Status

See `docs/DESIGN.md` for the brainstormed plan and phase breakdown.
