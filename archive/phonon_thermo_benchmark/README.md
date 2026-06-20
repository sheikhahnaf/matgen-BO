# phonon_thermo_benchmark

Per-atom phonon thermodynamic regression benchmark: predict heat capacity,
entropy, free energy and max phonon frequency (at 300 K) from crystal structure
descriptors, for materials computed with DFPT (Arm A) vs pheasy (Arm B) phonons
in the Materials Project.

## Data prep (live MP pull)

`src/prepare_phonon_data.py` builds the labeled dataset from the Materials
Project phonon endpoint.

```bash
export MP_API_KEY=...   # your Materials Project API key
python src/prepare_phonon_data.py --method dfpt --out data/phonon_dfpt.pkl [--limit N] [--temperature 300]
```

Output: a pickled `pandas.DataFrame` keyed by `material_id`, carrying the
pymatgen `Structure` plus the four per-atom targets (`Cv_300K`, `S_300K`,
`F_300K`, `max_phonon_freq`). Dynamically unstable materials (appreciable
imaginary-mode density) are dropped.

### Environment requirement

- The live data prep requires **Python >= 3.11** and a **recent `mp-api`**.
  The local `matinvent` env (Python 3.10) cannot even import `MPRester`
  (`emmet.core` needs `typing.NotRequired`, 3.11+). Run data prep in an ACES
  env, not matinvent.
- The script **prints the `mp_api` version** at the start of the live run so
  the log records exactly which API surface ACES resolved.
- **Method selection (dfpt vs pheasy) needs a recent `mp-api`.** The script is
  written to work on either API: it tries the method-aware kwarg/accessor and
  falls back to client-side filtering / the no-arg DOS accessor on older
  releases. But the old no-arg DOS accessor returns the **default (dfpt)**
  method only, so **pheasy (Arm B) requires a recent mp-api** — confirm the
  printed `mp_api` version before launching the pheasy run.

The pure row-builder (`build_row`) and its offline test
(`src/test_prepare_smoke.py`) need only pymatgen and run anywhere, including
matinvent (no network, no `mp_api`).

## Feature cache

Featurized descriptors are cached on disk under
`data/feat_cache/<descriptor>/`. Cache keys differ by descriptor family:

- **SOAP** is namespaced by a species signature and is width-guarded (the cache
  key encodes the element set / cutoff so different chemistries don't collide).
- **MACE / ORB / UMA** embeddings are keyed by **model NAME only**.

> **Warning:** because the learned-embedding caches key on model name alone, if
> you ever change a featurizer's **model** (e.g. a different MACE/ORB/UMA
> checkpoint) or change **SOAP hyperparameters** (`r_cut`, `n_max`, `l_max`),
> you **must delete `data/feat_cache`** first. Otherwise stale vectors from the
> previous configuration will be served silently with no error.

## Testing

```bash
eval "$(/Users/alvi/miniconda3/bin/conda shell.zsh hook)" && conda activate matinvent \
  && cd phonon_thermo_benchmark && python -m pytest src/ -v
```

## Deliberate divergences from `ASE_regression_test` (audit fixes)

`src/` started as a copy of `../ASE_regression_test/`. Beyond the data-prep
adaptations above (pickle loader, holdout cap, feature cache), two **intentional,
user-authorized** bug-fixes from an audit were applied to the copies. The copies
are therefore **expected** to differ from the originals on these points; do not
"revert to match the originals."

### PATCH-A — DGP-1: per-descriptor reduction search (`src/dgp_regression.py`)

The original `DESCRIPTOR_REDUCTIONS` dict assigned the **same** 7-value list
`[-45, -40, -35, -30, -20, -10, 0]` to every descriptor, contradicting both the
module docstring and the source notebook's per-descriptor `compare_models_with_split_*`
functions. The copy restores the **documented per-descriptor** search sets:

| descriptor | reduction search set |
|------------|----------------------|
| `soap`     | `[-45]` |
| `mace`     | `[-45, -40, -35, -30, -20]` |
| `orb`      | `[-20, -10, 0]` |
| `uma`      | `[-20, -10, 0]` (eSEN in the notebook) |

Nothing else in the DGP model/training logic changed.

### PATCH-B — DATA-1: exclude invalid-feature holdout rows from HOLDOUT metrics

When a structure cannot be **genuinely featurized** — SOAP encounters an element
absent from the fitted training-split species basis, or MACE/ORB/UMA featurization
fails and falls back to a zero vector — the resulting zero/constant feature row, if
it lands in the HOLDOUT with a valid target, was still being scored in holdout
metrics, **pessimizing** them.

The fix:

- `common.py`: each featurizer now records a per-row validity mask
  (`last_valid_mask_`, `np.ndarray[bool]`, length = number of input structures,
  `True` = genuinely featurized) for its most recent featurize call, aligned through
  the same per-row scatter the features use.
- `gp_regression.py`, `mtgp_regression.py`, `dgp_regression.py`: at HOLDOUT
  evaluation, the existing target-not-NaN holdout mask is intersected with the
  featurizer's `last_valid_mask_` (for DGP this is done by NaN-ing invalid rows'
  targets before building the long-format holdout pairs), and an excluded-row count
  is printed, e.g.
  `[holdout][soap] excluded N/<total> rows (invalid/unknown-species features)`.

**Training is unchanged.** SOAP fits species on the full train set, so train rows
never hit the unknown-species fallback by construction; internal-test and the metric
formulas are untouched. On data with no unknown species (e.g. the synthetic smoke
set) the excluded count is `0` and behavior is identical to the originals.
