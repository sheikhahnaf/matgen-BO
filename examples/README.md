# Examples

Two short scripts that show how the pieces of the paper are used. Both run on a
laptop CPU with the repository environment (`environment.yml`) and read data
that ships with the repository, so no GPU, cluster or download is needed.

| Script | What it shows | Time |
|---|---|---|
| `01_fit_surrogate.py` | Fit the default surrogate (ORB embedding, PCA, Gaussian process) to labeled structures and check accuracy, ranking and interval coverage on held-out structures | about a minute |
| `02_gate_selection.py` | Score one real batch of generated candidates with the gate's GP and apply the gate's selection rule, next to what the oracle later returned | a few seconds |

```bash
conda activate matinvent
python examples/01_fit_surrogate.py
python examples/02_gate_selection.py
```

## Using your own structures

`01_fit_surrogate.py` has two small functions to adapt.

- `embed(atoms_list, device)` turns a list of `ase.Atoms` into one 256-number
  ORB embedding per structure. The first call downloads the ORB weights.
- `load_data()` returns the embedding matrix and the target values. Replace its
  body with your own structures and property values.

The other featurizers of the benchmark (MACE, UMA/eSEN, SOAP) and the other
surrogates (multi-task GP, deep GP) are in `01_static_benchmark/src/`, with one
command-line script per surrogate, for example:

```bash
python 01_static_benchmark/src/gp_regression.py --dataset elastic_tensor_2015 \
    --descriptor orb --pca-components 50 --n-train 500 --n-splits 5 --device cuda
```

## Where the full pipelines are

- Static benchmark: `01_static_benchmark/` (README there, and `run.sh`).
- Closed loop with the gate: `02_closed_loop_bo/src/` for the code,
  `02_closed_loop_bo/src/scripts/v4*/` for the job scripts of the paper.
- Runs and analyses added in the revision: `04_revision_round1/`.
