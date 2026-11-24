# 04_revision_round1

Experiments and analyses added in the first revision of the paper. Nothing in
`01_static_benchmark/`, `02_closed_loop_bo/` or `03_synthesizability/` was
changed. Every new run reuses the published code through one extra setting, and
every new analysis reads the published run folders.

The state of the repository before this folder was added is tagged
`pre-revision-round1-20261124`.

## Layout

| Folder | What it holds |
|---|---|
| `closed_loop/` | Job script, submit script and two small modules for the new closed-loop arms |
| `calibration/` | Re-run of the static GP fits that keeps each holdout prediction with its standard deviation |
| `candidate_table/` | One table of every candidate that reached the property calculator in the published runs |
| `analysis/` | Scripts that turn the tables into the numbers and figures of the revision |
| `environments/` | `pip freeze` of the three Python environments used for these runs |
| `figures/` | Figures written by the analysis scripts |

## Closed-loop arms (`closed_loop/`)

All arms use `run_rev1_arm.slurm`. Each arm is one of the original command lines
(`02_closed_loop_bo/src/scripts/v4` and `v4_bm`) with a single change.

| Arm | Starts from | The one change |
|---|---|---|
| `cap4` | ungated run (BASE) | `sample_cfg.max_num=4`: at most the first four SUN survivors reach the oracle each cycle |
| `frozen_gate` | gated run (ACC) | `pipeline._target_=fme_rev.frozen_pipeline.MatInventFrozen`: the generator is never fine-tuned |
| `ei_only` | gated run (ACC) | `GP_DPP_LAMBDA=0`: the four picks are the top four by Expected Improvement |
| `div_only` | gated run (ACC) | `fme_rev.diversity_gate`: the Expected Improvement scores are replaced by zeros |

`frozen_pipeline.py` logs a SHA-256 fingerprint of the generator parameters
before the first cycle and after the last one and fails the job if they differ.

Seeds are 7, 17, 23, 99 and 113, with 20 cycles per run, as in the paper.

```bash
export REV_ROOT=/path/to/output/root   # runs/, ltm/ and logs/ are created here
export ACCOUNT=<slurm account>
bash closed_loop/submit_rev1.sh smoke      # short test jobs
bash closed_loop/submit_rev1.sh cap4       # CrystalFlow and ADiT, both targets
bash closed_loop/submit_rev1.sh frozen     # three backbones, both targets
bash closed_loop/submit_rev1.sh ei_only    # three backbones, both targets
bash closed_loop/submit_rev1.sh div_only   # MatterGen, both targets
bash closed_loop/status_rev1.sh            # queue and finished runs
```

The job script reads three locations that default to the cluster layout used
for the paper and can be overridden: `PROJ` (the contents of
`02_closed_loop_bo/src` with its `configs/` and `data/`), `PRISTINE` (MatInvent
with the two guards for cycles in which no oracle call returns a value) and
`ENV_PREFIX` (the Python environment, `environments/matinvent-hcap-bo.frozen.txt`).
Heat-capacity jobs request a 16 GB T4 GPU, because the out-of-memory failures of
the phonon oracle depend on GPU memory. Bulk-modulus jobs can run on any GPU
(`GRES_BM`).

## Static GP calibration (`calibration/`)

The published benchmark stored summary scores only. `gp_calibration_rerun.py`
calls the unmodified `gp_regression.py` with the per-property split loop
replaced by a copy that skips the parity plots and writes, for every holdout
sample, the true value, the posterior mean and the predictive standard
deviation with and without the fitted noise. `run_gp_calibration.slurm` runs
the full grid for one dataset (PCA 10/25/50, 100/250/500 training points, four
descriptors, five splits).

Check of the rerun against the published summary files (PCA 50, 500 training
points): SOAP reproduces the published holdout R2 exactly. For the three
foundation-model embeddings the values differ in the second or third decimal,
because the embeddings are recomputed on a GPU. The only larger differences are
on `poly_electronic`, where the published R2 is negative and varies by 0.3 to
1.1 between splits.

## Candidate table (`candidate_table/`)

`build_candidate_table.py` reads the raw run folders. For every cycle it takes
the candidates that were handed to the property calculator
(`samples/step_XXXX_eval.extxyz`), marks which of them came back with a value
(`samples/long_term_memory.csv`) and, for runs without the gate, assigns the
failure reason from the job's stderr log, where the oracle prints one line per
failed candidate in dispatch order. It also stores the mean-pooled ORB
embedding of every candidate and of the GP seed pools.

`out/published/` holds the tables for the runs of the paper.

| File | Content |
|---|---|
| `candidates_{cp,bm}.parquet` | one row per candidate: run, cycle, outcome, failure reason, reward, property value, formula, cell descriptors, structure |
| `orb_raw_{cp,bm}.npy` | 256-dimensional ORB embedding, row-aligned with the table |
| `seed_pool_{cp,bm}.parquet`, `orb_raw_seed_{cp,bm}.npy` | the GP seed pools |
| `run_summary_{cp,bm}.csv` | per-run counts and the cross-check between table and log |
| `gate_memory/` | the gate's own training set of every gated run |

In every ungated heat-capacity run the number of failures in the table equals
the number of failure lines in the log.

## Analyses (`analysis/`)

Run from this folder with the repository environment (`environment.yml`).

| Script | Question | Output |
|---|---|---|
| `analyze_oracle_failures.py` | How often does the heat-capacity oracle fail, why, and what does failure correlate with? | `out/cp_failure_*.csv`, `figures/fig_cp_oracle_failures.png` |
| `analyze_calibration.py` | Are the GP's predictive intervals calibrated, on the static benchmark and inside the loop? | `out/static_calibration_*.csv`, `out/loop_calibration.csv`, `figures/fig_gp_calibration.png` |
| `reconstruct_gate_picks.py` | Which candidates did the gate send to the oracle in each cycle of the published gated runs, and how was each pick made? | `out/gate_picks_*.csv` |
| `analyze_composition_baselines.py` | Does the ORB embedding add anything over composition-only models for the two loop targets? | `out/composition_baselines_*.csv` |

`rev_common.py` holds the shared pieces: data loading, the gate's GP and
selection rule, and the calibration metrics.
