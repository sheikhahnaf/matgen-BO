# three_way_comparison

DFT vs. eSEN-oracle vs. GP-surrogate comparison of bulk modulus K₀ for the closed-loop
discovered structures. Sibling of `dft_validation/` (which owns the DFT campaign + `K0.json`s);
this folder owns everything three-way: code, results, figures, and the paper section.

## Layout
```
drivers/
  three_way_causal.py   # MAIN: causal/per-cycle GP replay (train on RL_step<s, predict step s).
                        #   GPU. Reads memory (matinvent-hcap-bo), structures+K0 (dft_validation);
                        #   writes results/three_way_causal.csv + results/gp_causal_allpreds.csv.
  rank_analysis.py      # per-step GP RANKING analysis + figures. Pure pandas, runs anywhere.
  three_way.slurm       # FASTER GPU submit (any GPU, WebProxy, env-configured paths).
  three_way_gp.py       # env-configurable variant (single-fit; superseded by causal).
  legacy/
    three_way_gp_insample.py   # ARCHIVED flawed version (trained on memory containing the targets).
results/
  three_way_causal.csv        # 13 validated structures: DFT vs eSEN vs GP(causal) K0.
  gp_causal_allpreds.csv      # 3,802 step-ahead causal predictions (run, RL_step, n_train, gp, eSEN).
  rank_by_ntrain.csv, rank_by_step.csv, rank_within_step.csv
  three_way_property.png      # value scatter GP(causal) vs eSEN, colored by n_train.
  three_way_ranking.png       # ranking learning curve + per-step spread.
paper/
  three_way_comparison.md     # draft paper section (numbers + figures + caveats).
```

## CANONICAL = the FAITHFUL seed-inclusive replay (`*_seeded`)
`three_way_causal_seeded.py` is the correct driver. The earlier ORB/csv driver
(`three_way_causal.py`, `rank_analysis.py`, slurms) and its outputs are kept for the record but
were **seed-omitted/flawed** — see `results/legacy/*_NOSEED_flawed.*` and `paper/legacy/`. The flaw:
it trained per-run on generated structures only (no ~500 warm-start seed) and included BASE runs
that never used a GP, making the GP look cold-started. The faithful version trains on seed +
accumulated using the real per-run LTM parquets (`$HCAP/data/{bm,cp}/`), ACC runs only.

## Key findings — FAITHFUL, FULL SEED (see paper/three_way_comparison.md)
- eSEN oracle is a faithful DFT proxy: ρ≈0.885, MAPE 2.3%.
- GP(causal) vs eSEN: **BM ρ=0.944, RMSE 19.5 GPa (n=1209); Cp ρ=0.868, RMSE 0.112 J/g/K (n=596)**.
- Warm-started: ranking is ~flat-high vs loop-accumulated data (BM 0.88→0.96; Cp 0.79→0.89) — the
  seed pool, not per-cycle accumulation, makes the surrogate rank well.
- GP modestly underpredicts the extreme winners (BM bias −42.5 GPa, ρ=0.71, σ≈13–32) — not a collapse.
- 200-seed subsample reproduces the picture (BM ρ=0.90); flawed no-seed version said ρ=0.68, bias
  −161, "rising 0.23→0.97" — artifacts (archived `results/legacy/`, `_seed200` + `_NOSEED_flawed`).

## Reproduce (faithful, full seed)
```bash
sbatch drivers/three_way_seeded.slurm        # FASTER, full seed, all runs parallel (~1 h)
python drivers/rank_analysis_seeded.py bm    # figures + ranking (repeat with cp)
# (local quick subsample: N_SEED=200 + xargs -P8 over drivers/three_way_causal_seeded.py --one)
```
Why seed-inclusive + ACC-only: the real GP was warm-started on a seed pool and only ACC runs used a
GP at all. See `../dft_validation/insights.md` §12.
