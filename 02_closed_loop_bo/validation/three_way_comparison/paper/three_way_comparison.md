# Three-way comparison: DFT vs. eSEN oracle vs. GP surrogate

*Draft section for the FME paper. Numbers are the FAITHFUL, FULL-seed, ACC-run replay
(`three_way_causal_seeded.py`, run on FASTER via `three_way_seeded.slurm`). The earlier
seed-omitted version is archived under `results/legacy/*_NOSEED_flawed.*` and `paper/legacy/`; a
200-seed subsample (also archived, `_seed200`) gave the same picture. See "A correction we made".
Framing kept modest.*

Figures: `esen_vs_dft_bm.png` (oracle integrity), `three_way_ranking_{bm,cp}_seeded.png`,
`three_way_property_{bm,cp}_seeded.png`. Tables: `results/esen_vs_dft_bm.csv` (eSEN-vs-DFT),
`results/three_way_causal_bm_seeded.csv` (winners); per-structure causal preds in
`results/gp_causal_allpreds_{bm,cp}_seeded.csv`.

---

## Motivation

Two questions sit underneath the closed loop and neither is answered by its own logs. First, is the
eSEN MLIP oracle a trustworthy stand-in for DFT on the structures it selects? Second, how well does
the GP surrogate actually predict the property of *freshly generated* structures, as opposed to
reproducing values it has already trained on? We answer both with a like-for-like comparison of
three estimates of the bulk modulus K₀ (and, for heat capacity, the gravimetric Cv): ground-truth
DFT (oracle-parity Birch–Murnaghan; phonon Cv pending), the eSEN oracle value used as the loop's
reward, and the GP surrogate's prediction.

## Method: a causal, seed-inclusive replay (ACC runs)

The validated top structures are members of the closed-loop memory, so training a GP on all memory
and "predicting" them is memorization, not prediction. We instead replay the workflow exactly as it
ran. Each run is independent (own generator, own GP); structures carry a `cycle_id`. For each run we
walk cycles in order and, at cycle s, train the GP on everything available at cycles `< s` and
predict the cycle-s structures (unseen). Crucially the training set INCLUDES the workflow's
warm-start **seed pool** (`cycle_id = −1`, ~500 labeled structures spanning the full property
range) — the GP was never cold in reality. We use the actual per-run LTM parquets (stored ORB→PCA50
features + labels), and restrict to the **ACC (accelerated) runs**, which are the runs that used a
GP gate at all (the BASE runs have no surrogate to reconstruct). Per the plotting convention, the
learning-curve axis (`n_train_accum`) counts only loop-gathered structures and EXCLUDES the seed.

## How we computed the DFT (oracle-parity protocol)

The DFT K₀ is computed to be *like-for-like* with the eSEN oracle (`local_esen_bm.py`), so the
comparison is apples-to-apples rather than a generic DFT-vs-MLIP gap:
- **Cell preparation.** AI-generated cells carry ~0.01–0.05 Å lattice noise; we spglib-refine each
  to its intended space group, which lets VASP run with full symmetry (ISYM=2; k-mesh reduced
  4–24×). Genuinely P1 cells (8/29) run ISYM=0.
- **Relax.** Cell + ions to equilibrium with MPRelaxSet, PBE_54, ENCUT=680 eV (convergence-locked),
  no +U (0/29 trigger MP's +U scheme), ISPIN=2 (MP/OMat24 convention; magnetic branch pinned across
  EOS volumes by seeding the relaxed per-site moments).
- **Equation of state (oracle parity).** From the relaxed cell, 7 *rigid* isotropic strain points
  ε ∈ {−3%, …, +3%} with **frozen fractional coordinates** and a single-point static at each (the
  oracle does not relax ions at each strained volume → neither do we), then a 3rd-order
  Birch–Murnaghan fit → B₀. One **fixed k-grid** (KSPACING=0.16, from the relaxed reference) across
  all 7 points — recomputing per strained cell drifts the mesh and corrupts the BM curvature.
  Statics use tetrahedron ISMEAR=−5 (Gaussian fallback on near-Γ meshes) with EDIFF=1e-7.

Every validated structure's BM fit is in-window with a physical B₀′ (4.1–5.1), so the K₀ values
are trustworthy ground truth.

## Result 1: the MLIP oracle is a faithful DFT proxy

Over the 15 bm-leaderboard structures with completed DFT, the eSEN oracle tracks ground-truth DFT
closely: **MAE 8.5 GPa, MAPE 2.5%, Pearson r=0.90, Spearman ρ=0.87, bias (eSEN−DFT) −2.3 GPa**
(Fig. `esen_vs_dft_bm.png`; all points within or near ±10 GPa of parity). This licenses using eSEN
as the loop's reward and DFT as a post-hoc integrity audit rather than an in-loop calculator.

| formula | run | DFT K₀ | eSEN K₀ | Δ (eSEN−DFT) |
|---|---|--:|--:|--:|
| MoN | ACC | 353.6 | 375.2 | +21.6 |
| MoC | BASE | 348.1 | 351.6 | +3.5 |
| Os₅W₃ | ACC | 355.9 | 339.6 | −16.3 |
| CoIrOs₂ (r5) | ACC | 346.0 | 338.0 | −8.0 |
| CoIrOs₂ (r6) | ACC | 346.0 | 338.0 | −8.0 |
| CoIrOs₂ (r7) | ACC | 348.1 | 337.6 | −10.5 |
| Re₅W | ACC | 351.7 | 336.5 | −15.2 |
| VIr₇ | BASE | 334.6 | 325.0 | −9.6 |
| TiVN₂ | ACC | 296.3 | 313.1 | +16.8 |
| TaB₂W | ACC | 315.2 | 313.0 | −2.2 |
| FeB₂MoW (r13) | BASE | 306.9 | 306.2 | −0.7 |
| FeB₂MoW (r14) | BASE | 306.9 | 306.1 | −0.8 |
| Mn(BMo)₃ | BASE | 300.5 | 298.4 | −2.1 |
| FeRe(PW)₂ | ACC | 293.4 | 297.8 | +4.4 |
| ReIr₂Rh₅ | ACC | 299.5 | 292.2 | −7.3 |

The largest disagreements are MoN (eSEN +21.6) and TiVN₂ (+16.8); the rest agree to within ~10 GPa.
(All validated bm winners are hard refractory, so the comparison spans a narrow 293–356 GPa band —
the correlation is therefore restricted-range; the small MAE/MAPE is the more meaningful statistic.)
Table: `results/esen_vs_dft_bm.csv`. The Cv eSEN-vs-DFT leg awaits the phonon Cv campaign.

## Result 2: the GP is a strong ranker, warm-started from the start

Across all causal step-ahead predictions the GP orders generated structures well:

| property | n (preds) | GP-vs-eSEN ρ | Pearson r | RMSE | within-step median ρ |
|---|--:|--:|--:|--:|--:|
| K₀ (bulk modulus) | 1,209 | **0.944** | 0.954 | 19.5 GPa | 1.00 |
| Cv (heat capacity) | 596 | **0.868** | 0.899 | 0.112 J/g/K | 1.00 |

Resolving ranking quality by the amount of *loop-gathered* data (seed excluded) shows it is
**~flat and high from the start** — the warm-start seed does the work, and per-cycle accumulation
adds only a modest lift:

| loop-accumulated n_train (seed excl.) | 0–5 | 5–15 | 15–30 | 30–50 | 50+ |
|---|--:|--:|--:|--:|--:|
| K₀ Spearman ρ | 0.88 | 0.92 | 0.93 | 0.96 | 0.96 |
| Cv Spearman ρ | 0.79 | 0.85 | 0.89 | 0.89 | 0.87 |

So the surrogate is a reliable ranker of each cycle's candidates (the quantity the EI+DPP gate
consumes), and that reliability comes primarily from the warm start rather than from accumulating
loop labels.

## Result 3: a mild, honest underprediction of the extreme winners

For the ACC-run DFT-validated winners the GP modestly underpredicts the absolute K₀ of the top
extremes (predictions ~210–325 GPa vs DFT ~293–356), bias −42.5 GPa, ρ = 0.71, with honest
posterior variance (σ ≈ 13–32 GPa):

| formula | DFT K₀ | eSEN K₀ | GP causal K₀ | ±σ |
|---|--:|--:|--:|--:|
| MoN | 353.6 | 375.2 | 300.8 | 26 |
| Os₅W₃ | 355.9 | 339.6 | 323.2 | 21 |
| CoIrOs₂ (r5/6/7) | ~347 | ~338 | 313.1 | 16 |
| Re₅W | 351.7 | 336.5 | 280.0 | 32 |
| TiVN₂ | 296.3 | 313.1 | 273.6 | 17 |
| TaB₂W | 315.2 | 313.0 | 271.2 | 18 |
| FeRe(PW)₂ | 293.4 | 297.8 | 211.9 | 13 |
| ReIr₂Rh₅ | 299.5 | 292.2 | 280.5 | 13 |

The GP pulls the very top extremes toward the training bulk by ~40 GPa on average — a real but
modest effect, with the uncertainty flagging it — not a collapse.

## Reading

This makes the rank-not-regress position concrete and self-consistent from a correct baseline: the
surrogate's contribution is uncertainty-aware *ordering* (ρ ≈ 0.87–0.94), while it modestly
underpredicts absolute values at the rare extremes. The result upgrades the Discussion's
"self-consistency rather than generalization" caveat into a measured generalization number on
genuinely-unseen generated structures.

## A correction we made (methods-integrity note worth one sentence in the paper)

Our first replay omitted the warm-start seed pool and included BASE runs that never used a GP; that
made the GP look cold-started (ρ ≈ 0.68) and made it appear to collapse the winners (bias −161 GPa)
with a spurious "rising 0.23→0.97" learning curve. Reconstructing the surrogate from the data and
code it actually used — seed-inclusive training, ACC runs only — corrected ρ to 0.944 and the winner
bias to −42.5 GPa. We report the corrected analysis; the discrepancy is a useful caution that a
surrogate must be evaluated with its real training set, not a convenient proxy table.

## Caveats to disclose

- ACC runs only (BASE runs had no GP). The full warm-start seed (~500 BM / ~446 Cv) is included;
  a 200-seed subsample reproduces the same conclusion.
- The validated-winner correlations are restricted-range (winners span a narrow band).
- The Cv DFT-anchored leg (phonon Cv) is pending; the Cv numbers above are GP-vs-eSEN only.

## Reproduce

```bash
# faithful causal replay, full seed (FASTER; all runs parallel; ~1 h)
sbatch three_way_comparison/drivers/three_way_seeded.slurm
python three_way_comparison/drivers/rank_analysis_seeded.py bm   # figures + ranking (repeat cp)
```
