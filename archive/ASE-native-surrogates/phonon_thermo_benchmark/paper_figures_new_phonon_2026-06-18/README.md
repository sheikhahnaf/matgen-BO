# New phonon-thermo benchmark figures (for paper-figure comparison)

Built 2026-06-18. Replicates the FME paper's PHONON benchmark figure set using the new
phonon + heat-capacity dataset, with the SAME figure types/layout/styling as the paper but
the new target properties: **Cv_300K, S_300K, F_300K, max_phonon_freq** (per atom, 300 K).

**Nothing outside this directory was modified.** Figures were produced by COPYING the actual
`ASE_regression_test/analysis_v3_phonon_dielectric_mp/` + `combined_phonon_dielectric_mp/`
scripts and adapting only: property list, n-train, output paths, and Arm B's 2-model list.
All benchmark COMPUTE ran on ACES H100 GPUs; figure rendering is local CPU.

## Two arms
- **arm_a_dfpt/** — DFPT (1,253 materials), surrogates GP/MTGP/DGP, 5 splits, n∈{100,250,500}.
- **arm_b_pheasy/** — Pheasy (11,818 materials), GP/MTGP only (no DGP), 10 splits,
  GP n∈{100,250,500,1000,2000}, MTGP n∈{100,250,500,1000}. (n=2000 is GP-only.)

## Per arm
- `n<N>/figures/` — bar_charts/, heatmaps/, pca_sensitivity/, radar_charts/, property_difficulty/
  (each in R²/RMSE/Spearman + per-property variants). Maps to paper Figs 12c, S6, S7, S8, 14c.
- `combined/figures/aggregated/` — learning curves (R²/RMSE/Spearman vs n_train, ORB). Paper S10.
- `parity_orb_gp/` — ORB+GP holdout parity (Cv/S/F/max_phonon_freq), metrics-inset style. Paper S12c.

## contact_sheets/  ← START HERE for review
Side-by-side `paper | new Arm A | new Arm B` for each type: `compare_Fig12c_bar_R2.png`,
`compare_Fig14c_property_difficulty.png`, `compare_S6_heatmap_R2.png`,
`compare_S7_pca_sensitivity_R2.png`, `compare_S8_radar_orb_R2.png`,
`compare_S10_learning_curve_R2.png`, `compare_S12c_parity.png`.

## Key findings (see chat analysis for detail)
- ORB is the best featurizer (replicates paper, both arms). GP is a fine default; DGP gives no
  clear edge and collapses at PCA=50. Learning curves saturate by n=100 (high data efficiency).
- New targets predict well (R² ~0.7–0.98); heat capacity Cv ~0.83–0.97. **The higher R² vs the
  paper's phonon panel is target-driven** (paper's set mixed 1 phonon peak with 2 dielectric
  constants that R² can't capture, averaging to ~0.27), **not a method improvement** — frame
  accordingly. Swapping into the paper requires rewriting the phonon results text (Phase 2).
