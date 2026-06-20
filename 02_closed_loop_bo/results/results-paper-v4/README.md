# Curated per-run results for the paper's closed-loop figures

One directory per run, named `<target>_p3v4[_bm]_<backbone>_<policy>_seed<seed>_<jobid>`,
holding exactly the files the figure scripts read:

- `metrics.csv` — per-cycle run metrics
- `hparams.yaml` — run configuration
- `samples/long_term_memory.csv` — oracled candidates (reward, RL_step, composition, CIF)
- `rewards/heat_capacity/gp_routed_v4_log.csv` or `rewards/bulk_modulus/bm_gp_routed_v4_log.csv`
  — per-cycle GP diagnostics (ACC runs)

`results/` = heat-capacity runs (incl. the cap-4 ablation `hcap_mgabl_cap4_cp_*`,
with the two `_patched` re-runs of seeds that initially terminated on zero-reward
cycles); `results_bm/` = bulk-modulus runs (incl. `hcap_mgabl_cap4_bm_*` and
`hcap_mgabl_oracleall_bm_*`). Duplicate (backbone, policy, seed) directories from
re-submissions are resolved inside the figure scripts (keep latest job id).

Regenerate the closed-loop figures from a fresh clone:

    python figures_src/closed_loop_curves.py     # discovery curves + GP CV5 panels
    python figures_src/oracle_savings.py         # cumulative oracle calls
    python figures_src/mg_ablation_plots.py      # MatterGen budget ablation
    python figures_src/closed_loop_extras.py     # spacegroup / per-model / zoo extras
    python figures_src/synth_compare.py          # synthesizability comparison

Outputs land in `figures/` (override roots with MBO_REPO_ROOT / MBO_RESULTS_ROOT /
MBO_FIG_DIR). Fine-tuned diffusion checkpoints for all 59 runs are archived on
Hugging Face (SheikhAhnaf/apu-synthesizability-checkpoints, finetuned_diffusion/).
