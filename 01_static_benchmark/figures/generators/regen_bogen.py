"""Regenerate BO+Generative-model 5-seed figures with plot_style applied.

Imports plot_metrics helpers from analysis/v4_metrics_plots and re-renders into
FME_paper_refresh_v1/figures/ with fig_bogen_* prefix at 300 DPI Helvetica.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-bo/FME_paper_refresh_v1")
SRC = Path("/Volumes/SSD1_SMAAA/matinvent-hcap-bo/analysis")
DST = ROOT / "figures"

sys.path.insert(0, str(ROOT / "figures_src"))
from plot_style import apply_style  # noqa: E402


def load_plot_metrics():
    spec = importlib.util.spec_from_file_location(
        "pm", str(SRC / "v4_metrics_plots/plot_metrics.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regen_metric_plots(pm):
    apply_style()
    plt.rcParams.update({"savefig.dpi": 300, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    for prop, label, col, minv, maxv, target_str, tag in [
        ("Cp", "Cp", "Cp", 0.25, 2.0, "→ 1.5 J/g/K", "cp"),
        ("BM", "K_VRH", "K_VRH", 20.0, 400.0, "max", "bm"),
    ]:
        jobs = pm.discover(prop)
        if jobs.empty:
            print(f"  no jobs for {prop}, skipping")
            continue
        scratch = DST / "_scratch_bogen"
        scratch.mkdir(exist_ok=True)
        # best_running
        pm.plot_best_running(jobs, label, col, minv, maxv, target_str, scratch)
        shutil.move(str(scratch / "best_running.png"),
                    str(DST / f"fig_bogen_best_running_{tag}.png"))
        # gp_fit_quality
        pm.plot_gp_fit(jobs, label, scratch)
        shutil.move(str(scratch / "gp_fit_quality.png"),
                    str(DST / f"fig_bogen_gp_fit_quality_{tag}.png"))
        # oracle_split
        pm.plot_oracle_split(jobs, label, scratch)
        shutil.move(str(scratch / "oracle_split.png"),
                    str(DST / f"fig_bogen_oracle_split_{tag}.png"))
        # frac_NUS via plot_metric_panel
        df_nus = pm.load_metrics(jobs, "frac_novel_unique_stable_structures")
        if not df_nus.empty:
            pm.plot_metric_panel(
                df_nus,
                title=f"Stable+Unique+Novel fraction per cycle — {label}",
                ylabel="frac NUS",
                fname=scratch / "frac_nus.png",
            )
            shutil.move(str(scratch / "frac_nus.png"),
                        str(DST / f"fig_bogen_frac_nus_{tag}.png"))
        scratch.rmdir() if not any(scratch.iterdir()) else None


def copy_static(name: str, dst_name: str) -> bool:
    src = SRC / name
    if not src.exists():
        print(f"  MISSING: {src}")
        return False
    shutil.copy(str(src), str(DST / dst_name))
    return True


def main():
    pm = load_plot_metrics()
    regen_metric_plots(pm)
    # Top-structures (already paper-quality, copy as-is)
    copy_static("top_structures/zoo/zoo_cp.png", "fig_bogen_structure_zoo_cp.png")
    copy_static("top_structures/zoo/zoo_bm.png", "fig_bogen_structure_zoo_bm.png")
    copy_static("top_structures/figures/seed_overlap_cp.png",
                "fig_bogen_seed_overlap_cp.png")
    copy_static("top_structures/figures/seed_overlap_bm.png",
                "fig_bogen_seed_overlap_bm.png")
    copy_static("top_structures/figures/composition_top_cp.png",
                "fig_bogen_top_formula_leaderboard_cp.png")
    copy_static("top_structures/figures/composition_top_bm.png",
                "fig_bogen_top_formula_leaderboard_bm.png")
    print("\nDone. New fig_bogen_*.png written to:", DST)


if __name__ == "__main__":
    main()
