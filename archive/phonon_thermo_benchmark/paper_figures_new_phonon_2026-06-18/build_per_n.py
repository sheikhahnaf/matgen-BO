"""Port the ACTUAL staged ASE per-n figure scripts to the new phonon-thermo targets and run
them for one (arm, n), strictly inside this new directory.

Non-destructive: the staged originals (analysis_dfpt/figure_scripts_staged/per_n/, verbatim
copies of analysis_v3_phonon_dielectric_mp/n100/scripts/) are COPIED, never edited. The only
edits applied to the copies are: (1) property list -> the 4 new targets, (2) the hardcoded
n_train literal (n100 variant) -> the requested n. Everything writes under
<arm>/n<N>/{scripts,data,figures}; <arm>/aggregated_results.csv is the input.

Usage:  python build_per_n.py <arm_dir> <n>
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
NEW = BASE / "paper_figures_new_phonon_2026-06-18"
STAGED = BASE / "analysis_dfpt" / "figure_scripts_staged" / "per_n"
NEW_PROPS = "'Cv_300K', 'S_300K', 'F_300K', 'max_phonon_freq'"

# run order mirrors the original run_all.py (prepare_data first; the rest each read filtered_n{N}.csv)
RUN_ORDER = [
    "prepare_data.py",
    "bar_charts_averaged.py",
    "bar_charts_per_property.py",
    "heatmaps_averaged.py",
    "property_difficulty.py",
    "pca_sensitivity.py",
    "radar_charts.py",
]


def adapt(txt: str, n: int, models: list) -> str:
    # 1) property list (single-line literal in every staged script) -> new 4 targets
    txt = re.sub(r"'eps_electronic',\s*'eps_total',\s*'last phdos peak'", NEW_PROPS, txt)
    # 2) n-train literal (staged scripts are the n100 variant). Replacing "_n100" also fixes
    #    "filtered_n100" (it ends in _n100). Order the broad one last.
    txt = txt.replace("n_train'] == 100", f"n_train'] == {n}")
    txt = txt.replace("n=100", f"n={n}")
    txt = txt.replace("_n100", f"_n{n}")
    # 3) surrogate list -> this arm's models (Arm B pheasy has no DGP). Drives the model
    #    iteration / heatmap column order / bar groups. The colors_models & model_labels
    #    dicts keep their extra DGP key harmlessly. Also fix the hardcoded 3-panel grid in
    #    property_difficulty so a 2-model arm doesn't render an empty 3rd panel.
    models_literal = "[" + ", ".join(f"'{m}'" for m in models) + "]"
    txt = txt.replace("['gp', 'mtgp_2', 'dgp']", models_literal)
    if len(models) != 3:
        txt = txt.replace("plt.subplots(1, 3, figsize=(22, 10))",
                          f"plt.subplots(1, {len(models)}, figsize=({7 * len(models) + 1}, 10))")
    # 4) other hardcoded 3-model assumptions: heatmap x-tick labels, and the radar best-PCA
    #    text table (header + per-row formatting assume exactly GP/MTGP/DGP). Make them follow
    #    the model count so a 2-model arm renders cleanly (no-op when models == the 3 defaults).
    label_map = {"gp": "GP", "mtgp_2": "MTGP", "dgp": "DGP"}
    labels_literal = "[" + ", ".join(repr(label_map[m]) for m in models) + "]"
    txt = txt.replace("['GP', 'MTGP', 'DGP']", labels_literal)
    txt = txt.replace(
        'header = f"{\'Property\':25s} {\'GP\':15s} {\'MTGP\':15s} {\'DGP\':15s}"',
        'header = f"{\'Property\':25s} " + " ".join(f"{lbl:15s}" for lbl in ' + labels_literal + ")",
    )
    txt = txt.replace(
        'pca_text_lines.append(f"{row[0]:25s} {row[1]:15s} {row[2]:15s} {row[3]:15s}")',
        'pca_text_lines.append(f"{row[0]:25s} " + " ".join(f"{c:15s}" for c in row[1:]))',
    )
    return txt


def main():
    arm_dir = NEW / sys.argv[1]
    n = int(sys.argv[2])
    models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gp", "mtgp_2", "dgp"]
    dest = arm_dir / f"n{n}" / "scripts"
    dest.mkdir(parents=True, exist_ok=True)
    (arm_dir / f"n{n}" / "data").mkdir(exist_ok=True)
    (arm_dir / f"n{n}" / "figures").mkdir(exist_ok=True)

    # copy + adapt every staged .py (run only RUN_ORDER, but copy all so helpers resolve)
    for src in sorted(STAGED.glob("*.py")):
        out = dest / src.name
        out.write_text(adapt(src.read_text(), n, models))

    # property_difficulty draws one panel per model and indexes a 1-D axes array; that breaks
    # for a single-model arm (e.g. pheasy n=2000 has GP only, since MTGP's grid stops at 1000),
    # so skip it there. The single-model n=2000 point is still represented in the learning curve.
    run_order = [s for s in RUN_ORDER if not (len(models) == 1 and s == "property_difficulty.py")]

    print(f"=== {sys.argv[1]} n={n}: ported {len(list(STAGED.glob('*.py')))} scripts -> {dest}")
    for script in run_order:
        sp = dest / script
        r = subprocess.run([sys.executable, str(sp)], capture_output=True, text=True)
        ok = "OK" if r.returncode == 0 else "FAIL"
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"  [{ok}] {script}: {tail}")
        if r.returncode != 0:
            print("    STDERR:", r.stderr.strip().splitlines()[-3:] if r.stderr.strip() else "")
    figs = list((arm_dir / f"n{n}" / "figures").rglob("*.png"))
    print(f"=== produced {len(figs)} figures under n{n}/figures/")


if __name__ == "__main__":
    main()
