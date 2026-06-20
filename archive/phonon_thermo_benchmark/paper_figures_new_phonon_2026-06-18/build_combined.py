"""Port the ACTUAL staged combined learning-curve scripts and run them for one arm.

Produces the S10-style aggregated learning curves (R2 + Spearman vs n_train, ORB, mean+/-std
across properties). Non-destructive: staged originals copied, edits limited to (1) source path
(point at this arm's n<N>/data instead of the original analysis dir), (2) the n_train list /
x-ticks for this arm, (3) plot_per_property's stale property list -> the 4 new targets.

Usage:  python build_combined.py <arm_dir> <comma-separated n list>
"""
import re
import subprocess
import sys
from pathlib import Path

BASE = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
NEW = BASE / "paper_figures_new_phonon_2026-06-18"
STAGED = BASE / "analysis_dfpt" / "figure_scripts_staged" / "combined"
NEW_PROPS = "'Cv_300K', 'S_300K', 'F_300K', 'max_phonon_freq'"

arm = sys.argv[1]
n_list = [int(x) for x in sys.argv[2].split(",")]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gp", "mtgp_2", "dgp"]
models_literal = "[" + ", ".join(f"'{m}'" for m in models) + "]"
dest = NEW / arm / "combined" / "scripts"
dest.mkdir(parents=True, exist_ok=True)
(NEW / arm / "combined" / "data").mkdir(exist_ok=True)
(NEW / arm / "combined" / "figures").mkdir(exist_ok=True)


def adapt(name: str, txt: str) -> str:
    # point the source path at THIS arm's per-n dirs (drop the original analysis-dir segment)
    txt = txt.replace("'..', '..', 'analysis_v3_phonon_dielectric_mp', f'n{n_train}'",
                      "'..', '..', f'n{n_train}'")
    # per-arm n grid (covers the for-loop in prepare and xticks/summary loop in plot_aggregated)
    txt = txt.replace("for n in [100, 250, 500]", f"for n in {n_list}")
    txt = txt.replace("[100, 250, 500]", str(n_list))
    # plot_per_property carries a stale elastic property list; give it the new targets
    txt = re.sub(r"'eps_electronic',\s*'eps_total',\s*'last phdos peak'", NEW_PROPS, txt)
    txt = re.sub(r"'K_Voigt'.*?'poisson_ratio'", NEW_PROPS, txt, flags=re.DOTALL)
    # surrogate list -> this arm's models (drops the phantom empty-DGP line/legend entry for Arm B)
    txt = txt.replace("['gp', 'mtgp_2', 'dgp']", models_literal)
    return txt


for src in sorted(STAGED.glob("*.py")):
    (dest / src.name).write_text(adapt(src.name, src.read_text()))

print(f"=== {arm} combined: ported, n_list={n_list}")
for script in ["prepare_learning_curves.py", "plot_aggregated.py", "plot_per_property.py"]:
    r = subprocess.run([sys.executable, str(dest / script)], capture_output=True, text=True)
    ok = "OK" if r.returncode == 0 else "FAIL"
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    print(f"  [{ok}] {script}: {tail}")
    if r.returncode != 0:
        print("    STDERR:", r.stderr.strip().splitlines()[-3:])
figs = list((NEW / arm / "combined" / "figures").rglob("*.png"))
print(f"=== produced {len(figs)} learning-curve figures")
