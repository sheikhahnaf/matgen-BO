"""Build combined learning-curve tables for arm_b_pheasy_full WITH DGP.
Replicates build_combined.py's adapt logic (that script has no __main__ guard, so it
can't be safely imported). Writes ONLY into arm_b_pheasy_full/combined/ (new dir)."""
import re, subprocess, sys
from pathlib import Path
BASE = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
STAGED = BASE / "analysis_dfpt" / "figure_scripts_staged" / "combined"
NEW_PROPS = "'Cv_300K', 'S_300K', 'F_300K', 'max_phonon_freq'"
ARM = Path(__file__).resolve().parent.parent / "arm_b_pheasy_full"
n_list = [100, 250, 500, 1000, 2000]
models = ["gp", "mtgp_2", "dgp"]
models_literal = "[" + ", ".join(f"'{m}'" for m in models) + "]"
dest = ARM / "combined" / "scripts"; dest.mkdir(parents=True, exist_ok=True)
(ARM / "combined" / "data").mkdir(exist_ok=True); (ARM / "combined" / "figures").mkdir(exist_ok=True)

def adapt(txt):
    txt = txt.replace("'..', '..', 'analysis_v3_phonon_dielectric_mp', f'n{n_train}'",
                      "'..', '..', f'n{n_train}'")
    txt = txt.replace("for n in [100, 250, 500]", f"for n in {n_list}")
    txt = txt.replace("[100, 250, 500]", str(n_list))
    txt = re.sub(r"'eps_electronic',\s*'eps_total',\s*'last phdos peak'", NEW_PROPS, txt)
    txt = re.sub(r"'K_Voigt'.*?'poisson_ratio'", NEW_PROPS, txt, flags=re.DOTALL)
    txt = txt.replace("['gp', 'mtgp_2', 'dgp']", models_literal)
    return txt

for src in sorted(STAGED.glob("*.py")):
    (dest / src.name).write_text(adapt(src.read_text()))
for script in ["prepare_learning_curves.py", "plot_aggregated.py", "plot_per_property.py"]:
    r = subprocess.run([sys.executable, str(dest / script)], capture_output=True, text=True)
    print(f"  [{'OK' if r.returncode==0 else 'FAIL'}] {script}: {((r.stdout.strip().splitlines() or [''])[-1])[:70]}")
    if r.returncode:
        print("    ERR:", (r.stderr.strip().splitlines() or ['?'])[-3:])
