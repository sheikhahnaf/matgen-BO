"""Build per-n derived tables for arm_b_pheasy_full (WITH DGP), reusing the source
build_per_n.adapt()/RUN_ORDER. Writes ONLY into arm_b_pheasy_full/n<N>/ (new dir)."""
import sys, subprocess
from pathlib import Path
SRC = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
sys.path.insert(0, str(SRC / "paper_figures_new_phonon_2026-06-18"))
import build_per_n as B  # main() is __main__-guarded; importing only gives adapt/STAGED/RUN_ORDER
ARM = Path(__file__).resolve().parent.parent / "arm_b_pheasy_full"

def build(n, models):
    dest = ARM / f"n{n}" / "scripts"; dest.mkdir(parents=True, exist_ok=True)
    (ARM / f"n{n}" / "data").mkdir(exist_ok=True); (ARM / f"n{n}" / "figures").mkdir(exist_ok=True)
    for src in sorted(B.STAGED.glob("*.py")):
        (dest / src.name).write_text(B.adapt(src.read_text(), n, models))
    order = [s for s in B.RUN_ORDER if not (len(models) == 1 and s == "property_difficulty.py")]
    for script in order:
        r = subprocess.run([sys.executable, str(dest / script)], capture_output=True, text=True)
        tail = ((r.stdout.strip().splitlines() or [""])[-1])[:70]
        print(f"  [{'OK' if r.returncode==0 else 'FAIL'}] n{n}/{script}: {tail}")
        if r.returncode != 0:
            print("    ERR:", (r.stderr.strip().splitlines() or ['?'])[-2:])

for n, models in [(100,["gp","mtgp_2","dgp"]),(250,["gp","mtgp_2","dgp"]),
                  (500,["gp","mtgp_2","dgp"]),(1000,["gp","mtgp_2","dgp"]),(2000,["gp"])]:
    print(f"=== n{n} models={models} ===")
    build(n, models)
