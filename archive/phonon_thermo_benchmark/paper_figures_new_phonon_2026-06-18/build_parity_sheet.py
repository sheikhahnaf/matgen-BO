"""Parity contact sheet: paper's best-property parity vs the new arms' best-property parity.

Paper S12c parity is `last phdos peak` (ORB+GP). The new arms' best-predicted property is
S_300K (ORB+GP), so we show that as the analogue. Read-only on the paper figure; writes only
into <new_dir>/contact_sheets/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from pathlib import Path

PAPER = Path("/Users/alvi/fme_paper_work/FoundationalEmbeddings_2026/figures/fig_parity_phonon_phdos_peak_orb_gp.png")
NEW = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark/paper_figures_new_phonon_2026-06-18")
OUT = NEW / "contact_sheets"

panels = [
    ("PAPER S12c: last phdos peak (ORB+GP)", PAPER),
    ("NEW Arm A - DFPT: S_300K (ORB+GP)", NEW / "arm_a_dfpt/parity_orb_gp/parity_S_300K_holdout_split1.png"),
    ("NEW Arm B - pheasy: S_300K (ORB+GP)", NEW / "arm_b_pheasy/parity_orb_gp/parity_S_300K_holdout_split1.png"),
]
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
for ax, (title, p) in zip(axes, panels):
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    if p.exists():
        ax.imshow(mpimg.imread(p))
    else:
        ax.text(0.5, 0.5, "(file not found)", ha="center", va="center")
fig.suptitle("S12c_parity  -  best-predicted property, new phonon-thermo vs paper",
             fontsize=15, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = OUT / "compare_S12c_parity.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close()
print("wrote", out)
