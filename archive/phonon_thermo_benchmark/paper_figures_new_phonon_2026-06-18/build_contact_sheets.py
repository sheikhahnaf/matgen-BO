"""Side-by-side contact sheets: paper phonon figure | NEW Arm A (DFPT) | NEW Arm B (pheasy).

Read-only on the paper figures (rasterizes the PDFs via PyMuPDF); writes ONLY into
<new_dir>/contact_sheets/. Nothing in the paper tree is modified.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import fitz
from pathlib import Path

PAPER = Path("/Users/alvi/fme_paper_work/FoundationalEmbeddings_2026/figures")
NEW = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark/paper_figures_new_phonon_2026-06-18")
OUT = NEW / "contact_sheets"
OUT.mkdir(exist_ok=True)


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    if p.suffix.lower() == ".pdf":
        page = fitz.open(p)[0]
        pix = page.get_pixmap(dpi=150)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return mpimg.imread(p)


# (sheet name, paper file, new-figure path relative to each arm dir)
SHEETS = [
    ("Fig12c_bar_R2", "fig2_bar_phonon_R2_grouped.png",
     "n500/figures/bar_charts/averaged_R2_n500.png"),
    ("Fig14c_property_difficulty", "fig_difficulty_phonon_n500.pdf",
     "n500/figures/property_difficulty/difficulty_matrix_per_surrogate_n500.png"),
    ("S6_heatmap_R2", "fig_heatmap_phonon_R2_n500.pdf",
     "n500/figures/heatmaps/averaged_R2_n500.png"),
    ("S7_pca_sensitivity_R2", "fig_pca_phonon_R2_n500.pdf",
     "n500/figures/pca_sensitivity/averaged_R2_n500.png"),
    ("S8_radar_orb_R2", "fig_radar_phonon_orb_R2_n500.pdf",
     "n500/figures/radar_charts/orb_R2_n500.png"),
    ("S10_learning_curve_R2", "fig_combined_phonon_R2_learning_curve.png",
     "combined/figures/aggregated/averaged_R2_learning_curve.png"),
]

made = []
for name, paper_f, rel in SHEETS:
    panels = [
        ("PAPER (current: phdos-peak + 2 dielectric)", load(PAPER / paper_f)),
        ("NEW Arm A - DFPT 1.25k (GP/MTGP/DGP)", load(NEW / "arm_a_dfpt" / rel)),
        ("NEW Arm B - pheasy 11.8k (GP/MTGP)", load(NEW / "arm_b_pheasy" / rel)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    for ax, (title, im) in zip(axes, panels):
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")
        if im is None:
            ax.text(0.5, 0.5, "(file not found)", ha="center", va="center", fontsize=12)
        else:
            ax.imshow(im)
    fig.suptitle(f"{name}  —  new phonon+heat-capacity targets vs paper",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / f"compare_{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    made.append(out.name)

print("contact sheets written to", OUT)
for m in made:
    print("  ", m)
