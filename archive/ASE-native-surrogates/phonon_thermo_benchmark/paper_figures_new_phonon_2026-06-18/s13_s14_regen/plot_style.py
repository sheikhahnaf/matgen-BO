"""Global plot style for FME paper figure refresh (Phase 1).

One font family (Helvetica), coordinated sizes, paper-quality output.
Apply via `from plot_style import apply_style; apply_style()` at the top of every plotter.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


FONT_FAMILY = "Helvetica"

# Coordinated font sizes (per global plot quality memo)
SZ_AXIS = 11
SZ_TICK = 10
SZ_SUBTITLE = 11
SZ_SUPTITLE = 13
SZ_LEGEND = 9
SZ_ANNOT = 9
SZ_VALUE_LABEL = 8

# Consistent colour palette across the paper
SURROGATE_COLOURS = {
    "gp": "#1f77b4",
    "mtgp_2": "#ff7f0e",
    "mtgp": "#ff7f0e",
    "dgp": "#2ca02c",
}
SURROGATE_LABEL = {"gp": "GP", "mtgp_2": "MTGP", "mtgp": "MTGP", "dgp": "DGP"}

DESCRIPTOR_ORDER = ["mace", "orb", "soap", "uma"]
DESCRIPTOR_LABEL = {"mace": "MACE", "orb": "ORB", "soap": "SOAP", "uma": "UMA"}


def apply_style() -> None:
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": [FONT_FAMILY, "Arial", "DejaVu Sans"],
        "font.size": SZ_AXIS,
        "axes.titlesize": SZ_SUBTITLE,
        "axes.labelsize": SZ_AXIS,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": SZ_TICK,
        "ytick.labelsize": SZ_TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.fontsize": SZ_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
    })


def archive_to_legacy(figure_path: Path) -> Path | None:
    """Move an existing figure into ../legacy/ before overwriting. Idempotent."""
    figure_path = Path(figure_path)
    if not figure_path.exists():
        return None
    legacy_dir = figure_path.parent / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    target = legacy_dir / figure_path.name
    if target.exists():
        return target
    shutil.move(str(figure_path), str(target))
    return target


def save_figure(fig, out_path: Path) -> Path:
    """Save with tight bbox, archiving any pre-existing file to legacy/."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_to_legacy(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}" + (f"  (legacy ← {archived.name})" if archived else ""))
    return out_path
