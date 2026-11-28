#!/usr/bin/env python
"""Schematic of the static-benchmark pipeline (Fig. 1 of the paper).

Redrawn for the revision with two labels corrected to match the Methods:
the evaluation uses five random 80/20 splits of the training pool with a fixed
held-out test set (not five-fold cross-validation), and the ORB checkpoint is
orb_v3_conservative_inf_omat (not orbff-v2).

Writes figures/fig1_pipeline_schematic.pdf
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import rev_common as C

plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
TEAL, GREEN, ORANGE, BLUE, EDGE = '#66d4cc', '#a6e88a', '#ffbe66', '#99d3ee', '#444444'

BOXES = [
    (TEAL, '1,181 Crystal Structures (ASE Atoms)', ['elastic_tensor_2015  ·  8 target properties']),
    (TEAL, 'Five Random Splits', ['training pool = 500 (80/20 per split)  |  held-out test = 681']),
    (GREEN, 'ORB Featurizer (orb-v3)', [r'Forward pass $\rightarrow$ node embeddings', r'(N atoms × 256) $\rightarrow$ Mean pool $\rightarrow$ (1 × 256)']),
    (GREEN, 'PCA Dimensionality Reduction', [r'Fit on train  ·  d $\in$ {10, 25, 50}']),
    (ORANGE, 'BoTorch SingleTaskGP (×8 properties)', ['RBF Kernel + ARD lengthscales', r'Marginal Log-Likelihood Optimisation $\rightarrow$ Fitted Posterior']),
    (BLUE, 'Posterior Predictions', ['Mean ŷ (681 × 1)  +  Variance σ² (681 × 1)']),
    (BLUE, 'Evaluation Metrics', ['R²  ·  RMSE  ·  Spearman ρ  (per property, per split)']),
]
LEGEND = [(TEAL, 'Data / Input'), (GREEN, 'Processing'), (ORANGE, 'Surrogate Model'), (BLUE, 'Output / Evaluation')]


def main():
    fig, ax = plt.subplots(figsize=(3.62, 5.92))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16.4); ax.axis('off')
    x0, w, gap = 0.9, 8.2, 0.42
    y = 15.9
    centers = []
    for col, title, subs in BOXES:
        h = 1.45 if len(subs) == 1 else 1.75
        y -= h
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18',
                                    fc=col, ec=EDGE, lw=1.0))
        cy = y + h / 2
        if len(subs) == 1:
            ax.text(5, cy + 0.26, title, ha='center', va='center', fontsize=8.0, fontweight='bold')
            ax.text(5, cy - 0.32, subs[0], ha='center', va='center', fontsize=5.6)
        else:
            ax.text(5, cy + 0.46, title, ha='center', va='center', fontsize=8.0, fontweight='bold')
            ax.text(5, cy - 0.06, subs[0], ha='center', va='center', fontsize=5.6)
            ax.text(5, cy - 0.50, subs[1], ha='center', va='center', fontsize=5.6)
        centers.append((y, y + h))
        y -= gap
    for (lo, _), (_, hi_next) in zip(centers[:-1], centers[1:]):
        ax.annotate('', xy=(5, hi_next + 0.02), xytext=(5, lo - 0.02),
                    arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=1.0, mutation_scale=8))
    ly = y - 0.25
    for k, (col, lab) in enumerate(LEGEND):
        lx = 1.5 + 4.0 * (k % 2); yy = ly - 0.62 * (k // 2)
        ax.add_patch(FancyBboxPatch((lx, yy - 0.2), 0.45, 0.4, boxstyle='round,pad=0.02,rounding_size=0.06',
                                    fc=col, ec=EDGE, lw=0.8))
        ax.text(lx + 0.65, yy, lab, ha='left', va='center', fontsize=5.6)
    C.FIG.mkdir(exist_ok=True)
    out = C.FIG / 'fig1_pipeline_schematic.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(C.FIG / 'fig1_pipeline_schematic.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print('saved ->', out)


if __name__ == '__main__':
    main()
