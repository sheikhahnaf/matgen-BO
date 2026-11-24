"""Shared helpers for the revision analyses (data loading, the gate's GP, metrics)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'archive').is_dir())
REV = REPO / '04_revision_round1'
CAND = REV / 'candidate_table' / 'out'
OUT = REV / 'analysis' / 'out'
FIG = REV / 'figures'
R_GAS = 8.314462618  # J/mol/K

sys.path.insert(0, str(REPO / '02_closed_loop_bo'))

BACKBONE_LABEL = {'mg': 'MatterGen', 'cf': 'CrystalFlow', 'adit': 'ADiT'}
BACKBONE_COLOR = {'mg': '#1f77b4', 'cf': '#2ca02c', 'adit': '#d62728'}
TARGET_LABEL = {'cp': r'$C_p$ (J/g/K)', 'bm': r'$K_{\mathrm{VRH}}$ (GPa)'}


def set_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'Helvetica', 'font.size': 10, 'axes.labelsize': 10.5,
        'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'axes.linewidth': 0.8, 'savefig.dpi': 600,
        'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
    })
    return plt


def save_fig(fig, name):
    """Save under figures/; an existing file of the same name is moved to figures/legacy first."""
    import shutil
    import time
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    if path.exists():
        legacy = FIG / 'legacy'
        legacy.mkdir(exist_ok=True)
        shutil.move(str(path), str(legacy / f'{path.stem}_{time.strftime("%Y%m%d_%H%M%S")}{path.suffix}'))
    fig.savefig(path, bbox_inches='tight')
    print(f'saved -> {path}')
    return path


def load_candidates(target, tag='published'):
    d = CAND / tag
    df = pd.read_parquet(d / f'candidates_{target}.parquet')
    X = np.load(d / f'orb_raw_{target}.npy')
    seed = pd.read_parquet(d / f'seed_pool_{target}.parquet')
    Xs = np.load(d / f'orb_raw_seed_{target}.npy')
    assert len(df) == len(X) and len(seed) == len(Xs)
    return df, X, seed, Xs


# ----- the gate's surrogate ---------------------------------------------------

def fit_gate_gp(Z, y, age=None, noise_min=1e-6, decay=0.9):
    """The GP the gate fits: single-task GP on PCA-50 ORB features with the
    age-dependent fixed noise of the in-loop code (noise_min / decay**age)."""
    from src.surrogate import HCapSurrogate
    age = np.ones(len(y)) if age is None else np.asarray(age, dtype=float)
    y_var = noise_min / np.maximum(decay ** age, 1e-3)
    sur = HCapSurrogate(device='cpu')
    sur.fit(np.asarray(Z, dtype=float), np.asarray(y, dtype=float), y_var=y_var)
    return sur


def expected_improvement(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-8)
    z = (mu - f_best) / sigma
    return (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)


def greedy_dpp_topk(scores, features, k, lambda_div=0.5):
    """Same selection rule as the gate (EI score minus a cosine-similarity penalty)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) <= k:
        return np.argsort(-scores)
    fn = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    sim = fn @ fn.T
    selected = [int(np.argmax(scores))]
    remaining = set(range(len(scores))) - set(selected)
    for _ in range(k - 1):
        rem = np.array(sorted(remaining))
        adjusted = scores[rem] - lambda_div * sim[rem][:, selected].max(axis=1)
        pick = int(rem[np.argmax(adjusted)])
        selected.append(pick)
        remaining.discard(pick)
    return np.array(selected, dtype=int)


# ----- calibration metrics ----------------------------------------------------

LEVELS = (0.5, 0.68, 0.9, 0.95)


def calibration_metrics(y, mu, sd, scale=1.0):
    """Gaussian predictive-distribution diagnostics.

    scale: standard deviation of the training targets, used to report the
    negative log predictive density in standardized units so that values are
    comparable across properties (a predictor that returns the training mean
    with unit variance scores 1.419).
    """
    y, mu, sd = (np.asarray(a, dtype=float) for a in (y, mu, sd))
    sd = np.maximum(sd, 1e-12)
    z = (y - mu) / sd
    out = {'n': len(y), 'z_mean': z.mean(), 'z_rms': np.sqrt(np.mean(z ** 2))}
    for p in LEVELS:
        out[f'cov{int(round(p * 100))}'] = float(np.mean(np.abs(z) <= norm.ppf((1 + p) / 2)))
    s = np.asarray(scale, dtype=float)
    out['nlpd_std'] = float(np.mean(0.5 * np.log(2 * np.pi * (sd / s) ** 2) + 0.5 * z ** 2))
    q = np.linspace(0.01, 0.99, 99)
    emp = np.array([np.mean(norm.cdf(z) <= qi) for qi in q])
    out['miscal_area'] = float(np.mean(np.abs(emp - q)))
    from scipy.stats import spearmanr
    out['rho_err_sd'] = float(spearmanr(np.abs(y - mu), sd)[0]) if len(y) > 2 else np.nan
    out['sharpness'] = float(np.mean(sd / s))
    return out


def calibration_curve(y, mu, sd, n=99):
    z = (np.asarray(y, float) - np.asarray(mu, float)) / np.maximum(np.asarray(sd, float), 1e-12)
    q = np.linspace(0.01, 0.99, n)
    return q, np.array([np.mean(norm.cdf(z) <= qi) for qi in q])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (c - h, c + h)


# ----- chemistry descriptors --------------------------------------------------

def crystal_system(sg):
    if sg is None or sg < 1:
        return 'unknown'
    for hi, name in ((2, 'triclinic'), (15, 'monoclinic'), (74, 'orthorhombic'), (142, 'tetragonal'),
                     (167, 'trigonal'), (194, 'hexagonal'), (230, 'cubic')):
        if sg <= hi:
            return name
    return 'unknown'


def composition_flags(formula):
    from pymatgen.core import Composition
    els = list(Composition(formula).elements)
    return {
        'has_alkali': any(e.is_alkali for e in els),
        'has_alkaline_earth': any(e.is_alkaline for e in els),
        'has_transition_metal': any(e.is_transition_metal for e in els),
        'has_lanthanide_actinide': any(e.is_lanthanoid or e.is_actinoid for e in els),
        'has_oxygen': any(e.symbol == 'O' for e in els),
        'has_chalcogen_SSeTe': any(e.symbol in ('S', 'Se', 'Te') for e in els),
        'has_halogen': any(e.is_halogen for e in els),
        'has_BCN': any(e.symbol in ('B', 'C', 'N') for e in els),
        'has_hydrogen': any(e.symbol == 'H' for e in els),
    }


def dulong_petit_cp(mean_mass):
    """High-temperature limit of the mass-specific heat capacity, 3R / mean atomic mass (J/g/K)."""
    return 3.0 * R_GAS / np.asarray(mean_mass, dtype=float)
