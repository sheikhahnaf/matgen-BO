#!/usr/bin/env python
"""Posterior calibration of the GP surrogate.

Part A  static benchmark. Holdout predictive distributions from the GP rerun
        (calibration/gp_calibration_rerun.py) for the three datasets, four
        descriptors, PCA in {10, 25, 50} and n_train in {100, 250, 500}.
Part B  inside the loop. Step-ahead predictions of the gated runs (the causal
        replay of 02_closed_loop_bo/validation/three_way_comparison): at cycle s
        the GP is trained on the seed pool and on cycles before s and predicts
        the cycle-s candidates that the gate sent to the oracle.
Part C  does the miscalibration change what the gate selects? Expected
        Improvement top-4 sets on the real candidate pools of the ungated runs,
        with the GP standard deviation as fitted and rescaled by the factor
        that restores unit z-score spread.

Metrics: empirical coverage of the central 50/68/90/95 % intervals, RMS of the
z-scores (1 for a calibrated Gaussian predictive distribution), negative log
predictive density in standardized target units, miscalibration area, and the
Spearman correlation between absolute error and predicted standard deviation.
"""

import glob

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import rev_common as C

PROPS = {
    'elastic': ['K_VRH', 'K_Voigt', 'K_Reuss', 'G_VRH', 'G_Voigt', 'G_Reuss', 'elastic_anisotropy', 'poisson_ratio'],
    'dielectric': ['band_gap', 'n', 'poly_electronic', 'poly_total'],
    'dfpt': ['Cv_300K', 'S_300K', 'F_300K', 'max_phonon_freq'],
}
KIND_LABEL = {'elastic': 'Elastic (8)', 'dielectric': 'Dielectric (4)', 'dfpt': 'Phonon thermo. (4)'}
METRICS = ['cov50', 'cov68', 'cov90', 'cov95', 'z_rms', 'nlpd_std', 'miscal_area', 'rho_err_sd', 'sharpness']


def static_part():
    root = C.REV / 'calibration' / 'out'
    rows, curves = [], {}
    for kind in PROPS:
        files = sorted(glob.glob(str(root / kind / 'pca*_n*' / 'calib_preds_*.csv.gz')))
        for f in files:
            d = pd.read_csv(f)
            d = d[d['property'].isin(PROPS[kind])]
            for (desc, pca, n, prop, split), g in d.groupby(['descriptor', 'pca', 'n_train', 'property', 'split']):
                for sd_kind in ('sd_obs', 'sd_latent'):
                    m = C.calibration_metrics(g['y_true'], g['mu'], g[sd_kind], scale=g['y_train_std'].iloc[0])
                    rows.append(dict(kind=kind, descriptor=desc, pca=pca, n_train=n, property=prop,
                                     split=split, sd=sd_kind, **m))
            sel = d[(d['descriptor'] == 'orb') & (d['pca'] == 50) & (d['n_train'] == 500)]
            if len(sel):
                curves[kind] = sel
    if not rows:
        print('Part A: no calibration reruns found yet'); return None, None
    per = pd.DataFrame(rows)
    per.to_csv(C.OUT / 'static_calibration_per_split.csv.gz', index=False)
    prop = per.groupby(['kind', 'descriptor', 'pca', 'n_train', 'sd', 'property'])[METRICS].mean().reset_index()
    prop.to_csv(C.OUT / 'static_calibration_per_property.csv', index=False)
    ds = prop.groupby(['kind', 'descriptor', 'pca', 'n_train', 'sd'])[METRICS].mean().reset_index()
    ds.to_csv(C.OUT / 'static_calibration_per_dataset.csv', index=False)

    show = ['cov68', 'cov90', 'cov95', 'z_rms', 'nlpd_std', 'rho_err_sd']
    print('\n== Part A: ORB + GP, PCA = 50, n_train = 500, predictive sd including noise')
    print(ds[(ds.descriptor == 'orb') & (ds.pca == 50) & (ds.n_train == 500) & (ds.sd == 'sd_obs')]
          [['kind'] + show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\n   same cells, latent sd (noise excluded)')
    print(ds[(ds.descriptor == 'orb') & (ds.pca == 50) & (ds.n_train == 500) & (ds.sd == 'sd_latent')]
          [['kind'] + show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\n   all descriptors, PCA = 50, n_train = 500, sd including noise')
    print(ds[(ds.pca == 50) & (ds.n_train == 500) & (ds.sd == 'sd_obs')]
          .pivot(index='descriptor', columns='kind', values='cov95').to_string(float_format=lambda v: f'{v:.3f}'))
    print('\n   ORB + GP per property (PCA 50, n 500, sd including noise)')
    print(prop[(prop.descriptor == 'orb') & (prop.pca == 50) & (prop.n_train == 500) & (prop.sd == 'sd_obs')]
          [['kind', 'property'] + show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    return ds, curves


def loop_part():
    src = C.REPO / '02_closed_loop_bo' / 'validation' / 'three_way_comparison' / 'results'
    rows, data = [], {}
    for tgt in ('cp', 'bm'):
        d = pd.read_csv(src / f'gp_causal_allpreds_{tgt}_seeded.csv')
        data[tgt] = d
        scale = d['esen'].std()
        scopes = [('all backbones', d)] + [(C.BACKBONE_LABEL[b], d[d.paradigm == b]) for b in ('mg', 'cf', 'adit')]
        scopes += [('cycles 0-9', d[d.cycle_id <= 9]), ('cycles 10-19', d[d.cycle_id >= 10])]
        for label, g in scopes:
            m = C.calibration_metrics(g['esen'], g['gp_pred'], g['gp_sigma'], scale=scale)
            rows.append(dict(target=tgt, scope=label, **m))
        # one scalar variance factor learned on the first half of each run, applied to the second half
        early, late = d[d.cycle_id <= 9], d[d.cycle_id >= 10]
        c = np.sqrt(np.mean(((early.esen - early.gp_pred) / early.gp_sigma) ** 2))
        m = C.calibration_metrics(late['esen'], late['gp_pred'], c * late['gp_sigma'], scale=scale)
        rows.append(dict(target=tgt, scope=f'cycles 10-19, sd x {c:.2f} (factor from cycles 0-9)', **m))
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / 'loop_calibration.csv', index=False)
    print('\n== Part B: step-ahead predictions of the gated runs (latent sd, as used by the gate)')
    print(out[['target', 'scope', 'n', 'cov50', 'cov68', 'cov90', 'cov95', 'z_rms', 'nlpd_std', 'rho_err_sd']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    return out, data


def selection_part(z_rms):
    """Top-4 picks with the fitted sd and with the sd rescaled by the measured z-score spread."""
    rows = []
    for tgt in ('cp', 'bm'):
        try:
            df, X, seed, Xs = C.load_candidates(tgt)
        except FileNotFoundError:
            print(f'Part C: candidate table for {tgt} not available yet'); continue
        pca = PCA(n_components=50, random_state=0).fit(Xs)
        sur = C.fit_gate_gp(pca.transform(Xs), seed['y'].to_numpy())
        f_best = float(seed['y'].max())
        ok = np.isfinite(X).all(axis=1)
        ung = df[(~df['gated']) & ok & df['arm'].isin(['BASE', 'oracle_all'])]
        factor = z_rms[tgt]
        for (run, step), g in ung.groupby(['run', 'step']):
            if len(g) <= 4:
                continue
            Z = pca.transform(X[g.index.to_numpy()])
            mu, sd = sur.predict(Z)
            a = set(C.greedy_dpp_topk(C.expected_improvement(mu, sd, f_best), Z, 4))
            b = set(C.greedy_dpp_topk(C.expected_improvement(mu, factor * sd, f_best), Z, 4))
            m = set(np.argsort(-mu)[:4])
            rows.append(dict(target=tgt, backbone=g['backbone'].iloc[0], run=run, step=step, pool=len(g),
                             overlap_rescaled=len(a & b) / 4, overlap_mean_only=len(a & m) / 4))
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / 'ei_selection_sensitivity.csv', index=False)
    summ = out.groupby(['target', 'backbone'])[['pool', 'overlap_rescaled', 'overlap_mean_only']].mean()
    summ['pools'] = out.groupby(['target', 'backbone']).size()
    print('\n== Part C: share of the gate\'s four picks that stay the same')
    print('   overlap_rescaled: sd multiplied by the measured z-score spread; overlap_mean_only: ranking by the mean alone')
    print(summ.to_string(float_format=lambda v: f'{v:.3f}'))
    print(out.groupby('target')[['overlap_rescaled', 'overlap_mean_only']].mean().to_string(float_format=lambda v: f'{v:.3f}'))
    return out


def figure(curves, loop_data):
    plt = C.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    q = np.linspace(0.01, 0.99, 99)

    ax = axes[0]
    colors = {'elastic': '#1f77b4', 'dielectric': '#ff7f0e', 'dfpt': '#2ca02c'}
    if curves:
        for kind, d in curves.items():
            qq, emp = C.calibration_curve(d['y_true'], d['mu'], d['sd_obs'])
            ax.plot(qq, emp, color=colors[kind], lw=1.6, label=KIND_LABEL[kind])
    ax.plot([0, 1], [0, 1], '--', color='#777777', lw=1.0)
    ax.set_xlabel('predicted cumulative probability'); ax.set_ylabel('observed frequency')
    ax.set_title('(a) static benchmark, ORB + GP', loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper left'); ax.set_aspect('equal', 'box')

    ax = axes[1]
    for tgt, col, lab in (('cp', '#9467bd', r'$C_p$'), ('bm', '#8c564b', r'$K_{\mathrm{VRH}}$')):
        d = loop_data[tgt]
        qq, emp = C.calibration_curve(d['esen'], d['gp_pred'], d['gp_sigma'])
        ax.plot(qq, emp, color=col, lw=1.6, label=f'{lab} (n = {len(d)})')
    ax.plot([0, 1], [0, 1], '--', color='#777777', lw=1.0)
    ax.set_xlabel('predicted cumulative probability'); ax.set_ylabel('observed frequency')
    ax.set_title('(b) in the loop, step-ahead', loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper left'); ax.set_aspect('equal', 'box')

    ax = axes[2]
    from scipy.stats import norm
    grid = np.linspace(-6, 6, 400)
    for tgt, col, lab in (('cp', '#9467bd', r'$C_p$'), ('bm', '#8c564b', r'$K_{\mathrm{VRH}}$')):
        d = loop_data[tgt]
        z = ((d['esen'] - d['gp_pred']) / d['gp_sigma']).clip(-6, 6)
        ax.hist(z, bins=np.linspace(-6, 6, 49), density=True, histtype='step', color=col, lw=1.4, label=lab)
    ax.plot(grid, norm.pdf(grid), '--', color='#777777', lw=1.0, label='standard normal')
    ax.set_xlabel(r'z-score, (oracle $-$ GP mean) / GP sd'); ax.set_ylabel('density')
    ax.set_title('(c) z-scores in the loop', loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper right')
    C.save_fig(fig, 'fig_gp_calibration.png')


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    ds, curves = static_part()
    loop, loop_data = loop_part()
    z_rms = {t: float(loop[(loop.target == t) & (loop.scope == 'all backbones')]['z_rms'].iloc[0]) for t in ('cp', 'bm')}
    selection_part(z_rms)
    figure(curves, loop_data)


if __name__ == '__main__':
    main()
