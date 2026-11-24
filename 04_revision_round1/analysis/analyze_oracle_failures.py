#!/usr/bin/env python
"""Heat-capacity oracle failures: how often, why, and what they correlate with.

Population: the candidate table built from the raw run folders
(candidate_table/build_candidate_table.py).

  Part 1  failure rate and failure reason per backbone and arm. For runs without
          the gate every dispatched candidate has a known outcome. For gated
          runs the number of dispatched candidates per run comes from the gate
          log and the number of returned values from the run memory.
  Part 2  candidate-level association between an imaginary-mode failure and
          (a) the gate surrogate's view of the candidate, (b) chemistry,
          (c) cell geometry and symmetry, (d) generator identity. Uses the
          ungated runs only, where no selection has been applied. The surrogate
          is the gate's GP as it stands at the start of every gated run
          (trained on the seed pool only), so each candidate is scored by a
          model that has not seen it.
  Part 3  GPU out-of-memory failures against cell size.

Writes CSV tables to analysis/out/ and figures to figures/.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rev_common as C

RNG = np.random.default_rng(0)
ARM_ORDER = ['BASE', 'cap4', 'cap4ext', 'ACC']


def auc_ci(score, label, groups, n_boot=2000):
    """AUC of `score` for label==1, with a run-clustered bootstrap interval."""
    ok = np.isfinite(score)
    score, label, groups = score[ok], label[ok], groups[ok]
    auc = roc_auc_score(label, score)
    runs = np.unique(groups)
    idx_by_run = {r: np.where(groups == r)[0] for r in runs}
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(runs, size=len(runs), replace=True)
        idx = np.concatenate([idx_by_run[r] for r in pick])
        if label[idx].min() == label[idx].max():
            continue
        boots.append(roc_auc_score(label[idx], score[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def _count(v):
    return 0 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    df, X, seed, Xs = C.load_candidates('cp')
    summ = pd.read_csv(C.CAND / 'published' / 'run_summary_cp.csv')

    # ------------------------------------------------------------------ Part 1
    rows = []
    for _, r in summ.iterrows():
        gated = r['arm'] in ('ACC', 'frozen_gate')
        n_disp = int(r['gate_n_oracle']) if gated else int(r['n_candidates'])
        n_ok = int(r['n_success'])
        oom = _count(r.get('log_exception_OutOfMemoryError'))
        rows.append(dict(run=r['run'], backbone=r['backbone'], arm=r['arm'], seed=r['seed'],
                         dispatched=n_disp, returned=n_ok, failed=n_disp - n_ok,
                         fail_rate=(n_disp - n_ok) / n_disp,
                         imaginary=_count(r.get('log_imaginary_modes')), oom_ungated=oom if not gated else np.nan))
    per_run = pd.DataFrame(rows)
    per_run.to_csv(C.OUT / 'cp_failure_per_run.csv', index=False)

    agg = []
    for (bb, arm), g in per_run.groupby(['backbone', 'arm']):
        k, n = int(g['failed'].sum()), int(g['dispatched'].sum())
        lo, hi = C.wilson(k, n)
        agg.append(dict(backbone=bb, arm=arm, runs=len(g), dispatched=n, failed=k, fail_rate=k / n,
                        ci_lo=lo, ci_hi=hi, seed_mean=g['fail_rate'].mean(), seed_sd=g['fail_rate'].std(ddof=1)))
    agg = pd.DataFrame(agg)
    agg['arm'] = pd.Categorical(agg['arm'], ARM_ORDER, ordered=True)
    agg = agg.sort_values(['backbone', 'arm'])
    agg.to_csv(C.OUT / 'cp_failure_by_backbone_arm.csv', index=False)
    print('\n== Part 1: failure rate by backbone and arm')
    print(agg.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print('\n   gated vs ungated (seed-level Mann-Whitney on per-run failure rate)')
    for bb in ('mg', 'cf', 'adit'):
        a = per_run[(per_run.backbone == bb) & (per_run.arm == 'ACC')]['fail_rate']
        b = per_run[(per_run.backbone == bb) & (per_run.arm == 'BASE')]['fail_rate']
        p = mannwhitneyu(a, b, alternative='two-sided').pvalue
        print(f'   {bb}: ACC {a.mean():.3f} +- {a.std(ddof=1):.3f} (n={len(a)})  '
              f'BASE {b.mean():.3f} +- {b.std(ddof=1):.3f} (n={len(b)})  p={p:.3f}')

    ung = df[~df['gated']].copy()
    reasons = (ung[ung['success'] == False].groupby(['backbone', 'fail_reason']).size()
               .unstack(fill_value=0))
    reasons.to_csv(C.OUT / 'cp_failure_reasons.csv')
    print('\n   failure reasons, ungated runs'); print(reasons)

    # ------------------------------------------------------------------ Part 2
    # the gate's GP at the start of a run: PCA-50 and GP fitted on the seed pool
    pca = PCA(n_components=50, random_state=0).fit(Xs)
    sur = C.fit_gate_gp(pca.transform(Xs), seed['y'].to_numpy())
    f_best = float(seed['y'].max())
    emb_ok = np.isfinite(X).all(axis=1)
    mu = np.full(len(df), np.nan); sd = np.full(len(df), np.nan)
    mu[emb_ok], sd[emb_ok] = sur.predict(pca.transform(X[emb_ok]))
    df['gp_mu'], df['gp_sigma'] = mu, sd
    df['gp_ei'] = C.expected_improvement(mu, sd, f_best)
    df['cp_dulong_petit'] = C.dulong_petit_cp(df['mean_mass'])
    df['crystal_system'] = df['spacegroup'].map(C.crystal_system)
    df['is_P1'] = (df['spacegroup'] == 1).astype(float)
    flags = pd.DataFrame([C.composition_flags(f) for f in df['formula']], index=df.index).astype(float)
    df = pd.concat([df, flags], axis=1)
    df.drop(columns=['atoms_json']).to_csv(C.OUT / 'cp_candidates_scored.csv.gz', index=False)

    ung = df[(~df['gated']) & emb_ok].copy()
    phys = ung[ung['fail_reason'] != 'exception_OutOfMemoryError'].copy()   # returned a value or imaginary modes
    phys['fail'] = (phys['success'] == False).astype(int)
    print(f'\n== Part 2: {len(phys)} ungated candidates with a physical outcome, '
          f'{phys.fail.sum()} imaginary-mode failures ({phys.fail.mean():.3f})')

    scalar = ['gp_mu', 'gp_sigma', 'gp_ei', 'cp_dulong_petit', 'mean_mass', 'n_atoms', 'n_elements',
              'vol_per_atom', 'density', 'mlip_energy_per_atom', 'is_P1'] + list(flags.columns)
    assoc = []
    for name in scalar:
        for scope, sub in [('all', phys)] + [(bb, phys[phys.backbone == bb]) for bb in ('mg', 'cf', 'adit')]:
            s, y, g = sub[name].to_numpy(float), sub['fail'].to_numpy(), sub['run'].to_numpy()
            if np.nanstd(s) == 0:
                continue
            auc, lo, hi = auc_ci(s, y, g)
            assoc.append(dict(feature=name, scope=scope, n=int(np.isfinite(s).sum()), auc=auc, ci_lo=lo, ci_hi=hi,
                              mean_fail=np.nanmean(s[y == 1]), mean_ok=np.nanmean(s[y == 0])))
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(C.OUT / 'cp_failure_feature_auc.csv', index=False)
    piv = assoc.pivot(index='feature', columns='scope', values='auc').loc[scalar, ['all', 'mg', 'cf', 'adit']]
    print('   AUC of each feature for predicting an imaginary-mode failure (0.5 = no association)')
    print(piv.to_string(float_format=lambda v: f'{v:.3f}'))

    # failure rate across quintiles of the gate surrogate's outputs
    quint = []
    for name in ('gp_mu', 'gp_ei', 'gp_sigma', 'cp_dulong_petit'):
        for bb in ('mg', 'cf', 'adit'):
            sub = phys[phys.backbone == bb]
            q = pd.qcut(sub[name].rank(method='first'), 5, labels=False)
            for k, g in sub.groupby(q):
                lo, hi = C.wilson(int(g.fail.sum()), len(g))
                quint.append(dict(feature=name, backbone=bb, quintile=int(k) + 1, n=len(g),
                                  median=g[name].median(), fail_rate=g.fail.mean(), ci_lo=lo, ci_hi=hi))
    quint = pd.DataFrame(quint)
    quint.to_csv(C.OUT / 'cp_failure_by_quintile.csv', index=False)

    by_sys = (phys.groupby(['backbone', 'crystal_system'])['fail'].agg(n='size', fail_rate='mean').reset_index())
    by_sys.to_csv(C.OUT / 'cp_failure_by_crystal_system.csv', index=False)
    by_nel = (phys.groupby(['backbone', 'n_elements'])['fail'].agg(n='size', fail_rate='mean').reset_index())
    by_nel.to_csv(C.OUT / 'cp_failure_by_n_elements.csv', index=False)
    print('\n   failure rate by crystal system (all backbones pooled)')
    print(phys.groupby('crystal_system')['fail'].agg(n='size', fail_rate='mean').to_string(float_format=lambda v: f'{v:.3f}'))
    print('\n   failure rate by number of elements (pooled)')
    print(phys.groupby('n_elements')['fail'].agg(n='size', fail_rate='mean').to_string(float_format=lambda v: f'{v:.3f}'))

    # how predictable is failure from everything together (run-grouped cross-validation)
    feats = ['gp_mu', 'gp_sigma', 'gp_ei', 'mean_mass', 'n_atoms', 'n_elements', 'vol_per_atom', 'density',
             'mlip_energy_per_atom', 'is_P1'] + list(flags.columns)
    sets = {
        'backbone only': ['bb_mg', 'bb_cf'],
        'gate surrogate (mu, sigma, EI)': ['gp_mu', 'gp_sigma', 'gp_ei'],
        'composition and geometry': [f for f in feats if not f.startswith('gp_')],
        'all': feats + ['bb_mg', 'bb_cf'],
    }
    phys['bb_mg'] = (phys.backbone == 'mg').astype(float)
    phys['bb_cf'] = (phys.backbone == 'cf').astype(float)
    cv_rows = []
    for label, cols in sets.items():
        Xm, y, g = phys[cols].to_numpy(float), phys['fail'].to_numpy(), phys['run'].to_numpy()
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Xm, y, g):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(Xm[tr], y[tr])
            pred[te] = m.predict_proba(Xm[te])[:, 1]
        cv_rows.append(dict(feature_set=label, n_features=len(cols), cv_auc=roc_auc_score(y, pred)))
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(C.OUT / 'cp_failure_logistic_cv.csv', index=False)
    print('\n   logistic model, 5-fold cross-validation grouped by run'); print(cv.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(
        phys[sets['all']].to_numpy(float), phys['fail'].to_numpy())
    coef = pd.Series(m[-1].coef_[0], index=sets['all']).sort_values(key=np.abs, ascending=False)
    coef.rename('standardized_coefficient').to_csv(C.OUT / 'cp_failure_logistic_coefficients.csv')
    print('\n   standardized logistic coefficients (positive = more failures)'); print(coef.head(12).to_string(float_format=lambda v: f'{v:+.3f}'))

    # do the gated and ungated policies see different candidate pools?
    comp_cols = sets['composition and geometry']
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    phys['p_fail'] = np.nan
    for tr, te in GroupKFold(n_splits=5).split(phys[comp_cols], phys['fail'], phys['run']):
        clf.fit(phys.iloc[tr][comp_cols].to_numpy(float), phys.iloc[tr]['fail'].to_numpy())
        phys.iloc[te, phys.columns.get_loc('p_fail')] = clf.predict_proba(phys.iloc[te][comp_cols].to_numpy(float))[:, 1]
    clf.fit(phys[comp_cols].to_numpy(float), phys['fail'].to_numpy())
    gat = df[df['gated'] & emb_ok].copy()
    gat = gat[np.isfinite(gat[comp_cols].to_numpy(float)).all(axis=1)]
    gat['p_fail'] = clf.predict_proba(gat[comp_cols].to_numpy(float))[:, 1]
    pool_rows = []
    for bb in ('mg', 'cf', 'adit'):
        for label, sub in (('BASE pool', phys[(phys.backbone == bb) & (phys.arm == 'BASE')]),
                           ('ACC pool', gat[gat.backbone == bb])):
            pool_rows.append(dict(backbone=bb, pool=label, n=len(sub), predicted_fail=sub['p_fail'].mean(),
                                  gp_mu=sub['gp_mu'].mean(), mean_mass=sub['mean_mass'].mean(),
                                  frac_P1=sub['is_P1'].mean(), density=sub['density'].mean()))
    pools = pd.DataFrame(pool_rows)
    pools.to_csv(C.OUT / 'cp_pool_comparison.csv', index=False)
    print('\n   candidate pools seen by the two policies (failure probability from the composition-and-geometry model)')
    print(pools.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ------------------------------------------------------------------ Part 3
    oom = ung.copy(); oom['oom'] = (oom['fail_reason'] == 'exception_OutOfMemoryError').astype(int)
    print(f'\n== Part 3: out-of-memory failures {oom.oom.sum()} of {len(oom)} dispatched')
    print(oom.groupby('backbone')[['oom', 'n_atoms']].agg(oom_rate=('oom', 'mean'), n_atoms_median=('n_atoms', 'median')))
    print('   n_atoms: OOM median %.0f (min %d), others median %.0f' % (
        oom[oom.oom == 1].n_atoms.median(), oom[oom.oom == 1].n_atoms.min(), oom[oom.oom == 0].n_atoms.median()))
    oom.groupby('backbone').apply(lambda g: pd.Series({
        'dispatched': len(g), 'oom': int(g.oom.sum()), 'oom_rate': g.oom.mean(),
        'n_atoms_median_oom': g[g.oom == 1].n_atoms.median(), 'n_atoms_median_other': g[g.oom == 0].n_atoms.median(),
    })).to_csv(C.OUT / 'cp_oom_by_backbone.csv')

    # ------------------------------------------------------------------ Figure
    plt = C.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), constrained_layout=True)

    ax = axes[0]
    width = 0.2
    arms = ['BASE', 'cap4', 'ACC']
    arm_label = {'BASE': 'BASE (all dispatched)', 'cap4': 'cap-4', 'ACC': 'ACC (gate picks)'}
    shade = {'BASE': 0.45, 'cap4': 0.7, 'ACC': 1.0}
    for i, bb in enumerate(('mg', 'cf', 'adit')):
        present = [a for a in arms if ((agg.backbone == bb) & (agg.arm == a)).any()]
        for j, a in enumerate(present):
            r = agg[(agg.backbone == bb) & (agg.arm == a)].iloc[0]
            x = i + (j - (len(present) - 1) / 2) * width
            ax.bar(x, r.fail_rate, width * 0.92, color=C.BACKBONE_COLOR[bb], alpha=shade[a],
                   edgecolor='black', linewidth=0.5,
                   label=arm_label[a] if i == 0 else None)
            ax.errorbar(x, r.fail_rate, yerr=[[r.fail_rate - r.ci_lo], [r.ci_hi - r.fail_rate]],
                        color='black', lw=0.8, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels([C.BACKBONE_LABEL[b] for b in ('mg', 'cf', 'adit')])
    ax.set_ylabel('fraction of dispatched candidates\nwith no returned $C_p$'); ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    ax.set_title('(a) failure rate', loc='left', fontweight='bold')

    for ax, name, xl, tag in ((axes[1], 'gp_mu', 'quintile of GP-predicted $C_p$', '(b) against the surrogate mean'),
                              (axes[2], 'gp_ei', 'quintile of Expected Improvement', '(c) against the acquisition score')):
        for bb in ('mg', 'cf', 'adit'):
            g = quint[(quint.feature == name) & (quint.backbone == bb)].sort_values('quintile')
            ax.errorbar(g.quintile, g.fail_rate, yerr=[g.fail_rate - g.ci_lo, g.ci_hi - g.fail_rate],
                        marker='o', ms=4, lw=1.2, capsize=2, color=C.BACKBONE_COLOR[bb], label=C.BACKBONE_LABEL[bb])
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(['lowest', '2', '3', '4', 'highest'])
        ax.set_xlabel(xl); ax.set_ylim(0, 1.0); ax.set_ylabel('imaginary-mode failure rate')
        ax.set_title(tag, loc='left', fontweight='bold')
    axes[1].legend(frameon=False, loc='upper left', fontsize=8)
    C.save_fig(fig, 'fig_cp_oracle_failures.png')


if __name__ == '__main__':
    main()
