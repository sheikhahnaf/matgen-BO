#!/usr/bin/env python
"""Does the ORB embedding add anything over composition for the loop targets?

Labeled data: the GP seed pool plus every in-loop candidate with an exact
oracle value (gated runs: values stored by the gate; ungated runs: values
recovered from an unclipped reward).

Feature sets
    orb        mean-pooled ORB-v3 node embedding, PCA-50          (structure + composition)
    magpie     132 Magpie composition statistics, PCA-50          (composition only)
    mass       one feature, the mean atomic mass (1/M for C_p)    (composition only)
    dulong     C_p = 3R / M, no fitted parameter                   (C_p only)
Models: the benchmark's single-task GP for orb / magpie / mass, and a random
forest on the raw Magpie vector as a second composition-only learner.

Protocols
    P1  five-fold cross-validation on the pooled set, random folds
    P2  five-fold cross-validation with folds grouped by reduced formula
        (every test composition is unseen)
    P3  train on the seed pool, test on all in-loop candidates
        (what the gate knows at cycle 0)
    P4  pairs of different structures that share a reduced formula: fraction
        ordered correctly by out-of-fold ORB predictions from P2. A
        composition-only model cannot separate such pairs (0.5 by construction).
    P5  retrospective selection on the real candidate pools of the ungated
        runs: each model, trained on the seed pool, picks four candidates per
        cycle. Scored by the reward the loop would have received (0 for an
        oracle failure).
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

import rev_common as C

warnings.filterwarnings('ignore')
SEED = 0
MAX_LOOP = 2500


def magpie_matrix(formulas):
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    ep = ElementProperty.from_preset('magpie')
    cache = {}
    out = []
    for f in formulas:
        if f not in cache:
            cache[f] = np.asarray(ep.featurize(Composition(f)), dtype=float)
        out.append(cache[f])
    M = np.vstack(out)
    med = np.nanmedian(M, axis=0)
    idx = np.where(~np.isfinite(M))
    M[idx] = np.take(med, idx[1])
    return M


def gp_fit_predict(Xtr, ytr, Xte):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood
    torch.manual_seed(SEED)
    xs, ys = StandardScaler().fit(Xtr), StandardScaler().fit(ytr.reshape(-1, 1))
    Xt = torch.tensor(xs.transform(Xtr), dtype=torch.float64)
    yt = torch.tensor(ys.transform(ytr.reshape(-1, 1)), dtype=torch.float64)
    model = SingleTaskGP(Xt, yt)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    with torch.no_grad():
        post = model.posterior(torch.tensor(xs.transform(Xte), dtype=torch.float64))
    mu = ys.inverse_transform(post.mean.numpy().reshape(-1, 1)).ravel()
    sd = post.variance.sqrt().numpy().ravel() * float(ys.scale_[0])
    return mu, sd


def predict(model, feats, tr, te, y):
    """Fit `model` on rows tr and predict rows te. Returns (mu, sd or None)."""
    if model == 'orb_gp':
        p = PCA(n_components=min(50, len(tr) - 1), random_state=SEED).fit(feats['orb'][tr])
        return gp_fit_predict(p.transform(feats['orb'][tr]), y[tr], p.transform(feats['orb'][te]))
    if model == 'magpie_gp':
        s = StandardScaler().fit(feats['magpie'][tr])
        p = PCA(n_components=min(50, len(tr) - 1), random_state=SEED).fit(s.transform(feats['magpie'][tr]))
        return gp_fit_predict(p.transform(s.transform(feats['magpie'][tr])), y[tr],
                              p.transform(s.transform(feats['magpie'][te])))
    if model == 'magpie_rf':
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED).fit(feats['magpie'][tr], y[tr])
        return rf.predict(feats['magpie'][te]), None
    if model == 'mass_gp':
        return gp_fit_predict(feats['mass'][tr], y[tr], feats['mass'][te])
    if model == 'dulong':
        return feats['dulong'][te], None
    raise ValueError(model)


def score(y, mu):
    return dict(r2=r2_score(y, mu), rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
                mae=float(np.mean(np.abs(y - mu))), spearman=float(spearmanr(y, mu)[0]))


def run_target(tgt):
    from pymatgen.core import Composition
    df, X, seed, Xs = C.load_candidates(tgt)
    ok = np.isfinite(X).all(axis=1)
    lab = df[ok & (df['success'] == True) & df['y_exact']].copy()
    if len(lab) > MAX_LOOP:   # exact GPs on a CPU: cap the in-loop set with a fixed random subsample
        lab = lab.sample(n=MAX_LOOP, random_state=SEED).sort_index()
    lab_idx = lab.index.to_numpy()
    pool = pd.concat([
        pd.DataFrame(dict(source='seed', backbone='seed', run='seed', formula=seed['formula'], y=seed['y'].astype(float))),
        pd.DataFrame(dict(source='loop', backbone=lab['backbone'].values, run=lab['run'].values,
                          formula=lab['formula'].values, y=lab['y'].astype(float).values)),
    ], ignore_index=True)
    orb = np.vstack([Xs, X[lab_idx]])
    comps = [Composition(f) for f in pool['formula']]
    pool['reduced'] = [c.reduced_formula for c in comps]
    pool['mean_mass'] = [c.weight / c.num_atoms for c in comps]
    keep = np.isfinite(pool['y'].to_numpy()) & np.isfinite(orb).all(axis=1)
    pool, orb = pool[keep].reset_index(drop=True), orb[keep]
    y = pool['y'].to_numpy()
    mass = pool['mean_mass'].to_numpy()
    feats = dict(orb=orb, magpie=magpie_matrix(pool['formula'].tolist()),
                 mass=(1.0 / mass if tgt == 'cp' else mass).reshape(-1, 1),
                 dulong=C.dulong_petit_cp(mass))
    models = ['orb_gp', 'magpie_gp', 'magpie_rf', 'mass_gp'] + (['dulong'] if tgt == 'cp' else [])
    n_seed, n_loop = int((pool.source == 'seed').sum()), int((pool.source == 'loop').sum())
    print(f'\n######## {tgt}: {len(pool)} labeled structures ({n_seed} seed, {n_loop} in-loop), '
          f'{pool.reduced.nunique()} reduced formulas')
    if tgt == 'cp':
        lin = np.polyfit(feats['dulong'], y, 1)
        print(f'   Dulong-Petit value alone: R2 of the parameter-free prediction {r2_score(y, feats["dulong"]):.3f}, '
              f'R2 of a linear fit {r2_score(y, np.polyval(lin, feats["dulong"])):.3f}, '
              f'Spearman {spearmanr(y, feats["dulong"])[0]:.3f}')

    rows, oof = [], {}
    protocols = {
        'P1 random 5-fold': list(KFold(5, shuffle=True, random_state=SEED).split(orb)),
        'P2 formula-grouped 5-fold': list(GroupKFold(5).split(orb, y, pool['reduced'])),
        'P3 seed pool -> in-loop': [(np.where(pool.source == 'seed')[0], np.where(pool.source == 'loop')[0])],
    }
    for pname, splits in protocols.items():
        for m in models:
            mu = np.full(len(y), np.nan)
            for tr, te in splits:
                mu[te], _ = predict(m, feats, tr, te, y)
            te_all = np.isfinite(mu)
            rows.append(dict(target=tgt, protocol=pname, model=m, n_test=int(te_all.sum()), **score(y[te_all], mu[te_all])))
            if pname.startswith('P2'):
                oof[m] = mu
            print(f'   {pname:28s} {m:10s} R2 {rows[-1]["r2"]:.3f}  RMSE {rows[-1]["rmse"]:.4g}  rho {rows[-1]["spearman"]:.3f}', flush=True)
    res = pd.DataFrame(rows)

    # P4: same composition, different structure
    pairs = []
    thr = 0.02 * (np.percentile(y, 95) - np.percentile(y, 5))
    for red, g in pool.groupby('reduced'):
        if len(g) < 2:
            continue
        idx = g.index.to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if abs(y[i] - y[j]) > thr:
                    pairs.append((i, j))
    p4 = dict(target=tgt, n_formulas_with_polymorphs=int((pool.groupby('reduced').size() > 1).sum()),
              n_pairs=len(pairs), min_gap=thr)
    if pairs:
        i, j = np.array(pairs).T
        for m in ('orb_gp', 'magpie_gp'):
            d_true, d_pred = y[i] - y[j], oof[m][i] - oof[m][j]
            p4[f'{m}_pair_accuracy'] = float(np.mean(np.sign(d_true) == np.sign(d_pred)))
        within = pool.groupby('reduced')['y'].transform(lambda s: s - s.mean())
        p4['within_formula_sd'] = float(within[pool.groupby('reduced')['y'].transform('size') > 1].std())
        p4['total_sd'] = float(y.std())
    print('   P4 same-formula pairs:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p4.items()})

    # P5: retrospective selection of four per cycle on the ungated pools
    seed_idx = np.where(pool.source == 'seed')[0]
    ung = df[(~df['gated']) & ok & df['arm'].isin(['BASE', 'oracle_all'])].copy()
    ung['reward0'] = ung['reward'].fillna(0.0)
    Xu = X[ung.index.to_numpy()]
    cu = [Composition(f) for f in ung['formula']]
    mu_mass = np.array([c.weight / c.num_atoms for c in cu])
    fu = dict(orb=Xu, magpie=magpie_matrix(ung['formula'].tolist()),
              mass=(1.0 / mu_mass if tgt == 'cp' else mu_mass).reshape(-1, 1), dulong=C.dulong_petit_cp(mu_mass))
    preds = {}
    for m in models:
        if m == 'dulong':
            preds[m] = (fu['dulong'], None); continue
        joint = {k: np.vstack([feats[k][seed_idx], fu[k]]) if feats[k].ndim == 2 else None for k in ('orb', 'magpie', 'mass')}
        joint['dulong'] = None
        tr = np.arange(len(seed_idx)); te = np.arange(len(seed_idx), len(seed_idx) + len(ung))
        preds[m] = predict(m, joint, tr, te, np.concatenate([y[seed_idx], np.zeros(len(ung))]))
    f_best = float(y[seed_idx].max())
    sel_rows = []
    ung = ung.reset_index(drop=True)
    for (run, step), g in ung.groupby(['run', 'step']):
        if len(g) <= 4:
            continue
        ii = g.index.to_numpy()
        r = g['reward0'].to_numpy()
        base = dict(target=tgt, backbone=g['backbone'].iloc[0], run=run, step=step, pool=len(g),
                    random_mean=r.mean(), best4_mean=np.sort(r)[-4:].mean(), pool_max=r.max())
        for m in models:
            mu, sd = preds[m]
            pick = np.argsort(-mu[ii])[:4]
            base[f'{m}_mean'] = r[pick].mean(); base[f'{m}_max'] = r[pick].max()
            if sd is not None and m.endswith('_gp'):
                pick = np.argsort(-C.expected_improvement(mu[ii], sd[ii], f_best))[:4]
                base[f'{m}_ei_mean'] = r[pick].mean(); base[f'{m}_ei_max'] = r[pick].max()
        sel_rows.append(base)
    sel = pd.DataFrame(sel_rows)
    cols = [c for c in sel.columns if c.endswith('_mean')]
    print('   P5 mean reward of the four picks per cycle (ungated pools):')
    print(sel.groupby('backbone')[cols].mean().round(4).to_string())
    print('   pooled:', sel[cols].mean().round(4).to_dict())
    print('   share of cycles in which the pick contains the best candidate of the pool:')
    hit = {c[:-4]: float(np.mean(sel[c] >= sel['pool_max'] - 1e-12)) for c in sel.columns if c.endswith('_max') and c != 'pool_max'}
    print('  ', {k: round(v, 3) for k, v in hit.items()}, ' random expectation', round(float(np.mean(4 / sel['pool'])), 3))
    return res, p4, sel


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or ['cp', 'bm']
    all_res, all_p4, all_sel = [], [], []
    for t in targets:
        res, p4, sel = run_target(t)
        all_res.append(res); all_p4.append(p4); all_sel.append(sel)
    pd.concat(all_res).to_csv(C.OUT / 'composition_baselines_scores.csv', index=False)
    pd.DataFrame(all_p4).to_csv(C.OUT / 'composition_baselines_same_formula_pairs.csv', index=False)
    pd.concat(all_sel).to_csv(C.OUT / 'composition_baselines_selection.csv', index=False)


if __name__ == '__main__':
    main()
