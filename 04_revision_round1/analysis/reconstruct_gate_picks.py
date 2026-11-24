#!/usr/bin/env python
"""Reconstruct, cycle by cycle, which candidates the gate sent to the oracle in
the published gated (ACC) runs, and how each pick was made.

The gate logged how many candidates it dispatched per cycle and stored the
ones that returned a value, but not the identity of picks whose oracle call
failed. Everything needed to replay the selection is on disk:

    the candidates of every cycle            samples/step_XXXX_eval.extxyz  (candidate table)
    the gate's training set at every cycle   ltm_<run>.parquet, with the PCA-50 features used in the loop
    the selection rule                       EI on the GP posterior, then greedy EI - 0.5 * cosine similarity

For cycle s >= 1 we refit the GP on the stored rows of cycles < s with the
in-loop noise schedule, score that cycle's candidates, and run the same greedy
selection. Two checks tell whether the replay is faithful: the PCA basis must
reproduce the stored seed-pool features, and every candidate that the run
recorded as an oracle success must be among the reconstructed picks.

Each pick is labelled by how it was chosen:
    order 0      argmax of Expected Improvement
    order 1..3   greedy picks. We also record whether the same candidate would
                 have been chosen with the EI term removed (pure max-dissimilarity)
                 and whether it is in the top-K by EI alone.

Output: analysis/out/gate_picks_reconstructed_<target>.csv (one row per candidate
per cycle) and gate_picks_summary_<target>.csv (one row per run).
"""

import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import rev_common as C

MEM = C.CAND / 'published' / 'gate_memory'


def diversity_only(first, Z, k):
    fn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    sim = fn @ fn.T
    sel = [int(first)]
    while len(sel) < k:
        rem = [i for i in range(len(Z)) if i not in sel]
        sel.append(rem[int(np.argmin(sim[rem][:, sel].max(axis=1)))])
    return sel


def run_target(tgt):
    df, X, seed, Xs = C.load_candidates(tgt)
    rows, summ = [], []
    acc = df[df['arm'] == 'ACC']
    for run, g in acc.groupby('run'):
        mem_path = MEM / f'ltm_{run}.parquet'
        if not mem_path.exists():
            print(f'  {run}: gate memory not found, skipped'); continue
        mem = pd.read_parquet(mem_path)
        g = g.sort_values(['step', 'cand_idx'])
        c0 = g[g['step'] == 0]
        pca = PCA(n_components=50).fit(np.vstack([Xs, X[c0.index.to_numpy()]]))
        z_seed = pca.transform(Xs)
        z_stored = np.stack(mem[mem['cycle_id'] == -1]['Z_pca50'].apply(np.asarray).tolist())
        sign = np.sign((z_seed * z_stored).sum(axis=0))
        pca_err = float(np.abs(z_seed * sign - z_stored).max())

        n_succ = n_succ_in = n_picks = 0
        for s in sorted(g['step'].unique()):
            pool = g[g['step'] == s]
            N = len(pool)
            ok = pool['success'].to_numpy() == True
            if s == 0:      # cold start: every candidate goes to the oracle
                for i in range(N):
                    rows.append(dict(target=tgt, run=run, backbone=pool['backbone'].iloc[0], step=s, cand_idx=i, pool=N,
                                     picked=True, order=-1, how='cold_start', success=bool(ok[i])))
                continue
            tr = mem[mem['cycle_id'] < s]
            Ztr = np.stack(tr['Z_pca50'].apply(np.asarray).tolist())
            ytr = tr['y_cp'].to_numpy(dtype=float)
            cyc = tr['cycle_id'].to_numpy()
            age = np.maximum(np.where(cyc < 0, s, s - 1 - cyc).astype(float), 0.0)
            sur = C.fit_gate_gp(Ztr, ytr, age=age)
            Z = pca.transform(X[pool.index.to_numpy()]) * sign
            mu, sd = sur.predict(Z)
            f_best = float(np.nanmax(ytr))
            ei = C.expected_improvement(mu, sd, f_best)
            K = min(4, max(1, int(np.ceil(0.5 * N))))
            if K >= N:
                picks = list(range(N)); div = picks; top = set(picks)
            else:
                picks = [int(i) for i in C.greedy_dpp_topk(ei, Z, K)]
                div = diversity_only(picks[0], Z, K)
                top = set(int(i) for i in np.argsort(-ei)[:K])
            order = {i: k for k, i in enumerate(picks)}
            for i in range(N):
                rows.append(dict(
                    target=tgt, run=run, backbone=pool['backbone'].iloc[0], step=s, cand_idx=i, pool=N,
                    picked=i in order, order=order.get(i, -1),
                    how=('ei_argmax' if order.get(i) == 0 else 'greedy' if i in order else 'not_picked'),
                    same_as_diversity_only=(i in order and i in div), in_top_ei=i in top,
                    gp_mu=mu[i], gp_sigma=sd[i], ei=ei[i], z=(mu[i] - f_best) / sd[i], f_best=f_best,
                    success=bool(ok[i]),
                ))
            n_succ += int(ok.sum()); n_succ_in += int(sum(1 for i in np.where(ok)[0] if i in order)); n_picks += len(picks)
        r = pd.DataFrame([x for x in rows if x['run'] == run and x['step'] > 0])
        pk = r[r['picked']]
        summ.append(dict(target=tgt, run=run, backbone=g['backbone'].iloc[0], pca_max_abs_err=pca_err,
                         successes_after_cycle0=n_succ, successes_inside_picks=n_succ_in, picks=n_picks,
                         picks_same_as_diversity_only=float(pk['same_as_diversity_only'].mean()),
                         picks_in_top_ei=float(pk['in_top_ei'].mean()),
                         median_best_z=float(r.groupby('step')['z'].max().median()),
                         median_max_ei=float(r.groupby('step')['ei'].max().median()),
                         fail_rate_ei_pick=float(1 - pk[pk['order'] == 0]['success'].mean()),
                         fail_rate_greedy_picks=float(1 - pk[pk['order'] > 0]['success'].mean())))
        print('  ' + ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in summ[-1].items()), flush=True)
    pd.DataFrame(rows).to_csv(C.OUT / f'gate_picks_reconstructed_{tgt}.csv', index=False)
    s = pd.DataFrame(summ)
    s.to_csv(C.OUT / f'gate_picks_summary_{tgt}.csv', index=False)
    if len(s):
        print(f'\n== {tgt}: {len(s)} gated runs replayed')
        print(f'   recorded successes inside the reconstructed picks: {int(s.successes_inside_picks.sum())} of {int(s.successes_after_cycle0.sum())}')
        print(f'   picks identical to argmax-EI followed by pure max-dissimilarity: {np.average(s.picks_same_as_diversity_only, weights=s.picks):.3f}')
        print(f'   picks that are in the top-K by EI: {np.average(s.picks_in_top_ei, weights=s.picks):.3f}')
        print(s.groupby('backbone')[['median_best_z', 'median_max_ei', 'fail_rate_ei_pick', 'fail_rate_greedy_picks']].mean().round(4).to_string())


if __name__ == '__main__':
    C.OUT.mkdir(parents=True, exist_ok=True)
    for t in (sys.argv[1:] or ['cp', 'bm']):
        run_target(t)
