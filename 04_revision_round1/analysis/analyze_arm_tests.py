#!/usr/bin/env python
"""Two supporting comparisons for the closed-loop arms of the revision.

1. Pooled test of each control against the gate (ACC). Within every
   (target, backbone) cell the final running best of the two arms is
   standardized with the mean and standard deviation of the ten runs of that
   cell, and the standardized values of all cells are compared with a two-sided
   Mann-Whitney test. The number of cells in which ACC has the higher mean is
   reported next to it. The same is done for the running best averaged over
   budgets of 8 to 80 calls (out/value_cost_auc.csv describes that quantity).

2. Values returned by the oracle with and without generator updates. For the
   gate with fine-tuning (ACC) and the gate with a frozen generator, the mean
   oracle value of the candidates that returned a value is computed for cycles
   1-9 and for cycles 10-19 of every run.

Writes  out/arm_pooled_tests.csv, out/frozen_label_values.csv
"""
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import rev_common as C
import analyze_new_arms as A
import analyze_value_cost as V


def main():
    per = pd.read_csv(C.OUT / 'new_arms_per_run.csv')
    runs = A.collect().assign(run=lambda d: d['dir'].map(os.path.basename)).set_index('run')
    per['final'] = [np.nan_to_num(v, nan=A.LIM[t][0]) for v, t in zip(per.final_best, per.target)]
    avg = []
    for _, r in per.iterrows():
        calls, rb = A.run_arrays(runs.loc[r.run])[:2]
        avg.append(np.mean([V.at_budget(calls, rb, b, A.LIM[r.target][0]) for b in V.GRID]))
    per['budget_mean'] = avg

    rows = []
    for arm in ('cap4', 'frozen_gate', 'ei_only', 'div_only', 'BASE'):
        for q in ('final', 'budget_mean'):
            za, zb, higher, cells = [], [], 0, 0
            for (t, b), g in per.groupby(['target', 'backbone']):
                x = g[g.arm == 'ACC'][q].to_numpy(); y = g[g.arm == arm][q].to_numpy()
                if not len(x) or not len(y):
                    continue
                both = np.concatenate([x, y]); m, s = both.mean(), both.std(ddof=1)
                za += list((x - m) / s); zb += list((y - m) / s)
                cells += 1; higher += int(x.mean() > y.mean())
            rows.append(dict(control=arm, quantity=q, cells=cells, cells_ACC_higher=higher, n_ACC=len(za), n_control=len(zb),
                             mean_z_ACC=np.mean(za), mean_z_control=np.mean(zb),
                             p=mannwhitneyu(za, zb, alternative='two-sided').pvalue))
    pooled = pd.DataFrame(rows)
    pooled.to_csv(C.OUT / 'arm_pooled_tests.csv', index=False)
    print(pooled.round(4).to_string(index=False))

    rows = []
    for _, r in per[per.arm.isin(['ACC', 'frozen_gate'])].iterrows():
        mn, mx = A.LIM[r.target]
        l = pd.read_csv(f"{runs.loc[r.run]['dir']}/samples/long_term_memory.csv", usecols=['reward', 'RL_step'])
        l = l[l.reward > 0]
        v = l.reward * (mx - mn) + mn
        rows.append(dict(target=r.target, backbone=r.backbone, arm=r.arm, seed=r.seed,
                         cycles_1_9=v[(l.RL_step >= 1) & (l.RL_step < 10)].mean(), cycles_10_19=v[l.RL_step >= 10].mean()))
    lab = pd.DataFrame(rows)
    out = []
    for (t, b), g in lab.groupby(['target', 'backbone']):
        x = g[g.arm == 'ACC']; y = g[g.arm == 'frozen_gate']
        out.append(dict(target=t, backbone=b,
                        ACC_early=x.cycles_1_9.mean(), ACC_late=x.cycles_10_19.mean(),
                        frozen_early=y.cycles_1_9.mean(), frozen_late=y.cycles_10_19.mean(),
                        p_late=mannwhitneyu(x.cycles_10_19.dropna(), y.cycles_10_19.dropna(), alternative='two-sided').pvalue))
    out = pd.DataFrame(out)
    out.to_csv(C.OUT / 'frozen_label_values.csv', index=False)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
