#!/usr/bin/env python
"""Value against oracle cost for every arm, backbone and target.

The main-text cost figures compare arms by the running best reached at a given
number of dispatched oracle calls. This script draws the same curves for all
arms of the revision (figure) and reads them at fixed budgets (table).

For each run the running best is a step function of the cumulative number of
dispatched calls (one step per cycle). At a budget B we take the last cycle
whose cumulative cost is at most B. A run that has not returned any value by
then is counted at the lower clip of the reward (0.25 J/g/K, 20 GPa). A run
that ends before B keeps its final value.

Reads   the run folders listed in analyze_new_arms.py
Writes  out/value_cost_at_budget.csv      mean and sd of the running best at 20, 40, 60, 80 calls
        out/value_cost_auc.csv            mean running best over budgets 8..80 (step 4), with p against ACC
        figures/fig_rev_value_cost.png
"""
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import rev_common as C
import analyze_new_arms as A

BUDGETS = [20, 40, 60, 80]
GRID = np.arange(8, 81, 4)
ARMS = ['ACC', 'ei_only', 'div_only', 'frozen_gate', 'cap4', 'BASE']
LABEL = {'ACC': 'gate (ACC)', 'ei_only': 'EI only', 'div_only': 'diversity only',
         'frozen_gate': 'gate, generator frozen', 'cap4': 'cap-4', 'BASE': 'BASE'}
STYLE = {'ACC': ('#1f77b4', '-', 'o'), 'ei_only': ('#d62728', '-', 'P'), 'div_only': ('#8c564b', '-', 'X'),
         'frozen_gate': ('#ff7f0e', '-', 'v'), 'cap4': ('#2ca02c', '-.', '^'), 'BASE': ('#7f7f7f', '--', 's')}


def at_budget(calls, rb, b, floor):
    k = np.searchsorted(calls, b, side='right') - 1
    if k < 0 or np.isnan(rb[k]):
        return floor
    return rb[k]


def main():
    per = pd.read_csv(C.OUT / 'new_arms_per_run.csv')
    runs = A.collect().assign(run=lambda d: d['dir'].map(os.path.basename)).set_index('run')
    curves = {r: A.run_arrays(runs.loc[r])[:2] for r in per.run}

    rows = []
    for _, r in per.iterrows():
        calls, rb = curves[r.run]
        floor = A.LIM[r.target][0]
        d = dict(target=r.target, backbone=r.backbone, arm=r.arm, seed=r.seed)
        for b in BUDGETS:
            d[f'best_at_{b}'] = at_budget(calls, rb, b, floor)
        d['mean_over_budgets'] = float(np.mean([at_budget(calls, rb, b, floor) for b in GRID]))
        rows.append(d)
    pr = pd.DataFrame(rows)
    cols = [f'best_at_{b}' for b in BUDGETS]
    tab = pr.groupby(['target', 'backbone', 'arm'])[cols].agg(['mean', 'std']).round(3)
    tab.columns = [f'{c}_{s}' for c, s in tab.columns]
    tab.reset_index().to_csv(C.OUT / 'value_cost_at_budget.csv', index=False)

    auc = []
    for (t, b, a), g in pr.groupby(['target', 'backbone', 'arm']):
        ref = pr[(pr.target == t) & (pr.backbone == b) & (pr.arm == 'ACC')].mean_over_budgets
        p = mannwhitneyu(g.mean_over_budgets, ref, alternative='two-sided').pvalue if a != 'ACC' else np.nan
        auc.append(dict(target=t, backbone=b, arm=a, n=len(g), mean=g.mean_over_budgets.mean(),
                        sd=g.mean_over_budgets.std(ddof=1), p_vs_ACC=p))
    auc = pd.DataFrame(auc)
    auc['rank'] = auc.groupby(['target', 'backbone'])['mean'].rank(ascending=False).astype(int)
    auc.to_csv(C.OUT / 'value_cost_auc.csv', index=False)
    pd.set_option('display.width', 220)
    print(auc.round(3).to_string(index=False))
    print(tab.reset_index()[['target', 'backbone', 'arm', 'best_at_40_mean', 'best_at_40_std', 'best_at_80_mean', 'best_at_80_std']].to_string(index=False))

    plt = C.set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.0), constrained_layout=True)
    handles = {}
    for i, t in enumerate(('cp', 'bm')):
        for j, b in enumerate(('mg', 'cf', 'adit')):
            ax = axes[i, j]
            for a in ARMS[::-1]:
                g = per[(per.target == t) & (per.backbone == b) & (per.arm == a)]
                if g.empty:
                    continue
                xs = np.array([curves[r][0] for r in g.run])
                ys = np.array([np.nan_to_num(curves[r][1], nan=A.LIM[t][0]) for r in g.run])
                col, ls, mk = STYLE[a]
                h, = ax.plot(xs.mean(0), ys.mean(0), ls=ls, color=col, lw=1.8 if a == 'ACC' else 1.2, marker=mk,
                             ms=3.2, zorder=5 if a == 'ACC' else 3)
                handles[a] = h
                if a != 'BASE':
                    ax.errorbar(xs.mean(0)[-1], ys.mean(0)[-1], yerr=ys.std(0, ddof=1)[-1], color=col, lw=0.9, capsize=2)
            ax.set_xlim(0, 105)
            ax.set_title(f'{C.BACKBONE_LABEL[b]}, ' + ('heat capacity' if t == 'cp' else 'bulk modulus'), loc='left')
            if i == 1:
                ax.set_xlabel('cumulative oracle calls dispatched')
            if j == 0:
                ax.set_ylabel('running best ' + C.TARGET_LABEL[t])
            ax.grid(alpha=0.25, lw=0.5)
    fig.legend([handles[a] for a in ARMS if a in handles], [LABEL[a] for a in ARMS if a in handles],
               loc='outside lower center', ncol=6, frameon=False)
    C.save_fig(fig, 'fig_rev_value_cost.png')


if __name__ == '__main__':
    main()
