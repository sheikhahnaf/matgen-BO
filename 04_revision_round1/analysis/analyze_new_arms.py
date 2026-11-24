#!/usr/bin/env python
"""Closed-loop arms of the revision next to the published BASE and ACC runs.

Same conventions as the main-text cost figures
(02_closed_loop_bo/figures/generators/dispatched_regen/make_value_cost_dispatched.py):
running best from the rewards in samples/long_term_memory.csv, forward-filled
over 20 cycles, and oracle cost counted as dispatched calls (gate log for gated
arms, the `cost` column of metrics.csv for ungated arms).

Reads   02_closed_loop_bo/results/results-paper-v4   (published runs)
        04_revision_round1/closed_loop/results         (revision runs, curated files)
Writes  analysis/out/new_arms_per_run.csv, new_arms_summary.csv,
        figures/fig_new_arms_<target>.png
Only runs that completed all 20 cycles are used.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import rev_common as C

HORIZON = 20
LIM = {'cp': (0.25, 2.0), 'bm': (20.0, 400.0)}
PUB = C.REPO / '02_closed_loop_bo' / 'results' / 'results-paper-v4'
NEW = C.REV / 'closed_loop' / 'results'
GATED = {'ACC', 'frozen_gate', 'ei_only', 'div_only'}
ARM_LABEL = {'BASE': 'BASE', 'ACC': 'ACC (gate)', 'cap4': 'cap-4', 'oracle_all': 'oracle-all',
             'frozen_gate': 'gate, frozen generator', 'ei_only': 'EI only', 'div_only': 'diversity only'}
ARM_STYLE = {'BASE': ('#7f7f7f', '--', 's'), 'ACC': ('#1f77b4', '-', 'o'), 'cap4': ('#2ca02c', '-.', '^'),
             'oracle_all': ('#9467bd', ':', 'D'), 'frozen_gate': ('#ff7f0e', '-', 'v'),
             'ei_only': ('#d62728', '-', 'P'), 'div_only': ('#8c564b', '-', 'X')}


def collect():
    runs = []
    pats = [(PUB / 'results', 'cp'), (PUB / 'results_bm', 'bm'), (NEW, None)]
    for root, tgt in pats:
        for d in sorted(glob.glob(str(root / '*'))):
            n = os.path.basename(d)
            m = re.match(r'^(?:hcap|bm)_p3v4(?:_bm)?_(mg|cf|adit)_(accel|baseline)_seed(\d+)_(\d+)$', n)
            if m:
                runs.append(dict(dir=d, target=tgt, backbone=m.group(1), arm={'accel': 'ACC', 'baseline': 'BASE'}[m.group(2)],
                                 seed=int(m.group(3)), job=int(m.group(4)), patched=False)); continue
            m = re.match(r'^hcap_mgabl_(cap4|oracleall)_(cp|bm)_seed(\d+)(_patched)?_(\d+)$', n)
            if m:
                runs.append(dict(dir=d, target=m.group(2), backbone='mg', arm={'cap4': 'cap4', 'oracleall': 'oracle_all'}[m.group(1)],
                                 seed=int(m.group(3)), job=int(m.group(5)), patched=bool(m.group(4)))); continue
            m = re.match(r'^rev1_(cap4|frozen_gate|ei_only|div_only)_(mg|cf|adit)_(cp|bm)_seed(\d+)_(\d+)$', n)
            if m:
                runs.append(dict(dir=d, target=m.group(3), backbone=m.group(2), arm=m.group(1),
                                 seed=int(m.group(4)), job=int(m.group(5)), patched=False))
    return pd.DataFrame(runs)


def run_arrays(r):
    d, tgt = r['dir'], r['target']
    mn, mx = LIM[tgt]
    ltm = pd.read_csv(f'{d}/samples/long_term_memory.csv', usecols=['reward', 'RL_step'])
    lv = ltm[ltm.reward > 0]
    pv = lv.reward * (mx - mn) + mn
    pv = pv.where(lv.reward < 1.0, mx)
    rb = pv.groupby(lv.RL_step.astype(int)).max().reindex(range(HORIZON)).cummax().ffill().values
    if r['arm'] in GATED:
        g = pd.read_csv(glob.glob(f'{d}/rewards/*/*gp_routed_v4_log.csv')[0])
        n_cycles = len(g)
        calls = g.set_index('cycle')['n_oracle'].reindex(range(HORIZON), fill_value=0).cumsum().values.astype(float)
    else:
        m = pd.read_csv(f'{d}/metrics.csv')
        n_cycles = len(m)
        calls = m.set_index('step')['cost'].reindex(range(HORIZON)).ffill().values.astype(float)
    return calls, rb, n_cycles


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    runs = collect()
    rows, curves = [], {}
    for _, r in runs.iterrows():
        try:
            calls, rb, n_cycles = run_arrays(r)
        except Exception as e:
            continue
        if n_cycles < HORIZON:
            continue
        rows.append(dict(target=r.target, backbone=r.backbone, arm=r.arm, seed=r.seed, job=r.job, patched=r.patched,
                         final_best=rb[-1], total_calls=calls[-1], run=os.path.basename(r['dir'])))
        curves[os.path.basename(r['dir'])] = (calls, rb)
    per = pd.DataFrame(rows)
    # one run per (target, backbone, arm, seed): prefer a patched re-run, then the latest job
    per = (per.sort_values(['patched', 'job']).groupby(['target', 'backbone', 'arm', 'seed']).tail(1)
           .sort_values(['target', 'backbone', 'arm', 'seed']).reset_index(drop=True))
    per.to_csv(C.OUT / 'new_arms_per_run.csv', index=False)

    summ = []
    for (t, b, a), g in per.groupby(['target', 'backbone', 'arm']):
        fb = np.nan_to_num(g.final_best.to_numpy(), nan=LIM[t][0])
        ref = per[(per.target == t) & (per.backbone == b) & (per.arm == 'ACC')].final_best
        ref = np.nan_to_num(ref.to_numpy(), nan=LIM[t][0])
        p = mannwhitneyu(fb, ref, alternative='two-sided').pvalue if (a != 'ACC' and len(ref) and len(fb)) else np.nan
        summ.append(dict(target=t, backbone=b, arm=a, n=len(g), final_best_mean=fb.mean(), final_best_sd=fb.std(ddof=0),
                         calls_mean=g.total_calls.mean(), p_vs_ACC=p))
    summ = pd.DataFrame(summ)
    summ.to_csv(C.OUT / 'new_arms_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(summ.round(3).to_string(index=False))

    plt = C.set_style()
    for t in ('cp', 'bm'):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
        for ax, b in zip(axes, ('mg', 'cf', 'adit')):
            for a in ('BASE', 'ACC', 'cap4', 'frozen_gate', 'ei_only', 'div_only'):
                g = per[(per.target == t) & (per.backbone == b) & (per.arm == a)]
                if g.empty:
                    continue
                xs = np.array([curves[r][0] for r in g.run]); ys = np.array([np.nan_to_num(curves[r][1], nan=LIM[t][0]) for r in g.run])
                col, ls, mk = ARM_STYLE[a]
                ax.plot(xs.mean(0), ys.mean(0), ls=ls, color=col, lw=1.4, marker=mk, ms=3, label=f'{ARM_LABEL[a]} (n={len(g)})')
                ax.errorbar(xs.mean(0)[-1], ys.mean(0)[-1], xerr=xs.std(0)[-1], yerr=ys.std(0)[-1], color=col, lw=0.9, capsize=2)
            ax.set_title(C.BACKBONE_LABEL[b], loc='left', fontweight='bold')
            ax.set_xlabel('cumulative oracle calls dispatched')
            ax.set_ylabel('running best ' + C.TARGET_LABEL[t])
            ax.legend(frameon=False, fontsize=7.5, loc='lower right')
        C.save_fig(fig, f'fig_new_arms_{t}.png')


if __name__ == '__main__':
    main()
