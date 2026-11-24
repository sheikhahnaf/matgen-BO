#!/usr/bin/env python
"""Build one table of every candidate that reached the property calculator in
the published closed-loop runs, with its outcome.

For each run and RL cycle the pipeline wrote `samples/step_XXXX_eval.extxyz`
(the post-SUN candidates handed to the reward calculator, in order) and
`samples/long_term_memory.csv` (the candidates that came back with a value).

  Runs without the GP gate (BASE, cap-4, oracle-all): every candidate in the
  eval file was sent to the oracle. A candidate missing from the long-term
  memory for that cycle is an oracle failure. The reason is taken from the
  job's stderr log, where the oracle prints exactly one line per failed
  candidate, in dispatch order.

  Gated runs (ACC): the eval file holds the candidates the gate scored. The
  ones it sent to the oracle and that succeeded are in the gate's own memory
  (parquet, exact property values). Failed gate picks are not recorded per
  candidate; their number per cycle is in the gate log.

The script also stores the mean-pooled ORB-v3 node embedding (before PCA) of
every candidate and of the GP seed pool, so the surrogate analyses can be
repeated on a CPU.

Outputs in --out:
    candidates_<target>.parquet     one row per candidate
    orb_raw_<target>.npy            (n_candidates, 256) float32, row-aligned
    seed_pool_<target>.parquet      GP seed pool (atoms_json, y)
    orb_raw_seed_<target>.npy       row-aligned with the seed pool
    run_summary_<target>.csv        one row per run (counts, log cross-checks)
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from ase.io import read as ase_read

P3 = re.compile(r'^(?:hcap|bm)_p3v4(?:_bm)?_(mg|cf|adit)_(accel|baseline)_seed(\d+)_(\d+)$')
ABL = re.compile(r'^hcap_mgabl_(cap4ext|cap4|oracleall)_(?:cp|bm)_seed(\d+)(_patched)?_(\d+)$')
REV = re.compile(r'^rev1_(cap4|frozen_gate|ei_only|div_only)_(mg|cf|adit)_(?:cp|bm)_seed(\d+)_(\d+)$')
ARM = {'accel': 'ACC', 'baseline': 'BASE', 'cap4': 'cap4', 'cap4ext': 'cap4ext',
       'oracleall': 'oracle_all', 'frozen_gate': 'frozen_gate', 'ei_only': 'ei_only', 'div_only': 'div_only'}
GATED = {'ACC', 'frozen_gate', 'ei_only', 'div_only'}

FAIL_LINE = re.compile(r'^\[LocalESEN(?:_BM)?\] (.*)$')


def parse_run(name):
    m = P3.match(name)
    if m:
        return dict(backbone=m.group(1), arm=ARM[m.group(2)], seed=int(m.group(3)),
                    jobid=int(m.group(4)), patched=False)
    m = ABL.match(name)
    if m:
        return dict(backbone='mg', arm=ARM[m.group(1)], seed=int(m.group(2)),
                    jobid=int(m.group(4)), patched=bool(m.group(3)))
    m = REV.match(name)
    if m:
        return dict(backbone=m.group(2), arm=ARM[m.group(1)], seed=int(m.group(3)),
                    jobid=int(m.group(4)), patched=True)
    return None


def select_runs(names):
    """One run per (backbone, arm, seed): prefer a patched re-run, then the latest job id."""
    best = {}
    for n in names:
        info = parse_run(n)
        if info is None:
            continue
        key = (info['backbone'], info['arm'], info['seed'])
        rank = (info['patched'], info['jobid'])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, n, info)
    return [(n, info) for _, n, info in sorted(best.values(), key=lambda t: t[1])]


def comp_key(symbols):
    return tuple(sorted(Counter(symbols).items()))


def classify(msg):
    if 'imaginary modes' in msg:
        return 'imaginary_modes'
    if msg.startswith('relax failed'):
        return 'relax_failed'
    if 'energy failed' in msg:
        return 'strain_energy_failed'
    if msg.startswith('EOS fit failed'):
        return 'eos_fit_failed'
    if msg.startswith('unphysical K'):
        return 'unphysical_K'
    m = re.match(r'task \d+ failed: (\w+)', msg)
    if m:
        return 'exception_' + m.group(1)
    return 'other'


def align(frames_keys, items_keys, rtol=2e-3):
    """Match items (a subsequence of the frames, same order) to frame indices.

    Keys are (composition, volume). Returns a list of frame indices, one per
    item, or None if the sequential walk does not consume every item.
    """
    out, j = [], 0
    for i, (ck, vol) in enumerate(frames_keys):
        if j >= len(items_keys):
            break
        ick, ivol = items_keys[j]
        if ck == ick and abs(vol - ivol) <= rtol * max(vol, 1e-9):
            out.append(i)
            j += 1
    return out if j == len(items_keys) else None


def descriptors(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    n = len(atoms)
    masses = atoms.get_masses()
    vol = float(atoms.get_volume())
    try:
        sg = int(SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms), symprec=0.1)
                 .get_space_group_number())
    except Exception:
        sg = -1
    e = atoms.info.get('total_energy')
    return dict(
        formula=atoms.get_chemical_formula(),
        n_atoms=n,
        n_elements=len(set(atoms.get_chemical_symbols())),
        mean_mass=float(masses.mean()),
        vol_per_atom=vol / n,
        density=float(masses.sum()) / vol * 1.66053907,
        spacegroup=sg,
        mlip_energy_per_atom=float(e) / n if e is not None else np.nan,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['cp', 'bm'], required=True)
    ap.add_argument('--proj', required=True, help='closed-loop project root (configs, src, data, logs)')
    ap.add_argument('--results', nargs='+', required=True, help='directories holding run folders')
    ap.add_argument('--logs', nargs='+', required=True, help='directories holding <job>.err logs')
    ap.add_argument('--ltm', nargs='+', required=True, help='directories holding ltm_<run>.parquet')
    ap.add_argument('--seed-pool', required=True)
    ap.add_argument('--run-list', default=None, help='optional text file of run folder names to keep')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-orb', action='store_true')
    args = ap.parse_args()

    sys.path.insert(0, args.proj)
    from omegaconf import OmegaConf
    from pymatgen.core import Structure
    from src.ltm import atoms_from_json, atoms_to_json

    os.makedirs(args.out, exist_ok=True)
    keep = None
    if args.run_list:
        keep = {l.strip() for l in open(args.run_list) if l.strip()}

    run_dirs = {}
    for root in args.results:
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            name = os.path.basename(d)
            if keep is not None and name not in keep:
                continue
            if os.path.isfile(os.path.join(d, 'samples', 'long_term_memory.csv')):
                run_dirs[name] = d
    runs = select_runs(run_dirs)
    print(f'{len(runs)} runs selected for target={args.target}', flush=True)

    rows, atoms_all, summaries = [], [], []
    for name, info in runs:
        d = run_dirs[name]
        cfg = OmegaConf.load(os.path.join(d, '.hydra', 'config.yaml'))
        pc = cfg.reward.prop_cfg[0]
        minv, maxv = float(pc.get('minv', 0.0)), float(pc.maxv)
        gated = info['arm'] in GATED

        ltm = pd.read_csv(os.path.join(d, 'samples', 'long_term_memory.csv'))
        ltm['RL_step'] = ltm['RL_step'].astype(int)

        gate_mem = None
        if gated:
            for root in args.ltm:
                p = os.path.join(root, f'ltm_{name}.parquet')
                if os.path.isfile(p):
                    gate_mem = pd.read_parquet(p)
                    gate_mem = gate_mem[gate_mem['cycle_id'] >= 0]
                    break

        fail_msgs = []
        for root in args.logs:
            for p in glob.glob(os.path.join(root, f'*_{info["jobid"]}.err')):
                with open(p, errors='replace') as f:
                    for line in f:
                        m = FAIL_LINE.match(line.strip())
                        if m:
                            fail_msgs.append(classify(m.group(1)))

        run_rows, align_bad = [], 0
        step_files = sorted(glob.glob(os.path.join(d, 'samples', 'step_*_eval.extxyz')))
        for sf in step_files:
            step = int(re.search(r'step_(\d+)_eval', sf).group(1))
            frames = ase_read(sf, index=':')
            if not isinstance(frames, list):
                frames = [frames]
            fkeys = [(comp_key(a.get_chemical_symbols()), float(a.get_volume())) for a in frames]

            # successes recorded by the pipeline (reward, possibly clipped to [0, 1])
            sub = ltm[ltm['RL_step'] == step]
            ikeys = []
            for cif in sub['cif']:
                s = Structure.from_str(cif, fmt='cif')
                ikeys.append((comp_key([str(sp) for sp in s.species]), float(s.volume)))
            idx = align(fkeys, ikeys)
            if idx is None:
                align_bad += 1
                idx = []
            reward = {i: float(r) for i, r in zip(idx, sub['reward'].tolist())}

            # exact values from the gate memory (gated runs only)
            exact = {}
            if gate_mem is not None:
                gsub = gate_mem[gate_mem['cycle_id'] == step]
                gkeys = []
                for aj in gsub['atoms_json']:
                    a = atoms_from_json(aj)
                    gkeys.append((comp_key(a.get_chemical_symbols()), float(a.get_volume())))
                gidx = align(fkeys, gkeys)
                if gidx is not None:
                    exact = {i: float(y) for i, y in zip(gidx, gsub['y_cp'].tolist())}

            for i, a in enumerate(frames):
                ok = i in reward
                r = reward.get(i, np.nan)
                y_reward = minv + r * (maxv - minv) if ok else np.nan
                row = dict(
                    target=args.target, run=name, backbone=info['backbone'], arm=info['arm'],
                    seed=info['seed'], step=step, cand_idx=i, gated=gated,
                    # without a gate every candidate is dispatched; with a gate only
                    # the successful picks are identified per candidate
                    dispatched=(True if not gated else (True if ok else None)),
                    success=(ok if not gated else (True if ok else None)),
                    reward=r,
                    y=exact.get(i, y_reward),
                    y_exact=(i in exact) or (ok and 0.0 < r < 1.0),
                    censored_low=bool(ok and r <= 0.0 and i not in exact),
                    censored_high=bool(ok and r >= 1.0 and i not in exact),
                    fail_reason=None,
                )
                row.update(descriptors(a))
                row['atoms_json'] = atoms_to_json(a)
                run_rows.append(row)
                atoms_all.append(a)

        # failure reasons: one stderr line per failed candidate, in dispatch order
        n_fail = sum(1 for r in run_rows if r['success'] is False)
        reason_ok = False
        if not gated and n_fail == len(fail_msgs):
            it = iter(fail_msgs)
            for r in run_rows:
                if r['success'] is False:
                    r['fail_reason'] = next(it)
            reason_ok = True
        elif not gated:
            for r in run_rows:
                if r['success'] is False:
                    r['fail_reason'] = 'log_count_mismatch'

        gate_log = {}
        for p in glob.glob(os.path.join(d, 'rewards', '*', '*gp_routed_v4_log.csv')):
            g = pd.read_csv(p)
            gate_log = dict(gate_n_input=int(g['n_input'].sum()), gate_n_oracle=int(g['n_oracle'].sum()))

        summ = dict(run=name, **{k: info[k] for k in ('backbone', 'arm', 'seed', 'jobid')},
                    n_cycles=len(step_files), n_candidates=len(run_rows),
                    n_success=sum(1 for r in run_rows if r['success'] is True),
                    n_fail_table=n_fail, n_fail_log=len(fail_msgs), reason_assigned=reason_ok,
                    steps_align_failed=align_bad, minv=minv, maxv=maxv,
                    **{f'log_{k}': v for k, v in Counter(fail_msgs).items()}, **gate_log)
        summaries.append(summ)
        rows.extend(run_rows)
        print(f"  {name}: cand={summ['n_candidates']} ok={summ['n_success']} "
              f"fail_table={n_fail} fail_log={len(fail_msgs)} align_bad={align_bad}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out, f'candidates_{args.target}.parquet'), index=False)
    pd.DataFrame(summaries).to_csv(os.path.join(args.out, f'run_summary_{args.target}.csv'), index=False)
    print(f'{len(df)} candidates written', flush=True)

    seed = pd.read_parquet(args.seed_pool)
    seed = seed[['structure_id', 'formula', 'atoms_json', 'y_cp', 'oracle_source']].rename(columns={'y_cp': 'y'})
    seed.to_parquet(os.path.join(args.out, f'seed_pool_{args.target}.parquet'), index=False)
    seed_atoms = [atoms_from_json(aj) for aj in seed['atoms_json']]

    if not args.no_orb:
        from src.featurizer import ORBFeaturizer
        feat = ORBFeaturizer(n_components=0, device=args.device)
        for tag, atoms_list in ((f'orb_raw_seed_{args.target}', seed_atoms),
                                (f'orb_raw_{args.target}', atoms_all)):
            chunks, bad = [], 0
            for k in range(0, len(atoms_list), 500):
                x, mask = feat._featurize_raw(atoms_list[k:k + 500])
                x[mask] = np.nan
                bad += int(mask.sum())
                chunks.append(x.astype(np.float32))
                print(f'  {tag}: {min(k + 500, len(atoms_list))}/{len(atoms_list)}', flush=True)
            np.save(os.path.join(args.out, tag + '.npy'), np.concatenate(chunks))
            print(f'{tag}: {bad} structures could not be embedded', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
