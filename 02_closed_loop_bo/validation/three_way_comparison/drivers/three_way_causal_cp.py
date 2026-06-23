#!/usr/bin/env python
"""Causal (per-cycle) GP holdout — HEAT CAPACITY (Cv) version.

Cp twin of three_way_causal.py (which is the bm/K0 version and is left untouched as the
verified BM record). DIFFERENCES: reads the cp closed-loop runs (results_cp/hcap_p3v4_*) and
the cp leaderboard, target = Cv (reward*2.0 J/g/K), and the DFT leg is OFF — there is no
phonon Cv DFT yet, so this produces only the GP-vs-eSEN causal generalization + the per-step
ranking (gp_causal_allpreds_cp.csv). When the Cp phonon campaign finishes, the DFT-anchored
table can be added the same way the bm version does it.

Why this and not the simple version: the DFT-validated top structures are themselves
members of the closed-loop memory, so training a GP on all memory and then "predicting"
them is in-sample memorization, not a test of the surrogate. The honest question is the
one the workflow itself faced: as the loop runs, does the GP genuinely predict the
property of FRESHLY GENERATED structures it has not seen yet?

We replay exactly that, using the RL_step column the loop logged. Each run is independent
(its own generator + its own GP accumulating its own memory). For each run, sorted by
RL_step:
    at step s -> train the GP ONLY on structures from steps < s, then predict the
    structures generated AT step s (before their eSEN value would have been known).
Those are genuine step-ahead predictions on unseen, freshly-generated structures.

Outputs:
  * gp_causal_allpreds.csv  — every step-ahead prediction across the whole workflow
    (gp_pred vs eSEN), i.e. the real generalization signal over thousands of structures.
  * three_way_causal.csv    — for each DFT-validated top structure, its CAUSAL GP
    prediction (made at the step it first appeared) beside the eSEN oracle value and the
    ground-truth DFT K0.

Representation note: ORB->PCA50 is a FROZEN, unsupervised embedding fit once over all
memory (consistent with the paper's frozen-embedding thesis). Only the GP updates causally
on accumulating LABELS; the descriptor space is fixed, so it carries no K0-label leakage.
"""
import sys, glob, os, re, json, warnings
HCAP = os.environ.get("HCAP_ROOT", "/Volumes/SSD1_SMAAA/matinvent-hcap-bo/hcap_bo")
DATA_ROOT = os.environ.get("DATA_ROOT", "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation")        # inputs: structures + K0
OUT_ROOT = os.environ.get("OUT_ROOT", "/Volumes/SSD1_SMAAA/matinvent-bo/three_way_comparison")    # all three-way outputs
K0DIR = os.environ.get("K0_DIR", DATA_ROOT + "/results/faster")  # dir holding eos_<stem>/K0.json
MIN_TRAIN = int(os.environ.get("MIN_TRAIN", "10"))             # GP needs some history to mean anything
DEVICE = os.environ.get("DEVICE", "cpu")                       # 'cuda' on a GPU node -> ORB in minutes
sys.path.insert(0, HCAP)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from src.featurizer import ORBFeaturizer
from src.surrogate import HCapSurrogate

MAXV = 2.0     # reward*2.0 -> Cv (J/g/K); ranking metrics are scale-invariant (units only)
RB = HCAP + "/results-paper-v4/results_cp"                       # cp closed-loop runs (hcap_p3v4_*)
ESEN = HCAP + "/analysis/top_structures/cp/global_top20.csv"     # cp leaderboard
STRUCT = DATA_ROOT + "/structures"
DFT = K0DIR
OUT = OUT_ROOT + "/results/three_way_causal_cp.csv"
OUTALL = OUT_ROOT + "/results/gp_causal_allpreds_cp.csv"


def atoms(c):
    return AseAtomsAdaptor.get_atoms(Structure.from_str(c, fmt="cif"))


def stats_line(a, b, tag):
    from scipy.stats import pearsonr, spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float); e = a - b
    return ("%-22s n=%d RMSE=%.1f MAE=%.1f MAPE=%.1f%% bias=%+.1f | r=%.3f rho=%.3f"
            % (tag, len(a), (e**2).mean()**.5, np.abs(e).mean(),
               (100*np.abs(e)/np.abs(b)).mean(), e.mean(),
               pearsonr(a, b)[0], spearmanr(a, b).correlation))


def main():
    # ---- DFT leg OFF for cp: no phonon Cv DFT yet, so we do not build DFT targets.
    # This run produces only the GP-vs-eSEN causal generalization + per-step ranking.
    tgt = {}
    print("cp run: DFT leg skipped (no phonon Cv yet) -> GP-vs-eSEN + ranking only", flush=True)

    # ---- gather ALL unique generated structures across runs; freeze ORB->PCA50 once ----
    runs = sorted(glob.glob(RB + "/hcap_p3v4_*"))
    per_run = []
    allcifs = {}
    for run in runs:
        f = os.path.join(run, "samples", "long_term_memory.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        if not {"cif", "reward", "RL_step"}.issubset(d.columns):
            continue
        d = d[d["reward"] > 0].copy()
        d["cif"] = d["cif"].astype(str).str.strip()
        d["y"] = d["reward"] * MAXV
        d = d.drop_duplicates("cif").sort_values("RL_step").reset_index(drop=True)
        per_run.append((os.path.basename(run), d))
        for c in d["cif"]:
            allcifs.setdefault(c, len(allcifs))
    cif_list = list(allcifs.keys())
    print("runs=%d  unique generated structures=%d  -> featurizing (frozen ORB+PCA50) ..."
          % (len(per_run), len(cif_list)), flush=True)

    feat = ORBFeaturizer(n_components=50, device=DEVICE)        # ORB on GPU if DEVICE=cuda
    Zall = feat.fit_transform([atoms(c) for c in cif_list])    # frozen representation, fit once
    Z = {c: Zall[i] for i, c in enumerate(cif_list)}
    print("frozen embedding ready: %s" % (Zall.shape,), flush=True)

    # ---- per-run causal walk: train on steps<s, predict step==s ----
    all_rows, tgt_hits = [], {}
    for ri, (run_name, d) in enumerate(per_run):
        steps = sorted(d["RL_step"].unique())
        seen = []                                   # row indices from earlier steps
        for s in steps:
            cur = d[d["RL_step"] == s]
            if len(seen) >= MIN_TRAIN:
                Xtr = np.vstack([Z[c] for c in d["cif"].values[seen]])
                ytr = d["y"].values[seen].astype(float)
                g = HCapSurrogate(device="cpu"); g.fit(Xtr, ytr)
                Xq = np.vstack([Z[c] for c in cur["cif"].values])
                mu, sig = g.predict(Xq)
                for j, c in enumerate(cur["cif"].values):
                    rec = dict(run=run_name, RL_step=int(s), n_train=len(seen),
                               gp_pred_K0=float(mu[j]), gp_sigma=float(sig[j]),
                               esen_K0=float(d.loc[d["cif"] == c, "y"].iloc[0]))
                    all_rows.append({**rec, "cif": c})
                    if c in tgt and c not in tgt_hits:
                        tgt_hits[c] = dict(stem=tgt[c]["stem"], rank=tgt[c]["rank"],
                                           formula=tgt[c]["formula"], run=run_name, RL_step=int(s),
                                           n_train=len(seen), dft_K0=tgt[c]["dft_K0"],
                                           esen_K0=tgt[c]["esen_K0"], gp_causal_K0=float(mu[j]),
                                           gp_sigma=float(sig[j]))
            seen.extend(cur.index.tolist())
        print("  [%d/%d] %-42s structs=%-4d steps=%-3d" % (ri + 1, len(per_run), run_name, len(d), len(steps)), flush=True)

    # ---- (a) genuine generalization across the whole workflow ----
    allp = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTALL), exist_ok=True)
    allp.to_csv(OUTALL, index=False)
    print("\n=== GENUINE causal generalization (GP step-ahead prediction on UNSEEN generated structures) ===", flush=True)
    print("n=%d predictions across %d runs, n_train>=%d  ->  %s" % (len(allp), len(per_run), MIN_TRAIN, OUTALL), flush=True)
    print(stats_line(allp["gp_pred_K0"], allp["esen_K0"], "GP(causal) vs eSEN"), flush=True)

    # ---- (b) three-way for the DFT-validated structures, using their CAUSAL prediction ----
    rows = list(tgt_hits.values())
    miss = [tgt[c]["stem"] for c in tgt if c not in tgt_hits]
    if not rows:
        print("\n!! no validated target received a causal prediction (all appeared before n_train>=%d)" % MIN_TRAIN, flush=True)
        return
    df = pd.DataFrame(rows).sort_values("esen_K0", ascending=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n=== DFT vs eSEN vs GP(causal) for validated structures — %d matched%s ===" % (
        len(df), ("" if not miss else ", %d appeared too early: %s" % (len(miss), miss))), flush=True)
    print(df[["formula", "rank", "RL_step", "n_train", "dft_K0", "esen_K0", "gp_causal_K0", "gp_sigma"]].round(1).to_string(index=False), flush=True)
    print("\n-- ground-truth anchored --", flush=True)
    print(stats_line(df["esen_K0"], df["dft_K0"], "eSEN vs DFT"), flush=True)
    print(stats_line(df["gp_causal_K0"], df["dft_K0"], "GP(causal) vs DFT"), flush=True)
    print("\n-- surrogate vs oracle --", flush=True)
    print(stats_line(df["gp_causal_K0"], df["esen_K0"], "GP(causal) vs eSEN"), flush=True)
    print("\nsaved -> %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
