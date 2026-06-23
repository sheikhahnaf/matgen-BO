#!/usr/bin/env python
"""FAITHFUL causal GP replay from the real per-run LTM parquets (isolated-subprocess mode).

Fixes two flaws of the csv-based three_way_causal.py (kept in legacy/ for the record):
  1. SEED OMISSION: the workflow GP was warm-started on a ~500-structure seed pool
     (cycle_id = -1, full property range). The csv replay trained per-run on only the run's
     generated structures -> cold -> underpredicted the extreme winners. Here the GP trains
     on seed + everything accumulated at cycles < s, using the stored Z_pca50 + labels.
  2. BASE RUNS HAVE NO GP: only ACC runs used a GP gate / have an LTM parquet. We restrict
     to ACC runs (the csv replay fabricated predictions for BASE runs where no GP ran).

Tractability: each per-cycle GP refit on the full 500-seed is ~30 s, and an in-process pool
deadlocks on macOS (torch+spawn). So we (a) run each run as an ISOLATED subprocess (--one),
and (b) subsample the warm-start seed to N_SEED=200 (fixed rng). 200 diverse seed structures
spanning the full range is still a genuine warm start; the warm-vs-cold correction is robust
to seed size (200 vs 500). n_train_accum EXCLUDES the seed (plot axis); n_train_total includes
the kept seed.

Modes:
  python three_way_causal_seeded.py --one <parquet> <bm|cp>   # one run -> part csv
  python three_way_causal_seeded.py --agg <bm|cp>             # aggregate parts + stats + table
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, glob, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HCAP = os.environ.get("HCAP_ROOT", "/Volumes/SSD1_SMAAA/matinvent-hcap-bo/hcap_bo")
DATA_ROOT = os.environ.get("DATA_ROOT", "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation")
OUT_ROOT = os.environ.get("OUT_ROOT", "/Volumes/SSD1_SMAAA/matinvent-bo/three_way_comparison")
K0DIR = os.environ.get("K0_DIR", DATA_ROOT + "/results/faster")
N_SEED = int(os.environ.get("N_SEED", "200"))
PARTS = OUT_ROOT + "/results/seeded_parts"
sys.path.insert(0, HCAP)


def _run_key(path):
    m = re.search(r"(adit|cf|mg)_accel_seed(\d+)", os.path.basename(path))
    return ("%s_accel_seed%s" % (m.group(1), m.group(2))) if m else os.path.basename(path)


def one(path, prop):
    from src.surrogate import HCapSurrogate
    rk = _run_key(path)
    os.makedirs(PARTS, exist_ok=True)
    outp = os.path.join(PARTS, "%s__%s.csv" % (prop, rk))
    d = pd.read_parquet(path)
    cols = ["run", "paradigm", "cycle_id", "n_train_accum", "n_train_total", "gp_pred", "gp_sigma", "esen", "structure_id", "formula"]
    if not {"cycle_id", "Z_pca50", "y_cp"}.issubset(d.columns):
        pd.DataFrame(columns=cols).to_csv(outp, index=False); print("  %s: no cols" % rk, flush=True); return
    d = d.dropna(subset=["y_cp"]).reset_index(drop=True)
    Z = np.vstack(d["Z_pca50"].apply(np.asarray).values).astype(float)
    y = d["y_cp"].values.astype(float)
    cyc = d["cycle_id"].values.astype(int)
    seed_idx = np.where(cyc < 0)[0]
    if len(seed_idx) > N_SEED:                       # subsample warm start (fixed rng)
        keep = set(np.random.default_rng(0).choice(seed_idx, N_SEED, replace=False).tolist())
    else:
        keep = set(seed_idx.tolist())
    out = []
    for s in sorted(int(c) for c in np.unique(cyc) if c >= 0):
        tr = np.array([i for i in range(len(cyc)) if (cyc[i] < 0 and i in keep) or (0 <= cyc[i] < s)])
        cu = np.where(cyc == s)[0]
        if len(tr) < 5 or len(cu) == 0:
            continue
        g = HCapSurrogate(device="cpu"); g.fit(Z[tr], y[tr])
        mu, sig = g.predict(Z[cu])
        n_accum = int(np.sum(cyc[tr] >= 0))          # EXCLUDES seed -> plotting axis
        for j, i in enumerate(cu):
            out.append(dict(run=rk, paradigm=rk.split("_")[0], cycle_id=int(s),
                            n_train_accum=n_accum, n_train_total=int(len(tr)),
                            gp_pred=float(mu[j]), gp_sigma=float(sig[j]), esen=float(y[i]),
                            structure_id=str(d["structure_id"].iloc[i]), formula=str(d["formula"].iloc[i])))
    pd.DataFrame(out, columns=cols).to_csv(outp, index=False)
    print("  done %-26s rows=%d (N_SEED=%d)" % (rk, len(out), N_SEED), flush=True)


def agg(prop):
    from scipy.stats import spearmanr, pearsonr
    parts = sorted(glob.glob(os.path.join(PARTS, "%s__*.csv" % prop)))
    allp = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    allp = allp[allp["esen"].notna()]
    outall = OUT_ROOT + "/results/gp_causal_allpreds_%s_seeded.csv" % prop
    allp.to_csv(outall, index=False)
    a, b = allp["gp_pred"].values, allp["esen"].values
    units = "K0 (GPa)" if prop == "bm" else "Cv (J/g/K)"
    print("\n=== FAITHFUL causal GP-vs-eSEN (seed-warm, ACC runs) — %s ===" % units, flush=True)
    print("n=%d preds, %d ACC runs, parts=%d" % (len(allp), allp["run"].nunique(), len(parts)), flush=True)
    print("GP vs eSEN: rho=%.3f  r=%.3f  RMSE=%.3f" % (
        spearmanr(a, b).correlation, pearsonr(a, b)[0], ((a - b) ** 2).mean() ** .5), flush=True)
    print("saved -> %s" % outall, flush=True)
    if prop != "bm":
        print("\ncp: DFT leg skipped (no phonon Cv yet).", flush=True); return
    # parquet stores FULL formulas (Mo4N4); leaderboard uses reduced (MoN) -> match on reduced
    from pymatgen.core import Composition
    allp = allp.copy()
    allp["rformula"] = allp["formula"].apply(lambda s: Composition(str(s)).reduced_formula)
    esen = pd.read_csv(HCAP + "/analysis/top_structures/bm/global_top20.csv"); rows = []
    for cif in sorted(glob.glob(DATA_ROOT + "/structures/bm_*.cif")):
        stem = os.path.basename(cif)[:-4]
        m = re.search(r"bm_top(\d+)_(adit|cf|mg)_(ACC|BASE)_seed(\d+)_", stem)
        if not m:
            continue
        rank, para, setup, seed = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        dftj = os.path.join(K0DIR, "eos_" + stem, "K0.json")
        if setup != "ACC" or not os.path.exists(dftj) or rank > len(esen):
            continue
        rk = "%s_accel_seed%s" % (para, seed); formula = esen.iloc[rank - 1]["reduced_formula"]
        sub = allp[(allp["run"] == rk) & (allp["rformula"] == formula)]
        if sub.empty:
            continue
        f = sub.sort_values("cycle_id").iloc[0]
        rows.append(dict(formula=formula, rank=rank, run=rk, cycle_id=int(f["cycle_id"]),
                         n_train_accum=int(f["n_train_accum"]), n_train_total=int(f["n_train_total"]),
                         dft_K0=json.load(open(dftj))["B0_GPa"], esen_K0=float(esen.iloc[rank - 1]["value"]),
                         gp_causal_K0=float(f["gp_pred"]), gp_sigma=float(f["gp_sigma"])))
    if rows:
        df = pd.DataFrame(rows).sort_values("esen_K0", ascending=False)
        df.to_csv(OUT_ROOT + "/results/three_way_causal_bm_seeded.csv", index=False)
        def stt(x, yv, t):
            x = np.asarray(x, float); yv = np.asarray(yv, float); e = x - yv
            return "%-20s n=%d RMSE=%.1f MAE=%.1f bias=%+.1f rho=%.3f" % (
                t, len(x), (e ** 2).mean() ** .5, np.abs(e).mean(), e.mean(), spearmanr(x, yv).correlation)
        print("\n=== FAITHFUL DFT vs eSEN vs GP(causal) — ACC-run validated winners (%d) ===" % len(df), flush=True)
        print(df[["formula", "rank", "cycle_id", "n_train_accum", "dft_K0", "esen_K0", "gp_causal_K0", "gp_sigma"]].round(1).to_string(index=False), flush=True)
        print(stt(df["gp_causal_K0"], df["dft_K0"], "GP(causal) vs DFT"), flush=True)
        print(stt(df["gp_causal_K0"], df["esen_K0"], "GP(causal) vs eSEN"), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "--agg":
        agg(sys.argv[2])
    else:
        raise SystemExit("usage: --one <parquet> <bm|cp>  |  --agg <bm|cp>")
