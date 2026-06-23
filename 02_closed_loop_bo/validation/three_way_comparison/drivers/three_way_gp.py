#!/usr/bin/env python
"""DFT vs eSEN vs GP-pred three-way for the DFT-validated bm structures.

eSEN K0 = the closed-loop oracle 'value' (bm/global_top20.csv). DFT K0 = our campaign (K0.json).
GP-pred = reconstructed faithfully: the paper's exact surrogate (ORBFeaturizer orb_v3 mean-pool ->
PCA50 -> BoTorch SingleTaskGP) trained on the pooled bm closed-loop memory (target K0 ~= reward*maxv,
maxv=400), then predicting each top structure. The GP's per-structure predictions were not logged in
the runs, so this re-creates what the surrogate would say about these structures from ORB features.
"""
import sys, glob, os, re, json, warnings
# Roots are env-overridable so the SAME script runs locally (macOS) or on a cluster.
# NOTE: this is a CPU Torch/ORB job; on macOS it deadlocks in an OpenMP barrier
# (dual libomp/libgomp), so the real home is FASTER (Linux + the matinvent-hcap-bo env).
# On FASTER export HCAP_ROOT/DFT_ROOT/K0_DIR (see drivers/three_way.slurm).
HCAP = os.environ.get("HCAP_ROOT", "/Volumes/SSD1_SMAAA/matinvent-hcap-bo/hcap_bo")
DFT_ROOT = os.environ.get("DFT_ROOT", "/Volumes/SSD1_SMAAA/matinvent-bo/dft_validation")
K0DIR = os.environ.get("K0_DIR", DFT_ROOT + "/results/faster")  # dir holding eos_<stem>/K0.json
sys.path.insert(0, HCAP)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from src.featurizer import ORBFeaturizer
from src.surrogate import HCapSurrogate

MAXV = 400.0
RB = HCAP + "/results-paper-v4/results_bm"
ESEN = HCAP + "/analysis/top_structures/bm/global_top20.csv"
STRUCT = DFT_ROOT + "/structures"
DFT = K0DIR
OUT = DFT_ROOT + "/results/three_way_gp.csv"

def cif2atoms(c):
    return AseAtomsAdaptor.get_atoms(Structure.from_str(c, fmt="cif"))

def main():
    # 1. pool bm closed-loop memory (the production runs that made the top structures)
    mems = []
    for f in glob.glob(RB + "/bm_p3v4_bm_*/samples/long_term_memory.csv"):
        d = pd.read_csv(f)
        if "cif" in d and "reward" in d:
            d = d[d["reward"] > 0].copy(); d["y"] = d["reward"] * MAXV
            mems.append(d[["cif", "y"]])
    mem = pd.concat(mems).drop_duplicates("cif").reset_index(drop=True)
    mem = mem.sample(n=min(280, len(mem)), random_state=0).reset_index(drop=True)
    print("pooled bm memory used for GP training: %d structures" % len(mem), flush=True)

    # 2. featurize memory with the paper's exact ORB featurizer (fits PCA50)
    feat = ORBFeaturizer(n_components=50, device="cpu")
    atoms_tr = [cif2atoms(c) for c in mem["cif"]]
    Ztr = feat.fit_transform(atoms_tr)
    print("ORB+PCA50 features:", Ztr.shape, flush=True)

    # 3. fit the paper's GP surrogate (target = K0 in GPa)
    sur = HCapSurrogate(device="cpu"); sur.fit(Ztr, mem["y"].values.astype(float))

    # 4. the DFT-validated bm structures: eSEN value (by global rank) + DFT K0 + GP-pred
    esen = pd.read_csv(ESEN)
    rows = []
    for cif in sorted(glob.glob(STRUCT + "/bm_*.cif")):
        stem = os.path.basename(cif)[:-4]
        m = re.search(r"bm_top(\d+)_", stem)
        rank = int(m.group(1))
        dftj = os.path.join(DFT, "eos_" + stem, "K0.json")
        if not os.path.exists(dftj) or rank > len(esen):
            continue
        dft_k0 = json.load(open(dftj))["B0_GPa"]
        esen_k0 = float(esen.iloc[rank - 1]["value"])
        formula = esen.iloc[rank - 1]["reduced_formula"]
        Zq = feat.transform([cif2atoms(open(cif).read())])
        gp_mu, gp_sig = sur.predict(Zq)
        rows.append(dict(formula=formula, rank=rank, dft_K0=dft_k0, esen_K0=esen_k0,
                         gp_K0=float(gp_mu[0]), gp_sigma=float(gp_sig[0])))
    df = pd.DataFrame(rows).sort_values("esen_K0", ascending=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    from scipy.stats import pearsonr, spearmanr
    def stats(a, b, na, nb):
        e = a - b
        return ("%s-vs-%s: MAE=%.1f RMSE=%.1f MAPE=%.1f%% bias=%+.1f | r=%.3f rho=%.3f"
                % (na, nb, np.abs(e).mean(), (e**2).mean()**.5, (100*np.abs(e)/b).mean(),
                   e.mean(), pearsonr(a, b)[0], spearmanr(a, b).correlation))
    print("\n=== DFT vs eSEN vs GP-pred (K0, GPa) — %d structures ===" % len(df), flush=True)
    print(df.round(1).to_string(index=False), flush=True)
    print("\n" + stats(df["esen_K0"].values, df["dft_K0"].values, "eSEN", "DFT"), flush=True)
    print(stats(df["gp_K0"].values, df["dft_K0"].values, "GP", "DFT"), flush=True)
    print(stats(df["gp_K0"].values, df["esen_K0"].values, "GP", "eSEN"), flush=True)
    print("\nsaved -> %s" % OUT, flush=True)

if __name__ == "__main__":
    main()
