"""MatterGen 4-arm ablation plots — recovered 2026-06-09 from the 2026-05-28 session transcript.
The script originally lived only in /tmp (local + Grace) and was lost; this file is the exact
reconstruction: v2 ("Option A") + the three in-session patches, in order.
Run on Grace:
  PROJ=$SCRATCH/matinvent-hcap-bo OUT=$SCRATCH/matinvent-hcap-bo/analysis/mg_ablation_plots \
  $SCRATCH/envs/matinvent-hcap-bo/bin/python mg_ablation_plots.py
Generates fig_mg_ablation_discovery.png / fig_mg_ablation_oracle_cost.png used in the FME paper.
Option A: oracle-all on Cp ≡ BASE (cap never binds)."""
import os, glob, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os as _os
from pathlib import Path as _Path
# Repo-relative defaults; override with MBO_REPO_ROOT / MBO_FIG_DIR / MBO_RESULTS_ROOT.
_REPO = _Path(_os.environ.get("MBO_REPO_ROOT", _Path(__file__).resolve().parent.parent))
_FIGS = _Path(_os.environ.get("MBO_FIG_DIR", _REPO / "figures"))
_RES = _Path(_os.environ.get("MBO_RESULTS_ROOT", _REPO / "hcap_bo" / "results-paper-v4"))

PROJ = Path(os.environ.get("PROJ", "")) if os.environ.get("PROJ") else _RES
HORIZON = 20
OUT = Path(os.environ.get("OUT", "")) if os.environ.get("OUT") else _FIGS
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":"Helvetica","font.size":10.5,
    "axes.labelsize":11,"axes.titlesize":11.5,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,
    "legend.fontsize":8.8,"legend.frameon":False,
    "axes.spines.right":False,"axes.spines.top":False,
    "savefig.dpi":300,"savefig.bbox":"tight",
})

COLOR_ACC, COLOR_BASE, COLOR_CAP, COLOR_ORALL = "#1f77b4","#ff7f0e","#2ca02c","#9467bd"

# tuples: (display_name, glob_pattern, exclude_regex, color, marker, linestyle)
ARMS = {
 "Cp": [
  ("ACC",                       PROJ/"results"/"hcap_p3v4_mg_accel_seed*",     None,                 COLOR_ACC, "o", "-"),
  ("BASE",                      PROJ/"results"/"hcap_p3v4_mg_baseline_seed*",  None,                 COLOR_BASE,"s","--"),
  ("cap-4 (ours)",              PROJ/"results"/"hcap_mgabl_cap4_cp_seed*",     "_18636280|_18636282",COLOR_CAP, "^", "-"),
  # oracle-all on Cp coincides with BASE because the SUN cap=16 never binds for MatterGen Cp
  # (survival ~9 < 16). We plot BASE's data here, styled as oracle-all, with the explicit label.
  (r"oracle-all (≡ BASE; cap never binds)",
                                 PROJ/"results"/"hcap_p3v4_mg_baseline_seed*", None,                 COLOR_ORALL,"D",":"),
 ],
 "K_VRH": [
  ("ACC",                  PROJ/"results_bm"/"bm_p3v4_bm_mg_accel_seed*",    None, COLOR_ACC, "o", "-"),
  ("BASE",                 PROJ/"results_bm"/"bm_p3v4_bm_mg_baseline_seed*", None, COLOR_BASE,"s","--"),
  ("cap-4 (ours)",         PROJ/"results_bm"/"hcap_mgabl_cap4_bm_seed*",     None, COLOR_CAP, "^", "-"),
  ("oracle-all (ours)",    PROJ/"results_bm"/"hcap_mgabl_oracleall_bm_seed*",None, COLOR_ORALL,"D","-"),
 ],
}
LIMITS = {"Cp":(0.25,2.0),"K_VRH":(20.0,400.0)}

def collect(pat, excl, minv, maxv):
    rb_seeds=[]; oc_seeds=[]
    for d in sorted(glob.glob(str(pat))):
        if excl and re.search(excl, d): continue
        ltm = Path(d)/"samples"/"long_term_memory.csv"
        if not ltm.exists(): continue
        df = pd.read_csv(ltm, usecols=["reward","RL_step"])
        max_step = int(df["RL_step"].max())
        if max_step < 15:  # skip very-partial runs (e.g., just-launched resubmits)
            continue
        oc = df.groupby("RL_step").size().reindex(range(HORIZON), fill_value=0).cumsum().values
        df = df[df.reward>0].copy()
        df["prop"] = df.reward*(maxv-minv) + minv
        df.loc[df.reward>=1.0,"prop"] = maxv
        rb = df.groupby("RL_step")["prop"].max().reindex(range(HORIZON)).cummax().ffill().values
        rb_seeds.append(rb); oc_seeds.append(oc)
    return np.array(rb_seeds), np.array(oc_seeds)

def panel(ax, prop, mode):  # mode: "curves" or "oracle"
    minv,maxv = LIMITS[prop]
    for name,pat,excl,color,marker,ls in ARMS[prop]:
        rb, oc = collect(pat,excl,minv,maxv)
        if len(rb)==0: continue
        x = np.arange(HORIZON)
        if mode=="curves":
            m = np.nanmean(rb,axis=0); s = np.nanstd(rb,axis=0)
        else:
            m = oc.mean(axis=0); s = oc.std(axis=0)
        ax.fill_between(x, m-s, m+s, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, m, ls=ls, marker=marker, markersize=4.2, color=color,
                lw=1.5, label=f"{name} (n={len(rb)})")
    ax.set_xlabel("RL cycle"); ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(range(0, 20, 5))
    ax.set_xticks(range(0, 20), minor=True)
    ax.grid(alpha=0.25, lw=0.5)
    if mode=="curves":
        if prop=="Cp":
            ax.set_ylabel(r"running best $C_p$ (J/g/K)")
            ax.axhline(1.5, color="gray", ls=":", lw=0.9, alpha=0.7)
            ax.text(19.3, 1.515, "target = 1.5", color="gray", fontsize=8.5, ha="right", va="bottom")
            ax.set_title("MatterGen / heat capacity")
        else:
            ax.set_ylabel(r"running best $K_{\mathrm{VRH}}$ (GPa)")
            ax.set_title("MatterGen / bulk modulus")
        ax.legend(loc="upper left" if prop=="Cp" else "lower right", handlelength=2.4)
    else:
        ax.set_ylabel(r"cumulative oracle calls (" + (r"$C_p$" if prop=="Cp" else r"$K_{\mathrm{VRH}}$") + ")")
        ax.set_title(f"MatterGen / {'heat capacity' if prop=='Cp' else 'bulk modulus'}")
        ax.legend(loc="upper left", handlelength=2.4)

# Figure 1 — discovery curves
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
panel(axes[0],"Cp","curves"); panel(axes[1],"K_VRH","curves")
fig.suptitle("MatterGen ablation: running best vs RL cycle  (seed-mean $\\pm$ std)", y=1.02, fontsize=12)
fig.tight_layout()
p1 = OUT/"fig_mg_ablation_discovery.png"; fig.savefig(p1); print("saved",p1)

# Figure 2 — cumulative oracle calls
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
panel(axes[0],"Cp","oracle"); panel(axes[1],"K_VRH","oracle")
fig.suptitle("MatterGen ablation: cumulative oracle calls per run  (seed-mean $\\pm$ std)", y=1.02, fontsize=12)
fig.tight_layout()
p2 = OUT/"fig_mg_ablation_oracle_cost.png"; fig.savefig(p2); print("saved",p2)