"""Top-performing generated structures per v4 run — visualize, classify by space group,
   and cross-reference against the GP warm-start seed pool.

Outputs (read-only on results dirs; nothing is overwritten):
  cp/top_per_job.csv        — top-3 per (paradigm, setup, seed) for Cp
  bm/top_per_job.csv        — same for BM (K_VRH)
  cp/global_top20.csv       — overall top-20 across all Cp jobs
  bm/global_top20.csv       — overall top-20 across all BM jobs
  cp/global_top20_seedmatch.csv  — overlap with 446-row Cp warm-start pool
  bm/global_top20_seedmatch.csv  — overlap with 500-row BM warm-start pool
  figures/spacegroup_dist_cp.png   — space-group distribution among top structures
  figures/spacegroup_dist_bm.png
  figures/composition_top_cp.png   — top elements / formulas
  figures/composition_top_bm.png
  figures/seed_overlap_cp.png      — venn-style: top structures vs seed pool
  figures/seed_overlap_bm.png
  structures/cp_top<N>_<paradigm>_<setup>_seed<S>_<formula>_sg<SG>.cif
  structures/bm_top<N>_<paradigm>_<setup>_seed<S>_<formula>_sg<SG>.cif
"""
import io
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-hcap-bo")
OUT = ROOT / "analysis" / "top_structures"
PARADIGM_COLORS = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10, "figure.dpi": 110,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Discover jobs

def discover(prop: str):
    if prop == "Cp":
        roots = ROOT / "results"
        pat = re.compile(r"hcap_p3v4_(mg|cf|adit)_(baseline|accel)_seed(\d+)_(\d+)$")
        minv, maxv = 0.25, 2.0
    else:
        roots = ROOT / "results_bm"
        pat = re.compile(r"bm_p3v4_bm_(mg|cf|adit)_(baseline|accel)_seed(\d+)_(\d+)$")
        minv, maxv = 20.0, 400.0

    rows = []
    for d in sorted(roots.glob("*")):
        m = pat.match(d.name)
        if not m:
            continue
        paradigm = m.group(1)
        setup = "BASE" if m.group(2) == "baseline" else "ACC"
        seed = int(m.group(3))
        jobid = m.group(4)
        rows.append({"paradigm": paradigm, "setup": setup, "seed": seed,
                     "jobid": jobid, "dir": d})
    df = pd.DataFrame(rows)
    df = df.sort_values("jobid").drop_duplicates(
        subset=["paradigm", "setup", "seed"], keep="last").reset_index(drop=True)
    return df, minv, maxv


def reward_to_value(r, minv, maxv):
    if r >= 1.0:
        return maxv
    return r * (maxv - minv) + minv


# ---------------------------------------------------------------------------
# Per-job: extract top-N by reward, with structure metadata

def top_n_per_job(jobs, prop, minv, maxv, n=3):
    out_rows = []
    for _, j in jobs.iterrows():
        ltm = j["dir"] / "samples" / "long_term_memory.csv"
        if not ltm.exists():
            continue
        try:
            df = pd.read_csv(ltm, usecols=["comp", "ele_comb", "reward",
                                           "RL_step", "cif"])
        except Exception as e:
            print(f"  skip {ltm}: {e}")
            continue
        df = df[df.reward > 0].copy()
        if len(df) == 0:
            continue
        df = df.sort_values("reward", ascending=False).head(n).reset_index(drop=True)
        for rank, row in df.iterrows():
            sg, sg_num, n_atoms, density = analyze_cif(row["cif"])
            out_rows.append({
                "paradigm": j.paradigm, "setup": j.setup, "seed": j.seed,
                "jobid": j.jobid, "rank": rank + 1,
                "RL_step": int(row["RL_step"]),
                "comp": row["comp"], "ele_comb": row["ele_comb"],
                "reward": float(row["reward"]),
                "value": reward_to_value(float(row["reward"]), minv, maxv),
                "sg_symbol": sg, "sg_number": sg_num,
                "n_atoms": n_atoms, "density_gcc": density,
                "cif": row["cif"],
            })
    return pd.DataFrame(out_rows)


def analyze_cif(cif_str: str):
    try:
        s = Structure.from_str(cif_str, fmt="cif")
        sga = SpacegroupAnalyzer(s, symprec=0.1, angle_tolerance=5)
        return sga.get_space_group_symbol(), sga.get_space_group_number(), \
               len(s), float(s.density)
    except Exception:
        return "P1?", 0, 0, 0.0


# ---------------------------------------------------------------------------
# Seed pool match by reduced formula

def reduced_formula(comp_str: str) -> str:
    try:
        return Composition(comp_str).reduced_formula
    except Exception:
        return comp_str.strip()


def load_seed_pool(prop):
    if prop == "Cp":
        df = pd.read_parquet(ROOT / "data" / "ltm_seed_pool_v3.parquet")
    else:
        df = pd.read_parquet(ROOT / "data" / "bm" / "ltm_bm_seed_pool.parquet")
    df["reduced_formula"] = df["formula"].apply(reduced_formula)
    return df


def cross_reference(top_df, seed_df, prop):
    seed_set = set(seed_df["reduced_formula"])
    top_df = top_df.copy()
    top_df["reduced_formula"] = top_df["comp"].apply(reduced_formula)
    top_df["in_seed_pool"] = top_df["reduced_formula"].isin(seed_set)
    return top_df


# ---------------------------------------------------------------------------
# Visualizations

def plot_spacegroup_dist(df, prop_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # by paradigm
    ax = axes[0]
    rows = []
    for paradigm in ["mg", "cf", "adit"]:
        sub = df[df.paradigm == paradigm]
        crystal_systems = sub["sg_number"].apply(crystal_system_from_number)
        for sys, count in crystal_systems.value_counts().items():
            rows.append({"paradigm": paradigm, "system": sys, "count": count})
    if rows:
        plot_df = pd.DataFrame(rows).pivot(index="system", columns="paradigm",
                                           values="count").fillna(0)
        order = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
                 "trigonal", "hexagonal", "cubic"]
        plot_df = plot_df.reindex([o for o in order if o in plot_df.index])
        plot_df.plot(kind="barh", ax=ax,
                     color=[PARADIGM_COLORS[c] for c in plot_df.columns])
        ax.set_xlabel("# top structures"); ax.set_ylabel("crystal system")
        ax.set_title(f"Crystal systems among top-3 ({prop_label}) by paradigm")
        ax.grid(alpha=0.3, axis="x")

    # by setup
    ax = axes[1]
    rows = []
    for setup in ["BASE", "ACC"]:
        sub = df[df.setup == setup]
        crystal_systems = sub["sg_number"].apply(crystal_system_from_number)
        for sys, count in crystal_systems.value_counts().items():
            rows.append({"setup": setup, "system": sys, "count": count})
    if rows:
        plot_df = pd.DataFrame(rows).pivot(index="system", columns="setup",
                                           values="count").fillna(0)
        order = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
                 "trigonal", "hexagonal", "cubic"]
        plot_df = plot_df.reindex([o for o in order if o in plot_df.index])
        plot_df.plot(kind="barh", ax=ax, color=["#888888", "#cc4400"])
        ax.set_xlabel("# top structures"); ax.set_ylabel("crystal system")
        ax.set_title(f"Crystal systems: BASE vs ACC ({prop_label})")
        ax.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Symmetry distribution — top-3 generated structures ({prop_label})",
                 fontweight="bold", y=1.02)
    fig.savefig(fname); plt.close(fig)
    print(f"  → {fname.name}")


def crystal_system_from_number(n: int) -> str:
    if n == 0: return "unknown"
    if n <= 2: return "triclinic"
    if n <= 15: return "monoclinic"
    if n <= 74: return "orthorhombic"
    if n <= 142: return "tetragonal"
    if n <= 167: return "trigonal"
    if n <= 194: return "hexagonal"
    return "cubic"


def plot_composition_top(df, prop_label, fname):
    """Top elements (by frequency in top structures) and top formulas."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # element frequency
    ax = axes[0]
    elem_counts = Counter()
    for ec in df["ele_comb"]:
        for e in re.findall(r"[A-Z][a-z]?", ec):
            elem_counts[e] += 1
    top_elems = elem_counts.most_common(20)
    if top_elems:
        elems, counts = zip(*top_elems)
        ax.bar(elems, counts, color="#4477aa")
        ax.set_xlabel("element"); ax.set_ylabel("# top structures containing")
        ax.set_title(f"Most common elements in top structures ({prop_label})")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")

    # top formulas: highest-property value first (not freq-1 noise)
    ax = axes[1]
    df["reduced_formula"] = df["comp"].apply(reduced_formula)
    # collapse duplicates by formula keeping the max property value
    by_formula = (df.groupby("reduced_formula")
                    .agg(best=("value", "max"), count=("value", "size"))
                    .sort_values("best", ascending=False).head(20))
    if len(by_formula):
        colors = ["#cc6677" if c == 1 else "#117733" for c in by_formula["count"]]
        ax.barh(range(len(by_formula)), by_formula["best"].values, color=colors)
        for i, (idx, row) in enumerate(by_formula.iterrows()):
            label = f"{idx}" + (f" ({row['count']}×)" if row["count"] > 1 else "")
            ax.text(row["best"], i, "  " + label, va="center", fontsize=8)
        ax.set_yticks(range(len(by_formula)))
        ax.set_yticklabels([""] * len(by_formula))
        ax.invert_yaxis()
        unit = "J/g/K" if prop_label == "Cp" else "GPa"
        ax.set_xlabel(f"best {prop_label} value [{unit}]")
        ax.set_title(f"Top-20 reduced formulas by best property ({prop_label})\n"
                     "(red=unique hit, green=multi-seed hit)")
        ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(fname); plt.close(fig)
    print(f"  → {fname.name}")


def plot_seed_overlap(top_df, seed_df, prop_label, fname):
    """How many top structures match seed-pool compositions, broken down by paradigm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # by paradigm
    ax = axes[0]
    rows = []
    for paradigm in ["mg", "cf", "adit"]:
        sub = top_df[top_df.paradigm == paradigm]
        n_in = sub["in_seed_pool"].sum()
        n_out = (~sub["in_seed_pool"]).sum()
        rows.append({"paradigm": paradigm, "in_seed": n_in, "novel": n_out})
    pd_df = pd.DataFrame(rows).set_index("paradigm")
    pd_df.plot(kind="bar", ax=ax, color=["#888888", "#cc6677"], stacked=True)
    ax.set_xlabel("paradigm"); ax.set_ylabel("# top structures")
    ax.set_title(f"Seed-pool overlap by paradigm ({prop_label})")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(alpha=0.3, axis="y")

    # value distribution: in seed vs novel
    ax = axes[1]
    in_vals = top_df.loc[top_df.in_seed_pool, "value"]
    novel_vals = top_df.loc[~top_df.in_seed_pool, "value"]
    bins = np.linspace(top_df["value"].min(), top_df["value"].max(), 25)
    ax.hist([in_vals, novel_vals], bins=bins, label=["in seed", "novel"],
            color=["#888888", "#cc6677"], stacked=True)
    target_or_max = "Cp [J/g/K]" if prop_label == "Cp" else "K_VRH [GPa]"
    ax.set_xlabel(target_or_max); ax.set_ylabel("# top structures")
    ax.set_title(f"Property values: seed-overlap vs novel ({prop_label})")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fname); plt.close(fig)
    print(f"  → {fname.name}")


# ---------------------------------------------------------------------------
# Save top-K CIFs to structures/

def save_top_cifs(global_top, prop_label, structures_dir, n=20):
    saved = []
    for i, row in global_top.head(n).iterrows():
        formula = reduced_formula(row["comp"]).replace("/", "_")
        fname = (f"{prop_label.lower()}_top{i+1:02d}_{row['paradigm']}_{row['setup']}"
                 f"_seed{row['seed']}_{formula}_sg{row['sg_number']}.cif")
        path = structures_dir / fname
        path.write_text(row["cif"])
        saved.append(fname)
    return saved


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cp").mkdir(exist_ok=True)
    (OUT / "bm").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "structures").mkdir(exist_ok=True)

    summary_lines = []

    for prop in ["Cp", "BM"]:
        print(f"\n=== {prop} ===")
        prop_label = prop
        outdir = OUT / prop.lower()
        jobs, minv, maxv = discover(prop)
        print(f"  jobs: {len(jobs)}")

        top = top_n_per_job(jobs, prop, minv, maxv, n=3)
        print(f"  top-3 per job rows: {len(top)}")

        seed = load_seed_pool(prop)
        print(f"  seed pool size: {len(seed)} ({prop} warm-start)")

        top = cross_reference(top, seed, prop)
        top.drop(columns=["cif"]).to_csv(outdir / "top_per_job.csv", index=False)

        # global top 20
        global_top = top.sort_values("reward", ascending=False).head(20).reset_index(drop=True)
        global_top.drop(columns=["cif"]).to_csv(outdir / "global_top20.csv", index=False)
        seedmatch = global_top[["rank", "paradigm", "setup", "seed", "comp",
                                "reduced_formula", "value", "sg_symbol",
                                "sg_number", "n_atoms", "in_seed_pool"]].copy()
        seedmatch.to_csv(outdir / "global_top20_seedmatch.csv", index=False)

        plot_spacegroup_dist(top, prop_label,
                             OUT / "figures" / f"spacegroup_dist_{prop.lower()}.png")
        plot_composition_top(top, prop_label,
                             OUT / "figures" / f"composition_top_{prop.lower()}.png")
        plot_seed_overlap(top, seed, prop_label,
                          OUT / "figures" / f"seed_overlap_{prop.lower()}.png")
        save_top_cifs(global_top, prop, OUT / "structures", n=20)

        # text summary
        n_total = len(top)
        n_overlap = int(top["in_seed_pool"].sum())
        pct = 100.0 * n_overlap / max(n_total, 1)
        summary_lines.append(f"\n=== {prop} ===")
        summary_lines.append(f"  Top-3 structures: {n_total}")
        summary_lines.append(f"  Reduced-formula in seed pool: {n_overlap} ({pct:.1f}%)")
        summary_lines.append(f"  Best: {global_top.iloc[0]['value']:.3f} "
                             f"({global_top.iloc[0]['comp']}, "
                             f"sg {global_top.iloc[0]['sg_symbol']} #{global_top.iloc[0]['sg_number']}, "
                             f"{global_top.iloc[0]['paradigm']}/{global_top.iloc[0]['setup']}/seed{global_top.iloc[0]['seed']}, "
                             f"in_seed={bool(global_top.iloc[0]['in_seed_pool'])})")
        # crystal system breakdown
        cs_counts = top["sg_number"].apply(crystal_system_from_number).value_counts()
        summary_lines.append(f"  Crystal systems: {dict(cs_counts)}")
        # paradigm overlap
        for paradigm in ["mg", "cf", "adit"]:
            p = top[top.paradigm == paradigm]
            if len(p):
                pct_p = 100 * p["in_seed_pool"].sum() / len(p)
                summary_lines.append(f"  {paradigm}: {p['in_seed_pool'].sum()}/{len(p)} "
                                     f"({pct_p:.1f}%) match seed-pool formulas")

    summary = "\n".join(summary_lines)
    (OUT / "SUMMARY.txt").write_text(summary)
    print("\n=== SUMMARY ===")
    print(summary)


if __name__ == "__main__":
    main()
