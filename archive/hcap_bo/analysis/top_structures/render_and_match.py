"""(a) Render 3D images of top-3 structures per property + paradigm.
   (b) Structural similarity match between each top structure and the GP seed pool
       (lattice+sites via pymatgen StructureMatcher, not just reduced formula).
   (c) Structure-zoo overview figure: 4×5 grid of top-20 generated structures.

Outputs (read-only on results/data dirs):
  renders/<prop>/<paradigm>_<setup>_seed<S>_<formula>_sg<SG>_v<VAL>.png
  zoo/zoo_cp.png, zoo/zoo_bm.png            — 4×5 grids of top-20 per property
  match/<prop>_seed_match.csv               — closest seed match per top structure
  match/seed_distance_hist_<prop>.png       — distance distribution
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import jsonio
from ase.visualize.plot import plot_atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

warnings.filterwarnings("ignore")

ROOT = Path("/Volumes/SSD1_SMAAA/matinvent-hcap-bo")
ANA = ROOT / "analysis" / "top_structures"
OUT_RENDER = ANA / "renders"
OUT_ZOO = ANA / "zoo"
OUT_MATCH = ANA / "match"
for d in (OUT_RENDER, OUT_ZOO, OUT_MATCH, OUT_RENDER / "cp", OUT_RENDER / "bm"):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Helvetica", "savefig.dpi": 150,
                     "savefig.bbox": "tight"})

PARADIGM_COLORS = {"mg": "#1f77b4", "cf": "#2ca02c", "adit": "#d62728"}


def reduced_formula(comp_str: str) -> str:
    try:
        return Composition(comp_str).reduced_formula
    except Exception:
        return comp_str


def load_top_with_cif(prop: str) -> pd.DataFrame:
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
        paradigm, raw_setup, seed, jobid = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        setup = "BASE" if raw_setup == "baseline" else "ACC"
        ltm = d / "samples" / "long_term_memory.csv"
        if not ltm.exists():
            continue
        try:
            df = pd.read_csv(ltm, usecols=["comp", "ele_comb", "reward",
                                           "RL_step", "cif"])
        except Exception:
            continue
        df = df[df.reward > 0].sort_values("reward", ascending=False).head(3)
        df = df.reset_index(drop=True)
        for r, row in df.iterrows():
            rows.append({
                "paradigm": paradigm, "setup": setup, "seed": seed, "jobid": jobid,
                "rank": r + 1, "RL_step": int(row["RL_step"]),
                "comp": row["comp"], "ele_comb": row["ele_comb"],
                "reward": float(row["reward"]),
                "value": maxv if row["reward"] >= 1.0
                        else float(row["reward"]) * (maxv - minv) + minv,
                "cif": row["cif"],
            })
    df = pd.DataFrame(rows).sort_values("reward", ascending=False)
    df = df.drop_duplicates(subset=["paradigm", "setup", "seed", "rank"]).reset_index(drop=True)
    return df


def cif_to_structure(cif_str: str) -> Structure:
    return Structure.from_str(cif_str, fmt="cif")


def annotate_structure(s: Structure):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(s, symprec=0.1, angle_tolerance=5)
    return sga.get_space_group_symbol(), sga.get_space_group_number()


# ---------------------------------------------------------------------------
# (a) Per-structure 3D rendering

def render_structure(s: Structure, fname: Path, title: str):
    atoms = s.to_ase_atoms() if hasattr(s, "to_ase_atoms") else s.to(fmt="ase")
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    plot_atoms(atoms, ax, radii=0.45, rotation="20x,30y,0z",
               show_unit_cell=2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def render_top_per_property(df: pd.DataFrame, prop_label: str,
                            outdir: Path, n_per_paradigm_setup=3):
    """Render top-N (rank 1-3) per (paradigm × setup)."""
    rendered = []
    grouped = df.groupby(["paradigm", "setup"])
    for (paradigm, setup), grp in grouped:
        grp = grp.sort_values("reward", ascending=False).head(n_per_paradigm_setup)
        for _, row in grp.iterrows():
            try:
                s = cif_to_structure(row["cif"])
                sg_sym, sg_num = annotate_structure(s)
            except Exception as e:
                print(f"  skip {paradigm}/{setup}/{row['comp']}: {e}")
                continue
            unit = "J/g/K" if prop_label == "Cp" else "GPa"
            formula = reduced_formula(row["comp"]).replace("/", "_")
            fname = (outdir / f"{paradigm}_{setup}_seed{row['seed']}_rank{row['rank']}"
                              f"_{formula}_sg{sg_num}.png")
            title = (f"{paradigm}/{setup}/seed{row['seed']} #{row['rank']}\n"
                     f"{formula}  sg={sg_sym}({sg_num})\n"
                     f"{prop_label}={row['value']:.2f} {unit}")
            render_structure(s, fname, title)
            rendered.append(fname)
    print(f"  rendered {len(rendered)} structures → {outdir}")
    return rendered


# ---------------------------------------------------------------------------
# (b) Structural similarity to seed pool (lattice + sites)

def load_seed_structures(prop: str):
    if prop == "Cp":
        df = pd.read_parquet(ROOT / "data" / "ltm_seed_pool_v3.parquet")
    else:
        df = pd.read_parquet(ROOT / "data" / "bm" / "ltm_bm_seed_pool.parquet")
    df["reduced_formula"] = df["formula"].apply(reduced_formula)
    return df


def atoms_json_to_pmg(aj: str) -> Structure:
    atoms = jsonio.decode(aj)
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_structure(atoms)


def closest_seed_match(top_df: pd.DataFrame, seed_df: pd.DataFrame,
                       prop_label: str) -> pd.DataFrame:
    """For each top structure, find the closest seed entry by:
       1) reduced-formula match → True/False
       2) StructureMatcher fit (anonymous=False) — strict lattice+sites match
       3) anonymous StructureMatcher (any element substitution allowed) → distance proxy
       4) volume / density / lattice-param mismatch

    Returns enriched DataFrame.
    """
    # cache seed structures for formulas hit by top set
    needed_formulas = set(top_df["comp"].apply(reduced_formula))
    seed_sub = seed_df[seed_df["reduced_formula"].isin(needed_formulas)].copy()
    seed_struct_cache = {}
    for _, sr in seed_sub.iterrows():
        try:
            s = atoms_json_to_pmg(sr["atoms_json"])
            seed_struct_cache.setdefault(sr["reduced_formula"], []).append(
                (sr["structure_id"], s))
        except Exception:
            continue

    matcher_strict = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
                                      primitive_cell=True, scale=True)
    matcher_anon = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=10,
                                    primitive_cell=True, scale=True,
                                    attempt_supercell=True)

    out = []
    for _, row in top_df.iterrows():
        rf = reduced_formula(row["comp"])
        try:
            s_top = cif_to_structure(row["cif"])
        except Exception:
            continue
        rec = {
            "paradigm": row["paradigm"], "setup": row["setup"], "seed": row["seed"],
            "rank": row["rank"], "comp": row["comp"], "reduced_formula": rf,
            "value": row["value"],
            "in_seed_formula": rf in seed_struct_cache,
            "strict_match_id": None, "strict_match_dist": np.nan,
            "best_anon_match_id": None, "best_anon_dist": np.nan,
            "min_density_diff_pct": np.nan,
            "min_volume_diff_pct": np.nan,
            "n_seed_candidates_same_formula": 0,
        }
        if rf in seed_struct_cache:
            cands = seed_struct_cache[rf]
            rec["n_seed_candidates_same_formula"] = len(cands)
            best_strict_d, best_strict_id = np.inf, None
            best_anon_d, best_anon_id = np.inf, None
            best_dens, best_vol = np.inf, np.inf
            for sid, s_seed in cands:
                # strict — same elements + close lattice+sites
                try:
                    if matcher_strict.fit(s_top, s_seed):
                        d = matcher_strict.get_rms_dist(s_top, s_seed)
                        if d is not None and d[0] < best_strict_d:
                            best_strict_d = d[0]; best_strict_id = sid
                except Exception:
                    pass
                # anonymous — same stoichiometry, allow element relabel
                try:
                    d2 = matcher_anon.get_rms_dist(s_top, s_seed)
                    if d2 is not None and d2[0] < best_anon_d:
                        best_anon_d = d2[0]; best_anon_id = sid
                except Exception:
                    pass
                # density/volume diff (always defined)
                dens_diff = abs(s_top.density - s_seed.density) / s_seed.density * 100
                vol_diff = abs(s_top.volume / len(s_top)
                               - s_seed.volume / len(s_seed)) \
                           / (s_seed.volume / len(s_seed)) * 100
                if dens_diff < best_dens: best_dens = dens_diff
                if vol_diff < best_vol: best_vol = vol_diff
            rec["strict_match_id"] = best_strict_id if best_strict_d < np.inf else None
            rec["strict_match_dist"] = best_strict_d if best_strict_d < np.inf else np.nan
            rec["best_anon_match_id"] = best_anon_id if best_anon_d < np.inf else None
            rec["best_anon_dist"] = best_anon_d if best_anon_d < np.inf else np.nan
            rec["min_density_diff_pct"] = best_dens if best_dens < np.inf else np.nan
            rec["min_volume_diff_pct"] = best_vol if best_vol < np.inf else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def plot_seed_distance(match_df: pd.DataFrame, prop_label: str, fname: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    same = match_df[match_df["in_seed_formula"]]
    ax.bar(["unique formula", "shared formula\nw/ seed"],
           [(~match_df["in_seed_formula"]).sum(), len(same)],
           color=["#cc6677", "#888888"])
    ax.set_ylabel("# top structures"); ax.set_title("formula-level overlap")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate([(~match_df["in_seed_formula"]).sum(), len(same)]):
        ax.text(i, v + 0.3, str(v), ha="center")

    ax = axes[1]
    if same["best_anon_dist"].notna().any():
        ax.hist(same["best_anon_dist"].dropna(), bins=10, color="#4477aa",
                edgecolor="black")
        ax.set_xlabel("RMS site distance to closest seed [normalized]")
        ax.set_ylabel("# top structures")
        ax.set_title(f"For {len(same)} formula-shared cases:\n"
                     "anonymous-match RMS distance to closest seed")
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No formula-shared top structures",
                ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"Seed-pool distance for top structures ({prop_label})",
                 fontweight="bold", y=1.02)
    fig.savefig(fname); plt.close(fig)
    print(f"  → {fname.name}")


# ---------------------------------------------------------------------------
# (c) Structure zoo: 4×5 grid of top-20 per property

def build_zoo(df: pd.DataFrame, prop_label: str, fname: Path, n=20):
    fig, axes = plt.subplots(4, 5, figsize=(15, 13))
    axes = axes.flatten()
    df = df.sort_values("reward", ascending=False).head(n).reset_index(drop=True)
    for i, ax in enumerate(axes):
        if i >= len(df):
            ax.axis("off"); continue
        row = df.iloc[i]
        try:
            s = cif_to_structure(row["cif"])
            sg_sym, sg_num = annotate_structure(s)
            atoms = s.to_ase_atoms() if hasattr(s, "to_ase_atoms") else s.to(fmt="ase")
            plot_atoms(atoms, ax, radii=0.45, rotation="20x,30y,0z",
                       show_unit_cell=2)
        except Exception as e:
            ax.text(0.5, 0.5, f"render failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off"); continue
        ax.set_axis_off()
        unit = "J/g/K" if prop_label == "Cp" else "GPa"
        formula = reduced_formula(row["comp"])
        title = (f"#{i+1}  {formula}\n"
                 f"sg={sg_sym}({sg_num})\n"
                 f"{prop_label}={row['value']:.2f}{unit}\n"
                 f"{row['paradigm']}/{row['setup']}/s{row['seed']}")
        # color border by paradigm
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PARADIGM_COLORS[row["paradigm"]])
            spine.set_linewidth(2.5)
        ax.set_title(title, fontsize=8.5)
    fig.suptitle(f"Top-{n} generated structures — {prop_label}",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    print(f"  → {fname.name}")


# ---------------------------------------------------------------------------

def main():
    for prop, prop_label in [("Cp", "Cp"), ("BM", "K_VRH")]:
        print(f"\n=== {prop} ===")
        df = load_top_with_cif(prop)
        print(f"  top-3 jobs: {len(df)}")

        # (c) zoo
        build_zoo(df, prop_label, OUT_ZOO / f"zoo_{prop.lower()}.png", n=20)

        # (a) per-structure renders
        render_top_per_property(df, prop_label, OUT_RENDER / prop.lower(),
                                n_per_paradigm_setup=3)

        # (b) seed match
        seed = load_seed_structures(prop)
        match = closest_seed_match(df.head(20), seed, prop_label)
        match.to_csv(OUT_MATCH / f"{prop.lower()}_seed_match.csv", index=False)
        plot_seed_distance(match, prop_label,
                           OUT_MATCH / f"seed_distance_{prop.lower()}.png")

        # text summary for matches
        print(f"  formula-overlap: {match['in_seed_formula'].sum()}/{len(match)}")
        any_strict = match["strict_match_id"].notna().sum()
        print(f"  strict StructureMatcher fits: {any_strict}/{len(match)}")
        anon_q = match["best_anon_dist"].dropna()
        if len(anon_q):
            print(f"  anon-RMS distance (formula-shared subset): "
                  f"{anon_q.min():.3f} … {anon_q.max():.3f}")

    print(f"\nOutputs in: {ANA}")


if __name__ == "__main__":
    main()
