#!/usr/bin/env python
"""E_above_hull with MP2020 anion correction applied to the discovery entry (rigor fix).

The original ehull_aggregate.py compared a RAW (uncorrected) discovery static energy against an
MP convex hull built from MP2020-CORRECTED reference energies (the thermo endpoint's
energy_per_atom is the corrected value). That asymmetry inflates E_hull for every anion phase
(oxides + nitrides) by the size of the missing anion correction. Here we correct the discovery
entry with the SAME scheme (MaterialsProject2020Compatibility) so both sides are on equal footing,
and report old (uncorrected) vs new (corrected) E_hull side by side.

Metallic/boride/carbide/intermetallic entries get ~zero MP2020 correction, so their E_hull is
unchanged; only the anion phases move (downward, since the correction lowers the entry energy).
Writes results/ehull_summary_corrected.csv. No paper edits.
"""
import os, json, glob, requests, itertools
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram

KEY = [l.split("=", 1)[1].strip().strip('"').strip("'")
       for l in open(os.path.expanduser("~/.env")) if l.startswith(("MP_API_KEY", "PMG_MAPI_KEY"))][0]
H = {"X-API-KEY": KEY}
TH = "https://api.materialsproject.org/materials/thermo/"
HERE = os.path.dirname(os.path.abspath(__file__))
EH = os.path.join(HERE, "..", "results", "faster_ehull")
OUT = os.path.join(HERE, "..", "results", "ehull_summary_corrected.csv")

COMPAT = MaterialsProject2020Compatibility(check_potcar=False)


def mp_entries(els):
    """MP reference entries (GGA/GGA+U), corrected energy_per_atom (= MP2020-corrected)."""
    seen = {}
    for r in range(1, len(els) + 1):
        for combo in itertools.combinations(sorted(els), r):
            cs = "-".join(combo)
            resp = requests.get(TH, headers=H, params={
                "chemsys": cs, "thermo_types": "GGA_GGA+U",
                "_fields": "material_id,composition,energy_per_atom", "_per_page": 1000}, timeout=90)
            for d in resp.json().get("data", []):
                seen[d["material_id"]] = d
    ents = []
    for d in seen.values():
        c = Composition(d["composition"])
        ents.append(ComputedEntry(c, d["energy_per_atom"] * c.num_atoms))
    return ents


def corrected_entry(structure, energy):
    """Discovery entry with MP2020 anion (and, where applicable, +U) correction applied.
    Returns (corrected ComputedStructureEntry, total adjustment in eV) or (raw, 0.0) if rejected."""
    ce = ComputedStructureEntry(structure, energy, parameters={
        "run_type": "GGA", "is_hubbard": False, "hubbards": {}, "potcar_symbols": []})
    proc = COMPAT.process_entries([ce], clean=True, verbose=False)
    if proc:
        adj = sum(a.value for a in proc[0].energy_adjustments)
        return proc[0], adj
    return ComputedEntry(structure.composition, energy), 0.0


def main():
    ANIONS = {"O", "N", "F", "S", "Cl"}
    rows = []
    for j in sorted(glob.glob(os.path.join(EH, "ehull_*", "ehull.json"))):
        d = json.load(open(j))
        stem = d["stem"]
        comp = Composition(d["composition"])
        els = [str(e) for e in comp.elements]
        nat = comp.num_atoms
        is_anion = bool(set(els) & ANIONS)
        try:
            ents = mp_entries(els)
            # OLD: raw discovery energy on the hull
            raw = ComputedEntry(comp, d["energy"])
            eah_old = PhaseDiagram(ents + [raw]).get_e_above_hull(raw)
            # NEW: MP2020-corrected discovery energy on the hull
            s = Structure.from_dict(d["structure"])
            corr_entry, adj = corrected_entry(s, d["energy"])
            eah_new = PhaseDiagram(ents + [corr_entry]).get_e_above_hull(corr_entry)
            rows.append(dict(stem=stem, formula=comp.reduced_formula, n_mp=len(ents),
                             anion=("yes" if is_anion else ""),
                             e_hull_old=round(eah_old, 3), corr_eV=round(adj, 3),
                             corr_eV_per_atom=round(adj / nat, 3), e_hull_new=round(eah_new, 3)))
        except Exception as e:
            rows.append(dict(stem=stem, formula=comp.reduced_formula, n_mp=0, anion=("yes" if is_anion else ""),
                             e_hull_old=None, corr_eV=None, corr_eV_per_atom=None, e_hull_new=None,
                             err=repr(e)[:60]))
        r = rows[-1]
        flag = " <-- ANION (moved)" if (r.get("corr_eV") and abs(r["corr_eV"]) > 1e-6) else ""
        print("%-30s %-10s  old=%s  corr=%s eV (%s/atom)  new=%s%s"
              % (stem[:30], r["formula"], r["e_hull_old"], r["corr_eV"], r["corr_eV_per_atom"],
                 r["e_hull_new"], flag), flush=True)
    import csv
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "formula", "n_mp", "anion", "e_hull_old",
                                          "corr_eV", "corr_eV_per_atom", "e_hull_new", "err"],
                           extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in rows]
    print("\nsaved -> %s" % OUT)


if __name__ == "__main__":
    main()
