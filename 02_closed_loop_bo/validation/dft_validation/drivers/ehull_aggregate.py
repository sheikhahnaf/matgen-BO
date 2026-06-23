#!/usr/bin/env python
"""E_above_hull aggregation: our MP-compatible static energy (ehull.json) placed on the MP convex
hull. MP entries fetched via REST thermo for ALL sub-systems (hull needs elemental terminals),
deduped; PhaseDiagram -> get_e_above_hull. Metallic systems need no MP2020 correction (our raw
MP-static energy is comparable); OXIDES are flagged APPROXIMATE (anion/+U corrections not applied
to our entry, which would need the vasprun parameters). chemsys-absent structures are skipped
(no MP hull). Writes results/ehull_summary.csv."""
import os, json, glob, requests, itertools
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
KEY=[l.split("=",1)[1].strip().strip('"').strip("'") for l in open(os.path.expanduser("~/.env")) if l.startswith(("MP_API_KEY","PMG_MAPI_KEY"))][0]
H={"X-API-KEY":KEY}; TH="https://api.materialsproject.org/materials/thermo/"
EH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","faster_ehull")
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","ehull_summary.csv")

def mp_entries(els):
    seen={}; 
    for r in range(1,len(els)+1):
        for combo in itertools.combinations(sorted(els), r):
            cs="-".join(combo)
            resp=requests.get(TH, headers=H, params={"chemsys":cs,"thermo_types":"GGA_GGA+U","_fields":"material_id,composition,energy_per_atom","_per_page":1000}, timeout=90)
            for d in resp.json().get("data",[]):
                seen[d["material_id"]]=d
    ents=[]
    for d in seen.values():
        c=Composition(d["composition"]); ents.append(ComputedEntry(c, d["energy_per_atom"]*c.num_atoms))
    return ents

rows=[]
for j in sorted(glob.glob(os.path.join(EH,"ehull_*","ehull.json"))):
    d=json.load(open(j)); stem=d["stem"]; comp=Composition(d["composition"]); els=[str(e) for e in comp.elements]
    is_oxide="O" in els
    try:
        ents=mp_entries(els)
        comps=set(Composition(e.composition.reduced_formula).reduced_formula for e in ents)
        ours=ComputedEntry(comp, d["energy"])
        pd=PhaseDiagram(ents+[ours])
        eah=pd.get_e_above_hull(ours)
        rows.append(dict(stem=stem, formula=comp.reduced_formula, n_mp=len(ents),
                         e_above_hull=round(eah,3), approx=("oxide-no-corr" if is_oxide else "")))
    except Exception as e:
        rows.append(dict(stem=stem, formula=comp.reduced_formula, n_mp=0, e_above_hull=None, approx="no MP hull: "+repr(e)[:50]))
    print("%-44s %-12s E_hull=%s %s"%(stem[:44], rows[-1]["formula"], rows[-1]["e_above_hull"], rows[-1]["approx"]), flush=True)
import csv
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["stem","formula","n_mp","e_above_hull","approx"]); w.writeheader(); [w.writerow(r) for r in rows]
print("saved -> %s"%OUT)
