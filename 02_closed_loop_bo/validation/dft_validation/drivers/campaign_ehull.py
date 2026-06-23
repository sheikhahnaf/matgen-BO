#!/usr/bin/env python
"""E_above_hull static: MP-COMPATIBLE single-point on the DFT-relaxed cell.
Uses MPStaticSet (MP INCAR: ENCUT 520, +U where MP applies, MP POTCARs) so the energy is
comparable to the Materials Project hull (NOT our ENCUT=680/no-U EOS energy). The PhaseDiagram /
MP2020Compatibility step is done separately after these statics + MP-entry download.
Usage: campaign_ehull.py <relaxed_POSCAR> <K0.json (for magmoms)>  -> writes ehull_<stem>/ehull.json"""
import os, sys, json
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.outputs import Vasprun
from custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler,
    NonConvergingErrorHandler, PotimErrorHandler, FrozenJobErrorHandler, StdErrHandler, LargeSigmaHandler)
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator
VASP=["srun","--mpi=pmi2","vasp_std"]
H=[VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(), NonConvergingErrorHandler(),
   PotimErrorHandler(), FrozenJobErrorHandler(), StdErrHandler(), LargeSigmaHandler()]
V=[VasprunXMLValidator(), VaspFilesValidator()]

def static_done(d):
    try: return bool(Vasprun(os.path.join(d,"vasprun.xml"),parse_dos=False,parse_eigen=False,parse_potcar_file=False).converged_electronic)
    except Exception: return False

def main():
    poscar=sys.argv[1]; magj=sys.argv[2] if len(sys.argv)>2 else None
    # stem from the parent eos_<stem> dir
    stem=os.path.basename(os.path.dirname(os.path.abspath(poscar))).replace("eos_","")
    base=os.path.abspath("ehull_"+stem); os.makedirs(base,exist_ok=True)
    s=Structure.from_file(poscar)
    mag=None
    if magj and os.path.exists(magj):
        mag=json.load(open(magj)).get("magmoms")
    if mag and any(abs(m)>0.05 for m in mag):
        s.add_site_property("magmom", list(mag))
    if not static_done(base):
        # MP-compatible static (MP INCAR defaults). NSW=0 single-point on the relaxed cell.
        MPStaticSet(s, user_potcar_functional="PBE_54").write_input(base)
        cwd=os.getcwd(); os.chdir(base)
        try: Custodian(H,[VaspJob(VASP)],V,max_errors=6).run()
        finally: os.chdir(cwd)
    vr=Vasprun(os.path.join(base,"vasprun.xml"),parse_potcar_file=False)
    res=dict(stem=stem, energy=float(vr.final_energy), n_sites=len(s),
             energy_per_atom=float(vr.final_energy)/len(s),
             composition=s.composition.formula, reduced=s.composition.reduced_formula,
             structure=s.as_dict())
    json.dump(res, open(os.path.join(base,"ehull.json"),"w"))
    print("EHULL-STATIC %s  E=%.4f eV  (%.4f eV/atom)" % (stem, res["energy"], res["energy_per_atom"]), flush=True)

if __name__=="__main__":
    main()
