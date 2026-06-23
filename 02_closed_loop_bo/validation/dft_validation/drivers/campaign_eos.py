#!/usr/bin/env python
"""Campaign DFT EOS driver — K0 (bulk modulus) for the AI-generated structures,
matching the eSEN oracle's protocol (local_esen_bm.py) so DFT vs oracle is like-for-like.

ORACLE PARITY (the comparison must be apples-to-apples):
  * relax cell+ions to equilibrium, then
  * 7 isotropic strain points eps in {-0.03,-0.02,-0.01,0,+0.01,+0.02,+0.03}
  * RIGID scaling: cell -> cell*(1+eps)^(1/3), fractional coords FROZEN, single-point energy
    (the oracle does NOT relax ions at each strained volume -> neither do we)
  * 3rd-order Birch-Murnaghan fit -> B0 (GPa)

CAMPAIGN TRACTABILITY FIX (vs the ISYM=-1 conv test that stalled on MoN/Na2BO3):
  AI cells carry ~0.01-0.05 A lattice noise; spglib refines to the intended SG so VASP runs
  with full symmetry (ISYM=2; k-mesh reduced 4-24x). P1 cells (8/29) -> ISYM=0. The phonon
  stage independently checks the refined symmetry is a stable minimum.

PHYSICS / NUMERICS (hardened after the pre-flight audit):
  * ENCUT=680 (conv-locked, B0 flat 520->760); PBE_54; no +U (0/29 trigger MP's +U scheme).
  * ISPIN=2 explicit (MP convention -> parity with the MP/OMat-trained eSEN oracle, which
    approximates spin-polarized DFT). Magnetic state is PINNED across the 7 EOS volumes by
    seeding MAGMOM from the relaxed reference's moments, so a magnetic system stays on ONE
    branch and E(V) is smooth (non-magnetic systems -> moments ~0 -> a no-op).
  * relax: Gaussian ISMEAR=0/SIGMA=0.05 (gap-agnostic). EOS statics: tetrahedron ISMEAR=-5
    (most accurate total energy) WITH a sparse-mesh guard -> fall back to ISMEAR=0/0.05 when
    any k-axis has <4 subdivisions (tetrahedron is fragile on a near-Gamma mesh). EDIFF=1e-7
    for statics (B0 is a 2nd derivative; the +-3% energy window can be only a few meV).
  * ONE fixed k-grid from the relaxed reference cell across all 7 strain points (recomputing
    per cell makes the mesh drift mid-EOS -> corrupt BM; that was the 21/16/18.2 oscillation).
  * Energy harvested from vasprun gated on converged_electronic (rejects NELM-hit statics);
    a failed static -> None -> the BM fit is refused with a structure-named error, not a numpy
    TypeError; partial (V,E) is always persisted for diagnosis/resubmit.
"""
import os, sys, re, json, shutil, numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.analysis.eos import EOS
from custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (
    VaspErrorHandler, MeshSymmetryErrorHandler, UnconvergedErrorHandler,
    NonConvergingErrorHandler, PotimErrorHandler, PositiveEnergyErrorHandler,
    FrozenJobErrorHandler, StdErrHandler, LargeSigmaHandler, IncorrectSmearingHandler)
from custodian.vasp.validators import VasprunXMLValidator, VaspFilesValidator

VASP = ["srun", "--mpi=pmi2", "vasp_std"]
HANDLERS = [VaspErrorHandler(), MeshSymmetryErrorHandler(), UnconvergedErrorHandler(),
            NonConvergingErrorHandler(), PotimErrorHandler(), PositiveEnergyErrorHandler(),
            FrozenJobErrorHandler(), StdErrHandler(), LargeSigmaHandler(), IncorrectSmearingHandler()]
VALIDATORS = [VasprunXMLValidator(), VaspFilesValidator()]
POT = dict(user_potcar_functional="PBE_54")
STRAINS = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])   # oracle's 7-point grid
SYMPRECS = [1e-3, 1e-2, 5e-2, 1e-1]
MIN_SUBDIV_FOR_TETRA = 4   # ISMEAR=-5 needs a real 3D mesh; below this -> Gaussian fallback

def base_incar(isym, encut=680):
    # ENCUT default 680 (conv-locked); overridable so a convergence driver can sweep it.
    # ISYM per-structure (2 refined-symmetric / 0 P1). ISPIN=2 explicit: MP/OMat-trained oracle
    # approximates spin-polarized DFT, so this is the parity choice; magnetism is pinned across
    # volumes via MAGMOM seeding (see eos_b0).
    return dict(ENCUT=encut, ISYM=isym, ISPIN=2, EDIFF=1e-6, ALGO="Normal",
                LREAL=False, LWAVE=False, LCHARG=False, LASPH=True, NCORE=4)

def relax_incar(isym, encut=680):
    return dict(base_incar(isym, encut), ISMEAR=0, SIGMA=0.05,
                ISIF=3, IBRION=2, EDIFFG=-0.02, NSW=120, NELM=200, NELMIN=6)

def stat_incar(isym, tetra=True, encut=680):
    # EOS static: tetrahedron (ISMEAR=-5) for the most accurate total energy when the mesh
    # supports it; Gaussian fallback on sparse meshes. EDIFF tightened to 1e-7 for curvature.
    sm = dict(ISMEAR=-5) if tetra else dict(ISMEAR=0, SIGMA=0.05)
    return dict(base_incar(isym, encut), IBRION=-1, NSW=0, EDIFF=1e-7, **sm)

def intended_sg(fn):
    m = re.search(r"_sg(\d+)", fn)
    return int(m.group(1)) if m else None

def prep_cell(cif):
    """Clean AI lattice noise -> symmetric primitive cell + ISYM. Never invents symmetry
    beyond the intended SG. Returns (structure, isym, info)."""
    s0 = Structure.from_file(cif)
    want = intended_sg(os.path.basename(cif))
    if want is None or want == 1:
        s = s0.get_reduced_structure()
        return s, 0, dict(want=want, used="P1/Niggli", symprec=None,
                          sg_final=SpacegroupAnalyzer(s, symprec=1e-2).get_space_group_number(),
                          nat=len(s))
    for sp in SYMPRECS:
        sga = SpacegroupAnalyzer(s0, symprec=sp)
        if sga.get_space_group_number() == want:
            s = sga.get_primitive_standard_structure()
            sgn = SpacegroupAnalyzer(s, symprec=1e-3).get_space_group_number()
            return s, 2, dict(want=want, used="refined", symprec=sp, sg_final=sgn, nat=len(s))
    s = s0.get_reduced_structure()
    return s, 0, dict(want=want, used="P1-fallback(no-SG-match)", symprec=None,
                      sg_final=SpacegroupAnalyzer(s, symprec=1e-2).get_space_group_number(), nat=len(s))

def grid_for(s, spacing):
    # VASP KSPACING convention: N_i = max(1, ceil(|b_i| / spacing))
    b = s.lattice.reciprocal_lattice.abc
    return [max(1, int(np.ceil(bi / spacing))) for bi in b]

def write_grid(d, n):
    Kpoints.gamma_automatic(n).write_file(os.path.join(d, "KPOINTS"))

def assert_no_kspacing(d):
    # version-proof the fixed-grid guarantee: if a future MP set injects KSPACING into INCAR,
    # VASP ignores our KPOINTS and silently reverts to per-cell adaptive meshing.
    inc = Incar.from_file(os.path.join(d, "INCAR"))
    if "KSPACING" in inc:
        del inc["KSPACING"]; inc.write_file(os.path.join(d, "INCAR"))

def static_done(d):
    """True iff a complete, electronically-converged static already lives in d (idempotent resume)."""
    try:
        vr = Vasprun(os.path.join(d, "vasprun.xml"), parse_dos=False, parse_eigen=False,
                     parse_potcar_file=False)
        return bool(vr.converged_electronic)
    except Exception:
        return False

def cust(d, resume_ok=True):
    if resume_ok and static_done(d):
        return                                   # already complete -> don't rerun (resume)
    cwd = os.getcwd(); os.chdir(d)
    try:
        Custodian(HANDLERS, [VaspJob(VASP)], VALIDATORS, max_errors=8).run()
    finally:
        os.chdir(cwd)

def read_energy(d):
    """Converged total energy or None. Gated on electronic convergence so a NELM-hit static is
    rejected rather than silently fed to the BM fit."""
    try:
        vr = Vasprun(os.path.join(d, "vasprun.xml"), parse_dos=False, parse_eigen=False,
                     parse_potcar_file=False)
        if not vr.converged_electronic:
            return None
        return float(vr.final_energy)
    except Exception:
        return None

def read_magmoms(wd):
    """Per-atom magnetic moments from the relaxed reference (for pinning the EOS to one branch)."""
    try:
        mag = Outcar(os.path.join(wd, "OUTCAR")).magnetization
        if mag:
            return [float(m["tot"]) for m in mag]
    except Exception:
        pass
    return None

def relax_ok(wd):
    try:
        vr = Vasprun(os.path.join(wd, "vasprun.xml"), parse_dos=False, parse_eigen=False,
                     parse_potcar_file=False)
        return bool(vr.converged_electronic and vr.converged_ionic)
    except Exception:
        return False

def relax(s, wd, isym, spacing, encut=680):
    os.makedirs(wd, exist_ok=True)
    if not relax_ok(wd):                          # clean rerun if a prior relax didn't validate
        for f in os.listdir(wd):
            p = os.path.join(wd, f)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        MPRelaxSet(s, user_incar_settings=relax_incar(isym, encut), **POT).write_input(wd)
        write_grid(wd, grid_for(s, spacing)); assert_no_kspacing(wd)
        cust(wd, resume_ok=False)
    if not relax_ok(wd):
        raise RuntimeError("%s: reference relax did not converge (cell+ions); refusing to "
                           "build the EOS on an unrelaxed reference." % wd)
    return Structure.from_file(os.path.join(wd, "CONTCAR")), read_magmoms(wd)

def eos_b0(rel, wd, isym, spacing, magmoms=None, encut=680):
    # ONE fixed k-grid (from the relaxed reference cell) for ALL strain points — DO NOT
    # re-derive per strained cell or the mesh drifts mid-EOS and corrupts the BM curvature.
    n = grid_for(rel, spacing)
    tetra = min(n) >= MIN_SUBDIV_FOR_TETRA
    base_stat = stat_incar(isym, tetra=tetra, encut=encut)
    # pin one magnetic branch: seed the SAME per-site moments (from the relaxed reference) into
    # every strain static. MAGMOM is set via the structure's site property (pymatgen honors a
    # per-site 'magmom' list); it must NOT go into user_incar_settings, which expects a
    # species->moment dict, not a list (that AttributeErrors at write time).
    pin = magmoms is not None and any(abs(m) > 0.05 for m in magmoms)
    V, E = [], []
    for i, eps in enumerate(STRAINS):
        d = os.path.join(wd, "v%d" % i)
        st = rel.copy()
        if pin:
            st.add_site_property("magmom", list(magmoms))
        st.scale_lattice(rel.volume * (1 + eps))                    # rigid isotropic, frozen frac
        if not static_done(d):
            os.makedirs(d, exist_ok=True)
            MPStaticSet(st, user_incar_settings=base_stat, **POT).write_input(d)
            write_grid(d, n); assert_no_kspacing(d)                 # SAME grid every volume
        cust(d)
        V.append(st.volume); E.append(read_energy(d))
    V = np.asarray(V, float)
    Ef = np.asarray([np.nan if e is None else e for e in E], float)
    ok = np.isfinite(Ef)
    diag = dict(k_grid=n, encut=encut, smearing=("tetra-5" if tetra else "gauss-0/0.05"),
                pinned_magmom=pin, V=V.tolist(), E=[None if e is None else e for e in E])
    if ok.sum() < 5:
        raise RuntimeError("%s: only %d/7 statics converged (failed eps=%s); BM fit refused. %s"
                           % (wd, int(ok.sum()), [float(STRAINS[i]) for i in range(7) if not ok[i]], diag))
    fit = EOS("birch_murnaghan").fit(V[ok], Ef[ok])
    B0, V0, B0p = float(fit.b0_GPa), float(fit.v0), float(fit.b1)
    resid = float(np.sqrt(np.mean((fit.func(V[ok]) - Ef[ok]) ** 2)))
    inwin = bool(V[ok].min() < V0 < V[ok].max())
    diag.update(B0_GPa=B0, V0=V0, B0prime=B0p, n_ok=int(ok.sum()),
                rms_resid_eV=resid, inwindow=inwin, V0_offset_pct=100 * (V0 - rel.volume) / rel.volume)
    return diag

def main():
    cif = sys.argv[1]
    spacing = float(sys.argv[2]) if len(sys.argv) > 2 else 0.16   # conv-locked KSPACING
    stem = os.path.splitext(os.path.basename(cif))[0]
    base = os.path.abspath("eos_" + stem); os.makedirs(base, exist_ok=True)
    s, isym, info = prep_cell(cif)
    print("%s\n  prep: want_sg=%s used=%s symprec=%s sg_final=%s nat=%d -> ISYM=%d ISPIN=2, KSPACING=%.3f"
          % (stem, info["want"], info["used"], info["symprec"], info["sg_final"], info["nat"], isym, spacing), flush=True)
    rel, magmoms = relax(s, os.path.join(base, "relax"), isym, spacing)
    mtot = None if magmoms is None else sum(abs(m) for m in magmoms)
    print("  relaxed: V=%.3f  a=%.4f b=%.4f c=%.4f  |M|tot=%s" % (rel.volume, *rel.lattice.abc,
          "%.2f" % mtot if mtot is not None else "n/a"), flush=True)
    # Cp/phonon handoff: persist the relaxed cell + converged moments so the (separate, ISYM=0)
    # phonon stage starts from this geometry and seeds ISPIN=2 with the SAME moments. The full
    # relax/ dir (CONTCAR+OUTCAR) is also retained.
    rel.to(fmt="poscar", filename=os.path.join(base, "relaxed_POSCAR.vasp"))
    d = eos_b0(rel, base, isym, spacing, magmoms=magmoms)
    res = dict(stem=stem, isym=isym, ispin=2, spacing=spacing,
               magmoms=magmoms, relaxed_volume=rel.volume, relaxed_abc=list(rel.lattice.abc),
               relaxed_poscar="relaxed_POSCAR.vasp", relax_dir="relax", **info, **d)
    with open(os.path.join(base, "K0.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    flag = "" if d["inwindow"] else "  !! V0 OUTSIDE strain window — suspect (re-relax)"
    print("  K0 = %.1f GPa  V0=%.2f (in-window=%s, off=%.2f%%)  B0'=%.2f  resid=%.1e eV  [%s %s]%s"
          % (d["B0_GPa"], d["V0"], d["inwindow"], d["V0_offset_pct"], d["B0prime"], d["rms_resid_eV"],
             info["used"], d["smearing"], flag), flush=True)
    print("DONE %s" % stem, flush=True)

if __name__ == "__main__":
    main()
