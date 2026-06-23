#!/usr/bin/env python
"""Phonon / dynamical-stability / Cv(T) driver — finite displacement (phonopy + VASP).

For AI structures the imaginary-mode check is the dynamical-stability verdict, and Cv(300 K) is the
heat-capacity target. Used both for the small-scale physicality discovery (reference Si + supercell
convergence) and the production campaign, via the supercell min-length argument.

KEY physics:
  * ISYM=0 on every displaced supercell (the displacement breaks symmetry; phonopy does its own
    symmetry reduction and force-constant symmetrization — letting VASP symmetrize can average away
    a soft/imaginary mode, the exact signal we want).
  * PREC=Accurate + EDIFF=1e-7 (force constants need accurate forces).
  * ENCUT=680; gap-agnostic ISMEAR=0/SIGMA=0.05; ISPIN=2 with per-element MAGMOM seeded from the
    EOS-relaxed moments (consistent magnetic state; no-op for non-magnetic systems).
  * supercell k-mesh from a fixed reciprocal density (Gamma-centered) — auto-reduces as the
    supercell grows, keeping resolution roughly constant.
  * Cv normalized PER GRAM (J/g/K): basis-independent, matches the oracle, dodges the per-formula-
    unit factor-Z trap.

Args: <relaxed_POSCAR_or_CIF> [supercell_min_length_Angstrom=15] [magmoms_json=K0.json]
"""
import os, sys, json, numpy as np
from collections import defaultdict
from pymatgen.core import Structure, Lattice
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Vasprun
import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
# reuse the validated custodian/VASP machinery
from campaign_eos import VASP, HANDLERS, VALIDATORS, POT, cust, assert_no_kspacing, static_done, grid_for
from custodian import Custodian
from custodian.vasp.jobs import VaspJob

KSP_SUPER = 0.25       # reciprocal density on the supercell (coarser than primitive; BZ is folded)
QMESH = [24, 24, 24]   # q-mesh for DOS / thermal integration
DISP = 0.01            # displacement amplitude (A)
IMAG_TOL = -0.05       # THz: frequencies below this count as imaginary (acoustic-at-Gamma excluded)

PH_INCAR = dict(ENCUT=680, ISYM=0, ISPIN=2, PREC="Accurate", IBRION=-1, NSW=0,
                ISMEAR=0, SIGMA=0.05, EDIFF=1e-7, ALGO="Normal", LREAL=False,
                LWAVE=False, LCHARG=False, LASPH=True, NCORE=4)

def to_phonopy(s):
    return PhonopyAtoms(symbols=[sp.symbol for sp in s.species],
                        scaled_positions=s.frac_coords, cell=s.lattice.matrix)

def to_pmg(ph_atoms):
    return Structure(Lattice(ph_atoms.cell), list(ph_atoms.symbols), ph_atoms.scaled_positions)

def supercell_matrix(s, min_len):
    return np.diag([max(1, int(np.ceil(min_len / a))) for a in s.lattice.abc]).tolist()

def elem_magmoms(struct, magmoms):
    """Per-element average moment from the EOS-relaxed cell (None if non-magnetic)."""
    if not magmoms:
        return None
    acc = defaultdict(list)
    for sp, m in zip(struct.species, magmoms):
        acc[sp.symbol].append(m)
    d = {e: float(np.mean(v)) for e, v in acc.items()}
    return d if any(abs(x) > 0.05 for x in d.values()) else None

def read_forces(d):
    try:
        vr = Vasprun(os.path.join(d, "vasprun.xml"), parse_dos=False, parse_eigen=False,
                     parse_potcar_file=False)
        if not vr.converged_electronic:
            return None
        return np.array(vr.ionic_steps[-1]["forces"])
    except Exception:
        return None

def main():
    poscar = sys.argv[1]
    min_len = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    magjson = sys.argv[3] if len(sys.argv) > 3 else None
    stem = os.path.splitext(os.path.basename(poscar))[0]
    base = os.path.abspath("ph_%s_L%d" % (stem, round(min_len))); os.makedirs(base, exist_ok=True)
    s = Structure.from_file(poscar)
    magmoms = None
    if magjson and os.path.exists(magjson):
        try:
            magmoms = json.load(open(magjson)).get("magmoms")
        except Exception:
            magmoms = None
    emag = elem_magmoms(s, magmoms)

    smat = supercell_matrix(s, min_len)
    phonon = Phonopy(to_phonopy(s), supercell_matrix=smat, primitive_matrix="auto")
    phonon.generate_displacements(distance=DISP)
    disps = phonon.supercells_with_displacements
    n_super = len(phonon.supercell)
    n_k = grid_for(to_pmg(phonon.supercell), KSP_SUPER)
    print("%s\n  supercell=%s  natoms_super=%d  n_displacements=%d  k=%s  magnetic=%s"
          % (stem, [smat[0][0], smat[1][1], smat[2][2]], n_super, len(disps), n_k, bool(emag)), flush=True)

    forces, failed = [], []
    for i, sc in enumerate(disps):
        d = os.path.join(base, "disp-%03d" % i)
        if not static_done(d):
            os.makedirs(d, exist_ok=True)
            st = to_pmg(sc)
            inc = dict(PH_INCAR)
            if emag:
                inc["MAGMOM"] = [emag.get(sp.symbol, 0.0) for sp in st.species]
            MPStaticSet(st, user_incar_settings=inc, **POT).write_input(d)
            Kpoints.gamma_automatic(n_k).write_file(os.path.join(d, "KPOINTS")); assert_no_kspacing(d)
        cust(d)
        f = read_forces(d)
        if f is None:
            failed.append(i)
        forces.append(f)
        print("  disp %3d/%d  %s" % (i + 1, len(disps), "ok" if f is not None else "FAILED"), flush=True)

    if failed:
        json.dump(dict(stem=stem, supercell=smat, n_disp=len(disps), failed=failed),
                  open(os.path.join(base, "phonon.json"), "w"), indent=2)
        raise RuntimeError("%s: %d/%d displacement force calcs failed (idx=%s)"
                           % (stem, len(failed), len(disps), failed))

    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    phonon.run_mesh(QMESH)
    freqs = phonon.get_mesh_dict()["frequencies"]            # THz
    min_freq = float(np.min(freqs))
    n_imag = int(np.sum(freqs < IMAG_TOL))
    frac_imag = float(np.mean(freqs < IMAG_TOL))
    phonon.run_thermal_properties(t_min=300, t_max=300, t_step=1)
    tp = phonon.get_thermal_properties_dict()
    cv_mol = float(tp["heat_capacity"][0])                   # J/K/mol-primitive
    prim_mass = float(np.sum(phonon.primitive.masses))       # g/mol-primitive
    cv_pergram = cv_mol / prim_mass                          # J/g/K  (intensive, oracle-consistent)

    res = dict(stem=stem, supercell=[smat[0][0], smat[1][1], smat[2][2]], min_len=min_len,
               natoms_super=n_super, n_displacements=len(disps), kgrid_super=n_k, qmesh=QMESH,
               cv300_J_per_gK=cv_pergram, cv300_J_per_molK=cv_mol, prim_mass_g=prim_mass,
               min_freq_THz=min_freq, n_imaginary=n_imag, frac_imaginary=frac_imag,
               dynamically_stable=bool(min_freq >= IMAG_TOL))
    json.dump(res, open(os.path.join(base, "phonon.json"), "w"), indent=2)
    print("  Cv(300K) = %.4f J/g/K  (%.2f J/K/mol)  | min_freq = %.3f THz  imaginary=%d (%.1f%%)  -> %s"
          % (cv_pergram, cv_mol, min_freq, n_imag, 100 * frac_imag,
             "STABLE" if res["dynamically_stable"] else "UNSTABLE (imaginary modes)"), flush=True)
    print("DONE %s" % stem, flush=True)

if __name__ == "__main__":
    main()
