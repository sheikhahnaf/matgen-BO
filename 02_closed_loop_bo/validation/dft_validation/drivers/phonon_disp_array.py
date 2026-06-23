#!/usr/bin/env python
"""Compute ONE phonon displacement (by index) for SLURM job-array parallelization.

Reuses campaign_phonon's EXACT displacement generation (same structure, supercell, distance,
primitive_matrix='auto'), so the disp-<idx> this writes is byte-for-byte the one the serial
driver would produce at that index. The order is deterministic in phonopy given identical input,
so array workers and the gather agree on which disp is which.

Workflow: launch this as `--array=0-(N-1)` to compute all displacements in parallel; then run the
normal campaign_phonon.py as a gather step (every disp is static_done -> it skips compute via the
idempotent cust(), reads forces, and writes phonon.json with Cv + imaginary-mode verdict).

Usage: phonon_disp_array.py <relaxed_POSCAR> <supercell_min_length_Angstrom> <magmoms_json|none> <disp_index>
"""
import os, sys, json
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.vasp.inputs import Kpoints
from phonopy import Phonopy
# identical displacement/INCAR/k-mesh machinery as the serial driver
import campaign_phonon as cp
from campaign_eos import POT, cust, assert_no_kspacing, grid_for, static_done


def main():
    poscar = sys.argv[1]
    min_len = float(sys.argv[2])
    magjson = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] not in ("", "none", "None") else None
    idx = int(sys.argv[4])

    stem = os.path.splitext(os.path.basename(poscar))[0]
    base = os.path.abspath("ph_%s_L%d" % (stem, round(min_len)))
    os.makedirs(base, exist_ok=True)

    s = Structure.from_file(poscar)
    magmoms = None
    if magjson and os.path.exists(magjson):
        try:
            magmoms = json.load(open(magjson)).get("magmoms")
        except Exception:
            magmoms = None
    emag = cp.elem_magmoms(s, magmoms)

    smat = cp.supercell_matrix(s, min_len)
    phonon = Phonopy(cp.to_phonopy(s), supercell_matrix=smat, primitive_matrix="auto")
    phonon.generate_displacements(distance=cp.DISP)
    disps = phonon.supercells_with_displacements
    if idx >= len(disps):
        print("idx %d >= n_disp %d; nothing to do" % (idx, len(disps)), flush=True)
        return
    n_k = grid_for(cp.to_pmg(phonon.supercell), cp.KSP_SUPER)

    d = os.path.join(base, "disp-%03d" % idx)
    if static_done(d):
        print("disp-%03d already converged; skip" % idx, flush=True)
        return

    os.makedirs(d, exist_ok=True)
    st = cp.to_pmg(disps[idx])
    inc = dict(cp.PH_INCAR)
    if emag:
        inc["MAGMOM"] = [emag.get(sp.symbol, 0.0) for sp in st.species]
    MPStaticSet(st, user_incar_settings=inc, **POT).write_input(d)
    Kpoints.gamma_automatic(n_k).write_file(os.path.join(d, "KPOINTS"))
    assert_no_kspacing(d)
    cust(d)

    ok = static_done(d)
    print("disp-%03d %s" % (idx, "ok" if ok else "FAILED"), flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
