"""UMA (FAIRChem uma-s-1p1, task=omat) heat-capacity runner via phonopy.

Same I/O contract as run_orb_phonon.py and run_esen.py.

Pipeline mirrors run_orb_phonon exactly; only the calculator changes.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import sys

import numpy as np
import torch
from ase.io import read
from ase.optimize import FIRE
try:
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from pymatgen.io.ase import AseAtomsAdaptor


_UMA_PRED = None
_UMA_CALC = None


def _load_uma():
    global _UMA_PRED, _UMA_CALC
    if _UMA_CALC is None:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            _UMA_PRED = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
            _UMA_CALC = FAIRChemCalculator(_UMA_PRED, task_name="omat")
        finally:
            torch.set_default_dtype(prev)
    return _UMA_CALC


def _supercell_matrix_for_min_length(cell, min_len: float = 10.0) -> np.ndarray:
    a, b, c = np.linalg.norm(cell, axis=1)
    return np.diag([
        max(1, int(np.ceil(min_len / a))),
        max(1, int(np.ceil(min_len / b))),
        max(1, int(np.ceil(min_len / c))),
    ])


def _phonon_task(atoms):
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except Exception as e:
        print(f"[uma] phonopy import failed: {e}", file=sys.stderr)
        return np.nan

    try:
        calc = _load_uma()

        a = atoms.copy()
        a.set_positions(a.get_positions().astype(np.float32))
        a.set_cell(a.get_cell().array.astype(np.float32))
        a.calc = calc
        try:
            opt = FIRE(ExpCellFilter(a), logfile=None)
            opt.run(fmax=0.01, steps=500)
        except Exception as e:
            print(f"[uma] relax failed: {e}", file=sys.stderr)
            return np.nan

        sc_matrix = _supercell_matrix_for_min_length(a.get_cell().array, 10.0)
        unit = PhonopyAtoms(
            symbols=a.get_chemical_symbols(),
            scaled_positions=a.get_scaled_positions(),
            cell=a.get_cell().array,
        )
        phonon = Phonopy(unit, supercell_matrix=sc_matrix)
        phonon.generate_displacements(distance=0.01)
        sc_atoms = phonon.supercells_with_displacements

        forces = []
        for s in sc_atoms:
            from ase import Atoms as ASEAtoms
            asx = ASEAtoms(
                symbols=s.symbols,
                scaled_positions=s.scaled_positions,
                cell=s.cell,
                pbc=True,
            )
            asx.set_positions(asx.get_positions().astype(np.float32))
            asx.calc = calc
            forces.append(asx.get_forces())
        phonon.forces = forces
        phonon.produce_force_constants()

        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(t_step=10, t_max=300, t_min=0)
        tp = phonon.get_thermal_properties_dict()

        mesh = phonon.get_mesh_dict()
        if mesh is not None:
            if (mesh["frequencies"] < -0.1).any():
                print("[uma] imaginary modes detected", file=sys.stderr)
                return np.nan

        idx = int(np.argmin(np.abs(tp["temperatures"] - 300.0)))
        cv_unit = float(tp["heat_capacity"][idx])
        struc = AseAtomsAdaptor.get_structure(atoms)
        cp_per_g = cv_unit / float(struc.composition.weight)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cp_per_g

    except Exception as e:
        print(f"[uma] failed: {type(e).__name__}: {e}", file=sys.stderr)
        return np.nan


def main():
    if len(sys.argv) != 4:
        print("usage: run_uma_phonon.py <xyz_in> <out_txt> <n_workers>", file=sys.stderr)
        sys.exit(2)
    xyz_in, out_txt, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    atoms_list = read(xyz_in, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    if n_workers <= 1:
        results = [_phonon_task(a) for a in atoms_list]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_phonon_task, atoms_list)

    np.savetxt(out_txt, np.array(results, dtype=np.float64), fmt="%.6f")


if __name__ == "__main__":
    main()
