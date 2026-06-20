"""ORB-v3 heat-capacity runner via phonopy.

Same input/output contract as run_esen.py:
    python runners/run_orb_phonon.py <xyz_in> <out_txt> <n_workers>
    -> Cp@300K (J/g/K), one per line, NaN on failure.

Pipeline per structure:
    1. Relax (cell + positions, FIRE, fmax=0.01) with ORBCalculator.
    2. Build phonopy supercell with min_lengths >= 10 Å.
    3. Generate displacements; compute forces on each via ORB.
    4. Phonopy assembles dynamical matrix, computes phonon DOS.
    5. Cp@300K = thermal_properties at T=300 K, divided by composition mass.
    6. Reject structures with imaginary modes (any negative frequency below
       threshold) — return NaN.
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
    # ase >= 3.23 keeps ExpCellFilter in ase.filters; older versions in ase.constraints.
    from ase.filters import ExpCellFilter
except ImportError:
    from ase.constraints import ExpCellFilter
from pymatgen.io.ase import AseAtomsAdaptor


_ORB_CALC = None  # singleton per process


def _load_orb():
    global _ORB_CALC
    if _ORB_CALC is None:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device, precision="float32-high"
        )
        _ORB_CALC = ORBCalculator(orbff, device=device)
    return _ORB_CALC


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
        print(f"[orb] phonopy import failed: {e}", file=sys.stderr)
        return np.nan

    try:
        calc = _load_orb()

        # 1. Relax (cell + positions). ExpCellFilter for cell relaxation.
        a = atoms.copy()
        a.calc = calc
        try:
            opt = FIRE(ExpCellFilter(a), logfile=None)
            opt.run(fmax=0.01, steps=500)
        except Exception as e:
            print(f"[orb] relax failed: {e}", file=sys.stderr)
            return np.nan

        # 2. Phonopy setup with minimum supercell length 10 Å.
        sc_matrix = _supercell_matrix_for_min_length(a.get_cell().array, 10.0)
        unit = PhonopyAtoms(
            symbols=a.get_chemical_symbols(),
            scaled_positions=a.get_scaled_positions(),
            cell=a.get_cell().array,
        )
        phonon = Phonopy(unit, supercell_matrix=sc_matrix)
        phonon.generate_displacements(distance=0.01)
        sc_atoms = phonon.supercells_with_displacements

        # 3. Forces on each displaced supercell.
        forces = []
        for s in sc_atoms:
            from ase import Atoms as ASEAtoms
            asx = ASEAtoms(
                symbols=s.symbols,
                scaled_positions=s.scaled_positions,
                cell=s.cell,
                pbc=True,
            )
            asx.calc = calc
            forces.append(asx.get_forces())
        phonon.forces = forces
        phonon.produce_force_constants()

        # 4. Mesh + thermal properties.
        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(t_step=10, t_max=300, t_min=0)
        tp = phonon.get_thermal_properties_dict()

        # 5. Reject imaginary modes (heuristic: any frequency below -0.1 THz).
        phonon.run_total_dos()
        freqs = phonon.get_total_dos_dict()["frequency_points"]  # THz
        # phonon.get_total_dos_dict() actually returns 'frequency_points'? safer:
        try:
            band = phonon.get_band_structure_dict()
        except Exception:
            band = None
        # Use mesh frequencies for imaginary check
        mesh = phonon.get_mesh_dict()
        if mesh is not None:
            mesh_freqs = mesh["frequencies"]  # in THz
            if (mesh_freqs < -0.1).any():
                print("[orb] imaginary modes detected", file=sys.stderr)
                return np.nan

        # 6. Cp at 300K (J/mol/K) -> J/g/K via composition weight.
        # heat_capacity from phonopy is in J/K/mol of the unit cell.
        idx = int(np.argmin(np.abs(tp["temperatures"] - 300.0)))
        cv_unit = float(tp["heat_capacity"][idx])  # J/K/mol of unit cell
        struc = AseAtomsAdaptor.get_structure(atoms)
        mass_g_per_mol = float(struc.composition.weight)
        cp_per_g = cv_unit / mass_g_per_mol

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cp_per_g

    except Exception as e:
        print(f"[orb] failed: {type(e).__name__}: {e}", file=sys.stderr)
        return np.nan


def main():
    if len(sys.argv) != 4:
        print("usage: run_orb_phonon.py <xyz_in> <out_txt> <n_workers>", file=sys.stderr)
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
