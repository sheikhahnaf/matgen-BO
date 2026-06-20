"""LocalESEN — drop-in replacement for upstream `rewards.calculators.FairChem`.

Signature compatibility:
    upstream:  FairChem(root_dir, task='heat_capacity', env_name='fair-chem-v1', worker=5)
    ours:      LocalESEN(root_dir, task='heat_capacity', env_name=None, worker=1)

Behavioural difference:
    upstream invokes a subprocess `conda run -n fair-chem-v1 python phonon.py ...`
    that runs eSEN-30M-OAM via quacc inside a SEPARATE conda env.

    LocalESEN runs eSEN-30M-OAM IN-PROCESS using fairchem-core 1.10's
    OCPCalculator + phonopy directly. This is faster (no subprocess startup,
    no fork-and-conda-init overhead) and avoids the broken quacc/fairchem
    API combination we discovered.

The .calc() return value is identical: an np.ndarray of Cp(300 K) in J/g/K,
with NaN for structures whose phonon flow failed.

Used by Phase 1 to replicate MatInvent's exact Cp RL run with our env.
Phase 2 / Phase 3 then add the GP-routed σ-threshold on top of this same
calculator.
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from ase.io import read as ase_read
from pymatgen.core.structure import Structure


class LocalESEN:
    """In-process eSEN-30M-OAM heat-capacity oracle (matches upstream FairChem API)."""

    def __init__(
        self,
        root_dir: str,
        task: str = "heat_capacity",
        env_name: str | None = None,   # ignored — we run in-process
        worker: int = 1,                # currently single-process
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "heat_capacity":
            raise ValueError(
                f"LocalESEN currently only supports task='heat_capacity', got {task!r}"
            )
        self.worker = int(worker)
        # Cache the OCPCalculator load on first .calc() call (heavy: ~15 s + GPU mem).
        self._calc = None

    # ----- model load ----------------------------------------------------

    def _load_calculator(self):
        if self._calc is None:
            import torch
            from fairchem.core import OCPCalculator
            local_cache = os.environ.get(
                "FAIRCHEM_LOCAL_CACHE",
                os.path.join(os.environ.get("SCRATCH", "/tmp"), ".cache", "huggingface"),
            )
            self._calc = OCPCalculator(
                model_name="eSEN-30M-OAM",
                local_cache=local_cache,
                cpu=not torch.cuda.is_available(),
            )
        return self._calc

    # ----- main entrypoint -----------------------------------------------

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray:
        """Compute Cp@300K for each structure.

        Args:
            samples: tuple (List[pymatgen.Structure], xyz_path).
                We use the xyz_path (faithful to upstream pattern) — it's an
                extxyz file already written by the caller (`pipeline/mat_invent.py`).
            label: subdirectory tag for any per-call output (currently unused
                since we don't write intermediate files — kept for signature
                compatibility).

        Returns:
            np.ndarray of shape (N,) with Cp values in J/g/K. NaN where the
            phonon flow failed.
        """
        xyz_path = samples[1] if isinstance(samples, (tuple, list)) and len(samples) >= 2 else None
        if xyz_path is None or not os.path.isfile(xyz_path):
            # Fall back to converting samples[0] (List[Structure]) directly.
            from pymatgen.io.ase import AseAtomsAdaptor
            adaptor = AseAtomsAdaptor()
            atoms_list = [adaptor.get_atoms(s) for s in samples[0]]
        else:
            atoms_list = ase_read(xyz_path, index=":")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

        out = np.empty(len(atoms_list), dtype=np.float64)
        for i, atoms in enumerate(atoms_list):
            try:
                out[i] = self._phonon_task(atoms)
            except Exception as e:
                print(f"[LocalESEN] task {i} failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
                out[i] = np.nan
        return out

    # ----- per-structure phonon flow -------------------------------------

    def _phonon_task(self, atoms) -> float:
        import gc
        import torch
        from ase.optimize import FIRE
        try:
            from ase.filters import ExpCellFilter
        except ImportError:
            from ase.constraints import ExpCellFilter
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from pymatgen.io.ase import AseAtomsAdaptor

        calc = self._load_calculator()

        # 1. Relax cell + positions (FIRE, fmax=0.01).
        a = atoms.copy()
        a.calc = calc
        try:
            opt = FIRE(ExpCellFilter(a), logfile=None)
            opt.run(fmax=0.01, steps=500)
        except Exception as e:
            print(f"[LocalESEN] relax failed: {e}", file=sys.stderr)
            return float("nan")

        # 2. Build phonopy supercell with min lengths >= 10 Å.
        cell = a.get_cell().array
        sc = np.diag([
            max(1, int(np.ceil(10.0 / np.linalg.norm(cell[0])))),
            max(1, int(np.ceil(10.0 / np.linalg.norm(cell[1])))),
            max(1, int(np.ceil(10.0 / np.linalg.norm(cell[2])))),
        ])
        unit = PhonopyAtoms(
            symbols=a.get_chemical_symbols(),
            scaled_positions=a.get_scaled_positions(),
            cell=cell,
        )
        phonon = Phonopy(unit, supercell_matrix=sc)
        phonon.generate_displacements(distance=0.01)

        # 3. Forces on each displaced supercell.
        forces = []
        for s in phonon.supercells_with_displacements:
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

        # 4. Mesh + thermal properties at 0..300K (10K steps).
        phonon.run_mesh([20, 20, 20])
        phonon.run_thermal_properties(t_step=10, t_max=300, t_min=0)
        tp = phonon.get_thermal_properties_dict()

        # 5. Reject imaginary modes.
        mesh = phonon.get_mesh_dict()
        if mesh is not None and (mesh["frequencies"] < -0.1).any():
            print("[LocalESEN] imaginary modes detected", file=sys.stderr)
            return float("nan")

        # 6. Cp@300K → J/g/K.
        temps = np.asarray(tp["temperatures"])
        idx = int(np.argmin(np.abs(temps - 300.0)))
        cv_unit = float(tp["heat_capacity"][idx])  # J/K/mol of unit cell
        struc = AseAtomsAdaptor.get_structure(atoms)
        cp_per_g = cv_unit / float(struc.composition.weight)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cp_per_g
