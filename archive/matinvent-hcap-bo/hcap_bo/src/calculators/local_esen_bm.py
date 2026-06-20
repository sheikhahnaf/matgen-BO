"""LocalESEN_BM — drop-in BM calculator (mirrors LocalESEN's API for bulk modulus).

Returns Voigt-Reuss-Hill-equivalent isotropic K (bulk modulus, GPa) computed
from a 3rd-order Birch-Murnaghan EOS fit on 7 isotropic-strain points around
the eSEN-relaxed equilibrium volume.

Compatible with `rewards.calculators.FairChem` API (constructor + .calc()).
Used for Phase-3 BM RL loop. Hydra:
    reward.prop_cfg.0.calculator._target_=src.calculators.LocalESEN_BM
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
from ase.io import read as ase_read
from pymatgen.core.structure import Structure


# eV/Å³ → GPa
EV_PER_A3_TO_GPA = 160.21766208


class LocalESEN_BM:
    """In-process eSEN-30M-OAM bulk modulus oracle.

    Workflow per structure:
        1. eSEN-relax cell+positions (FIRE, fmax=0.05)
        2. 7 isotropic-strain points: a → a*(1+ε)**(1/3), ε ∈ {-0.03, -0.02, ..., +0.03}
        3. ASE EquationOfState 3rd-order Birch-Murnaghan fit → V0, E0, B (eV/Å³)
        4. Return K = B in GPa (B*EV_PER_A3_TO_GPA)
    """

    def __init__(
        self,
        root_dir: str,
        task: str = "bulk_modulus",
        env_name: str | None = None,
        worker: int = 1,
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        os.makedirs(self.root_dir, exist_ok=True)
        if task != "bulk_modulus":
            raise ValueError(
                f"LocalESEN_BM only supports task='bulk_modulus', got {task!r}"
            )
        self.worker = int(worker)
        self._calc = None

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

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray:
        xyz_path = samples[1] if isinstance(samples, (tuple, list)) and len(samples) >= 2 else None
        if xyz_path is None or not os.path.isfile(xyz_path):
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
                out[i] = self._bm_task(atoms)
            except Exception as e:
                print(f"[LocalESEN_BM] task {i} failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
                out[i] = np.nan
        return out

    def _bm_task(self, atoms) -> float:
        """Per-structure relax + BM3-EOS bulk modulus."""
        import gc
        import torch
        from ase.optimize import FIRE
        try:
            from ase.filters import ExpCellFilter
        except ImportError:
            from ase.constraints import ExpCellFilter
        from ase.eos import EquationOfState

        calc = self._load_calculator()

        # 1. Relax cell + positions to V0
        a = atoms.copy()
        a.calc = calc
        try:
            opt = FIRE(ExpCellFilter(a), logfile=None)
            opt.run(fmax=0.05, steps=300)
        except Exception as e:
            print(f"[LocalESEN_BM] relax failed: {e}", file=sys.stderr)
            return float("nan")

        # 2. 7 isotropic strain points around relaxed cell
        strains = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
        cell0 = a.get_cell().array.copy()
        positions_frac0 = a.get_scaled_positions().copy()
        symbols = a.get_chemical_symbols()
        volumes, energies = [], []
        for eps in strains:
            scale = (1.0 + eps) ** (1.0 / 3.0)
            new_cell = cell0 * scale
            from ase import Atoms as ASEAtoms
            ax = ASEAtoms(
                symbols=symbols,
                scaled_positions=positions_frac0,
                cell=new_cell,
                pbc=True,
            )
            ax.calc = calc
            try:
                e = float(ax.get_potential_energy())
            except Exception as ex:
                print(f"[LocalESEN_BM] strain {eps:+.3f} energy failed: {ex}",
                      file=sys.stderr)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return float("nan")
            volumes.append(ax.get_volume())
            energies.append(e)

        # 3. Fit 3rd-order Birch-Murnaghan
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B_evA3 = eos.fit()
        except Exception as e:
            print(f"[LocalESEN_BM] EOS fit failed: {e}", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("nan")

        K_GPa = float(B_evA3) * EV_PER_A3_TO_GPA

        # Sanity: K must be positive and within physical range
        if not np.isfinite(K_GPa) or K_GPa < 1.0 or K_GPa > 800.0:
            print(f"[LocalESEN_BM] unphysical K={K_GPa:.2f} GPa, returning NaN",
                  file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float("nan")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return K_GPa
