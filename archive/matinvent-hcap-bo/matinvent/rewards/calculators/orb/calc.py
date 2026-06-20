"""ORB Calculator for direct property prediction.

Uses ORB (Orbital Materials) model for fast property predictions via MaterialsFramework.
Serves as cheap calculator in GP-routing system (cost: 0.001).
"""

import os
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.base import Calculator


class ORBCalculator(Calculator):
    """
    ORB-based property calculator using MaterialsFramework.

    Uses ORB model for direct property prediction via elastic constants
    and formation energy calculations. This is the cheapest calculator
    option (cost: 0.001) and is used for initial screening before
    expensive calculators like VASP.

    Supported tasks:
        - bulk_modulus: Bulk modulus in GPa (from elastic constants)
        - shear_modulus: Shear modulus in GPa (from elastic constants)
        - formation_energy: Formation energy per atom in eV
        - band_gap: Band gap in eV (placeholder - not supported by ORB)

    Note: Uses MaterialsFramework's ORBCalculator and analyzers for
    proper stress-strain calculations.
    """

    def __init__(
        self,
        root_dir: str,
        task: str,
        device: str = 'cpu',
        fmax: float = 0.05,
        model: str = 'orb-v2',
        **kwargs
    ):
        """
        Initialize ORB Calculator.

        Args:
            root_dir: Directory to save results
            task: Property to calculate (bulk_modulus, shear_modulus, formation_energy)
            device: Device to run on ('cpu' or 'cuda')
            fmax: Force convergence criterion (eV/Å)
            model: ORB model version ('orb-v2' recommended)
        """
        super().__init__(root_dir, task)
        self.device = device
        self.fmax = fmax
        self.model = model

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:
        """
        Calculate property using ORB model via MaterialsFramework.

        Args:
            samples: Tuple of (structures, xyz_path)
                - structures: List of pymatgen Structure objects
                - xyz_path: Path to XYZ file (unused for ORB)
            label: Label for saving results

        Returns:
            np.ndarray: Property values (one per structure)
        """
        structures, _ = samples

        # Calculate property for each structure
        properties = []
        for i, structure in enumerate(structures):
            try:
                prop_value = self._calc_single_structure(structure)
                properties.append(prop_value)
                print(f"  Structure {i+1}/{len(structures)}: {self.task} = {prop_value:.4f}")
            except Exception as e:
                print(f"  Warning: ORB calculation failed for structure {i+1}: {str(e)[:100]}")
                properties.append(np.nan)

        properties = np.array(properties, dtype=float)

        # Save results
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        out_path = os.path.abspath(out_path)
        np.savetxt(out_path, properties, fmt="%.6f")

        return properties

    def _calc_single_structure(self, structure: Structure) -> float:
        """
        Calculate property for a single structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            float: Property value
        """
        # Import MaterialsFramework components
        from materialsframework.calculators import ORBCalculator as MFORBCalculator
        from materialsframework.analysis import (
            CubicElasticConstantsAnalyzer,
            ElasticConstantsAnalyzer,
            FormationEnergyAnalyzer
        )

        # Convert pymatgen Structure to ASE Atoms
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)

        if self.task in ['bulk_modulus', 'shear_modulus']:
            # Initialize ORB calculator
            calc = MFORBCalculator(
                fmax=self.fmax,
                model=self.model,
                relax_cell=True,
                verbose=False
            )

            # Relax structure first
            relax_result = calc.relax(atoms)
            relaxed_structure = relax_result["final_structure"]

            # Check if structure is cubic
            is_cubic = self._is_cubic(structure)

            if is_cubic:
                # Use CubicElasticConstantsAnalyzer (faster, more accurate)
                elastic = CubicElasticConstantsAnalyzer(calculator=calc)
                elastic_result = elastic.calculate(
                    relaxed_structure,
                    is_relaxed=True,
                    delta_max=0.05,
                    step_size=0.01
                )
            else:
                # Use general ElasticConstantsAnalyzer
                elastic = ElasticConstantsAnalyzer(calculator=calc)
                elastic_result = elastic.calculate(
                    relaxed_structure,
                    is_relaxed=True
                )

            # Extract requested property
            if self.task == 'bulk_modulus':
                return elastic_result['bulk_modulus']  # K_VRH in GPa
            elif self.task == 'shear_modulus':
                return elastic_result['shear_modulus']  # G_VRH in GPa

        elif self.task == 'formation_energy':
            # Initialize ORB calculator
            calc = MFORBCalculator(
                fmax=self.fmax,
                model=self.model,
                relax_cell=True,
                verbose=False
            )

            # Calculate formation energy
            formation = FormationEnergyAnalyzer(calculator=calc)
            formation_result = formation.calculate(structure)

            return formation_result['formation_energy']  # eV/atom

        elif self.task == 'band_gap':
            # Band gap not directly supported by ORB
            # Return NaN as placeholder
            print("  Warning: Band gap calculation not supported by ORB. Returning NaN.")
            return np.nan

        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _is_cubic(self, structure: Structure) -> bool:
        """
        Check if structure has cubic symmetry.

        Args:
            structure: pymatgen Structure

        Returns:
            bool: True if cubic, False otherwise
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            crystal_system = sga.get_crystal_system()
            return crystal_system == 'cubic'
        except:
            # If symmetry analysis fails, assume non-cubic
            return False
