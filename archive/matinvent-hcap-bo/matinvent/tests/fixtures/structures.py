"""Test fixtures for crystal structures."""

import numpy as np
import pytest
from pymatgen.core import Structure, Lattice


@pytest.fixture
def simple_cubic_structure():
    """Simple cubic structure (e.g., Po)."""
    lattice = Lattice.cubic(3.0)
    structure = Structure(lattice, ["Po"], [[0, 0, 0]])
    return structure


@pytest.fixture
def fcc_structure():
    """FCC structure (e.g., Cu)."""
    lattice = Lattice.cubic(3.6)
    coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    structure = Structure(lattice, ["Cu"] * 4, coords)
    return structure


@pytest.fixture
def bcc_structure():
    """BCC structure (e.g., Fe)."""
    lattice = Lattice.cubic(2.87)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = Structure(lattice, ["Fe"] * 2, coords)
    return structure


@pytest.fixture
def hcp_structure():
    """HCP structure (e.g., Mg)."""
    a = 3.2
    c = 5.2
    lattice = Lattice.hexagonal(a, c)
    coords = [[1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
    structure = Structure(lattice, ["Mg"] * 2, coords)
    return structure


@pytest.fixture
def rocksalt_structure():
    """Rock salt structure (e.g., NaCl)."""
    lattice = Lattice.cubic(5.64)
    coords = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],  # Na
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]   # Cl
    ]
    species = ["Na"] * 4 + ["Cl"] * 4
    structure = Structure(lattice, species, coords)
    return structure


@pytest.fixture
def perovskite_structure():
    """Perovskite structure (e.g., BaTiO3)."""
    lattice = Lattice.cubic(4.0)
    coords = [
        [0, 0, 0],      # Ba
        [0.5, 0.5, 0.5],  # Ti
        [0.5, 0.5, 0],    # O
        [0.5, 0, 0.5],    # O
        [0, 0.5, 0.5]     # O
    ]
    species = ["Ba", "Ti", "O", "O", "O"]
    structure = Structure(lattice, species, coords)
    return structure


@pytest.fixture
def structure_list(fcc_structure, bcc_structure, hcp_structure):
    """List of diverse structures for batch testing."""
    return [fcc_structure, bcc_structure, hcp_structure]


@pytest.fixture
def xyz_file_path(tmp_path, structure_list):
    """Create temporary XYZ file for testing."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write

    adaptor = AseAtomsAdaptor()
    atoms_list = [adaptor.get_atoms(s) for s in structure_list]

    xyz_path = tmp_path / "test_structures.extxyz"
    write(str(xyz_path), atoms_list)
    return str(xyz_path)


@pytest.fixture
def mock_rewards():
    """Mock reward values for testing."""
    return np.array([0.8, 0.6, 0.9, 0.7, 0.5])


@pytest.fixture
def mock_property_values():
    """Mock property values (e.g., bulk modulus in GPa)."""
    return np.array([150.0, 180.0, 200.0, 120.0, 160.0])
