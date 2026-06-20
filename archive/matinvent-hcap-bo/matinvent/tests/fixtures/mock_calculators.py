"""Mock calculators for testing."""

import numpy as np
import pytest
from typing import List, Tuple
from pymatgen.core import Structure


class MockCalculator:
    """Mock calculator that returns deterministic fake values."""

    def __init__(self, root_dir: str, task: str, mean_value: float = 150.0, std_value: float = 20.0):
        self.root_dir = root_dir
        self.task = task
        self.mean_value = mean_value
        self.std_value = std_value
        self.call_count = 0

    def calc(self, samples: Tuple[List[Structure], str], label: str = 'tmp') -> np.ndarray:
        """Return mock property values."""
        structures, _ = samples
        n = len(structures)
        self.call_count += 1

        # Deterministic but varied values
        np.random.seed(hash(label) % 2**32)
        values = np.random.normal(self.mean_value, self.std_value, n)
        return values


class MockCheapCalculator(MockCalculator):
    """Mock cheap calculator (e.g., ORB)."""

    def __init__(self, root_dir: str, task: str):
        super().__init__(root_dir, task, mean_value=150.0, std_value=30.0)


class MockExpensiveCalculator(MockCalculator):
    """Mock expensive calculator (e.g., VASP)."""

    def __init__(self, root_dir: str, task: str):
        super().__init__(root_dir, task, mean_value=155.0, std_value=10.0)


@pytest.fixture
def mock_orb_calculator(tmp_path):
    """Mock ORB calculator."""
    return MockCheapCalculator(str(tmp_path / "orb"), "bulk_modulus")


@pytest.fixture
def mock_alignn_calculator(tmp_path):
    """Mock ALIGNN calculator."""
    return MockCalculator(str(tmp_path / "alignn"), "bulk_modulus", mean_value=152.0, std_value=15.0)


@pytest.fixture
def mock_vasp_calculator(tmp_path):
    """Mock VASP calculator."""
    return MockExpensiveCalculator(str(tmp_path / "vasp"), "bulk_modulus")


@pytest.fixture
def mock_calculator_dict(mock_orb_calculator, mock_alignn_calculator, mock_vasp_calculator):
    """Dictionary of mock calculators."""
    return {
        'orb': mock_orb_calculator,
        'alignn': mock_alignn_calculator,
        'vasp': mock_vasp_calculator
    }


@pytest.fixture
def mock_cost_model():
    """Mock cost model for acquisition function."""
    return {
        'orb': 0.001,
        'alignn': 0.01,
        'vasp': 1.0
    }
