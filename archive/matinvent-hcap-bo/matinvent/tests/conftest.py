"""Pytest configuration and shared fixtures."""

import sys
import os
import pytest
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import fixtures
from tests.fixtures.structures import *
from tests.fixtures.mock_calculators import *


# Configure pytest
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return 'cpu'


@pytest.fixture
def seed():
    """Random seed for reproducibility."""
    np.random.seed(42)
    return 42
