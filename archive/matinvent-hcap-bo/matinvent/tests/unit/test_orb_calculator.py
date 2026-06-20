"""Unit tests for ORB calculator.

Note: These tests use mock ORB calculations to avoid heavy computations.
For full integration tests with real ORB, see integration tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pymatgen.core import Structure

from rewards.calculators.ORBCalculator import ORBCalculator


@pytest.fixture
def orb_calculator(tmp_path, device):
    """Create ORB calculator instance."""
    return ORBCalculator(
        root_dir=str(tmp_path / "orb_test"),
        task='bulk_modulus',
        device=device,
        fmax=0.05,
        model='orb-v2'
    )


def test_orb_calculator_initialization(orb_calculator, tmp_path):
    """Test ORB calculator initializes correctly."""
    assert orb_calculator.task == 'bulk_modulus'
    assert orb_calculator.fmax == 0.05
    assert orb_calculator.model == 'orb-v2'
    assert str(tmp_path / "orb_test") in orb_calculator.root_dir


def test_orb_calculator_supported_tasks(tmp_path, device):
    """Test ORB calculator supports required tasks."""
    supported_tasks = ['bulk_modulus', 'shear_modulus', 'formation_energy']

    for task in supported_tasks:
        calc = ORBCalculator(
            root_dir=str(tmp_path / f"orb_{task}"),
            task=task,
            device=device
        )
        assert calc.task == task


def test_orb_calculator_band_gap_not_supported(tmp_path, device):
    """Test that band gap task returns None (not supported by elastic constants)."""
    calc = ORBCalculator(
        root_dir=str(tmp_path / "orb_bandgap"),
        task='band_gap',
        device=device
    )

    # Band gap should raise NotImplementedError or return None
    from tests.fixtures.structures import fcc_structure
    structure = fcc_structure()

    with pytest.raises((NotImplementedError, ValueError)):
        calc._calc_single_structure(structure)


@patch('rewards.calculators.ORBCalculator.MFORBCalculator')
@patch('rewards.calculators.ORBCalculator.CubicElasticConstantsAnalyzer')
def test_calc_bulk_modulus_cubic(mock_elastic, mock_mf_orb, orb_calculator, fcc_structure):
    """Test bulk modulus calculation for cubic structure."""
    # Mock the MaterialsFramework calculator
    mock_calc_instance = Mock()
    mock_mf_orb.return_value = mock_calc_instance

    # Mock relaxation result
    mock_relax_result = {
        'final_structure': fcc_structure,
        'energy': -10.5,
        'converged': True
    }
    mock_calc_instance.relax.return_value = mock_relax_result

    # Mock elastic constants result
    mock_elastic_instance = Mock()
    mock_elastic.return_value = mock_elastic_instance
    mock_elastic_result = {
        'bulk_modulus': 150.0,  # K_VRH in GPa
        'shear_modulus': 80.0,  # G_VRH in GPa
        'C11': 200.0,
        'C12': 120.0,
        'C44': 90.0
    }
    mock_elastic_instance.calculate.return_value = mock_elastic_result

    # Test calculation
    result = orb_calculator._calc_single_structure(fcc_structure)

    assert result == 150.0
    mock_calc_instance.relax.assert_called_once()
    mock_elastic_instance.calculate.assert_called_once()


@patch('rewards.calculators.ORBCalculator.MFORBCalculator')
@patch('rewards.calculators.ORBCalculator.ElasticConstantsAnalyzer')
def test_calc_bulk_modulus_non_cubic(mock_elastic, mock_mf_orb, orb_calculator, hcp_structure):
    """Test bulk modulus calculation for non-cubic structure."""
    # Mock setup
    mock_calc_instance = Mock()
    mock_mf_orb.return_value = mock_calc_instance

    mock_relax_result = {
        'final_structure': hcp_structure,
        'energy': -8.3,
        'converged': True
    }
    mock_calc_instance.relax.return_value = mock_relax_result

    # Mock elastic constants for non-cubic
    mock_elastic_instance = Mock()
    mock_elastic.return_value = mock_elastic_instance
    mock_elastic_result = {
        'bulk_modulus': 160.0,
        'shear_modulus': 70.0
    }
    mock_elastic_instance.calculate.return_value = mock_elastic_result

    result = orb_calculator._calc_single_structure(hcp_structure)

    assert result == 160.0
    # Should use ElasticConstantsAnalyzer for non-cubic
    mock_elastic.assert_called_once()


@patch('rewards.calculators.ORBCalculator.MFORBCalculator')
@patch('rewards.calculators.ORBCalculator.CubicElasticConstantsAnalyzer')
def test_calc_shear_modulus(mock_elastic, mock_mf_orb, tmp_path, device, fcc_structure):
    """Test shear modulus calculation."""
    calc = ORBCalculator(
        root_dir=str(tmp_path / "orb_shear"),
        task='shear_modulus',
        device=device
    )

    # Mock setup
    mock_calc_instance = Mock()
    mock_mf_orb.return_value = mock_calc_instance

    mock_relax_result = {
        'final_structure': fcc_structure,
        'energy': -10.5,
        'converged': True
    }
    mock_calc_instance.relax.return_value = mock_relax_result

    mock_elastic_instance = Mock()
    mock_elastic.return_value = mock_elastic_instance
    mock_elastic_result = {
        'bulk_modulus': 150.0,
        'shear_modulus': 80.0
    }
    mock_elastic_instance.calculate.return_value = mock_elastic_result

    result = calc._calc_single_structure(fcc_structure)

    assert result == 80.0


@patch('rewards.calculators.ORBCalculator.MFORBCalculator')
@patch('rewards.calculators.ORBCalculator.FormationEnergyAnalyzer')
def test_calc_formation_energy(mock_formation, mock_mf_orb, tmp_path, device, rocksalt_structure):
    """Test formation energy calculation."""
    calc = ORBCalculator(
        root_dir=str(tmp_path / "orb_formation"),
        task='formation_energy',
        device=device
    )

    # Mock calculator
    mock_calc_instance = Mock()
    mock_mf_orb.return_value = mock_calc_instance

    # Mock formation energy analyzer
    mock_formation_instance = Mock()
    mock_formation.return_value = mock_formation_instance
    mock_formation_result = {
        'formation_energy': -2.5  # eV/atom
    }
    mock_formation_instance.calculate.return_value = mock_formation_result

    result = calc._calc_single_structure(rocksalt_structure)

    assert result == -2.5
    mock_formation_instance.calculate.assert_called_once()


def test_calc_batch_structures(orb_calculator, structure_list, xyz_file_path):
    """Test batch calculation of structures."""
    # Mock _calc_single_structure to avoid actual ORB computation
    orb_calculator._calc_single_structure = Mock(side_effect=[150.0, 180.0, 160.0])

    samples = (structure_list, xyz_file_path)
    results = orb_calculator.calc(samples, label='test')

    assert results.shape == (len(structure_list),)
    assert orb_calculator._calc_single_structure.call_count == len(structure_list)


def test_calc_saves_results(orb_calculator, structure_list, xyz_file_path, tmp_path):
    """Test that results are saved to file."""
    # Mock calculations
    orb_calculator._calc_single_structure = Mock(side_effect=[150.0, 180.0, 160.0])

    samples = (structure_list, xyz_file_path)
    results = orb_calculator.calc(samples, label='test_save')

    # Check save file exists
    save_file = tmp_path / "orb_test" / "test_save.txt"
    assert save_file.exists()

    # Verify content
    with open(save_file, 'r') as f:
        lines = f.readlines()
        values = [float(line.strip()) for line in lines]
        assert len(values) == len(structure_list)
        np.testing.assert_array_equal(values, [150.0, 180.0, 160.0])


def test_calc_handles_calculation_failure(orb_calculator, fcc_structure, xyz_file_path):
    """Test handling of calculation failures."""
    # Mock to raise exception
    orb_calculator._calc_single_structure = Mock(side_effect=RuntimeError("Calculation failed"))

    samples = ([fcc_structure], xyz_file_path)
    results = orb_calculator.calc(samples, label='test_fail')

    # Should return NaN for failed calculations
    assert np.isnan(results[0])


@pytest.mark.parametrize("fmax", [0.01, 0.05, 0.1])
def test_different_convergence_criteria(tmp_path, device, fmax):
    """Test calculator with different convergence criteria."""
    calc = ORBCalculator(
        root_dir=str(tmp_path / f"orb_fmax_{fmax}"),
        task='bulk_modulus',
        device=device,
        fmax=fmax
    )

    assert calc.fmax == fmax


def test_is_cubic_detection(orb_calculator, fcc_structure, hcp_structure):
    """Test cubic crystal system detection."""
    # FCC should be cubic
    assert orb_calculator._is_cubic(fcc_structure)

    # HCP should not be cubic
    assert not orb_calculator._is_cubic(hcp_structure)
