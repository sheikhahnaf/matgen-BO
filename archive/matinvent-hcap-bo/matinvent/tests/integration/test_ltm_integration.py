"""Integration tests for Long-Term Memory with GP data storage."""

import pytest
import numpy as np
import pandas as pd

from memory.ltm import LongTimeMem


@pytest.mark.integration
def test_ltm_store_gp_data(structure_list, mock_rewards, mock_property_values):
    """Test that LTM stores GP-related data correctly."""
    ltm = LongTimeMem()

    n = len(structure_list)
    features = [np.random.randn(50) for _ in range(n)]
    calculators_used = ['orb', 'alignn', 'orb']

    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        property_values=mock_property_values[:n],
        features=features,
        calculators_used=calculators_used
    )

    # Check data is stored
    assert len(ltm) == n
    assert 'property_value' in ltm.memory.columns
    assert 'features' in ltm.memory.columns
    assert 'calculator_used' in ltm.memory.columns

    # Verify values
    np.testing.assert_array_equal(ltm.memory['property_value'].values, mock_property_values[:n])
    assert list(ltm.memory['calculator_used'].values) == calculators_used


@pytest.mark.integration
def test_ltm_backwards_compatibility(structure_list, mock_rewards):
    """Test that LTM works without GP data (backwards compatibility)."""
    ltm = LongTimeMem()

    # Old-style extend (without GP data)
    ltm.extend(structure_list, mock_rewards[:len(structure_list)], step=0)

    assert len(ltm) == len(structure_list)

    # New columns should be None
    assert ltm.memory['property_value'].isna().all()
    assert ltm.memory['features'].isna().all()
    assert ltm.memory['calculator_used'].isna().all()


@pytest.mark.integration
def test_ltm_mixed_data(structure_list, mock_rewards, mock_property_values):
    """Test LTM with mixed old and new data."""
    ltm = LongTimeMem()

    n = len(structure_list)

    # Add old-style data (step 0)
    ltm.extend(structure_list, mock_rewards[:n], step=0)

    # Add new-style data with GP metadata (step 1)
    features = [np.random.randn(50) for _ in range(n)]
    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=1,
        property_values=mock_property_values[:n],
        features=features,
        calculators_used=['orb'] * n
    )

    assert len(ltm) == 2 * n

    # First batch should have None GP data
    assert ltm.memory.iloc[0]['property_value'] is None or pd.isna(ltm.memory.iloc[0]['property_value'])

    # Second batch should have GP data
    assert not pd.isna(ltm.memory.iloc[n]['property_value'])


@pytest.mark.integration
def test_ltm_feature_retrieval(structure_list, mock_rewards):
    """Test retrieving features from LTM."""
    ltm = LongTimeMem()

    n = len(structure_list)
    features = [np.random.randn(50) for _ in range(n)]

    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        features=features
    )

    # Retrieve features
    stored_features = ltm.memory['features'].values

    # Should be able to reconstruct feature matrix
    feature_matrix = np.stack(stored_features)
    assert feature_matrix.shape == (n, 50)


@pytest.mark.integration
def test_ltm_save_with_gp_data(tmp_path, structure_list, mock_rewards, mock_property_values):
    """Test saving LTM with GP data to CSV."""
    ltm = LongTimeMem()

    n = len(structure_list)
    features = [np.random.randn(50) for _ in range(n)]

    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        property_values=mock_property_values[:n],
        features=features,
        calculators_used=['orb'] * n
    )

    # Save to CSV
    save_path = tmp_path / 'ltm_test.csv'
    ltm.save(str(save_path))

    assert save_path.exists()

    # Load and verify
    df = pd.read_csv(save_path)

    assert 'property_value' in df.columns
    assert 'calculator_used' in df.columns
    # Features are complex objects, might be serialized differently


@pytest.mark.integration
def test_ltm_gp_data_for_training(structure_list, mock_rewards, mock_property_values):
    """Test extracting GP training data from LTM."""
    ltm = LongTimeMem()

    n = len(structure_list)
    features = [np.random.randn(50) for _ in range(n)]

    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        property_values=mock_property_values[:n],
        features=features
    )

    # Extract training data
    valid_mask = ltm.memory['property_value'].notna()
    X = np.stack(ltm.memory.loc[valid_mask, 'features'].values)
    y = ltm.memory.loc[valid_mask, 'property_value'].values.astype(float)

    assert X.shape == (n, 50)
    assert y.shape == (n,)
    np.testing.assert_array_equal(y, mock_property_values[:n])


@pytest.mark.integration
def test_ltm_incremental_gp_data(structure_list, mock_rewards, mock_property_values):
    """Test incrementally adding GP data over multiple RL steps."""
    ltm = LongTimeMem()

    for step in range(5):
        n = len(structure_list)
        features = [np.random.randn(50) for _ in range(n)]
        prop_vals = mock_property_values[:n] + step * 10  # Slightly different each step

        ltm.extend(
            structure_list,
            mock_rewards[:n],
            step=step,
            property_values=prop_vals,
            features=features,
            calculators_used=['orb' if step < 2 else 'alignn'] * n
        )

    # Should have accumulated data
    assert len(ltm) == 5 * len(structure_list)

    # All should have GP data
    assert ltm.memory['property_value'].notna().all()
    assert ltm.memory['features'].notna().all()

    # Check calculator distribution
    calc_counts = ltm.memory['calculator_used'].value_counts()
    assert 'orb' in calc_counts
    assert 'alignn' in calc_counts


@pytest.mark.integration
def test_ltm_diversity_filter_with_gp_data(structure_list, mock_rewards):
    """Test that diversity filter still works with GP data."""
    ltm = LongTimeMem()

    # Add structures with GP data
    n = len(structure_list)
    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        property_values=np.random.rand(n) * 100,
        features=[np.random.randn(50) for _ in range(n)]
    )

    # Use diversity filter
    new_rewards, penalty_idx, tol_n, buff_n = ltm.div_filter(
        structure_list,
        mock_rewards[:n],
        tol=1,
        buff=3
    )

    # Should still work
    assert len(new_rewards) == n
    assert isinstance(penalty_idx, list)


@pytest.mark.integration
def test_ltm_calc_metrics_with_gp_data(structure_list, mock_rewards):
    """Test that metrics calculation works with GP data."""
    ltm = LongTimeMem()

    n = len(structure_list)
    ltm.extend(
        structure_list,
        mock_rewards[:n],
        step=0,
        property_values=np.random.rand(n) * 100
    )

    # Calculate metrics
    burden, div_ratio = ltm.calc_metrics(threshold=0.5, budget=1000, num_candidate=10)

    # Metrics should still compute
    assert burden is not None or div_ratio is not None


@pytest.mark.integration
def test_ltm_get_baseline_with_gp_data(structure_list, mock_rewards):
    """Test baseline calculation with GP data."""
    ltm = LongTimeMem()

    for step in range(3):
        ltm.extend(
            structure_list,
            mock_rewards[:len(structure_list)] + step * 0.1,
            step=step,
            property_values=np.random.rand(len(structure_list)) * 100
        )

    baseline = ltm.get_baseline(step=2, prev=2)

    # Should compute baseline from recent steps
    assert baseline is not None
    assert not np.isnan(baseline)
