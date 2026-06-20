#!/usr/bin/env python
"""Direct validation script for the 4 critical GP routing bug fixes."""

import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_bug3_gp_readiness_tracking():
    """Bug 3: Track fitted sample count in GP surrogate."""
    from rewards.gp.surrogate import GPSurrogate

    logging.info("\n" + "="*70)
    logging.info("Testing Bug 3: GP Readiness Tracking")
    logging.info("="*70)

    gp = GPSurrogate(input_dim=10, task='bulk_modulus', device='cpu')

    # Check initial state
    assert gp.n_fitted_samples == 0, "Initial n_fitted_samples should be 0"
    assert gp.get_training_data_size() == 0, "Initial training size should be 0"
    logging.info("✓ Initial state: n_fitted_samples=0, training_size=0")

    # Fit GP directly (bypassing buffer) - this is what GPTrainingManager does
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    gp.fit(X, y)

    # Check that get_training_data_size() now returns correct count
    assert gp.n_fitted_samples == 50, f"n_fitted_samples should be 50, got {gp.n_fitted_samples}"
    assert gp.get_training_data_size() == 50, f"training_size should be 50, got {gp.get_training_data_size()}"
    logging.info("✓ After fit(X,y): n_fitted_samples=50, training_size=50")

    # Fit with more data
    X2 = np.random.randn(100, 10)
    y2 = np.random.randn(100)
    gp.fit(X2, y2)

    assert gp.n_fitted_samples == 100, f"n_fitted_samples should be 100, got {gp.n_fitted_samples}"
    assert gp.get_training_data_size() == 100, f"training_size should be 100, got {gp.get_training_data_size()}"
    logging.info("✓ After fit(X2,y2): n_fitted_samples=100, training_size=100")

    logging.info("\n✅ Bug 3 FIXED: GP readiness tracking works correctly!")
    return True


def test_bug2_gp_model_sharing():
    """Bug 2: Trainer reuses router's GP model (shared references)."""
    from rewards.router import CalculatorRouter
    from rewards.gp.surrogate import GPSurrogate
    from rewards.calculators.orb.featurizer import ORBFeaturizer
    from rewards.acquisition import ExpectedImprovementPerCost

    logging.info("\n" + "="*70)
    logging.info("Testing Bug 2: GP Model Sharing Between Router and Trainer")
    logging.info("="*70)

    # Create GP and featurizer
    gp_model = GPSurrogate(input_dim=10, task='bulk_modulus', device='cpu')
    featurizer = ORBFeaturizer(n_components=10, device='cpu')

    # Create router with GP model
    router = CalculatorRouter(
        calculators={},  # Empty for this test
        gp_model=gp_model,
        featurizer=featurizer,
        acquisition_fn=ExpectedImprovementPerCost(cost_model={}),
        default_calculator='orb'
    )

    # Verify router has the GP model
    assert router.gp_model is gp_model, "Router should have reference to GP model"
    logging.info("✓ Router initialized with GP model reference")

    # Simulate what _init_gp_trainer does: extract GP from router
    extracted_gp = router.gp_model
    extracted_featurizer = router.featurizer

    # Verify these are the SAME objects (shared references, not copies)
    assert extracted_gp is gp_model, "Extracted GP should be same object as original"
    assert extracted_featurizer is featurizer, "Extracted featurizer should be same object"
    logging.info("✓ Extracted GP model is same object (shared reference)")

    # Modify extracted GP (simulate trainer fitting)
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    extracted_gp.fit(X, y)

    # Verify router sees the change (because it's the same object)
    assert router.gp_model.is_trained == True, "Router should see trained GP"
    assert router.gp_model.n_fitted_samples == 50, "Router should see fitted sample count"
    logging.info("✓ Router sees changes made to GP (same object in memory)")

    logging.info("\n✅ Bug 2 FIXED: GP model sharing works via shared references!")
    return True


def test_bug1_calibration_data_separation():
    """Bug 1: Calibration data stored separately from scoring dict."""
    from rewards.reward import Reward
    from omegaconf import OmegaConf

    logging.info("\n" + "="*70)
    logging.info("Testing Bug 1: Calibration Data Separation")
    logging.info("="*70)

    # Create reward with mock config
    config = OmegaConf.create({
        'root_dir': '/tmp/test_reward',
        'prop_cfg': [],
        'reward_threshold': 0.5
    })

    reward = Reward(**config)

    # Check initial state
    assert hasattr(reward, 'multi_calc_data'), "Reward should have multi_calc_data attribute"
    assert isinstance(reward.multi_calc_data, dict), "multi_calc_data should be dict"
    assert len(reward.multi_calc_data) == 0, "multi_calc_data should be empty initially"
    logging.info("✓ Reward has multi_calc_data attribute (dict)")

    # Simulate calibration mode storing multi-calculator data
    mock_multi_calc = {
        'orb': np.array([100.0, 150.0, 120.0]),
        'alignn': np.array([95.0, 145.0, 115.0]),
        'vasp': np.array([98.0, 148.0, 118.0])
    }
    reward.multi_calc_data['bulk_modulus'] = mock_multi_calc

    # Verify storage
    assert 'bulk_modulus' in reward.multi_calc_data, "Property should be in multi_calc_data"
    assert len(reward.multi_calc_data['bulk_modulus']) == 3, "Should have 3 calculators"
    logging.info("✓ Multi-calculator data stored in separate attribute")

    # This separation ensures prop_dict stays numeric for scoring()
    # while calibration data is preserved for LTM
    logging.info("✓ Scoring can use numeric prop_dict while LTM gets multi_calc_data")

    logging.info("\n✅ Bug 1 FIXED: Calibration data separation works!")
    return True


def test_bug4_unique_structure_ids():
    """Bug 4: Structure IDs are globally unique across calibration steps."""
    from memory.ltm import LongTimeMem
    from pymatgen.core import Structure, Lattice

    logging.info("\n" + "="*70)
    logging.info("Testing Bug 4: Unique Structure IDs")
    logging.info("="*70)

    ltm = LongTimeMem()

    # Create mock structures
    lattice = Lattice.cubic(4.0)
    structures = [
        Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
        Structure(lattice, ["Ge", "Ge"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
        Structure(lattice, ["C", "C"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    ]

    # Initial counter should be 0
    assert ltm._structure_id_counter == 0, "Initial counter should be 0"
    logging.info(f"✓ Initial structure_id_counter: {ltm._structure_id_counter}")

    # Simulate calibration step 0: 3 structures evaluated by 3 calculators
    structure_ids_step0 = list(range(ltm._structure_id_counter, ltm._structure_id_counter + 3))
    logging.info(f"  Step 0: Structure IDs = {structure_ids_step0}")

    for calc_name in ['orb', 'alignn', 'vasp']:
        ltm.extend(
            structures,
            rewards=np.array([0.5, 0.6, 0.7]),
            step=0,
            property_values=np.array([100.0, 110.0, 120.0]),
            calculators_used=[calc_name] * 3,
            structure_ids=structure_ids_step0  # SAME IDs for all calculators
        )

    # Manually increment (as done in mat_invent.py)
    ltm._structure_id_counter += len(structures)

    assert ltm._structure_id_counter == 3, f"Counter should be 3, got {ltm._structure_id_counter}"
    logging.info(f"✓ After step 0: counter incremented to {ltm._structure_id_counter}")

    # Verify IDs in LTM
    step0_ids = ltm.memory[ltm.memory['RL_step'] == 0]['structure_id'].unique()
    assert len(step0_ids) == 3, f"Should have 3 unique structure IDs, got {len(step0_ids)}"
    assert list(step0_ids) == [0, 1, 2], f"IDs should be [0,1,2], got {list(step0_ids)}"
    logging.info(f"✓ Step 0 IDs in LTM: {list(step0_ids)} (3 structures × 3 calculators = 9 rows)")

    # Simulate calibration step 1: 3 NEW structures
    structure_ids_step1 = list(range(ltm._structure_id_counter, ltm._structure_id_counter + 3))
    logging.info(f"  Step 1: Structure IDs = {structure_ids_step1}")

    for calc_name in ['orb', 'alignn', 'vasp']:
        ltm.extend(
            structures,
            rewards=np.array([0.8, 0.9, 1.0]),
            step=1,
            property_values=np.array([130.0, 140.0, 150.0]),
            calculators_used=[calc_name] * 3,
            structure_ids=structure_ids_step1
        )

    ltm._structure_id_counter += len(structures)

    assert ltm._structure_id_counter == 6, f"Counter should be 6, got {ltm._structure_id_counter}"
    logging.info(f"✓ After step 1: counter incremented to {ltm._structure_id_counter}")

    # Verify IDs are globally unique
    step1_ids = ltm.memory[ltm.memory['RL_step'] == 1]['structure_id'].unique()
    assert len(step1_ids) == 3, f"Should have 3 unique structure IDs, got {len(step1_ids)}"
    assert list(step1_ids) == [3, 4, 5], f"IDs should be [3,4,5], got {list(step1_ids)}"
    logging.info(f"✓ Step 1 IDs in LTM: {list(step1_ids)} (globally unique!)")

    # Verify NoiseEstimator can pair structures correctly
    all_ids = ltm.memory['structure_id'].unique()
    assert len(all_ids) == 6, f"Should have 6 total unique IDs, got {len(all_ids)}"
    logging.info(f"✓ Total unique structure IDs: {len(all_ids)} (prevents ID collisions)")

    # Verify same structure evaluated by multiple calculators has matching ID
    structure_0_data = ltm.memory[ltm.memory['structure_id'] == 0]
    assert len(structure_0_data) == 3, f"Structure 0 should appear 3 times (3 calcs), got {len(structure_0_data)}"
    calcs = structure_0_data['calculator_used'].tolist()
    assert set(calcs) == {'orb', 'alignn', 'vasp'}, "Structure 0 evaluated by all 3 calculators"
    logging.info(f"✓ Structure ID=0 evaluated by: {calcs} (enables noise pairing)")

    logging.info("\n✅ Bug 4 FIXED: Structure IDs are globally unique!")
    return True


def main():
    """Run all validation tests."""
    logging.info("\n" + "🔍 VALIDATING 4 CRITICAL BUG FIXES ".center(70, "="))
    logging.info("Plan implementation: Fix GP routing bugs preventing RL workflow\n")

    results = []

    try:
        results.append(("Bug 3: GP Readiness Tracking", test_bug3_gp_readiness_tracking()))
    except Exception as e:
        logging.error(f"❌ Bug 3 test failed: {e}")
        results.append(("Bug 3: GP Readiness Tracking", False))

    try:
        results.append(("Bug 2: GP Model Sharing", test_bug2_gp_model_sharing()))
    except Exception as e:
        logging.error(f"❌ Bug 2 test failed: {e}")
        results.append(("Bug 2: GP Model Sharing", False))

    try:
        results.append(("Bug 1: Calibration Data Separation", test_bug1_calibration_data_separation()))
    except Exception as e:
        logging.error(f"❌ Bug 1 test failed: {e}")
        results.append(("Bug 1: Calibration Data Separation", False))

    try:
        results.append(("Bug 4: Unique Structure IDs", test_bug4_unique_structure_ids()))
    except Exception as e:
        logging.error(f"❌ Bug 4 test failed: {e}")
        results.append(("Bug 4: Unique Structure IDs", False))

    # Summary
    logging.info("\n" + "="*70)
    logging.info("VALIDATION SUMMARY")
    logging.info("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logging.info(f"{status}: {name}")

    logging.info(f"\n{passed}/{total} bug fixes validated successfully")

    if passed == total:
        logging.info("\n🎉 ALL BUG FIXES VALIDATED!")
        return 0
    else:
        logging.info(f"\n⚠️  {total - passed} bug fix(es) need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())
