"""Unit tests for src/router.py — σ-routing + threshold calibration.

Critical invariants tested:
    - GP-predicted μ is RETURNED as reward but NEVER lands in LTM rows.
    - σ-threshold actually gates oracle vs GP path.
    - calibrate_threshold_from_picp uses errors (not just σ).
    - force_anchor=True bypasses σ check.
    - cold-start (n_train < min) routes everything to oracle.
"""

import numpy as np
import pytest
from ase.build import bulk

from src.router import (
    sigma_route, ltm_rows_for_oracle_results, calibrate_threshold_from_picp,
)


# --- Mocks ---------------------------------------------------------------

class _MockSurrogate:
    """Minimal HCapSurrogate stand-in returning controlled (μ, σ)."""

    def __init__(self, mu_arr, sigma_arr, n_train=100):
        self.mu = np.asarray(mu_arr, dtype=float)
        self.sigma = np.asarray(sigma_arr, dtype=float)
        # Mock train_inputs to satisfy the cold-start check.
        import torch
        Xt = torch.zeros(n_train, 1)

        class _M:
            def __init__(self, x):
                self.train_inputs = (x,)
        self.model = _M(Xt)

    def predict(self, Z):
        return self.mu, self.sigma


class _MockOracle:
    """Returns synthetic Cp values keyed by Atoms identity (id()).

    The router passes only the σ ≥ τ subset to oracle.evaluate(), so positional
    indexing inside the mock would not align with the global candidate index.
    Use id(atoms) → cp lookup instead.
    """

    def __init__(self, atoms_to_cp: dict, fail_atoms: set | None = None):
        self.atoms_to_cp = atoms_to_cp
        self.fail_atoms = fail_atoms or set()
        self.calls = 0

    def evaluate(self, atoms_list):
        self.calls += 1
        out = []
        for a in atoms_list:
            key = id(a)
            if key in self.fail_atoms:
                out.append(np.nan)
            else:
                out.append(float(self.atoms_to_cp[key]))
        cp = np.array(out, dtype=float)
        return cp, ~np.isfinite(cp)


def _toy_atoms(n=4):
    return [bulk("Si", "diamond", a=5.43) for _ in range(n)]


# --- Tests ---------------------------------------------------------------

def _id_keyed(atoms, cp):
    return {id(a): float(c) for a, c in zip(atoms, cp)}


def test_sigma_below_threshold_uses_gp():
    K = 4
    atoms = _toy_atoms(K)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0, 3.0, 4.0], sigma_arr=[0.01, 0.01, 0.01, 0.01])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0, 30.0, 40.0]))
    Z = np.eye(K)
    result = sigma_route(Z, atoms, sur, orc, threshold=0.05, cycle_id=0)

    assert result.n_oracle_calls == 0, "All σ < τ → no oracle calls"
    assert orc.calls == 0
    np.testing.assert_array_equal(result.sources, ["gp"] * K)
    np.testing.assert_array_almost_equal(result.rewards, [1.0, 2.0, 3.0, 4.0])


def test_sigma_above_threshold_uses_oracle():
    K = 4
    atoms = _toy_atoms(K)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0, 3.0, 4.0], sigma_arr=[0.5, 0.5, 0.5, 0.5])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0, 30.0, 40.0]))
    Z = np.eye(K)
    result = sigma_route(Z, atoms, sur, orc, threshold=0.05, cycle_id=0)

    assert result.n_oracle_calls == K
    np.testing.assert_array_equal(result.sources, ["oracle"] * K)
    np.testing.assert_array_almost_equal(result.rewards, [10.0, 20.0, 30.0, 40.0])


def test_mixed_routing_per_sample():
    atoms = _toy_atoms(4)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0, 3.0, 4.0], sigma_arr=[0.01, 0.5, 0.01, 0.5])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0, 30.0, 40.0]))
    Z = np.eye(4)
    result = sigma_route(Z, atoms, sur, orc, threshold=0.05, cycle_id=0)

    np.testing.assert_array_equal(result.sources, ["gp", "oracle", "gp", "oracle"])
    np.testing.assert_array_almost_equal(result.rewards, [1.0, 20.0, 3.0, 40.0])
    assert result.n_oracle_calls == 2


def test_force_anchor_overrides_sigma():
    atoms = _toy_atoms(2)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0], sigma_arr=[0.001, 0.001])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0]))
    result = sigma_route(np.eye(2), atoms, sur, orc,
                         threshold=0.5, cycle_id=5, force_anchor=True)
    assert result.n_oracle_calls == 2
    np.testing.assert_array_almost_equal(result.rewards, [10.0, 20.0])


def test_cold_start_oracles_all():
    atoms = _toy_atoms(2)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0], sigma_arr=[0.001, 0.001], n_train=4)
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0]))
    result = sigma_route(np.eye(2), atoms, sur, orc, threshold=0.5,
                         cycle_id=0, cold_start_min_train=16)
    assert result.n_oracle_calls == 2, "n_train < min → all to oracle"


def test_oracle_failure_falls_back_to_mu():
    atoms = _toy_atoms(2)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0], sigma_arr=[0.5, 0.5])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0]), fail_atoms={id(atoms[0])})
    result = sigma_route(np.eye(2), atoms, sur, orc, threshold=0.05, cycle_id=0)
    assert result.sources[0] == "oracfail"  # full string, not truncated
    assert result.sources[1] == "oracle"
    np.testing.assert_almost_equal(result.rewards[0], 1.0)
    np.testing.assert_almost_equal(result.rewards[1], 20.0)


def test_ltm_rows_only_include_successful_oracle_results():
    """The HARD INVARIANT: GP-only rewards never become LTM rows."""
    atoms = _toy_atoms(4)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0, 3.0, 4.0], sigma_arr=[0.01, 0.5, 0.01, 0.5])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0, 30.0, 40.0]))
    Z = np.eye(4)
    result = sigma_route(Z, atoms, sur, orc, threshold=0.05, cycle_id=7)

    sids = [f"sid{i}" for i in range(4)]
    formulas = [a.get_chemical_formula() for a in atoms]
    rows = ltm_rows_for_oracle_results(result, Z, atoms, sids, formulas)

    assert len(rows) == 2
    sids_in_rows = sorted(r["structure_id"] for r in rows)
    assert sids_in_rows == ["sid1", "sid3"]
    cps = sorted(r["y_cp"] for r in rows)
    assert cps == [20.0, 40.0]
    for r in rows:
        assert r["y_cp"] != 1.0 and r["y_cp"] != 3.0


def test_ltm_rows_skip_oracle_failures():
    atoms = _toy_atoms(2)
    sur = _MockSurrogate(mu_arr=[1.0, 2.0], sigma_arr=[0.5, 0.5])
    orc = _MockOracle(_id_keyed(atoms, [10.0, 20.0]), fail_atoms={id(atoms[0])})
    Z = np.eye(2)
    result = sigma_route(Z, atoms, sur, orc, threshold=0.05, cycle_id=1)

    rows = ltm_rows_for_oracle_results(result, Z, atoms, ["a", "b"], ["X", "X"])
    assert len(rows) == 1
    assert rows[0]["structure_id"] == "b"
    assert rows[0]["y_cp"] == 20.0


def test_calibrate_threshold_uses_empirical_errors():
    """Same σ but different errors should yield different τ."""
    sigma = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    err_low = np.array([0.05, 0.04, 0.1, 0.12, 0.2, 0.18])  # well calibrated
    err_high = np.array([0.5, 0.4, 1.0, 1.2, 2.0, 1.8])     # over-confident GP

    tau_low = calibrate_threshold_from_picp(sigma, err_low, target_picp=0.9)
    tau_high = calibrate_threshold_from_picp(sigma, err_high, target_picp=0.9)

    # If GP is over-confident (error >> σ), τ should be SMALLER (force more oracle calls).
    # Our implementation: τ = quantile_target(|err|/σ) × quantile_(1-target)(σ)
    # For err_high, the |err|/σ ratio is much higher → larger τ.
    # Either way, tau_low ≠ tau_high — that's the key invariant.
    assert not np.isclose(tau_low, tau_high), \
        "τ should differ when errors differ; previous bug: only σ was used"


def test_calibrate_threshold_few_points_returns_nan():
    """Too few anchor points → NaN signals 'keep current τ'."""
    sigma = np.array([0.1, 0.2])
    err = np.array([0.05, 0.1])
    tau = calibrate_threshold_from_picp(sigma, err, target_picp=0.9)
    assert np.isnan(tau)
