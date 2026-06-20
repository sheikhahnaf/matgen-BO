"""Physics-grounded tests for the phonon thermodynamic target module.

The central acceptance check is the Dulong-Petit limit: as T -> infinity the
constant-volume heat capacity per atom must converge to 3R ~= 24.94
J/(K*mol-atom). If it does not, the unit handling or DOS normalization is
wrong -- we fix the physics, not the tolerance.
"""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.dos import PhononDos

from phonon_thermo import (
    cv_per_atom,
    entropy_per_atom,
    free_energy_per_atom,
    has_imaginary_modes,
    max_phonon_freq,
)

# 3R, the classical Dulong-Petit heat capacity per mole of atoms.
DULONG_PETIT = 3.0 * 8.314462618  # ~= 24.943 J/(K*mol-atom)
N_ATOMS = 2


def _debye_dos(w_max=10.0, n=4000, w_min=1e-3, n_atoms=N_ATOMS):
    """Synthetic Debye DOS g(w) ~ w^2 on [w_min, w_max].

    Normalized so that the integral of g over frequency equals 3 * n_atoms,
    i.e. the total number of vibrational modes (3 per atom). This is the
    normalization that makes C_v -> 3R per atom in the high-T limit.

    ``w_min`` defaults to a small positive value rather than exactly 0:
    pymatgen's thermodynamic integrals evaluate csch^2(w/2kT), which is
    singular at w=0 and produces inf*0 = NaN when a frequency bin sits exactly
    on zero. The physical Debye spectrum carries zero weight at w=0, so
    starting just above zero leaves both the normalization and the physics
    unchanged.
    """
    freqs = np.linspace(w_min, w_max, n)
    dens = np.where(freqs > 0, freqs**2, 0.0)
    integral = np.trapezoid(dens, x=freqs)
    dens = dens * (3.0 * n_atoms) / integral
    return PhononDos(freqs, dens)


def _two_atom_structure():
    """A minimal 2-atom cubic structure (num_sites == N_ATOMS)."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert struct.num_sites == N_ATOMS
    return struct


def test_cv_below_dulong_petit_at_300K():
    """At 300 K, C_v per atom is positive and below the 3R classical ceiling."""
    dos = _debye_dos()
    struct = _two_atom_structure()
    cv = cv_per_atom(dos, struct, T=300)
    assert 0 < cv < 24.95, f"C_v(300K) = {cv}, expected 0 < cv < 24.95"


def test_cv_approaches_dulong_petit_high_T():
    """THE key physics check: C_v per atom -> 3R ~= 24.94 as T -> infinity."""
    dos = _debye_dos()
    struct = _two_atom_structure()
    cv = cv_per_atom(dos, struct, T=5000)
    assert abs(cv - DULONG_PETIT) < 1.0, (
        f"C_v(5000K) = {cv}, expected ~{DULONG_PETIT:.3f} (Dulong-Petit)"
    )


def test_max_freq_positive_and_near_wmax():
    """For a DOS up to 10 THz, the max phonon frequency is ~10 THz."""
    dos = _debye_dos(w_max=10.0)
    wmax = max_phonon_freq(dos)
    assert wmax > 0
    assert abs(wmax - 10.0) < 0.5, f"max_phonon_freq = {wmax}, expected ~10.0"


def test_imaginary_modes_flag():
    """A DOS extending to -2 THz with real density there is flagged imaginary."""
    dos = _debye_dos(w_min=-2.0, w_max=10.0)
    # Force non-negligible density in the negative-frequency region so the
    # flag has something to detect (Debye w^2 vanishes near w=0).
    f = np.asarray(dos.frequencies)
    d = np.asarray(dos.densities)
    d = d.copy()
    d[f < 0] = 0.5
    dos_neg = PhononDos(f, d)
    assert has_imaginary_modes(dos_neg) is True

    # Sanity: a purely-positive DOS is not flagged.
    assert has_imaginary_modes(_debye_dos()) is False


def test_entropy_and_free_energy_signs():
    """Sanity check on the other two targets at 300 K.

    Vibrational entropy is non-negative; Helmholtz free energy F = U - TS is
    typically negative once the entropic term dominates.
    """
    dos = _debye_dos()
    struct = _two_atom_structure()
    s = entropy_per_atom(dos, struct, T=300)
    f = free_energy_per_atom(dos, struct, T=300)
    assert s >= 0, f"entropy = {s}, expected non-negative"
    assert np.isfinite(f), f"free energy = {f}, expected finite"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
