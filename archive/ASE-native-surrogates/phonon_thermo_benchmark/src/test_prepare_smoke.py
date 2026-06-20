"""Offline smoke test for the data-prep row-builder.

No network and no ``mp_api``: we exercise ``build_row`` directly with a
synthetic Debye-like phonon DOS and a 2-atom cubic structure. The live MP
query path lives behind a lazy import inside ``prepare_phonon_data.main()`` and
is intentionally not touched here -- the matinvent env cannot import MPRester
(emmet/typing.NotRequired needs Python >= 3.11), but the pure builder only
needs pymatgen, which is available.

Checks:
* a valid (structure, dos) pair yields a dict with the right id, plain-float
  targets, and a physically bounded ``Cv_300K`` (0 < Cv < 3R per atom);
* a DOS with imaginary modes is dropped (``build_row`` returns ``None``).
"""

import os
import sys

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.dos import PhononDos

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_phonon_data import _method_of, build_row

N_ATOMS = 2


def _debye_dos(w_max=10.0, n=4000, w_min=1e-3, n_atoms=N_ATOMS):
    """Synthetic Debye DOS g(w) ~ w^2, normalized so int g dw = 3 * n_atoms.

    The 3-modes-per-atom normalization is what makes C_v -> 3R per atom in the
    high-T limit. ``w_min`` starts just above zero because pymatgen's thermo
    integrals are singular at exactly w = 0 (csch^2 -> inf), and the physical
    Debye spectrum carries no weight there anyway.
    """
    freqs = np.linspace(w_min, w_max, n)
    dens = np.where(freqs > 0, freqs**2, 0.0)
    integral = np.trapezoid(dens, x=freqs)
    dens = dens * (3.0 * n_atoms) / integral
    return PhononDos(freqs, dens)


def _imaginary_dos(w_min=-2.0, w_max=10.0, n=4000, n_atoms=N_ATOMS):
    """A Debye-like DOS extending to -2 THz with real density there.

    Mirrors the imaginary-mode case in test_phonon_thermo: force non-negligible
    density in the negative-frequency region so the imaginary-mode flag has
    something to detect (the bare Debye w^2 vanishes near w = 0).
    """
    dos = _debye_dos(w_max=w_max, n=n, w_min=w_min, n_atoms=n_atoms)
    f = np.asarray(dos.frequencies)
    d = np.asarray(dos.densities).copy()
    d[f < 0] = 0.5
    return PhononDos(f, d)


def _two_atom_structure():
    """A minimal 2-atom cubic structure (num_sites == N_ATOMS)."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert struct.num_sites == N_ATOMS
    return struct


def test_build_row_valid_material():
    """A stable material yields a dict with plain-float, bounded targets."""
    dos = _debye_dos()
    struct = _two_atom_structure()
    row = build_row("mp-149", struct, dos)

    assert row is not None
    assert isinstance(row, dict)
    assert row["material_id"] == "mp-149"

    for key in ("Cv_300K", "S_300K", "F_300K", "max_phonon_freq"):
        assert isinstance(row[key], float), f"{key} is {type(row[key])}, expected float"

    # Heat capacity per atom is positive and below the 3R Dulong-Petit ceiling.
    assert 0 < row["Cv_300K"] < 24.95, (
        f"Cv_300K = {row['Cv_300K']}, expected 0 < Cv < 24.95"
    )


def test_build_row_drops_imaginary_modes():
    """A dynamically unstable DOS (imaginary modes) is dropped -> None."""
    dos = _imaginary_dos()
    struct = _two_atom_structure()
    assert build_row("mp-bad", struct, dos) is None


def test_build_row_drops_missing_inputs():
    """Missing structure or DOS is dropped -> None (defensive)."""
    dos = _debye_dos()
    struct = _two_atom_structure()
    assert build_row("mp-none", None, dos) is None
    assert build_row("mp-none", struct, None) is None


def test_method_of_enum_and_str():
    """_method_of: enum-like object -> its .value; plain str -> itself.

    Offline, no network: fakes the emmet phonon doc's phonon_method field both
    as an enum (has .value) and as a bare string.
    """

    class _FakeEnum:
        value = "pheasy"

    class _DocEnum:
        phonon_method = _FakeEnum()

    class _DocStr:
        phonon_method = "dfpt"

    class _DocNone:
        phonon_method = None

    assert _method_of(_DocEnum()) == "pheasy"
    assert _method_of(_DocStr()) == "dfpt"
    assert _method_of(_DocNone()) is None
    # Missing the attribute entirely also yields None (getattr default).
    assert _method_of(object()) is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
