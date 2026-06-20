"""Phonon thermodynamic targets from a phonon density of states (DOS).

This module converts a pymatgen ``PhononDos`` / ``CompletePhononDos`` into the
per-atom regression targets used as benchmark labels:

* constant-volume heat capacity ``C_v``
* entropy ``S``
* Helmholtz free energy ``F``

all evaluated at a reference temperature (300 K by default), plus the maximum
phonon frequency and an imaginary-mode flag.

Per-atom convention
--------------------
pymatgen's ``PhononDos.cv`` / ``entropy`` / ``helmholtz_free_energy`` return
*per mole-cell* values (Avogadro's number times the atoms in the unit cell)
when called with the temperature only. Passing ``structure=`` to pymatgen
instead normalizes per *formula unit*, which is NOT what we want. We therefore
compute per-atom targets ourselves by dividing the per-cell value by
``structure.num_sites``.

Frequencies are assumed to be in THz (pymatgen's convention).
"""

import numpy as np


def _per_atom(value_per_cell, structure):
    """Divide a per-cell extensive quantity by the number of atoms in the cell."""
    return float(value_per_cell) / structure.num_sites


def cv_per_atom(dos, structure, T=300):
    """Constant-volume heat capacity per atom at temperature ``T``.

    Returns C_v in J/(K*mol-atom). Bounded above by the Dulong-Petit
    classical limit 3R ~= 24.94 J/(K*mol-atom) as T -> infinity.
    """
    return _per_atom(dos.cv(T), structure)


def entropy_per_atom(dos, structure, T=300):
    """Vibrational entropy per atom at temperature ``T``, in J/(K*mol-atom)."""
    return _per_atom(dos.entropy(T), structure)


def free_energy_per_atom(dos, structure, T=300):
    """Helmholtz free energy per atom at temperature ``T``, in J/mol-atom."""
    return _per_atom(dos.helmholtz_free_energy(T), structure)


def max_phonon_freq(dos, frac=1e-3):
    """Highest phonon frequency carrying non-negligible density.

    Returns the largest frequency whose density exceeds ``frac`` times the
    maximum density. Falls back to ``max(frequencies)`` if no bin clears the
    threshold (e.g. a degenerate DOS).
    """
    f = np.asarray(dos.frequencies)
    d = np.asarray(dos.densities)
    mask = d > frac * d.max()
    return float(f[mask].max()) if mask.any() else float(f.max())


def has_imaginary_modes(dos, tol_thz=0.1):
    """True if the DOS has appreciable density at frequencies below -tol_thz.

    Imaginary (soft) phonon modes are conventionally encoded as negative
    frequencies. A small tolerance avoids flagging numerical noise near zero.
    """
    f = np.asarray(dos.frequencies)
    d = np.asarray(dos.densities)
    return bool(((f < -tol_thz) & (d > 1e-8)).any())
