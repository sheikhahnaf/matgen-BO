"""Build a small synthetic phonon-thermo dataset for offline integration smoke tests.

This produces a pickled ``pandas.DataFrame`` with EXACTLY the schema the real
data-prep driver (``prepare_phonon_data.py``) emits, so the GP/MTGP/DGP drivers
can be exercised end-to-end with no Materials Project query, no GPU, and no
network. It is NOT scientific data -- the four targets are deterministic
functions of simple structural descriptors plus a little noise, chosen so a GP
sees real (non-pure-noise) signal to fit.

Schema (matches prepare_phonon_data.build_row):

    material_id      : str   ("mp-0" .. "mp-79")
    structure        : pymatgen Structure (cubic, a in [3, 5] A, 2-4 atoms)
    Cv_300K          : float64
    S_300K           : float64
    F_300K           : float64
    max_phonon_freq  : float64

The compositions are drawn from a small element set {Si, O, Na, Cl, Mg, Al} so
the SOAP species basis is non-trivial and, in cubic cells of this size, every
atom has periodic neighbours well inside the r_cut=5 A SOAP cutoff.

Run:
    python src/make_synthetic_pickle.py
    -> wrote 80 rows -> data/smoke_synth.pkl
"""

import os
import sys

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

# Output is anchored to the package root (parent of src/) so the script works
# regardless of the caller's cwd, and lands where the drivers expect data/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SRC_DIR)
OUT_PATH = os.path.join(PKG_ROOT, "data", "smoke_synth.pkl")

N_ROWS = 80
SEED = 12345

# Small element set: light through heavier, so mean atomic mass varies enough to
# drive a real signal across rows, and the SOAP species basis spans 6 elements.
ELEMENTS = ["Si", "O", "Na", "Cl", "Mg", "Al"]


def _random_structure(rng):
    """Return one cubic pymatgen Structure with 2-4 atoms drawn from ELEMENTS.

    The lattice constant (3-5 A) and site count are randomised so the dataset
    spans a range of densities and compositions; fractional coordinates are
    spread across the cell so SOAP sees genuine local environments.
    """
    a = float(rng.uniform(3.0, 5.0))
    lattice = Lattice.cubic(a)
    n_sites = int(rng.integers(2, 5))  # 2, 3, or 4 atoms

    species = list(rng.choice(ELEMENTS, size=n_sites, replace=True))

    # Deterministic, well-separated fractional coordinates (a small "Wyckoff"
    # menu) so two sites never coincide and SOAP gets distinct environments.
    coord_menu = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    coords = coord_menu[:n_sites]

    return Structure(lattice, species, coords)


def _structural_descriptor(structure):
    """A simple, smooth scalar descriptor of a structure.

    Combines mean atomic mass, number density, and site count -- all cheap and
    well-defined -- into a single number the synthetic targets are built from.
    """
    masses = np.array([sp.atomic_mass for sp in structure.species], dtype=float)
    mean_mass = float(masses.mean())
    volume = float(structure.volume)
    n_sites = int(structure.num_sites)
    number_density = n_sites / volume  # atoms per A^3
    return mean_mass, number_density, n_sites


def build_dataframe(seed=SEED, n_rows=N_ROWS):
    """Build the synthetic DataFrame with the real benchmark schema."""
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_rows):
        structure = _random_structure(rng)
        mean_mass, number_density, n_sites = _structural_descriptor(structure)

        # Deterministic target functions of the structural descriptor + small
        # Gaussian noise (noise std << signal range) so a GP has real signal.
        # Forms are loosely physically motivated but make no claim of accuracy:
        #   - Cv (heat capacity per atom) trends with mass / softness, bounded.
        #   - S (entropy per atom) grows with mass and number of sites.
        #   - F (free energy per atom) is large and negative, scaling with mass.
        #   - max phonon freq falls with mass and rises with number density.
        noise = rng.normal(0.0, 1.0, size=4)

        cv = 18.0 + 0.02 * mean_mass + 2.0 * number_density + 0.05 * noise[0]
        s = 5.0 + 0.04 * mean_mass + 1.5 * n_sites + 0.05 * noise[1]
        f = -2000.0 - 8.0 * mean_mass - 200.0 * number_density + 1.0 * noise[2]
        max_freq = 18.0 - 0.04 * mean_mass + 30.0 * number_density + 0.05 * noise[3]

        rows.append(
            {
                "material_id": f"mp-{i}",
                "structure": structure,
                "Cv_300K": float(cv),
                "S_300K": float(s),
                "F_300K": float(f),
                "max_phonon_freq": float(max_freq),
            }
        )

    df = pd.DataFrame(rows)

    # Match the real data-prep driver: force the four targets to float64.
    target_cols = ["Cv_300K", "S_300K", "F_300K", "max_phonon_freq"]
    for col in target_cols:
        df[col] = df[col].astype("float64")

    return df


def main():
    df = build_dataframe()

    # Defensive: the GP per-property filter needs non-NaN targets, and
    # auto_detect_targets drops >20% NaN columns -- assert we emit none.
    target_cols = ["Cv_300K", "S_300K", "F_300K", "max_phonon_freq"]
    assert not df[target_cols].isna().any().any(), "synthetic targets must have no NaNs"
    assert (df[target_cols].dtypes == "float64").all(), "targets must be float64"

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_pickle(OUT_PATH)
    print(f"wrote {len(df)} rows -> data/smoke_synth.pkl")


if __name__ == "__main__":
    main()
