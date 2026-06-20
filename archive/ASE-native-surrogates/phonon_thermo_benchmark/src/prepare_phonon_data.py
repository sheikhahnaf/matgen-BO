"""Build per-atom phonon thermodynamic regression targets from Materials Project.

This is the data-prep driver for the phonon-thermo benchmark. It queries the
Materials Project phonon endpoint for all materials computed with a chosen
phonon method (DFPT or pheasy), pulls the phonon DOS and crystal structure for
each, and converts them into the per-atom regression labels defined in
``phonon_thermo.py``:

* ``Cv_300K``            -- constant-volume heat capacity, J/(K*mol-atom)
* ``S_300K``             -- vibrational entropy, J/(K*mol-atom)
* ``F_300K``             -- Helmholtz free energy, J/mol-atom
* ``max_phonon_freq``    -- highest phonon frequency carrying density, THz

Dynamically unstable materials (those with appreciable imaginary-mode density)
are dropped. The result is written to a pickled ``pandas.DataFrame`` keyed by
``material_id``, carrying the pymatgen ``Structure`` alongside the four targets.

The live Materials Project query (``mp_api``) is imported lazily inside
``main()`` so that the pure row-builder and its offline tests do not require an
API key or network access.
"""

import argparse
import os
import sys

import pandas as pd

# The phonon_thermo module lives next to this file in src/. Inserting the
# script's own directory makes the import work regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonon_thermo import (
    cv_per_atom,
    entropy_per_atom,
    free_energy_per_atom,
    has_imaginary_modes,
    max_phonon_freq,
)


def _method_of(doc):
    """Extract the phonon method string from an emmet phonon doc.

    The ``phonon_method`` field is an enum ({dfpt, phonopy, pheasy}) on recent
    emmet docs; older/serialized docs may carry a plain string. Return the enum
    ``.value`` when present, otherwise the value itself (str passthrough).
    """
    m = getattr(doc, "phonon_method", None)
    return getattr(m, "value", m)


def build_row(material_id, structure, dos, T=300):
    """Build one benchmark row from a material's structure and phonon DOS.

    Returns ``None`` (the row is dropped) when the structure or DOS is missing,
    or when the DOS carries appreciable imaginary-mode density -- such materials
    are dynamically unstable and excluded from the regression set.

    Otherwise returns a dict with the four per-atom targets (as plain Python
    floats) plus the structure and id:

        {
            "material_id": str,
            "structure": pymatgen Structure,
            "Cv_300K": float,
            "S_300K": float,
            "F_300K": float,
            "max_phonon_freq": float,
        }
    """
    if structure is None or dos is None:
        return None
    if has_imaginary_modes(dos):
        return None
    return {
        "material_id": str(material_id),
        "structure": structure,
        "Cv_300K": float(cv_per_atom(dos, structure, T=T)),
        "S_300K": float(entropy_per_atom(dos, structure, T=T)),
        "F_300K": float(free_energy_per_atom(dos, structure, T=T)),
        "max_phonon_freq": float(max_phonon_freq(dos)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build per-atom phonon thermo regression targets from the "
            "Materials Project phonon endpoint."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["dfpt", "pheasy"],
        help="Phonon calculation method to query (MP phonon_method).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the pickled DataFrame.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of materials to process.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Reference temperature in K for the thermo targets (default 300).",
    )
    a = parser.parse_args()

    # Lazy import: only the live data-prep path needs mp_api / network / a key.
    # (matinpy's py3.10 env can't even import MPRester; this stays out of the
    # module top so the offline build_row path and its test never touch it.)
    import mp_api
    from mp_api.client import MPRester

    # Log the version we actually got: the API surface differs across mp_api
    # releases, and ACES pip-installs whatever resolves at run time.
    try:
        from importlib.metadata import version as _pkg_version
        print("mp_api", _pkg_version("mp-api"))
    except Exception:
        print("mp_api", getattr(mp_api, "__version__", "?"))

    rows = []
    with MPRester(os.environ.get("MP_API_KEY")) as mpr:
        # Bulk search for (id, structure) in one chunked query. NOTE: the phonon
        # route uses 'identifier' for the MP id (NOT 'material_id'), and it can
        # return 'structure' directly -- using that avoids a second per-material
        # structure call (big speedup for the ~26k pheasy pull). Recent mp_api
        # accepts the phonon_method kwarg; older mp_api does not, so fall back to
        # client-side filtering on the doc's phonon_method enum field.
        try:
            docs = mpr.materials.phonon.search(
                phonon_method=a.method, fields=["identifier", "structure"]
            )
            items = [(str(d.identifier), getattr(d, "structure", None)) for d in docs]
        except TypeError:
            docs = mpr.materials.phonon.search(
                fields=["identifier", "structure", "phonon_method"]
            )
            items = [
                (str(d.identifier), getattr(d, "structure", None))
                for d in docs
                if _method_of(d) == a.method
            ]

        # Apply --limit AFTER the method filter, and use `is not None` so that
        # --limit 0 means zero materials (not "no cap").
        if a.limit is not None:
            items = items[: a.limit]

        for mid, st in items:
            try:
                # DOS fetch. Recent mp_api has a method-aware accessor on the
                # phonon subclient; older mp_api only exposes the no-arg
                # top-level accessor, which returns the DEFAULT method (dfpt).
                # So pheasy (Arm B) selection requires a recent mp_api --
                # confirm the printed version on ACES before the pheasy run.
                try:
                    dos = mpr.materials.phonon.get_dos_from_material_id(
                        mid, phonon_method=a.method
                    )
                except (AttributeError, TypeError):
                    dos = mpr.get_phonon_dos_by_material_id(mid)
                # The method-aware accessor returns an emmet ``PhononDOS`` which
                # lacks the thermodynamic methods (.cv/.entropy/...). Convert it
                # to the pymatgen ``PhononDos`` via ``.to_pmg`` (a property on
                # recent emmet; tolerate a callable too). The no-arg fallback
                # accessor already returns a pymatgen object with .cv.
                if dos is not None and not hasattr(dos, "cv"):
                    to_pmg = getattr(dos, "to_pmg", None)
                    if to_pmg is not None:
                        dos = to_pmg() if callable(to_pmg) else to_pmg
                # Structure came from the bulk search; only fetch per-material as
                # a fallback if the search didn't carry it.
                if st is None:
                    st = mpr.get_structure_by_material_id(mid)
                r = build_row(str(mid), st, dos, a.temperature)
                if r is not None:
                    rows.append(r)
            except Exception as err:  # noqa: BLE001 - skip & continue per material
                print(f"[skip] {mid}: {err}")
                continue

    df = pd.DataFrame(rows)
    target_cols = ["Cv_300K", "S_300K", "F_300K", "max_phonon_freq"]
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    df.to_pickle(a.out)
    print(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
