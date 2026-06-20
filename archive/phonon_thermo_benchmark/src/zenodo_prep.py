"""Build per-atom phonon-thermo targets for the PHEASY arm from the Zenodo deposit.

Source: "Materials Project Phonon database v1.1 - (26,413 compounds)",
Zenodo record 20196565 (the paper's complete dataset; the mp-api mirror is
incomplete). Each material is a directory `data_share_V3/<mp-id>/06_harmonic_phonons/`
containing PRECOMPUTED thermodynamics, so no DOS integration is needed here:

* `thermal_properties.yaml.gz` -- phonopy thermo on a 0..1000 K grid; per UNIT CELL,
  heat_capacity & entropy in J/(K*mol-cell), free_energy in kJ/mol-cell.
* `summary.json.gz` -- scalar metadata: mp_id, nsites, has_imaginary_modes,
  predicted_phonon_stable, max_frequency (THz), e_above_hull, ...
* `POSCAR.gz` -- primitive structure (pymatgen Structure).
* `total_dos.dat.gz` -- 2 cols (freq THz, DOS); used for the DOS-based max phonon
  frequency so it matches the dfpt arm's `max_phonon_freq` definition.

Targets are normalized PER ATOM to match the dfpt arm: Cv/S = per-cell / nsites
(J/(K*mol-atom)); F = per-cell kJ/mol -> J/mol then / nsites (J/mol-atom).
Dynamically unstable materials (imaginary modes) are dropped.
"""

import argparse
import glob
import gzip
import io
import json
import os

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure


def _value_at_T(arr, key, T):
    """Nearest-temperature value for `key` from the thermal_properties list."""
    i = min(range(len(arr)), key=lambda j: abs(arr[j]["temperature"] - T))
    return float(arr[i][key])


def _max_freq_from_dos(path, frac=1e-3):
    """Highest frequency carrying DOS (matches dfpt arm's max_phonon_freq)."""
    data = np.loadtxt(io.BytesIO(gzip.open(path).read()))
    f, d = data[:, 0], data[:, 1]
    mask = d > frac * d.max()
    return float(f[mask].max()) if mask.any() else float(f.max())


def build_row(hdir, T=300):
    """One benchmark row from a `06_harmonic_phonons/` dir, or None to drop."""
    sm = json.loads(gzip.open(os.path.join(hdir, "summary.json.gz")).read())
    # Drop dynamically unstable materials (Cv/S ill-defined with imaginary modes).
    if sm.get("has_imaginary_modes") or sm.get("predicted_phonon_stable") is False:
        return None
    n = sm.get("nsites")
    if not n:
        return None
    tp = yaml.safe_load(gzip.open(os.path.join(hdir, "thermal_properties.yaml.gz")).read())
    arr = tp["thermal_properties"]
    cv = _value_at_T(arr, "heat_capacity", T)   # J/(K*mol-cell)
    s = _value_at_T(arr, "entropy", T)          # J/(K*mol-cell)
    f = _value_at_T(arr, "free_energy", T)      # kJ/mol-cell
    st = Structure.from_str(gzip.open(os.path.join(hdir, "POSCAR.gz")).read().decode(), fmt="poscar")
    try:
        mf = _max_freq_from_dos(os.path.join(hdir, "total_dos.dat.gz"))
    except Exception:
        mf = sm.get("max_frequency")
    row = {
        "material_id": str(sm["mp_id"]),
        "structure": st,
        "Cv_300K": cv / n,            # J/(K*mol-atom)
        "S_300K": s / n,             # J/(K*mol-atom)
        "F_300K": f * 1000.0 / n,    # kJ/mol-cell -> J/mol-atom (matches dfpt arm)
        "max_phonon_freq": float(mf),
    }
    # Drop pathological materials whose thermo is non-finite (e.g. a near-zero
    # mode or a corrupted thermal_properties entry). Phonopy Cv is physically
    # bounded by 3NkB, so an inf/nan signals bad data that would poison the GP.
    if not all(np.isfinite(row[k]) for k in ("Cv_300K", "S_300K", "F_300K", "max_phonon_freq")):
        return None
    return row


def main():
    ap = argparse.ArgumentParser(description="Build pheasy-arm DataFrame from extracted Zenodo dirs.")
    ap.add_argument("--root", required=True, help="Dir containing data_share_V3/<mp-id>/06_harmonic_phonons/")
    ap.add_argument("--out", required=True, help="Output pickled DataFrame path.")
    ap.add_argument("--temperature", type=float, default=300.0)
    a = ap.parse_args()

    hdirs = sorted(glob.glob(os.path.join(a.root, "**", "06_harmonic_phonons"), recursive=True))
    rows, skipped = [], 0
    for h in hdirs:
        try:
            r = build_row(h, a.temperature)
            if r is not None:
                rows.append(r)
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001 - per-material skip-and-continue
            skipped += 1
            print(f"[skip] {h}: {e}")
    df = pd.DataFrame(rows)
    for c in ("Cv_300K", "S_300K", "F_300K", "max_phonon_freq"):
        if c in df.columns:
            df[c] = df[c].astype("float64")
    df.to_pickle(a.out)
    print(f"wrote {len(df)} rows -> {a.out} (from {len(hdirs)} dirs, {skipped} skipped/dropped)")


if __name__ == "__main__":
    main()
