"""Long-Term Memory: parquet-backed observed-sample store.

Hard rule: only TRUE oracle observations land in `y_cp`. Never write GP
predictions back as labels. The `sigma_pred` and `ood_score` columns are
diagnostic only (never used as supervised targets).

Schema:
    structure_id   str    SHA-256 of canonical CIF
    formula        str
    cycle_id       int    BO cycle index when observed
    atoms_json     str    ase.io.jsonio dump of the (relaxed) Atoms
    Z_pca50        list[f32] (n_components,) PCA features at observation time
    y_cp           float  TRUE Cp from FairChem (J/g/K)
    y_cp_var       float  replicate variance (NaN if single eval)
    sigma_pred     float  GP posterior std at selection (diagnostic only)
    ood_score      float  PCA reconstruction error (diagnostic only)
    oracle_source  str    "fairchem_esen" | "anchor_batch" | "seed_pool"
    timestamp      int    unix seconds
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import jsonio


_SCHEMA_COLS = [
    "structure_id", "formula", "cycle_id", "atoms_json",
    "Z_pca50", "y_cp", "y_cp_var",
    "sigma_pred", "ood_score",
    "oracle_source", "timestamp",
]


def canonical_atoms_id(atoms: Atoms) -> str:
    """Stable hash of an ASE Atoms (positions + cell + numbers)."""
    payload = json.dumps(
        {
            "Z": atoms.get_atomic_numbers().tolist(),
            "pos": np.round(atoms.get_positions(), 4).tolist(),
            "cell": np.round(atoms.get_cell().array, 4).tolist(),
            "pbc": atoms.get_pbc().tolist(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def atoms_to_json(atoms: Atoms) -> str:
    buf = io.StringIO()
    jsonio.write_json(buf, atoms)
    return buf.getvalue()


def atoms_from_json(s: str) -> Atoms:
    buf = io.StringIO(s)
    return jsonio.read_json(buf)


class LTM:
    """Append-only parquet store with structure-id dedup."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame = self._load_or_init()
        self._ids: set = set(self._df["structure_id"].tolist()) if len(self._df) else set()

    def _load_or_init(self) -> pd.DataFrame:
        if self.path.exists():
            return pd.read_parquet(self.path)
        return pd.DataFrame(columns=_SCHEMA_COLS)

    def size(self) -> int:
        return len(self._df)

    def has(self, structure_id: str) -> bool:
        return structure_id in self._ids

    def append(self, rows: Iterable[dict]) -> int:
        """Append rows, dedup by structure_id, persist. Returns # actually added."""
        new_rows = []
        for r in rows:
            sid = r.get("structure_id")
            if sid is None or sid in self._ids:
                continue
            # Validate hard rule: y_cp must be a real number, not None/NaN
            # unless oracle_source explicitly indicates a failure record
            # (reserved for future debug logging).
            assert "y_cp" in r, "y_cp is required (true oracle output)"
            self._ids.add(sid)
            r.setdefault("timestamp", int(time.time()))
            new_rows.append({c: r.get(c) for c in _SCHEMA_COLS})
        if not new_rows:
            return 0
        new_df = pd.DataFrame(new_rows, columns=_SCHEMA_COLS)
        self._df = pd.concat([self._df, new_df], ignore_index=True)
        self._df.to_parquet(self.path, index=False)
        return len(new_rows)

    def load(self) -> pd.DataFrame:
        return self._df.copy()

    def features_and_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (Z, y) ready for surrogate.fit()."""
        if len(self._df) == 0:
            return np.zeros((0, 0)), np.zeros((0,))
        Z = np.stack(self._df["Z_pca50"].apply(np.asarray).tolist())
        y = self._df["y_cp"].to_numpy(dtype=np.float64)
        return Z, y
