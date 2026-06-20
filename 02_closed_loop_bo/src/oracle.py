"""Heat-capacity oracle (300 K, J/g/K) — three backends with a unified API.

Backends:
    eSEN (FairChem eSEN-30M-OAM via quacc.phonon_flow): matches upstream
        MatInvent. Most paper-compatible.
    ORB (orb_v3_conservative_inf_omat via phonopy): same model as our featurizer;
        zero new pip-install footprint beyond phonopy.
    UMA (FAIRChem uma-s-1p1 via phonopy): the multi-domain alternative.

Common signature:
    cp, fail_mask = OracleX(env_prefix=...).evaluate(atoms_list)

Subprocess pattern:
    The oracles are called from a runner script via the env's Python directly
    (not `conda run`). This pattern works on FASTER's conda where `conda run`
    mis-resolves the Python binary.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


class _SubprocessOracle:
    """Base: serialize Atoms to xyz, run runner script in target env, read txt."""

    runner_relpath: str = ""  # path relative to this file's directory

    def __init__(
        self,
        env_prefix: str,
        runner_script: Optional[str] = None,
        n_workers: int = 1,
        scratch_dir: Optional[str] = None,
    ):
        self.env_prefix = env_prefix
        # Path.with_name() can't accept paths containing '/'; build via parent.
        self.runner_script = runner_script or str(
            Path(__file__).parent / self.runner_relpath
        )
        self.n_workers = int(n_workers)
        self.scratch_dir = scratch_dir
        if not Path(self.runner_script).exists():
            raise FileNotFoundError(f"Runner script missing: {self.runner_script}")
        if not Path(self.env_prefix).exists():
            raise FileNotFoundError(f"Env prefix missing: {self.env_prefix}")

    def evaluate(self, atoms_list: list[Atoms]) -> tuple[np.ndarray, np.ndarray]:
        if not atoms_list:
            return np.zeros(0), np.zeros(0, dtype=bool)
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmp:
            xyz_path = os.path.join(tmp, "in.xyz")
            out_path = os.path.join(tmp, "out.txt")
            ase_write(xyz_path, atoms_list, format="extxyz")

            python_bin = os.path.join(self.env_prefix, "bin", "python")
            cmd = [python_bin, self.runner_script, xyz_path, out_path, str(self.n_workers)]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "QUACC_RESULTS_DIR": "/tmp"},
            )
            if proc.returncode != 0 and not os.path.isfile(out_path):
                raise RuntimeError(
                    f"{self.__class__.__name__} runner failed (code {proc.returncode}). "
                    f"stderr (tail):\n{proc.stderr.decode()[-2000:]}"
                )
            cp = np.atleast_1d(np.genfromtxt(out_path, dtype=np.float64))
            if cp.shape[0] != len(atoms_list):
                pad = np.full(len(atoms_list) - cp.shape[0], np.nan)
                cp = np.concatenate([cp, pad])[: len(atoms_list)]
            fail_mask = ~np.isfinite(cp)
            # If everything is NaN, surface the runner's stderr for diagnosis.
            if fail_mask.all() and proc.stderr:
                tail = proc.stderr.decode()[-3000:]
                if tail.strip():
                    print(
                        f"[{self.__class__.__name__}] all results NaN — runner stderr (tail):\n{tail}",
                        flush=True,
                    )
            return cp, fail_mask


class HCapOracle_eSEN(_SubprocessOracle):
    """FairChem eSEN-30M-OAM heat-capacity oracle (paper-compatible).

    Implementation: quacc.recipes.mlp.phonons.phonon_flow with method="fairchem",
    model="eSEN-30M-OAM". See src/runners/run_esen.py.
    """
    runner_relpath = "runners/run_esen.py"


class HCapOracle_ORB(_SubprocessOracle):
    """ORB-v3 heat-capacity oracle (uses our featurizer's potential).

    Implementation: phonopy + ASE force calls via ORBCalculator.
    See src/runners/run_orb_phonon.py.
    """
    runner_relpath = "runners/run_orb_phonon.py"


class HCapOracle_UMA(_SubprocessOracle):
    """UMA heat-capacity oracle (FAIRChem uma-s-1p1, task=omat).

    Implementation: phonopy + ASE force calls via FAIRChemCalculator.
    See src/runners/run_uma_phonon.py.
    """
    runner_relpath = "runners/run_uma_phonon.py"


def get_oracle(kind: str, **kwargs) -> _SubprocessOracle:
    kind = kind.lower()
    if kind in ("esen", "fairchem", "esen-30m-oam"):
        return HCapOracle_eSEN(**kwargs)
    if kind == "orb":
        return HCapOracle_ORB(**kwargs)
    if kind == "uma":
        return HCapOracle_UMA(**kwargs)
    raise ValueError(f"Unknown oracle kind: {kind!r} (expected: esen, orb, uma)")
