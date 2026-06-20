"""RemoteGeneratorAdapter — call any GeneratorAdapter that lives in a different
conda env via subprocess + extxyz IPC.

Same pattern as our LocalESEN oracle (src/oracle.py): env_prefix points at the
conda env that hosts the model; we invoke that env's Python with a thin runner
script (src/diffusion/runners/remote_runner.py), pass the request as JSON, read
the output as extxyz.

Usage in Hydra:
    # configs/diffusion/crystalflow.yaml
    _target_: src.diffusion.RemoteGeneratorAdapter
    env_prefix: ${oc.env:SCRATCH}/envs/mat-zoo-modern
    model_name: crystalflow
    adapter_kwargs:
        checkpoint: ${oc.env:SCRATCH}/checkpoints/crystalflow_mp20.ckpt
        device: cuda

The BO loop in `matinvent-hcap-bo` env can then call any adapter regardless of
where the actual model is installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from src.diffusion.base import GeneratorAdapter


class RemoteGeneratorAdapter(GeneratorAdapter):
    """Cross-env wrapper: invokes another adapter via subprocess in a target conda env."""

    def __init__(
        self,
        env_prefix: str,
        model_name: str,
        adapter_kwargs: Optional[dict] = None,
        scratch_dir: Optional[str] = None,
        runner_script: Optional[str] = None,
        timeout_seconds: int = 1800,
    ):
        self.env_prefix = env_prefix
        self.model_name = model_name
        self.adapter_kwargs = adapter_kwargs or {}
        self.scratch_dir = scratch_dir
        self.timeout_seconds = int(timeout_seconds)
        self.runner_script = runner_script or str(
            Path(__file__).parent / "runners" / "remote_runner.py"
        )
        if not Path(self.runner_script).exists():
            raise FileNotFoundError(f"Runner missing: {self.runner_script}")
        if not Path(self.env_prefix).exists():
            raise FileNotFoundError(f"Env missing: {self.env_prefix}")

        # Pretty name/code/paper inherited from the target adapter via lookup
        self.name = f"remote::{model_name}"
        try:
            from src.diffusion.registry import get_adapter
            target_cls = get_adapter(model_name)
            self.code_url = getattr(target_cls, "code_url", "")
            self.paper_url = getattr(target_cls, "paper_url", "")
        except Exception:
            self.code_url = ""
            self.paper_url = ""

    # ------------------------------------------------------------------

    def sample(
        self,
        n: int = 64,
        chemical_system: Optional[list[str]] = None,
        property_conditions: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> list[Atoms]:
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmp:
            req_path = os.path.join(tmp, "req.json")
            out_path = os.path.join(tmp, "out.xyz")
            with open(req_path, "w") as f:
                json.dump({
                    "model_name": self.model_name,
                    "adapter_kwargs": self.adapter_kwargs,
                    "n": n,
                    "chemical_system": chemical_system,
                    "property_conditions": property_conditions,
                    "seed": seed,
                }, f)

            python_bin = os.path.join(self.env_prefix, "bin", "python")
            cmd = [
                python_bin, self.runner_script,
                "--request", req_path,
                "--output", out_path,
            ]
            # Pass our project root via PYTHONPATH so the target env's Python
            # can find src.diffusion.* (the adapter classes).
            env = {**os.environ}
            proj = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = f"{proj}:{env.get('PYTHONPATH', '')}"
            env["PYTHONNOUSERSITE"] = "1"

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=self.timeout_seconds,
            )
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"Remote {self.model_name} failed (code {proc.returncode}). "
                    f"stderr (tail):\n{proc.stderr.decode()[-2000:]}"
                )

            from ase.io import read
            atoms = read(out_path, index=":")
            if not isinstance(atoms, list):
                atoms = [atoms]
            return atoms

    def supports(self) -> dict:
        # Cross-env query is heavy; mirror the target adapter's static support flags.
        try:
            from src.diffusion.registry import get_adapter
            target_cls = get_adapter(self.model_name)
            # Support-flags are instance methods; for static metadata we expose
            # an empty dict and recommend instantiating to query.
            return {
                "unconditional": True,
                "chemical_system": False,
                "properties": [],
                "disordered": False,
                "remote": True,
            }
        except Exception:
            return {"remote": True, "unconditional": True}
