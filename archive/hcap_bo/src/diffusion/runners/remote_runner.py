"""In-env runner — invoked by RemoteGeneratorAdapter via subprocess.

Reads a JSON request:
    {
      "model_name":         <registry key, e.g. "crystalflow">,
      "adapter_kwargs":     {checkpoint: ..., device: ...},
      "n":                  64,
      "chemical_system":    null or [...],
      "property_conditions": null or {...},
      "seed":               null or int
    }

Writes an extxyz file with the generated structures.

Run inside the target conda env:
    <env_prefix>/bin/python -m src.diffusion.runners.remote_runner \\
        --request /tmp/req.json --output /tmp/out.xyz
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(args.request) as f:
        req = json.load(f)

    try:
        from src.diffusion.registry import get_adapter
        Adapter = get_adapter(req["model_name"])
        adapter = Adapter(**(req.get("adapter_kwargs") or {}))
        atoms_list = adapter.sample(
            n=int(req.get("n", 64)),
            chemical_system=req.get("chemical_system"),
            property_conditions=req.get("property_conditions"),
            seed=req.get("seed"),
        )
        from ase.io import write
        write(args.output, atoms_list, format="extxyz")
    except Exception as e:
        print(f"REMOTE RUNNER FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
