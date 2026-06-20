"""Command-line entrypoint for matinvent-hcap-bo phases.

Usage:
    python -m src.cli phase1_static  --config configs/hcap_bo.yaml --output-dir <out>
    python -m src.cli phase2_open    --config configs/hcap_bo.yaml --output-dir <out>
    python -m src.cli smoke          --config configs/hcap_bo.yaml --output-dir <out>

Phase 1 implementations live in src.phases.phase1_static (added next).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


_PHASES = {
    "phase1_label": "src.phases.phase1_label",
    "phase2_open": "src.phases.phase2_open",
    "phase3_rl": "src.phases.phase3_rl",
    "smoke": "src.phases.smoke",
    "oracle_smoke": "src.phases.oracle_smoke",
    "feat_compare": "src.phases.feat_compare",
    "oracle_3way": "src.phases.oracle_3way",
    "diffusion_smoke": "src.phases.diffusion_smoke",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="matinvent-hcap-bo")
    p.add_argument("phase", choices=list(_PHASES.keys()))
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(_PHASES[args.phase])
    return int(mod.run(config_path=args.config, output_dir=args.output_dir) or 0)


if __name__ == "__main__":
    sys.exit(main())
