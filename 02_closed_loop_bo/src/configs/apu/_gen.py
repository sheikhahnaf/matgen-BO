"""
Config grid generator for APU synthesizability screening.

Deployable feature sets (no DFT-derived features at inference time):
  - orb_pca               : ORB embeddings PCA-reduced
  - orb_pca+magpie        : ORB PCA + Magpie composition features
  - orb_pca+magpie+stab   : ORB PCA + Magpie + stability proxy
  - magpie                : Magpie composition features only
  - orb_pca+cgnf          : ORB PCA + CGNF synthesizability score

Reference feature sets (include DFT mp_props — NOT deployable):
  - orb_pca+mp_props      : xgboost upper-bound reference
  - magpie+mp_props       : xgboost upper-bound reference (composition only)

Grid:
  Deployable:
    5 feature sets × {xgboost, rf, mlp}             = 15 configs (n_bags=20)
    5 feature sets × {nnpu}                          =  5 configs
    4 n_bags variants (n_bags=10 and n_bags=30 for
      two xgboost + two rf configs)                  =  4 configs
    ─────────────────────────────────────────────────
    Total deployable                                  = 24 configs

  Reference (deployable=False):
    2 configs (mp_props variants)
    ─────────────────────────────────────────────────
    Grand total                                       = 26 configs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _feature_tag(features: list[str]) -> str:
    """Short hyphenated tag for a feature-set list."""
    abbrev = {
        "orb_pca": "orb",
        "magpie": "mag",
        "stability": "stab",
        "cgnf_score": "cgnf",
        "mp_props": "mp",
    }
    return "_".join(abbrev.get(f, f) for f in features)


def build_configs() -> list[dict[str, Any]]:
    """Return a list of config dicts for the APU grid."""

    # ------------------------------------------------------------------ #
    # Deployable feature sets (inference-ready, no DFT properties)        #
    # ------------------------------------------------------------------ #
    deploy_fsets: list[list[str]] = [
        ["orb_pca"],
        ["orb_pca", "magpie"],
        ["orb_pca", "magpie", "stability"],
        ["magpie"],
        ["orb_pca", "cgnf_score"],
    ]

    configs: list[dict[str, Any]] = []

    # -- 5 × {xgboost, rf, mlp} = 15 base configs (default n_bags=20) -- #
    for features in deploy_fsets:
        for arch in ("xgboost", "rf", "mlp"):
            ftag = _feature_tag(features)
            name = f"f-{ftag}__a-{arch}"
            configs.append(
                dict(
                    name=name,
                    features=features,
                    arch=arch,
                    pu_scheme="mv_bagging",
                    n_bags=20,
                    deployable=True,
                )
            )

    # -- 5 × {nnpu} = 5 configs (all feature sets include orb_pca or magpie) -- #
    for features in deploy_fsets:
        ftag = _feature_tag(features)
        name = f"f-{ftag}__a-nnpu"
        configs.append(
            dict(
                name=name,
                features=features,
                arch="nnpu",
                pu_scheme="nnpu",
                n_bags=1,        # not used by nnpu, sentinel value
                deployable=True,
            )
        )

    # -- 4 n_bags variants to enrich the xgboost/rf configs -- #
    # n_bags=10 for orb_pca/xgboost and orb_pca+magpie/rf
    # n_bags=30 for magpie/xgboost and orb_pca+magpie+stability/rf
    nbags_variants = [
        (["orb_pca"], "xgboost", 10),
        (["orb_pca", "magpie"], "rf", 10),
        (["magpie"], "xgboost", 30),
        (["orb_pca", "magpie", "stability"], "rf", 30),
    ]
    for features, arch, n_bags in nbags_variants:
        ftag = _feature_tag(features)
        name = f"f-{ftag}__a-{arch}__nb-{n_bags}"
        configs.append(
            dict(
                name=name,
                features=features,
                arch=arch,
                pu_scheme="mv_bagging",
                n_bags=n_bags,
                deployable=True,
            )
        )

    # ------------------------------------------------------------------ #
    # Reference configs (include DFT mp_props — deployable=False)         #
    # ------------------------------------------------------------------ #
    ref_fsets: list[tuple[list[str], str]] = [
        (["orb_pca", "mp_props"], "xgboost"),
        (["magpie", "mp_props"], "xgboost"),
    ]
    for features, arch in ref_fsets:
        ftag = _feature_tag(features)
        name = f"f-{ftag}__a-{arch}__ref"
        configs.append(
            dict(
                name=name,
                features=features,
                arch=arch,
                pu_scheme="mv_bagging",
                n_bags=20,
                deployable=False,
            )
        )

    return configs


# ------------------------------------------------------------------ #
# __main__: write one YAML per config when called with --write        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import yaml  # pyyaml

    write_mode = "--write" in sys.argv

    cfgs = build_configs()
    out_dir = Path(__file__).parent

    if write_mode:
        written = 0
        for cfg in cfgs:
            out_path = out_dir / f"{cfg['name']}.yaml"
            with open(out_path, "w") as fh:
                yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
            written += 1
        print(f"Wrote {written} config YAMLs to {out_dir}")
    else:
        print(f"build_configs() returns {len(cfgs)} configs "
              f"({sum(c['deployable'] for c in cfgs)} deployable, "
              f"{sum(not c['deployable'] for c in cfgs)} reference). "
              f"Pass --write to emit YAMLs.")
