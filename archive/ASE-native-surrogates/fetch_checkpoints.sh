#!/usr/bin/env bash
# Pre-download the three foundation-model checkpoints the benchmark uses
# (MACE-MP-0 medium, ORB v3 conservative-inf-omat, UMA-S-1p1), exactly as
# common.py loads them. UMA is gated on Hugging Face: put HF_TOKEN=hf_xxx in
# a .env next to this script first (see .env.example).
set -e
set -a; [ -f "$(dirname "$0")/.env" ] && . "$(dirname "$0")/.env"; set +a
PY=${PYTHON:-python}

echo "[1/3] MACE-MP-0 (medium) ..."
$PY -c 'from mace.calculators import mace_mp; mace_mp(model="medium", default_dtype="float64", device="cpu"); print("  cached")'

echo "[2/3] ORB v3 conservative-inf-omat ..."
$PY -c 'from orb_models.forcefield import pretrained; pretrained.orb_v3_conservative_inf_omat(device="cpu", precision="float32-high"); print("  cached")'

echo "[3/3] UMA-S-1p1 (gated; needs HF_TOKEN) ..."
$PY -c 'from fairchem.core import pretrained_mlip; pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu"); print("  cached")'

echo "All checkpoints cached."
