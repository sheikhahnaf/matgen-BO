#!/bin/bash
# setup_ase_test_env.sh
# Creates $SCRATCH/envs/ase-test with all featurizers for ASE regression tests.
# Run inside tmux on Grace login node.
# Usage: bash setup_ase_test_env.sh 2>&1 | tee $SCRATCH/ase_test_install.log

set -e

module purge
module load Anaconda3/2024.02-1 CUDA/12.4.0

ENVDIR="$SCRATCH/envs/ase-test"
echo "============================================================"
echo "Creating conda env at: $ENVDIR"
echo "============================================================"

conda create -p "$ENVDIR" python=3.11 -y

source activate "$ENVDIR"

echo ""
echo "--- [1/8] PyTorch 2.5.1 + CUDA 12.4 ---"
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "--- [2/8] torch-geometric ecosystem (for FAIRChem) ---"
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install torch-geometric

echo ""
echo "--- [3/8] BoTorch / GPyTorch / linear_operator ---"
pip install botorch gpytorch linear_operator

echo ""
echo "--- [4/8] DScribe (SOAP) ---"
pip install dscribe

echo ""
echo "--- [5/8] MACE ---"
pip install mace-torch

echo ""
echo "--- [6/8] ORB ---"
pip install orb-models

echo ""
echo "--- [7/8] FAIRChem (UMA / eSEN) ---"
pip install fairchem-core

echo ""
echo "--- [8/8] Materials science + scientific Python stack ---"
pip install matminer pymatgen ase spglib
pip install numpy scipy pandas matplotlib scikit-learn seaborn tqdm

echo ""
echo "============================================================"
echo "INSTALL COMPLETE — verifying key packages"
echo "============================================================"
python - <<'EOF'
import importlib, sys

packages = [
    ("torch",             "torch.__version__"),
    ("botorch",           "botorch.__version__"),
    ("gpytorch",          "gpytorch.__version__"),
    ("dscribe",           "dscribe.__version__"),
    ("mace",              "mace.__version__"),
    ("orb_models",        "orb_models.__version__"),
    ("fairchem.core",     "fairchem.core.__version__"),
    ("matminer",          "matminer.__version__"),
    ("pymatgen",          "pymatgen.__version__"),
    ("ase",               "ase.__version__"),
]

ok = True
for mod, ver_expr in packages:
    try:
        m = importlib.import_module(mod)
        ver = eval(ver_expr)
        print(f"  OK  {mod:<25} {ver}")
    except Exception as e:
        print(f"  ERR {mod:<25} {e}")
        ok = False

import torch
print(f"\n  CUDA available: {torch.cuda.is_available()}")
print(f"  PyTorch CUDA build: {torch.version.cuda}")

sys.exit(0 if ok else 1)
EOF

echo ""
echo "Done! Activate with:"
echo "  module load Anaconda3/2024.02-1 CUDA/12.4.0"
echo "  source activate \$SCRATCH/envs/ase-test"
