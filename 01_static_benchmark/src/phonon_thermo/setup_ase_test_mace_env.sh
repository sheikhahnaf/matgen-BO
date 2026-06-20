#!/bin/bash
# setup_ase_test_mace_env.sh
# Creates /scratch/user/u.sa119259/envs/ase-test-mace with SOAP + MACE + botorch/gpytorch.
# NO fairchem-core, NO orb-models (those go in ase-test env).
# Pins torch to 2.5.1+cu124 so torchvision stays compatible.
# Usage: bash setup_ase_test_mace_env.sh 2>&1 | tee /scratch/user/u.sa119259/ase_test_mace_install.log

set -e

module purge
# WebProxy/0000 ensures pip downloads work on ACES build nodes (no direct outbound route otherwise).
module load Anaconda3/2024.02-1 CUDA/12.4.0 WebProxy/0000

ENVDIR="/scratch/user/u.sa119259/envs/ase-test-mace"
echo "============================================================"
echo "Creating conda env at: $ENVDIR"
echo "============================================================"

conda create -p "$ENVDIR" python=3.11 -y

source activate "$ENVDIR"

echo ""
echo "--- [1/7] PyTorch 2.5.1 + CUDA 12.4 (pinned, no fairchem to bump it) ---"
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "--- [2/7] BoTorch / GPyTorch / linear_operator ---"
pip install botorch gpytorch linear_operator

echo ""
echo "--- [3/7] DScribe (SOAP) ---"
pip install dscribe

echo ""
echo "--- [4/7] MACE ---"
pip install mace-torch

echo ""
echo "--- [5/7] Materials science stack ---"
pip install matminer pymatgen ase spglib

echo ""
echo "--- [6/7] Scientific Python ---"
pip install numpy scipy pandas matplotlib scikit-learn seaborn tqdm

echo ""
echo "--- [7/7] Materials Project API (for data-prep) ---"
pip install mp-api

echo ""
echo "============================================================"
echo "INSTALL COMPLETE — verifying"
echo "============================================================"

python -c "
import torch, botorch, gpytorch, dscribe, mace, matminer, pymatgen, ase, spglib, mp_api
from mace.calculators import mace_mp
print(f'torch:      {torch.__version__}')
print(f'botorch:    {botorch.__version__}')
print(f'gpytorch:   {gpytorch.__version__}')
print(f'dscribe:    {dscribe.__version__}')
print(f'mace:       {mace.__version__}')
print(f'matminer:   {matminer.__version__}')
print(f'pymatgen:   {pymatgen.__version__}')
print(f'ase:        {ase.__version__}')
print(f'spglib:     {spglib.__version__}')
print(f'mp_api:     {mp_api.__version__}')
print(f'mace_mp:    OK')
print(f'CUDA build: {torch.version.cuda}')
"

echo ""
echo "Done! Activate with:"
echo "  module load Anaconda3/2024.02-1 CUDA/12.4.0"
echo "  source activate /scratch/user/u.sa119259/envs/ase-test-mace"
