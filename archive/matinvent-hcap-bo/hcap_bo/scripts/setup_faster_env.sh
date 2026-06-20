#!/usr/bin/env bash
# Build the dedicated conda env on FASTER for this project.
# Run inside a tmux session on FASTER login node:
#   ssh faster
#   tmux new -s hcap_install
#   cd $SCRATCH/matinvent-hcap-bo && bash scripts/setup_faster_env.sh
#   (Ctrl-B D to detach; reattach with `tmux attach -t hcap_install`)

set -euo pipefail

ENV_PREFIX="$SCRATCH/envs/matinvent-hcap-bo"
LOG_FILE="$SCRATCH/matinvent-hcap-bo/logs/env_install.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "[env] Loading modules..." | tee "$LOG_FILE"
module purge
module load GCC/13.2.0 CUDA/12.2.0 Anaconda3/2024.02-1 WebProxy

echo "[env] Creating conda env at $ENV_PREFIX (this takes 20-40 min)" | tee -a "$LOG_FILE"
conda env create -f configs/env.yml -p "$ENV_PREFIX" 2>&1 | tee -a "$LOG_FILE"

echo "[env] Activating..." | tee -a "$LOG_FILE"
source activate "$ENV_PREFIX"

echo "[env] Smoke-test imports..." | tee -a "$LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import torch, botorch, gpytorch
print(f"torch {torch.__version__}  cuda? {torch.cuda.is_available()}")
print(f"botorch {botorch.__version__}  gpytorch {gpytorch.__version__}")
try:
    import orb_models
    print(f"orb_models {orb_models.__version__}")
except Exception as e:
    print(f"orb_models import FAILED: {e}")
try:
    import fairchem
    print(f"fairchem {fairchem.__version__}")
except Exception as e:
    print(f"fairchem import FAILED (may be ok if not installed yet): {e}")
PY

echo "EXIT_CODE:$?" | tee -a "$LOG_FILE"
echo "[env] Done. Log: $LOG_FILE"
