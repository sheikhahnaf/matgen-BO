#!/usr/bin/env bash
# Sync this dedicated workspace to FASTER scratch.
# Excludes data/results/logs (large, generated on FASTER).
# Idempotent: safe to re-run.

set -euo pipefail

LOCAL_DIR="/Volumes/SSD1_SMAAA/matinvent-hcap-bo/"
REMOTE_HOST="faster"
REMOTE_DIR="/scratch/user/ahnafalvi/matinvent-hcap-bo/"

echo "[sync] $LOCAL_DIR -> $REMOTE_HOST:$REMOTE_DIR"

rsync -avz --progress \
    --exclude='data/' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='__pycache__/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='.DS_Store' \
    --exclude='*.pyc' \
    "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

echo "[sync] done."
