#!/usr/bin/env bash
# Pull results + logs from FASTER scratch back to local for analysis.
set -euo pipefail

LOCAL_DIR="/Volumes/SSD1_SMAAA/matinvent-hcap-bo/"
REMOTE_HOST="faster"
REMOTE_DIR="/scratch/user/ahnafalvi/matinvent-hcap-bo/"

echo "[sync-back] $REMOTE_HOST:$REMOTE_DIR{results,logs} -> $LOCAL_DIR"

rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/results/" "$LOCAL_DIR/results/"
rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"

echo "[sync-back] done."
