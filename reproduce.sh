#!/usr/bin/env bash
# Regenerate the paper figures for all three subsystems from in-tree data.
# Requires the `matinvent` conda env (see README / environment.yml). No external downloads.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "matgen-BO reproduce — regenerating figures from in-tree data"
echo "env: $(python -c 'import sys; print(sys.executable)' 2>/dev/null || echo 'activate matinvent first')"
for sub in 01_static_benchmark 02_closed_loop_bo 03_synthesizability; do
  if [ -f "$HERE/$sub/run.sh" ]; then
    echo "=== $sub ==="
    bash "$HERE/$sub/run.sh" || echo "  ($sub/run.sh reported errors — see notes in $sub/README.md)"
  fi
done
echo "Done. Rendered figures are under each 0*/figures/rendered/."
