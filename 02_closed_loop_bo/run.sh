#!/usr/bin/env bash
# Regenerate closed-loop discovery figures from in-tree archive/ data (no external downloads).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/figures/generators"
echo "== Closed-loop BO: regenerating discovery figures (figures/regenerated/) =="
run() { ( cd "$1" && echo "-- $2" && python "$2" >/dev/null ) && echo "   ok" || echo "   ($2 reported errors — see README.md)"; }
run "$G" closed_loop_curves.py
run "$G" closed_loop_extras.py
echo "Done. Rendered figures under figures/regenerated/."
