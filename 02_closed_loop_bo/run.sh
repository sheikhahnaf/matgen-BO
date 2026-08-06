#!/usr/bin/env bash
# Regenerate closed-loop discovery figures from in-tree archive/ data (no external downloads).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
G="$HERE/figures/generators"
echo "== Closed-loop BO: regenerating discovery figures (figures/regenerated/) =="
run() { ( cd "$1" && echo "-- $2" && python "$2" >/dev/null ) && echo "   ok" || echo "   ($2 reported errors — see README.md)"; }
run "$G" closed_loop_curves.py
run "$G" closed_loop_extras.py
# Final manuscript versions (dispatched cost basis; main-text Figs 3 & 5, SI Figs S1-S3, S5):
run "$G/dispatched_regen" make_value_cost_dispatched.py
run "$G/dispatched_regen" make_si_curves_corrected.py
run "$G/dispatched_regen" make_si_oracle_dispatched.py
echo "Done. Rendered figures under figures/regenerated/."
