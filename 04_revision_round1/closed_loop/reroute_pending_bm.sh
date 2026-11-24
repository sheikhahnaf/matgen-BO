#!/bin/bash
# Operational helper: move this user's PENDING bulk-modulus revision jobs from the
# T4 request to other GPU types, to shorten the queue. Running jobs are not touched.
# The bulk-modulus oracle does not hit GPU memory limits, so the GPU type does not
# change which candidates return a value. Heat-capacity jobs stay on T4.
#
#   REV_ROOT=... ACCOUNT=... GRES_BM="gpu:a10:1 gpu:a30:1" bash reroute_pending_bm.sh

set -euo pipefail
REV_ROOT="${REV_ROOT:?}"; ACCOUNT="${ACCOUNT:?}"; : "${GRES_BM:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
squeue -u "$USER" -h -t PENDING -o "%i %j" | awk '$2 ~ /^rev1_.*_bm_s[0-9]+$/' > "$REV_ROOT/reroute_list.txt" || true
n=$(wc -l < "$REV_ROOT/reroute_list.txt")
echo "pending bulk-modulus jobs to move: $n"
[ "$n" -eq 0 ] && exit 0
read -r -a LIST <<< "$GRES_BM"
i=0
while read -r jid name; do
    if [[ "$name" =~ ^rev1_(.+)_(mg|cf|adit)_bm_s([0-9]+)$ ]]; then
        arm="${BASH_REMATCH[1]}"; bb="${BASH_REMATCH[2]}"; seed="${BASH_REMATCH[3]}"
        wall=24:00:00; [ "$bb" = "mg" ] && wall=48:00:00
        scancel "$jid"
        GRES_BM="${LIST[$((i % ${#LIST[@]}))]}" REV_ROOT="$REV_ROOT" ACCOUNT="$ACCOUNT" \
            bash "$HERE/submit_rev1.sh" one "$arm" "$bb" bm "$seed" 20 "$wall"
        i=$((i + 1))
    fi
done < "$REV_ROOT/reroute_list.txt"
