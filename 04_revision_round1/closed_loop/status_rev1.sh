#!/bin/bash
# One-screen status of the revision runs: queue, finished runs per arm, failed jobs.
#   REV_ROOT=... bash status_rev1.sh
REV_ROOT="${REV_ROOT:?set REV_ROOT}"
echo "== $(date '+%F %T')"
echo "== queue (this user's rev1/gpcal jobs)"
squeue -u "$USER" -h -o "%j %T" 2>/dev/null | awk '$1 ~ /^(rev1|gpcal|candtab)/ {c[$2]++} END {for (k in c) printf "  %s %d\n", k, c[k]}'
echo "== runs: finished / started, per arm and target"
for d in "$REV_ROOT"/runs/rev1_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    key=$(echo "$n" | sed -E 's/^rev1_(.+)_(mg|cf|adit)_(cp|bm)_seed[0-9]+_[0-9]+$/\1 \2 \3/')
    done_flag=0; grep -q "RL END" "$d/main.log" 2>/dev/null && done_flag=1
    loops=$(grep -c "LOOP [0-9]* FINISH" "$d/main.log" 2>/dev/null)
    echo "$key $done_flag $loops"
done | awk '{k=$1" "$2" "$3; s[k]++; f[k]+=$4; l[k]+=$5} END {for (k in s) printf "  %-26s finished %d / started %d   cycles done %d\n", k, f[k], s[k], l[k]}' | sort
echo "== jobs that ended without finishing (last 2 days)"
sacct -u "$USER" -S now-2days -X -n --format=JobID,JobName%34,State%14,Elapsed 2>/dev/null \
    | awk '$2 ~ /^(rev1|gpcal)/ && $3 !~ /COMPLETED|RUNNING|PENDING/' | tail -n 15
echo "== calibration cells done"
for k in elastic dielectric dfpt; do
    echo "  $k $(ls "$REV_ROOT"/calibration/out/$k/pca*_n*/calib_preds_*.csv.gz 2>/dev/null | wc -l) / 36"
done
