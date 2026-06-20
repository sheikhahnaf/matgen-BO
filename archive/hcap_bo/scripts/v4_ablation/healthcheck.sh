#!/usr/bin/env bash
# Health check for the MatterGen ablation sweep (run ON Grace).
# Watches the Hydra results dirs (main.log + samples/long_term_memory.csv),
# NOT the buffered SLURM .out. Verifies the max_num cap once cycles appear.
set -uo pipefail
U="$USER"
PROJ="$SCRATCH/matinvent-hcap-bo"

echo "===== mgabl health @ $(date '+%F %T') ====="
squeue -u "$U" -n mgabl -o '%.12i %.8T %.10M %R' -h | \
  awk '{printf "  job %s  %-9s  t=%-8s  %s\n",$1,$2,$3,$4}'
NQ=$(squeue -u "$U" -n mgabl -h | wc -l)
echo "  in-queue: $NQ / 15"

echo "--- finished (last 4h, non-RUNNING) ---"
sacct -u "$U" --name=mgabl --starttime=now-4hours \
  --format=JobID,State,ExitCode,Elapsed -n 2>/dev/null | \
  grep -vE '\.(batch|extern)' | grep -ivE 'RUNNING|PENDING' | \
  awk 'NF{printf "  %s %s %s %s\n",$1,$2,$3,$4}' | head -20

echo "--- per-run scan (errors / last cycle / cap-applied) ---"
shopt -s nullglob
for base in "$PROJ/results" "$PROJ/results_bm"; do
  for D in "$base"/hcap_mgabl_*; do
    [ -d "$D" ] || continue
    name=$(basename "$D")
    arm=$(echo "$name" | sed -E 's/hcap_mgabl_(cap4|oracleall)_(cp|bm)_seed([0-9]+)_.*/\1\/\2 s\3/')
    log="$D/main.log"; ltm="$D/samples/long_term_memory.csv"
    # error scan
    err=$(grep -iE 'Traceback|CUDA out of memory|RuntimeError|Killed|OutOfMemory|Error:' "$log" 2>/dev/null | tail -1)
    if [ -n "$err" ]; then echo "  [FAIL?] $arm :: ${err:0:90}"; continue; fi
    # progress + cap check from LTM (authoritative: per-RL_step row counts)
    if [ -f "$ltm" ]; then
      read -r ncyc maxadd <<<"$(python3 - "$ltm" <<'PY' 2>/dev/null
import sys,csv,collections
c=collections.Counter()
try:
    with open(sys.argv[1]) as f:
        r=csv.DictReader(f)
        key='RL_step' if 'RL_step' in (r.fieldnames or []) else None
        for row in r:
            if key: c[row[key]]+=1
    print(len(c), max(c.values()) if c else 0)
except Exception:
    print(0,0)
PY
)"
      # cap verdict
      verdict=""
      case "$arm" in
        cap4*)      [ "${maxadd:-0}" -le 4 ] && verdict="cap OK (max add=$maxadd<=4)" || verdict="CAP VIOLATION max add=$maxadd>4" ;;
        oracleall*) [ "${maxadd:-0}" -gt 16 ] && verdict="cap OK (max add=$maxadd>16)" || verdict="check: max add=$maxadd (expect >16 once warm)" ;;
      esac
      echo "  [ok]    $arm :: cycles_logged=$ncyc  $verdict"
    else
      # no LTM yet — show last cycle hint from main.log
      lc=$(grep -oiE 'cycle[ =:]*[0-9]+|epoch[ =:]*[0-9]+|RL[_ ]step[ =:]*[0-9]+' "$log" 2>/dev/null | tail -1)
      echo "  [ok]    $arm :: ${lc:-loading (no LTM yet)}"
    fi
  done
done
echo "===== end ====="
