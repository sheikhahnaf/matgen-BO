#!/bin/bash
# Fetch the still-missing Zenodo tarballs WITHOUT $SCRATCH inode pressure:
# extract each tarball's small phonon files to node-local /tmp, parse immediately
# into a tiny per-tarball pickle on $SCRATCH (1 file each), then delete the /tmp
# files. (The earlier "failures" were $SCRATCH hitting its 250k-inode limit, not
# Zenodo — /tmp is a separate filesystem with its own inodes.)
set -u
ROOT="/scratch/user/u.sa119259/phonon_thermo_benchmark"
BASE="https://zenodo.org/api/records/20196565/files"
DONE="$ROOT/data/zenodo/done"
MAN="$ROOT/data/zenodo/manifest.tsv"
ROWS="$ROOT/data/zenodo/rows"          # one small pickle per tarball (cheap inodes)
PY="/scratch/user/u.sa119259/envs/ase-test-mace/bin/python"
WORK="/tmp/zwork.$$"
mkdir -p "$ROWS"

for tb in $(comm -23 <(tail -n +2 "$MAN" | cut -f1 | sort) <(ls "$DONE" 2>/dev/null | sort)); do
  echo "[inc] $tb $(date +%H:%M:%S)"
  rm -rf "$WORK"; mkdir -p "$WORK"
  if curl -fsSL --retry 5 --retry-delay 10 --connect-timeout 30 "$BASE/$tb/content" \
       | tar -C "$WORK" -x --wildcards --no-anchored \
         "*/06_harmonic_phonons/thermal_properties.yaml.gz" "*/06_harmonic_phonons/summary.json.gz" \
         "*/06_harmonic_phonons/POSCAR.gz" "*/06_harmonic_phonons/total_dos.dat.gz" ; then
    if "$PY" "$ROOT/src/zenodo_prep.py" --root "$WORK" --out "$ROWS/${tb%.tar}.pkl" ; then
      touch "$DONE/$tb"
      echo "  ok: $("$PY" -c "import pandas as pd; print(len(pd.read_pickle('$ROWS/${tb%.tar}.pkl')),'usable rows')")"
    else
      echo "  PARSE-FAIL $tb"
    fi
  else
    echo "  FETCH-FAIL $tb"
  fi
  rm -rf "$WORK"
  sleep 5
done
rm -rf "$WORK"
echo "INC complete: $(ls "$DONE" | wc -l)/87 tarballs done; row-pickles: $(ls "$ROWS"/*.pkl 2>/dev/null | wc -l)"
