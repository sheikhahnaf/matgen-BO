#!/bin/bash
# Stream each Zenodo tar (record 20196565) over HTTP and extract ONLY the small
# per-mp-id phonon files (thermal_properties, summary, POSCAR, total_dos) — never
# writes the full ~61.5 GB to disk. Resumable: skips a tarball if a marker exists.
#
# Usage: bash stream_zenodo.sh [tarball1 tarball2 ...]
#   With no args, streams every tarball listed in the manifest.
set -u
ROOT="/scratch/user/u.sa119259/phonon_thermo_benchmark"
BASE="https://zenodo.org/api/records/20196565/files"
EXTRACT="$ROOT/data/zenodo/extract"
DONE="$ROOT/data/zenodo/done"
MAN="$ROOT/data/zenodo/manifest.tsv"
mkdir -p "$EXTRACT" "$DONE"

if [ "$#" -gt 0 ]; then
  TARBALLS="$*"
else
  TARBALLS=$(tail -n +2 "$MAN" | cut -f1)
fi

for tb in $TARBALLS; do
  if [ -f "$DONE/$tb" ]; then echo "[skip-done] $tb"; continue; fi
  echo "[stream] $tb @ $(date +%H:%M:%S)"
  if curl -fsSL "$BASE/$tb/content" \
      | tar -C "$EXTRACT" -x --wildcards --no-anchored \
          '*/06_harmonic_phonons/thermal_properties.yaml.gz' \
          '*/06_harmonic_phonons/summary.json.gz' \
          '*/06_harmonic_phonons/POSCAR.gz' \
          '*/06_harmonic_phonons/total_dos.dat.gz' ; then
    touch "$DONE/$tb"
  else
    echo "[warn] $tb failed (curl/tar); will retry on next run"
  fi
done
echo "stream_zenodo complete: $(ls "$DONE" | wc -l) tarballs done; $(find "$EXTRACT" -name summary.json.gz | wc -l) materials extracted"
