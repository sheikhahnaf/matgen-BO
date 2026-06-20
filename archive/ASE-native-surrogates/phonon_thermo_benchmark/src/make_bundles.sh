#!/bin/bash
# Build two download bundles on ACES node-local /tmp (not $SCRATCH, so no inode cost):
#   1) metrics_bundle.tgz  -- CSVs + SLURM logs + source pickles, in a clean
#      results/ + slurm_logs/ + data_pickles/ layout (the analysis-essential ~16 MB).
#   2) bench_bundle.tgz     -- the FULL result tree incl. all 12.5k diagnostic PNGs (~5 GB).
set -u
RR=/scratch/user/u.sa119259/phonon_thermo_benchmark
cd "$RR" || exit 1

# --- 1) metrics-only bundle (clean subfolder layout) ---
rm -rf /tmp/mb
mkdir -p /tmp/mb/results /tmp/mb/slurm_logs /tmp/mb/data_pickles
while IFS= read -r f; do
  rel="${f#results/}"
  mkdir -p "/tmp/mb/results/$(dirname "$rel")"
  cp "$f" "/tmp/mb/results/$rel"
done < <(find results -name '*_holdout_summary.csv')
cp *pheasy_*.out *pheasy_*.err gapfill_dfpt_*.out gapfill_dfpt_*.err /tmp/mb/slurm_logs/ 2>/dev/null
cp data/dfpt_phonon_thermo.pkl data/pheasy_phonon_thermo.pkl /tmp/mb/data_pickles/
tar czf /tmp/metrics_bundle.tgz -C /tmp/mb .

# --- 2) full archive (everything, incl. plots) ---
rm -f /tmp/bench_bundle.tgz
tar czf /tmp/bench_bundle.tgz results data/dfpt_phonon_thermo.pkl data/pheasy_phonon_thermo.pkl \
    *pheasy_*.out *pheasy_*.err gapfill_dfpt_*.out gapfill_dfpt_*.err 2>/dev/null

echo "METRICS_TGZ: $(ls -la /tmp/metrics_bundle.tgz)"
echo "FULL_TGZ:    $(ls -la /tmp/bench_bundle.tgz)"
echo "metrics_csv_count: $(tar tzf /tmp/metrics_bundle.tgz | grep -c holdout_summary.csv)"
echo "BUNDLES_DONE"
