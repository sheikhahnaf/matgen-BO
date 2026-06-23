#!/bin/bash
# Resubmit the Cp-phonon campaign after the 16c/8h batch timed out.
#   - Na2BO3 DROPPED (charge-imbalanced bad generation; DFT E_hull + CALPHAD both reject it).
#   - top03 Li4Mg sg139 already COMPLETE -> skipped.
#   - MID cells (<=36 disp): resume the driver at 64c/24h (banked disps are skipped).
#   - BIG cells (>=72 disp): per-displacement job array (32c each) + a gather job (afterany).
# Run from $SCRATCH/dft_validation:  bash submit_cp_phonon.sh
set -euo pipefail
cd "$SCRATCH/dft_validation"
PI=ph_inputs

# stem (without the cp_ prefix); POSCAR=ph_inputs/cp_<stem>.vasp ; MAG=eos_cp_<stem>/K0.json
MID=(
  top06_cf_ACC_seed23_NaLi2_sg11
  top17_adit_ACC_seed23_LiMg2_sg40
  top02_mg_ACC_seed7_Li4Mg_sg12
  top14_cf_ACC_seed7_LiMg_sg6
  top01_mg_BASE_seed23_Li13Mg3_sg38
  top13_cf_ACC_seed99_Li5CaMg2_sg6
  top04_cf_BASE_seed7_NaLi5_sg1
)
# big stem -> n_displacements
BIG_STEMS=(
  top16_cf_BASE_seed113_Li6TiO5_sg1
  top19_adit_BASE_seed17_Li3PO4_sg1
  top05_mg_BASE_seed23_Li13Mg3Al2_sg1
  top07_mg_BASE_seed113_Li7Mg3_sg1
)
BIG_NDISP=(72 96 108 120)

echo "=== MID cells (resume 64c/24h) ==="
for s in "${MID[@]}"; do
  pos="$PI/cp_$s.vasp"; mag="eos_cp_$s/K0.json"
  [ -f "$pos" ] || { echo "MISSING $pos -- skip"; continue; }
  jid=$(sbatch --parsable --job-name="phr_$s" \
        --export=ALL,POSCAR="$pos",L=12,MAG="$mag" phonon_run.slurm)
  echo "  MID  $s -> $jid"
done

echo "=== BIG cells (per-disp array + gather) ==="
for i in "${!BIG_STEMS[@]}"; do
  s="${BIG_STEMS[$i]}"; n="${BIG_NDISP[$i]}"; last=$((n-1))
  pos="$PI/cp_$s.vasp"; mag="eos_cp_$s/K0.json"
  [ -f "$pos" ] || { echo "MISSING $pos -- skip"; continue; }
  aid=$(sbatch --parsable --job-name="pha_$s" --array=0-${last}%24 \
        --export=ALL,POSCAR="$pos",L=12,MAG="$mag" phonon_array.slurm)
  gid=$(sbatch --parsable --job-name="phg_$s" --dependency=afterany:"$aid" \
        --export=ALL,POSCAR="$pos",L=12,MAG="$mag" phonon_run.slurm)
  echo "  BIG  $s  array=$aid (0-$last %24)  gather=$gid"
done
echo "submitted. watch:  squeue -u \$USER -o '%.10i %.22j %.2t %.4C %R'"
