#!/bin/bash
# Step 0 of the phonon swap: archive the current phonon figures, then drop in the new DFPT ones.
# Archive-then-replace (D5): every original is copied to figures/legacy_phonon_database/ first.
set -u
P=/Users/alvi/fme_paper_work/FoundationalEmbeddings_2026
FIG="$P/figures"
LEG="$FIG/legacy_phonon_database"
A=/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark/paper_figures_new_phonon_2026-06-18/arm_a_dfpt
mkdir -p "$LEG"

archive() { [ -f "$FIG/$1" ] && cp -p "$FIG/$1" "$LEG/$1" && echo "  archived $1"; }
place()   { cp -f "$2" "$FIG/$1" && echo "  placed   $1  <-  $(basename "$2")"; }

echo "=== regenerate the 4 PDF panels from DFPT (re-run the n500 scripts with PDF output) ==="
eval "$(/Users/alvi/miniconda3/bin/conda shell.zsh hook)" && conda activate matinvent
for s in heatmaps_averaged pca_sensitivity radar_charts property_difficulty; do
  perl -pi -e 's/\.png/.pdf/g' "$A/n500/scripts/$s.py"
  python "$A/n500/scripts/$s.py" >/dev/null 2>&1 && echo "  regenerated $s (pdf)" || echo "  FAIL $s"
done

echo "=== archive originals (7 main + old parity + S11) ==="
for f in fig2_bar_phonon_R2_grouped.png fig_difficulty_phonon_n500.pdf fig_heatmap_phonon_R2_n500.pdf \
         fig_pca_phonon_R2_n500.pdf fig_radar_phonon_orb_R2_n500.pdf \
         fig_combined_phonon_R2_learning_curve.png fig_combined_phonon_Spearman_learning_curve.png \
         fig_parity_phonon_phdos_peak_orb_gp.png fig_metric_disagreement.png; do archive "$f"; done

echo "=== place new DFPT figures at the paper filenames ==="
place fig2_bar_phonon_R2_grouped.png            "$A/n500/figures/bar_charts/averaged_R2_n500.png"
place fig_difficulty_phonon_n500.pdf            "$A/n500/figures/property_difficulty/difficulty_matrix_per_surrogate_n500.pdf"
place fig_heatmap_phonon_R2_n500.pdf            "$A/n500/figures/heatmaps/averaged_R2_n500.pdf"
place fig_pca_phonon_R2_n500.pdf                "$A/n500/figures/pca_sensitivity/averaged_R2_n500.pdf"
place fig_radar_phonon_orb_R2_n500.pdf          "$A/n500/figures/radar_charts/orb_R2_n500.pdf"
place fig_combined_phonon_R2_learning_curve.png "$A/combined/figures/aggregated/averaged_R2_learning_curve.png"
place fig_combined_phonon_Spearman_learning_curve.png "$A/combined/figures/aggregated/averaged_Spearman_learning_curve.png"
place fig_parity_phonon_F_300K_orb_gp.png       "$A/parity_orb_gp/parity_F_300K_holdout_split1.png"

echo "=== drop S11 (now unreferenced): move original to legacy ==="
[ -f "$FIG/fig_metric_disagreement.png" ] && mv "$FIG/fig_metric_disagreement.png" "$LEG/" && echo "  moved fig_metric_disagreement.png -> legacy (dropped)"

echo "=== verify the new files are present + right type ==="
for f in fig2_bar_phonon_R2_grouped.png fig_difficulty_phonon_n500.pdf fig_heatmap_phonon_R2_n500.pdf \
         fig_pca_phonon_R2_n500.pdf fig_radar_phonon_orb_R2_n500.pdf \
         fig_combined_phonon_R2_learning_curve.png fig_combined_phonon_Spearman_learning_curve.png \
         fig_parity_phonon_F_300K_orb_gp.png; do
  [ -f "$FIG/$f" ] && echo "  OK  $f  ($(file -b --mime-type "$FIG/$f"))" || echo "  MISSING $f"
done
