"""Aggregate Arm B (pheasy) holdout CSVs into the long-form aggregated_results.csv.

Non-destructive driver: imports the real `analysis_pheasy/aggregate_results.py` module
(MODELS = ['gp','mtgp_2'], no DGP) and only redirects input to the pulled results and output
into THIS new directory. Originals are never modified.
"""
import sys
from pathlib import Path

BASE = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
sys.path.insert(0, str(BASE / "analysis_pheasy"))
import aggregate_results as agg  # noqa: E402

RESULTS = BASE / "aces_results_phonon_heatcap_benchmark_2026-06-18" / "results"
OUT = Path(__file__).parent

df = agg.load_all_results(RESULTS, models=agg.MODELS)
df.to_csv(OUT / "aggregated_results.csv", index=False)
agg.generate_summary_report(df, OUT / "data_summary.txt")
print(f"ARM B (pheasy): {len(df)} records | {df['config'].nunique()} configs "
      f"| models={sorted(df['model'].unique())} | props={sorted(df['property'].unique())}")
