"""Aggregate Arm A (DFPT) holdout CSVs into the long-form aggregated_results.csv.

Non-destructive driver: it IMPORTS the real, already-validated aggregator module
(`analysis_dfpt/aggregate_results.py`) and only redirects the input path to the locally
pulled results and the output into THIS new directory. The original aggregator and its
prior outputs are never touched.
"""
import sys
from pathlib import Path

BASE = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
sys.path.insert(0, str(BASE / "analysis_dfpt"))
import aggregate_results as agg  # noqa: E402

RESULTS = BASE / "aces_results_phonon_heatcap_benchmark_2026-06-18" / "results"
OUT = Path(__file__).parent

df = agg.load_all_results(RESULTS, models=agg.MODELS)
df.to_csv(OUT / "aggregated_results.csv", index=False)
agg.generate_summary_report(df, OUT / "data_summary.txt")
print(f"ARM A (dfpt): {len(df)} records | {df['config'].nunique()} configs "
      f"| models={sorted(df['model'].unique())} | props={sorted(df['property'].unique())}")
