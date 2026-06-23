"""Aggregate pheasy Arm B WITH DGP into long-form aggregated_results.csv.
Reuses the source analysis_pheasy/aggregate_results.py module (read-only import);
input = the staged gp+mtgp_2+dgp raw cells; output into arm_b_pheasy_full/. Non-destructive."""
import sys
from pathlib import Path
SRC = Path("/Volumes/SSD1_SMAAA/matinvent-bo/phonon_thermo_benchmark")
sys.path.insert(0, str(SRC / "analysis_pheasy"))
import aggregate_results as agg  # noqa: E402
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parent / "arm_b_pheasy_full"
OUT.mkdir(parents=True, exist_ok=True)
df = agg.load_all_results(RESULTS, models=["gp", "mtgp_2", "dgp"])
df.to_csv(OUT / "aggregated_results.csv", index=False)
try:
    agg.generate_summary_report(df, OUT / "data_summary.txt")
except Exception as e:
    print("summary report skipped:", e)
print("PHEASY+DGP: %d records | models=%s | n_train=%s | props=%s | descs=%s"
      % (len(df), sorted(df['model'].unique()), sorted(df['n_train'].unique()),
         sorted(df['property'].unique()), sorted(df['descriptor'].unique())))
