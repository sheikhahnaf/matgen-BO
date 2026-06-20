"""
Run All Combined Analysis Scripts

This script orchestrates the complete combined learning curve analysis:
1. Prepare learning curves data (ORB only, all n_train values)
2. Plot per-property learning curves (24 PNGs)
3. Plot aggregated learning curves (3 PNGs)

Prerequisites:
- n100, n250, and n500 analyses must be completed first
- Each must have best_pca_per_property.csv in their data/ directory

Total output: 1 CSV + 27 PNGs
"""

import subprocess
import sys
import os

def run_script(script_name, script_dir):
    """
    Run a Python script and check for errors

    Args:
        script_name: Name of script to run
        script_dir: Directory containing the script
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print('='*80)

    result = subprocess.run(
        [sys.executable, script_name],
        cwd=script_dir
    )

    if result.returncode != 0:
        print(f"\n✗ ERROR: {script_name} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n✓ {script_name} completed successfully")

def main():
    print("="*80)
    print("COMBINED LEARNING CURVE ANALYSIS")
    print("="*80)
    print("\nThis will:")
    print("  1. Prepare learning curves data (ORB only)")
    print("  2. Generate 24 per-property learning curve plots")
    print("  3. Generate 3 aggregated learning curve plots")
    print("\nTotal output: 1 CSV + 27 PNGs")

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Scripts to run in order
    scripts = [
        'prepare_learning_curves.py',
        'plot_per_property.py',
        'plot_aggregated.py'
    ]

    # Run each script
    for script in scripts:
        run_script(script, script_dir)

    # Final summary
    print("\n" + "="*80)
    print("✓ COMBINED ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput files:")
    print("  Data: ../data/learning_curves_orb.csv")
    print("  Per-property figures: ../figures/per_property/ (24 PNGs)")
    print("  Aggregated figures: ../figures/aggregated/ (3 PNGs)")
    print("\nTotal: 1 CSV + 27 PNGs")

if __name__ == '__main__':
    main()
