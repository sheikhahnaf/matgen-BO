"""
Phase 7: Master Execution Script

This script runs all analyses in sequence:
0. Data Preparation
1. Bar Charts (Averaged)
2. Bar Charts (Per Property)
3. Heatmaps (Averaged)
4. Property Difficulty Matrix
5. PCA Sensitivity Analysis
6. Radar Charts (ORB Only)
"""

import subprocess
import sys
import os
import time

def print_header(phase_num, phase_name):
    """Print a nice header for each phase"""
    print("\n" + "="*80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("="*80)

def run_script(script_name, phase_num, phase_name):
    """
    Run a Python script and report success/failure

    Args:
        script_name: Name of the script to run
        phase_num: Phase number (for display)
        phase_name: Phase name (for display)

    Returns:
        True if successful, False otherwise
    """
    print_header(phase_num, phase_name)

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )

        # Print output
        print(result.stdout)

        elapsed = time.time() - start_time
        print(f"\n✓ Phase {phase_num} completed successfully in {elapsed:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in Phase {phase_num}:")
        print(e.stdout)
        print(e.stderr)
        elapsed = time.time() - start_time
        print(f"\nPhase {phase_num} failed after {elapsed:.1f}s")
        return False

def print_summary(results):
    """Print summary of all phases"""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    total_phases = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total_phases - successful

    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} Phase {result['num']}: {result['name']} "
             f"({result['time']:.1f}s)")

    print("\n" + "="*80)
    print(f"Total: {successful}/{total_phases} phases completed successfully")
    if failed > 0:
        print(f"       {failed} phase(s) failed")
    print("="*80)

def main():
    """Run all analysis phases"""
    print("="*80)
    print("SIMPLIFIED ASE REGRESSION ANALYSIS (n=250)")
    print("Running all phases...")
    print("="*80)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Define all phases
    phases = [
        (0, "prepare_data.py", "Data Preparation"),
        (1, "bar_charts_averaged.py", "Bar Charts (Averaged)"),
        (2, "bar_charts_per_property.py", "Bar Charts (Per Property)"),
        (3, "heatmaps_averaged.py", "Heatmaps (Averaged)"),
        (4, "property_difficulty.py", "Property Difficulty Matrix"),
        (5, "pca_sensitivity.py", "PCA Sensitivity Analysis"),
        (6, "radar_charts.py", "Radar Charts (ORB Only)")
    ]

    # Track results
    results = []
    overall_start = time.time()

    # Run each phase
    for phase_num, script_name, phase_name in phases:
        start_time = time.time()
        success = run_script(script_name, phase_num, phase_name)
        elapsed = time.time() - start_time

        results.append({
            'num': phase_num,
            'name': phase_name,
            'success': success,
            'time': elapsed
        })

        if not success:
            print(f"\n✗ Stopping execution due to failure in Phase {phase_num}")
            break

    overall_elapsed = time.time() - overall_start

    # Print summary
    print_summary(results)
    print(f"\nTotal execution time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")

    # Check if all succeeded
    if all(r['success'] for r in results):
        print("\n" + "="*80)
        print("ALL ANALYSES COMPLETE! ✓")
        print("="*80)
        print("\nGenerated outputs:")
        print("  - Data: analysis_v2/data/")
        print("  - Figures: analysis_v2/figures/")
        return 0
    else:
        print("\n" + "="*80)
        print("ANALYSIS INCOMPLETE - Some phases failed")
        print("="*80)
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
