#!/usr/bin/env python3
"""
run_all.py - Master Orchestration Script for IBM Z Datathon Cyber Threat Detection Pipeline

Executes all seven programs in the correct dependency order:
  1. Program 1: Dataset Analysis & Preprocessing
  2. Program 2: XGBoost Classifier
  3. Program 3: Neural Network Autoencoder
  4. Program 4: Random Forest Classifier
  5. Program 5: Isolation Forest Anomaly Detection
  6. Program 6: Hybrid XGBoost + Autoencoder
  7. Program 7: Final Model Comparison & Results Dashboard

Stops on first error, displays clear progress, and prints final summary.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Program definitions in execution order
PROGRAMS = [
    ("Program 1: Dataset Analysis & Preprocessing", "program_1_data_analysis.py"),
    ("Program 2: XGBoost Classifier", "program_2_xgboost.py"),
    ("Program 3: Neural Network Autoencoder", "program_3_autoencoder.py"),
    ("Program 4: Random Forest Classifier", "program_4_randomforest.py"),
    ("Program 5: Isolation Forest Anomaly Detection", "program_5_isolationforest.py"),
    ("Program 6: Hybrid XGBoost + Autoencoder", "program_6_hybrid_model.py"),
    ("Program 7: Final Model Comparison & Dashboard", "program_7_comparison.py"),
]

BASE_DIR = Path(__file__).parent.resolve()
PYTHON_EXE = sys.executable


def run_program(name: str, filename: str) -> tuple[bool, float]:
    """Execute a single program and return (success, elapsed_seconds)."""
    script_path = BASE_DIR / filename

    if not script_path.exists():
        print(f"\n{'='*75}")
        print(f"ERROR: {name}")
        print(f"Script not found: {script_path}")
        print(f"{'='*75}\n")
        return False, 0.0

    print(f"\n{'='*75}")
    print(f"EXECUTING: {name}")
    print(f"Script: {filename}")
    print(f"{'='*75}")

    start_time = time.time()

    try:
        # Run with the same Python interpreter, unbuffered output
        result = subprocess.run(
            [PYTHON_EXE, "-u", str(script_path)],
            cwd=BASE_DIR,
            capture_output=False,  # Let output stream directly to console
            check=False
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {name} COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n✗ {name} FAILED with exit code {result.returncode} after {elapsed:.1f}s")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {name} FAILED with exception: {e} after {elapsed:.1f}s")
        return False, elapsed


def main():
    print("=" * 75)
    print("IBM Z DATATHON - AI-POWERED CYBER THREAT DETECTION PIPELINE")
    print("Master Orchestration Script (run_all.py)")
    print("=" * 75)
    print(f"Working Directory: {BASE_DIR}")
    print(f"Python Interpreter: {PYTHON_EXE}")

    # Check for required input files
    data_dir = BASE_DIR / "data"
    required_files = [
        "CTIJI ormaI_Traffic.csv",
        "CTU13_Attack_Traffic.csv"
    ]

    print("\nChecking dataset files...")
    missing = []
    for fname in required_files:
        fpath = data_dir / fname
        if fpath.exists():
            print(f"  ✓ Found: {fname} ({fpath.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  ✗ Missing: {fname}")
            missing.append(fname)

    if missing:
        print(f"\nERROR: Required dataset files missing: {missing}")
        print("Please place the CSV files in the data/ directory.")
        sys.exit(1)

    # Execute programs sequentially
    results = []
    total_start = time.time()

    for name, filename in PROGRAMS:
        success, elapsed = run_program(name, filename)
        results.append((name, filename, success, elapsed))

        if not success:
            print("\n" + "!" * 75)
            print("PIPELINE HALTED: A program failed. Fix the error and re-run.")
            print("!" * 75)
            break

    total_elapsed = time.time() - total_start

    # Print final summary
    print("\n" + "=" * 75)
    print("EXECUTION SUMMARY")
    print("=" * 75)
    print(f"{'Status':<8} {'Program':<45} {'Time':>8}")
    print("-" * 75)

    all_passed = True
    for name, filename, success, elapsed in results:
        status = "PASS" if success else "FAIL"
        if not success:
            all_passed = False
        print(f"{status:<8} {name:<45} {elapsed:>7.1f}s")

    print("-" * 75)
    print(f"Total Pipeline Time: {total_elapsed:.1f}s")
    print(f"Overall Status: {'ALL PROGRAMS PASSED ✓' if all_passed else 'FAILURES DETECTED ✗'}")

    if all_passed:
        print("\n" + "=" * 75)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("All artifacts are saved under: outputs/")
        print("=" * 75)
        print("\nKey Output Directories:")
        print("  outputs/dataset_analysis/  - Prepared data splits & EDA charts")
        print("  outputs/xgboost/           - XGBoost model, predictions, metrics")
        print("  outputs/autoencoder/       - Autoencoder model, errors, metrics")
        print("  outputs/randomforest/      - Random Forest model, metrics")
        print("  outputs/isolationforest/   - Isolation Forest model, metrics")
        print("  outputs/hybrid/            - Hybrid model config, metrics, comparisons")
        print("  outputs/comparison/        - Final comparison CSV, JSON, charts")
        print("\nNext Steps:")
        print("  1. Review model_comparison.csv in outputs/comparison/")
        print("  2. Examine grouped_metrics_comparison.png for visual benchmark")
        print("  3. Check summary_report.json for structured results")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()