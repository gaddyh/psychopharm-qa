#!/usr/bin/env python
"""Run the evaluation suite and produce a release report.

Usage:
    python scripts/run_evals.py [--repeats N] [--datasets golden_v1,abstention_v1]

Defaults:
    --repeats 1
    --datasets golden_v1,abstention_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from psych_qa.evaluation.runner import (
    compute_release_report,
    run_dataset,
    save_report,
)


def main():
    parser = argparse.ArgumentParser(description="Run psychopharm-qa evaluation suite")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Number of times to run each case (for stochastic variance)",
    )
    parser.add_argument(
        "--datasets", type=str, default="golden_v1,abstention_v1",
        help="Comma-separated dataset names to run",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the release report JSON",
    )
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Running dataset: {dataset}")
        print(f"{'='*60}")
        result = run_dataset(dataset, repeats=args.repeats)
        results[dataset] = result

        # Print per-case summary
        for i, case_result in enumerate(result["results"]):
            status = "PASS" if case_result["all_passed"] else "FAIL"
            print(f"  [{status}] Case {i+1}: {case_result['question'][:60]}...")

    # Compute release report
    print(f"\n{'='*60}")
    print("Computing release report...")
    print(f"{'='*60}")

    gold_results = results.get("golden_v1", {})
    abstention_results = results.get("abstention_v1", {})
    regression_results = results.get("regression")

    report = compute_release_report(gold_results, abstention_results, regression_results)

    # Print gate results
    print()
    for gate_name, gate_info in report["gates"].items():
        status = "PASS" if gate_info["passed"] else ("N/A" if gate_info["passed"] is None else "FAIL")
        value = gate_info["value"]
        gate = gate_info["gate"]
        print(f"  [{status}] {gate_name}: {value} (gate: {gate})")
        print(f"         {gate_info['description']}")

    print(f"\n  All computable gates passed: {report['all_computable_gates_passed']}")
    print(f"  Ready for release: {report['ready_for_release']}")

    # Save report
    output_path = Path(args.output) if args.output else None
    saved_path = save_report(report, output_path)
    print(f"\n  Report saved to: {saved_path}")


if __name__ == "__main__":
    main()
