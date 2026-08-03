"""Run all benchmarks and generate comprehensive report."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent


def _run_benchmark(name: str, module: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}\n")
    try:
        result = os.system(f"{sys.executable} -m benchmarks.{module}")
        return result == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> int:
    start = time.perf_counter()
    results = {}

    benchmarks = [
        ("Numerical Accuracy (25 ops, 87k iterations)", "numerical_accuracy"),
        ("Performance Scaling (real gcc -O2)", "performance_scaling"),
        ("Edge Cases (IEEE 754)", "edge_cases"),
        ("Code Quality Metrics", "code_metrics"),
        ("C Code Quality Analysis", "c_code_quality"),
    ]

    for name, module in benchmarks:
        results[name] = _run_benchmark(name, module)

    elapsed = time.perf_counter() - start

    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"\n  Results:")
    for name, passed in results.items():
        status = "OK" if passed else "FAIL"
        print(f"    [{status}] {name}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
