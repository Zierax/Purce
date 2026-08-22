"""Run all benchmarks and report results."""
import subprocess
import sys
import time


def _run_benchmark(name: str, module: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")
    try:
        result = subprocess.run([sys.executable, "-m", f"benchmarks.{module}"])
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main() -> int:
    start = time.perf_counter()
    results = {}

    benchmarks = [
        ("Performance Scaling (gcc -O2, 10 samples)", "performance_scaling"),
        ("Detailed Benchmark (NumPy + Fuzzing)", "detailed_benchmark"),
        ("Numerical Accuracy (25 ops)", "numerical_accuracy"),
        ("Edge Cases (IEEE 754)", "edge_cases"),
        ("Code Quality Metrics", "code_metrics"),
        ("C Code Quality Analysis", "c_code_quality"),
        ("Matrix Multiplication", "benchmark_matmul"),
        ("Reproducible Corpus Gate (812 C kernels, byte-identical)", "reproducible"),
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
