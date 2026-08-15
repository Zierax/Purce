"""Detailed benchmark: NumPy matmul baseline + differential fuzzing."""
import platform
import sys
import time

import numpy as np

from purce.verifier.fuzzer import DifferentialFuzzer

SIZES = [10, 30, 50, 100, 200, 500]
ITERATIONS = 20
WARMUP = 3
FUZZ_ITERATIONS = 500


def _get_platform_info() -> str:
    lines = [
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}",
        f"NumPy: {np.__version__}",
    ]
    try:
        blas = np.show_config(mode="dict")
        if blas and "BLAS" in str(blas):
            lines.append("BLAS: detected")
    except Exception:
        pass
    return "\n".join(lines)


def _bench_matmul() -> None:
    print("=" * 70)
    print("NUMPY MATMUL BASELINE")
    print("=" * 70)
    print()
    print(_get_platform_info())
    print(f"Matrix sizes: {SIZES}")
    print(f"Iterations per size: {ITERATIONS} (after {WARMUP} warmup)")
    print()
    print("-" * 70)
    print(f"{'Size':<12} {'Time (ms)':<12} {'FLOPS':<15} {'GFLOPS':<12}")
    print("-" * 70)

    for n in SIZES:
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)

        for _ in range(WARMUP):
            _ = A @ B

        times = []
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            C = A @ B
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        flops = 2.0 * n * n * n
        gflops = flops / mean_time / 1e9

        std_time = 0.0
        if len(times) > 1:
            var = sum((t - mean_time) ** 2 for t in times) / (len(times) - 1)
            std_time = var ** 0.5

        print(f"{n}x{n:<8} {mean_time * 1000:>8.3f} +/- {std_time * 1000:>6.3f}  {flops:>13,.0f}  {gflops:>8.3f}")

    print("-" * 70)
    print()


def _bench_fuzzing() -> None:
    print("=" * 70)
    print("DIFFERENTIAL FUZZING (C99 vs Python)")
    print("=" * 70)
    print()

    try:
        fuzzer, compiled = DifferentialFuzzer.with_c_backend(seed=42)
        print("Backend: compiled C99 via gcc")
    except (RuntimeError, OSError):
        fuzzer = DifferentialFuzzer(seed=42)
        compiled = None
        print("Backend: Python-only (gcc not available)")

    print(f"Iterations per operation: {FUZZ_ITERATIONS}")
    print()
    print("-" * 70)
    print(f"{'Operation':<20} {'Iters':<8} {'Passed':<8} {'Rate':<10} {'Max Err':<12} {'Status':<8}")
    print("-" * 70)

    results = fuzzer.fuzz_all(iterations=FUZZ_ITERATIONS)
    total_passed = 0
    total_iters = 0

    for op, result in results.items():
        status = "PASS" if result.all_passed else "FAIL"
        err_str = f"{result.max_error:.2e}" if result.tested_c else "N/A"
        print(f"{op:<20} {result.iterations:<8} {result.passed:<8} {result.success_rate:>8.1%}  {err_str:<12} {status:<8}")
        total_passed += result.passed
        total_iters += result.iterations

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_iters:<8} {total_passed:<8} {total_passed / total_iters if total_iters else 0:>8.1%}")
    print()

    failed_ops = [name for name, r in results.items() if not r.all_passed]
    if failed_ops:
        print(f"FAILED: {', '.join(failed_ops)}")
    else:
        print("All 25 operations passed with compiled C99 code.")

    if compiled:
        compiled.close()


if __name__ == "__main__":
    _bench_matmul()
    _bench_fuzzing()
