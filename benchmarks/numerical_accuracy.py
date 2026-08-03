"""Intensive numerical accuracy benchmarks.

Tests all 25 operations with compiled C99 code via ctypes bridge.
Reports per-operation statistics including max error and mean error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from purce.verifier.fuzzer import DifferentialFuzzer, FuzzResult

try:
    from purce.verifier.ctypes_bridge import compile_kernels
    _HAS_GCC = True
except (RuntimeError, OSError):
    _HAS_GCC = False


@dataclass
class BenchmarkResult:
    name: str
    category: str
    iterations: int
    passed: int
    failed: int
    tested_c: bool
    max_error: float
    mean_error: float
    duration_ms: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.iterations if self.iterations > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def _fuzz_op(fuzzer, method_name: str, iters: int, category: str) -> BenchmarkResult:
    fn = getattr(fuzzer, method_name)
    start = time.perf_counter()
    r = fn(iterations=iters)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name=method_name.replace("fuzz_", ""), category=category,
        iterations=r.iterations, passed=r.passed, failed=r.failed,
        tested_c=r.tested_c, max_error=r.max_error, mean_error=r.mean_error,
        duration_ms=elapsed,
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    if _HAS_GCC:
        fuzzer, compiled = DifferentialFuzzer.with_c_backend(seed=42)
        print("Backend: compiled C99 via gcc")
    else:
        fuzzer = DifferentialFuzzer(seed=42)
        compiled = None
        print("Backend: Python-only (gcc not available)")

    ops = [
        ("fuzz_element_add", 5000, "elementwise_binary"),
        ("fuzz_element_sub", 5000, "elementwise_binary"),
        ("fuzz_element_mul", 5000, "elementwise_binary"),
        ("fuzz_element_div", 5000, "elementwise_binary"),
        ("fuzz_element_tan", 5000, "elementwise_unary"),
        ("fuzz_element_sqrt", 5000, "elementwise_unary"),
        ("fuzz_element_exp", 5000, "elementwise_unary"),
        ("fuzz_element_log", 5000, "elementwise_unary"),
        ("fuzz_element_sin", 5000, "elementwise_unary"),
        ("fuzz_element_cos", 5000, "elementwise_unary"),
        ("fuzz_element_abs", 5000, "elementwise_unary"),
        ("fuzz_reduce_sum", 5000, "reduction"),
        ("fuzz_reduce_mean", 5000, "reduction"),
        ("fuzz_reduce_max", 5000, "reduction"),
        ("fuzz_reduce_min", 5000, "reduction"),
        ("fuzz_matmul", 2000, "linear_algebra"),
        ("fuzz_linalg_solve", 500, "linear_algebra"),
        ("fuzz_linalg_inv", 500, "linear_algebra"),
        ("fuzz_linalg_cholesky", 500, "linear_algebra"),
        ("fuzz_linalg_eig", 500, "linear_algebra"),
        ("fuzz_fft", 1000, "signal_processing"),
        ("fuzz_ifft", 1000, "signal_processing"),
        ("fuzz_alloc_zeros", 2000, "allocation"),
        ("fuzz_alloc_ones", 2000, "allocation"),
        ("fuzz_alloc_eye", 2000, "allocation"),
    ]

    results = []
    for method, iters, cat in ops:
        print(f"  Running {method}...")
        results.append(_fuzz_op(fuzzer, method, iters, cat))

    if compiled:
        compiled.close()

    return results


def format_results_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Operation | Category | Iterations | Passed | Rate | Max Error | Mean Error | Tested C | Duration |",
        "|-----------|----------|------------|--------|------|-----------|------------|----------|----------|",
    ]
    for r in sorted(results, key=lambda x: x.name):
        err_max = f"{r.max_error:.2e}" if r.tested_c else "N/A"
        err_mean = f"{r.mean_error:.2e}" if r.tested_c else "N/A"
        c_mark = "Yes" if r.tested_c else "No"
        lines.append(
            f"| {r.name:20s} | {r.category:20s} | {r.iterations:>10d} | {r.passed:>6d} | {r.pass_rate:>5.1%} | {err_max:>10s} | {err_mean:>10s} | {c_mark:>8s} | {r.duration_ms:>8.1f}ms |"
        )
    total_iter = sum(r.iterations for r in results)
    total_pass = sum(r.passed for r in results)
    total_fail = sum(r.failed for r in results)
    total_time = sum(r.duration_ms for r in results)
    rate = total_pass / total_iter if total_iter else 0
    lines.append(
        f"| **TOTAL** | | **{total_iter}** | **{total_pass}** | **{rate:.1%}** | | | | **{total_time:.1f}ms** |"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running numerical accuracy benchmarks (C99 vs Python)...")
    print()
    results = run_all_benchmarks()
    print()
    print(format_results_table(results))
