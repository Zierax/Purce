"""Intensive numerical accuracy benchmarks.

Tests numerical accuracy of generated C99 code against Python/NumPy reference
implementations across a wide range of input distributions and edge cases.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from purce.verifier.fuzzer import DifferentialFuzzer, FuzzResult


@dataclass
class BenchmarkResult:
    name: str
    category: str
    iterations: int
    passed: int
    failed: int
    max_error: float = 0.0
    mean_error: float = 0.0
    duration_ms: float = 0.0
    edge_cases_tested: int = 0
    edge_cases_passed: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.iterations if self.iterations > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def _bench_element_add(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_add(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_add", category="elementwise_binary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_sub(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_sub(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_sub", category="elementwise_binary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_mul(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_mul(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_mul", category="elementwise_binary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_div(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_div(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_div", category="elementwise_binary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_tan(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_tan(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_tan", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_sqrt(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_sqrt(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_sqrt", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_exp(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_exp(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_exp", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_log(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_log(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_log", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_sin(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_sin(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_sin", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_cos(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_cos(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_cos", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_element_abs(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_element_abs(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="element_abs", category="elementwise_unary",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_reduce_sum(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_reduce_sum(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="reduce_sum", category="reduction",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_reduce_mean(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_reduce_mean(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="reduce_mean", category="reduction",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_reduce_max(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_reduce_max(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="reduce_max", category="reduction",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_reduce_min(iterations: int = 5000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_reduce_min(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="reduce_min", category="reduction",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_matmul(iterations: int = 2000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_matmul(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="matmul", category="linear_algebra",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_linalg_solve(iterations: int = 500) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_linalg_solve(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="linalg_solve", category="linear_algebra",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_linalg_inv(iterations: int = 500) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_linalg_inv(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="linalg_inv", category="linear_algebra",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_linalg_cholesky(iterations: int = 500) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_linalg_cholesky(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="linalg_cholesky", category="linear_algebra",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_linalg_eig(iterations: int = 500) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_linalg_eig(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="linalg_eig", category="linear_algebra",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_fft(iterations: int = 1000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_fft(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="fft", category="signal_processing",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_ifft(iterations: int = 1000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_ifft(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="ifft", category="signal_processing",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_alloc_zeros(iterations: int = 2000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_alloc_zeros(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="alloc_zeros", category="allocation",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_alloc_ones(iterations: int = 2000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_alloc_ones(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="alloc_ones", category="allocation",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def _bench_alloc_eye(iterations: int = 2000) -> BenchmarkResult:
    fuzzer = DifferentialFuzzer(seed=42)
    start = time.perf_counter()
    result = fuzzer.fuzz_alloc_eye(iterations)
    elapsed = (time.perf_counter() - start) * 1000
    return BenchmarkResult(
        name="alloc_eye", category="allocation",
        iterations=result.iterations, passed=result.passed, failed=result.failed,
        duration_ms=elapsed,
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    benchmarks = [
        _bench_element_add, _bench_element_sub, _bench_element_mul, _bench_element_div,
        _bench_element_tan, _bench_element_sqrt, _bench_element_exp, _bench_element_log,
        _bench_element_sin, _bench_element_cos, _bench_element_abs,
        _bench_reduce_sum, _bench_reduce_mean, _bench_reduce_max, _bench_reduce_min,
        _bench_matmul, _bench_linalg_solve, _bench_linalg_inv,
        _bench_linalg_cholesky, _bench_linalg_eig,
        _bench_fft, _bench_ifft,
        _bench_alloc_zeros, _bench_alloc_ones, _bench_alloc_eye,
    ]
    results = []
    for bench_fn in benchmarks:
        print(f"  Running {bench_fn.__name__}...")
        results.append(bench_fn())
    return results


def format_results_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "| Operation | Category | Iterations | Passed | Failed | Pass Rate | Duration (ms) |",
        "|-----------|----------|------------|--------|--------|-----------|---------------|",
    ]
    for r in sorted(results, key=lambda x: x.name):
        lines.append(
            f"| {r.name:20s} | {r.category:20s} | {r.iterations:10d} | {r.passed:6d} | {r.failed:6d} | {r.pass_rate:8.1%} | {r.duration_ms:13.1f} |"
        )
    total_iter = sum(r.iterations for r in results)
    total_pass = sum(r.passed for r in results)
    total_fail = sum(r.failed for r in results)
    total_time = sum(r.duration_ms for r in results)
    lines.append(
        f"| **TOTAL** | | **{total_iter}** | **{total_pass}** | **{total_fail}** | **{total_pass/total_iter:.1%}** | **{total_time:.1f}** |"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running intensive benchmarks...")
    results = run_all_benchmarks()
    print()
    print(format_results_table(results))
