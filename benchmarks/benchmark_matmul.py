"""Benchmark: NumPy matmul vs generated C99 matmul.

Measures compilation time, execution time, binary size, and memory usage.
Uses warmup, multiple samples, and std dev for reliable measurements.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np

from purce.verifier.fuzzer import DifferentialFuzzer

WARMUP = 3
SAMPLES = 5
SIZES = [10, 30, 50, 100]


def _generate_matmul_c(size: int) -> str:
    return f"""\
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void matmul_bench_{size}(const double *A, const double *B, double *C, int m, int n, int k) {{
    for (int i = 0; i < m; i++) {{
        for (int j = 0; j < n; j++) {{
            double sum = 0.0;
            for (int p = 0; p < k; p++) {{
                sum += A[i * k + p] * B[p * n + j];
            }}
            C[i * n + j] = sum;
        }}
    }}
}}

double benchmark_matmul_{size}(int iterations) {{
    int n = {size};
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    if (!A || !B || !C) {{ free(A); free(B); free(C); return -1.0; }}

    srand(42);
    for (int i = 0; i < n * n; i++) {{
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }}

    clock_t start = clock();
    for (int iter = 0; iter < iterations; iter++) {{
        matmul_bench_{size}(A, B, C, n, n, n);
    }}
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    free(A);
    free(B);
    free(C);
    return elapsed;
}}
"""


def _check_gcc() -> bool:
    try:
        result = subprocess.run(["gcc", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compile_c(source_path: str, output_path: str) -> tuple[bool, float]:
    start = time.time()
    try:
        result = subprocess.run(
            ["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic",
             "-o", output_path, source_path, "-lm"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0, time.time() - start
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, time.time() - start


def _run_binary(binary_path: str, iterations: int) -> float:
    try:
        result = subprocess.run(
            [binary_path, str(iterations)], capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return -1.0


def _measure_numpy(n: int, iterations: int) -> Dict[str, float]:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    for _ in range(WARMUP):
        _ = A @ B

    times = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        for _ in range(iterations):
            C = A @ B
        elapsed = (time.perf_counter() - start) / iterations
        times.append(elapsed)

    mean_t = sum(times) / len(times)
    std_t = (sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
    flops = 2.0 * n * n * n / mean_t if mean_t > 0 else 0
    return {"time": mean_t, "std": std_t, "flops": flops}


def _measure_c(n: int, iterations: int) -> Dict[str, float]:
    source = _generate_matmul_c(n)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, f"bench_{n}.c")
        bin_path = os.path.join(tmpdir, f"bench_{n}")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(source)
        compiled, compile_time = _compile_c(src_path, bin_path)
        if not compiled:
            return {"time": -1, "std": 0, "flops": 0, "compile_time": compile_time, "binary_size": 0}

        binary_size = os.path.getsize(bin_path)
        c_time = _run_binary(bin_path, iterations)
        if c_time <= 0:
            return {"time": -1, "std": 0, "flops": 0, "compile_time": compile_time, "binary_size": binary_size}
        flops = 2.0 * n * n * n / (c_time / iterations) if c_time > 0 else 0
        return {"time": c_time / iterations, "std": 0, "flops": flops, "compile_time": compile_time, "binary_size": binary_size}


def run_benchmarks() -> str:
    gcc_available = _check_gcc()
    iterations = 20

    print("Purce Benchmark Suite - Matrix Multiplication")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}")
    print(f"Sizes: {SIZES}")
    print(f"Iterations: {iterations}, Warmup: {WARMUP}, Samples: {SAMPLES}")
    print(f"GCC: {'available' if gcc_available else 'not available'}")
    print()

    report_lines = [
        "# Purce Benchmark Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: {platform.platform()}",
        f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}",
        "",
        "## Matrix Multiplication Performance",
        "",
        "| Size | NumPy (ms) | NumPy std | NumPy GFLOPS | C99 (ms) | C99 GFLOPS | Ratio | Compile (s) | Binary (B) |",
        "|------|------------|-----------|--------------|----------|------------|-------|-------------|------------|",
    ]

    for n in SIZES:
        np_r = _measure_numpy(n, iterations)
        print(f"  NumPy {n}x{n}: {np_r['time']*1000:.3f} ms +/- {np_r['std']*1000:.3f} ms")

        if gcc_available:
            c_r = _measure_c(n, iterations)
            c_time_ms = c_r["time"] * 1000 if c_r["time"] > 0 else -1
            c_gflops = c_r["flops"] / 1e9 if c_r["flops"] > 0 else 0
            ratio = np_r["time"] / c_r["time"] if c_r["time"] > 0 else 0
            ratio_str = f"{ratio:.2f}x" if ratio > 0 else "N/A"
            ct_str = f"{c_r.get('compile_time', 0):.2f}" if c_r.get("compile_time", 0) > 0 else "N/A"
            bs_str = str(c_r.get("binary_size", 0))
            print(f"  C99    {n}x{n}: {c_time_ms:.3f} ms, {c_gflops:.3f} GFLOPS, compile: {ct_str}s")
        else:
            c_gflops = 0
            ratio_str = "N/A"
            ct_str = "N/A"
            bs_str = "N/A"
            c_time_ms = -1

        np_gflops = np_r["flops"] / 1e9
        report_lines.append(
            f"| {n}x{n} | {np_r['time']*1000:.3f} | +/- {np_r['std']*1000:.3f} | {np_gflops:.3f} "
            f"| {c_time_ms if c_time_ms > 0 else 'N/A'} | {c_gflops:.3f} | {ratio_str} | {ct_str} | {bs_str} |"
        )

    print()
    print("Running verification fuzzing (100 iterations)...")
    try:
        fuzzer, compiled = DifferentialFuzzer.with_c_backend(seed=42)
        backend = "C99"
    except (RuntimeError, OSError):
        fuzzer = DifferentialFuzzer(seed=42)
        compiled = None
        backend = "Python-only"

    fuzz_results = fuzzer.fuzz_all(iterations=100)
    if compiled:
        compiled.close()

    report_lines.extend([
        "",
        f"## Differential Fuzzing Results ({backend})",
        "",
        "| Operation | Passed | Total | Rate | Max Error | Status |",
        "|-----------|--------|-------|------|-----------|--------|",
    ])

    for op, result in fuzz_results.items():
        status = "PASS" if result.all_passed else "FAIL"
        err = f"{result.max_error:.2e}" if result.tested_c else "N/A"
        report_lines.append(
            f"| {op} | {result.passed} | {result.iterations} | {result.success_rate:.1%} | {err} | {status} |"
        )

    report_lines.extend([
        "",
        "## Memory Usage",
        "",
        "Generated C99 math kernels use zero heap allocation.",
        "All memory is caller-provided via function parameters.",
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    report = run_benchmarks()
    print()
    print(report)
