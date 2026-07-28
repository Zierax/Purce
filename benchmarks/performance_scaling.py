"""Performance scaling benchmarks: real C99 compilation vs Python reference.

Compiles actual C99 code with gcc -O2 and measures execution time.
Compares against Python list comprehensions and NumPy for reference.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


@dataclass
class ScalingPoint:
    input_size: int
    python_time_us: float
    c99_time_us: float
    numpy_time_us: float | None
    speedup_vs_python: float
    speedup_vs_numpy: float | None


@dataclass
class ScalingResult:
    operation: str
    category: str
    points: list[ScalingPoint]
    python_complexity: str
    c99_complexity: str


def _has_gcc() -> bool:
    try:
        result = subprocess.run(
            ["gcc", "--version"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _compile_c(source: str, optimization: str = "-O2") -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, dir="/tmp") as f:
        f.write(source)
        src_path = f.name
    bin_path = src_path.replace(".c", "_bench")
    try:
        result = subprocess.run(
            ["gcc", "-std=c99", optimization, "-o", bin_path, src_path, "-lm"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            try:
                os.unlink(src_path)
            except OSError:
                pass
            return bin_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        os.unlink(src_path)
    except OSError:
        pass
    return None


def _measure_c_binary(binary_path: str, n_iters: int = 500) -> float:
    try:
        result = subprocess.run(
            [binary_path, str(n_iters)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split("\n")[-1].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return -1.0


def _measure_python(fn, args, iterations: int = 500) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = (time.perf_counter() - start) / iterations
    return elapsed * 1_000_000


C_ELEMENT_ADD = """\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void element_add(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int n = %(n)d;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) { a[i] = (double)rand()/RAND_MAX; b[i] = (double)rand()/RAND_MAX; }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) element_add(a, b, c, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%%.6f\\n", elapsed / iters * 1e6);
    free(a); free(b); free(c);
    return 0;
}
"""

C_ELEMENT_MUL = """\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void element_mul(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] * b[i];
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int n = %(n)d;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) { a[i] = (double)rand()/RAND_MAX; b[i] = (double)rand()/RAND_MAX; }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) element_mul(a, b, c, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%%.6f\\n", elapsed / iters * 1e6);
    free(a); free(b); free(c);
    return 0;
}
"""

C_REDUCE_SUM = """\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double reduce_sum(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int n = %(n)d;
    double *x = malloc(n * sizeof(double));
    srand(42);
    for (int i = 0; i < n; i++) x[i] = (double)rand()/RAND_MAX;
    volatile double s;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) s = reduce_sum(x, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%%.6f\\n", elapsed / iters * 1e6);
    free(x);
    return 0;
}
"""

C_MATMUL = """\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void matmul(const double *A, const double *B, double *C, int n) {
    int i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) C[i*n+j] = 0.0;
    for (i = 0; i < n; i++)
        for (k = 0; k < n; k++) {
            double a_ik = A[i*n+k];
            for (j = 0; j < n; j++) C[i*n+j] += a_ik * B[k*n+j];
        }
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int n = %(n)d;
    double *A = malloc(n*n*sizeof(double));
    double *B = malloc(n*n*sizeof(double));
    double *C = malloc(n*n*sizeof(double));
    srand(42);
    for (int i = 0; i < n*n; i++) { A[i] = (double)rand()/RAND_MAX; B[i] = (double)rand()/RAND_MAX; }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) matmul(A, B, C, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%%.6f\\n", elapsed / iters * 1e6);
    free(A); free(B); free(C);
    return 0;
}
"""

C_FFT = """\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void fft(double *re, double *im, int n) {
    int log_n = 0, tmp = n;
    while (tmp > 1) { tmp >>= 1; log_n++; }
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int bit = 0, mask = n >> 1; bit < log_n; bit++, mask >>= 1)
            if (i & (1 << bit)) j |= mask;
        if (i < j) {
            double t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2;
        double angle = -2.0 * M_PI / (double)size;
        double w_re = cos(angle), w_im = sin(angle);
        for (int i = 0; i < n; i += size) {
            double cur_re = 1.0, cur_im = 0.0;
            for (int j = 0; j < half; j++) {
                int u = i + j, v = i + j + half;
                double t_re = cur_re*re[v] - cur_im*im[v];
                double t_im = cur_re*im[v] + cur_im*re[v];
                re[v] = re[u] - t_re; im[v] = im[u] - t_im;
                re[u] += t_re; im[u] += t_im;
                double nr = cur_re*w_re - cur_im*w_im;
                double ni = cur_re*w_im + cur_im*w_re;
                cur_re = nr; cur_im = ni;
            }
        }
    }
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 500;
    int n = %(n)d;
    double *re = malloc(n * sizeof(double));
    double *im = malloc(n * sizeof(double));
    srand(42);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < n; i++) { re[i] = (double)rand()/RAND_MAX; im[i] = (double)rand()/RAND_MAX; }
        fft(re, im, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%%.6f\\n", elapsed / iters * 1e6);
    free(re); free(im);
    return 0;
}
"""


def _compile_and_measure(c_template: str, n: int, iters: int = 500) -> float:
    source = c_template % {"n": n}
    bin_path = _compile_c(source)
    if not bin_path:
        return -1.0
    result = _measure_c_binary(bin_path, iters)
    try:
        os.unlink(bin_path)
    except OSError:
        pass
    return result


def _try_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        return None


def _bench_element_add() -> ScalingResult:
    np = _try_numpy()
    points = []
    for n in [16, 64, 256, 1024, 4096, 16384]:
        a = [random.uniform(-100, 100) for _ in range(n)]
        b = [random.uniform(-100, 100) for _ in range(n)]
        py_us = _measure_python(lambda a, b: [a[i] + b[i] for i in range(len(a))], (a, b))
        c99_us = _compile_and_measure(C_ELEMENT_ADD, n)
        np_us = None
        if np:
            arr_a = np.array(a)
            arr_b = np.array(b)
            np_us = _measure_python(lambda: np.add(arr_a, arr_b, out=np.empty_like(arr_a)), (), 10000)
        sp_py = py_us / c99_us if c99_us > 0 else 0
        sp_np = py_us / np_us if np_us and np_us > 0 else None
        points.append(ScalingPoint(n, py_us, c99_us, np_us, sp_py, sp_np))
    return ScalingResult("element_add", "elementwise_binary", points, "O(n)", "O(n)")


def _bench_element_mul() -> ScalingResult:
    np = _try_numpy()
    points = []
    for n in [16, 64, 256, 1024, 4096, 16384]:
        a = [random.uniform(-100, 100) for _ in range(n)]
        b = [random.uniform(-100, 100) for _ in range(n)]
        py_us = _measure_python(lambda a, b: [a[i] * b[i] for i in range(len(a))], (a, b))
        c99_us = _compile_and_measure(C_ELEMENT_MUL, n)
        np_us = None
        if np:
            arr_a = np.array(a)
            arr_b = np.array(b)
            np_us = _measure_python(lambda: np.multiply(arr_a, arr_b, out=np.empty_like(arr_a)), (), 10000)
        sp_py = py_us / c99_us if c99_us > 0 else 0
        sp_np = py_us / np_us if np_us and np_us > 0 else None
        points.append(ScalingPoint(n, py_us, c99_us, np_us, sp_py, sp_np))
    return ScalingResult("element_mul", "elementwise_binary", points, "O(n)", "O(n)")


def _bench_reduce_sum() -> ScalingResult:
    np = _try_numpy()
    points = []
    for n in [16, 64, 256, 1024, 4096, 16384]:
        x = [random.uniform(-1000, 1000) for _ in range(n)]
        py_us = _measure_python(lambda x: sum(x), (x,))
        c99_us = _compile_and_measure(C_REDUCE_SUM, n)
        np_us = None
        if np:
            arr_x = np.array(x)
            np_us = _measure_python(lambda: np.sum(arr_x), (), 10000)
        sp_py = py_us / c99_us if c99_us > 0 else 0
        sp_np = py_us / np_us if np_us and np_us > 0 else None
        points.append(ScalingPoint(n, py_us, c99_us, np_us, sp_py, sp_np))
    return ScalingResult("reduce_sum", "reduction", points, "O(n)", "O(n)")


def _bench_matmul() -> ScalingResult:
    np = _try_numpy()
    points = []
    for n in [2, 4, 8, 16, 32, 64]:
        A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
        B = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]

        def mm(A, B, n):
            C = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += A[i][k] * B[k][j]
                    C[i][j] = s
            return C

        iters = 10 if n <= 8 else (5 if n <= 16 else 1)
        py_us = _measure_python(lambda A, B, n: mm(A, B, n), (A, B, n), iters)
        c_iters = max(iters * 10, 100)
        c99_us = _compile_and_measure(C_MATMUL, n, c_iters)
        np_us = None
        if np:
            np_A = np.array(A)
            np_B = np.array(B)
            np_iters = 100 if n <= 16 else 10
            np_us = _measure_python(lambda: np_A @ np_B, (), np_iters)
        sp_py = py_us / c99_us if c99_us > 0 else 0
        sp_np = py_us / np_us if np_us and np_us > 0 else None
        points.append(ScalingPoint(n * n, py_us, c99_us, np_us, sp_py, sp_np))
    return ScalingResult("matmul", "linear_algebra", points, "O(n^3)", "O(n^3)")


def _bench_fft() -> ScalingResult:
    np = _try_numpy()
    points = []
    for exp in [2, 3, 4, 5, 6, 7]:
        n = 2 ** exp
        real = [random.uniform(-10, 10) for _ in range(n)]
        imag = [random.uniform(-10, 10) for _ in range(n)]

        def fft_py(r, i, n):
            if n <= 1:
                return list(r), list(i)
            log_n = 0
            t = n
            while t > 1:
                t >>= 1
                log_n += 1
            r_out = list(r)
            i_out = list(i)
            for ii in range(n):
                j = 0
                for bit in range(log_n):
                    if ii & (1 << bit):
                        j |= n >> (bit + 1)
                if ii < j:
                    r_out[ii], r_out[j] = r_out[j], r_out[ii]
                    i_out[ii], i_out[j] = i_out[j], i_out[ii]
            size = 2
            while size <= n:
                half = size // 2
                angle = -2.0 * math.pi / size
                w_re = math.cos(angle)
                w_im = math.sin(angle)
                for ii in range(0, n, size):
                    cur_re = 1.0
                    cur_im = 0.0
                    for jj in range(half):
                        u = ii + jj
                        v = ii + jj + half
                        t_re = cur_re * r_out[v] - cur_im * i_out[v]
                        t_im = cur_re * i_out[v] + cur_im * r_out[v]
                        r_out[v] = r_out[u] - t_re
                        i_out[v] = i_out[u] - t_im
                        r_out[u] += t_re
                        i_out[u] += t_im
                        nr = cur_re * w_re - cur_im * w_im
                        ni = cur_re * w_im + cur_im * w_re
                        cur_re, cur_im = nr, ni
                size *= 2
            return r_out, i_out

        py_iters = 50 if n <= 256 else (10 if n <= 1024 else 2)
        py_us = _measure_python(lambda r, i, n: fft_py(r, i, n), (real, imag, n), py_iters)
        c_iters = max(py_iters, 50)
        c99_us = _compile_and_measure(C_FFT, n, c_iters)
        np_us = None
        if np:
            np_real = np.array(real)
            np_imag = np.array(imag)
            np_us = _measure_python(lambda: np.fft.fft(np_real + 1j * np_imag), (), 1000)
        sp_py = py_us / c99_us if c99_us > 0 else 0
        sp_np = py_us / np_us if np_us and np_us > 0 else None
        points.append(ScalingPoint(n, py_us, c99_us, np_us, sp_py, sp_np))
    return ScalingResult("fft", "signal_processing", points, "O(n log n)", "O(n log n)")


ALL_BENCHMARKS = [
    _bench_element_add,
    _bench_element_mul,
    _bench_reduce_sum,
    _bench_matmul,
    _bench_fft,
]


def run_all() -> list[ScalingResult]:
    results = []
    for bench_fn in ALL_BENCHMARKS:
        print(f"  Running {bench_fn.__name__}...")
        results.append(bench_fn())
    return results


def format_table(results: list[ScalingResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"### {r.operation} ({r.category}) -- {r.python_complexity}")
        lines.append("")
        has_numpy = any(p.numpy_time_us is not None for p in r.points)
        if has_numpy:
            lines.append("| Input Size | Python (us) | C99 -O2 (us) | NumPy (us) | C99/Python | C99/NumPy |")
            lines.append("|------------|-------------|--------------|------------|------------|-----------|")
        else:
            lines.append("| Input Size | Python (us) | C99 -O2 (us) | C99/Python |")
            lines.append("|------------|-------------|--------------|-----------|")
        for p in r.points:
            np_str = f"{p.numpy_time_us:>12.1f}" if p.numpy_time_us is not None else "         N/A"
            c99_str = f"{p.c99_time_us:>12.1f}" if p.c99_time_us > 0 else "         N/A"
            sp_np = f"{p.speedup_vs_numpy:>9.0f}x" if p.speedup_vs_numpy is not None else "      N/A"
            if has_numpy:
                lines.append(
                    f"| {p.input_size:>10} | {p.python_time_us:>11.1f} | {c99_str} | {np_str} | {p.speedup_vs_python:>9.0f}x | {sp_np} |"
                )
            else:
                lines.append(
                    f"| {p.input_size:>10} | {p.python_time_us:>11.1f} | {c99_str} | {p.speedup_vs_python:>9.0f}x |"
                )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if not _has_gcc():
        print("ERROR: gcc not found. Cannot run real benchmarks.")
        sys.exit(1)
    print("Running performance benchmarks (real gcc -O2 compilation)...")
    results = run_all()
    print()
    print(format_table(results))
