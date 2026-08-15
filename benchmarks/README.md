# Purce Benchmark Suite

## Overview

Benchmarks for Purce's generated C99 code. All benchmarks that compare C vs Python use compiled C99 code via ctypes bridge (requires gcc).

## Running Benchmarks

```bash
# Full numerical accuracy (25 ops, C99 vs Python)
python -m benchmarks.numerical_accuracy

# Performance scaling (gcc -O2, warmup + statistics)
python -m benchmarks.performance_scaling

# Edge case robustness (IEEE 754, C99 vs Python)
python -m benchmarks.edge_cases

# Code quality metrics (Python codebase)
python -m benchmarks.code_metrics

# C code quality analysis (generated C99)
python -m benchmarks.c_code_quality

# Matrix multiplication (NumPy vs C99, warmup + samples)
python benchmarks/benchmark_matmul.py

# Full suite
python -m benchmarks.run_all
```

## Benchmark Categories

### 1. Numerical Accuracy (`numerical_accuracy.py`)

Tests all 25 operations with compiled C99 code vs Python reference.

- **25 operations** tested (elementwise, reduction, linear algebra, FFT, allocation)
- **5,000 iterations** per element-wise operation
- Reports max error and mean error per operation
- **Requires gcc** for C backend; falls back to Python-only

### 2. Performance Scaling (`performance_scaling.py`)

Measures gcc -O2 compiled C execution time across input sizes.

- **5 operations** profiled (element_add, element_mul, reduce_sum, matmul, fft)
- **6 input sizes** per operation (16 to 16,384)
- **5 warmup runs, 10 samples** with mean +/- std dev
- Reports GFLOPS and scaling curves
- Cross-platform (Windows clock() / POSIX clock_gettime)

### 3. Edge Cases (`edge_cases.py`)

Tests IEEE 754 edge cases with compiled C99 code.

- **NaN, Inf, -Inf, denormalized, overflow, underflow**
- **Zero, negative zero, very large/small values**
- **7 operations** covering 41 IEEE 754 edge cases (element_add 8, element_mul 7, element_div 7, reduce_sum 6, reduce_max 6, matmul 4, fft 3)
- All edge cases verified against compiled C99

### 4. Code Quality (`code_metrics.py`)

Analyzes Purce Python codebase for quality metrics.

- Lines of code (total, code, comments, blank)
- Cyclomatic complexity per function
- Function lengths and import counts

### 5. C Code Quality (`c_code_quality.py`)

Analyzes generated C99 code for structural quality.

- Header presence (PURCE OUTPUT, function headers)
- Include directives and line length limits
- Function length statistics (avg, max)
- Heap allocation detection (malloc/free/calloc/realloc)
- Zero-heap check validation

### 6. Matrix Multiplication (`benchmark_matmul.py`)

NumPy BLAS-optimized vs generated C99 naive loops.

- Matrix sizes: 10x10 to 100x100
- **3 warmup runs, 5 samples** with mean +/- std dev
- Compilation time and binary size
- GFLOPS comparison
- Differential fuzzing verification

### 7. Full Suite (`run_all.py`)

Runs all benchmarks in sequence with timing.

```bash
python -m benchmarks.run_all
```

### 7b. Real-World Pipeline Benchmark (`real_benchmark.py`)

Runs the full pipeline over the 23 real-world sources in `tests/realworld/` and writes
the results to `benchmarks/results.md` (per-file node counts, generated C files, gcc
compile status, Z3 verification, fuzz results, aggregated totals).

```bash
python -m benchmarks.real_benchmark
```

### 8. Reproducible Corpus Gate (`reproducible.py`)

Extracts the 14 harder real-world kernels in `benchmarks/corpus/`, fingerprints the
full environment, compiles every generated `.c` with strict gcc, hashes the
normalized content, and verifies drift against the committed `benchmarks/baseline.json`.

```bash
python -m benchmarks.reproducible --baseline benchmarks/baseline.json
```

Extraction output is byte-reproducible across processes: caller→callee edges are
iterated in sorted order, so the emitted `nested_deps` (and provenance) no longer
depend on the per-process `PYTHONHASHSEED`.
