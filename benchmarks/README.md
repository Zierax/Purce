# Purce Benchmark Suite

## Overview

This directory contains the comprehensive benchmark suite for Purce.

## Running Benchmarks

```bash
# Full numerical accuracy (87,000 iterations, 25 operations)
python -m benchmarks.numerical_accuracy

# Performance scaling curves
python -m benchmarks.performance_scaling

# Edge case robustness
python -m benchmarks.edge_cases

# Code quality metrics
python -m benchmarks.code_metrics

# C code quality analysis
python -m benchmarks.c_code_quality

# Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py

# Detailed benchmark
python benchmarks/detailed_benchmark.py
```

## Benchmark Categories

### 1. Numerical Accuracy (`numerical_accuracy.py`)

Tests correctness of generated C99 against Python reference implementations.

- **25 operations** tested
- **5,000 iterations** per element-wise operation
- **2,000 iterations** per linear algebra operation
- **1,000 iterations** per signal processing operation
- **Tolerance**: rtol=1e-5, atol=1e-8

### 2. Performance Scaling (`performance_scaling.py`)

Measures execution time across input sizes.

- **7 operations** profiled
- **6 input sizes** per operation (16 to 16,384)
- Establishes O(n), O(n log n), O(n^3) scaling curves

### 3. Edge Cases (`edge_cases.py`)

Tests numerical behavior under extreme inputs.

- **NaN, Inf, -Inf, denormalized, overflow, underflow**
- **Zero, negative zero, very large/small values**
- **7 operations** × ~8 edge cases each

### 4. Code Quality (`code_metrics.py`)

Analyzes Purce codebase for quality metrics.

- Lines of code (total, code, comments, blank)
- Cyclomatic complexity per function
- Function lengths
- Import counts

### 5. C Code Quality (`c_code_quality.py`)

Analyzes generated C99 code for structural quality.

- Header presence (PURCE OUTPUT, function headers)
- Include directives
- Line length limits
- Function body presence
- Provenance coverage

### 6. Matrix Multiplication (`benchmark_matmul.py`)

NumPy BLAS-optimized vs generated C99 naive loops.

- Matrix sizes: 10×10 to 500×500
- Compilation time and binary size
- FLOPS and GFLOPS

### 7. Fuzz Testing (via `purce verify`)

Differential fuzzing: 25 operations × N iterations.

- Random input generation
- Python reference as oracle
- Output comparison within tolerance

## Results

See `benchmarks/results.md` for comprehensive results.
