# Purce Benchmark Results

Generated: 2026-08-03 17:29:42
Python: 3.11.15

## Pipeline Performance

Time to process Python source through full pipeline: parse → IR → slice → generate C99.

| Source | Nodes | C Files | Clean C | #error C | C Lines | Time (ms) |
|--------|-------|---------|---------|----------|---------|-----------|
| fixtures/ (3 files) | 5 | 5 | 3 | 2 | 59 | 2.4 |
| realworld/ (15 files) | 141 | 141 | 95 | 46 | 1058 | 74.8 |

### Per-File Breakdown

| File | Nodes | C Files | Clean | #error | C Lines | Time (ms) |
|------|-------|---------|-------|--------|---------|-----------|
| activations.py | 15 | 15 | 5 | 10 | 100 | 3.9 |
| attention.py | 7 | 7 | 3 | 4 | 56 | 3.4 |
| convolution.py | 5 | 5 | 5 | 0 | 30 | 2.8 |
| data_pipeline.py | 10 | 10 | 5 | 5 | 64 | 3.7 |
| extra_patterns.py | 0 | 0 | 0 | 0 | 0 | 0.2 |
| jax_ops.py | 16 | 16 | 11 | 5 | 141 | 5.1 |
| layers.py | 10 | 10 | 7 | 3 | 82 | 4.4 |
| linear_algebra.py | 11 | 11 | 11 | 0 | 97 | 4.7 |
| losses.py | 10 | 10 | 6 | 4 | 71 | 4.1 |
| model.py | 6 | 6 | 2 | 4 | 38 | 3.6 |
| normalization.py | 5 | 5 | 4 | 1 | 50 | 2.6 |
| optimizers.py | 6 | 6 | 6 | 0 | 49 | 8.8 |
| pytorch_ops.py | 16 | 16 | 8 | 8 | 113 | 5.1 |
| scipy_ops.py | 15 | 15 | 14 | 1 | 99 | 7.7 |
| signal_processing.py | 11 | 11 | 8 | 3 | 80 | 5.2 |

## Verification Results

### Z3 SMT Bounds Checking

- Total nodes: 141
- Verified (safe): 80
- Violated (potential overflow): 59
- Unknown (Z3 timeout): 2

### Differential Fuzzing (Python-only, 1000 iterations)

| Operation | Passed | Total | Rate | Max Error | Status |
|-----------|--------|-------|------|-----------|--------|
| alloc_eye | 500 | 500 | 100.0% | N/A | PASS |
| alloc_ones | 500 | 500 | 100.0% | N/A | PASS |
| alloc_zeros | 500 | 500 | 100.0% | N/A | PASS |
| element_abs | 1000 | 1000 | 100.0% | N/A | PASS |
| element_add | 1000 | 1000 | 100.0% | N/A | PASS |
| element_cos | 1000 | 1000 | 100.0% | N/A | PASS |
| element_div | 1000 | 1000 | 100.0% | N/A | PASS |
| element_exp | 1000 | 1000 | 100.0% | N/A | PASS |
| element_log | 1000 | 1000 | 100.0% | N/A | PASS |
| element_mul | 1000 | 1000 | 100.0% | N/A | PASS |
| element_sin | 1000 | 1000 | 100.0% | N/A | PASS |
| element_sqrt | 1000 | 1000 | 100.0% | N/A | PASS |
| element_sub | 1000 | 1000 | 100.0% | N/A | PASS |
| element_tan | 1000 | 1000 | 100.0% | N/A | PASS |
| fft | 500 | 500 | 100.0% | N/A | PASS |
| ifft | 500 | 500 | 100.0% | N/A | PASS |
| linalg_cholesky | 200 | 200 | 100.0% | N/A | PASS |
| linalg_eig | 200 | 200 | 100.0% | N/A | PASS |
| linalg_inv | 200 | 200 | 100.0% | N/A | PASS |
| linalg_solve | 200 | 200 | 100.0% | N/A | PASS |
| matmul | 1000 | 1000 | 100.0% | N/A | PASS |
| reduce_max | 1000 | 1000 | 100.0% | N/A | PASS |
| reduce_mean | 1000 | 1000 | 100.0% | N/A | PASS |
| reduce_min | 1000 | 1000 | 100.0% | N/A | PASS |
| reduce_sum | 1000 | 1000 | 100.0% | N/A | PASS |

All 25 operations passed differential fuzzing.

## Generated Code Quality

- Total .c files: 141
- Files with #error (unsupported ops): 46
- Clean compilable files: 95
- Total C lines (excluding comments): 1058
- Total function definitions: 141
- Average lines per function: 8

## Operation Coverage

### Supported (generates valid C99)

| Operation | Algorithm | Kernel Count |
|-----------|-----------|--------------|
| various | alloc_eye | 6 |
| various | alloc_ones | 1 |
| various | alloc_zeros | 34 |
| various | element_abs | 1 |
| various | element_add | 9 |
| various | element_div | 2 |
| various | element_exp | 1 |
| various | element_log | 3 |
| various | element_mul | 16 |
| various | element_sqrt | 3 |
| various | element_sub | 16 |
| various | matmul | 6 |
| various | reduce_mean | 9 |
| various | reduce_sum | 3 |

### Unsupported (generates #error)

- Total nodes with 'unknown' algorithm: 31
- These contain operations like: numpy.power, numpy.maximum, numpy.minimum, numpy.where, numpy.clip, etc.
- Adding these to MATH_KERNEL_BODIES would eliminate the #error directives.

## Summary

| Metric | Value |
|--------|-------|
| Python source files processed | 18 |
| Total Math-IR nodes extracted | 11 |
| Generated .c files | 141 |
| Clean compilable files | 95 (67%) |
| Files with #error | 46 (33%) |
| Z3 verified nodes | 80/141 |
| Fuzz operations passed | 25/25 |
| Pipeline time (all files) | 384.3s |
