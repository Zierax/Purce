# Purce Benchmark Results

Generated: 2026-08-16 00:51:41
Python: 3.13.14

## Pipeline Performance

Time to process Python source through full pipeline: parse → IR → slice → generate C99.

| Source | Nodes | C Files | Clean C | #error C | C Lines | Time (ms) |
|--------|-------|---------|---------|----------|---------|-----------|
| fixtures/ (3 files) | 5 | 5 | 4 | 1 | 112 | 6.5 |
| realworld/ (22 files) | 1450 | 1450 | 1450 | 0 | 16620 | 1303.0 |

### Per-File Breakdown

| File | Nodes | C Files | Clean | #error | C Lines | Time (ms) |
|------|-------|---------|-------|--------|---------|-----------|
| activations.py | 234 | 234 | 234 | 0 | 2573 | 134.3 |
| attention.py | 44 | 44 | 44 | 0 | 558 | 46.4 |
| computer_vision.py | 57 | 57 | 57 | 0 | 671 | 57.1 |
| convolution.py | 39 | 39 | 39 | 0 | 401 | 31.4 |
| data_pipeline.py | 26 | 26 | 26 | 0 | 282 | 24.0 |
| extra_patterns.py | 0 | 0 | 0 | 0 | 0 | 0.4 |
| generative.py | 71 | 71 | 71 | 0 | 809 | 34.2 |
| graph_neural_networks.py | 35 | 35 | 35 | 0 | 478 | 19.8 |
| jax_ops.py | 85 | 85 | 85 | 0 | 964 | 58.8 |
| layers.py | 48 | 48 | 48 | 0 | 588 | 36.2 |
| linear_algebra.py | 36 | 36 | 36 | 0 | 478 | 34.3 |
| losses.py | 80 | 80 | 80 | 0 | 894 | 65.1 |
| model.py | 30 | 30 | 30 | 0 | 341 | 41.8 |
| normalization.py | 24 | 24 | 24 | 0 | 297 | 17.8 |
| optimizers.py | 86 | 86 | 86 | 0 | 911 | 69.7 |
| pytorch_ops.py | 100 | 100 | 100 | 0 | 1125 | 86.9 |
| recommendation.py | 60 | 60 | 60 | 0 | 725 | 64.7 |
| reinforcement_learning.py | 48 | 48 | 48 | 0 | 526 | 36.0 |
| scipy_ops.py | 24 | 24 | 24 | 0 | 239 | 32.8 |
| signal_processing.py | 87 | 87 | 87 | 0 | 901 | 68.3 |
| time_series.py | 38 | 38 | 38 | 0 | 513 | 23.1 |
| transformers.py | 71 | 71 | 71 | 0 | 880 | 50.2 |

## Verification Results

### Z3 SMT Bounds Checking

- Total nodes: 1450
- Verified (safe): 788
- Violated (potential overflow): 0
- Unknown (Z3 timeout): 662

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

- Total .c files: 1450
- Files with #error (unsupported ops): 0
- Clean compilable files: 1450
- Total C lines (excluding comments): 16620
- Total function definitions: 1450
- Average lines per function: 11

## Operation Coverage

### Supported (generates valid C99)

| Operation | Algorithm | Kernel Count |
|-----------|-----------|--------------|
| various | alloc_arange | 12 |
| various | alloc_eye | 12 |
| various | alloc_full | 4 |
| various | alloc_linspace | 2 |
| various | alloc_ones | 15 |
| various | alloc_random | 2 |
| various | alloc_zeros | 65 |
| various | array_argsort | 1 |
| various | array_concat | 2 |
| various | array_literal | 1 |
| various | array_permutation | 1 |
| various | array_reshape | 25 |
| various | array_take | 3 |
| various | element_abs | 11 |
| various | element_add | 184 |
| various | element_clip | 3 |
| various | element_conj | 4 |
| various | element_copy | 12 |
| various | element_cos | 8 |
| various | element_div | 123 |
| various | element_exp | 79 |
| various | element_greater | 27 |
| various | element_less | 1 |
| various | element_log | 26 |
| various | element_max | 25 |
| various | element_min | 10 |
| various | element_mul | 308 |
| various | element_neg | 13 |
| various | element_power | 55 |
| various | element_real | 1 |
| various | element_sign | 1 |
| various | element_sin | 4 |
| various | element_sqrt | 38 |
| various | element_sub | 121 |
| various | element_tanh | 6 |
| various | element_where | 37 |
| various | linalg_inv | 1 |
| various | loop_concat | 2 |
| various | matmul | 70 |
| various | matrix_diag | 2 |
| various | matrix_tril | 1 |
| various | outer_product | 1 |
| various | reduce_max | 16 |
| various | reduce_mean | 41 |
| various | reduce_sum | 48 |
| various | reduce_var | 12 |
| various | transpose | 14 |

### Unsupported (generates #error)

- Total nodes with 'unknown' algorithm: 0

## Summary

| Metric | Value |
|--------|-------|
| Python source files processed | 25 |
| Total Math-IR nodes extracted | 1323 |
| Generated .c files | 1450 |
| Clean compilable files | 1450 (100%) |
| Files with #error | 0 (0%) |
| Z3 verified nodes | 788/1450 |
| Fuzz operations passed | 25/25 |
| Pipeline time (all files) | 1.0s |
