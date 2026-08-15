# Architecture

## Pipeline Overview

```
Python Source → Parser → Math-IR → Slicer → C99 Code → Verifier
    ↓              ↓         ↓         ↓          ↓          ↓
  ast.parse    extract   DAG build   prune    code gen   fuzz+z3
  92 bodies    kernels   topological dead     headers    full sweep
  diagnostics           sort        code     headers    91 reachable
```

## Data Flow

1. **Parser** (`parser/python_parser.py`): Parses Python source, detects NumPy operations, produces `OperationInfo` list.

2. **Builder** (`ir/builder.py`): Converts `OperationInfo` into a Math-IR DAG (`MathIRGraph`). Each node represents a kernel with inputs/outputs, effects, and algorithm metadata.

3. **Slicer** (`slicer/semantic_slicer.py`): Resolves call-graph dependencies, classifies nodes (math/alloc/transform/external), prunes dead code.

4. **C99 Generator** (`backend/c99_generator.py`): Generates self-contained C99 code with file headers, function headers, zero-heap-allocation math kernels.

5. **Verifier** (`verifier/`):
   - `fuzzer.py`: Differential testing — compares C output against Python reference (25 legacy ops, configurable iterations).
   - `coverage_sweep.py`: Compiles every reachable kernel body (91/92 `MATH_KERNEL_BODIES`; `array_diff` is documented-unreachable) and verifies against NumPy over dense grids.
   - `z3_verifier.py`: Symbolic verification using SMT constraints.
   - `equivalence.py`: End-to-end kernel equivalence engine used by `test_generated_kernel_runtime.py` and `test_reproducible.py`.

## Key Design Decisions

- **DAG representation**: Enables topological sort for deterministic code generation and dependency tracking.
- **Zero heap allocation**: Math kernels use stack arrays only — suitable for embedded/real-time targets.
- **Provenance tracking**: Every generated C file and function carries origin metadata (source file, line, commit, signature).
- **Algorithm mapping**: `MATH_KERNEL_BODIES` in c99_generator.py maps 92 kernel bodies (91 reachable) to C implementations.
- **No Jinja2**: All C code generated programmatically (no template engine).

## Operation Categories

| Category | Operations |
|----------|-----------|
| Element-wise arithmetic | add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tan, tanh, power, sign, floor, ceil, trunc, clip, round, isclose, isnan, isinf, greater, less, log10, logaddexp, conj, angle, real, imag, copy, where, maximum, minimum |
| Reduction | sum, mean, max, min, var, prod, cumsum, diff, argmax, argmin, any, all |
| Matrix | matmul, transpose, outer, diag, tril, triu, sort |
| Linear algebra | solve, inverse, cholesky, eig, det, norm, qr, svd |
| Signal processing | fft, ifft |
| Allocation | zeros, ones, eye, arange, linspace, full, full_like, ones_like, zeros_like, random |
| Array ops | concatenate, take, argsort, permutation, reshape, squeeze, expand_dims, flatten, tile, repeat, flip, roll, split, unique, searchsorted, stack, vstack, hstack |
