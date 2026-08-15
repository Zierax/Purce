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

## Key Design Decisions

- **DAG representation**: Enables topological sort for deterministic code generation and dependency tracking.
- **Zero heap allocation**: Math kernels use stack arrays only — suitable for embedded/real-time targets.
- **Provenance tracking**: Every generated C file and function carries origin metadata (source file, line, commit, signature).
- **Algorithm mapping**: `MATH_KERNEL_BODIES` in c99_generator.py maps 25 algorithms to C implementations.
- **No Jinja2**: All C code generated programmatically (previously used templates, now removed).

## Operation Categories

| Category | Operations |
|----------|-----------|
| Element-wise arithmetic | add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tan, power |
| Reduction | sum, mean, max, min |
| Matrix | matmul, transpose |
| Linear algebra | solve, inverse, determinant, cholesky, eig |
| Signal processing | fft, ifft |
| Sorting | sort |
