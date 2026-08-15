# Changelog

All notable changes to Purce will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

#### v0.2.0 — Extended Operations
- `np.dot` for >2D arrays
- `np.einsum` (generalized tensor contraction)
- `np.sort` with multiple algorithms (quicksort, mergesort, heapsort)
- `np.linalg.det` (determinant)

#### v0.3.0 — Code Generation Improvements
- SIMD-optimized kernels (SSE, AVX, NEON)
- OpenMP parallelization support
- Amalgamated single-file output (like SQLite)
- Custom target profiles (user-defined type mappings)

#### v0.4.0 — Tooling
- Plugin system for custom algorithms
- WebAssembly target
- Docker container for reproducible builds
- IDE plugin (VS Code)

---

## [0.1.0] - 2026-08-16

### Added

#### Core Pipeline
- **Math-IR**: Custom intermediate representation (DAG-based, language-agnostic)
  - `MathIRNode` with full provenance tracking
  - `MathIRGraph` with iterative topological sort and cycle detection
  - Dependency classification (MATH_KERNEL, SYSTEM_PAL, DATA_ASSET, META_UTIL)
- **Python Parser**: Extracts math kernels from Python/NumPy source
  - Supports 92 NumPy kernel bodies (91 reachable)
  - Type hint extraction (float32, float64, int32, int64)
  - Structured diagnostics for unsupported constructs
- **Semantic Slicer**: Call graph analysis and dead code elimination
  - Reachability analysis from entry points
  - Transitive dependency resolution (3+ levels deep)
  - DAG validation (cycle detection, missing dependency detection)
- **C99 Backend**: Programmatic code generation (no template engine)
  - C99-SOS compliant output (file headers, function headers, naming)
  - Zero heap allocation in math kernels
  - Provenance metadata in every generated file
  - Auto-generated CMakeLists.txt

#### Supported Operations (91 reachable kernels, 92 bodies)
- **Element-wise**: add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tan, tanh, power, sign, floor, ceil, trunc, clip, where, greater, less, log10, logaddexp, conj, angle, real, imag, copy, round, isclose, isnan, isinf
- **Reductions**: sum, mean, max, min, var, prod, cumsum, diff, argmax, argmin, any, all
- **Matrix**: matmul, transpose, outer, diag, tril, triu, sort
- **Linear algebra**: solve, inverse, cholesky, eig, norm, det, qr, svd
- **Signal processing**: fft, ifft
- **Allocation**: zeros, ones, eye, arange, linspace, full, full_like, ones_like, zeros_like, random
- **Array ops**: concatenate, take, argsort, permutation, reshape, squeeze, expand_dims, flatten, tile, repeat, flip, roll, split, unique, searchsorted, stack, vstack, hstack

#### Verification
- **Z3 SMT Verifier**: Symbolic verification for all operations
  - Dimension bounds, division-by-zero, overflow, non-singular, FFT power-of-two
- **Differential Fuzzer**: 25 legacy operations, 1000 iterations each
  - Random input generation
  - Python reference comparison with configurable tolerance
- **Full-Coverage Sweep**: every reachable kernel body (91/92; `array_diff` documented-unreachable) verified against NumPy over dense grids at multiple scales
- **5-Phase Verification Agent**:
  1. Unit tests (487 tests)
  2. Fuzz tests (25 legacy ops + 91-reachable coverage sweep)
  3. Pipeline integration (real-world ML sources: 23 files → 1413 C kernels, 0 `#error`)
  4. Synthetic pipeline sanity checks
  5. Memory safety verification (heap-free, provenance)

#### CLI
- `purce extract`: Library extraction mode
- `purce compile`: User code compilation mode
- `purce verify`: Full verification suite
- Target profiles: `generic-c99`, `bare-arm-q31`, `bare-arm-q15`

#### Testing
- 487 tests across 17 test modules (gcc-dependent tests skip when gcc unavailable)
- Real-world test projects: 23 ML/scientific source files (1413 C kernels generated)
- Semantic compiler features: multi-statement decomposition, recursive expression decomposition, .shape/.transpose resolution, 91 reachable kernels
- C compilation verification tests (requires gcc)
- Full pipeline integration tests

#### Benchmarks
- Numerical accuracy: 87,000 iterations across 25 operations
- Performance scaling: real gcc -O2 compilation (5 operations, 6 input sizes)
- Edge cases: 41 IEEE 754 test cases
- Code quality: cyclomatic complexity, LOC, structural analysis
- Reproducible corpus gate: 14 harder kernels → 812 C files, strict-gcc compile gate

#### Documentation
- Architecture guide with pipeline diagram
- API reference
- User guide (quick start, memory model, targets)
- Testing guide
- Contributing guide

#### CI/CD
- GitHub Actions workflow (Python 3.11/3.12/3.13)
- Test matrix with fuzz verification
- Type checking with mypy

#### Developer Tooling
- Pre-commit hook (tests + fuzz + TODO check)
- Setup script for new contributors
- PEP 561 type marker (py.typed)

### Fixed
- **Corpus-gate non-determinism**: caller→callee edges were stored in a Python `set`
  and iterated unsorted (`builder.py` `_wire_caller_roots`), so emitted `nested_deps`
  order depended on the per-process `PYTHONHASHSEED`. Callees are now iterated in
  sorted order, making generated provenance byte-reproducible across processes and
  machines.
- **Semantic Slicer dropped `scalar_constants`** when copying nodes, causing scalar
  constants to emit as invalid `double _const_N[i]` array indexing — now preserved
  through both `slice()` and `resolve_transitive_deps()`
- Indentation bug in `_resolve_arg_for_full_body` constant handler that could
  return `None` for repeated constants
- **Benchmark summary aggregation**: `real_benchmark.py` reported only the last
  file's node count and a modulo-derived pipeline time; both are now aggregated
  across all files
