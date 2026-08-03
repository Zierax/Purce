# Changelog

All notable changes to Purce will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-28

### Added

#### Core Pipeline
- **Math-IR**: Custom intermediate representation (DAG-based, language-agnostic)
  - `MathIRNode` with full provenance tracking
  - `MathIRGraph` with iterative topological sort and cycle detection
  - Dependency classification (MATH_KERNEL, SYSTEM_PAL, DATA_ASSET, META_UTIL)
- **Python Parser**: Extracts math kernels from Python/NumPy source
  - Supports 25+ NumPy operations
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

#### Supported Operations (25)
- **Element-wise**: add, sub, mul, div, neg, abs, sqrt, exp, log, sin, cos, tan, power
- **Reductions**: sum, mean, max, min
- **Matrix**: matmul, transpose
- **Linear algebra**: solve, inverse, determinant, cholesky, eig
- **Signal processing**: fft, ifft
- **Sorting**: sort
- **Allocation**: zeros, ones, eye

#### Verification
- **Z3 SMT Verifier**: Symbolic verification for all operations
  - Dimension bounds, division-by-zero, overflow, non-singular, FFT power-of-two
- **Differential Fuzzer**: 25 operations, 200 iterations each
  - Random input generation via Hypothesis
  - Python reference comparison with configurable tolerance
- **5-Phase Verification Agent**:
  1. Unit tests (174+)
  2. Fuzz tests (25 ops × 200 iterations = 5,000 test cases)
  3. Pipeline integration (real-world ML sources)
  4. Synthetic pipeline sanity checks
  5. Memory safety verification (heap-free, provenance)

#### CLI
- `purce extract`: Library extraction mode
- `purce compile`: User code compilation mode
- `purce verify`: Full verification suite
- Target profiles: `generic-c99`, `bare-arm-q31`, `bare-arm-q15`

#### Testing
- 174 tests across 8 test modules
- 100% pass rate
- Real-world test projects: 14 ML modules (210+ functions)
- Extra patterns: transformer, RNN, conv1d, batch norm, focal loss, AdamW

#### Benchmarks
- Numerical accuracy: 87,000 iterations across 25 operations
- Performance scaling: real gcc -O2 compilation (6x-303x speedup)
- Edge cases: 50 IEEE 754 test cases
- Code quality: cyclomatic complexity, LOC, structural analysis

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
- N/A (initial release)

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Security
- N/A (initial release)

---

## [Unreleased]

### Planned

#### v0.2.0 — Extended Operations
- `element_pow` (np.power) — most common unsupported op
- `np.dot` for >2D arrays
- `np.einsum` (generalized tensor contraction)
- `np.clip` / `np.where` (conditional element-wise)
- `np.sort` with multiple algorithms (quicksort, mergesort, heapsort)

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
