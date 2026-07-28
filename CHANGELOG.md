# Changelog

All notable changes to Purce will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-26

### Added

#### Core Pipeline
- **Math-IR**: Custom intermediate representation (DAG-based, language-agnostic)
  - `MathIRNode` with full provenance tracking
  - `MathIRGraph` with topological sort and cycle detection
  - Dependency classification (MATH_KERNEL, SYSTEM_PAL, DATA_ASSET, META_UTIL)
- **Python Parser**: Extracts math kernels from Python/NumPy source
  - Supports 20+ NumPy operations (MVP subset)
  - Type hint extraction (float32, float64, int32, int64)
  - Structured diagnostics for unsupported constructs
- **Semantic Slicer**: Call graph analysis and dead code elimination
  - Reachability analysis from entry points
  - Transitive dependency resolution (3+ levels deep)
  - DAG validation (cycle detection, missing dependency detection)
- **C99 Backend**: Jinja2-based code generation
  - C99-SOS compliant output (file headers, function headers, naming)
  - Zero heap allocation in math kernels
  - Q31/Q15 fixed-point support
  - Provenance JSON for every .c file
  - Auto-generated CMakeLists.txt

#### Supported Operations
- **Linear Algebra**: `matmul`, `linalg_solve`, `linalg_inv`, `linalg_cholesky`, `linalg_eig`
- **Element-wise**: `element_add`, `element_sub`, `element_mul`, `element_div`
- **Reductions**: `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min`
- **FFT**: `fft` (Cooley-Tukey), `ifft`
- **Allocation**: `alloc_zeros`, `alloc_ones`, `alloc_eye`

#### Verification
- **Z3 SMT Verifier**: Bounds checking for all operations
  - Dimension bounds verification
  - Division by zero detection
  - Overflow checks
  - Non-singular matrix checks
  - FFT power-of-two constraints
- **Differential Fuzzer**: Property-based testing
  - Random input generation via Hypothesis
  - Python reference comparison
  - Configurable tolerance (rtol, atol)
  - 10,000 iterations per operation

#### CLI
- `purce extract`: Library extraction mode
- `purce compile`: User code compilation mode
- `purce verify`: Full verification suite
- Target profiles: `generic-c99`, `bare-arm-q31`, `bare-arm-q15`
- Flags: `--embed-assets`, `--amalgamate`, `--verbose`, `--provenance-only`

#### Testing
- 119 tests across 6 test modules
- 100% pass rate
- Test fixtures for all MVP operations

#### Documentation
- Comprehensive README with architecture diagram
- API reference documentation
- Contributing guide
- Benchmark results

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

#### Phase 2 Features
- C++ parser via libclang
- libcst integration for Python mutation
- Amalgamated single-file output (like SQLite)
- SIMD-optimized kernels (SSE, AVX, NEON)
- OpenMP parallelization support
- Additional NumPy operations (dot for >2D, einsum, etc.)

#### Phase 3 Features
- Custom target profiles (user-defined type mappings)
- Plugin system for custom algorithms
- WebAssembly target
- Rust backend
- Interactive mode (REPL)

#### Phase 4 Features
- Integration with CI/CD pipelines
- Docker container for reproducible builds
- Package registry for generated C libraries
- IDE plugin (VS Code)
- Online playground
