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
  1. Unit tests (486 tests)
  2. Fuzz tests (25 legacy ops + 91-reachable coverage sweep)
  3. Pipeline integration (real-world ML sources: 23 files → 499 C kernels, 100% clean)
  4. Synthetic pipeline sanity checks
  5. Memory safety verification (heap-free, provenance)

#### CLI
- `purce extract`: Library extraction mode
- `purce compile`: User code compilation mode
- `purce verify`: Full verification suite
- Target profiles: `generic-c99`, `bare-arm-q31`, `bare-arm-q15`

#### Testing
- 486 tests across 14 test modules (gcc-dependent tests skip when gcc unavailable)
- Real-world test projects: 23 ML/scientific source files (499 C kernels generated)
- Semantic compiler features: multi-statement decomposition, recursive expression decomposition, .shape/.transpose resolution, 91 reachable kernels
- C compilation verification tests (requires gcc)
- Full pipeline integration tests

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

### Added
- **Full-coverage verification sweep** (`coverage_sweep.py`): compiles every reachable
  kernel body in `MATH_KERNEL_BODIES` and verifies each against its NumPy reference over
  dense grid sizes (8×8 → 2000×80) under multiple seeds — 92 total bodies, 91 reachable
  (`array_diff` is a documented-unreachable body), all matching.
- **Stress-scale sweeps**: dense grid sizes from 8×8 up to 2000×80 (plus linalg/FFT
  single-dimension inputs) across multiple seeds — 91/91 kernels reachable in every
  step, with the full mega-sweep covering ~14.5M kernel cases.
- **Edge/corner case coverage** (`test_edge_coverage.py`): IEEE-754 corner inputs and
  boundary shapes for every element-wise kernel (0% crash rate).
- **Reproducible corpus gate hardening**: extraction output is now byte-identical
  regardless of `PYTHONHASHSEED` (see Fixed), making the committed
  `benchmarks/baseline.json` a true drift detector.
- **499 C kernels** generated from 23 realworld test files
- **0 unresolved inputs** (was 169/18.4%)
- **92 NumPy kernel bodies** with full C99 implementations (91 reachable)
- Multi-statement body decomposition with symbol table tracking
- Recursive expression decomposition for nested BinOps/Calls
- If/else control flow as conditional IR nodes (element_where)
- For-loop unrolling (loop_concat pattern detection)
- `.T` transpose attribute access resolved in all arg resolvers
- `.reshape()` / `.flatten()` / `.squeeze()` method call resolution
- `np.pi` / `np.e` / `np.inf` as scalar constants
- UnaryOp negation creates element_mul IR nodes for non-constants
- Local function intermediate resolution in composed function decomposition
- ast.Tuple handling alongside ast.List in all arg resolution paths
- Type cast mapping (np.float64 → element_copy)
- .shape tuple unpacking and BinOp assignment decomposition
- Complex constant handling (`-2j`, `1j`) in all arg resolvers
- NumPy dtype attribute constants (`np.complex128`, `np.float64`) resolved as scalars
- Python builtin type names (`float`, `int`, `complex`) resolved as scalars
- Subscript/slicing base resolution (`x[0::2]`, `V[:, :1, :]`, `feature_map[x0, y0]`)
- Unknown/local function calls resolve to first-argument identity
- Local function call assignments tracked in symbol table
- `.T` transpose on Call results (e.g. `mel_filterbank(...).T`)

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
