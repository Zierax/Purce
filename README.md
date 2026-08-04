# Purce — Pure-C Semantic Compiler

> **"Code is not the source. Mathematics is the source. Code is merely a transient projection of mathematical intent."**

Purce is a semantic compiler that takes Python/NumPy code and produces clean, self-contained, production-grade C99 codebases. It does not transpile — it **extracts mathematical intent** from source code and **re-expresses it** as verifiable C99.

---

## Table of Contents

- [Why Purce](#why-purce)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [C99-SOS: Semantic Output Standard](#c99-sos-semantic-output-standard)
- [Supported Operations](#supported-operations)
- [Verification Pipeline](#verification-pipeline)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Why Purce

| Problem | Solution |
|---------|----------|
| Python/NumPy is too slow for embedded/real-time | Generates pure C99 — no Python runtime needed |
| NumPy has 500+ functions, you only use 10 | Purce extracts only what you actually call (semantic slicing) |
| Generated C code is unreadable/unmaintainable | C99-SOS standard enforces provenance, memory contracts, and naming |
| Limited verification of generated code | Z3 SMT bounds checking + differential fuzzing (Python reference) |
| Platform-specific code is hard to port | PAL stubs for bare-metal targets (ARM Q31/Q15) |

---

## Architecture

```
                        PURCE PIPELINE
                        ==============

 ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
 │  Python /   │     │  Multi-Lang │     │   Math-IR   │
 │  NumPy      │────▶│  Parser     │────▶│   (DAG)     │
 │  Source      │     │  (ast)      │     │             │
 │             │     │             │     │  Language-   │
 └─────────────┘     └─────────────┘     │  Agnostic    │
                                         └──────┬──────┘
                                                │
                                                ▼
 ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
 │   Output    │     │  Programmatic│     │  Semantic   │
 │  .c / .h    │◀────│  C99        │◀────│  Slicer     │
 │  .prov.json │     │  Generator  │     │  (Dead Code │
 │  CMakeLists │     │             │     │   Elim.)    │
 └─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                      │  Z3 SMT      │ │ Differential │ │  ctypes      │
                      │  Bounds      │ │ Fuzzing      │ │  Bridge      │
                      │  Checker     │ │ (C99 vs Py)  │ │  (gcc→DLL)   │
                      └──────────────┘ └──────────────┘ └──────────────┘
```

### Pipeline Stages

1. **Parser** — Uses Python's `ast` module to extract function definitions and NumPy call graphs
2. **Math-IR Builder** — Converts AST nodes into a language-agnostic intermediate representation (DAG of semantic units)
3. **Semantic Slicer** — Resolves call graphs, eliminates dead code, classifies dependencies (math kernel vs PAL vs data asset)
4. **C99 Backend** — Generates C99 code programmatically via Python string templates in `c99_generator.py`, following the C99-SOS standard
5. **Verification** — Compiles generated C99 to a shared library via gcc, loads via ctypes, and performs differential fuzzing comparing compiled C output against Python reference implementations. Z3 SMT for bounds checking.

---

## Installation

### Requirements

- Python 3.11+
- GCC or Clang (for C backend verification and benchmarking)
- NumPy (for verification benchmarks)

### From Source

```bash
git clone https://github.com/Zierax/Purce.git
cd Purce
pip install -e ".[dev]"
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `z3-solver` | ≥4.12 | SMT solver for bounds verification |
| `hypothesis` | ≥6.80 | Property-based testing / differential fuzzing |
| `click` | ≥8.1 | CLI framework |
| `pytest` | ≥7.4 | Test runner |

---

## Quick Start

### Extract math kernels from a library

```bash
purce extract ./numpy/linalg/ --target generic-c99 -o ./out/
```

### Compile your Python project to C99

```bash
purce compile ./my_project/ --target generic-c99 -o ./out/

# With verification (requires gcc)
purce compile ./my_project/ --target generic-c99 -o ./out/ --verify
```

### Run the verification suite

```bash
purce verify --iterations 1000
```

---

## CLI Reference

### `purce extract`

Library extraction mode: extract math kernels from a directory of Python files.

```bash
purce extract <SOURCE_DIR> [OPTIONS]

Options:
  --target {generic-c99,bare-arm-q31,bare-arm-q15}
                          Target profile (default: generic-c99)
  -o, --output PATH       Output directory (default: out/)
  --verbose               Show extraction diagnostics
  --provenance-only       Generate manifest without code
```

### `purce compile`

User code compilation mode: compile a Python project to C99 with full dependency resolution.

```bash
purce compile <SOURCE_DIR> [OPTIONS]

Options:
  --target {generic-c99,bare-arm-q31,bare-arm-q15}
                          Target profile (default: generic-c99)
  -o, --output PATH       Output directory (default: out/)
  --verify                Run Z3 + differential fuzzing after compilation
```

### `purce verify`

Run the full verification suite: Z3 SMT bounds checking + differential fuzzing.

```bash
purce verify [OPTIONS]

Options:
  --iterations N          Fuzzing iterations per operation (default: 1000)
```

---

## C99-SOS: Semantic Output Standard

Every generated file follows the **C99 Semantic Output Standard** — a strict commenting and naming convention that ensures provenance, traceability, and correctness.

### File Header

```c
/* ═══════════════════════════════════════════════════════════════════════════
 * PURCE OUTPUT
 * GENERATED FILE:   numpy_linalg_matmul.c
 * SOURCE MODULE:    numpy.linalg
 * GENERATED BY:     purce v0.1.0
 * TARGET PROFILE:   generic-c99
 * ═══════════════════════════════════════════════════════════════════════════ */
```

### Function Header

```c
/* ───────────────────────────────────────────────────────────────────────────
 * SEMANTIC UNIT:    linalg.matmul_a1b2c3d4
 * ORIGIN SYMBOL:    numpy.linalg.matmul
 * MEMORY CONTRACT:
 *   - Stack:   512 bytes
 *   - Heap:    NONE
 * ─────────────────────────────────────────────────────────────────────────── */
```

---

## Supported Operations

### Linear Algebra

| Python | C99 Algorithm |
|--------|---------------|
| `numpy.dot(A, B)` | `matmul` |
| `numpy.matmul(A, B)` | `matmul` |
| `numpy.linalg.solve(A, b)` | `linalg_solve` |
| `numpy.linalg.inv(A)` | `linalg_inv` |
| `numpy.linalg.cholesky(A)` | `linalg_cholesky` |
| `numpy.linalg.eig(A)` | `linalg_eig` |

### Element-wise Operations

| Python | C99 Algorithm |
|--------|---------------|
| `numpy.add(A, B)` | `element_add` |
| `numpy.subtract(A, B)` | `element_sub` |
| `numpy.multiply(A, B)` | `element_mul` |
| `numpy.divide(A, B)` | `element_div` |
| `numpy.sqrt(x)` | `element_sqrt` |
| `numpy.abs(x)` | `element_abs` |
| `numpy.exp(x)` | `element_exp` |
| `numpy.log(x)` | `element_log` |
| `numpy.sin(x)` | `element_sin` |
| `numpy.cos(x)` | `element_cos` |
| `numpy.tan(x)` | `element_tan` |

### Reductions

| Python | C99 Algorithm |
|--------|---------------|
| `numpy.sum(x)` | `reduce_sum` |
| `numpy.mean(x)` | `reduce_mean` |
| `numpy.max(x)` | `reduce_max` |
| `numpy.min(x)` | `reduce_min` |

### FFT

| Python | C99 Algorithm |
|--------|---------------|
| `numpy.fft.fft(x)` | `fft` (Cooley-Tukey) |
| `numpy.fft.ifft(x)` | `ifft` |

### Allocation

| Python | C99 Algorithm | Notes |
|--------|---------------|-------|
| `numpy.zeros(n)` | `alloc_zeros` | Zero-initialized |
| `numpy.ones(n)` | `alloc_ones` | One-initialized |
| `numpy.eye(n)` | `alloc_eye` | Identity matrix |

---

## Verification Pipeline

### Differential Fuzzing

For each supported operation, Purce compiles the generated C99 code into a shared library via gcc, loads it via Python's ctypes module, and compares:

1. **Reference**: Python/NumPy implementation
2. **Target**: Compiled C99 code (called via ctypes)
3. **Tolerance**: Configurable per-operation

The `ctypes_bridge.py` module handles C99 compilation, shared library loading, and typed call interfaces for all 25 operations.

### Z3 SMT Bounds Checking

For each Math-IR node, Purce generates verification conditions:

- **Dimension bounds**: Matrix dimensions are within safe ranges
- **Division by zero**: No division by zero in element_div
- **Overflow**: Output within representable range for target dtype
- **Non-singular**: Diagonal elements non-zero for linalg operations
- **Power of two**: FFT input size is power of 2

---

## Project Structure

```
purce/
├── purce/
│   ├── __init__.py              # Package version
│   ├── cli.py                   # CLI entry point (Click)
│   ├── parser/
│   │   └── python_parser.py     # Python ast + NumPy detection
│   ├── ir/
│   │   ├── nodes.py             # MathIRNode, MathIRGraph, Effect, Dtype
│   │   └── builder.py           # Build IR from AST
│   ├── slicer/
│   │   └── semantic_slicer.py   # Dead code elimination, call graph
│   ├── backend/
│   │   └── c99_generator.py     # C99 code generation
│   └── verifier/
│       ├── z3_verifier.py       # SMT bounds checking
│       ├── fuzzer.py            # Differential fuzzing
│       └── ctypes_bridge.py     # C99 compilation + ctypes loading
├── tests/                       # 188 tests (165 passing, 42 skipped)
│   ├── test_ir.py               # Math-IR node and graph tests
│   ├── test_parser.py           # Python parser tests
│   ├── test_slicer.py           # Semantic slicer tests
│   ├── test_backend.py          # C99 generator tests
│   ├── test_verifier.py         # Fuzzer, Z3, ctypes tests
│   ├── test_integration.py      # Full pipeline tests
│   ├── test_realworld.py        # Real-world ML code tests
│   ├── test_c_compilation.py    # C compilation verification (requires gcc)
│   ├── test_cli.py              # CLI integration tests
│   ├── verification_agent.py    # 5-phase verification agent
│   ├── fixtures/                # Sample Python files
│   └── realworld/               # 15 ML/scientific test sources
├── benchmarks/                  # 7 benchmark scripts
├── docs/                        # Documentation
├── pyproject.toml
└── README.md
```

---

## Development

### Running Tests

```bash
pytest tests/ -v                 # Run all tests
pytest tests/test_ir.py -v      # Run specific test module
pytest -x                       # Stop on first failure
```

### Benchmarks

```bash
python -m benchmarks.run_all    # Run full benchmark suite
```

---

## License

MIT
