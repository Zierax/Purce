# Purce User Guide

## What is Purce?

Purce (Pure-C Semantic Compiler) converts Python/NumPy code into clean, self-contained C99 codebases. It extracts mathematical kernels from Python, builds an intermediate representation (Math-IR), and generates C99 code with full provenance tracing.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Compile a Python project to C99
purce compile ./my_project/ -o ./out/ --verify

# Verify the output
purce verify --iterations 10000
```

## Pipeline Overview

```
Python/NumPy → Parser → Math-IR → Slicer → C99 Generator → Verified Output
```

### Stage 1: Parsing

The parser uses Python's `ast` module to extract function definitions and detect NumPy operations.

**Supported operations:**
- Element-wise: `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.abs`, `np.sqrt`, `np.exp`, `np.log`, `np.sin`, `np.cos`, `np.tan`
- Reductions: `np.sum`, `np.mean`, `np.max`, `np.min`
- Linear algebra: `np.matmul`, `np.linalg.solve`, `np.linalg.inv`, `np.linalg.cholesky`, `np.linalg.eig`
- Signal processing: `np.fft.fft`, `np.fft.ifft`
- Allocation: `np.zeros`, `np.ones`, `np.eye`

### Stage 2: Math-IR Construction

Each function becomes a `MathIRNode` in a dependency graph (DAG). Nodes record:
- Origin (file, line, commit, signature)
- Mathematical intent and algorithm
- Input/output types and shapes
- Side effects and dependency classification

### Stage 3: Semantic Slicing

The slicer classifies dependencies:
- **MATH_KERNEL**: Pure math → included in output
- **SYSTEM_PAL**: OS calls → stubs generated
- **DATA_ASSET**: File I/O → error (or embed with `--embed-assets`)
- **META_UTIL**: Small helpers → inlined/eliminated

Dead code is eliminated. DAG cycles are detected and rejected.

### Stage 4: C99 Code Generation

For each node in topological order:
1. Generate C99-SOS compliant function header
2. Substitute kernel body from template library
3. Map canonical parameters to IR parameters
4. Write `.c` file, `.prov.json` provenance file
5. Generate `.h` header with forward declarations
6. Generate `CMakeLists.txt`

### Stage 5: Verification

**Z3 SMT Verification:**
- Dimension bounds checking
- Division-by-zero detection
- Overflow analysis
- Non-singular matrix verification
- FFT power-of-two constraints

**Differential Fuzzing:**
- 25 operations × N iterations
- Python reference implementation as oracle
- Random input generation
- Output comparison within tolerance (rtol=1e-5, atol=1e-8)

## Generated Code Structure

```
out/
├── module_function1.c          # One .c per Math-IR node
├── module_function1.prov.json  # Full provenance
├── module_function2.c
├── module_function2.prov.json
├── module.h                    # Header with forward declarations
└── CMakeLists.txt              # Build system
```

### C99-SOS Compliance

Every generated file and function has mandatory comment blocks:

```c
/* ═══════════════════════════════════════════════════════════════
 * PURCE OUTPUT: C99-SOS Compliant Generated Code
 * Source: mymodule.py:42
 * Algorithm: matmul
 * Commit: abc123
 * Generated: 2026-07-28T12:00:00Z
 * ═══════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════
 * PURCE FUNCTION: numpy_matmul_a1b2c3d4
 * Origin: mymodule.py:42 → matmul
 * Signature: (double*, double*, double*, int, int, int) -> void
 * ═══════════════════════════════════════════════════════════════ */
void numpy_matmul_a1b2c3d4(
    const double *restrict A,
    const double *restrict B,
    double *restrict C,
    int m, int n, int k
) {
    /* Matrix multiplication: C[m][n] = A[m][k] * B[k][n] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
```

## Memory Model

**Zero heap allocation** in all math kernels. All memory is caller-provided:

```c
// Caller allocates
double *A = malloc(m * k * sizeof(double));
double *B = malloc(k * n * sizeof(double));
double *C = malloc(m * n * sizeof(double));

// Generated code uses arrays in-place
numpy_matmul_a1b2c3d4(A, B, C, m, n, k);

// Caller frees
free(A); free(B); free(C);
```

This makes generated code suitable for:
- Bare-metal / embedded systems
- Real-time systems (deterministic memory)
- Safety-critical applications
- RTOS environments

## Target Profiles

| Profile | Description | Use Case |
|---------|-------------|----------|
| `generic-c99` | Standard C99 with `double` | Desktop, server |
| `bare-arm-q31` | ARM fixed-point Q31 | Microcontrollers |
| `bare-arm-q15` | ARM fixed-point Q15 | DSP applications |

## Verification

### Z3 SMT Verification

```python
from purce.verifier.z3_verifier import Z3Verifier

verifier = Z3Verifier()
reports = verifier.verify_graph(graph)

for nid, report in reports.items():
    print(f"{nid}: {report.verified_count}/{len(report.conditions)} verified")
```

### Differential Fuzzing

```python
from purce.verifier.fuzzer import DifferentialFuzzer

fuzzer = DifferentialFuzzer(rtol=1e-5, atol=1e-8)
results = fuzzer.fuzz_all(iterations=10000)

for op, result in results.items():
    print(f"{op}: {result.success_rate:.1%}")
```

### Verification Agent

```bash
python -m tests.verification_agent
```

Runs 4-phase validation:
1. Unit tests (pytest)
2. Fuzz tests (25 operations)
3. Pipeline integration (real-world sources)
4. Synthetic pipeline test

## CLI Reference

### `purce compile`

```bash
purce compile <SOURCE_DIR> [OPTIONS]

Options:
  --target {generic-c99,bare-arm-q31,bare-arm-q15}
  -o, --output PATH          Output directory (default: out/)
  --embed-assets             Embed data assets
  --amalgamate               Single .c/.h output
  --verify                   Run Z3 + fuzzing after compilation
  --verbose                  Detailed diagnostics
```

### `purce extract`

```bash
purce extract <SOURCE_DIR> [OPTIONS]

Options:
  --target {generic-c99,bare-arm-q31,bare-arm-q15}
  -o, --output PATH          Output directory
  --embed-assets             Embed data assets
  --provenance-only          Manifest without code
```

### `purce verify`

```bash
purce verify [OPTIONS]

Options:
  --iterations N             Fuzzing iterations per operation (default: 10000)
  --seed N                   Random seed for reproducibility
```

## Adding New Operations

1. Add parser mapping in `purce/parser/python_parser.py`
2. Add IR builder mapping in `purce/ir/builder.py`
3. Add C99 kernel body in `purce/backend/c99_generator.py` (MATH_KERNEL_BODIES)
4. Add parameter mapping in `purce/backend/c99_generator.py` (BODY_PARAM_MAP)
5. Add derived params in `purce/backend/c99_generator.py` (DERIVED_PARAMS)
6. Add fuzzer method in `purce/verifier/fuzzer.py`
7. Add tests in `tests/`

## Benchmarks

```bash
# Numerical accuracy (87,000 iterations)
python -m benchmarks.numerical_accuracy

# Performance scaling
python -m benchmarks.performance_scaling

# Edge cases
python -m benchmarks.edge_cases

# Code metrics
python -m benchmarks.code_metrics

# C code quality
python -m benchmarks.c_code_quality
```

## Limitations

- **Not all NumPy operations are supported**: Only the 25 operations listed above
- **No dynamic shapes**: All dimensions must be known at compile time (passed as parameters)
- **No in-place mutation**: Generated functions take separate input/output arrays
- **Stack-allocated linalg**: `linalg_solve` and `linalg_inv` use fixed-size stack arrays (max 64×64)
- **No GPU/CUDA**: Generated code is pure C99, no GPU offloading

## Troubleshooting

**"No math kernels found"**: Your Python code doesn't use recognized NumPy operations. Add `import numpy as np` and use supported functions.

**`#error` in generated code**: The operation was recognized but cannot be mapped to a known kernel. Check if the operation is in the supported list.

**Z3 not available**: Install `z3-solver`: `pip install z3-solver`

**Fuzz failures**: Increase tolerance or check if the operation has known numerical precision limits.
