# Architecture Deep Dive

This document describes the internal architecture of Purce in detail.

## Pipeline Overview

```
Input (Python/NumPy)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: PARSING                                           │
│  ─────────────────                                          │
│  • Python ast.Module → FunctionDef nodes                    │
│  • Detect NumPy calls (numpy.*, np.*)                       │
│  • Extract type hints, argument names, call graphs          │
│  • Emit diagnostics for unsupported constructs              │
│  • Output: List[ParsedFunction]                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Math-IR CONSTRUCTION                              │
│  ──────────────────────────                                 │
│  • Convert ParsedFunction → MathIRNode                      │
│  • Classify effects (PURE, IO, TEMPORAL, RANDOM, ALLOC)     │
│  • Map NumPy operations to algorithms                       │
│  • Build dependency graph (nested_deps)                     │
│  • Output: MathIRGraph (DAG)                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: SEMANTIC SLICING                                  │
│  ──────────────────────                                     │
│  • Resolve call graph from entry points                     │
│  • Classify dependencies:                                   │
│    - MATH_KERNEL → include in output                        │
│    - SYSTEM_PAL → generate stub                             │
│    - DATA_ASSET → error (or embed)                          │
│    - META_UTIL → inline/eliminate                           │
│  • Dead code elimination                                    │
│  • Validate DAG (no cycles, no missing deps)                │
│  • Output: Sliced MathIRGraph                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: C99 CODE GENERATION                              │
│  ───────────────────────────                                │
│  • Topological sort of nodes                                │
│  • Generate .c files with C99-SOS headers                   │
│  • Generate .h header file                                  │
│  • Generate .prov.json provenance files                     │
│  • Generate CMakeLists.txt                                  │
│  • Zero heap allocation in math kernels                     │
│  • Output: GeneratedFile list                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: VERIFICATION                                      │
│  ────────────────────                                       │
│  • Z3 SMT: Generate verification conditions                 │
│    - Dimension bounds                                       │
│    - Division by zero                                       │
│    - Overflow checks                                        │
│    - Non-singular matrix checks                             │
│    - FFT power-of-two constraints                           │
│  • Differential Fuzzing:                                    │
│    - Generate random inputs via Hypothesis                  │
│    - Run Python reference implementation                    │
│    - Run generated C code (ctypes)                          │
│    - Compare outputs within tolerance                       │
│    - Minimum 10,000 iterations per operation                │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Stage 1: Parser Input/Output

**Input**: Python source file
**Output**: `ParseResult` containing:
- `functions: List[ParsedFunction]` — extracted math kernels
- `diagnostics: List[Diagnostic]` — errors and warnings

```python
@dataclass
class ParsedFunction:
    name: str                    # "dot_product"
    module: str                  # "mymodule"
    node_id: str                 # "mymodule.dot_product_a1b2c3d4"
    origin_file: str             # "mymodule.py"
    origin_line: int             # 42
    origin_signature: str        # "float64 a -> float64 b -> float64"
    math_intent: str             # "Math kernel 'dot_product' implementing matmul"
    algorithm: str               # "matmul"
    inputs: List[Tuple[str, Dtype, str]]
    outputs: List[Tuple[str, Dtype, str]]
    effects: List[Effect]
    reductions: List[ReductionEntry]
    nested_deps: List[str]
    stack_usage: Optional[int]
    heap_usage: Optional[int]
    reentrant: bool
    raw_body: str
```

### Stage 2: Math-IR Node

Each node in the Math-IR graph represents one atomic mathematical operation.

```python
@dataclass
class MathIRNode:
    node_id: str                    # Unique identifier
    origin_symbol: str              # Original qualified name
    origin_file: str                # Source file
    origin_line: int                # Source line
    origin_commit: Optional[str]    # Git commit
    origin_signature: str           # Type signature
    
    math_intent: str                # Human-readable description
    inputs: List[Tuple[str, Dtype, str]]
    outputs: List[Tuple[str, Dtype, str]]
    
    effects: List[Effect]           # Side effects
    algorithm: str                  # Implementation algorithm
    
    reductions: List[ReductionEntry]
    nested_deps: List[str]          # Dependencies
    
    stack_usage: Optional[int]      # Stack bytes
    heap_usage: Optional[int]       # Heap bytes
    reentrant: bool                 # Thread-safe
    dep_kind: DepKind               # MATH_KERNEL, SYSTEM_PAL, etc.
```

### Stage 3: Slicer Classification

The slicer classifies each dependency:

```
Node → Check effects
  │
  ├─ PURE + no PAL calls → MATH_KERNEL
  │   └─ Include in output, resolve deps recursively
  │
  ├─ IO, TEMPORAL, RANDOM → SYSTEM_PAL
  │   └─ Generate stub, add to manifest
  │
  ├─ DATA fetching → DATA_ASSET
  │   └─ Error unless --embed-assets
  │
  └─ Small utility → META_UTIL
      └─ Inline or eliminate
```

### Stage 4: Code Generation

The generator produces files in topological order:

```
For each node in topological_sort(graph):
  1. Generate C99-SOS function header
  2. Generate function body from algorithm template
  3. Write .c file
  4. Write .prov.json provenance file

Then:
  5. Generate .h header with forward declarations
  6. Generate CMakeLists.txt
```

### Stage 5: Verification

**Z3 SMT Verification**:
```python
# For each node, generate verification conditions:
s = z3.Solver()
n = z3.Int("n")
s.add(n > 0, n <= 1000000)  # Dimension bounds

# Check if conditions are satisfiable
result = s.check()
# UNSAT = safe, SAT = violation found
```

**Differential Fuzzing**:
```python
for _ in range(10000):
    # Generate random input
    a = random_array()
    b = random_array()
    
    # Reference implementation
    expected = numpy.add(a, b)
    
    # Generated C code
    actual = ctypes_callGenerated(a, b)
    
    # Compare
    assert allclose(expected, actual, rtol=1e-5)
```

## Key Design Decisions

### 1. Zero Heap Allocation

Math kernels never call `malloc()` or `free()`. All memory is caller-provided:

```c
// Caller allocates
double *C = malloc(m * n * sizeof(double));

// Generated code uses it in-place
void numpy_linalg_matmul_a1b2c3d4(
    const double *A, const double *B, double *C,
    int m, int n, int k
) {
    // No malloc, no free — just computation
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

### 2. C99-SOS Provenance

Every generated file has complete provenance — you can trace any C function back to the exact Python line that generated it.

### 3. Dependency Classification

Not all dependencies can be translated to C99. The slicer classifies them:

- **MATH_KERNEL**: Pure math → C99
- **SYSTEM_PAL**: OS calls → stubs (user implements)
- **DATA_ASSET**: File I/O → error (or embed)
- **META_UTIL**: Small helpers → inline

### 4. Algorithm Templates

Each algorithm has a fixed C99 template:

```
matmul      → Triple loop (i, j, k)
element_add → Single loop
reduce_sum  → Single loop with accumulator
fft         → Cooley-Tukey in-place
linalg_solve → Gaussian elimination with partial pivoting
```

Templates are parameterized by input types and dimensions.

### 5. Fixed-Point Support

For bare-metal targets, Purce generates Q31/Q15 fixed-point arithmetic:

```c
typedef int32_t q31_t;
#define Q31_ONE ((q31_t)(1 << 30))

static inline q31_t q31_mul(q31_t a, q31_t b) {
    return (q31_t)(((int64_t)a * (int64_t)b) >> 30);
}
```
