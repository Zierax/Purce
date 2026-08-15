# API Reference

## CLI Commands

### `purce extract`

```bash
purce extract <SOURCE_DIR> [OPTIONS]
```

**Arguments:**
- `SOURCE_DIR` — Directory containing Python files to extract from

**Options:**
- `--target {generic-c99,bare-arm-q31,bare-arm-q15}` — Target profile (default: `generic-c99`)
- `-o, --output PATH` — Output directory (default: `out/`)
- `--embed-assets` — Embed data assets instead of erroring
- `--amalgamate` — Generate single .c/.h output
- `--verbose` — Show extraction diagnostics
- `--provenance-only` — Generate manifest without code
- `--entry NAME` — Only emit this top-level function (repeatable)

**Exit Codes:**
- `0` — Success (also used when no math kernels are found, with a warning)
- `1` — Data assets found without `--embed-assets`

**Example:**
```bash
$ purce extract ./mylib/ --target generic-c99 -o ./out/ --verbose
purce extract: ./mylib/ -> ./out/ (target: generic-c99)
Found 5 Python files
  Extracted: matmul_kernel (matmul)
  Extracted: fft_impl (fft)
  Extracted: reduce_sum (reduce_sum)
Extracted 3 math kernels

Output written to ./out/
  3 .c files
  1 .h files
  3 .prov.json files
  1 CMakeLists.txt

Done.
```

### `purce compile`

```bash
purce compile <SOURCE_DIR> [OPTIONS]
```

**Arguments:**
- `SOURCE_DIR` — Directory containing Python files to compile

**Options:**
- `--target {generic-c99,bare-arm-q31,bare-arm-q15}` — Target profile (default: `generic-c99`)
- `-o, --output PATH` — Output directory (default: `out/`)
- `--embed-assets` — Embed data assets instead of erroring
- `--amalgamate` — Generate single .c/.h output
- `--verbose` — Show compilation diagnostics
- `--verify` — Run Z3 + differential fuzzing after compilation
- `--entry NAME` — Only emit this top-level function (repeatable)

**Exit Codes:**
- `0` — Success (also used when no math kernels are found, with a warning)
- `1` — No Python files found, data assets without `--embed-assets`, Z3 `SAT` counterexample (`--verify`), or fuzz failure (`--verify`)

**Example:**
```bash
$ purce compile ./myproject/ -o ./out/ --verify
purce compile: ./myproject/ -> ./out/ (target: generic-c99)
Found 8 Python files
Found 5 math kernel functions

Output written to ./out/
  5 .c files
  1 .h files
  5 .prov.json files
  1 CMakeLists.txt

Running verification...
  Z3 verification: 5/5 nodes verified
  Fuzz matmul: PASS (200/200)
  Fuzz element_add: PASS (200/200)
  ...

Done.
```

### `purce verify`

```bash
purce verify [OPTIONS]
```

**Options:**
- `--iterations N` — Fuzzing iterations per operation (default: 10000)
- `--seed N` — Random seed for reproducible fuzzing

**Exit Codes:**
- `0` — All operations passed differential fuzzing against compiled C
- `1` — Z3 verification violation (SAT), fuzz failure, or Z3 unavailable
- `3` — No C compiler available; results were Python self-comparison only (unverified)

**Example:**
```bash
$ purce verify --iterations 1000
purce verify: running 1000 iterations per operation
Z3 solver: available

Running differential fuzzing (1000 iterations)...
  matmul                PASS (1000/1000 passed)
  element_add           PASS (1000/1000 passed)
  element_sub           PASS (1000/1000 passed)
  element_mul           PASS (1000/1000 passed)
  element_div           PASS (1000/1000 passed)
  reduce_sum            PASS (1000/1000 passed)
  reduce_mean           PASS (1000/1000 passed)
  reduce_max            PASS (1000/1000 passed)
  reduce_min            PASS (1000/1000 passed)
  linalg_solve          PASS (1000/1000 passed)
  linalg_inv            PASS (1000/1000 passed)
  fft                   PASS (1000/1000 passed)

All operations passed differential fuzzing.
```

---

## Python API

### `PythonParser`

```python
from purce.parser.python_parser import PythonParser

parser = PythonParser(target_profile="generic-c99")
result = parser.parse_source(source_code, filename="mymodule.py")

# Access extracted functions
for fn in result.functions:
    print(f"{fn.name}: {fn.algorithm}")

# Access diagnostics
for diag in result.diagnostics:
    print(f"[{diag.severity}] {diag.reason}")

# Convert to Math-IR graph
graph = result.to_graph()
```

### `MathIRBuilder`

```python
from purce.ir.builder import MathIRBuilder

builder = MathIRBuilder(origin_file="mymodule.py")
graph = builder.build_from_source(source_code, module="mymodule")

# Access nodes
for nid, node in graph.nodes.items():
    print(f"{node.node_id}: {node.algorithm}")

# Topological sort
ordered = graph.topological_sort()
```

### `SemanticSlicer`

```python
from purce.slicer.semantic_slicer import SemanticSlicer

slicer = SemanticSlicer(embed_assets=False)
result = slicer.slice(graph, entry_points=["main"])

# Sliced graph contains only reachable nodes
sliced_graph = result.graph

# PAL stubs generated
print(result.pal_stubs)

# Data assets that need embedding
print(result.data_assets)
```

### `C99Generator`

```python
from purce.backend.c99_generator import C99Generator

generator = C99Generator(
    target_profile="generic-c99",
    fixed_point=False,
    amalgamate=False,
)
result = generator.generate(graph, module_name="mymodule")

# Write files
result.write_all("./out/")

# Access generated files
for gf in result.files:
    print(f"{gf.path}: {gf.file_type} ({len(gf.content)} bytes)")
```

### `DifferentialFuzzer`

```python
from purce.verifier.fuzzer import DifferentialFuzzer

fuzzer = DifferentialFuzzer(rtol=1e-5, atol=1e-8)

# Fuzz specific operation
result = fuzzer.fuzz_matmul(iterations=10000)
print(f"{result.passed}/{result.iterations} passed")

# Fuzz all operations
results = fuzzer.fuzz_all(iterations=10000)
for op, res in results.items():
    print(f"{op}: {res.success_rate:.1%}")
```

### `Z3Verifier`

```python
from purce.verifier.z3_verifier import Z3Verifier

verifier = Z3Verifier()

# Verify single node
report = verifier.verify_node(node)
print(f"{report.verified_count}/{len(report.conditions)} conditions verified")

# Verify entire graph
reports = verifier.verify_graph(graph)
for nid, report in reports.items():
    print(f"{nid}: {'PASS' if report.all_verified else 'FAIL'}")
```

---

## Data Types

### `Dtype`

```python
class Dtype(Enum):
    FLOAT32 = auto()
    FLOAT64 = auto()
    INT32 = auto()
    INT64 = auto()
    Q15 = auto()       # Fixed-point 1.15
    Q31 = auto()       # Fixed-point 1.31
    BOOL = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()
```

### `Effect`

```python
class Effect(Enum):
    PURE = auto()      # No side effects, deterministic
    IO = auto()        # Reads/writes external state
    TEMPORAL = auto()  # Depends on time
    RANDOM = auto()    # Uses randomness
    ALLOC = auto()     # Performs memory allocation
```

### `DepKind`

```python
class DepKind(Enum):
    MATH_KERNEL = 1    # Translatable to C99
    SYSTEM_PAL = 2     # OS/Platform dependent
    DATA_ASSET = 3     # Runtime data fetching
    META_UTIL = 4      # Small utilities
```
