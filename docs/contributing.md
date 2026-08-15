# Contributing to Purce

## Development Setup

```bash
# Clone the repository
git clone https://github.com/youruser/purce.git
cd purce

# Install in development mode
pip install -e ".[dev]"

# Verify installation
purce --version
```

## Code Style

Purce follows strict code quality standards:

### Python

- **Formatter/Linter**: `ruff` (configured in `pyproject.toml`)
- **Type hints**: Required everywhere
- **Line length**: 100 characters
- **Target**: Python 3.11+

```bash
# Check for issues
make lint

# Auto-fix
make format
```

### Generated C99

- **Standard**: C99 (C89 + `//` comments, mid-block declarations)
- **No heap**: Math kernels never use `malloc()` or `free()`
- **C99-SOS**: Every file/function has mandatory comment blocks
- **Naming**: `{lib}_{module}_{op}_{variant}` for functions

## Testing

### Running Tests

```bash
# Run all 486 tests
pytest tests/ -v

# Run specific module
pytest tests/test_ir.py -v

# Run with coverage
pytest tests/ --cov=purce --cov-report=html

# Stop on first failure
pytest -x
```

### Test Structure

```
tests/
├── test_ir.py              # 24 tests - MathIRNode, MathIRGraph, MathIRBuilder
├── test_parser.py          # 26 tests - PythonParser (all MVP operations)
├── test_slicer.py          # 18 tests - SemanticSlicer (pruning, classification)
├── test_backend.py         # 18 tests - C99Generator (headers, output, naming)
├── test_verifier.py        # 38 tests - DifferentialFuzzer, Z3Verifier, CBackend (requires gcc)
├── test_cli.py             # 12 tests - CLI extract/compile/verify commands
├── test_integration.py     # 33 tests - Full pipeline integration
├── test_realworld.py       # 42 tests - Real-world ML/scientific code
├── test_c_compilation.py   # 15 tests - C compilation verification (requires gcc)
├── verification_agent.py   # 5-phase verification agent
├── fixtures/               # Sample Python files for testing
└── realworld/              # 23 ML/scientific test source files
```

### Writing Tests

```python
import pytest
from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode

def _make_node(node_id: str) -> MathIRNode:
    return MathIRNode(
        node_id=node_id,
        origin_symbol=f"mod.{node_id}",
        origin_file="test.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="double -> double",
        math_intent="test kernel",
        inputs=[("x", Dtype.FLOAT64, "scalar")],
        outputs=[("result", Dtype.FLOAT64, "scalar")],
        effects=[Effect.PURE],
        algorithm="test",
        stack_usage=64,
    )

class TestMyFeature:
    def test_basic(self) -> None:
        node = _make_node("test")
        assert node.node_id == "test"
        assert node.is_pure() is True
```

## Project Structure

```
purce/
├── purce/                    # Main package
│   ├── __init__.py          # Version
│   ├── cli.py               # CLI entry point
│   ├── parser/              # Source code parsing
│   │   └── python_parser.py
│   ├── ir/                  # Intermediate representation
│   │   ├── nodes.py
│   │   └── builder.py
│   ├── slicer/              # Semantic slicing
│   │   └── semantic_slicer.py
│   ├── backend/             # Code generation
│   │   └── c99_generator.py
│   ├── verifier/            # Verification
│   │   ├── z3_verifier.py
│   │   └── fuzzer.py
│   └── pal/                 # Platform abstraction layer
├── tests/                   # Test suite
├── benchmarks/              # Performance benchmarks
├── docs/                    # Documentation
├── pyproject.toml
└── README.md
```

## Adding New Operations

To add support for a new NumPy operation:

### 1. Add to parser mapping

```python
# purce/parser/python_parser.py
SUPPORTED_MODULES = {
    "numpy": {
        # ... existing operations
        "new_operation",  # Add here
    },
}
```

### 2. Add to IR builder mapping

```python
# purce/ir/builder.py
NUMPY_OP_MAP = {
    # ... existing mappings
    "numpy.new_operation": "new_algorithm",
}
```

### 3. Add C99 template

```python
# purce/backend/c99_generator.py
MATH_KERNEL_BODIES = {
    # ... existing templates
    "new_algorithm": """\
    /* New algorithm description */
    for (int i = 0; i < n; i++) {
        out[i] = /* implementation */;
    }""",
}
```

### 4. Add verification

```python
# purce/verifier/fuzzer.py
class DifferentialFuzzer:
    def fuzz_new_operation(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_unary_op(
            "new_operation",
            self._new_operation_python,
            iterations,
        )
    
    @staticmethod
    def _new_operation_python(x: List[float], n: int) -> float:
        # Reference implementation
        return sum(x) / n
```

### 5. Add tests

```python
# tests/test_parser.py
class TestPythonParserNewOperation:
    def test_new_operation(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def new_op(x):
    return np.new_operation(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "new_algorithm"
```

## Benchmarks

```bash
# Run full benchmark suite
make benchmark

# Results written to benchmarks/results.md
```

## Documentation

```bash
# Build docs (if using Sphinx)
cd docs/
make html
```

## Pull Request Checklist

- [ ] All tests pass (486 collected; gcc-dependent tests skip gracefully without a compiler)
- [ ] No lint errors (`make lint`)
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Benchmark results included (if performance-related)
- [ ] C99-SOS standard followed for generated code
