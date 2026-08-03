# Purce Testing Guide

## Test Structure

```
tests/
├── test_ir.py              MathIRNode, MathIRGraph, topological sort
├── test_parser.py           PythonParser (all 25+ operations)
├── test_slicer.py           SemanticSlicer (pruning, classification)
├── test_backend.py          C99Generator (headers, naming, output)
├── test_verifier.py         DifferentialFuzzer, Z3Verifier, CBackend (requires gcc)
├── test_cli.py              CLI extract/compile/verify commands
├── test_integration.py      Full pipeline integration tests
├── test_realworld.py        Real-world ML/scientific pipeline tests
├── test_c_compilation.py    C compilation verification (requires gcc)
├── verification_agent.py    5-phase verification sub-agent
└── realworld/               Real-world test projects
    ├── layers.py            Neural network layers
    ├── activations.py       Activation functions
    ├── losses.py            Loss functions
    ├── optimizers.py        Optimizer implementations
    ├── normalization.py     Normalization layers
    ├── attention.py         Attention mechanisms
    ├── convolution.py       Convolution operations
    ├── linear_algebra.py    Linear algebra operations
    ├── signal_processing.py Signal processing
    ├── data_pipeline.py     Data processing
    ├── model.py             Model definitions
    ├── jax_ops.py           JAX-style operations
    ├── pytorch_ops.py       PyTorch-style operations
    └── scipy_ops.py         SciPy-style operations
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_backend.py -v

# With coverage
pytest tests/ --cov=purce --cov-report=html

# Stop on first failure
pytest -x

# Run specific test class
pytest tests/test_backend.py::TestC99GeneratorHeaders -v

# Run specific test
pytest tests/test_backend.py::TestC99GeneratorHeaders::test_generates_purce_output_header -v
```

## Verification Agent

```bash
python -m tests.verification_agent
```

Runs 5 phases:
1. **Unit tests**: All pytest tests
2. **Fuzz tests**: 25 operations × 200 iterations (C99 vs Python)
3. **Pipeline integration**: All real-world sources
4. **Synthetic pipeline**: 5 hand-crafted sources
5. **Memory safety**: Heap-free and provenance verification

Exit code 0 = all pass, 1 = failure.

## Test Categories

### Unit Tests (188 total)

**Math-IR (`test_ir.py`)**:
- Node creation and validation
- Graph operations (add, remove, get)
- Topological sort (Kahn's algorithm)
- Cycle detection
- DAG validation
- Dependency tracking

**Parser (`test_parser.py`)**:
- All 25+ NumPy operations
- Function extraction
- Type inference
- Diagnostic generation
- Multi-function files
- Edge cases (nested, lambda, comprehension)

**Slicer (`test_slicer.py`)**:
- Dead code elimination
- Dependency classification
- Call graph resolution
- PAL stub generation
- Data asset detection
- Entry point resolution

**Backend (`test_backend.py`)**:
- C99-SOS headers
- Function naming
- Parameter mapping
- Kernel body substitution
- Provenance JSON
- CMakeLists.txt generation
- Header file generation

**Verifier (`test_verifier.py`)**:
- All 25 fuzz operations
- Z3 verification conditions
- Tolerance checking
- Edge case handling
- Reproducibility (seed)

**CLI (`test_cli.py`)**:
- Extract command
- Compile command
- Verify command
- Error handling
- Output formatting

### Integration Tests (`test_integration.py`)

Full pipeline tests: parse → IR → slice → generate → write → verify

### Real-World Tests (`test_realworld.py`)

Tests against real-world-style codebases (JAX, PyTorch, SciPy patterns).

## Writing New Tests

```python
import pytest
from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer


def _run_pipeline(source: str, module_name: str):
    parser = PythonParser(target_profile="generic-c99")
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    slice_result = slicer.slice(graph, list(graph.nodes.keys()))
    generator = C99Generator(target_profile="generic-c99")
    return generator.generate(slice_result.graph, module_name=module_name), graph


class TestMyNewFeature:
    def test_basic(self) -> None:
        src = """\
import numpy as np
def my_func(x):
    return np.add(x, x)
"""
        gen, graph = _run_pipeline(src, "test_mod")
        assert len(graph.nodes) >= 1
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        assert "PURCE OUTPUT" in c_files[0].content

    def test_error_handling(self) -> None:
        with pytest.raises(SomeError):
            # test error case
            pass
```

## Coverage Targets

- **Math-IR**: 100% of node/graph operations
- **Parser**: All 25+ operations tested
- **Slicer**: All dependency types tested
- **Backend**: All kernel bodies tested
- **Verifier**: All fuzz operations tested
- **CLI**: All commands tested

## Performance Benchmarks

```bash
# Numerical accuracy
python -m benchmarks.numerical_accuracy

# Scaling curves
python -m benchmarks.performance_scaling

# Edge cases
python -m benchmarks.edge_cases

# Code metrics
python -m benchmarks.code_metrics

# C code quality
python -m benchmarks.c_code_quality
```
