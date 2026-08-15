# Purce Testing Guide

## Test Structure

```
tests/
├── test_ir.py               MathIRNode, MathIRGraph, topological sort
├── test_ir_coverage_a.py    IR builder coverage sweep A (34 tests)
├── test_ir_coverage_b.py    IR builder coverage sweep B (28 tests)
├── test_ir_coverage_c.py    IR builder coverage sweep C (29 tests)
├── test_parser.py           PythonParser (parseable NumPy operations)
├── test_slicer.py           SemanticSlicer (pruning, classification)
├── test_backend.py          C99Generator (headers, naming, output)
├── test_verifier.py         DifferentialFuzzer, Z3Verifier, CBackend (requires gcc)
├── test_verifier_edges.py   Verifier edge/corner cases (62 tests)
├── test_coverage_sweep.py   End-to-end coverage sweep (91 reachable kernels vs NumPy)
├── test_edge_coverage.py    IEEE-754 corner inputs for every element-wise kernel
├── test_generated_kernel_runtime.py  Runtime correctness gate for generated kernels
├── test_reproducible.py     Corpus-gate determinism checks
├── test_cli.py              CLI extract/compile/verify commands
├── test_integration.py      Full pipeline integration tests
├── test_realworld.py        Real-world ML/scientific pipeline tests
├── test_c_compilation.py    C compilation verification (requires gcc)
├── verification_agent.py    5-phase verification sub-agent
└── realworld/               Real-world test projects (23 .py sources)
    ├── activations.py          Activation functions
    ├── attention.py            Attention mechanisms
    ├── computer_vision.py      Computer vision operations
    ├── convolution.py          Convolution operations
    ├── data_pipeline.py        Data processing
    ├── extra_patterns.py       Additional usage patterns
    ├── generative.py           Generative model operations
    ├── graph_neural_networks.py  GNN operations
    ├── jax_ops.py              JAX-style operations
    ├── layers.py               Neural network layers
    ├── linear_algebra.py       Linear algebra operations
    ├── losses.py               Loss functions
    ├── model.py                Model definitions
    ├── normalization.py        Normalization layers
    ├── optimizers.py           Optimizer implementations
    ├── pytorch_ops.py          PyTorch-style operations
    ├── recommendation.py       Recommendation operations
    ├── reinforcement_learning.py  RL operations
    ├── scipy_ops.py            SciPy-style operations
    ├── signal_processing.py    Signal processing
    ├── time_series.py          Time series operations
    └── transformers.py         Transformer operations
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
2. **Fuzz tests**: legacy 25-op fuzzer + full 91-reachable coverage sweep (C99 vs Python)
3. **Pipeline integration**: All real-world sources
4. **Synthetic pipeline**: 5 hand-crafted sources
5. **Memory safety**: Heap-free and provenance verification

Exit code 0 = all pass, 1 = failure.

## Test Categories

### Unit Tests (487 total)

**Math-IR (`test_ir.py` / `test_ir_coverage_a/b/c.py`)**:
- Node creation and validation
- Graph operations (add, remove, get)
- Topological sort (Kahn's algorithm)
- Cycle detection
- DAG validation
- Dependency tracking
- IR builder coverage sweeps (34 + 28 + 29 tests across A/B/C)

**Parser (`test_parser.py`)**:
- Full coverage of the 91 reachable kernel bodies
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

**Verifier (`test_verifier.py` / `test_verifier_edges.py` / `test_coverage_sweep.py`)**:
- Legacy fuzzer: all 25 operations
- Z3 verification conditions
- Tolerance checking
- Edge case handling
- Reproducibility (seed)
- Verifier edge/corner cases (62 tests)

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
- **Parser**: All 91 reachable operations' parse paths tested (+ edge/corner cases)
- **Slicer**: All dependency types tested
- **Backend**: All 91 reachable kernel bodies tested (+ documented-unreachable `array_diff`)
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

# Real-world pipeline benchmark (generates benchmarks/results.md)
python -m benchmarks.real_benchmark

# Reproducible corpus gate (14 kernels → 812 C files, byte-reproducible)
python -m benchmarks.reproducible --baseline benchmarks/baseline.json

# Everything in one run
python -m benchmarks.run_all
```
