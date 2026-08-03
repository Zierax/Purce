"""Integration tests: full pipeline from Python source to generated C99 code.

These tests exercise the complete purce pipeline end-to-end:
  Python source -> parse -> build IR -> slice -> generate C99 -> verify

Each test constructs a Python source string, runs it through every stage,
and asserts correctness of the intermediate and final artifacts.
"""
from __future__ import annotations

import os
import re
import tempfile

import pytest

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_pipeline(
    source: str,
    module_name: str = "test_mod",
    target: str = "generic-c99",
) -> tuple:
    """Run the full pipeline and return (gen_result, graph, slice_result)."""
    parser = PythonParser(target_profile=target)
    parsed = parser.parse_source(source, f"{module_name}.py")

    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)

    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)

    generator = C99Generator(target_profile=target)
    gen_result = generator.generate(slice_result.graph, module_name=module_name)

    return gen_result, graph, slice_result


def _has_c_function(content: str) -> bool:
    return bool(re.search(r'\b(void|int|double|float)\s+\w+\s*\(', content))


# ── Single-operation pipelines ───────────────────────────────────────────────


class TestPipelineElementAdd:
    SRC = "import numpy as np\ndef add_arrays(a, b):\n    return np.add(a, b)"

    def test_generates_c_file(self) -> None:
        gen, graph, sliced = _run_pipeline(self.SRC)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 1

    def test_has_provenance(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        assert len(prov_files) == 1

    def test_has_header(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        h_files = [f for f in gen.files if f.file_type == "h"]
        assert len(h_files) == 1
        assert "#ifndef" in h_files[0].content

    def test_c_file_has_purce_header(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert "PURCE OUTPUT" in c_files[0].content

    def test_c_file_includes_math_h(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert "#include <math.h>" in c_files[0].content

    def test_c_file_has_function_signature(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert _has_c_function(c_files[0].content)
        assert "restrict" in c_files[0].content

    def test_no_malloc_in_output(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        for f in gen.files:
            if f.file_type == "c":
                assert "malloc" not in f.content
                assert "calloc" not in f.content
                assert "realloc" not in f.content

    def test_has_cmake(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        cmake = [f for f in gen.files if f.file_type == "cmake"]
        assert len(cmake) == 1
        assert "cmake_minimum_required" in cmake[0].content

    def test_provenance_json_valid(self) -> None:
        import json
        gen, _, _ = _run_pipeline(self.SRC)
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        prov = json.loads(prov_files[0].content)
        assert "source" in prov
        assert "ir_node" in prov
        assert "memory" in prov


# ── Matmul pipeline ──────────────────────────────────────────────────────────


class TestPipelineMatmul:
    SRC = "import numpy as np\ndef matmul_op(A, B):\n    return np.matmul(A, B)"

    def test_matmul_generates_c(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 1
        assert _has_c_function(c_files[0].content)

    def test_matmul_has_dimension_params(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        h_files = [f for f in gen.files if f.file_type == "h"]
        header = h_files[0].content
        assert "void" in header, "Header should declare a void function"
        assert "restrict" in header
        assert "double" in header


# ── Reduce sum pipeline ──────────────────────────────────────────────────────


class TestPipelineReduceSum:
    def test_reduce_sum_generates_c(self) -> None:
        src = "import numpy as np\ndef sum_op(x):\n    return np.sum(x)"
        gen, _, _ = _run_pipeline(src)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 1
        assert _has_c_function(c_files[0].content)


# ── Unary operations ─────────────────────────────────────────────────────────


class TestPipelineElementUnary:
    @pytest.mark.parametrize("op_name,op_call", [
        ("sqrt", "np.sqrt"),
        ("exp", "np.exp"),
        ("log", "np.log"),
        ("sin", "np.sin"),
        ("cos", "np.cos"),
        ("abs", "np.abs"),
        ("tan", "np.tan"),
    ])
    def test_unary_op_generates_c(self, op_name: str, op_call: str) -> None:
        src = f"import numpy as np\ndef my_{op_name}(x):\n    return {op_call}(x)"
        gen, _, _ = _run_pipeline(src)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 1
        assert _has_c_function(c_files[0].content)


# ── Allocation operations ────────────────────────────────────────────────────


class TestPipelineAllocOps:
    @pytest.mark.parametrize("op_name,op_call", [
        ("zeros", "np.zeros"),
        ("ones", "np.ones"),
        ("eye", "np.eye"),
    ])
    def test_alloc_op_generates_c(self, op_name: str, op_call: str) -> None:
        src = f"import numpy as np\ndef my_{op_name}(n):\n    return {op_call}(n)"
        gen, _, _ = _run_pipeline(src)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 1
        assert _has_c_function(c_files[0].content)


# ── Multi-operation pipelines ────────────────────────────────────────────────


class TestPipelineMultiOp:
    def test_multi_op_extracts_function(self) -> None:
        src = """\
import numpy as np

def add_then_mul(a, b, c):
    x = np.add(a, b)
    return np.multiply(x, c)
"""
        gen, graph, sliced = _run_pipeline(src)
        assert len(graph.nodes) >= 1

    def test_multi_op_generates_c_files(self) -> None:
        src = """\
import numpy as np

def add_then_mul(a, b, c):
    x = np.add(a, b)
    return np.multiply(x, c)
"""
        gen, _, _ = _run_pipeline(src)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content)

    def test_multi_function_extracts_all(self) -> None:
        src = """\
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)

def mul_arrays(a, b):
    return np.multiply(a, b)
"""
        gen, graph, _ = _run_pipeline(src)
        assert len(graph.nodes) == 2

    def test_linalg_pipeline(self) -> None:
        src = """\
import numpy as np

def solve_system(A, b):
    return np.linalg.solve(A, b)

def invert_matrix(A):
    return np.linalg.inv(A)
"""
        gen, graph, _ = _run_pipeline(src)
        assert len(graph.nodes) == 2


# ── Write-to-disk integration ────────────────────────────────────────────────


class TestPipelineWriteDisk:
    SRC = "import numpy as np\ndef add_arrays(a, b):\n    return np.add(a, b)"

    def test_write_all_creates_files(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.write_all(tmpdir)
            files = os.listdir(tmpdir)
            c_files = [f for f in files if f.endswith(".c")]
            h_files = [f for f in files if f.endswith(".h")]
            prov_files = [f for f in files if f.endswith(".prov.json")]
            assert len(c_files) == 1
            assert len(h_files) == 1
            assert len(prov_files) == 1
            assert "CMakeLists.txt" in files

    def test_written_c_files_are_valid_utf8(self) -> None:
        gen, _, _ = _run_pipeline(self.SRC)
        with tempfile.TemporaryDirectory() as tmpdir:
            gen.write_all(tmpdir)
            for f in os.listdir(tmpdir):
                if f.endswith(".c") or f.endswith(".h"):
                    with open(os.path.join(tmpdir, f), encoding="utf-8") as fh:
                        content = fh.read()
                    assert len(content) > 0
                    assert _has_c_function(content)


# ── Error handling ───────────────────────────────────────────────────────────


class TestPipelineErrorHandling:
    def test_syntax_error_produces_empty_graph(self) -> None:
        src = "def broken(\n"
        builder = MathIRBuilder(origin_file="test")
        graph = builder.build_from_source(src, module="test")
        assert len(graph.nodes) == 0

    def test_no_numpy_produces_empty_graph(self) -> None:
        src = "def pure_python(x):\n    return x + 1"
        builder = MathIRBuilder(origin_file="test")
        graph = builder.build_from_source(src, module="test")
        assert len(graph.nodes) == 0

    def test_empty_source_produces_empty_graph(self) -> None:
        builder = MathIRBuilder(origin_file="test")
        graph = builder.build_from_source("", module="test")
        assert len(graph.nodes) == 0


# ── Verification integration ─────────────────────────────────────────────────


class TestPipelineVerification:
    def test_fuzz_element_add(self) -> None:
        fuzzer = DifferentialFuzzer(seed=42)
        result = fuzzer.fuzz_element_add(iterations=100)
        assert result.all_passed
        assert not result.tested_c

    def test_fuzz_matmul(self) -> None:
        fuzzer = DifferentialFuzzer(seed=42)
        result = fuzzer.fuzz_matmul(iterations=100)
        assert result.all_passed

    def test_fuzz_all_passes(self) -> None:
        fuzzer = DifferentialFuzzer(seed=42)
        results = fuzzer.fuzz_all(iterations=50)
        for op, result in results.items():
            assert result.all_passed, f"Fuzz failed for {op}: {result.failures[:2]}"
