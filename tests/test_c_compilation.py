"""C compilation verification tests.

Generates C99 code from Python sources and compiles each .c file with gcc
to verify the generated code is syntactically valid C99.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.slicer.semantic_slicer import SemanticSlicer

_HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")


def _run_pipeline(source: str, module_name: str = "test_mod") -> C99Generator.GenerateResult:
    parser = __import__("purce.parser.python_parser", fromlist=["PythonParser"]).PythonParser(target_profile="generic-c99")
    parsed = parser.parse_source(source, f"{module_name}.py")
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)
    generator = C99Generator(target_profile="generic-c99")
    return generator.generate(slice_result.graph, module_name=module_name)


def _find_gcc() -> str:
    for name in ("gcc", "cc"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("gcc not found")


def _compile_c_file(c_path: str, gcc: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            [gcc, "-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic",
             "-o", out_path, c_path, "-lm"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


SOURCES = {
    "element_add": "import numpy as np\ndef add_arrays(a, b):\n    return np.add(a, b)",
    "matmul": "import numpy as np\ndef matmul_op(A, B):\n    return np.matmul(A, B)",
    "reduce_sum": "import numpy as np\ndef sum_op(x):\n    return np.sum(x)",
    "element_sqrt": "import numpy as np\ndef my_sqrt(x):\n    return np.sqrt(x)",
    "element_exp": "import numpy as np\ndef my_exp(x):\n    return np.exp(x)",
    "element_log": "import numpy as np\ndef my_log(x):\n    return np.log(x)",
    "element_sin": "import numpy as np\ndef my_sin(x):\n    return np.sin(x)",
    "element_cos": "import numpy as np\ndef my_cos(x):\n    return np.cos(x)",
    "element_tan": "import numpy as np\ndef my_tan(x):\n    return np.tan(x)",
    "element_abs": "import numpy as np\ndef my_abs(x):\n    return np.abs(x)",
    "alloc_zeros": "import numpy as np\ndef my_zeros(n):\n    return np.zeros(n)",
    "linalg_solve": "import numpy as np\ndef solve(A, b):\n    return np.linalg.solve(A, b)",
    "softmax": "import numpy as np\ndef sm(x):\n    return np.divide(np.exp(x), np.sum(np.exp(x)))",
    "layer_norm": (
        "import numpy as np\n"
        "def ln(x, g, b):\n"
        "    m = np.mean(x)\n"
        "    v = np.mean(np.power(np.subtract(x, m), 2.0))\n"
        "    n = np.divide(np.subtract(x, m), np.sqrt(np.add(v, 1e-5)))\n"
        "    return np.add(np.multiply(g, n), b)"
    ),
}


@RequiresGcc
class TestCCompilation:
    @pytest.mark.parametrize("name,source", list(SOURCES.items()))
    def test_generated_c_compiles(self, name: str, source: str) -> None:
        gen = _run_pipeline(source, name)
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1, f"No C files generated for {name}"

        gcc = _find_gcc()
        with tempfile.TemporaryDirectory() as tmpdir:
            for cf in c_files:
                c_path = os.path.join(tmpdir, cf.path)
                os.makedirs(os.path.dirname(c_path), exist_ok=True)
                with open(c_path, "w", encoding="utf-8") as fh:
                    fh.write(cf.content)
                ok, stderr = _compile_c_file(c_path, gcc)
                assert ok, f"Compilation failed for {name}/{cf.path}:\n{stderr}"

    def test_multi_op_compiles(self) -> None:
        src = """\
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)

def mul_arrays(a, b):
    return np.multiply(a, b)
"""
        gen = _run_pipeline(src, "multi_op")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) == 2

        gcc = _find_gcc()
        with tempfile.TemporaryDirectory() as tmpdir:
            for cf in c_files:
                c_path = os.path.join(tmpdir, cf.path)
                os.makedirs(os.path.dirname(c_path), exist_ok=True)
                with open(c_path, "w", encoding="utf-8") as fh:
                    fh.write(cf.content)
                ok, stderr = _compile_c_file(c_path, gcc)
                assert ok, f"Compilation failed for {cf.path}:\n{stderr}"


@RequiresGcc
class TestCCompilationNoWarnings:
    def test_element_add_no_warnings(self) -> None:
        gen = _run_pipeline(SOURCES["element_add"], "element_add")
        c_files = [f for f in gen.files if f.file_type == "c"]
        gcc = _find_gcc()
        with tempfile.TemporaryDirectory() as tmpdir:
            for cf in c_files:
                c_path = os.path.join(tmpdir, cf.path)
                os.makedirs(os.path.dirname(c_path), exist_ok=True)
                with open(c_path, "w", encoding="utf-8") as fh:
                    fh.write(cf.content)
                result = subprocess.run(
                    [gcc, "-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic",
                     "-fsyntax-only", c_path],
                    capture_output=True, text=True, timeout=15,
                )
                warnings = [l for l in result.stderr.splitlines() if "warning:" in l.lower()]
                assert result.returncode == 0, f"Compilation error:\n{result.stderr}"
                assert len(warnings) == 0, f"Warnings found:\n" + "\n".join(warnings)
