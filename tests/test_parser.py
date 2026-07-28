import pytest

from purce.parser.python_parser import PythonParser


@pytest.fixture
def parser() -> PythonParser:
    return PythonParser(target_profile="generic-c99")


class TestPythonParserDotProduct:
    def test_simple_dot(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        fn = result.functions[0]
        assert fn.name == "dot_product"
        assert fn.algorithm == "matmul"
        assert len(fn.inputs) == 2
        assert fn.inputs[0][0] == "a"
        assert fn.inputs[1][0] == "b"
        assert len(parser.diagnostics) == 0

    def test_matmul(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def matmul(A, B):
    return np.matmul(A, B)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        assert result.functions[0].algorithm == "matmul"


class TestPythonParserLinearAlgebra:
    def test_solve(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def solve_system(A, b):
    return np.linalg.solve(A, b)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        fn = result.functions[0]
        assert fn.algorithm == "linalg_solve"

    def test_cholesky(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def cholesky_decomp(A):
    return np.linalg.cholesky(A)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        assert result.functions[0].algorithm == "linalg_cholesky"

    def test_inv(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def invert_matrix(A):
    return np.linalg.inv(A)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        assert result.functions[0].algorithm == "linalg_inv"

    def test_eig(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def eigenvalues(A):
    return np.linalg.eig(A)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        assert result.functions[0].algorithm == "linalg_eig"


class TestPythonParserElementwise:
    def test_add(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "element_add"

    def test_subtract(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def sub_arrays(a, b):
    return np.subtract(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "element_sub"

    def test_multiply(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def mul_arrays(a, b):
    return np.multiply(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "element_mul"

    def test_divide(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def div_arrays(a, b):
    return np.divide(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "element_div"


class TestPythonParserReductions:
    def test_sum(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def total(x):
    return np.sum(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "reduce_sum"

    def test_mean(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def average(x):
    return np.mean(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "reduce_mean"

    def test_max(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def maximum(x):
    return np.max(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "reduce_max"

    def test_min(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def minimum(x):
    return np.min(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "reduce_min"


class TestPythonParserFFT:
    def test_fft(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def compute_fft(x):
    return np.fft.fft(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "fft"

    def test_ifft(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def compute_ifft(x):
    return np.fft.ifft(x)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "ifft"


class TestPythonParserAllocation:
    def test_zeros(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def make_zeros(n):
    return np.zeros(n)
"""
        result = parser.parse_source(source, "test.py")
        fn = result.functions[0]
        assert fn.algorithm == "alloc_zeros"

    def test_ones(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def make_ones(n):
    return np.ones(n)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "alloc_ones"

    def test_eye(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def make_identity(n):
    return np.eye(n)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].algorithm == "alloc_eye"


class TestPythonParserEdgeCases:
    def test_syntax_error(self, parser: PythonParser) -> None:
        result = parser.parse_source("def broken(:\n", "bad.py")
        assert len(result.functions) == 0
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "ERROR"

    def test_non_numpy_ignored(self, parser: PythonParser) -> None:
        source = """
def pure_python(x):
    return x + 1
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 0

    def test_multiple_functions(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)

def dot_product(a, b):
    return np.dot(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 2
        names = {fn.name for fn in result.functions}
        assert names == {"add_arrays", "dot_product"}

    def test_composition(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def composed(A, B, c):
    return np.add(np.matmul(A, B), c)
"""
        result = parser.parse_source(source, "test.py")
        assert len(result.functions) == 1
        fn = result.functions[0]
        assert len(fn.reductions) == 2

    def test_type_hint_float32(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def dot_f32(a: np.float32, b: np.float32) -> np.float32:
    return np.dot(a, b)
"""
        result = parser.parse_source(source, "test.py")
        assert result.functions[0].inputs[0][1].name == "FLOAT32"
        assert result.functions[0].outputs[0][1].name == "FLOAT32"

    def test_to_graph(self, parser: PythonParser) -> None:
        source = """
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)
"""
        result = parser.parse_source(source, "test.py")
        graph = result.to_graph()
        assert len(graph.nodes) == 1
        assert len(graph.entry_points) == 1

    def test_parse_file_missing(self, parser: PythonParser) -> None:
        result = parser.parse_file("nonexistent_file.py")
        assert len(result.functions) == 0
        assert result.diagnostics[0].severity == "ERROR"
