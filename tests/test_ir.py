from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import (
    Dtype,
    Effect,
    MathIRGraph,
    MathIRNode,
    ReductionEntry,
)


def _make_node(node_id: str, deps: list[str] | None = None) -> MathIRNode:
    return MathIRNode(
        node_id=node_id,
        origin_symbol=f"mod.{node_id}",
        origin_file="test.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="float64 x -> float64",
        math_intent="test kernel",
        inputs=[("x", Dtype.FLOAT64, "scalar")],
        outputs=[("result", Dtype.FLOAT64, "scalar")],
        effects=[Effect.PURE],
        algorithm="test",
        nested_deps=deps or [],
        stack_usage=64,
    )


class TestMathIRNode:
    def test_create_node(self) -> None:
        node = _make_node("a")
        assert node.node_id == "a"
        assert node.is_pure() is True
        assert node.input_dtypes() == [Dtype.FLOAT64]
        assert node.output_dtypes() == [Dtype.FLOAT64]

    def test_node_with_reductions(self) -> None:
        node = _make_node("b")
        node.reductions.append(
            ReductionEntry(
                rule="inline",
                description="inlined helper",
                original="helper_fn",
            )
        )
        assert len(node.reductions) == 1
        assert node.reductions[0].rule == "inline"

    def test_node_not_pure_with_io(self) -> None:
        node = _make_node("c")
        node.effects = [Effect.PURE, Effect.IO]
        assert node.is_pure() is False

    def test_node_heap_usage(self) -> None:
        node = _make_node("d")
        assert node.heap_usage is None
        node.heap_usage = 1024
        assert node.heap_usage == 1024


class TestMathIRGraph:
    def test_add_and_get(self) -> None:
        g = MathIRGraph()
        node = _make_node("x")
        g.add_node(node)
        assert g.get_node("x") is node
        assert g.get_node("missing") is None

    def test_remove_node(self) -> None:
        g = MathIRGraph()
        n1 = _make_node("a")
        n2 = _make_node("b", deps=["a"])
        g.add_node(n1)
        g.add_node(n2)
        g.remove_node("a")
        assert g.get_node("a") is None
        assert "a" not in g.get_node("b").nested_deps

    def test_topological_sort_linear(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("c", deps=["b"]))
        g.add_node(_make_node("b", deps=["a"]))
        g.add_node(_make_node("a"))
        g.entry_points = ["c"]

        order = g.topological_sort()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_topological_sort_diamond(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("d", deps=["b", "c"]))
        g.add_node(_make_node("b", deps=["a"]))
        g.add_node(_make_node("c", deps=["a"]))
        g.add_node(_make_node("a"))
        g.entry_points = ["d"]

        order = g.topological_sort()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_reachable_from(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b", deps=["a"]))
        g.add_node(_make_node("c"))
        g.add_node(_make_node("d", deps=["c"]))

        sub = g.reachable_from(["b"])
        assert "a" in sub.nodes
        assert "b" in sub.nodes
        assert "c" not in sub.nodes
        assert "d" not in sub.nodes

    def test_validate_dag_no_errors(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b", deps=["a"]))
        errors = g.validate_dag()
        assert errors == []

    def test_validate_dag_missing_dep(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a", deps=["ghost"]))
        errors = g.validate_dag()
        assert len(errors) == 1
        assert "ghost" in errors[0]

    def test_validate_dag_cycle(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a", deps=["b"]))
        g.add_node(_make_node("b", deps=["a"]))
        errors = g.validate_dag()
        assert any("Cycle" in e for e in errors)

    def test_empty_graph(self) -> None:
        g = MathIRGraph()
        order = g.topological_sort()
        assert order == []


class TestMathIRBuilder:
    def test_build_from_dot_product(self) -> None:
        source = """
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")

        assert len(graph.nodes) == 1
        node = next(iter(graph.nodes.values()))
        assert node.algorithm == "matmul"
        assert len(node.inputs) == 2
        assert builder.diagnostics == []

    def test_build_with_linalg_solve(self) -> None:
        source = """
import numpy as np

def solve_system(A, b):
    return np.linalg.solve(A, b)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")

        assert len(graph.nodes) == 1
        node = next(iter(graph.nodes.values()))
        assert node.algorithm == "linalg_solve"

    def test_build_fft(self) -> None:
        source = """
import numpy as np

def compute_fft(x):
    return np.fft.fft(x)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")

        assert len(graph.nodes) == 1
        node = next(iter(graph.nodes.values()))
        assert node.algorithm == "fft"

    def test_build_elementwise(self) -> None:
        source = """
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")

        assert len(graph.nodes) == 1
        node = next(iter(graph.nodes.values()))
        assert node.algorithm == "element_add"

    def test_build_multiple_functions(self) -> None:
        source = """
import numpy as np

def dot_product(a, b):
    return np.dot(a, b)

def add_arrays(a, b):
    return np.add(a, b)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")

        assert len(graph.nodes) == 2

    def test_skip_non_numpy(self) -> None:
        source = """
def pure_python(x):
    return x + 1
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")
        assert len(graph.nodes) == 0

    def test_syntax_error(self) -> None:
        source = "def broken(:\n"
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source)
        assert len(graph.nodes) == 0
        assert any(d["severity"] == "ERROR" for d in builder.diagnostics)

    def test_link_deps(self) -> None:
        g = MathIRGraph()
        n1 = _make_node("a")
        n2 = _make_node("b", deps=["a"])
        g.add_node(n1)
        g.add_node(n2)

        builder = MathIRBuilder()
        result = builder.link_deps(g)
        assert result.get_node("b").nested_deps == ["a"]

    def test_link_deps_missing(self) -> None:
        g = MathIRGraph()
        n1 = _make_node("a", deps=["ghost"])
        g.add_node(n1)

        builder = MathIRBuilder()
        builder.link_deps(g)
        assert len(builder.diagnostics) == 1
        assert "ghost" in builder.diagnostics[0]["reason"]

    def test_cpp_ast_stub(self) -> None:
        builder = MathIRBuilder()
        graph = builder.from_cpp_ast(None)
        assert len(graph.nodes) == 0
        assert any("C++ parser" in d["reason"] for d in builder.diagnostics)

    def test_complex_composition(self) -> None:
        source = """
import numpy as np

def matmul_then_add(A, B, c):
    return np.add(np.matmul(A, B), c)
"""
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.build_from_source(source, module="testmod")
        nodes = list(graph.nodes.values())
        assert len(nodes) == 2
        algos = [n.algorithm for n in nodes]
        assert "matmul" in algos
        assert "element_add" in algos
