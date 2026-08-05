import json
import os
import re

from purce.backend.c99_generator import C99Generator
from purce.ir.nodes import (
    Dtype,
    Effect,
    MathIRGraph,
    MathIRNode,
    ReductionEntry,
)


def _make_matmul_node() -> MathIRNode:
    return MathIRNode(
        node_id="test.mod_matmul_a1b2c3d4",
        origin_symbol="mod.matmul_fn",
        origin_file="test.py",
        origin_line=5,
        origin_commit="abc123",
        origin_signature="double A -> double B -> double",
        math_intent="Matrix multiplication kernel",
        inputs=[("A", Dtype.FLOAT64, "(m,k)"), ("B", Dtype.FLOAT64, "(k,n)")],
        outputs=[("C", Dtype.FLOAT64, "(m,n)")],
        effects=[Effect.PURE],
        algorithm="matmul",
        reductions=[
            ReductionEntry(rule="numpy_op_extraction", description="Extracted numpy.matmul as matmul kernel", original="numpy.matmul"),
        ],
        stack_usage=512,
        heap_usage=None,
        reentrant=True,
    )


def _make_element_add_node() -> MathIRNode:
    return MathIRNode(
        node_id="test.mod_add_e5f6g7h8",
        origin_symbol="mod.add_fn",
        origin_file="test.py",
        origin_line=10,
        origin_commit=None,
        origin_signature="double a -> double b -> double",
        math_intent="Element-wise addition kernel",
        inputs=[("a", Dtype.FLOAT64, "array"), ("b", Dtype.FLOAT64, "array")],
        outputs=[("c", Dtype.FLOAT64, "array")],
        effects=[Effect.PURE],
        algorithm="element_add",
        reductions=[],
        stack_usage=64,
    )


def _make_reduce_sum_node() -> MathIRNode:
    return MathIRNode(
        node_id="test.mod_sum_i9j0k1l2",
        origin_symbol="mod.sum_fn",
        origin_file="test.py",
        origin_line=15,
        origin_commit=None,
        origin_signature="double x -> double",
        math_intent="Reduction sum kernel",
        inputs=[("x", Dtype.FLOAT64, "array")],
        outputs=[("result", Dtype.FLOAT64, "scalar")],
        effects=[Effect.PURE],
        algorithm="reduce_sum",
        reductions=[],
        stack_usage=32,
    )


def _make_graph(*nodes: MathIRNode) -> MathIRGraph:
    g = MathIRGraph()
    for n in nodes:
        g.add_node(n)
        g.entry_points.append(n.node_id)
    return g


class TestC99GeneratorHeaders:
    def test_file_header_present(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        assert len(c_files) == 1
        assert "PURCE OUTPUT" in c_files[0].content
        assert "mymod" in c_files[0].content
        assert "matmul" in c_files[0].content.lower()

    def test_function_header_present(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        content = c_files[0].content
        assert "SEMANTIC UNIT:" in content
        assert "ORIGIN SYMBOL:" in content
        assert "MATH INTENT:" in content
        assert "REDUCTION LOG:" in content
        assert "MEMORY CONTRACT:" in content
        assert "CORRECTNESS:" in content


class TestC99GeneratorOutput:
    def test_header_file(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        h_files = [f for f in result.files if f.file_type == "h"]
        assert len(h_files) == 1
        assert "MYMOD_H" in h_files[0].content
        assert "#include <stdint.h>" in h_files[0].content
        assert "#endif" in h_files[0].content

    def test_multiple_c_files(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node(), _make_element_add_node())
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        assert len(c_files) == 2

    def test_provenance_files(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        prov_files = [f for f in result.files if f.file_type == "prov"]
        assert len(prov_files) == 1
        prov = json.loads(prov_files[0].content)
        assert prov["purce_version"] is not None
        assert prov["source"]["symbol"] == "mod.matmul_fn"
        assert prov["ir_node"]["algorithm"] == "matmul"

    def test_cmake_file(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        cmake_files = [f for f in result.files if f.file_type == "cmake"]
        assert len(cmake_files) == 1
        assert "cmake_minimum_required" in cmake_files[0].content
        assert "CMAKE_C_STANDARD 99" in cmake_files[0].content


class TestC99GeneratorCodeQuality:
    def test_no_malloc(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        for f in c_files:
            assert "malloc" not in f.content
            assert "free" not in f.content

    def test_c99_standard(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_element_add_node())
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        for f in c_files:
            assert "#include <stdint.h>" in f.content
            assert "#include <math.h>" in f.content

    def test_naming_convention(self) -> None:
        gen = C99Generator()
        node = _make_matmul_node()
        graph = _make_graph(node)
        result = gen.generate(graph, "mymod")

        c_files = [f for f in result.files if f.file_type == "c"]
        assert "test_mod_matmul_a1b2c3d4" in c_files[0].content

    def test_provenance_json_valid(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node(), _make_element_add_node())
        result = gen.generate(graph, "mymod")

        prov_files = [f for f in result.files if f.file_type == "prov"]
        for pf in prov_files:
            prov = json.loads(pf.content)
            assert "source" in prov
            assert "ir_node" in prov
            assert "memory" in prov
            assert "reductions" in prov


class TestC99GeneratorFixedPoint:
    def test_q31_types_in_header(self) -> None:
        gen = C99Generator(fixed_point=True)
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        h_files = [f for f in result.files if f.file_type == "h"]
        assert "q31_t" in h_files[0].content
        assert "q31_mul" in h_files[0].content
        assert "Q31_ONE" in h_files[0].content


class TestC99GeneratorAlgorithms:
    def test_matmul_body(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")
        c_files = [f for f in result.files if f.file_type == "c"]
        content = c_files[0].content
        assert "Matrix multiplication" in content
        assert re.search(r'\bfor\s*\(', content), "matmul should contain for loops"
        assert re.search(r'\bvoid\s+\w+\s*\(', content), "matmul should have a function signature"

    def test_element_add_body(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_element_add_node())
        result = gen.generate(graph, "mymod")
        c_files = [f for f in result.files if f.file_type == "c"]
        content = c_files[0].content
        assert "Element-wise addition" in content
        assert re.search(r'\bfor\s*\(', content), "element_add should contain a for loop"
        assert re.search(r'\bvoid\s+\w+\s*\(', content), "element_add should have a function signature"

    def test_reduce_sum_body(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_reduce_sum_node())
        result = gen.generate(graph, "mymod")
        c_files = [f for f in result.files if f.file_type == "c"]
        content = c_files[0].content
        assert "Reduction sum" in content
        assert re.search(r'\bfor\s*\(', content), "reduce_sum should contain a for loop"
        assert re.search(r'\bvoid\s+\w+\s*\(', content), "reduce_sum should have a function signature"


class TestC99GeneratorScalarParams:
    """Regression: scalar/None-shaped inputs must be emitted as scalar params
    and de-subscripted in the body, while tuple-shaped inputs stay pointers."""

    @staticmethod
    def _mul_node(b_shape):
        return MathIRNode(
            node_id="t.mod_mul_x9y8z7w6",
            origin_symbol="mod.mul_fn",
            origin_file="t.py",
            origin_line=3,
            origin_commit=None,
            origin_signature="double A -> double B -> double",
            math_intent="Element-wise multiplication kernel",
            inputs=[("A", Dtype.FLOAT64, "array"), ("B", Dtype.FLOAT64, b_shape)],
            outputs=[("C", Dtype.FLOAT64, "array")],
            effects=[Effect.PURE],
            algorithm="element_mul",
            reductions=[],
            stack_usage=64,
        )

    def _render(self, node) -> str:
        gen = C99Generator()
        graph = _make_graph(node)
        result = gen.generate(graph, "mymod")
        c_files = [f for f in result.files if f.file_type == "c"]
        return c_files[0].content

    def test_none_shape_input_is_scalar_param_and_not_subscripted(self) -> None:
        content = self._render(self._mul_node(None))
        assert re.search(r'\bvoid\s+\w+\(int\s+n,\s*const\s+double\s*\*\s*restrict\s+A,\s*double\s+B,\s*double\s*\*\s*restrict\s+C\)', content), content
        assert re.search(r'C\[i\] = A\[i\] \* B;', content), content
        assert "B[i]" not in content

    def test_tuple_shape_input_is_pointer_and_stays_subscripted(self) -> None:
        content = self._render(self._mul_node("(m,n)"))
        assert re.search(r'\bconst\s+double\s*\*\s*restrict\s+B\b', content), content
        assert "B[i]" in content

    def test_scalar_shape_input_is_scalar_param(self) -> None:
        content = self._render(self._mul_node("scalar"))
        assert re.search(r'\bdouble\s+B\b', content), content
        assert "B[i]" not in content


class TestC99GeneratorWriteAll:
    def test_write_files(self, tmp_path: object) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_matmul_node())
        result = gen.generate(graph, "mymod")

        out_dir = str(tmp_path) + "/out"
        result.write_all(out_dir)

        c_files = [f for f in result.files if f.file_type == "c"]
        h_files = [f for f in result.files if f.file_type == "h"]
        prov_files = [f for f in result.files if f.file_type == "prov"]

        for f in c_files:
            assert os.path.exists(os.path.join(out_dir, f.path))
        for f in h_files:
            assert os.path.exists(os.path.join(out_dir, f.path))
        for f in prov_files:
            assert os.path.exists(os.path.join(out_dir, f.path))


class TestC99GeneratorEdgeCases:
    def test_empty_graph(self) -> None:
        gen = C99Generator()
        graph = MathIRGraph()
        result = gen.generate(graph, "empty")
        assert len(result.files) == 2  # header + cmake

    def test_single_node(self) -> None:
        gen = C99Generator()
        graph = _make_graph(_make_element_add_node())
        result = gen.generate(graph, "single")
        c_files = [f for f in result.files if f.file_type == "c"]
        assert len(c_files) == 1

    def test_topological_order_in_output(self) -> None:
        gen = C99Generator()
        n1 = MathIRNode(
            node_id="a", origin_symbol="mod.a", origin_file="t.py",
            origin_line=1, origin_commit=None, origin_signature="double -> double",
            math_intent="a", inputs=[("x", Dtype.FLOAT64, "scalar")],
            outputs=[("r", Dtype.FLOAT64, "scalar")], effects=[Effect.PURE],
            algorithm="element_add", stack_usage=32,
        )
        n2 = MathIRNode(
            node_id="b", origin_symbol="mod.b", origin_file="t.py",
            origin_line=5, origin_commit=None, origin_signature="double -> double",
            math_intent="b", inputs=[("x", Dtype.FLOAT64, "scalar")],
            outputs=[("r", Dtype.FLOAT64, "scalar")], effects=[Effect.PURE],
            algorithm="reduce_sum", stack_usage=32, nested_deps=["a"],
        )
        g = MathIRGraph()
        g.add_node(n1)
        g.add_node(n2)
        g.entry_points = ["b"]

        result = gen.generate(g, "order_test")
        c_files = [f for f in result.files if f.file_type == "c"]
        a_idx = next(i for i, f in enumerate(c_files) if "a" in f.path)
        b_idx = next(i for i, f in enumerate(c_files) if "b" in f.path)
        assert a_idx < b_idx
