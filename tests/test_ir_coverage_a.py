"""Coverage-focused tests for purce.ir.builder edges (slice A).

Exercises dtype annotation resolution, qualified-name extraction, IO effect
tagging, nested numpy call detection through local functions, and the
_resolve_arg_to_name branches that plain single-op sources never reach.
"""

import ast
import math

from purce.ir.builder import MathIRBuilder, _get_qualified_name
from purce.ir.nodes import Dtype, Effect

try:
    from conftest import run_pipeline
except ImportError:  # pragma: no cover
    from tests.conftest import run_pipeline


def _local_funcs(source: str) -> dict[str, ast.FunctionDef]:
    mod = ast.parse(source)
    return {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef)}


def _build(source: str, module: str = "covmod") -> MathIRBuilder:
    bld = MathIRBuilder(origin_file="covtest.py")
    bld.build_from_source(source, module=module)
    return bld


class TestDtypeAnnotations:
    def test_name_bool_annotation_returns_bool(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def bf(x: bool):\n"
            "    return np.sin(x)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert node.input_dtypes() == [Dtype.BOOL]

    def test_name_complex_annotation_returns_complex128(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def cf(z: complex):\n"
            "    return np.sqrt(z)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert node.input_dtypes() == [Dtype.COMPLEX128]

    def test_attribute_dtype_annotation_uses_dtype_map(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def af(x: np.float32):\n"
            "    return np.abs(x)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert node.input_dtypes() == [Dtype.FLOAT32]

    def test_unknown_annotations_fall_back_to_float64(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def uf(x: MyType, y: np.notadtype):\n"
            "    return np.add(x, y)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert node.input_dtypes() == [Dtype.FLOAT64, Dtype.FLOAT64]


class TestQualifiedName:
    def test_attr_attribute_and_name_and_fallback(self) -> None:
        assert _get_qualified_name(_from_attr("np.dot")) == "np.dot"
        assert _get_qualified_name(_from_name("x")) == "x"
        assert _get_qualified_name(ast.parse("3 + 4").body[0].value) == ""

    def test_attr_chain_joins_with_dots(self) -> None:
        node = _from_attr("np.linalg.solve")
        assert _get_qualified_name(node) == "np.linalg.solve"


class TestNestedLocalDetection:
    """Exercises _has_nested_numpy_calls local-function walk passes."""

    def test_recursion_pass_finds_direct_numpy_arg(self) -> None:
        bld = MathIRBuilder()
        local_funcs = _local_funcs(
            "def helper(x):\n"
            "    return np.add(np.exp(x), x)\n"
        )
        call = _from_attr("np.subtract(a, helper(b))")
        assert bld._has_nested_numpy_calls(call, local_funcs) is True

    def test_second_pass_attribute_inner_arg(self) -> None:
        bld = MathIRBuilder()
        local_funcs = _local_funcs(
            "def helper(x):\n"
            "    return np.add(x, np.unknown_thing(x))\n"
        )
        call = _from_attr("np.multiply(a, helper(b))")
        assert bld._has_nested_numpy_calls(call, local_funcs) is False

    def test_second_pass_name_inner_arg(self) -> None:
        bld = MathIRBuilder()
        local_funcs = _local_funcs(
            "def helper(x):\n"
            "    return np.add(x, unknown_name(y))\n"
        )
        call = _from_attr("np.multiply(a, helper(b))")
        assert bld._has_nested_numpy_calls(call, local_funcs) is False


class TestArgResolution:
    def _resolve(self, node: ast.expr, inputs=None, existing=None):
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        inputs = inputs or []
        existing = existing or set()
        return bld._resolve_arg_to_name(node, inputs, scalar, {}, set(existing))

    def test_name_in_existing_found_in_inputs(self) -> None:
        result = self._resolve(_from_name("x"), [("x", Dtype.FLOAT64, "array")], {"x"})
        assert result == ("x", Dtype.FLOAT64, "array")

    def test_name_in_existing_missing_from_inputs(self) -> None:
        result = self._resolve(_from_name("y"), [("x", Dtype.FLOAT64, "array")], {"y"})
        assert result[0] == "y"
        assert result[1] == Dtype.FLOAT64

    def test_float_constant_creates_scalar_input(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        existing = {"x"}
        result = bld._resolve_arg_to_name(ast.Constant(2.5), [("x", Dtype.FLOAT64, "array")], scalar, {}, existing)
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar["_const_0"] == 2.5
        assert "_const_0" in existing

    def test_constant_already_registered(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        existing = {"_const_0"}
        result = bld._resolve_arg_to_name(ast.Constant(2.5), [], scalar, {}, existing)
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert "_const_0" not in scalar

    def test_complex_constant_uses_real_part(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(ast.Constant(3 + 2j, kind=None), [], scalar, {}, set())
        assert result[0] == "_const_0"
        assert result[2] == "scalar"
        assert scalar["_const_0"] == 3.0

    def test_transpose_on_registered_input(self) -> None:
        result = self._resolve(_from_attr("a.T"),[("a", Dtype.FLOAT64, "array")], {"a"})
        assert result == ("a", Dtype.FLOAT64, "array")

    def test_transpose_on_intermediate(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(
            _from_attr("b.T"), [], scalar, {"b": "inter_0"}, set()
        )
        assert result == ("inter_0", Dtype.FLOAT64, "array")

    def test_np_pi_attribute_becomes_const(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        existing = {"a"}
        result = bld._resolve_arg_to_name(_from_attr("np.pi"), [], scalar, {}, existing)
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert math.isclose(scalar["_const_0"], math.pi)

    def test_np_e_attribute_becomes_const(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        bld._resolve_arg_to_name(_from_attr("np.e"), [], scalar, {}, set())
        assert scalar["_const_0"] == math.e

    def test_numpy_dtype_attribute_becomes_zero_const(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(_from_attr("np.float32"), [], scalar, {}, set())
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar["_const_0"] == 0.0

    def test_call_to_attr_numpy_target(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(
            _from_attr("np.dot(A, B)"), [], scalar, {"numpy.dot": "dot_0"}, set()
        )
        assert result == ("dot_0", Dtype.FLOAT64, "array")

    def test_call_to_name_target_intermediate(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(
            _from_call("helper(x)"), [], scalar, {"helper": "h_0"}, set()
        )
        assert result == ("h_0", Dtype.FLOAT64, "array")

    def test_unary_neg_constant(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(_unary_neg("2.5"), [], scalar, {}, set())
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar["_const_0"] == -2.5

    def test_unary_neg_intermediate_uses_minus_one(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(
            _unary_neg("b"), [], scalar, {"b": "b_0"}, set()
        )
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar["_const_0"] == -1.0

    def test_subscript_on_registered_name(self) -> None:
        result = self._resolve(_from_attr("a[0]"), [("a", Dtype.FLOAT64, "array")], {"a"})
        assert result == ("a", Dtype.FLOAT64, "array")

    def test_builtin_type_name_becomes_zero_const(self) -> None:
        bld = MathIRBuilder()
        scalar: dict[str, float] = {}
        result = bld._resolve_arg_to_name(_from_name("float"), [], scalar, {}, {"x"})
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar["_const_0"] == 0.0

    def test_name_matched_in_func_inputs(self) -> None:
        result = self._resolve(_from_name("b"), [("b", Dtype.FLOAT64, "array")], {"x"})
        assert result == ("b", Dtype.FLOAT64, "array")

    def test_unresolvable_name_returns_flag(self) -> None:
        result = self._resolve(_from_name("ghost"), [("b", Dtype.FLOAT64, "array")], {"x"})
        assert result[0].startswith("_unresolved_")
        assert result[1] == Dtype.FLOAT64
        assert result[2] is None


class TestEndpointBehavior:
    def test_print_marks_io_effect(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def noisy(x, y):\n"
            "    print(x)\n"
            "    return np.add(x, y)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert Effect.IO in node.effects
        assert node.is_pure() is False

    def test_local_helper_call_inlined_into_expression_return(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def plain(a, b):\n"
            "    return a + b\n"
            "def outer(a, b):\n"
            "    return plain(a, np.sin(b))\n"
        )
        nodes = list(builder.graph.nodes.values())
        assert len(nodes) == 2
        sin_nodes = [n for n in nodes if n.algorithm == "element_sin"]
        add_nodes = [n for n in nodes if n.algorithm == "element_add"]
        assert len(sin_nodes) == 1
        assert len(add_nodes) == 1
        assert [i[0] for i in sin_nodes[0].inputs] == ["b"]
        assert add_nodes[0].outputs == [("result", Dtype.FLOAT64, "array")]
        assert sorted(n.algorithm for n in nodes) == ["element_add", "element_sin"]
        assert builder.graph.entry_points == [add_nodes[0].node_id]

    def test_composed_multi_op_graph(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def comp(a, b):\n"
            "    return np.subtract(np.multiply(a, b), 3.5)\n"
        )
        nodes = list(builder.graph.nodes.values())
        assert len(nodes) == 2
        algos = [n.algorithm for n in nodes]
        assert "element_mul" in algos
        assert "element_sub" in algos

    def test_composed_method_call_receiver_is_kernel_input(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def comp(a, b):\n"
            "    return np.multiply(a, b.transpose())\n"
        )
        nodes = list(builder.graph.nodes.values())
        assert len(nodes) == 2
        by_algo = {n.algorithm for n in nodes}
        assert by_algo == {"transpose", "element_mul"}
        transpose_node = next(n for n in nodes if n.algorithm == "transpose")
        assert [i[0] for i in transpose_node.inputs] == ["b"]
        mul_node = next(n for n in nodes if n.algorithm == "element_mul")
        assert mul_node.inputs[0][0] == "a"
        assert transpose_node.node_id in mul_node.nested_deps

    def test_unary_neg_assign_in_full_body_binds_operand(self) -> None:
        builder = _build(
            "import numpy as np\n"
            "def f(a, b, c):\n"
            "    s = np.sin(b)\n"
            "    r = -a\n"
            "    return np.subtract(s, r)\n"
        )
        nodes = list(builder.graph.nodes.values())
        assert len(nodes) == 3
        by_algo = {n.algorithm for n in nodes}
        assert by_algo == {"element_mul", "element_sub", "element_sin"}
        neg_node = next(n for n in nodes if n.algorithm == "element_mul")
        assert neg_node.inputs[1][0] == "a"
        sub_node = next(n for n in nodes if n.algorithm == "element_sub")
        assert neg_node.node_id in sub_node.nested_deps


class TestBuilderFixture:
    def test_builder_fixture_builds_node(self, builder) -> None:
        builder.build_from_source(
            "import numpy as np\n"
            "def dot(a, b):\n"
            "    return np.dot(a, b)\n"
        )
        node = next(iter(builder.graph.nodes.values()))
        assert node.algorithm == "matmul"

    def test_pipeline_end_to_end(self) -> None:
        gen_result, graph, _slice = run_pipeline(
            "import numpy as np\n"
            "def prod(a, b):\n"
            "    return np.matmul(a, b)\n"
        )
        assert graph.nodes
        c_files = [f for f in gen_result.files if f.file_type == "c"]
        assert "matmul" in c_files[0].content.lower()


def _from_name(src: str) -> ast.Name:
    node = _from_attr(src)
    if isinstance(node, ast.Name):
        return node
    raise AssertionError(f"expected Name, got {type(node)}")


def _from_attr(src: str) -> ast.expr:
    return ast.parse(src).body[0].value


def _from_call(src: str) -> ast.Call:
    node = _from_attr(src)
    if isinstance(node, ast.Call):
        return node
    raise AssertionError(f"expected Call, got {type(node)}")


def _unary_neg(operand: str) -> ast.UnaryOp:
    node = ast.parse(f"-{operand}").body[0].value
    assert isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    return node