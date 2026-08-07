"""Edge coverage for purce.ir.builder — composed/decompose/loop paths.

Targets the decomposition machinery of MathIRBuilder: composed-function
argument wiring, expression decomposition fallbacks, guard-clause detection,
and loop-concatenation detection.
"""

import ast

from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import Dtype


def _build(source: str, module: str = "mod"):
    builder = MathIRBuilder(origin_file="test.py")
    graph = builder.build_from_source(source, module=module)
    return builder, graph


def _algos(graph):
    return [n.algorithm for n in graph.nodes.values()]


class TestComposedFunctionArgs:
    def test_list_arg_containing_unprocessed_calls(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    return np.divide(np.add(a, b), np.concatenate([np.multiply(a, c), b]))\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_mul" in algos
        assert "array_concat" in algos
        assert "element_div" in algos
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert concat.inputs[0][0].startswith("_inter_")

    def test_nested_list_element_extends_inputs(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    return np.divide(np.add(a, b), np.concatenate([[np.multiply(a, c), b]]))\n"
        )
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 2
        assert concat.inputs[0][0].startswith("_inter_")

    def test_non_list_arg_that_resolves_to_list_extends_inputs(self) -> None:
        _, graph = _build(
            "def f(a, b, c, d):\n"
            "    return np.divide(np.add(a, b), ([c, np.add(b, c)] | d))\n"
        )
        div = next(n for n in graph.nodes.values() if n.algorithm == "element_div")
        assert len(div.inputs) >= 3
        assert "element_add" in _algos(graph)


class TestResolveCallTarget:
    def test_flatten_and_squeeze_method_targets(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    u = np.multiply(a.flatten(), b)\n"
            "    v = np.multiply(c.squeeze(), b)\n"
            "    return np.subtract(u, v)\n"
        )
        assert "element_mul" in _algos(graph)
        assert "element_sub" in _algos(graph)


class TestCollectAllNumpyCalls:
    def test_return_tuple_calls_collected_but_not_emitted(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    s = np.add(a, b)\n"
            "    return (np.multiply(s, b), np.subtract(a, s))\n"
        )
        assert _algos(graph) == ["element_add"]


class TestDecomposeExpr:
    def test_name_in_intermediates_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("tmp").body[0].value
        result = builder._decompose_expr(
            expr,
            [],
            {},
            {"tmp": "inter0"},
            {},
            set(),
            {},
            "mod", "f", "file", [], [0],
        )
        assert result == ("inter0", Dtype.FLOAT64, "array")

    def test_name_builtin_type_becomes_zero_const(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    s = np.add(a, float)\n"
            "    t = np.multiply(s, b)\n"
            "    return t\n"
        )
        add = next(n for n in graph.nodes.values() if n.algorithm == "element_add")
        assert add.inputs[1][2] == "scalar"

    def test_list_containing_list_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, d):\n"
            "    s = np.concatenate([[a, b], [c, d]])\n"
            "    return s\n"
        )
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 4

    def test_np_prefixed_call_target_normalized(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    s = np.add(a, b)\n"
            "    t = np.multiply(s, b)\n"
            "    return t\n"
        )
        assert "element_add" in _algos(graph)

    def test_call_with_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    s = np.concatenate([[a, b], [c]])\n"
            "    return s\n"
        )
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 3

    def test_call_arg_binop_list_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, d):\n"
            "    s = np.divide([a, b] | c, d)\n"
            "    t = np.multiply(s, d)\n"
            "    return t\n"
        )
        div = next(n for n in graph.nodes.values() if n.algorithm == "element_div")
        assert len(div.inputs) >= 2

    def test_transpose_arg_uses_input_shape(self) -> None:
        _, graph = _build(
            "def f(a: float, b):\n"
            "    s = np.multiply(a.T, b)\n"
            "    t = np.add(s, b)\n"
            "    return t\n"
        )
        mul = next(n for n in graph.nodes.values() if n.algorithm == "element_mul")
        assert mul.inputs[0] == ("a", Dtype.FLOAT64, "scalar")

    def test_call_with_nested_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, e):\n"
            "    s = np.divide(np.concatenate([[[a, b], c]]), e)\n"
            "    t = np.multiply(s, e)\n"
            "    return t\n"
        )
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 3
        assert any(n.algorithm == "element_div" for n in graph.nodes.values())

    def test_call_with_list_and_plain_arg_append(self) -> None:
        _, graph = _build(
            "def f(a, b, c, e):\n"
            "    s = np.divide(np.concatenate([[a, b], c]), e)\n"
            "    t = np.multiply(s, e)\n"
            "    return t\n"
        )
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 3

    def test_call_arg_binop_list_extends_decomposed(self) -> None:
        _, graph = _build(
            "def f(a, b, c, e):\n"
            "    s = np.divide(np.concatenate([a, b] | c), e)\n"
            "    t = np.multiply(s, e)\n"
            "    return t\n"
        )
        assert any(n.algorithm == "element_div" for n in graph.nodes.values())

    def test_transpose_of_call_creates_intermediate(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("np.sqrt(x).T").body[0].value
        result = builder._decompose_expr(
            expr,
            [],
            {},
            {"x": "inter_x"},
            {},
            set(),
            {},
            "mod", "f", "file", [], [0],
        )
        assert result[0].startswith("_inter_expr_")

    def test_np_e_attribute(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    s = np.multiply(np.e, b)\n"
            "    t = np.add(s, a)\n"
            "    return t\n"
        )
        mul = next(n for n in graph.nodes.values() if n.algorithm == "element_mul")
        assert mul.inputs[0][2] == "scalar"

    def test_np_inf_attribute(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    s = np.divide(a, np.inf)\n"
            "    t = np.multiply(s, b)\n"
            "    return t\n"
        )
        div = next(n for n in graph.nodes.values() if n.algorithm == "element_div")
        assert div.inputs[1][2] == "scalar"

    def test_subscript_on_intermediate_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x[i]").body[0].value
        result = builder._decompose_expr(
            expr,
            [],
            {},
            {"x": "inter_x"},
            {"y": ("y", Dtype.FLOAT64)},
            set(),
            {},
            "mod", "f", "file", [], [0],
        )
        assert result[0] == "inter_x"


class TestGuardClauses:
    def test_lt_guard_flattened_in_full_body(self) -> None:
        _, graph = _build(
            "def f(a, b, n):\n"
            "    s = np.add(a, b)\n"
            "    if n < 5:\n"
            "        return np.multiply(s, b)\n"
            "    return np.subtract(a, s)\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_mul" in algos

    def test_name_test_guard_flattened_in_full_body(self) -> None:
        _, graph = _build(
            "def f(a, b, flag):\n"
            "    s = np.add(a, b)\n"
            "    if flag:\n"
            "        return np.multiply(s, b)\n"
            "    return np.subtract(a, s)\n"
        )
        assert "element_add" in _algos(graph)
        assert "element_mul" in _algos(graph)


class TestDetectLoopConcat:
    def test_list_var_never_appended(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    scratch = []\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    return np.concatenate(out)\n"
        )
        assert "loop_concat" in _algos(graph)

    def test_np_prefixed_concatenate_return(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    return np.concatenate(out)\n"
        )
        assert "loop_concat" in _algos(graph)

    def test_return_tuple_with_concatenate_binding(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    y = np.concatenate(out)\n"
            "    return (y, out)\n"
        )
        assert "loop_concat" in _algos(graph)

    def test_assign_binding_concatenate(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    y = np.concatenate(out)\n"
            "    return y\n"
        )
        assert "loop_concat" in _algos(graph)

    def test_return_tuple_before_concatenate_binding(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    return (y, out)\n"
            "    y = np.concatenate(out)\n"
        )
        assert "loop_concat" in _algos(graph)

    def test_no_loop_pattern_returns_none(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    return np.subtract(s, w)\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_sub" in algos


class TestDecomposeFullBodyEmptyOps:
    def test_no_collectable_operations_returns_early(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    s = np.add(a, b) + c\n"
            "    t = np.multiply(a, b) + c\n"
            "    return s\n"
        )
        assert len(graph.nodes) == 0
