"""Edge coverage for purce.ir.builder — full-body/loop/if-else resolution.

Targets the remaining decomposition branches of MathIRBuilder: loop-body
operation wiring, if/else flattening and conditional-select emission, shape
unpacking, duplicate-call deduplication, return-statement search, and the
full-body argument resolver fallbacks.
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


class TestLoopConcatBody:
    def test_loop_var_as_operand_input(self) -> None:
        _, graph = _build(
            "def f(x, n, w):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(s, i))\n"
            "    return np.concatenate(out)\n"
        )
        mul = next(n for n in graph.nodes.values() if n.algorithm == "element_mul")
        assert ("i", Dtype.INT64, "loop_var") in mul.inputs

    def test_loop_body_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(x, n, w, a, b):\n"
            "    s = np.add(x, w)\n"
            "    out = []\n"
            "    for i in range(n):\n"
            "        out.append(np.multiply(x, [a, b]))\n"
            "    return np.concatenate(out)\n"
        )
        mul = next(n for n in graph.nodes.values() if n.algorithm == "element_mul")
        assert len(mul.inputs) == 3


class TestIfElseFlattening:
    def test_if_else_different_vars_flattened(self) -> None:
        _, graph = _build(
            "def f(a, b, flag):\n"
            "    s = np.add(a, b)\n"
            "    if flag:\n"
            "        u = np.multiply(a, b)\n"
            "    else:\n"
            "        v = np.subtract(a, s)\n"
            "    return np.add(s, v)\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_mul" in algos
        assert "element_sub" in algos

    def test_same_var_if_else_emits_where(self) -> None:
        _, graph = _build(
            "def f(a, b, flag):\n"
            "    s = np.add(a, b)\n"
            "    if flag:\n"
            "        s = np.multiply(a, s)\n"
            "    else:\n"
            "        s = np.subtract(a, b)\n"
            "    return s\n"
        )
        algos = _algos(graph)
        assert "element_where" in algos
        assert "element_mul" in algos
        assert "element_sub" in algos
        where = next(n for n in graph.nodes.values() if n.algorithm == "element_where")
        assert len(where.inputs) == 3

    def test_if_branch_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, flag):\n"
            "    if flag:\n"
            "        s = np.concatenate([[np.multiply(a, c), b]])\n"
            "    else:\n"
            "        s = np.subtract(a, b)\n"
            "    return s\n"
        )
        where = next(n for n in graph.nodes.values() if n.algorithm == "element_where")
        assert len(where.inputs) == 3

    def test_if_branch_non_numpy_value(self) -> None:
        _, graph = _build(
            "def f(a, b, c, flag):\n"
            "    t = np.add(a, c)\n"
            "    if flag:\n"
            "        s = a\n"
            "    else:\n"
            "        s = np.multiply(a, b)\n"
            "    return s\n"
        )
        where = next(n for n in graph.nodes.values() if n.algorithm == "element_where")
        assert where.inputs[1][0] == "a"

    def test_else_branch_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, flag):\n"
            "    t = np.add(a, c)\n"
            "    if flag:\n"
            "        s = np.multiply(a, b)\n"
            "    else:\n"
            "        s = np.concatenate([[np.subtract(a, b), c]])\n"
            "    return s\n"
        )
        where = next(n for n in graph.nodes.values() if n.algorithm == "element_where")
        assert len(where.inputs) == 3


class TestShapeUnpacking:
    def test_shape_single_target(self) -> None:
        _, graph = _build(
            "def f(a, b):\n"
            "    m = a.shape\n"
            "    s = np.multiply(a, b)\n"
            "    t = np.add(s, m)\n"
            "    return t\n"
        )
        algos = _algos(graph)
        assert "element_mul" in algos
        assert "element_add" in algos
        add = next(n for n in graph.nodes.values() if n.algorithm == "element_add")
        assert add.inputs[1][2] == "scalar"


class TestProcessedCallDedup:
    def test_duplicate_call_node_processed_once(self) -> None:
        tree = ast.parse(
            "def f(a, b):\n"
            "    s1 = np.add(a, b)\n"
            "    s2 = np.add(a, b)\n"
            "    return np.subtract(s1, s2)\n"
        )
        func = tree.body[0]
        shared_call = func.body[0].value
        func.body[1].value = shared_call
        builder = MathIRBuilder(origin_file="test.py")
        graph = builder.from_python_ast(tree, module_name="mod")
        add_count = sum(1 for n in graph.nodes.values() if n.algorithm == "element_add")
        assert add_count == 1
        assert "element_sub" in _algos(graph)


class TestReturnSearch:
    def test_return_inside_if_orelse(self) -> None:
        _, graph = _build(
            "def f(a, b, flag):\n"
            "    s = np.add(a, b)\n"
            "    if flag:\n"
            "        pass\n"
            "    else:\n"
            "        return np.multiply(s, b)\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_mul" in algos

    def test_return_list_args_emit_nodes(self) -> None:
        _, graph = _build(
            "def f(a, b, c):\n"
            "    s = np.add(a, b)\n"
            "    return np.concatenate([[np.multiply(a, c), b]])\n"
        )
        algos = _algos(graph)
        assert "element_add" in algos
        assert "element_mul" in algos
        assert "array_concat" in algos
        concat = next(n for n in graph.nodes.values() if n.algorithm == "array_concat")
        assert len(concat.inputs) == 2

    def test_return_binop_list_arg_extends(self) -> None:
        _, graph = _build(
            "def f(a, b, c, d):\n"
            "    s = np.add(a, b)\n"
            "    return np.divide([a, b] | c, d)\n"
        )
        div = next(n for n in graph.nodes.values() if n.algorithm == "element_div")
        assert len(div.inputs) >= 2

    def test_return_keyword_constant(self) -> None:
        _, graph = _build(
            "def f(a, m):\n"
            "    s = np.add(a, m)\n"
            "    return np.full(m, fill_value=5.0)\n"
        )
        assert "alloc_full" in _algos(graph)
        full = next(n for n in graph.nodes.values() if n.algorithm == "alloc_full")
        assert any(shape == "scalar" for _, _, shape in full.inputs)


class TestResolveArgForFullBody:
    def test_helper_arg_forms_hit_resolve_branches(self) -> None:
        _, graph = _build(
            "def helper(x, y):\n"
            "    return x\n"
            "def f(a, b, n, zz):\n"
            "    u = helper(np.transpose(a), b)\n"
            "    v = helper(a.transpose(), b)\n"
            "    w = helper(a.shape, b)\n"
            "    x = helper(a.T, b)\n"
            "    y = helper(np.pi, b)\n"
            "    z = helper(np.e, b)\n"
            "    p = helper(a.shape[n], b)\n"
            "    q = helper(a.shape[zz], b)\n"
            "    r = helper(-a, b)\n"
            "    t = np.subtract(u, r)\n"
            "    return t\n"
        )
        nodes = list(graph.nodes.values())
        assert len(nodes) == 1
        assert nodes[0].algorithm == "element_sub"
        assert nodes[0].inputs[0][0] == "a"
        assert nodes[0].inputs[1][2] == "scalar"

    def test_name_in_existing_not_in_func_inputs_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {}, {}, {"x"}, {},
        )
        assert result == ("x", Dtype.FLOAT64, "array")

    def test_name_in_intermediates_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {"x": "inter_x"}, {}, set(), {},
        )
        assert result == ("inter_x", Dtype.FLOAT64, "array")

    def test_name_in_constant_assignments_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("two").body[0].value
        scalar_constants = {}
        result = builder._resolve_arg_for_full_body(
            expr, [], scalar_constants, {}, {}, set(), {"two": 2.0},
        )
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar_constants["_const_0"] == 2.0

    def test_subscript_on_intermediate_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x[i]").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {"x": "inter_x"}, {}, set(), {},
        )
        assert result == ("inter_x", Dtype.FLOAT64, "array")

    def test_negated_constant_assignment_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("-two").body[0].value
        scalar_constants = {}
        result = builder._resolve_arg_for_full_body(
            expr, [], scalar_constants, {}, {}, set(), {"two": 2.0},
        )
        assert result == ("_const_0", Dtype.FLOAT64, "scalar")
        assert scalar_constants["_const_0"] == -2.0

    def test_name_in_existing_matches_func_inputs_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr,
            [("x", Dtype.FLOAT64, "scalar")],
            {},
            {},
            {},
            {"x"},
            {},
        )
        assert result == ("x", Dtype.FLOAT64, "scalar")

    def test_transpose_method_returns_input_shape_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.transpose()").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr,
            [("a", Dtype.FLOAT64, "scalar")],
            {},
            {},
            {"a": ("a", Dtype.FLOAT64)},
            set(),
            {},
        )
        assert result == ("a", Dtype.FLOAT64, "scalar")

    def test_transpose_attr_returns_input_shape_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.T").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr,
            [("a", Dtype.FLOAT64, "scalar")],
            {},
            {},
            {"a": ("a", Dtype.FLOAT64)},
            set(),
            {},
        )
        assert result == ("a", Dtype.FLOAT64, "scalar")

    def test_transpose_method_fallback_array_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.transpose()").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {}, {"a": ("a", Dtype.FLOAT64)}, set(), {},
        )
        assert result == ("a", Dtype.FLOAT64, "array")

    def test_transpose_attr_fallback_array_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.T").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {}, {"a": ("a", Dtype.FLOAT64)}, set(), {},
        )
        assert result == ("a", Dtype.FLOAT64, "array")

    def test_transpose_attr_on_intermediate_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x.T").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {"x": "inter_x"}, {}, set(), {},
        )
        assert result == ("inter_x", Dtype.FLOAT64, "array")

    def test_shape_subscript_with_symbol_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.shape[n]").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr,
            [],
            {},
            {},
            {"a": ("a", Dtype.FLOAT64), "n": ("n", Dtype.FLOAT64)},
            set(),
            {},
        )
        assert result[0] == "_const_0"
        assert result[2] == "scalar"

    def test_shape_subscript_without_symbol_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("a.shape[zz]").body[0].value
        scalar_constants = {}
        result = builder._resolve_arg_for_full_body(
            expr,
            [],
            scalar_constants,
            {},
            {"a": ("a", Dtype.FLOAT64)},
            set(),
            {},
        )
        assert result == ("_const_0", Dtype.INT64, "scalar")
        assert scalar_constants["_const_0"] == 0.0

    def test_unresolved_unary_minus_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("-zz").body[0].value
        result = builder._resolve_arg_for_full_body(
            expr, [], {}, {}, {}, set(), {},
        )
        assert result[0].startswith("_unresolved_")

    def test_decompose_transpose_on_intermediate_direct(self) -> None:
        builder = MathIRBuilder(origin_file="unit.py")
        expr = ast.parse("x.T").body[0].value
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
        assert result == ("inter_x", Dtype.FLOAT64, "array")
