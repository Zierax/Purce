
from purce.ir.nodes import (
    DepKind,
    Dtype,
    Effect,
    MathIRGraph,
    MathIRNode,
)
from purce.slicer.semantic_slicer import SemanticSlicer


def _make_node(
    node_id: str,
    deps: list[str] | None = None,
    effects: list[Effect] | None = None,
    dep_kind: DepKind = DepKind.MATH_KERNEL,
    scalar_constants: dict[str, float] | None = None,
) -> MathIRNode:
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
        effects=effects or [Effect.PURE],
        algorithm="test",
        nested_deps=deps or [],
        stack_usage=64,
        dep_kind=dep_kind,
        scalar_constants=scalar_constants or {},
    )


def _build_linear_graph() -> MathIRGraph:
    g = MathIRGraph()
    g.add_node(_make_node("a"))
    g.add_node(_make_node("b", deps=["a"]))
    g.add_node(_make_node("c", deps=["b"]))
    g.entry_points = ["c"]
    return g


def _build_diamond_graph() -> MathIRGraph:
    g = MathIRGraph()
    g.add_node(_make_node("a"))
    g.add_node(_make_node("b", deps=["a"]))
    g.add_node(_make_node("c", deps=["a"]))
    g.add_node(_make_node("d", deps=["b", "c"]))
    g.entry_points = ["d"]
    return g


class TestSemanticSlicerBasic:
    def test_linear_prune(self) -> None:
        g = _build_linear_graph()
        slicer = SemanticSlicer()
        result = slicer.slice(g, ["c"])

        assert "c" in result.graph.nodes
        assert "b" in result.graph.nodes
        assert "a" in result.graph.nodes
        assert len(result.graph.nodes) == 3

    def test_diamond_prune(self) -> None:
        g = _build_diamond_graph()
        slicer = SemanticSlicer()
        result = slicer.slice(g, ["d"])

        assert len(result.graph.nodes) == 4
        assert "d" in result.graph.nodes
        assert "a" in result.graph.nodes

    def test_unused_nodes_removed(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("used", deps=["helper"]))
        g.add_node(_make_node("helper"))
        g.add_node(_make_node("orphan"))
        g.entry_points = ["used"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["used"])

        assert "used" in result.graph.nodes
        assert "helper" in result.graph.nodes
        assert "orphan" not in result.graph.nodes

    def test_multiple_entry_points(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a"))
        g.add_node(_make_node("b"))
        g.add_node(_make_node("shared", deps=["a", "b"]))
        g.entry_points = ["shared"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["shared"])

        assert len(result.graph.nodes) == 3


class TestSemanticSlicerClassification:
    def test_pal_stub_generated(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node(
            "io_node",
            dep_kind=DepKind.SYSTEM_PAL,
        ))
        g.entry_points = ["io_node"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["io_node"])

        assert "io_node" in result.pal_stubs
        assert "io_node" not in result.graph.nodes

    def test_meta_util_inlined(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node(
            "util_fn",
            dep_kind=DepKind.META_UTIL,
        ))
        g.entry_points = ["util_fn"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["util_fn"])

        assert "util_fn" in result.inlined_utils
        assert "util_fn" not in result.graph.nodes

    def test_data_asset_error(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node(
            "fetch_data",
            dep_kind=DepKind.DATA_ASSET,
        ))
        g.entry_points = ["fetch_data"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["fetch_data"])

        assert "fetch_data" in result.data_assets
        assert any(d["severity"] == "ERROR" for d in result.diagnostics)

    def test_data_asset_embed(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node(
            "fetch_data",
            dep_kind=DepKind.DATA_ASSET,
        ))
        g.entry_points = ["fetch_data"]

        slicer = SemanticSlicer(embed_assets=True)
        result = slicer.slice(g, ["fetch_data"])

        assert "fetch_data" in result.graph.nodes
        assert not any(d["severity"] == "ERROR" for d in result.diagnostics)

    def test_mixed_classification(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("kernel"))
        g.add_node(_make_node("helper", deps=["kernel"]))
        g.add_node(_make_node("io_fn", dep_kind=DepKind.SYSTEM_PAL))
        g.add_node(_make_node("dead_code"))
        g.entry_points = ["helper"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["helper"])

        assert "kernel" in result.graph.nodes
        assert "helper" in result.graph.nodes
        assert "dead_code" not in result.graph.nodes


class TestSemanticSlicerPruneUnused:
    def test_prune_linear(self) -> None:
        g = _build_linear_graph()
        slicer = SemanticSlicer()
        pruned = slicer.prune_unused(g, ["a"])

        assert "a" in pruned.nodes
        assert "b" not in pruned.nodes
        assert "c" not in pruned.nodes

    def test_prune_diamond(self) -> None:
        g = _build_diamond_graph()
        slicer = SemanticSlicer()
        pruned = slicer.prune_unused(g, ["b"])

        assert "b" in pruned.nodes
        assert "a" in pruned.nodes
        assert "c" not in pruned.nodes
        assert "d" not in pruned.nodes


class TestSemanticSlicerResolveTransitive:
    def test_resolve_all_present(self) -> None:
        g = _build_linear_graph()
        slicer = SemanticSlicer()
        resolved = slicer.resolve_transitive_deps(g)

        assert "a" in resolved.nodes
        assert "b" in resolved.nodes
        assert "c" in resolved.nodes

    def test_resolve_missing_deps(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("x", deps=["ghost"]))
        slicer = SemanticSlicer()
        resolved = slicer.resolve_transitive_deps(g)

        assert "x" in resolved.nodes
        assert resolved.nodes["x"].nested_deps == []


class TestSemanticSlicerDAGValidation:
    def test_cycle_detected(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a", deps=["b"]))
        g.add_node(_make_node("b", deps=["a"]))
        g.entry_points = ["a"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["a"])

        assert any("Cycle" in d.get("reason", "") for d in result.diagnostics)

    def test_missing_dep_detected(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("x", deps=["nonexistent"]))
        g.entry_points = ["x"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["x"])

        assert len(result.graph.nodes) == 1
        assert result.graph.nodes["x"].nested_deps == []


class TestSemanticSlicerEdgeCases:
    def test_empty_graph(self) -> None:
        g = MathIRGraph()
        slicer = SemanticSlicer()
        result = slicer.slice(g, [])
        assert len(result.graph.nodes) == 0

    def test_entry_not_in_graph(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a"))
        slicer = SemanticSlicer()
        result = slicer.slice(g, ["ghost"])
        assert len(result.graph.nodes) == 0

    def test_deep_chain_3_levels(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("level0"))
        g.add_node(_make_node("level1", deps=["level0"]))
        g.add_node(_make_node("level2", deps=["level1"]))
        g.add_node(_make_node("level3", deps=["level2"]))
        g.entry_points = ["level3"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["level3"])

        assert len(result.graph.nodes) == 4
        node3 = result.graph.nodes["level3"]
        assert node3.nested_deps == ["level2"]
        node2 = result.graph.nodes["level2"]
        assert node2.nested_deps == ["level1"]
        node1 = result.graph.nodes["level1"]
        assert node1.nested_deps == ["level0"]


class TestSemanticSlicerScalarConstants:
    def test_scalar_constants_preserved_through_slice(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a", scalar_constants={"_const_0": 3.141592653589793}))
        g.entry_points = ["a"]

        slicer = SemanticSlicer()
        result = slicer.slice(g, ["a"])

        assert result.graph.nodes["a"].scalar_constants == {"_const_0": 3.141592653589793}

    def test_scalar_constants_preserved_through_resolve(self) -> None:
        g = MathIRGraph()
        g.add_node(_make_node("a", scalar_constants={"_const_1": 2.0}))
        g.entry_points = ["a"]

        slicer = SemanticSlicer()
        resolved = slicer.resolve_transitive_deps(g)

        assert resolved.nodes["a"].scalar_constants == {"_const_1": 2.0}
