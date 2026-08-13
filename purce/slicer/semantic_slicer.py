from __future__ import annotations

from dataclasses import dataclass, field

from purce.ir.nodes import DepKind, MathIRGraph, MathIRNode


@dataclass
class SliceResult:
    graph: MathIRGraph
    pal_stubs: list[str] = field(default_factory=list)
    data_assets: list[str] = field(default_factory=list)
    inlined_utils: list[str] = field(default_factory=list)
    diagnostics: list[dict[str, str]] = field(default_factory=list)


class SemanticSlicer:
    """Performs call-graph analysis, dependency resolution, and dead code elimination."""

    def __init__(self, embed_assets: bool = False):
        self.embed_assets = embed_assets

    def slice(self, graph: MathIRGraph, entry_points: list[str]) -> SliceResult:
        result = SliceResult(graph=MathIRGraph(entry_points=list(entry_points)))

        reachable = self._reachability(graph, entry_points)
        self._classify_and_add(graph, reachable, result)

        errors = result.graph.validate_dag()
        for err in errors:
            result.diagnostics.append({
                "severity": "ERROR",
                "construct": "dag_validation",
                "reason": err,
            })

        return result

    def _reachability(self, graph: MathIRGraph, entry_points: list[str]) -> set[str]:
        visited: set[str] = set()
        stack = list(entry_points)

        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            if nid not in graph.nodes:
                continue
            visited.add(nid)
            node = graph.nodes[nid]
            for dep_id in node.nested_deps:
                if dep_id not in visited:
                    stack.append(dep_id)

        return visited

    def _classify_and_add(
        self,
        source_graph: MathIRGraph,
        reachable: set[str],
        result: SliceResult,
    ) -> None:
        for nid in reachable:
            node = source_graph.nodes.get(nid)
            if node is None:
                continue

            kind = self._classify(node)

            if kind == DepKind.MATH_KERNEL:
                resolved_deps = []
                for dep_id in node.nested_deps:
                    if dep_id in reachable:
                        resolved_deps.append(dep_id)

                new_node = MathIRNode(
                    node_id=node.node_id,
                    origin_symbol=node.origin_symbol,
                    origin_file=node.origin_file,
                    origin_line=node.origin_line,
                    origin_commit=node.origin_commit,
                    origin_signature=node.origin_signature,
                    math_intent=node.math_intent,
                    inputs=list(node.inputs),
                    outputs=list(node.outputs),
                    effects=list(node.effects),
                    algorithm=node.algorithm,
                    reductions=list(node.reductions),
                    nested_deps=resolved_deps,
                    stack_usage=node.stack_usage,
                    heap_usage=node.heap_usage,
                    reentrant=node.reentrant,
                    dep_kind=kind,
                    scalar_constants=dict(node.scalar_constants),
                )
                result.graph.add_node(new_node)

            elif kind == DepKind.SYSTEM_PAL:
                result.pal_stubs.append(nid)

            elif kind == DepKind.DATA_ASSET:
                if self.embed_assets:
                    result.graph.add_node(node)
                else:
                    result.data_assets.append(nid)
                    result.diagnostics.append({
                        "severity": "ERROR",
                        "construct": "data_asset",
                        "reason": (
                            f"Node '{nid}' requires runtime data fetching. "
                            "Use --embed-assets to include, or refactor to remove data dependency."
                        ),
                    })

            elif kind == DepKind.META_UTIL:
                result.inlined_utils.append(nid)

    def _classify(self, node: MathIRNode) -> DepKind:
        # Classification is taken from the builder's explicit dep_kind.  Effects
        # (IO/TEMPORAL/RANDOM) are verification metadata, NOT grounds for
        # downgrading a compilable MATH_KERNEL to SYSTEM_PAL and deleting it:
        # a function that calls np.add() and also prints is still a real math
        # kernel and must be emitted.  (Formerly such kernels were silently
        # dropped, losing observable behavior.)
        return node.dep_kind

    def prune_unused(self, graph: MathIRGraph, entry_points: list[str]) -> MathIRGraph:
        reachable = self._reachability(graph, entry_points)
        pruned = MathIRGraph(entry_points=list(entry_points))
        for nid in reachable:
            if nid in graph.nodes:
                pruned.nodes[nid] = graph.nodes[nid]
        return pruned

    def resolve_transitive_deps(self, graph: MathIRGraph) -> MathIRGraph:
        all_ids = set(graph.nodes.keys())
        resolved = MathIRGraph(entry_points=list(graph.entry_points))

        for node in graph.nodes.values():
            new_deps = []
            for dep_id in node.nested_deps:
                if dep_id in all_ids:
                    new_deps.append(dep_id)

            new_node = MathIRNode(
                node_id=node.node_id,
                origin_symbol=node.origin_symbol,
                origin_file=node.origin_file,
                origin_line=node.origin_line,
                origin_commit=node.origin_commit,
                origin_signature=node.origin_signature,
                math_intent=node.math_intent,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                effects=list(node.effects),
                algorithm=node.algorithm,
                reductions=list(node.reductions),
                nested_deps=new_deps,
                stack_usage=node.stack_usage,
                heap_usage=node.heap_usage,
                reentrant=node.reentrant,
                dep_kind=node.dep_kind,
                scalar_constants=dict(node.scalar_constants),
            )
            resolved.add_node(new_node)

        return resolved
