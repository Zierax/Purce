from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Effect(Enum):
    """Classifies function side effects."""

    PURE = auto()
    IO = auto()
    TEMPORAL = auto()
    RANDOM = auto()
    ALLOC = auto()


class Dtype(Enum):
    """Supported data types for IR nodes."""

    FLOAT32 = auto()
    FLOAT64 = auto()
    INT32 = auto()
    INT64 = auto()
    Q15 = auto()
    Q31 = auto()
    BOOL = auto()
    COMPLEX64 = auto()
    COMPLEX128 = auto()


class DepKind(Enum):
    """Dependency classification per C99-SOS standard."""

    MATH_KERNEL = 1
    SYSTEM_PAL = 2
    DATA_ASSET = 3
    META_UTIL = 4


@dataclass
class ReductionEntry:
    """Records a transformation applied during IR construction."""

    rule: str
    description: str
    original: str | None = None


@dataclass
class MathIRNode:
    """A single node in the Math-IR graph representing one semantic unit."""

    node_id: str
    origin_symbol: str
    origin_file: str
    origin_line: int
    origin_commit: str | None
    origin_signature: str

    math_intent: str
    inputs: list[tuple[str, Dtype, str]]
    outputs: list[tuple[str, Dtype, str]]

    effects: list[Effect]
    algorithm: str

    reductions: list[ReductionEntry] = field(default_factory=list)
    nested_deps: list[str] = field(default_factory=list)

    stack_usage: int | None = None
    heap_usage: int | None = None
    reentrant: bool = True

    dep_kind: DepKind = DepKind.MATH_KERNEL
    scalar_constants: dict[str, float] = field(default_factory=dict)

    def is_pure(self) -> bool:
        return self.effects == [Effect.PURE]

    def input_dtypes(self) -> list[Dtype]:
        return [dt for _, dt, _ in self.inputs]

    def output_dtypes(self) -> list[Dtype]:
        return [dt for _, dt, _ in self.outputs]


@dataclass
class MathIRGraph:
    """DAG of Math-IR nodes representing an entire computation."""

    nodes: dict[str, MathIRNode] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)

    def add_node(self, node: MathIRNode) -> None:
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
            for n in self.nodes.values():
                if node_id in n.nested_deps:
                    n.nested_deps.remove(node_id)

    def get_node(self, node_id: str) -> MathIRNode | None:
        return self.nodes.get(node_id)

    def topological_sort(self) -> list[MathIRNode]:
        """Return nodes in dependency order (leaves first, entry points last).

        Uses iterative Kahn's algorithm to avoid stack overflow on deep graphs.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        dependents: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for nid, node in self.nodes.items():
            for dep_id in node.nested_deps:
                if dep_id in self.nodes:
                    in_degree[nid] += 1
                    dependents[dep_id].append(nid)

        for dep_list in dependents.values():
            dep_list.sort()
        queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
        order: list[MathIRNode] = []

        while queue:
            nid = queue.pop(0)
            order.append(self.nodes[nid])
            for dependent_id in dependents[nid]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
            queue.sort()

        for nid in sorted(self.nodes):
            if nid not in [n.node_id for n in order]:
                order.append(self.nodes[nid])

        return order

    def reachable_from(self, entry_ids: list[str]) -> MathIRGraph:
        """Return a new graph containing only nodes reachable from entry_ids."""
        reachable: set[str] = set()
        stack = list(entry_ids)

        while stack:
            nid = stack.pop()
            if nid in reachable or nid not in self.nodes:
                continue
            reachable.add(nid)
            node = self.nodes[nid]
            for dep_id in node.nested_deps:
                if dep_id not in reachable:
                    stack.append(dep_id)

        sub = MathIRGraph(entry_points=list(entry_ids))
        for nid in sorted(reachable):
            sub.nodes[nid] = self.nodes[nid]
        return sub

    def validate_dag(self) -> list[str]:
        """Return list of errors if graph has cycles or broken deps."""
        errors: list[str] = []
        for nid, node in sorted(self.nodes.items()):
            for dep_id in sorted(node.nested_deps):
                if dep_id not in self.nodes:
                    errors.append(f"Node '{nid}' depends on missing node '{dep_id}'")

        if not errors:
            visited: set[str] = set()
            in_stack: set[str] = set()

            def _check_cycle(nid: str) -> bool:
                if nid in in_stack:
                    return True
                if nid in visited:
                    return False
                visited.add(nid)
                in_stack.add(nid)
                node = self.nodes.get(nid)
                if node:
                    for dep_id in sorted(node.nested_deps):
                        if _check_cycle(dep_id):
                            errors.append(f"Cycle detected involving '{nid}'")
                            return True
                in_stack.discard(nid)
                return False

            for nid in sorted(self.nodes):
                _check_cycle(nid)

        return errors
