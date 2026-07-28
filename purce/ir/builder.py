from __future__ import annotations

import ast
import hashlib

from purce.ir.nodes import (
    DepKind,
    Dtype,
    Effect,
    MathIRGraph,
    MathIRNode,
    ReductionEntry,
)

# ── NumPy operation mapping ──────────────────────────────────────────────────

NUMPY_OP_MAP: dict[str, str] = {
    "numpy.dot": "matmul",
    "numpy.matmul": "matmul",
    "numpy.linalg.solve": "linalg_solve",
    "numpy.linalg.inv": "linalg_inv",
    "numpy.linalg.cholesky": "linalg_cholesky",
    "numpy.linalg.eig": "linalg_eig",
    "numpy.add": "element_add",
    "numpy.subtract": "element_sub",
    "numpy.multiply": "element_mul",
    "numpy.divide": "element_div",
    "numpy.sum": "reduce_sum",
    "numpy.mean": "reduce_mean",
    "numpy.max": "reduce_max",
    "numpy.min": "reduce_min",
    "numpy.fft.fft": "fft",
    "numpy.fft.ifft": "ifft",
    "numpy.zeros": "alloc_zeros",
    "numpy.ones": "alloc_ones",
    "numpy.eye": "alloc_eye",
    "numpy.array": "array_literal",
    "numpy.sqrt": "element_sqrt",
    "numpy.abs": "element_abs",
    "numpy.exp": "element_exp",
    "numpy.log": "element_log",
    "numpy.sin": "element_sin",
    "numpy.cos": "element_cos",
    "numpy.tan": "element_tan",
}

NUMPY_DTYPE_MAP: dict[str, Dtype] = {
    "float32": Dtype.FLOAT32,
    "float64": Dtype.FLOAT64,
    "int32": Dtype.INT32,
    "int64": Dtype.INT64,
    "bool": Dtype.BOOL,
    "complex64": Dtype.COMPLEX64,
    "complex128": Dtype.COMPLEX128,
}


def _make_node_id(prefix: str, symbol: str) -> str:
    base = f"{prefix}.{symbol}"
    h = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()[:8]
    clean = symbol.replace(".", "_").replace("-", "_")
    return f"{prefix}.{clean}_{h}"


def _resolve_dtype_annotation(annotation: ast.expr | None) -> Dtype:
    if annotation is None:
        return Dtype.FLOAT64
    if isinstance(annotation, ast.Name):
        return NUMPY_DTYPE_MAP.get(annotation.id, Dtype.FLOAT64)
    if isinstance(annotation, ast.Attribute):
        full = f"{_get_attr_string(annotation)}"
        for key, dt in NUMPY_DTYPE_MAP.items():
            if key in full:
                return dt
    return Dtype.FLOAT64


def _get_attr_string(node: ast.Attribute) -> str:
    parts = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _get_qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return _get_attr_string(node)
    if isinstance(node, ast.Name):
        return node.id
    return ""


class MathIRBuilder:
    """Builds Math-IR graphs from Python AST or Clang AST nodes."""

    def __init__(self, origin_file: str = "<unknown>", origin_commit: str | None = None):
        self.origin_file = origin_file
        self.origin_commit = origin_commit
        self.graph = MathIRGraph()
        self._diagnostics: list[dict] = []
        self._node_counter = 0

    @property
    def diagnostics(self) -> list[dict]:
        return list(self._diagnostics)

    def from_python_ast(self, tree: ast.Module, module_name: str = "module") -> MathIRGraph:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node, module_name)
        return self.graph

    def _process_function(self, func: ast.FunctionDef, module_name: str) -> None:
        has_numpy = False
        call_targets: list[str] = []
        effects = [Effect.PURE]

        for child in ast.walk(func):
            if isinstance(child, ast.Call):
                target = ""
                if isinstance(child.func, ast.Attribute):
                    target = _get_qualified_name(child.func)
                elif isinstance(child.func, ast.Name):
                    target = child.func.id

                if target.startswith("np."):
                    target = "numpy." + target[3:]

                if target in NUMPY_OP_MAP or target.startswith("numpy."):
                    has_numpy = True
                    call_targets.append(target)

                if target in ("eval", "exec", "getattr", "setattr", "open", "print"):
                    effects.append(Effect.IO)

        if not has_numpy:
            return

        inputs: list[tuple[str, Dtype, str]] = []
        for arg in func.args.args:
            dt = _resolve_dtype_annotation(arg.annotation)
            shape = "scalar" if arg.arg in ("self", "cls") else "array"
            inputs.append((arg.arg, dt, shape))

        outputs: list[tuple[str, Dtype, str]] = [("result", Dtype.FLOAT64, "array")]

        if func.returns:
            ret_dt = _resolve_dtype_annotation(func.returns)
            outputs = [("result", ret_dt, "array")]

        algorithms = []
        nested_deps: list[str] = []
        reductions: list[ReductionEntry] = []

        for target in call_targets:
            algo = NUMPY_OP_MAP.get(target, "unknown")
            if algo not in algorithms:
                algorithms.append(algo)
            reductions.append(ReductionEntry(
                rule="numpy_op_extraction",
                description=f"Extracted {target} as {algo} kernel",
                original=target,
            ))

        algorithm = algorithms[0] if algorithms else "composite"
        sig_parts = []
        for name, dt, _ in inputs:
            sig_parts.append(f"{dt.name.lower()} {name}")
        origin_sig = f"{' -> '.join(sig_parts)} -> {outputs[0][1].name.lower()}"

        self._node_counter += 1
        node_id = _make_node_id(module_name, func.name)

        math_intent_parts = []
        for target in call_targets:
            algo = NUMPY_OP_MAP.get(target, target)
            math_intent_parts.append(f"{algo}")
        math_intent = " composed of: " + ", ".join(math_intent_parts) if math_intent_parts else "math kernel"

        node = MathIRNode(
            node_id=node_id,
            origin_symbol=f"{module_name}.{func.name}",
            origin_file=self.origin_file,
            origin_line=func.lineno,
            origin_commit=self.origin_commit,
            origin_signature=origin_sig,
            math_intent=f"Function '{func.name}'{math_intent}",
            inputs=inputs,
            outputs=outputs,
            effects=effects,
            algorithm=algorithm,
            reductions=reductions,
            nested_deps=nested_deps,
            stack_usage=256,
            heap_usage=None,
            reentrant=Effect.ALLOC not in effects,
            dep_kind=DepKind.MATH_KERNEL,
        )

        self.graph.add_node(node)
        self.graph.entry_points.append(node_id)

    def from_cpp_ast(self, clang_node: object) -> MathIRGraph:
        """Stub for future libclang integration."""
        self._diagnostics.append({
            "severity": "WARNING",
            "file": self.origin_file,
            "line": 0,
            "construct": "cpp_parser",
            "reason": "C++ parser not yet implemented; libclang required",
            "suggestion": "Install libclang and implement cpp_parser.py",
        })
        return self.graph

    def link_deps(self, graph: MathIRGraph) -> MathIRGraph:
        """Resolve nested_deps references between graph nodes."""
        all_ids = set(graph.nodes.keys())
        for node in graph.nodes.values():
            resolved: list[str] = []
            for dep_id in node.nested_deps:
                if dep_id in all_ids:
                    resolved.append(dep_id)
                else:
                    self._diagnostics.append({
                        "severity": "ERROR",
                        "file": node.origin_file,
                        "line": node.origin_line,
                        "construct": "dep_resolution",
                        "reason": f"Dependency '{dep_id}' not found in graph",
                        "suggestion": "Ensure all dependencies are parsed before linking",
                    })
            node.nested_deps = resolved
        return graph

    def build_from_source(self, source: str, module: str = "module") -> MathIRGraph:
        """Parse Python source string and build IR graph."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            self._diagnostics.append({
                "severity": "ERROR",
                "file": self.origin_file,
                "line": e.lineno or 0,
                "construct": "syntax",
                "reason": str(e.msg),
                "suggestion": "Fix Python syntax errors",
            })
            return self.graph

        return self.from_python_ast(tree, module)
