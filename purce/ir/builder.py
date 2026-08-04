from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field

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
    "numpy.linalg.norm": "linalg_norm",
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
    "numpy.tanh": "element_tanh",
    "numpy.maximum": "element_max",
    "numpy.minimum": "element_min",
    "numpy.power": "element_power",
    "numpy.where": "element_where",
    "numpy.clip": "element_clip",
    "numpy.negative": "element_neg",
    "numpy.sign": "element_sign",
    "numpy.floor": "element_floor",
    "numpy.transpose": "transpose",
    "numpy.greater": "element_greater",
    "numpy.less": "element_less",
    "numpy.var": "reduce_var",
    "numpy.log10": "element_log10",
    "numpy.logaddexp": "element_logaddexp",
    "numpy.conj": "element_conj",
    "numpy.angle": "element_angle",
    "numpy.real": "element_real",
    "numpy.imag": "element_imag",
    "numpy.outer": "outer_product",
    "numpy.diag": "matrix_diag",
    "numpy.copy": "element_copy",
    "numpy.arange": "alloc_arange",
    "numpy.linspace": "alloc_linspace",
    "numpy.full": "alloc_full",
    "numpy.full_like": "alloc_full",
    "numpy.ones_like": "alloc_ones",
    "numpy.zeros_like": "alloc_zeros",
    "numpy.concatenate": "array_concat",
    "numpy.take": "array_take",
    "numpy.take_along_axis": "array_take",
    "numpy.argsort": "array_argsort",
    "numpy.tril": "matrix_tril",
    "numpy.triu": "matrix_triu",
    "numpy.random.randn": "alloc_random",
    "numpy.random.random": "alloc_random",
    "numpy.random.randint": "alloc_random",
    "numpy.random.uniform": "alloc_random",
    "numpy.random.seed": "noop_seed",
    "numpy.random.beta": "alloc_random",
    "numpy.random.permutation": "array_permutation",
    "numpy.reshape": "array_reshape",
    "numpy.squeeze": "array_squeeze",
    "numpy.expand_dims": "array_expand_dims",
    "numpy.flatten": "array_flatten",
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


@dataclass
class _CallInfo:
    target: str
    algo: str
    args: list
    is_nested: bool
    intermediate_name: str | None = None
    node_id: str | None = None


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
        local_funcs: dict[str, ast.FunctionDef] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                local_funcs[node.name] = node

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node, module_name, local_funcs)
        return self.graph

    def _process_function(self, func: ast.FunctionDef, module_name: str,
                          local_funcs: dict[str, ast.FunctionDef] | None = None) -> None:
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

        existing_input_names = {arg.arg for arg in func.args.args}
        scalar_counter = 0
        scalar_constants: dict[str, float] = {}

        constant_assignments: dict[str, float] = {}
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, (int, float)):
                        constant_assignments[target.id] = float(stmt.value.value)

        for child in ast.walk(func):
            if not isinstance(child, ast.Call):
                continue
            if not isinstance(child.func, ast.Attribute):
                continue
            target = _get_qualified_name(child.func)
            if target.startswith("np."):
                target = "numpy." + target[3:]
            if target not in NUMPY_OP_MAP:
                continue
            for arg_node in child.args:
                if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, (int, float)):
                    const_val = float(arg_node.value)
                    const_name = f"_const_{scalar_counter}"
                    scalar_counter += 1
                    if const_name not in existing_input_names:
                        dt = Dtype.FLOAT64 if isinstance(arg_node.value, float) else Dtype.INT64
                        inputs.append((const_name, dt, "scalar"))
                        existing_input_names.add(const_name)
                        scalar_constants[const_name] = const_val
                elif isinstance(arg_node, ast.Name) and arg_node.id in constant_assignments:
                    const_val = constant_assignments[arg_node.id]
                    const_name = f"_const_{scalar_counter}"
                    scalar_counter += 1
                    if const_name not in existing_input_names:
                        dt = Dtype.FLOAT64 if isinstance(const_val, float) else Dtype.INT64
                        inputs.append((const_name, dt, "scalar"))
                        existing_input_names.add(const_name)
                        scalar_constants[const_name] = const_val

        outputs: list[tuple[str, Dtype, str]] = [("result", Dtype.FLOAT64, "array")]

        if func.returns:
            ret_dt = _resolve_dtype_annotation(func.returns)
            outputs = [("result", ret_dt, "array")]

        return_stmt = None
        for child in ast.walk(func):
            if isinstance(child, ast.Return):
                return_stmt = child
                break

        has_nested = False
        if return_stmt is not None and isinstance(return_stmt.value, ast.Call):
            has_nested = self._has_nested_numpy_calls(return_stmt.value, local_funcs or {})

        if has_nested and isinstance(return_stmt.value, ast.Call):
            self._decompose_composed_function(
                func, return_stmt.value, module_name, inputs,
                outputs, effects, existing_input_names, scalar_constants, local_funcs or {},
            )
            return

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
            scalar_constants=scalar_constants,
        )

        self.graph.add_node(node)
        self.graph.entry_points.append(node_id)

    def _has_nested_numpy_calls(self, call_node: ast.Call,
                                local_funcs: dict[str, ast.FunctionDef]) -> bool:
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                target = ""
                if isinstance(arg.func, ast.Attribute):
                    target = _get_qualified_name(arg.func)
                elif isinstance(arg.func, ast.Name):
                    target = arg.func.id
                if target.startswith("np."):
                    target = "numpy." + target[3:]
                if target in NUMPY_OP_MAP:
                    return True
                if target in local_funcs:
                    local_def = local_funcs[target]
                    for stmt in ast.walk(local_def):
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                            if self._has_nested_numpy_calls(stmt.value, local_funcs):
                                return True
                    for stmt in ast.walk(local_def):
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                            for inner_arg in stmt.value.args:
                                if isinstance(inner_arg, ast.Call):
                                    inner_target = ""
                                    if isinstance(inner_arg.func, ast.Attribute):
                                        inner_target = _get_qualified_name(inner_arg.func)
                                    elif isinstance(inner_arg.func, ast.Name):
                                        inner_target = inner_arg.func.id
                                    if inner_target.startswith("np."):
                                        inner_target = "numpy." + inner_target[3:]
                                    if inner_target in NUMPY_OP_MAP:
                                        return True
        return False

    def _resolve_arg_to_name(self, arg_node: ast.expr, func_inputs: list[tuple[str, Dtype, str]],
                             scalar_constants: dict[str, float],
                             intermediates: dict[str, str],
                             existing_names: set[str]) -> tuple[str, Dtype, str | None]:
        if isinstance(arg_node, ast.Name) and arg_node.id in existing_names:
            for name, dt, shape in func_inputs:
                if name == arg_node.id:
                    return (name, dt, None)
            return (arg_node.id, Dtype.FLOAT64, None)

        if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, (int, float)):
            const_val = float(arg_node.value)
            const_name = f"_const_{len(scalar_constants)}"
            if const_name not in existing_names:
                dt = Dtype.FLOAT64 if isinstance(arg_node.value, float) else Dtype.INT64
                scalar_constants[const_name] = const_val
                existing_names.add(const_name)
            return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Call):
            target = ""
            if isinstance(arg_node.func, ast.Attribute):
                target = _get_qualified_name(arg_node.func)
            elif isinstance(arg_node.func, ast.Name):
                target = arg_node.func.id
            if target.startswith("np."):
                target = "numpy." + target[3:]
            if target in intermediates:
                inter_name = intermediates[target]
                return (inter_name, Dtype.FLOAT64, None)

        return (f"_unresolved_{len(existing_names)}", Dtype.FLOAT64, None)

    def _decompose_composed_function(
        self,
        func: ast.FunctionDef,
        outer_call: ast.Call,
        module_name: str,
        func_inputs: list[tuple[str, Dtype, str]],
        func_outputs: list[tuple[str, Dtype, str]],
        effects: list[Effect],
        existing_names: set[str],
        scalar_constants: dict[str, float],
        local_funcs: dict[str, ast.FunctionDef] | None = None,
    ) -> None:
        operations: list[tuple[str, ast.Call]] = []
        self._collect_numpy_calls(outer_call, operations, local_funcs or {})

        if len(operations) <= 1:
            return

        intermediates: dict[str, str] = {}
        all_reductions: list[ReductionEntry] = []
        all_dep_ids: list[str] = []
        origin_file = self.origin_file

        for idx, (target, call_node) in enumerate(operations):
            algo = NUMPY_OP_MAP.get(target, "unknown")
            is_last = (idx == len(operations) - 1)

            node_inputs: list[tuple[str, Dtype, str]] = []
            for arg in call_node.args:
                resolved = self._resolve_arg_to_name(
                    arg, func_inputs, scalar_constants, intermediates, existing_names,
                )
                node_inputs.append(resolved)

            if is_last:
                node_outputs = list(func_outputs)
            else:
                inter_name = f"_inter_{func.name}_{idx}"
                node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
                intermediates[target] = inter_name

            inter_key = f"{target}_{idx}"
            intermediates[inter_key] = node_outputs[0][0]

            reductions = [ReductionEntry(
                rule="numpy_op_extraction",
                description=f"Extracted {target} as {algo} kernel",
                original=target,
            )]
            all_reductions.extend(reductions)

            sig_parts = []
            for name, dt, _ in node_inputs:
                sig_parts.append(f"{dt.name.lower()} {name}")
            origin_sig = f"{' -> '.join(sig_parts)} -> {node_outputs[0][1].name.lower()}"

            self._node_counter += 1
            suffix = f"_{idx}" if idx > 0 else ""
            node_id = _make_node_id(module_name, f"{func.name}{suffix}")

            math_intent = f"Step {idx + 1}/{len(operations)} of '{func.name}' implementing {algo}"

            node = MathIRNode(
                node_id=node_id,
                origin_symbol=f"{module_name}.{func.name}",
                origin_file=origin_file,
                origin_line=func.lineno,
                origin_commit=self.origin_commit,
                origin_signature=origin_sig,
                math_intent=math_intent,
                inputs=node_inputs,
                outputs=node_outputs,
                effects=effects if is_last else [Effect.PURE],
                algorithm=algo,
                reductions=reductions,
                nested_deps=list(all_dep_ids),
                stack_usage=256,
                heap_usage=None,
                reentrant=True,
                dep_kind=DepKind.MATH_KERNEL,
                scalar_constants=dict(scalar_constants),
            )

            self.graph.add_node(node)
            self.graph.entry_points.append(node_id)
            all_dep_ids.append(node_id)

    def _collect_numpy_calls(self, call_node: ast.Call, operations: list[tuple[str, ast.Call]],
                             local_funcs: dict[str, ast.FunctionDef]) -> None:
        target = ""
        if isinstance(call_node.func, ast.Attribute):
            target = _get_qualified_name(call_node.func)
        elif isinstance(call_node.func, ast.Name):
            target = call_node.func.id
        if target.startswith("np."):
            target = "numpy." + target[3:]

        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                arg_target = ""
                if isinstance(arg.func, ast.Attribute):
                    arg_target = _get_qualified_name(arg.func)
                elif isinstance(arg.func, ast.Name):
                    arg_target = arg.func.id
                if arg_target.startswith("np."):
                    arg_target = "numpy." + arg_target[3:]
                if arg_target in NUMPY_OP_MAP:
                    self._collect_numpy_calls(arg, operations, local_funcs)
                elif arg_target in local_funcs:
                    local_def = local_funcs[arg_target]
                    for stmt in ast.walk(local_def):
                        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                            self._collect_numpy_calls(stmt.value, operations, local_funcs)

        if target in NUMPY_OP_MAP:
            already_added = any(t == target for t, _ in operations)
            if not already_added:
                operations.append((target, call_node))

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
