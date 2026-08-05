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
    "numpy.astype": "element_copy",
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
    "numpy.sort": "array_sort",
    "numpy.linalg.det": "linalg_det",
    "numpy.argmax": "reduce_argmax",
    "numpy.argmin": "reduce_argmin",
    "numpy.any": "reduce_any",
    "numpy.all": "reduce_all",
    "numpy.round": "element_round",
    "numpy.ceil": "element_ceil",
    "numpy.trunc": "element_trunc",
    "numpy.stack": "array_concat",
    "numpy.vstack": "array_concat",
    "numpy.hstack": "array_concat",
    "numpy.tile": "array_tile",
    "numpy.repeat": "array_repeat",
    "numpy.flip": "array_flip",
    "numpy.roll": "array_roll",
    "numpy.split": "array_split",
    "numpy.array_split": "array_split",
    "numpy.unique": "array_unique",
    "numpy.sort": "array_sort",
    "numpy.linalg.qr": "linalg_qr",
    "numpy.linalg.svd": "linalg_svd",
    "numpy.isclose": "element_isclose",
    "numpy.isnan": "element_isnan",
    "numpy.isinf": "element_isinf",
    "numpy.finfo": "element_finfo",
    "numpy.prod": "reduce_prod",
    "numpy.cumsum": "reduce_cumsum",
    "numpy.diff": "reduce_diff",
    "numpy.searchsorted": "array_searchsorted",
    "numpy.float64": "element_copy",
    "numpy.float32": "element_copy",
    "numpy.int64": "element_copy",
    "numpy.int32": "element_copy",
    "numpy.bool_": "element_copy",
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


_NUMPY_DTYPE_NAMES = {
    "np.float16", "np.float32", "np.float64", "np.float128",
    "np.int8", "np.int16", "np.int32", "np.int64",
    "np.uint8", "np.uint16", "np.uint32", "np.uint64",
    "np.complex64", "np.complex128", "np.bool_", "np.object_",
    "numpy.float16", "numpy.float32", "numpy.float64", "numpy.float128",
    "numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64",
    "numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64",
    "numpy.complex64", "numpy.complex128", "numpy.bool_", "numpy.object_",
}

_PY_BUILTIN_TYPE_NAMES = {"float", "int", "complex", "bool", "str", "bytes", "bytearray"}


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
        for i, arg in enumerate(func.args.args):
            dt = _resolve_dtype_annotation(arg.annotation)
            shape = "scalar" if arg.arg in ("self", "cls") else "array"
            defaults_offset = len(func.args.args) - len(func.args.defaults)
            default_idx = i - defaults_offset
            if default_idx >= 0 and default_idx < len(func.args.defaults):
                def_val = func.args.defaults[default_idx]
                if isinstance(def_val, ast.Constant) and isinstance(def_val.value, (int, float)):
                    shape = "scalar"
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

        has_multi_stmt = self._has_multi_statement_numpy(func)
        if has_multi_stmt:
            self._decompose_full_body(
                func, module_name, inputs, outputs, effects,
                existing_input_names, scalar_constants, local_funcs or {},
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

        if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, (int, float, complex)):
            const_val = float(arg_node.value.real) if isinstance(arg_node.value, complex) else float(arg_node.value)
            const_name = f"_const_{len(scalar_constants)}"
            if const_name not in existing_names:
                dt = Dtype.FLOAT64 if isinstance(arg_node.value, float) else Dtype.INT64
                scalar_constants[const_name] = const_val
                existing_names.add(const_name)
            return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Attribute) and arg_node.attr == "T":
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in existing_names:
                return (arg_node.value.id, Dtype.FLOAT64, None)
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in intermediates:
                return (intermediates[arg_node.value.id], Dtype.FLOAT64, None)

        if isinstance(arg_node, ast.Attribute):
            full_name = ""
            if isinstance(arg_node.value, ast.Name):
                full_name = f"{arg_node.value.id}.{arg_node.attr}"
            if full_name in ("np.pi", "numpy.pi"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.pi
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in ("np.e", "numpy.e"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.e
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in _NUMPY_DTYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
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

        if isinstance(arg_node, ast.UnaryOp) and isinstance(arg_node.op, ast.USub):
            if isinstance(arg_node.operand, ast.Constant) and isinstance(arg_node.operand.value, (int, float, complex)):
                const_val = -float(arg_node.operand.value.real) if isinstance(arg_node.operand.value, complex) else -float(arg_node.operand.value)
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if isinstance(arg_node.operand, ast.Name) and arg_node.operand.id in intermediates:
                inter_name = intermediates[arg_node.operand.id]
                neg_name = f"_const_{len(scalar_constants)}"
                if neg_name not in existing_names:
                    scalar_constants[neg_name] = -1.0
                    existing_names.add(neg_name)
                return (neg_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Subscript):
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in existing_names:
                return (arg_node.value.id, Dtype.FLOAT64, "array")

        if isinstance(arg_node, ast.Name):
            if arg_node.id in _PY_BUILTIN_TYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            for name, dt, shape in func_inputs:
                if name == arg_node.id:
                    return (name, dt, None)

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

        local_func_call_ids: dict[int, str] = {}
        if local_funcs:
            for arg in outer_call.args:
                if isinstance(arg, ast.Call):
                    target = ""
                    if isinstance(arg.func, ast.Name):
                        target = arg.func.id
                    elif isinstance(arg.func, ast.Attribute):
                        target = _get_qualified_name(arg.func)
                    if target.startswith("np."):
                        target = "numpy." + target[3:]
                    if target in local_funcs:
                        local_func_call_ids[id(arg)] = target
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Call):
                            sub_target = ""
                            if isinstance(sub.func, ast.Name):
                                sub_target = sub.func.id
                            if sub_target in local_funcs:
                                local_func_call_ids[id(sub)] = sub_target

        intermediates: dict[str, str] = {}
        all_reductions: list[ReductionEntry] = []
        all_dep_ids: list[str] = []
        origin_file = self.origin_file

        symbol_table: dict[str, tuple[str, Dtype]] = {}
        for name, dt, shape in func_inputs:
            symbol_table[name] = (name, dt)

        constant_assignments: dict[str, float] = {}
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, (int, float)):
                        constant_assignments[target.id] = float(stmt.value.value)

        node_idx = 0
        processed_call_ids: set[int] = set()

        for idx, (target, call_node) in enumerate(operations):
            algo = NUMPY_OP_MAP.get(target, "unknown")
            is_last = (idx == len(operations) - 1)
            processed_call_ids.add(id(call_node))

            node_idx_ref = [node_idx]
            node_inputs: list[tuple[str, Dtype, str]] = []
            for arg in call_node.args:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    for elt in arg.elts:
                        if isinstance(elt, ast.Call) and id(elt) in processed_call_ids:
                            resolved = self._resolve_arg_to_name(
                                elt, func_inputs, scalar_constants, intermediates, existing_names,
                            )
                        else:
                            resolved = self._decompose_expr(
                                elt, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func.name, origin_file, all_dep_ids,
                                node_idx_ref,
                            )
                        if isinstance(resolved, list):
                            node_inputs.extend(resolved)
                        else:
                            node_inputs.append(resolved)
                else:
                    if isinstance(arg, ast.Call) and id(arg) in processed_call_ids:
                        resolved = self._resolve_arg_to_name(
                            arg, func_inputs, scalar_constants, intermediates, existing_names,
                        )
                    elif isinstance(arg, ast.Call) and id(arg) in local_func_call_ids:
                        local_func_name = local_func_call_ids[id(arg)]
                        last_inter = None
                        for op_idx in range(len(operations) - 1, -1, -1):
                            op_target, op_call = operations[op_idx]
                            inter_key = f"{op_target}_{op_idx}"
                            if inter_key in intermediates:
                                last_inter = intermediates[inter_key]
                                break
                        if last_inter:
                            resolved = (last_inter, Dtype.FLOAT64, None)
                        else:
                            resolved = self._resolve_arg_to_name(
                                arg, func_inputs, scalar_constants, intermediates, existing_names,
                            )
                    else:
                        resolved = self._decompose_expr(
                            arg, func_inputs, scalar_constants, intermediates,
                            symbol_table, existing_names, constant_assignments,
                            module_name, func.name, origin_file, all_dep_ids,
                            node_idx_ref,
                        )
                    if isinstance(resolved, list):
                        node_inputs.extend(resolved)
                    else:
                        node_inputs.append(resolved)

            node_idx = node_idx_ref[0]

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

    def _has_multi_statement_numpy(self, func: ast.FunctionDef) -> bool:
        numpy_calls_in_assignments = 0
        numpy_calls_in_return = 0
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign):
                for child in ast.walk(stmt.value):
                    if isinstance(child, ast.Call):
                        target = ""
                        if isinstance(child.func, ast.Attribute):
                            target = _get_qualified_name(child.func)
                        elif isinstance(child.func, ast.Name):
                            target = child.func.id
                        if target.startswith("np."):
                            target = "numpy." + target[3:]
                        if target in NUMPY_OP_MAP:
                            numpy_calls_in_assignments += 1
        for child in ast.walk(func):
            if isinstance(child, ast.Return) and child.value is not None:
                for sub in ast.walk(child.value):
                    if isinstance(sub, ast.Call):
                        target = ""
                        if isinstance(sub.func, ast.Attribute):
                            target = _get_qualified_name(sub.func)
                        elif isinstance(sub.func, ast.Name):
                            target = sub.func.id
                        if target.startswith("np."):
                            target = "numpy." + target[3:]
                        if target in NUMPY_OP_MAP:
                            numpy_calls_in_return += 1
        return numpy_calls_in_assignments > 0 and (numpy_calls_in_assignments + numpy_calls_in_return) > 1

    def _resolve_call_target(self, call_node: ast.Call) -> str:
        target = ""
        if isinstance(call_node.func, ast.Attribute):
            target = _get_qualified_name(call_node.func)
            if call_node.func.attr in ("transpose", "T"):
                target = "numpy.transpose"
            elif call_node.func.attr == "reshape" and isinstance(call_node.func.value, ast.Name):
                target = "numpy.reshape"
            elif call_node.func.attr == "flatten" and isinstance(call_node.func.value, ast.Name):
                target = "numpy.flatten"
            elif call_node.func.attr == "squeeze" and isinstance(call_node.func.value, ast.Name):
                target = "numpy.squeeze"
            elif call_node.func.attr == "astype" and isinstance(call_node.func.value, ast.Name):
                target = "numpy.astype"
            elif call_node.func.attr == "copy" and isinstance(call_node.func.value, ast.Name):
                target = "numpy.copy"
        elif isinstance(call_node.func, ast.Name):
            target = call_node.func.id
        if target.startswith("np."):
            target = "numpy." + target[3:]
        return target

    def _collect_all_numpy_calls_from_body(self, func: ast.FunctionDef,
                                           local_funcs: dict[str, ast.FunctionDef]) -> list[tuple[str, ast.Call]]:
        operations: list[tuple[str, ast.Call]] = []
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Call):
                    self._collect_numpy_calls(stmt.value, operations, local_funcs)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                if isinstance(stmt.value, ast.Call):
                    self._collect_numpy_calls(stmt.value, operations, local_funcs)
                elif isinstance(stmt.value, ast.Tuple):
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Call):
                            self._collect_numpy_calls(elt, operations, local_funcs)
        return operations

    def _decompose_expr(
        self,
        expr: ast.expr,
        func_inputs: list[tuple[str, Dtype, str]],
        scalar_constants: dict[str, float],
        intermediates: dict[str, str],
        symbol_table: dict[str, tuple[str, Dtype]],
        existing_names: set[str],
        constant_assignments: dict[str, float],
        module_name: str,
        func_name: str,
        origin_file: str,
        all_dep_ids: list[str],
        node_idx: list[int],
    ) -> tuple[str, Dtype, str | None]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float, complex)):
            const_val = float(expr.value.real) if isinstance(expr.value, complex) else float(expr.value)
            const_name = f"_const_{len(scalar_constants)}"
            if const_name not in existing_names:
                scalar_constants[const_name] = const_val
                existing_names.add(const_name)
            return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(expr, ast.Name):
            if expr.id in symbol_table:
                name, dt = symbol_table[expr.id]
                return (name, dt, None)
            if expr.id in intermediates:
                return (intermediates[expr.id], Dtype.FLOAT64, None)
            if constant_assignments and expr.id in constant_assignments:
                const_val = constant_assignments[expr.id]
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if expr.id in _PY_BUILTIN_TYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            return (expr.id, Dtype.FLOAT64, None)

        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            inner = self._decompose_expr(
                expr.operand, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
                module_name, func_name, origin_file, all_dep_ids, node_idx,
            )
            if inner[0] in scalar_constants:
                const_val = -scalar_constants[inner[0]]
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            neg_const_name = f"_const_{len(scalar_constants)}"
            if neg_const_name not in existing_names:
                scalar_constants[neg_const_name] = -1.0
                existing_names.add(neg_const_name)
            inter_name = f"_inter_neg_{node_idx[0]}"
            node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
            node_idx[0] += 1
            self._node_counter += 1
            node_id = _make_node_id(module_name, f"{func_name}_neg_{node_idx[0]}")
            node = MathIRNode(
                node_id=node_id,
                origin_symbol=f"{module_name}.{func_name}",
                origin_file=origin_file,
                origin_line=0,
                origin_commit=self.origin_commit,
                origin_signature=f"double {neg_const_name}, double {inner[0]} -> array",
                math_intent="Negation via element_mul with -1",
                inputs=[(neg_const_name, Dtype.FLOAT64, "scalar"), inner],
                outputs=node_outputs,
                effects=[Effect.PURE],
                algorithm="element_mul",
                reductions=[ReductionEntry(rule="unary_neg", description="Negation via multiply", original="neg")],
                nested_deps=list(all_dep_ids),
                stack_usage=256, heap_usage=None, reentrant=True,
                dep_kind=DepKind.MATH_KERNEL,
                scalar_constants=dict(scalar_constants),
            )
            self.graph.add_node(node)
            self.graph.entry_points.append(node_id)
            all_dep_ids.append(node_id)
            intermediates[f"_neg_{node_idx[0]}"] = inter_name
            return (inter_name, Dtype.FLOAT64, "array")

        if isinstance(expr, ast.BinOp):
            left = self._decompose_expr(
                expr.left, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
                module_name, func_name, origin_file, all_dep_ids, node_idx,
            )
            right = self._decompose_expr(
                expr.right, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
                module_name, func_name, origin_file, all_dep_ids, node_idx,
            )
            binop_map = {
                ast.Add: "element_add", ast.Sub: "element_sub",
                ast.Mult: "element_mul", ast.Div: "element_div",
                ast.Pow: "element_pow", ast.Mod: "element_mod",
            }
            if type(expr.op) in binop_map:
                algo = binop_map[type(expr.op)]
                node_inputs = [left, right]
                inter_name = f"_inter_expr_{node_idx[0]}"
                node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
                node_idx[0] += 1

                self._node_counter += 1
                node_id = _make_node_id(module_name, f"{func_name}_expr_{node_idx[0]}")
                sig_parts = [f"{dt.name.lower()} {name}" for name, dt, _ in node_inputs]
                origin_sig = f"{' -> '.join(sig_parts)} -> array"

                reductions = [ReductionEntry(
                    rule="expr_decomposition",
                    description=f"Decomposed BinOp as {algo}",
                    original=algo,
                )]
                node = MathIRNode(
                    node_id=node_id,
                    origin_symbol=f"{module_name}.{func_name}",
                    origin_file=origin_file,
                    origin_line=0,
                    origin_commit=self.origin_commit,
                    origin_signature=origin_sig,
                    math_intent=f"Binary operation {algo}",
                    inputs=node_inputs,
                    outputs=node_outputs,
                    effects=[Effect.PURE],
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
                intermediates[f"_binop_{node_idx[0]}"] = inter_name
                symbol_table[f"_binop_{node_idx[0]}"] = (inter_name, Dtype.FLOAT64)
                return (inter_name, Dtype.FLOAT64, "array")
            return left

        if isinstance(expr, (ast.List, ast.Tuple)):
            resolved_elts = []
            for elt in expr.elts:
                r = self._decompose_expr(
                    elt, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                    module_name, func_name, origin_file, all_dep_ids, node_idx,
                )
                if isinstance(r, list):
                    resolved_elts.extend(r)
                else:
                    resolved_elts.append(r)
            return resolved_elts[0] if len(resolved_elts) == 1 else resolved_elts

        if isinstance(expr, ast.Call):
            call_target = self._resolve_call_target(expr)
            if call_target.startswith("np."):
                call_target = "numpy." + call_target[3:]
            if call_target in NUMPY_OP_MAP:
                algo = NUMPY_OP_MAP[call_target]
                node_inputs = []
                for arg in expr.args:
                    if isinstance(arg, ast.List):
                        for elt in arg.elts:
                            r = self._decompose_expr(
                                elt, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func_name, origin_file, all_dep_ids, node_idx,
                            )
                            if isinstance(r, list):
                                node_inputs.extend(r)
                            else:
                                node_inputs.append(r)
                    else:
                        r = self._decompose_expr(
                            arg, func_inputs, scalar_constants, intermediates,
                            symbol_table, existing_names, constant_assignments,
                            module_name, func_name, origin_file, all_dep_ids, node_idx,
                        )
                        if isinstance(r, list):
                            node_inputs.extend(r)
                        else:
                            node_inputs.append(r)
                for kw in expr.keywords:
                    if isinstance(kw.value, ast.Constant):
                        const_val = float(kw.value.value)
                        const_name = f"_const_{len(scalar_constants)}"
                        if const_name not in existing_names:
                            scalar_constants[const_name] = const_val
                            existing_names.add(const_name)
                        node_inputs.append((const_name, Dtype.FLOAT64, "scalar"))

                inter_name = f"_inter_expr_{node_idx[0]}"
                node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
                node_idx[0] += 1

                self._node_counter += 1
                node_id = _make_node_id(module_name, f"{func_name}_expr_{node_idx[0]}")
                sig_parts = [f"{dt.name.lower()} {name}" for name, dt, _ in node_inputs]
                origin_sig = f"{' -> '.join(sig_parts)} -> array"

                reductions = [ReductionEntry(
                    rule="expr_decomposition",
                    description=f"Decomposed {call_target} as {algo}",
                    original=call_target,
                )]
                node = MathIRNode(
                    node_id=node_id,
                    origin_symbol=f"{module_name}.{func_name}",
                    origin_file=origin_file,
                    origin_line=expr.lineno if hasattr(expr, 'lineno') else 0,
                    origin_commit=self.origin_commit,
                    origin_signature=origin_sig,
                    math_intent=f"Expression {algo}",
                    inputs=node_inputs,
                    outputs=node_outputs,
                    effects=[Effect.PURE],
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
                intermediates[f"_call_{call_target}_{node_idx[0]}"] = inter_name
                return (inter_name, Dtype.FLOAT64, "array")
            return self._resolve_arg_for_full_body(
                expr, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
            )

        if isinstance(expr, ast.Attribute) and expr.attr == "shape":
            if isinstance(expr.value, ast.Name) and expr.value.id in symbol_table:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.INT64, "scalar")

        if isinstance(expr, ast.Attribute) and expr.attr == "T":
            if isinstance(expr.value, ast.Name) and expr.value.id in symbol_table:
                name, dt = symbol_table[expr.value.id]
                return (name, dt, None)
            if isinstance(expr.value, ast.Name) and expr.value.id in intermediates:
                inter_name = intermediates[expr.value.id]
                return (inter_name, Dtype.FLOAT64, None)
            if isinstance(expr.value, ast.Call):
                inner = self._decompose_expr(
                    expr.value, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                    module_name, func_name, origin_file, all_dep_ids, node_idx,
                )
                if isinstance(inner, tuple) and not inner[0].startswith("_unresolved_"):
                    return (inner[0], inner[1], "array")

        if isinstance(expr, ast.Attribute):
            full_name = ""
            if isinstance(expr.value, ast.Name):
                full_name = f"{expr.value.id}.{expr.attr}"
            if full_name in ("np.pi", "numpy.pi"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.pi
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in ("np.e", "numpy.e"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.e
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in ("np.inf", "numpy.inf"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = float('inf')
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in _NUMPY_DTYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(expr, ast.Subscript):
            if isinstance(expr.slice, ast.Constant) and isinstance(expr.slice.value, int):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.INT64, "scalar")
            base = expr.value
            if isinstance(base, ast.Name) and base.id in symbol_table:
                name, dt = symbol_table[base.id]
                return (name, dt, "array")
            if isinstance(base, ast.Name) and base.id in intermediates:
                return (intermediates[base.id], Dtype.FLOAT64, "array")

        return self._resolve_arg_for_full_body(
            expr, func_inputs, scalar_constants, intermediates,
            symbol_table, existing_names, constant_assignments,
        )

    def _is_guard_clause(self, if_node: ast.If) -> bool:
        test = if_node.test
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1 and isinstance(test.ops[0], (ast.IsNot, ast.Is)):
                if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                    return True
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.Gt):
                return True
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.Lt):
                return True
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.NotEq):
                return True
        if isinstance(test, ast.Name):
            return True
        return False

    def _if_else_same_variable(self, if_node: ast.If) -> bool:
        if_names = set()
        for stmt in if_node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                if isinstance(stmt.targets[0], ast.Name):
                    if_names.add(stmt.targets[0].id)
        else_names = set()
        for stmt in if_node.orelse:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                if isinstance(stmt.targets[0], ast.Name):
                    else_names.add(stmt.targets[0].id)
        return bool(if_names & else_names)

    def _detect_loop_concat(self, func: ast.FunctionDef) -> tuple[ast.For, str, str, ast.Call] | None:
        list_vars = []
        for_node = None
        for stmt in func.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.List) and len(stmt.value.elts) == 0:
                if isinstance(stmt.targets[0], ast.Name):
                    list_vars.append(stmt.targets[0].id)
            if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call):
                if isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == "range":
                    for_node = stmt
        if not list_vars or for_node is None:
            return None
        append_vars = {}
        for stmt in ast.walk(for_node):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                    if isinstance(call.func.value, ast.Name) and call.func.value.id in list_vars:
                        if call.args:
                            append_vars[call.func.value.id] = call.args[0]
        for lv in list_vars:
            if lv not in append_vars:
                continue
            for stmt in func.body:
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    target = self._resolve_call_target(call)
                    if target.startswith("np."):
                        target = "numpy." + target[3:]
                    if target == "numpy.concatenate" and call.args:
                        arg0 = call.args[0]
                        if isinstance(arg0, ast.Name) and arg0.id == lv:
                            return (for_node, lv, for_node.target.id, append_vars[lv])
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            for s2 in func.body:
                                if isinstance(s2, ast.Assign) and isinstance(s2.targets[0], ast.Name):
                                    if s2.targets[0].id == elt.id and isinstance(s2.value, ast.Call):
                                        call = s2.value
                                        target = self._resolve_call_target(call)
                                        if target.startswith("np."):
                                            target = "numpy." + target[3:]
                                        if target == "numpy.concatenate" and call.args:
                                            arg0 = call.args[0]
                                            if isinstance(arg0, ast.Name) and arg0.id == lv:
                                                return (for_node, lv, for_node.target.id, append_vars[lv])
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    target = self._resolve_call_target(call)
                    if target.startswith("np."):
                        target = "numpy." + target[3:]
                    if target == "numpy.concatenate" and call.args:
                        arg0 = call.args[0]
                        if isinstance(arg0, ast.Name) and arg0.id == lv:
                            return (for_node, lv, for_node.target.id, append_vars[lv])
        return None

    def _decompose_full_body(
        self,
        func: ast.FunctionDef,
        module_name: str,
        func_inputs: list[tuple[str, Dtype, str]],
        func_outputs: list[tuple[str, Dtype, str]],
        effects: list[Effect],
        existing_names: set[str],
        scalar_constants: dict[str, float],
        local_funcs: dict[str, ast.FunctionDef],
    ) -> None:
        all_operations = self._collect_all_numpy_calls_from_body(func, local_funcs)
        if not all_operations:
            return

        symbol_table: dict[str, tuple[str, Dtype]] = {}
        for name, dt, shape in func_inputs:
            symbol_table[name] = (name, dt)

        constant_assignments: dict[str, float] = {}
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, (int, float)):
                        constant_assignments[target.id] = float(stmt.value.value)

        intermediates: dict[str, str] = {}
        all_reductions: list[ReductionEntry] = []
        all_dep_ids: list[str] = []
        origin_file = self.origin_file
        node_idx = 0

        processed_calls: set[int] = set()

        loop_concat = self._detect_loop_concat(func)
        if loop_concat is not None:
            for_node, list_var, loop_var, append_expr = loop_concat
            iter_count_arg = for_node.iter.args[0]
            iter_count = self._decompose_expr(
                iter_count_arg, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
                module_name, func.name, origin_file, all_dep_ids,
                [node_idx],
            )
            loop_body_ops = self._collect_all_numpy_calls_from_body(
                ast.FunctionDef(
                    name=func.name + "_loop_body",
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], defaults=[]),
                    body=[ast.Assign(
                        targets=[ast.Name(id="_loop_result", ctx=ast.Store())],
                        value=append_expr,
                        lineno=0,
                    )],
                    decorator_list=[],
                    returns=None,
                ),
                local_funcs,
            )
            loop_intermediates: dict[str, str] = {}
            loop_symbol_table = dict(symbol_table)
            loop_existing = set(existing_names)
            loop_scalar = dict(scalar_constants)
            loop_dep_ids: list[str] = []
            loop_node_idx = [0]
            last_inter = None
            loop_algos = []
            for op_target, op_call in loop_body_ops:
                algo = NUMPY_OP_MAP.get(op_target, "unknown")
                loop_algos.append(algo)
                op_inputs = []
                for arg in op_call.args:
                    if isinstance(arg, ast.Name) and arg.id == loop_var:
                        op_inputs.append((loop_var, Dtype.INT64, "loop_var"))
                    else:
                        r = self._decompose_expr(
                            arg, func_inputs, loop_scalar, loop_intermediates,
                            loop_symbol_table, loop_existing, constant_assignments,
                            module_name, func.name, origin_file, loop_dep_ids,
                            loop_node_idx,
                        )
                        if isinstance(r, list):
                            op_inputs.extend(r)
                        else:
                            op_inputs.append(r)
                inter_name = f"_inter_loop_{op_target}_{loop_node_idx[0]}"
                loop_node_idx[0] += 1
                last_inter = inter_name
                self._node_counter += 1
                nid = _make_node_id(module_name, f"{func.name}_loop_{op_target}")
                self.graph.add_node(MathIRNode(
                    node_id=nid,
                    origin_symbol=f"{module_name}.{func.name}",
                    origin_file=origin_file,
                    origin_line=for_node.lineno,
                    origin_commit=self.origin_commit,
                    origin_signature=f"loop body: {algo}",
                    math_intent=f"Loop body iteration: {algo}",
                    inputs=op_inputs,
                    outputs=[(inter_name, Dtype.FLOAT64, "array")],
                    effects=[Effect.PURE],
                    algorithm=algo,
                    reductions=[ReductionEntry(rule="loop_concat_body", description=f"Loop body: {algo}", original=op_target)],
                    nested_deps=list(loop_dep_ids),
                    stack_usage=256, heap_usage=None, reentrant=True,
                    dep_kind=DepKind.MATH_KERNEL,
                    scalar_constants=dict(loop_scalar),
                ))
                self.graph.entry_points.append(nid)
                loop_dep_ids.append(nid)
                loop_intermediates[op_target] = inter_name
                loop_symbol_table["_loop_result"] = (inter_name, Dtype.FLOAT64)

            node_idx += loop_node_idx[0]
            all_dep_ids.extend(loop_dep_ids)

            concat_node_id = _make_node_id(module_name, f"{func.name}_loop_concat")
            self._node_counter += 1
            self.graph.add_node(MathIRNode(
                node_id=concat_node_id,
                origin_symbol=f"{module_name}.{func.name}",
                origin_file=origin_file,
                origin_line=for_node.lineno,
                origin_commit=self.origin_commit,
                origin_signature=f"loop_concat({'+'.join(loop_algos)}) * {len(loop_body_ops)} body ops",
                math_intent=f"Loop-concatenated multi-head computation ({len(loop_body_ops)} body ops, N iterations)",
                inputs=[
                    (last_inter, Dtype.FLOAT64, "array"),
                    (iter_count[0] if isinstance(iter_count, tuple) else iter_count, iter_count[1] if isinstance(iter_count, tuple) else Dtype.INT64, "scalar"),
                ],
                outputs=list(func_outputs),
                effects=list(effects),
                algorithm="loop_concat",
                reductions=[ReductionEntry(
                    rule="loop_concat",
                    description=f"Loop-concatenated multi-head attention: {len(loop_body_ops)} ops per head",
                    original="loop_concat",
                )],
                nested_deps=list(all_dep_ids),
                stack_usage=256, heap_usage=None,
                reentrant=Effect.ALLOC not in effects,
                dep_kind=DepKind.MATH_KERNEL,
                scalar_constants=dict(scalar_constants),
            ))
            self.graph.entry_points.append(concat_node_id)
            return

        flat_stmts: list[ast.stmt] = []
        for stmt in func.body:
            if isinstance(stmt, ast.If):
                is_guard = self._is_guard_clause(stmt)
                if not stmt.orelse:
                    if is_guard:
                        for body_stmt in stmt.body:
                            flat_stmts.append(body_stmt)
                    else:
                        for body_stmt in stmt.body:
                            flat_stmts.append(body_stmt)
                else:
                    same_var = self._if_else_same_variable(stmt)
                    if same_var:
                        flat_stmts.append(stmt)
                    else:
                        for body_stmt in stmt.body:
                            flat_stmts.append(body_stmt)
                        for else_stmt in stmt.orelse:
                            flat_stmts.append(else_stmt)
            elif isinstance(stmt, (ast.Assign, ast.Return, ast.Expr)):
                flat_stmts.append(stmt)

        for stmt in flat_stmts:
            if isinstance(stmt, ast.If) and stmt.orelse:
                if_names = {}
                for s in stmt.body:
                    if isinstance(s, ast.Assign) and len(s.targets) == 1:
                        if isinstance(s.targets[0], ast.Name):
                            if_names[s.targets[0].id] = s.value
                else_names = {}
                for s in stmt.orelse:
                    if isinstance(s, ast.Assign) and len(s.targets) == 1:
                        if isinstance(s.targets[0], ast.Name):
                            else_names[s.targets[0].id] = s.value
                for var_name in if_names:
                    if var_name in else_names:
                        if_value = if_names[var_name]
                        else_value = else_names[var_name]
                        if isinstance(if_value, ast.Call) and isinstance(else_value, ast.Call):
                            if_target = self._resolve_call_target(if_value)
                            else_target = self._resolve_call_target(else_value)
                            if if_target.startswith("np."):
                                if_target = "numpy." + if_target[3:]
                            if else_target.startswith("np."):
                                else_target = "numpy." + else_target[3:]
                            if if_target in NUMPY_OP_MAP and else_target in NUMPY_OP_MAP:
                                node_idx_ref = [node_idx]
                                if_inputs = []
                                for arg in if_value.args:
                                    r = self._decompose_expr(
                                        arg, func_inputs, scalar_constants, intermediates,
                                        symbol_table, existing_names, constant_assignments,
                                        module_name, func.name, origin_file, all_dep_ids,
                                        node_idx_ref,
                                    )
                                    if isinstance(r, list):
                                        if_inputs.extend(r)
                                    else:
                                        if_inputs.append(r)
                                else_inputs = []
                                for arg in else_value.args:
                                    r = self._decompose_expr(
                                        arg, func_inputs, scalar_constants, intermediates,
                                        symbol_table, existing_names, constant_assignments,
                                        module_name, func.name, origin_file, all_dep_ids,
                                        node_idx_ref,
                                    )
                                    if isinstance(r, list):
                                        else_inputs.extend(r)
                                    else:
                                        else_inputs.append(r)
                                inter_if = f"_inter_if_{var_name}_{node_idx}"
                                node_idx += 1
                                self._node_counter += 1
                                if_id = _make_node_id(module_name, f"{func.name}_if_{var_name}")
                                if_sig = " -> ".join([f"{dt.name.lower()} {n}" for n, dt, _ in if_inputs]) + " -> array"
                                self.graph.add_node(MathIRNode(
                                    node_id=if_id,
                                    origin_symbol=f"{module_name}.{func.name}",
                                    origin_file=origin_file,
                                    origin_line=stmt.lineno,
                                    origin_commit=self.origin_commit,
                                    origin_signature=if_sig,
                                    math_intent=f"If-branch for '{var_name}'",
                                    inputs=if_inputs,
                                    outputs=[(inter_if, Dtype.FLOAT64, "array")],
                                    effects=[Effect.PURE],
                                    algorithm=NUMPY_OP_MAP[if_target],
                                    reductions=[ReductionEntry(rule="if_branch", description=f"If-branch {if_target}", original=if_target)],
                                    nested_deps=list(all_dep_ids),
                                    stack_usage=256, heap_usage=None, reentrant=True,
                                    dep_kind=DepKind.MATH_KERNEL,
                                    scalar_constants=dict(scalar_constants),
                                ))
                                self.graph.entry_points.append(if_id)
                                all_dep_ids.append(if_id)
                                inter_else = f"_inter_else_{var_name}_{node_idx}"
                                node_idx += 1
                                self._node_counter += 1
                                else_id = _make_node_id(module_name, f"{func.name}_else_{var_name}")
                                else_sig = " -> ".join([f"{dt.name.lower()} {n}" for n, dt, _ in else_inputs]) + " -> array"
                                self.graph.add_node(MathIRNode(
                                    node_id=else_id,
                                    origin_symbol=f"{module_name}.{func.name}",
                                    origin_file=origin_file,
                                    origin_line=stmt.lineno,
                                    origin_commit=self.origin_commit,
                                    origin_signature=else_sig,
                                    math_intent=f"Else-branch for '{var_name}'",
                                    inputs=else_inputs,
                                    outputs=[(inter_else, Dtype.FLOAT64, "array")],
                                    effects=[Effect.PURE],
                                    algorithm=NUMPY_OP_MAP[else_target],
                                    reductions=[ReductionEntry(rule="else_branch", description=f"Else-branch {else_target}", original=else_target)],
                                    nested_deps=list(all_dep_ids),
                                    stack_usage=256, heap_usage=None, reentrant=True,
                                    dep_kind=DepKind.MATH_KERNEL,
                                    scalar_constants=dict(scalar_constants),
                                ))
                                self.graph.entry_points.append(else_id)
                                all_dep_ids.append(else_id)
                                cond_name = f"_cond_{var_name}_{node_idx}"
                                node_idx += 1
                                self._node_counter += 1
                                where_id = _make_node_id(module_name, f"{func.name}_where_{var_name}")
                                where_sig = f"double {cond_name}, double {inter_if}, double {inter_else} -> array"
                                self.graph.add_node(MathIRNode(
                                    node_id=where_id,
                                    origin_symbol=f"{module_name}.{func.name}",
                                    origin_file=origin_file,
                                    origin_line=stmt.lineno,
                                    origin_commit=self.origin_commit,
                                    origin_signature=where_sig,
                                    math_intent=f"Conditional select for '{var_name}'",
                                    inputs=[
                                        (cond_name, Dtype.FLOAT64, "scalar"),
                                        (inter_if, Dtype.FLOAT64, "array"),
                                        (inter_else, Dtype.FLOAT64, "array"),
                                    ],
                                    outputs=[(f"_inter_{var_name}", Dtype.FLOAT64, "array")],
                                    effects=[Effect.PURE],
                                    algorithm="element_where",
                                    reductions=[ReductionEntry(rule="conditional_select", description=f"element_where for {var_name}", original="numpy.where")],
                                    nested_deps=list(all_dep_ids),
                                    stack_usage=256, heap_usage=None, reentrant=True,
                                    dep_kind=DepKind.MATH_KERNEL,
                                    scalar_constants=dict(scalar_constants),
                                ))
                                self.graph.entry_points.append(where_id)
                                all_dep_ids.append(where_id)
                                symbol_table[var_name] = (f"_inter_{var_name}", Dtype.FLOAT64)
                continue

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target_name = stmt.targets[0].id if isinstance(stmt.targets[0], ast.Name) else None
                if target_name is None:
                    continue

                if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, float)):
                    const_val = float(stmt.value.value)
                    const_name = f"_const_{len(scalar_constants)}"
                    dt = Dtype.FLOAT64 if isinstance(stmt.value.value, float) else Dtype.INT64
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                    symbol_table[target_name] = (const_name, dt)
                    continue

                if isinstance(stmt.value, ast.Name) and stmt.value.id in symbol_table:
                    symbol_table[target_name] = symbol_table[stmt.value.id]
                    continue

                if isinstance(stmt.value, ast.Attribute) and stmt.value.attr == "shape":
                    if isinstance(stmt.targets[0], ast.Tuple):
                        shape_var = stmt.value.value.id if isinstance(stmt.value.value, ast.Name) else None
                        if shape_var:
                            for i, elt in enumerate(stmt.targets[0].elts):
                                if isinstance(elt, ast.Name):
                                    dim_name = f"_dim_{shape_var}_{i}"
                                    const_name = f"_const_{len(scalar_constants)}"
                                    if const_name not in existing_names:
                                        scalar_constants[const_name] = float(i)
                                        existing_names.add(const_name)
                                    symbol_table[elt.id] = (const_name, Dtype.INT64)
                        continue
                    elif isinstance(stmt.targets[0], ast.Name):
                        shape_var = stmt.value.value.id if isinstance(stmt.value.value, ast.Name) else None
                        if shape_var:
                            const_name = f"_const_{len(scalar_constants)}"
                            if const_name not in existing_names:
                                scalar_constants[const_name] = 0.0
                                existing_names.add(const_name)
                            symbol_table[target_name] = (const_name, Dtype.INT64)
                        continue

                if isinstance(stmt.value, ast.BinOp):
                    left = self._decompose_expr(
                        stmt.value.left, func_inputs, scalar_constants, intermediates,
                        symbol_table, existing_names, constant_assignments,
                        module_name, func.name, origin_file, all_dep_ids,
                        [node_idx],
                    )
                    right = self._decompose_expr(
                        stmt.value.right, func_inputs, scalar_constants, intermediates,
                        symbol_table, existing_names, constant_assignments,
                        module_name, func.name, origin_file, all_dep_ids,
                        [node_idx],
                    )
                    binop_map = {
                        ast.Add: "element_add", ast.Sub: "element_sub",
                        ast.Mult: "element_mul", ast.Div: "element_div",
                        ast.Pow: "element_power", ast.Mod: "element_mod",
                        ast.FloorDiv: "element_div",
                    }
                    if type(stmt.value.op) in binop_map:
                        algo = binop_map[type(stmt.value.op)]
                        node_inputs = [left, right]
                        inter_name = f"_inter_{target_name}"
                        node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
                        self._node_counter += 1
                        nid = _make_node_id(module_name, f"{func.name}_{target_name}")
                        sig_parts = [f"{dt.name.lower()} {n}" for n, dt, _ in node_inputs]
                        origin_sig = f"{' -> '.join(sig_parts)} -> array"
                        self.graph.add_node(MathIRNode(
                            node_id=nid,
                            origin_symbol=f"{module_name}.{func.name}",
                            origin_file=origin_file,
                            origin_line=stmt.lineno,
                            origin_commit=self.origin_commit,
                            origin_signature=origin_sig,
                            math_intent=f"Assignment '{target_name}' implementing {algo}",
                            inputs=node_inputs,
                            outputs=node_outputs,
                            effects=[Effect.PURE],
                            algorithm=algo,
                            reductions=[ReductionEntry(rule="binop_decomposition", description=f"Decomposed BinOp as {algo}", original=algo)],
                            nested_deps=list(all_dep_ids),
                            stack_usage=256, heap_usage=None, reentrant=True,
                            dep_kind=DepKind.MATH_KERNEL,
                            scalar_constants=dict(scalar_constants),
                        ))
                        self.graph.entry_points.append(nid)
                        all_dep_ids.append(nid)
                        node_idx += 1
                        symbol_table[target_name] = (inter_name, Dtype.FLOAT64)
                    continue

                if isinstance(stmt.value, ast.Call):
                    call_target = self._resolve_call_target(stmt.value)
                    if call_target.startswith("np."):
                        call_target = "numpy." + call_target[3:]
                    if call_target in NUMPY_OP_MAP:
                        stmt_id = id(stmt.value)
                        if stmt_id in processed_calls:
                            continue
                        processed_calls.add(stmt_id)

                        algo = NUMPY_OP_MAP[call_target]

                        node_idx_ref = [node_idx]
                        node_inputs: list[tuple[str, Dtype, str]] = []
                        for arg in stmt.value.args:
                            if isinstance(arg, (ast.List, ast.Tuple)):
                                for elt in arg.elts:
                                    resolved = self._decompose_expr(
                                        elt, func_inputs, scalar_constants, intermediates,
                                        symbol_table, existing_names, constant_assignments,
                                        module_name, func.name, origin_file, all_dep_ids,
                                        node_idx_ref,
                                    )
                                    if isinstance(resolved, list):
                                        node_inputs.extend(resolved)
                                    else:
                                        node_inputs.append(resolved)
                            else:
                                resolved = self._decompose_expr(
                                    arg, func_inputs, scalar_constants, intermediates,
                                    symbol_table, existing_names, constant_assignments,
                                    module_name, func.name, origin_file, all_dep_ids,
                                    node_idx_ref,
                                )
                                if isinstance(resolved, list):
                                    node_inputs.extend(resolved)
                                else:
                                    node_inputs.append(resolved)
                        for kw in stmt.value.keywords:
                            if isinstance(kw.value, ast.Constant):
                                const_val = float(kw.value.value)
                                const_name = f"_const_{len(scalar_constants)}"
                                if const_name not in existing_names:
                                    scalar_constants[const_name] = const_val
                                    existing_names.add(const_name)
                                node_inputs.append((const_name, Dtype.FLOAT64, "scalar"))

                        inter_name = f"_inter_{target_name}"
                        node_outputs = [(inter_name, Dtype.FLOAT64, "array")]
                        intermediates[call_target] = inter_name
                        intermediates[f"{call_target}_{node_idx}"] = inter_name
                        symbol_table[target_name] = (inter_name, Dtype.FLOAT64)

                        reductions = [ReductionEntry(
                            rule="numpy_op_extraction",
                            description=f"Extracted {call_target} as {algo} kernel",
                            original=call_target,
                        )]
                        all_reductions.extend(reductions)

                        sig_parts = []
                        for name, dt, _ in node_inputs:
                            sig_parts.append(f"{dt.name.lower()} {name}")
                        origin_sig = f"{' -> '.join(sig_parts)} -> array"

                        self._node_counter += 1
                        node_id = _make_node_id(module_name, f"{func.name}_{target_name}")

                        math_intent = f"Assignment '{target_name}' implementing {algo}"

                        node = MathIRNode(
                            node_id=node_id,
                            origin_symbol=f"{module_name}.{func.name}",
                            origin_file=origin_file,
                            origin_line=stmt.lineno,
                            origin_commit=self.origin_commit,
                            origin_signature=origin_sig,
                            math_intent=math_intent,
                            inputs=node_inputs,
                            outputs=node_outputs,
                            effects=[Effect.PURE],
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
                        node_idx += 1
                    else:
                        if stmt.value.args:
                            resolved = self._resolve_arg_for_full_body(
                                stmt.value.args[0], func_inputs, scalar_constants,
                                intermediates, symbol_table, existing_names,
                                constant_assignments,
                            )
                            if not resolved[0].startswith("_unresolved_"):
                                symbol_table[target_name] = (resolved[0], resolved[1])

        return_stmt = None
        for stmt in func.body:
            if isinstance(stmt, ast.Return):
                return_stmt = stmt
                break
            if isinstance(stmt, ast.If):
                for inner in stmt.body:
                    if isinstance(inner, ast.Return):
                        return_stmt = inner
                        break
                if return_stmt is not None:
                    break
                for inner in stmt.orelse:
                    if isinstance(inner, ast.Return):
                        return_stmt = inner
                        break
                if return_stmt is not None:
                    break

        if return_stmt is not None and return_stmt.value is not None:
            if isinstance(return_stmt.value, ast.Call):
                call_target = self._resolve_call_target(return_stmt.value)
                if call_target.startswith("np."):
                    call_target = "numpy." + call_target[3:]
                if call_target in NUMPY_OP_MAP:
                    algo = NUMPY_OP_MAP[call_target]

                    node_idx_ref = [node_idx]
                    node_inputs: list[tuple[str, Dtype, str]] = []
                    for arg in return_stmt.value.args:
                        if isinstance(arg, (ast.List, ast.Tuple)):
                            for elt in arg.elts:
                                resolved = self._decompose_expr(
                                    elt, func_inputs, scalar_constants, intermediates,
                                    symbol_table, existing_names, constant_assignments,
                                    module_name, func.name, origin_file, all_dep_ids,
                                    node_idx_ref,
                                )
                                if isinstance(resolved, list):
                                    node_inputs.extend(resolved)
                                else:
                                    node_inputs.append(resolved)
                        else:
                            resolved = self._decompose_expr(
                                arg, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func.name, origin_file, all_dep_ids,
                                node_idx_ref,
                            )
                            if isinstance(resolved, list):
                                node_inputs.extend(resolved)
                            else:
                                node_inputs.append(resolved)
                    for kw in return_stmt.value.keywords:
                        if isinstance(kw.value, ast.Constant):
                            const_val = float(kw.value.value)
                            const_name = f"_const_{len(scalar_constants)}"
                            if const_name not in existing_names:
                                scalar_constants[const_name] = const_val
                                existing_names.add(const_name)
                            node_inputs.append((const_name, Dtype.FLOAT64, "scalar"))

                    node_outputs = list(func_outputs)
                    reductions = [ReductionEntry(
                        rule="numpy_op_extraction",
                        description=f"Extracted {call_target} as {algo} kernel",
                        original=call_target,
                    )]
                    all_reductions.extend(reductions)

                    sig_parts = []
                    for name, dt, _ in node_inputs:
                        sig_parts.append(f"{dt.name.lower()} {name}")
                    origin_sig = f"{' -> '.join(sig_parts)} -> {func_outputs[0][1].name.lower()}"

                    self._node_counter += 1
                    node_id = _make_node_id(module_name, func.name)

                    math_intent = f"Return of '{func.name}' implementing {algo}"

                    node = MathIRNode(
                        node_id=node_id,
                        origin_symbol=f"{module_name}.{func.name}",
                        origin_file=origin_file,
                        origin_line=return_stmt.lineno,
                        origin_commit=self.origin_commit,
                        origin_signature=origin_sig,
                        math_intent=math_intent,
                        inputs=node_inputs,
                        outputs=node_outputs,
                        effects=effects,
                        algorithm=algo,
                        reductions=reductions,
                        nested_deps=list(all_dep_ids),
                        stack_usage=256,
                        heap_usage=None,
                        reentrant=Effect.ALLOC not in effects,
                        dep_kind=DepKind.MATH_KERNEL,
                        scalar_constants=dict(scalar_constants),
                    )

                    self.graph.add_node(node)
                    self.graph.entry_points.append(node_id)
                    all_dep_ids.append(node_id)

    def _resolve_arg_for_full_body(self, arg_node: ast.expr,
                                   func_inputs: list[tuple[str, Dtype, str]],
                                   scalar_constants: dict[str, float],
                                   intermediates: dict[str, str],
                                   symbol_table: dict[str, tuple[str, Dtype]],
                                   existing_names: set[str],
                                   constant_assignments: dict[str, float] | None = None) -> tuple[str, Dtype, str | None]:
        if isinstance(arg_node, ast.Name):
            if arg_node.id in symbol_table:
                name, dt = symbol_table[arg_node.id]
                return (name, dt, None)
            if arg_node.id in existing_names:
                for name, dt, shape in func_inputs:
                    if name == arg_node.id:
                        return (name, dt, None)
                return (arg_node.id, Dtype.FLOAT64, None)
            if arg_node.id in intermediates:
                inter_name = intermediates[arg_node.id]
                return (inter_name, Dtype.FLOAT64, None)
            if constant_assignments and arg_node.id in constant_assignments:
                const_val = constant_assignments[arg_node.id]
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if arg_node.id in _PY_BUILTIN_TYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, (int, float, complex)):
            const_val = float(arg_node.value.real) if isinstance(arg_node.value, complex) else float(arg_node.value)
            const_name = f"_const_{len(scalar_constants)}"
            if const_name not in existing_names:
                dt = Dtype.FLOAT64 if isinstance(arg_node.value, float) else Dtype.INT64
                scalar_constants[const_name] = const_val
                existing_names.add(const_name)
            return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Call):
            call_target = self._resolve_call_target(arg_node)
            if call_target in intermediates:
                inter_name = intermediates[call_target]
                return (inter_name, Dtype.FLOAT64, None)
            if call_target in ("numpy.transpose", "numpy.T"):
                if arg_node.args:
                    inner = arg_node.args[0]
                    resolved = self._resolve_arg_for_full_body(
                        inner, func_inputs, scalar_constants, intermediates,
                        symbol_table, existing_names, constant_assignments,
                    )
                    return (resolved[0], resolved[1], resolved[2])
                if isinstance(arg_node.func, ast.Attribute) and isinstance(arg_node.func.value, ast.Name):
                    base_name = arg_node.func.value.id
                    if base_name in symbol_table:
                        name, dt = symbol_table[base_name]
                        return (name, dt, None)
            if call_target not in NUMPY_OP_MAP:
                if arg_node.args:
                    resolved = self._resolve_arg_for_full_body(
                        arg_node.args[0], func_inputs, scalar_constants, intermediates,
                        symbol_table, existing_names, constant_assignments,
                    )
                    return (resolved[0], resolved[1], resolved[2])

        if isinstance(arg_node, ast.Attribute) and arg_node.attr == "shape":
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in symbol_table:
                dim_name = f"_dim_{arg_node.value.id}"
                const_val = 0.0
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.INT64, "scalar")

        if isinstance(arg_node, ast.Attribute) and arg_node.attr == "T":
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in symbol_table:
                name, dt = symbol_table[arg_node.value.id]
                return (name, dt, None)
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in intermediates:
                inter_name = intermediates[arg_node.value.id]
                return (inter_name, Dtype.FLOAT64, None)

        if isinstance(arg_node, ast.Attribute):
            full_name = ""
            if isinstance(arg_node.value, ast.Name):
                full_name = f"{arg_node.value.id}.{arg_node.attr}"
            if full_name in ("np.pi", "numpy.pi"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.pi
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in ("np.e", "numpy.e"):
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    import math
                    scalar_constants[const_name] = math.e
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if full_name in _NUMPY_DTYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")

        if isinstance(arg_node, ast.Subscript):
            if isinstance(arg_node.value, ast.Attribute) and arg_node.value.attr == "shape":
                if isinstance(arg_node.value.value, ast.Name):
                    if isinstance(arg_node.slice, ast.Constant) and isinstance(arg_node.slice.value, int):
                        dim_idx = arg_node.slice.value
                        dim_name = f"_dim_{arg_node.value.value.id}_{dim_idx}"
                        const_val = 0.0
                        const_name = f"_const_{len(scalar_constants)}"
                        if const_name not in existing_names:
                            scalar_constants[const_name] = const_val
                            existing_names.add(const_name)
                        return (const_name, Dtype.INT64, "scalar")
            if isinstance(arg_node.value, ast.Name):
                if arg_node.value.id in symbol_table:
                    name, dt = symbol_table[arg_node.value.id]
                    return (name, dt, "array")
                if arg_node.value.id in intermediates:
                    return (intermediates[arg_node.value.id], Dtype.FLOAT64, "array")

        if isinstance(arg_node, ast.UnaryOp) and isinstance(arg_node.op, ast.USub):
            if isinstance(arg_node.operand, ast.Constant) and isinstance(arg_node.operand.value, (int, float, complex)):
                const_val = -float(arg_node.operand.value.real) if isinstance(arg_node.operand.value, complex) else -float(arg_node.operand.value)
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = const_val
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            if isinstance(arg_node.operand, ast.Name):
                if arg_node.operand.id in symbol_table:
                    name, dt = symbol_table[arg_node.operand.id]
                    neg_name = f"_const_{len(scalar_constants)}"
                    if neg_name not in existing_names:
                        scalar_constants[neg_name] = -1.0
                        existing_names.add(neg_name)
                    return (neg_name, Dtype.FLOAT64, "scalar")
                if arg_node.operand.id in constant_assignments:
                    const_val = -constant_assignments[arg_node.operand.id]
                    const_name = f"_const_{len(scalar_constants)}"
                    if const_name not in existing_names:
                        scalar_constants[const_name] = const_val
                        existing_names.add(const_name)
                    return (const_name, Dtype.FLOAT64, "scalar")
            return (f"_unresolved_{len(existing_names)}", Dtype.FLOAT64, None)

        if isinstance(arg_node, ast.BinOp):
            binop_map = {
                ast.Add: "element_add", ast.Sub: "element_sub",
                ast.Mult: "element_mul", ast.Div: "element_div",
                ast.Pow: "element_power", ast.Mod: "element_mod",
            }
            if type(arg_node.op) in binop_map:
                left = self._resolve_arg_for_full_body(
                    arg_node.left, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                )
                right = self._resolve_arg_for_full_body(
                    arg_node.right, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                )
                return (f"_unresolved_{len(existing_names)}", Dtype.FLOAT64, None)

        return (f"_unresolved_{len(existing_names)}", Dtype.FLOAT64, None)

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
            self._collect_calls_from_expr(arg, operations, local_funcs)

        if target in NUMPY_OP_MAP:
            operations.append((target, call_node))

    def _collect_calls_from_expr(self, expr: ast.expr, operations: list[tuple[str, ast.Call]],
                                 local_funcs: dict[str, ast.FunctionDef]) -> None:
        if isinstance(expr, ast.Call):
            inner_target = ""
            if isinstance(expr.func, ast.Attribute):
                inner_target = _get_qualified_name(expr.func)
            elif isinstance(expr.func, ast.Name):
                inner_target = expr.func.id
            if inner_target.startswith("np."):
                inner_target = "numpy." + inner_target[3:]
            if inner_target in NUMPY_OP_MAP:
                self._collect_numpy_calls(expr, operations, local_funcs)
            elif inner_target in local_funcs:
                local_def = local_funcs[inner_target]
                for stmt in ast.walk(local_def):
                    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                        self._collect_numpy_calls(stmt.value, operations, local_funcs)
            else:
                for sub_arg in expr.args:
                    self._collect_calls_from_expr(sub_arg, operations, local_funcs)
        elif isinstance(expr, ast.BinOp):
            self._collect_calls_from_expr(expr.left, operations, local_funcs)
            self._collect_calls_from_expr(expr.right, operations, local_funcs)
        elif isinstance(expr, ast.UnaryOp):
            self._collect_calls_from_expr(expr.operand, operations, local_funcs)
        elif isinstance(expr, ast.Compare):
            self._collect_calls_from_expr(expr.left, operations, local_funcs)
            for comparator in expr.comparators:
                self._collect_calls_from_expr(comparator, operations, local_funcs)
        elif isinstance(expr, ast.IfExp):
            self._collect_calls_from_expr(expr.test, operations, local_funcs)
            self._collect_calls_from_expr(expr.body, operations, local_funcs)
            self._collect_calls_from_expr(expr.orelse, operations, local_funcs)
        elif isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Call):
            self._collect_calls_from_expr(expr.value, operations, local_funcs)

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
