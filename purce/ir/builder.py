from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

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

SCALAR_OUTPUT_CALLS: frozenset[str] = frozenset({
    "numpy.sum", "numpy.mean", "numpy.max", "numpy.min",
    "numpy.var", "numpy.prod", "numpy.argmax", "numpy.argmin",
    "numpy.any", "numpy.all", "numpy.linalg.norm",
})

SCALAR_OUTPUT_ALGOS: frozenset[str] = frozenset({
    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
    "reduce_var", "reduce_prod", "reduce_argmax", "reduce_argmin",
    "reduce_any", "reduce_all", "linalg_norm",
})


def _make_node_id(prefix: str, symbol: str) -> str:
    base = f"{prefix}.{symbol}"
    h = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()[:8]
    clean = symbol.replace(".", "_").replace("-", "_")
    return f"{prefix}.{clean}_{h}"


def _resolve_dtype_annotation(annotation: ast.expr | None) -> Dtype:
    if annotation is None:
        return Dtype.FLOAT64
    if isinstance(annotation, ast.Name):
        if annotation.id == "int":
            return Dtype.INT64
        if annotation.id == "float":
            return Dtype.FLOAT64
        if annotation.id == "bool":
            return Dtype.BOOL
        if annotation.id == "complex":
            return Dtype.COMPLEX128
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


def _substitute_into(node, substitution: dict[str, ast.expr]):
    """Recursively replace ``ast.Name`` nodes matching ``substitution``.

    Used for interprocedural inlining: a callee's parameters (as Name nodes)
    are replaced by the caller's actual argument expressions.  Substituted
    expressions are copied so a caller argument used for several parameters
    does not share AST nodes across multiple locations.
    """
    if isinstance(node, ast.Name):
        if node.id in substitution:
            return ast.copy_location(ast.parse(
                ast.unparse(substitution[node.id]), mode="eval"
            ).body, node)
        return node
    if isinstance(node, ast.Constant):
        return node
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.expr):
            setattr(node, field, _substitute_into(value, substitution))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, ast.expr):
                    value[i] = _substitute_into(item, substitution)
    return node


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
        self._scalar_names: set[str] = set()
        self._local_funcs: dict[str, ast.FunctionDef] = {}
        # Interprocedural call edges: module-local function name -> set of
        # module-local callee names it calls (gathered from the ORIGINAL AST
        # before inlining, so the call graph reflects the source).
        self._callees: dict[str, set[str]] = {}
        # Recursion guard for local-call inlining.
        self._inline_visiting: set[str] = set()

    def _shape_of(self, name: str) -> str:
        """Resolve the shape of a known local name (scalar or array)."""
        return "scalar" if name in self._scalar_names else "array"

    def _collect_scalar_locals(self, func: ast.FunctionDef) -> None:
        """Pre-pass: mark locals that are provably scalar-shaped.

        Sources: scalar-annotated params, constant assignments, .shape/.size/.ndim
        accesses, reduce-family calls, and fixed-point propagation through
        scalar-only expressions.
        """
        scalars: set[str] = set()
        for arg in func.args.args:
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id in _PY_BUILTIN_TYPE_NAMES:
                scalars.add(arg.arg)
            defaults_offset = len(func.args.args) - len(func.args.defaults)
            default_idx = func.args.args.index(arg) - defaults_offset
            if default_idx >= 0 and default_idx < len(func.args.defaults):
                def_val = func.args.defaults[default_idx]
                if isinstance(def_val, ast.Constant) and isinstance(def_val.value, (int, float)):
                    scalars.add(arg.arg)

        changed = True
        while changed:
            changed = False
            for stmt in func.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id in scalars:
                    continue
                value = stmt.value
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) or isinstance(value, ast.Attribute) and value.attr in ("shape", "size", "ndim") or isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute) \
                        and value.value.attr in ("shape", "size"):
                    scalars.add(target.id)
                    changed = True
                elif isinstance(value, ast.Call):
                    call_target = self._resolve_call_target(value)
                    if call_target.startswith("np."):
                        call_target = "numpy." + call_target[3:]
                    if call_target in SCALAR_OUTPUT_CALLS:
                        scalars.add(target.id)
                        changed = True
                    else:
                        leaf_names = {
                            node.id for node in ast.walk(value)
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        }
                        if leaf_names and leaf_names <= scalars:
                            scalars.add(target.id)
                            changed = True
                else:
                    leaf_names = {
                        node.id for node in ast.walk(value)
                        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    }
                    if leaf_names and leaf_names <= scalars:
                        scalars.add(target.id)
                        changed = True

        self._scalar_names |= scalars

    @property
    def diagnostics(self) -> list[dict]:
        return list(self._diagnostics)

    def from_python_ast(self, tree: ast.Module, module_name: str = "module") -> MathIRGraph:
        local_funcs: dict[str, ast.FunctionDef] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                local_funcs[node.name] = node

        self._local_funcs = local_funcs
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                self._process_function(node, module_name, local_funcs)
        self._finalize_call_graph(module_name)
        return self.graph

    def _finalize_call_graph(self, module_name: str) -> None:
        """Wire interprocedural edges and set meaningful entry points.

        Two-phase build so that call edges can reference callee node ids that
        only exist once every function has been processed:

        * Interprocedural edges: if function F calls module-local function G,
          every node belonging to F gains a nested_dep edge to G's root node.
          The slicer's reachability then correctly keeps callees alive and the
          call graph is a real graph (previously there were no cross-function
          edges at all, so reachability-based DCE could neither keep callees
          nor prune orphans soundly).

        * Entry points: the root node of each top-level function — the last
          node emitted for a function (which, through nested_deps chains,
          reaches every earlier decomposition step).  Decomposition
          intermediates are no longer treated as independent entry points,
          so the slicer actually performs reachability instead of treating
          every node as a liveness root.
        """
        if not self._local_funcs:
            return

        # Group nodes by the function that produced them (origin_symbol is
        # "<module>.<funcname>" for both simple and decomposed nodes).
        func_nodes: dict[str, list[str]] = {}
        for nid, node in self.graph.nodes.items():
            func_nodes.setdefault(node.origin_symbol, []).append(nid)

        # The root of a function is the last node created for it.  The current
        # entry_points list was appended in creation order, so the last id per
        # origin_symbol is the function's root.
        func_roots: dict[str, str] = {}
        for nid in self.graph.entry_points:
            node = self.graph.nodes.get(nid)
            if node is not None:
                func_roots[node.origin_symbol] = nid

        # Wire callee roots into every caller node. Iterate callees in sorted
        # order: `self._callees` stores a set, so unordered iteration would make
        # the emitted `nested_deps` (and generated provenance) depend on the
        # per-process PYTHONHASHSEED — breaking the reproducible corpus gate.
        for caller, callees in self._callees.items():
            caller_sym = f"{module_name}.{caller}"
            for callee in sorted(callees):
                callee_root = func_roots.get(f"{module_name}.{callee}")
                if callee_root is None:
                    continue
                for nid in func_nodes.get(caller_sym, []):
                    node = self.graph.nodes.get(nid)
                    if node is not None and callee_root not in node.nested_deps:
                        node.nested_deps.append(callee_root)

        # Entry points = one root per top-level function, deterministically
        # ordered by source symbol.
        self.graph.entry_points = [
            func_roots[sym] for sym in sorted(func_roots)
        ]

    def _register_literal_elts(self, arg_node: ast.expr, inputs: list[tuple[str, Dtype, str]],
                               existing_names: set[str], scalar_constants: dict[str, float],
                               constant_assignments: dict[str, float],
                               scalar_counter: int) -> int:
        """Register literal elements of a List/Tuple call argument in order.

        Constants become scalar constants; named elements become inputs with
        their known shape. Nested lists are flattened in order.
        """
        for elt in arg_node.elts:
            if isinstance(elt, (ast.List, ast.Tuple)):
                scalar_counter = self._register_literal_elts(
                    elt, inputs, existing_names, scalar_constants,
                    constant_assignments, scalar_counter,
                )
            elif isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                const_val = float(elt.value)
                const_name = f"_const_{scalar_counter}"
                scalar_counter += 1
                if const_name not in existing_names:
                    dt = Dtype.FLOAT64 if isinstance(elt.value, float) else Dtype.INT64
                    inputs.append((const_name, dt, "scalar"))
                    existing_names.add(const_name)
                    scalar_constants[const_name] = const_val
            elif isinstance(elt, ast.Name) and elt.id in constant_assignments:
                const_val = constant_assignments[elt.id]
                const_name = f"_const_{scalar_counter}"
                scalar_counter += 1
                if const_name not in existing_names:
                    dt = Dtype.FLOAT64 if isinstance(const_val, float) else Dtype.INT64
                    inputs.append((const_name, dt, "scalar"))
                    existing_names.add(const_name)
                    scalar_constants[const_name] = const_val
            elif isinstance(elt, ast.Name):
                if elt.id not in existing_names:
                    inputs.append((elt.id, Dtype.FLOAT64, self._shape_of(elt.id)))
                    existing_names.add(elt.id)
        return scalar_counter

    def _algorithm_for(self, target: str, call_node: ast.Call | None = None) -> str:
        """Map a numpy call target to an algorithm, with shape-driven refinements.

        Known limitation: np.diag(x) on a 1-D array variable compiles as
        matrix_diag (extract diagonal), matching the 2-D case. NumPy treats
        diag(1-D) as constructing a diagonal matrix, but the input's ndim is
        not statically known here; only literal list arguments are provably
        1-D and correctly compile to matrix_diag_from.
        """
        algo = NUMPY_OP_MAP.get(target, "unknown")
        if call_node is not None and target == "numpy.diag" and call_node.args:
            if isinstance(call_node.args[0], (ast.List, ast.Tuple)):
                algo = "matrix_diag_from"
        return algo

    def _output_shape_for(self, algo: str,
                          inputs: list[tuple[str, Dtype, str]]) -> str:
        """Shape of a kernel output: reduce* and scalar-elementwise are scalar.

        Allocation/construction kernels (alloc_*, array_*, matrix_*,
        linalg_*) always produce arrays even when every input is scalar.
        """
        if algo in SCALAR_OUTPUT_ALGOS:
            return "scalar"
        if algo.startswith("element_") and all(s == "scalar" for _, _, s in inputs):
            return "scalar"
        return "array"

    def _process_function(self, func: ast.FunctionDef, module_name: str,
                          local_funcs: dict[str, ast.FunctionDef] | None = None) -> None:
        local_funcs = local_funcs or {}

        self._record_local_callees(func, local_funcs)
        func = self._inline_local_calls(func, local_funcs)

        has_numpy = False
        call_targets: list[str] = []
        effects = [Effect.PURE]

        self._scalar_names.clear()
        self._collect_scalar_locals(func)

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

                # Only true I/O and dynamic-eval constructs are side-effecting
                # enough to require reclassification.  getattr/setattr are NOT
                # I/O: they are dynamic dispatch and must NOT delete a kernel
                # that contains a numpy op (dynamic calls are a known analysis
                # limitation, treated conservatively).
                if target in ("eval", "exec", "open", "print"):
                    effects.append(Effect.IO)
                elif target in ("getattr", "setattr"):
                    # Dynamic dispatch: conservatively treat as TEMPORAL so the
                    # slicer keeps the node (unresolved calls are a barrier).
                    effects.append(Effect.TEMPORAL)

        if not has_numpy:
            return

        inputs: list[tuple[str, Dtype, str]] = []
        for i, arg in enumerate(func.args.args):
            dt = _resolve_dtype_annotation(arg.annotation)
            is_py_scalar = False
            if isinstance(arg.annotation, ast.Name):
                is_py_scalar = arg.annotation.id in _PY_BUILTIN_TYPE_NAMES
            shape = "scalar" if (arg.arg in ("self", "cls") or is_py_scalar) else "array"
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
                if isinstance(arg_node, (ast.List, ast.Tuple)):
                    scalar_counter = self._register_literal_elts(
                        arg_node, inputs, existing_input_names,
                        scalar_constants, constant_assignments, scalar_counter,
                    )
                elif isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, (int, float)):
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

        # A non-call expression return (e.g. ``return a + np.sin(b)``) carries
        # the function's result in an operator expression, not in a numpy call.
        # Route it through full-body decomposition so the BinOp/UnaryOp is
        # materialized as the result kernel instead of being silently dropped.
        has_expr_return = return_stmt is not None and isinstance(
            return_stmt.value, (ast.BinOp, ast.UnaryOp)
        )

        has_multi_stmt = self._has_multi_statement_numpy(func)
        if has_multi_stmt or has_expr_return:
            self._decompose_full_body(
                func, module_name, inputs, outputs, effects,
                existing_input_names, scalar_constants, local_funcs or {},
            )
            return

        if has_nested and isinstance(return_stmt.value, ast.Call):
            self._decompose_composed_function(
                func, return_stmt.value, module_name, inputs,
                outputs, effects, existing_input_names, scalar_constants, local_funcs or {},
            )
            return

        algorithms = []
        nested_deps: list[str] = []
        reductions: list[ReductionEntry] = []

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
            algo = self._algorithm_for(target, child)
            if algo not in algorithms:
                algorithms.append(algo)
            reductions.append(ReductionEntry(
                rule="numpy_op_extraction",
                description=f"Extracted {target} as {algo} kernel",
                original=target,
            ))

        algorithm = algorithms[0] if algorithms else "composite"
        # Multi-output kernels (qr/svd/eig) return tuples — expand outputs
        if algorithm in ("linalg_qr", "linalg_svd"):
            # q, r = qr(A)  or  u, s, vh = svd(A)  — two/three outputs
            if algorithm == "linalg_qr":
                outputs = [("out_q", Dtype.FLOAT64, "array"), ("out_r", Dtype.FLOAT64, "array")]
            elif algorithm == "linalg_svd":
                outputs = [("out_u", Dtype.FLOAT64, "array"), ("out_s", Dtype.FLOAT64, "array"), ("out_v", Dtype.FLOAT64, "array")]
        elif algorithm == "linalg_eig":
            outputs = [("eigenvalues", Dtype.FLOAT64, "array")]
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

    def _record_local_callees(self, func: ast.FunctionDef,
                              local_funcs: dict[str, ast.FunctionDef]) -> None:
        """Record which module-local functions ``func`` calls.

        Gathered from the ORIGINAL AST (before inlining) so the interprocedural
        edges in the final graph mirror the source-level call graph.
        """
        callees: set[str] = set()
        for child in ast.walk(func):
            if not isinstance(child, ast.Call):
                continue
            target = ""
            if isinstance(child.func, ast.Attribute):
                target = _get_qualified_name(child.func)
            elif isinstance(child.func, ast.Name):
                target = child.func.id
            if target in local_funcs:
                callees.add(target)
        if callees:
            self._callees[func.name] = callees

    def _inline_local_calls(self, func: ast.FunctionDef,
                            local_funcs: dict[str, ast.FunctionDef]) -> ast.FunctionDef:
        """Inline module-local function calls into ``func``.

        A call to a module-local function is replaced by the callee's body with
        the callee's parameters substituted by the caller's actual arguments.
        Single-expression callees are replaced entirely; multi-statement
        callees have their statements spliced before the statement containing
        the call.  Nested local calls are inlined recursively.  Recursive /
        unresolved calls are left intact (they surface as unresolved inputs and
        are flagged by the identifier checker), never silently dropped.

        Without this, a composed expression such as ``np.multiply(helper(a), b)``
        silently discarded the helper's computation (the caller became just
        ``a * b``).
        """
        if not local_funcs:
            return func

        called: set[str] = set()
        for child in ast.walk(func):
            if isinstance(child, ast.Call):
                target = ""
                if isinstance(child.func, ast.Attribute):
                    target = _get_qualified_name(child.func)
                elif isinstance(child.func, ast.Name):
                    target = child.func.id
                if target in local_funcs:
                    called.add(target)
        if not called:
            return func

        new_body: list[ast.stmt] = []
        for stmt in func.body:
            new_body.extend(self._inline_stmt(stmt, local_funcs))

        new_func = ast.FunctionDef(
            name=func.name,
            args=func.args,
            body=new_body,
            decorator_list=func.decorator_list,
            returns=func.returns,
            lineno=func.lineno,
        )
        ast.fix_missing_locations(new_func)
        return new_func

    def _inline_stmt(self, stmt: ast.stmt, local_funcs: dict[str, ast.FunctionDef]) -> list[ast.stmt]:
        """Inline local calls found anywhere in ``stmt``.

        Returns a list of statements: any statements spliced out of
        multi-statement callees first, then the rewritten statement.
        """
        pre: list[ast.stmt] = []
        result = self._rewrite_node(stmt, local_funcs, pre)
        return pre + [result]

    def _rewrite_node(self, node, local_funcs: dict[str, ast.FunctionDef],
                      pre: list[ast.stmt]):
        """Rewrite ``node`` in place, inlining local calls in its expressions.

        ``pre`` accumulates statements spliced from multi-statement callees.
        """
        if isinstance(node, ast.expr):
            new_expr, spliced = self._inline_expr(node, local_funcs)
            pre.extend(spliced)
            return new_expr
        # Statement / generic nodes: rewrite all child expressions.
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                new_value, spliced = self._inline_expr(value, local_funcs)
                if spliced:
                    pre.extend(spliced)
                setattr(node, field, new_value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.expr):
                        new_item, spliced = self._inline_expr(item, local_funcs)
                        if spliced:
                            pre.extend(spliced)
                        value[i] = new_item
        return node

    def _inline_expr(self, expr: ast.expr,
                     local_funcs: dict[str, ast.FunctionDef]) -> tuple[ast.expr, list[ast.stmt]]:
        """Inline local calls inside ``expr``.

        Returns ``(new_expr, spliced_statements)``.  Spliced statements come
        from multi-statement callees and must execute before the statement
        containing ``expr``.
        """
        if not isinstance(expr, ast.Call):
            return expr, []

        target = ""
        if isinstance(expr.func, ast.Attribute):
            target = _get_qualified_name(expr.func)
        elif isinstance(expr.func, ast.Name):
            target = expr.func.id

        callee = local_funcs.get(target) if target else None
        if callee is not None:
            # Recursion guard: an in-progress callee is left intact (unresolved)
            # rather than infinitely recursing.
            if target in self._inline_visiting:
                return expr, []

            new_args, spliced_args = self._inline_call_args(expr.args, local_funcs)

            substitution: dict[str, ast.expr] = {}
            for param, arg_expr in zip(
                [a.arg for a in callee.args.args], new_args
            ):
                substitution[param] = arg_expr

            self._inline_visiting.add(target)
            try:
                ret_expr, body_stmts = self._inline_callee_body(
                    callee, substitution, local_funcs
                )
            finally:
                self._inline_visiting.discard(target)

            return ret_expr, spliced_args + body_stmts

        # Non-local call: rewrite arguments recursively.
        new_args, spliced_args = self._inline_call_args(expr.args, local_funcs)
        for i, arg in enumerate(new_args):
            expr.args[i] = arg
        return expr, spliced_args

    def _inline_call_args(self, args: list[ast.expr],
                          local_funcs: dict[str, ast.FunctionDef]) -> tuple[list[ast.expr], list[ast.stmt]]:
        new_args: list[ast.expr] = []
        spliced: list[ast.stmt] = []
        for arg in args:
            new_arg, s = self._inline_expr(arg, local_funcs)
            new_args.append(new_arg)
            spliced.extend(s)
        return new_args, spliced

    def _inline_callee_body(self, callee: ast.FunctionDef,
                            substitution: dict[str, ast.expr],
                            local_funcs: dict[str, ast.FunctionDef]) -> tuple[ast.expr, list[ast.stmt]]:
        """Produce (return_expr, spliced_statements) for an inlined callee."""
        body = list(callee.body)
        if not body:
            return self._substitute_expr(
                ast.Constant(value=0.0), substitution
            ), []

        ret_expr: ast.expr | None = None
        spliced: list[ast.stmt] = []
        for stmt in body:
            if isinstance(stmt, ast.Return):
                ret_value = stmt.value if stmt.value is not None else ast.Constant(value=None)
                ret_expr = self._substitute_expr(ret_value, substitution)
                continue
            # Non-return statement: substitute params, then splice.  Local calls
            # inside it are inlined recursively so nested helpers work.
            new_stmt = self._substitute_stmt(stmt, substitution)
            new_stmt = self._rewrite_node(new_stmt, local_funcs, spliced)
            spliced.append(new_stmt)
        if ret_expr is None:
            ret_expr = ast.Constant(value=0.0)
        return ret_expr, spliced

    def _substitute_stmt(self, stmt: ast.stmt, substitution: dict[str, ast.expr]) -> ast.stmt:
        return _substitute_into(stmt, substitution)

    def _substitute_expr(self, node, substitution: dict[str, ast.expr]):
        return _substitute_into(node, substitution)

    def _has_nested_numpy_calls(self, call_node: ast.Call,
                                local_funcs: dict[str, ast.FunctionDef]) -> bool:
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                target = self._resolve_call_target(arg)
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
                                    inner_target = self._resolve_call_target(inner_arg)
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
                    return (name, dt, shape)
            return (arg_node.id, Dtype.FLOAT64, self._shape_of(arg_node.id))

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
                return (arg_node.value.id, Dtype.FLOAT64, self._shape_of(arg_node.value.id))
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in intermediates:
                return (intermediates[arg_node.value.id], Dtype.FLOAT64, self._shape_of(arg_node.value.id))

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
            target = self._resolve_call_target(arg_node)
            if target in intermediates:
                inter_name = intermediates[target]
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))

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
                return (arg_node.value.id, Dtype.FLOAT64, self._shape_of(arg_node.value.id))

        if isinstance(arg_node, ast.Name):
            if arg_node.id in _PY_BUILTIN_TYPE_NAMES:
                const_name = f"_const_{len(scalar_constants)}"
                if const_name not in existing_names:
                    scalar_constants[const_name] = 0.0
                    existing_names.add(const_name)
                return (const_name, Dtype.FLOAT64, "scalar")
            for name, dt, shape in func_inputs:
                if name == arg_node.id:
                    return (name, dt, shape)

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
            algo = self._algorithm_for(target, call_node)
            is_last = (idx == len(operations) - 1)
            processed_call_ids.add(id(call_node))

            node_idx_ref = [node_idx]
            node_inputs: list[tuple[str, Dtype, str]] = []
            for arg in self._call_input_exprs(call_node):
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
                            resolved = (last_inter, Dtype.FLOAT64, self._shape_of(last_inter))
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
                out_shape = self._output_shape_for(algo, node_inputs)
                node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                if out_shape == "scalar":
                    self._scalar_names.add(inter_name)
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
        for stmt in func.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._resolve_call_target(stmt.value) == "numpy.random.seed":
                    return True
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

    def _is_module_ref(self, node: ast.expr) -> bool:
        """True if an expression is a reference to the numpy module (possibly
        through dotted submodules, e.g. ``np``, ``numpy``, ``np.random``)."""
        while isinstance(node, ast.Attribute):
            node = node.value
        return isinstance(node, ast.Name) and node.id in ("np", "numpy")

    def _call_input_exprs(self, call_node: ast.Call) -> list[ast.expr]:
        """Argument expressions of a numpy call, with the method receiver
        prepended for method-style calls such as ``x.transpose()``.

        ``x.transpose()`` carries the receiver ``x`` as an implicit first
        argument; ``np.transpose(x)`` does not.  Prepending the receiver keeps
        the kernel input list complete for method-style numpy calls.
        """
        exprs: list[ast.expr] = list(call_node.args)
        if isinstance(call_node.func, ast.Attribute) and not self._is_module_ref(call_node.func.value):
            exprs.insert(0, call_node.func.value)
        return exprs

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
                elif isinstance(stmt.value, (ast.BinOp, ast.UnaryOp, ast.IfExp)):
                    self._collect_calls_from_expr(stmt.value, operations, local_funcs)
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
                for fi_name, fi_dt, fi_shape in func_inputs:
                    if fi_name == name:
                        return (name, dt, fi_shape)
                if name in scalar_constants:
                    return (name, dt, "scalar")
                return (name, dt, self._shape_of(name))
            if expr.id in intermediates:
                return (intermediates[expr.id], Dtype.FLOAT64, self._shape_of(expr.id))
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
            return (expr.id, Dtype.FLOAT64, self._shape_of(expr.id))

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
            return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))

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
                out_shape = self._output_shape_for(algo, node_inputs)
                node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                if out_shape == "scalar":
                    self._scalar_names.add(inter_name)
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
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))
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
            algo = self._algorithm_for(call_target, expr)
            if algo is not None and algo != "unknown":
                node_inputs = []
                for arg in self._call_input_exprs(expr):
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
                out_shape = self._output_shape_for(algo, node_inputs)
                node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                if out_shape == "scalar":
                    self._scalar_names.add(inter_name)
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
                for in_name, _, fi_shape in func_inputs:
                    if in_name == name:
                        return (name, dt, fi_shape)
                return (name, dt, "array")
            if isinstance(expr.value, ast.Name) and expr.value.id in intermediates:
                inter_name = intermediates[expr.value.id]
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))
            if isinstance(expr.value, ast.Call):
                inner = self._decompose_expr(
                    expr.value, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                    module_name, func_name, origin_file, all_dep_ids, node_idx,
                )
                if isinstance(inner, tuple) and not inner[0].startswith("_unresolved_"):
                    inner_shape = inner[2] if len(inner) > 2 else "array"
                    return (inner[0], inner[1], inner_shape)

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
                return (intermediates[base.id], Dtype.FLOAT64, self._shape_of(base.id))

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
                        if_target = self._resolve_call_target(if_value) if isinstance(if_value, ast.Call) else None
                        else_target = self._resolve_call_target(else_value) if isinstance(else_value, ast.Call) else None
                        if if_target is not None:
                            if if_target.startswith("np."):
                                if_target = "numpy." + if_target[3:]
                        if else_target is not None:
                            if else_target.startswith("np."):
                                else_target = "numpy." + else_target[3:]
                        if_operand = None
                        else_operand = None
                        if if_target in NUMPY_OP_MAP:
                            node_idx_ref = [node_idx]
                            if_inputs = []
                            for arg in self._call_input_exprs(if_value):
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
                            if_operand = inter_if
                        else:
                            r = self._decompose_expr(
                                if_value, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func.name, origin_file, all_dep_ids,
                                [node_idx],
                            )
                            if not isinstance(r, list) and r[2] != "scalar":
                                if_operand = r[0]
                        if else_target in NUMPY_OP_MAP:
                            node_idx_ref = [node_idx]
                            else_inputs = []
                            for arg in self._call_input_exprs(else_value):
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
                            else_operand = inter_else
                        else:
                            r = self._decompose_expr(
                                else_value, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func.name, origin_file, all_dep_ids,
                                [node_idx],
                            )
                            if not isinstance(r, list) and r[2] != "scalar":
                                else_operand = r[0]
                        if if_operand is not None and else_operand is not None:
                            cond_name = f"_cond_{var_name}_{node_idx}"
                            node_idx += 1
                            self._node_counter += 1
                            where_id = _make_node_id(module_name, f"{func.name}_where_{var_name}")
                            where_sig = f"double {cond_name}, double {if_operand}, double {else_operand} -> array"
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
                                    (if_operand, Dtype.FLOAT64, "array"),
                                    (else_operand, Dtype.FLOAT64, "array"),
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

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                expr_target = self._resolve_call_target(stmt.value)
                if expr_target.startswith("np."):
                    expr_target = "numpy." + expr_target[3:]
                if expr_target in NUMPY_OP_MAP:
                    expr_id = id(stmt.value)
                    if expr_id in processed_calls:
                        continue
                    processed_calls.add(expr_id)
                    algo = self._algorithm_for(expr_target, stmt.value)
                    node_idx_ref = [node_idx]
                    node_inputs = []
                    for arg in self._call_input_exprs(stmt.value):
                        if isinstance(arg, (ast.List, ast.Tuple)):
                            for elt in arg.elts:
                                r = self._decompose_expr(
                                    elt, func_inputs, scalar_constants, intermediates,
                                    symbol_table, existing_names, constant_assignments,
                                    module_name, func.name, origin_file, all_dep_ids,
                                    node_idx_ref,
                                )
                                if isinstance(r, list):
                                    node_inputs.extend(r)
                                else:
                                    node_inputs.append(r)
                        else:
                            r = self._decompose_expr(
                                arg, func_inputs, scalar_constants, intermediates,
                                symbol_table, existing_names, constant_assignments,
                                module_name, func.name, origin_file, all_dep_ids,
                                node_idx_ref,
                            )
                            if isinstance(r, list):
                                node_inputs.extend(r)
                            else:
                                node_inputs.append(r)
                    for kw in stmt.value.keywords:
                        if isinstance(kw.value, ast.Constant):
                            const_val = float(kw.value.value)
                            const_name = f"_const_{len(scalar_constants)}"
                            if const_name not in existing_names:
                                scalar_constants[const_name] = const_val
                                existing_names.add(const_name)
                            node_inputs.append((const_name, Dtype.FLOAT64, "scalar"))
                    inter_name = f"_inter_expr_{node_idx}"
                    node_idx += 1
                    out_shape = self._output_shape_for(algo, node_inputs)
                    node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                    if out_shape == "scalar":
                        self._scalar_names.add(inter_name)
                    sig_parts = [f"{dt.name.lower()} {n}" for n, dt, _ in node_inputs]
                    origin_sig = f"{' -> '.join(sig_parts)} -> {out_shape}"
                    self._node_counter += 1
                    node_id = _make_node_id(module_name, f"{func.name}_expr_{node_idx}")
                    self.graph.add_node(MathIRNode(
                        node_id=node_id,
                        origin_symbol=f"{module_name}.{func.name}",
                        origin_file=origin_file,
                        origin_line=stmt.lineno,
                        origin_commit=self.origin_commit,
                        origin_signature=origin_sig,
                        math_intent=f"Side-effect call {expr_target} as {algo}",
                        inputs=node_inputs,
                        outputs=node_outputs,
                        effects=[Effect.ALLOC] if algo == "noop_seed" else [Effect.PURE],
                        algorithm=algo,
                        reductions=[ReductionEntry(rule="numpy_op_extraction",
                                                   description=f"Extracted {expr_target} as {algo} kernel",
                                                   original=expr_target)],
                        nested_deps=list(all_dep_ids),
                        stack_usage=256, heap_usage=None,
                        reentrant=algo != "noop_seed",
                        dep_kind=DepKind.MATH_KERNEL,
                        scalar_constants=dict(scalar_constants),
                    ))
                    self.graph.entry_points.append(node_id)
                    all_dep_ids.append(node_id)
                continue

            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                # Handle tuple unpacking for multi-output kernels: q, r = np.linalg.qr(A)
                if isinstance(stmt.targets[0], ast.Tuple) and isinstance(stmt.value, ast.Call):
                    target = _get_qualified_name(stmt.value.func)
                    if target.startswith("np."):
                        target = "numpy." + target[3:]
                    if target in NUMPY_OP_MAP:
                        algo = NUMPY_OP_MAP[target]
                        # Only handle known multi-output algos
                        if algo in ("linalg_qr", "linalg_svd", "linalg_eig"):
                            call_inputs = []
                            for arg in stmt.value.args:
                                if isinstance(arg, ast.Name) and arg.id in symbol_table:
                                    nm, dt = symbol_table[arg.id]
                                    for fi in func_inputs:
                                        if fi[0] == nm:
                                            call_inputs.append((nm, dt, fi[2]))
                                            break
                                    else:
                                        call_inputs.append((nm, dt, "array"))
                                elif isinstance(arg, ast.Name):
                                    call_inputs.append((arg.id, Dtype.FLOAT64, "array"))
                            # Outputs are the tuple elements
                            # For linalg_eig, Python returns (w, v) but Purce's kernel only models eigenvalues (w)
                            # So for eig with 2 targets, keep only the first (w)
                            elts = stmt.targets[0].elts
                            if algo == "linalg_eig" and len(elts) == 2:
                                elts = [elts[0]]
                            node_outputs = []
                            for elt in elts:
                                if isinstance(elt, ast.Name):
                                    out_name = elt.id
                                    # Register in symbol table as array
                                    symbol_table[out_name] = (out_name, Dtype.FLOAT64)
                                    node_outputs.append((out_name, Dtype.FLOAT64, "array"))
                                    existing_names.add(out_name)
                            if node_outputs:
                                self._node_counter += 1
                                node_id = _make_node_id(module_name, f"{func.name}_qr_{self._node_counter}")
                                self.graph.add_node(MathIRNode(
                                    node_id=node_id,
                                    origin_symbol=f"{module_name}.{func.name}",
                                    origin_file=origin_file,
                                    origin_line=stmt.lineno,
                                    origin_commit=self.origin_commit,
                                    origin_signature=f"{', '.join([f'{dt.name.lower()} {n}' for n,dt,_ in call_inputs])} -> {', '.join([o[0] for o in node_outputs])}",
                                    math_intent=f"Multi-output {algo}",
                                    inputs=call_inputs,
                                    outputs=node_outputs,
                                    effects=[Effect.PURE],
                                    algorithm=algo,
                                    reductions=[ReductionEntry(rule="numpy_op_extraction", description=f"Extracted {target} as {algo}", original=target)],
                                    nested_deps=list(all_dep_ids),
                                    stack_usage=256, heap_usage=None, reentrant=True,
                                    dep_kind=DepKind.MATH_KERNEL,
                                    scalar_constants=dict(scalar_constants),
                                ))
                                self.graph.entry_points.append(node_id)
                                all_dep_ids.append(node_id)
                            continue
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
                        out_shape = self._output_shape_for(algo, node_inputs)
                        node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                        if out_shape == "scalar":
                            self._scalar_names.add(inter_name)
                        self._node_counter += 1
                        nid = _make_node_id(module_name, f"{func.name}_{target_name}")
                        sig_parts = [f"{dt.name.lower()} {n}" for n, dt, _ in node_inputs]
                        origin_sig = f"{' -> '.join(sig_parts)} -> {out_shape}"
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

                if isinstance(stmt.value, ast.UnaryOp) and isinstance(stmt.value.op, ast.USub):
                    node_idx_ref = [node_idx]
                    resolved = self._decompose_expr(
                        stmt.value, func_inputs, scalar_constants, intermediates,
                        symbol_table, existing_names, constant_assignments,
                        module_name, func.name, origin_file, all_dep_ids, node_idx_ref,
                    )
                    node_idx = node_idx_ref[0]
                    if isinstance(resolved, list):
                        resolved = resolved[0]
                    symbol_table[target_name] = (resolved[0], resolved[1])
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

                        algo = self._algorithm_for(call_target, stmt.value)

                        node_idx_ref = [node_idx]
                        node_inputs: list[tuple[str, Dtype, str]] = []
                        for arg in self._call_input_exprs(stmt.value):
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
                        out_shape = self._output_shape_for(algo, node_inputs)
                        node_outputs = [(inter_name, Dtype.FLOAT64, out_shape)]
                        if out_shape == "scalar":
                            self._scalar_names.add(inter_name)
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
                    algo = self._algorithm_for(call_target, return_stmt.value)

                    node_idx_ref = [node_idx]
                    node_inputs: list[tuple[str, Dtype, str]] = []
                    for arg in self._call_input_exprs(return_stmt.value):
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

        if return_stmt is not None and isinstance(return_stmt.value, ast.BinOp):
            binop_map = {
                ast.Add: "element_add", ast.Sub: "element_sub",
                ast.Mult: "element_mul", ast.Div: "element_div",
                ast.Pow: "element_power", ast.Mod: "element_mod",
                ast.FloorDiv: "element_div",
            }
            op = type(return_stmt.value.op)
            if op in binop_map:
                node_idx_ref = [node_idx]
                left = self._decompose_expr(
                    return_stmt.value.left, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                    module_name, func.name, origin_file, all_dep_ids, node_idx_ref,
                )
                right = self._decompose_expr(
                    return_stmt.value.right, func_inputs, scalar_constants, intermediates,
                    symbol_table, existing_names, constant_assignments,
                    module_name, func.name, origin_file, all_dep_ids, node_idx_ref,
                )
                node_idx = node_idx_ref[0]
                node_inputs: list[tuple[str, Dtype, str]] = []
                for operand in (left, right):
                    if isinstance(operand, list):
                        node_inputs.extend(operand)
                    else:
                        node_inputs.append(operand)
                algo = binop_map[op]
                node_outputs = list(func_outputs)
                sig_parts = [f"{dt.name.lower()} {name}" for name, dt, _ in node_inputs]
                origin_sig = f"{' -> '.join(sig_parts)} -> {func_outputs[0][1].name.lower()}"

                self._node_counter += 1
                node_id = _make_node_id(module_name, func.name)
                node = MathIRNode(
                    node_id=node_id,
                    origin_symbol=f"{module_name}.{func.name}",
                    origin_file=origin_file,
                    origin_line=return_stmt.lineno,
                    origin_commit=self.origin_commit,
                    origin_signature=origin_sig,
                    math_intent=f"Return of '{func.name}' implementing {algo}",
                    inputs=node_inputs,
                    outputs=node_outputs,
                    effects=effects,
                    algorithm=algo,
                    reductions=[ReductionEntry(
                        rule="binop_return_decomposition",
                        description=f"Decomposed return BinOp as {algo}",
                        original=algo,
                    )],
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

        if return_stmt is not None and (
            isinstance(return_stmt.value, ast.UnaryOp)
            and isinstance(return_stmt.value.op, ast.USub)
        ):
            node_idx_ref = [node_idx]
            inner = self._decompose_expr(
                return_stmt.value.operand, func_inputs, scalar_constants, intermediates,
                symbol_table, existing_names, constant_assignments,
                module_name, func.name, origin_file, all_dep_ids, node_idx_ref,
            )
            node_idx = node_idx_ref[0]
            if isinstance(inner, list):
                inner = inner[0]
            neg_const_name = f"_const_{len(scalar_constants)}"
            if neg_const_name not in existing_names:
                scalar_constants[neg_const_name] = -1.0
                existing_names.add(neg_const_name)
            node_inputs: list[tuple[str, Dtype, str]] = [
                (neg_const_name, Dtype.FLOAT64, "scalar"), inner,
            ]
            node_outputs = list(func_outputs)
            sig_parts = [f"{dt.name.lower()} {name}" for name, dt, _ in node_inputs]
            origin_sig = f"{' -> '.join(sig_parts)} -> {func_outputs[0][1].name.lower()}"

            self._node_counter += 1
            node_id = _make_node_id(module_name, func.name)
            node = MathIRNode(
                node_id=node_id,
                origin_symbol=f"{module_name}.{func.name}",
                origin_file=origin_file,
                origin_line=return_stmt.lineno,
                origin_commit=self.origin_commit,
                origin_signature=origin_sig,
                math_intent=f"Return of '{func.name}' implementing element_mul negation",
                inputs=node_inputs,
                outputs=node_outputs,
                effects=effects,
                algorithm="element_mul",
                reductions=[ReductionEntry(
                    rule="unary_return_decomposition",
                    description="Decomposed return negation as element_mul",
                    original="neg",
                )],
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
                for fi_name, fi_dt, fi_shape in func_inputs:
                    if fi_name == name:
                        return (name, dt, fi_shape)
                if name in scalar_constants:
                    return (name, dt, "scalar")
                return (name, dt, self._shape_of(name))
            if arg_node.id in existing_names:
                for name, dt, shape in func_inputs:
                    if name == arg_node.id:
                        return (name, dt, shape)
                return (arg_node.id, Dtype.FLOAT64, self._shape_of(arg_node.id))
            if arg_node.id in intermediates:
                inter_name = intermediates[arg_node.id]
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))
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
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))
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
                        for in_name, _, fi_shape in func_inputs:
                            if in_name == name:
                                return (name, dt, fi_shape)
                        return (name, dt, "array")
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
                for in_name, _, fi_shape in func_inputs:
                    if in_name == name:
                        return (name, dt, fi_shape)
                return (name, dt, "array")
            if isinstance(arg_node.value, ast.Name) and arg_node.value.id in intermediates:
                inter_name = intermediates[arg_node.value.id]
                return (inter_name, Dtype.FLOAT64, self._shape_of(inter_name))

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
                    if isinstance(arg_node.slice, ast.Name) or isinstance(arg_node.slice, ast.Attribute):
                        if isinstance(arg_node.slice, ast.Name) and arg_node.slice.id in symbol_table:
                            dim_name = f"_dim_{arg_node.value.value.id}_{arg_node.slice.id}"
                            const_val = 0.0
                            const_name = f"_const_{len(scalar_constants)}"
                            if const_name not in existing_names:
                                scalar_constants[const_name] = const_val
                                existing_names.add(const_name)
                            return (const_name, Dtype.INT64, "scalar")
                        dim_name = f"_dim_{arg_node.value.value.id}_dyn"
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
                    return (intermediates[arg_node.value.id], Dtype.FLOAT64, self._shape_of(arg_node.value.id))

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
        target = self._resolve_call_target(call_node)

        for arg in self._call_input_exprs(call_node):
            self._collect_calls_from_expr(arg, operations, local_funcs)

        if target in NUMPY_OP_MAP:
            operations.append((target, call_node))

    def _collect_calls_from_expr(self, expr: ast.expr, operations: list[tuple[str, ast.Call]],
                                 local_funcs: dict[str, ast.FunctionDef]) -> None:
        if isinstance(expr, ast.Call):
            inner_target = self._resolve_call_target(expr)
            if inner_target in NUMPY_OP_MAP:
                self._collect_numpy_calls(expr, operations, local_funcs)
            elif inner_target in local_funcs:
                local_def = local_funcs[inner_target]
                for stmt in ast.walk(local_def):
                    if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                        self._collect_numpy_calls(stmt.value, operations, local_funcs)
            else:
                for sub_arg in self._call_input_exprs(expr):
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
