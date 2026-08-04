from __future__ import annotations

import ast
from dataclasses import dataclass, field

from purce.ir.builder import NUMPY_OP_MAP
from purce.ir.nodes import DepKind, Dtype, Effect, MathIRGraph, MathIRNode, ReductionEntry

SUPPORTED_MODULES = {
    "numpy": {
        "dot", "matmul", "add", "subtract", "multiply", "divide",
        "zeros", "ones", "eye", "array",
        "sum", "mean", "max", "min", "var",
        "sqrt", "abs", "exp", "log", "sin", "cos", "tan", "tanh",
        "maximum", "minimum", "power", "where", "clip",
        "negative", "sign", "floor", "transpose",
        "greater", "less", "log10", "logaddexp",
        "conj", "angle", "real", "imag",
        "outer", "diag", "copy",
        "arange", "linspace", "full", "full_like",
        "ones_like", "zeros_like",
        "concatenate", "take", "take_along_axis",
        "argsort", "tril", "triu",
        "reshape", "squeeze", "expand_dims", "flatten",
    },
    "numpy.linalg": {"solve", "inv", "cholesky", "eig", "norm"},
    "numpy.fft": {"fft", "ifft"},
    "numpy.random": {"randn", "random", "randint", "uniform", "seed", "beta", "permutation"},
}


@dataclass
class Diagnostic:
    severity: str
    file: str
    line: int
    construct: str
    reason: str
    suggestion: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "file": self.file,
            "line": str(self.line),
            "construct": self.construct,
            "reason": self.reason,
            "suggestion": self.suggestion,
        }


@dataclass
class ParsedFunction:
    name: str
    module: str
    node_id: str
    origin_file: str
    origin_line: int
    origin_signature: str
    math_intent: str
    algorithm: str
    inputs: list[tuple[str, Dtype, str]]
    outputs: list[tuple[str, Dtype, str]]
    effects: list[Effect]
    reductions: list[ReductionEntry]
    nested_deps: list[str]
    stack_usage: int | None
    heap_usage: int | None
    reentrant: bool
    raw_body: str


@dataclass
class ParseResult:
    functions: list[ParsedFunction] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_graph(self) -> MathIRGraph:
        graph = MathIRGraph()
        for fn in self.functions:
            node = MathIRNode(
                node_id=fn.node_id,
                origin_symbol=f"{fn.module}.{fn.name}",
                origin_file=fn.origin_file,
                origin_line=fn.origin_line,
                origin_commit=None,
                origin_signature=fn.origin_signature,
                math_intent=fn.math_intent,
                inputs=fn.inputs,
                outputs=fn.outputs,
                effects=fn.effects,
                algorithm=fn.algorithm,
                reductions=fn.reductions,
                nested_deps=fn.nested_deps,
                stack_usage=fn.stack_usage,
                heap_usage=fn.heap_usage,
                reentrant=fn.reentrant,
                dep_kind=DepKind.MATH_KERNEL,
            )
            graph.add_node(node)
            graph.entry_points.append(fn.node_id)
        return graph


class PythonParser:
    """Parses Python source files and extracts math kernel functions."""

    def __init__(self, target_profile: str = "generic-c99"):
        self.target_profile = target_profile
        self._diagnostics: list[Diagnostic] = []
        self._source_lines: list[str] = []

    @property
    def diagnostics(self) -> list[Diagnostic]:
        return list(self._diagnostics)

    def parse_source(self, source: str, filename: str = "<string>") -> ParseResult:
        self._diagnostics = []
        self._source_lines = source.splitlines()
        result = ParseResult()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            result.diagnostics.append(Diagnostic(
                severity="ERROR",
                file=filename,
                line=e.lineno or 0,
                construct="syntax",
                reason=str(e.msg),
                suggestion="Fix Python syntax errors",
            ))
            return result

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parsed = self._parse_function(node, filename)
                if parsed is not None:
                    result.functions.append(parsed)

        result.diagnostics = list(self._diagnostics)
        return result

    def parse_file(self, path: str) -> ParseResult:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except (OSError, UnicodeDecodeError) as e:
            return ParseResult(diagnostics=[Diagnostic(
                severity="ERROR",
                file=path,
                line=0,
                construct="file_read",
                reason=str(e),
                suggestion="Check file path and encoding",
            )])
        return self.parse_source(source, filename=path)

    def _parse_function(self, func: ast.FunctionDef, filename: str) -> ParsedFunction | None:
        numpy_calls = self._find_numpy_calls(func)
        if not numpy_calls:
            return None

        unsupported = self._check_unsupported(func)
        if unsupported:
            return None

        inputs = self._extract_inputs(func)
        outputs = self._extract_outputs(func)
        algorithm, reductions = self._classify_calls(numpy_calls)
        math_intent = self._build_math_intent(func.name, numpy_calls)

        sig_parts = [f"{dt.name.lower()} {name}" for name, dt, _ in inputs]
        origin_sig = f"{' -> '.join(sig_parts)} -> {outputs[0][1].name.lower()}"

        stack_usage = 64 + len(inputs) * 8
        alloc_ops = {"numpy.zeros", "numpy.ones", "numpy.eye"}
        has_alloc = any(r.original in alloc_ops for r in reductions)

        node_id_base = f"{filename.replace('/', '_').replace('.', '_')}.{func.name}"

        return ParsedFunction(
            name=func.name,
            module=filename.replace("/", ".").replace("\\", ".").replace(".py", ""),
            node_id=node_id_base,
            origin_file=filename,
            origin_line=func.lineno,
            origin_signature=origin_sig,
            math_intent=math_intent,
            algorithm=algorithm,
            inputs=inputs,
            outputs=outputs,
            effects=[Effect.ALLOC] if has_alloc else [Effect.PURE],
            reductions=reductions,
            nested_deps=[],
            stack_usage=stack_usage,
            heap_usage=None if not has_alloc else 0,
            reentrant=not has_alloc,
            raw_body=self._extract_source_segment(func),
        )

    def _extract_source_segment(self, node: ast.AST) -> str:
        try:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            if start < 0 or end > len(self._source_lines):
                return ""
            return "\n".join(self._source_lines[start:end])
        except (AttributeError, IndexError):
            return ""

    def _find_numpy_calls(self, func: ast.FunctionDef) -> list[str]:
        calls: list[str] = []
        for child in ast.walk(func):
            if not isinstance(child, ast.Call):
                continue
            target = self._resolve_call_target(child.func)
            if target is None:
                continue
            normalized = self._normalize_numpy_target(target)
            if normalized is not None:
                calls.append(normalized)
        return calls

    def _resolve_call_target(self, func_node: ast.expr) -> str | None:
        if isinstance(func_node, ast.Attribute):
            return self._get_full_attribute_name(func_node)
        if isinstance(func_node, ast.Name):
            return func_node.id
        return None

    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        parts: list[str] = []
        current: ast.expr = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _normalize_numpy_target(self, target: str) -> str | None:
        if target.startswith("np."):
            target = "numpy." + target[3:]
        if not target.startswith("numpy."):
            return None

        parts = target.split(".", 1)
        if len(parts) < 2:
            return None

        module_path = "numpy"
        op_name = parts[1]

        if "." in op_name:
            sub_parts = op_name.split(".", 1)
            module_path = f"numpy.{sub_parts[0]}"
            op_name = sub_parts[1]

        supported = SUPPORTED_MODULES.get(module_path, set())
        if op_name in supported:
            return f"{module_path}.{op_name}"

        return None

    def _check_unsupported(self, func: ast.FunctionDef) -> list[Diagnostic]:
        issues: list[Diagnostic] = []
        for child in ast.walk(func):
            if isinstance(child, ast.Call):
                target = self._resolve_call_target(child.func)
                if target in ("eval", "exec", "getattr", "setattr"):
                    issues.append(Diagnostic(
                        severity="ERROR",
                        file="",
                        line=child.lineno,
                        construct=target,
                        reason=f"Dynamic dispatch '{target}()' cannot be translated to C99",
                        suggestion="Refactor to use static dispatch",
                    ))
                elif target == "open":
                    issues.append(Diagnostic(
                        severity="ERROR",
                        file="",
                        line=child.lineno,
                        construct="open",
                        reason="File I/O cannot be translated to C99 math kernel",
                        suggestion="Separate I/O from computation",
                    ))
        for issue in issues:
            self._diagnostics.append(issue)
        return issues

    def _extract_inputs(self, func: ast.FunctionDef) -> list[tuple[str, Dtype, str]]:
        inputs: list[tuple[str, Dtype, str]] = []
        for arg in func.args.args:
            if arg.arg in ("self", "cls"):
                continue
            dt = self._resolve_type_hint(arg.annotation)
            shape = "scalar" if arg.arg in ("n", "m", "k", "i", "j", "axis", "ord") else "array"
            inputs.append((arg.arg, dt, shape))
        return inputs

    def _extract_outputs(self, func: ast.FunctionDef) -> list[tuple[str, Dtype, str]]:
        if func.returns:
            dt = self._resolve_type_hint(func.returns)
            return [("result", dt, "array")]
        return [("result", Dtype.FLOAT64, "array")]

    def _resolve_type_hint(self, annotation: ast.expr | None) -> Dtype:
        if annotation is None:
            return Dtype.FLOAT64
        if isinstance(annotation, ast.Name):
            hint_map = {
                "float": Dtype.FLOAT64,
                "float32": Dtype.FLOAT32,
                "float64": Dtype.FLOAT64,
                "int": Dtype.INT32,
                "int32": Dtype.INT32,
                "int64": Dtype.INT64,
                "bool": Dtype.BOOL,
                "complex": Dtype.COMPLEX128,
                "ndarray": Dtype.FLOAT64,
            }
            return hint_map.get(annotation.id, Dtype.FLOAT64)
        if isinstance(annotation, ast.Attribute):
            full = self._get_full_attribute_name(annotation)
            if "float32" in full:
                return Dtype.FLOAT32
            if "float64" in full:
                return Dtype.FLOAT64
            if "int32" in full:
                return Dtype.INT32
            if "int64" in full:
                return Dtype.INT64
        return Dtype.FLOAT64

    def _classify_calls(self, numpy_calls: list[str]) -> tuple[str, list[ReductionEntry]]:
        algorithms: list[str] = []
        reductions: list[ReductionEntry] = []

        for call in numpy_calls:
            algo = NUMPY_OP_MAP.get(call, "unknown")
            if algo not in algorithms:
                algorithms.append(algo)
            reductions.append(ReductionEntry(
                rule="numpy_op_extraction",
                description=f"Extracted {call} as {algo} kernel",
                original=call,
            ))

        primary = algorithms[0] if algorithms else "composite"
        return primary, reductions

    def _build_math_intent(self, func_name: str, numpy_calls: list[str]) -> str:
        ops = []
        for call in numpy_calls:
            algo = NUMPY_OP_MAP.get(call, call)
            ops.append(algo)
        op_str = ", ".join(ops) if ops else "unknown"
        return f"Math kernel '{func_name}' implementing {op_str}"
