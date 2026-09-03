"""Tier-R emitter: Python AST -> C99 calling the Tier-R runtime.

Contract (do not break):
- Every construct maps to runtime calls; anything unimplemented raises
  :class:`UnsupportedError` at emit time — compile-time failure with a
  clear message, never silent wrong behavior.
- After every runtime call the generated code checks
  ``if (p_err(S)) return NULL;`` so exceptions propagate; NULL with err
  set is the single error channel.
- Frame layout: slot 0 of every frame = enclosing frame. Module frame
  slot 0 = ``p_none``. Function frames are created per call; the env of
  a closure is the frame in which the ``def`` executed. Name reads
  resolve lexically: depth-0 reads the current frame ``F``; depth-d
  reads ``c->u.g.env`` followed by d-1 hops through slot 0.
- Evaluation order matches CPython: lazy ``and/or``, lazy chained
  comparisons, target/operand order in assignment and aug-assign,
  single evaluation of every expression (no double-emitted operands).
- Recursion depth is enforced in ``p_call`` (PE_RECUR).
- The module entry is ``pf_mod_entry(pst*, pval*, pval**, int)``
  (non-static) and is returned by ``purce_tir_entry`` for the ctypes
  harness.

Documented restrictions (all fail loudly at emit time):
- ``is`` / ``is not`` only with a ``None`` literal on one side
  (pointer identity matches CPython for None; anything else would be a
  silent divergence).
- ``assert`` with message, keyword args, defaults, varargs,
  posonly/kwonly args, decorators, method calls, import/class/try,
  comprehensions, lambdas, bitwise ops, slices: unsupported.
"""

from __future__ import annotations

import ast
import math
import re

from purce.runtime.runtime_c import RUNTIME_C_SOURCE

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class UnsupportedError(NotImplementedError):
    """Compile-time failure: construct outside the Tier-R surface."""

    def __init__(self, node: ast.AST, msg: str) -> None:
        self.msg = msg
        lineno = getattr(node, "lineno", None)
        if lineno:
            msg = f"line {lineno}: {msg}"
        super().__init__(msg)


BUILTINS = frozenset(
    {
        "len",
        "print",
        "str",
        "int",
        "float",
        "bool",
        "abs",
        "min",
        "max",
        "sum",
        "range",
        "list",
    }
)


def c_str(s: str) -> str:
    """C string literal from a Python str (UTF-8 bytes, octal escapes)."""
    out = ['"']
    for b in s.encode("utf-8"):
        if b == 0x22:
            out.append('\\"')
        elif b == 0x5C:
            out.append("\\\\")
        elif b == 0x0A:
            out.append("\\n")
        elif b == 0x09:
            out.append("\\t")
        elif b == 0x0D:
            out.append("\\r")
        elif 0x20 <= b < 0x7F:
            out.append(chr(b))
        else:
            out.append(f"\\{b:03o}")
    out.append('"')
    return "".join(out)


def _float_c(v: float) -> str:
    """C99 double literal from a Python float (incl. inf/nan)."""
    if math.isnan(v):
        return "(0.0/0.0)"
    if math.isinf(v):
        return f"({'-' if v < 0 else ''}1.0/0.0)"
    return repr(v)


_BINOP = {
    ast.Add: "p_add",
    ast.Sub: "p_sub",
    ast.Mult: "p_mul",
    ast.Div: "p_truediv",
    ast.FloorDiv: "p_floordiv",
    ast.Mod: "p_mod",
    ast.Pow: "p_pow",
}

_CMPOP = {
    ast.Eq: "p_eq",
    ast.NotEq: "p_ne",
    ast.Lt: "p_lt",
    ast.LtE: "p_le",
    ast.Gt: "p_gt",
    ast.GtE: "p_ge",
}


class _Frame:
    """Lexical mapping of variable names to frame slots (1-based slots).

    ``resolve(name)`` returns ``(depth, slot)`` where depth is the
    number of enclosing frames to traverse (0 = current frame).

    Python's local-vs-closure rule is modeled exactly: ``assigned``
    holds every name that is assigned anywhere in this scope (so it is
    a local even before its textual assignment); ``assigned_so_far``
    tracks textual order during emission so a local read before its
    first assignment is rejected at compile time (CPython would raise
    UnboundLocalError at runtime).
    """

    __slots__ = ("map", "next", "parent", "assigned", "assigned_so_far")

    def __init__(self, parent: _Frame | None = None):
        self.map: dict[str, int] = {}
        self.next = 1  # slot 0 is reserved for the enclosing frame
        self.parent = parent
        self.assigned: set[str] = set()
        self.assigned_so_far: set[str] = set()

    def alloc(self, name: str) -> int:
        i = self.map.get(name)
        if i is not None:
            return i
        i = self.next
        self.next += 1
        self.map[name] = i
        return i

    def mark_assigned(self, name: str) -> int:
        self.assigned.add(name)
        self.assigned_so_far.add(name)
        return self.alloc(name)

    def resolve(self, name: str) -> tuple[int, int] | None:
        if name in self.assigned:
            return (0, self.alloc(name))
        f: _Frame | None = self.parent
        d = 1
        while f is not None:
            i = f.map.get(name)
            if i is not None:
                return (d, i)
            if name in f.assigned:
                return (d, f.alloc(name))
            f = f.parent
            d += 1
        return None


class Emitter:
    """Compiles one Python source string into C99.

    Supported grammar (anything else raises UnsupportedError):
      literals (None/True/False/int/float/str), Name, List/Tuple/Dict,
      BinOp (+ - * / // % **), UnaryOp (- + not), IfExp, Compare
      (incl. chained, short-circuit, left-to-right), BoolOp (and/or,
      lazy), Call (builtins + user functions), Subscript read/write,
      statements: assign, augassign, if/elif/else, while, for (Name
      target), break, continue, return, pass, assert (no message),
      def (module-level and nested; exact arity, no defaults).
    """

    def __init__(self, source: str) -> None:
        self.src = source
        try:
            self.tree = ast.parse(source)
        except SyntaxError as e:  # pragma: no cover
            raise UnsupportedError(ast.Module(body=[]), f"syntax error in source: {e}") from None
        self.tmpn = 0
        self.fncount = 0
        self.cfuncs: list[str] = []  # generated static C functions

    # ── entry point ──────────────────────────────────────────────────
    def compile_module(self) -> str:
        module_frame = _Frame()
        self._prescan(module_frame, self.tree.body)
        entry = self._module_body(module_frame)
        return "\n".join(
            [
                "/* generated by purce Tier-R emitter (tier-R runtime) */",
                RUNTIME_C_SOURCE,
                "",
                *self.cfuncs,
                entry,
                "",
            ]
        )

    def tmp(self) -> str:
        self.tmpn += 1
        return f"pv{self.tmpn}"

    def gen_function(self, fn: ast.FunctionDef, parent: _Frame) -> str:
        """Emit one C function for ``fn``; returns its C name."""
        a = fn.args
        if fn.decorator_list:
            raise UnsupportedError(fn, "decorators unsupported")
        if (
            a.vararg is not None
            or a.kwarg is not None
            or a.posonlyargs
            or a.kwonlyargs
            or a.defaults
            or a.kw_defaults
        ):
            raise UnsupportedError(
                fn, f"function '{fn.name}': defaults/varargs/kwonly/posonly args unsupported"
            )
        if not _IDENT.match(fn.name):
            raise UnsupportedError(fn, f"function name '{fn.name!r}' not an ASCII identifier")
        cname = f"pf{self.fncount}"
        self.fncount += 1

        frame = _Frame(parent)
        argnames = []
        for arg in a.args:
            argnames.append(arg.arg)
            frame.mark_assigned(arg.arg)
        self._prescan(frame, fn.body)
        body_lines: list[str] = []
        for s in fn.body:
            body_lines.extend(_Stmts.emit(self, frame, s))

        lines = [
            f"static pval *{cname}(pst *S, pval *c, pval **argv, int narg) {{",
            "    (void)narg;",
            f"    pval *F = p_newframe(S, {frame.next});",
            "    if (!F) return NULL;",
            "    p_frame_set(S, F, 0, c->u.g.env);",
            "    if (p_err(S)) return NULL;",
        ]
        if not argnames:
            lines.append("    (void)argv;")
        for i, name in enumerate(argnames):
            lines.append(f"    p_frame_set(S, F, {frame.map[name]}, argv[{i}]);")
            lines.append("    if (p_err(S)) return NULL;")
        lines.extend(body_lines)
        lines.append("    return p_none(S);")
        lines.append("}")
        self.cfuncs.append("\n".join(lines))
        return cname

    @staticmethod
    def _prescan(frame: _Frame, body: list[ast.stmt]) -> None:
        """Collect every name assigned anywhere in this scope so that
        forward references resolve as locals (Python rule)."""
        for s in body:
            if isinstance(s, ast.FunctionDef):
                frame.assigned.add(s.name)
            elif isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        frame.assigned.add(t.id)
            elif isinstance(s, ast.AugAssign):
                if isinstance(s.target, ast.Name):
                    frame.assigned.add(s.target.id)
            elif isinstance(s, ast.For):
                if isinstance(s.target, ast.Name):
                    frame.assigned.add(s.target.id)
            if isinstance(s, (ast.If, ast.While, ast.For)):
                Emitter._prescan(frame, s.body)
                Emitter._prescan(frame, s.orelse)

    def _module_body(self, frame: _Frame) -> str:
        body_lines: list[str] = []
        for s in self.tree.body:
            if isinstance(s, ast.FunctionDef):
                cname = self.gen_function(s, frame)
                v = self.tmp()
                lines = [
                    f"    pval *{v} = p_mkfunc(S, {cname}, F, {len(s.args.args)}, {c_str(s.name)});",
                    f"    if (!{v}) return NULL;",
                    f"    p_frame_set(S, F, {frame.alloc(s.name)}, {v});",
                ]
                body_lines.extend(lines)
            else:
                body_lines.extend(_Stmts.emit(self, frame, s))
        out = [
            "pval *pf_mod_entry(pst *S, pval *c, pval **argv, int narg) {",
            "    (void)c; (void)argv; (void)narg;",
            f"    pval *F = p_newframe(S, {frame.next});",
            "    if (!F) return NULL;",
            "    p_frame_set(S, F, 0, p_none(S));",
        ]
        out.extend(body_lines)
        out.append("    return p_none(S);")
        out.append("}")
        return "\n".join(out)


class _Stmts:
    """Statement emission: appends C lines with 4-space base indent."""

    @staticmethod
    def emit(em: Emitter, frame: _Frame, node) -> list[str]:
        lines: list[str] = []

        def store(name: str, val: str) -> None:
            i = frame.mark_assigned(name)
            lines.append(f"    p_frame_set(S, F, {i}, {val});")
            lines.append("    if (p_err(S)) return NULL;")

        def expr(n) -> str:
            return _Expr.emit(em, frame, n, lines)

        if isinstance(node, ast.Expr):
            e = expr(node.value)
            lines.append(f"    (void){e};")
            return lines
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                raise UnsupportedError(node, "multi-target assignment unsupported")
            t = node.targets[0]
            value = expr(node.value)
            if isinstance(t, ast.Name):
                store(t.id, value)
            elif isinstance(t, ast.Subscript):
                obj = expr(t.value)
                key = expr(t.slice)
                lines.append(f"    p_setitem(S, {obj}, {key}, {value});")
                lines.append("    if (p_err(S)) return NULL;")
            else:
                raise UnsupportedError(t, "unsupported assignment target")
            return lines
        if isinstance(node, ast.AugAssign):
            fn = _BINOP.get(type(node.op))
            if fn is None:
                raise UnsupportedError(node, f"unsupported augassign op {type(node.op).__name__}")
            b = expr(node.value)
            if isinstance(node.target, ast.Name):
                if node.target.id not in frame.assigned:
                    raise UnsupportedError(
                        node.target,
                        f"augassign makes "
                        f"'{node.target.id}' "
                        f"a local but it is "
                        f"never assigned in "
                        f"this scope "
                        f"unsupported",
                    )
                a = _Expr.frame_read(frame, (0, frame.alloc(node.target.id)))
                res = em.tmp()
                lines.append(f"    pval *{res} = {fn}(S, {a}, {b});")
                lines.append(f"    if (!{res}) return NULL;")
                store(node.target.id, res)
            elif isinstance(node.target, ast.Subscript):
                obj = expr(node.target.value)
                key = expr(node.target.slice)
                a = em.tmp()
                lines.append(f"    pval *{a} = p_getitem(S, {obj}, {key});")
                lines.append(f"    if (!{a}) return NULL;")
                res = em.tmp()
                lines.append(f"    pval *{res} = {fn}(S, {a}, {b});")
                lines.append(f"    if (!{res}) return NULL;")
                lines.append(f"    p_setitem(S, {obj}, {key}, {res});")
                lines.append("    if (p_err(S)) return NULL;")
            else:
                raise UnsupportedError(node.target, "unsupported augassign target")
            return lines
        if isinstance(node, ast.Return):
            v = expr(node.value)
            lines.append(f"    return {v};")
            return lines
        if isinstance(node, ast.Pass):
            return lines
        if isinstance(node, ast.Assert):
            if node.msg is not None:
                raise UnsupportedError(node, "assert with message unsupported")
            test = expr(node.test)
            lines.append(
                f"    if (!p_truth(S, {test})) "
                f'{{ p_seterr(S, PE_ASSERT, "assertion failed"); '
                f"return NULL; }}"
            )
            return lines
        if isinstance(node, ast.Break):
            lines.append("    break;")
            return lines
        if isinstance(node, ast.Continue):
            lines.append("    continue;")
            return lines
        if isinstance(node, ast.If):
            return _Stmts.if_(em, frame, node, lines)
        if isinstance(node, ast.While):
            return _Stmts.while_(em, frame, node, lines)
        if isinstance(node, ast.For):
            return _Stmts.for_(em, frame, node, lines)
        if isinstance(node, ast.FunctionDef):
            cname = em.gen_function(node, frame)
            v = em.tmp()
            lines.append(
                f"    pval *{v} = p_mkfunc(S, {cname}, F, "
                f"{len(node.args.args)}, {c_str(node.name)});"
            )
            lines.append(f"    if (!{v}) return NULL;")
            lines.append(f"    p_frame_set(S, F, {frame.alloc(node.name)}, {v});")
            frame.assigned_so_far.add(node.name)
            lines.append("    if (p_err(S)) return NULL;")
            return lines
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.Try,
                ast.TryStar,
                ast.With,
                ast.AsyncWith,
                ast.AsyncFor,
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Global,
                ast.Nonlocal,
                ast.Delete,
                ast.Lambda,
            ),
        ):
            raise UnsupportedError(node, f"{type(node).__name__} statements unsupported")
        raise UnsupportedError(node, f"statement {type(node).__name__} unsupported")

    @staticmethod
    def if_(em, frame, node, lines) -> list[str]:
        t = _Expr.emit(em, frame, node.test, lines)
        lines.append(f"    if (p_truth(S, {t})) {{")
        for s in node.body:
            lines.extend(_Stmts.emit(em, frame, s))
        lines.append("    }")
        orelse = node.orelse
        while orelse and len(orelse) == 1 and isinstance(orelse[0], ast.If):
            sub = orelse[0]
            c = _Expr.emit(em, frame, sub.test, lines)
            lines.append(f"    else if (p_truth(S, {c})) {{")
            for s in sub.body:
                lines.extend(_Stmts.emit(em, frame, s))
            orelse = sub.orelse
            lines.append("    }")
        if orelse:
            lines.append("    else {")
            for s in orelse:
                lines.extend(_Stmts.emit(em, frame, s))
            lines.append("    }")
        return lines

    @staticmethod
    def while_(em, frame, node, lines) -> list[str]:
        lines.append("    for (;;) {")
        cond = _Expr.emit(em, frame, node.test, lines)
        lines.append(f"        if (!p_truth(S, {cond})) break;")
        for s in node.body:
            lines.extend(_Stmts.emit(em, frame, s))
        lines.append("        p_maybe_gc(S);")
        lines.append("    }")
        return lines

    @staticmethod
    def for_(em, frame, node, lines) -> list[str]:
        target = node.target
        if not isinstance(target, ast.Name):
            if isinstance(target, (ast.Tuple, ast.List)):
                raise UnsupportedError(target, "tuple targets in for unsupported")
            raise UnsupportedError(target, "unsupported for target")
        iter = _Expr.emit(em, frame, node.iter, lines)
        i = frame.mark_assigned(target.id)
        it = em.tmp()
        x = em.tmp()
        lines.append(f"    pval *{it} = p_iter(S, {iter});")
        lines.append(f"    if (!{it}) return NULL;")
        lines.append(f"    pval *{x} = NULL;")
        lines.append(f"    while (p_iter_next(S, {it}, &{x}) != 0) {{")
        lines.append(f"        p_frame_set(S, F, {i}, {x});")
        lines.append("        p_maybe_gc(S);")
        for s in node.body:
            lines.extend(_Stmts.emit(em, frame, s))
        lines.append("        p_maybe_gc(S);")
        lines.append("    }")
        lines.append("    if (p_err(S)) return NULL;")
        return lines


class _Expr:
    """Expression emission: appends lines, returns C expr naming a pval."""

    @staticmethod
    def frame_read(frame: _Frame, r: tuple[int, int]) -> str:
        """C read of a name at (depth, slot) from the current frame F."""
        d, i = r
        if d == 0:
            return f"p_frame_get(S, F, {i})"
        base = "c->u.g.env"
        for _ in range(1, d):
            base = f"p_frame_get(S, {base}, 0)"
        return f"p_frame_get(S, {base}, {i})"

    @staticmethod
    def emit(em: Emitter, frame: _Frame, node, lines) -> str:
        if isinstance(node, ast.Constant):
            return _Expr._const(em, node, lines)
        if isinstance(node, ast.Name):
            if node.id in frame.assigned and node.id not in frame.assigned_so_far:
                raise UnsupportedError(
                    node, f"local variable '{node.id}' referenced before assignment"
                )
            r = frame.resolve(node.id)
            if r is None:
                raise UnsupportedError(node, f"name '{node.id}' not defined")
            return _Expr.frame_read(frame, r)
        if isinstance(node, ast.BinOp):
            return _Expr._binop(em, frame, node, lines)
        if isinstance(node, ast.UnaryOp):
            return _Expr._unary(em, frame, node, lines)
        if isinstance(node, ast.BoolOp):
            return _Expr._boolop(em, frame, node, lines)
        if isinstance(node, ast.Compare):
            return _Expr._compare(em, frame, node, lines)
        if isinstance(node, ast.IfExp):
            return _Expr._ifexp(em, frame, node, lines)
        if isinstance(node, ast.Subscript):
            obj = _Expr.emit(em, frame, node.value, lines)
            key = _Expr.emit(em, frame, node.slice, lines)
            t = em.tmp()
            lines.append(f"    pval *{t} = p_getitem(S, {obj}, {key});")
            lines.append(f"    if (!{t}) return NULL;")
            return t
        if isinstance(node, ast.List):
            return _Expr._seq(em, frame, node, lines, "P_LIST")
        if isinstance(node, ast.Tuple):
            return _Expr._seq(em, frame, node, lines, "P_TUPLE")
        if isinstance(node, ast.Dict):
            return _Expr._dict(em, frame, node, lines)
        if isinstance(node, ast.Call):
            return _Expr._call(em, frame, node, lines)
        if isinstance(
            node,
            (
                ast.Lambda,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Set,
                ast.Slice,
                ast.Starred,
                ast.Attribute,
            ),
        ):
            raise UnsupportedError(node, f"{type(node).__name__} expression unsupported")
        raise UnsupportedError(node, f"unsupported expression node {type(node).__name__}")

    @staticmethod
    def _const(em, node, lines) -> str:
        v = node.value
        if v is None:
            return "p_none(S)"
        if isinstance(v, bool):
            return f"p_newbool(S, {'1' if v else '0'})"
        if isinstance(v, int):
            s = str(v)
            t = em.tmp()
            lines.append(f"    pval *{t} = NULL;")
            lines.append(f"    if (!p_int_from_str(S, {c_str(s)}, {len(s)}, &{t})) return NULL;")
            return t
        if isinstance(v, float):
            t = em.tmp()
            lines.append(f"    pval *{t} = p_newfloat(S, {_float_c(v)});")
            return t
        if isinstance(v, str):
            t = em.tmp()
            lines.append(f"    pval *{t} = p_newstrn(S, {c_str(v)});")
            lines.append(f"    if (!{t}) return NULL;")
            return t
        if isinstance(v, (bytes, complex)):
            raise UnsupportedError(node, f"literal of type {type(v).__name__}")
        raise UnsupportedError(node, f"unsupported constant {v!r}")

    @staticmethod
    def _binop(em, frame, node, lines) -> str:
        fn = _BINOP.get(type(node.op))
        if fn is None:
            raise UnsupportedError(node, f"operator {type(node.op).__name__} unsupported")
        a = _Expr.emit(em, frame, node.left, lines)
        b = _Expr.emit(em, frame, node.right, lines)
        t = em.tmp()
        lines.append(f"    pval *{t} = {fn}(S, {a}, {b});")
        lines.append(f"    if (!{t}) return NULL;")
        return t

    @staticmethod
    def _unary(em, frame, node, lines) -> str:
        if isinstance(node.op, (ast.Invert,)):
            raise UnsupportedError(node, "bitwise ~ unsupported")
        a = _Expr.emit(em, frame, node.operand, lines)
        if isinstance(node.op, ast.UAdd):
            return a
        if isinstance(node.op, ast.USub):
            t = em.tmp()
            lines.append(f"    pval *{t} = p_neg(S, {a});")
            lines.append(f"    if (!{t}) return NULL;")
            return t
        if isinstance(node.op, ast.Not):
            t = em.tmp()
            lines.append(f"    pval *{t} = p_newbool(S, !p_truth(S, {a}));")
            return t
        raise UnsupportedError(node, "unsupported unary operator")

    @staticmethod
    def _boolop(em, frame, node, lines) -> str:
        acc = _Expr.emit(em, frame, node.values[0], lines)
        is_and = isinstance(node.op, ast.And)
        for v in node.values[1:]:
            t = em.tmp()
            lines.append(f"    pval *{t} = NULL;")
            lines.append(f"    if (p_truth(S, {acc})) {{")
            if is_and:
                b = _Expr.emit(em, frame, v, lines)
                lines.append(f"        {t} = {b};")
                lines.append("    } else {")
                lines.append(f"        {t} = {acc};")
                lines.append("    }")
            else:
                lines.append(f"        {t} = {acc};")
                lines.append("    } else {")
                b = _Expr.emit(em, frame, v, lines)
                lines.append(f"        {t} = {b};")
                lines.append("    }")
            acc = t
        return acc

    @staticmethod
    def _compare(em, frame, node, lines) -> str:
        def is_none_lit(e) -> bool:
            return isinstance(e, ast.Constant) and e.value is None

        def one(op, a, b, la, ra) -> str:
            t = em.tmp()
            if isinstance(op, (ast.Is, ast.IsNot)):
                if not (is_none_lit(la) or is_none_lit(ra)):
                    raise UnsupportedError(node, "'is'/'is not' supported only with a None literal")
                opc = "==" if isinstance(op, ast.Is) else "!="
                lines.append(f"    pval *{t} = p_newbool(S, ({a}) {opc} ({b}));")
                return t
            if isinstance(op, ast.In):
                lines.append(f"    pval *{t} = p_newbool(S, p_contains(S, {b}, {a}));")
            elif isinstance(op, ast.NotIn):
                lines.append(f"    pval *{t} = p_newbool(S, !p_contains(S, {b}, {a}));")
            else:
                fn = _CMPOP.get(type(op))
                if fn is None:
                    raise UnsupportedError(node, f"comparison {type(op).__name__} unsupported")
                lines.append(f"    pval *{t} = p_newbool(S, {fn}(S, {a}, {b}));")
            lines.append("    if (p_err(S)) return NULL;")
            return t

        left = _Expr.emit(em, frame, node.left, lines)
        acc = None
        prev = left
        prev_ast = node.left
        for i, right in enumerate(node.comparators):
            if i == 0:
                b = _Expr.emit(em, frame, right, lines)
                acc = one(node.ops[i], prev, b, prev_ast, right)
            else:
                t = em.tmp()
                lines.append(f"    pval *{t} = NULL;")
                lines.append(f"    if (p_truth(S, {acc})) {{")
                b = _Expr.emit(em, frame, right, lines)
                r = one(node.ops[i], prev, b, prev_ast, right)
                lines.append(f"        {t} = {r};")
                lines.append("    } else {")
                lines.append(f"        {t} = p_newbool(S, 0);")
                lines.append("    }")
                acc = t
            prev = b
            prev_ast = right
        return acc

    @staticmethod
    def _ifexp(em, frame, node, lines) -> str:
        cond = _Expr.emit(em, frame, node.test, lines)
        t = em.tmp()
        lines.append(f"    pval *{t} = NULL;")
        lines.append(f"    if (p_truth(S, {cond})) {{")
        a = _Expr.emit(em, frame, node.body, lines)
        lines.append(f"        {t} = {a};")
        lines.append("    } else {")
        b = _Expr.emit(em, frame, node.orelse, lines)
        lines.append(f"        {t} = {b};")
        lines.append("    }")
        return t

    @staticmethod
    def _seq(em, frame, node, lines, tag) -> str:
        t = em.tmp()
        lines.append(f"    pval *{t} = p_newobj(S, {tag});")
        lines.append(f"    if (!{t}) return NULL;")
        for e in node.elts:
            v = _Expr.emit(em, frame, e, lines)
            lines.append(f"    p_seq_append(S, {t}, {v});")
            lines.append("    if (p_err(S)) return NULL;")
            lines.append("    p_maybe_gc(S);")
        return t

    @staticmethod
    def _dict(em, frame, node, lines) -> str:
        t = em.tmp()
        lines.append(f"    pval *{t} = p_newobj(S, P_DICT);")
        lines.append(f"    if (!{t}) return NULL;")
        for k, v in zip(node.keys, node.values):
            ke = _Expr.emit(em, frame, k, lines)
            ve = _Expr.emit(em, frame, v, lines)
            lines.append(f"    p_dict_set(S, &{t}->u.d, {ke}, {ve});")
            lines.append("    if (p_err(S)) return NULL;")
            lines.append("    p_maybe_gc(S);")
        return t

    @staticmethod
    def _call(em, frame, node, lines) -> str:
        if isinstance(node.func, ast.Attribute):
            raise UnsupportedError(node, "method calls unsupported")
        if not isinstance(node.func, ast.Name):
            raise UnsupportedError(node, "unsupported callee")
        if node.keywords:
            raise UnsupportedError(node, "keyword arguments unsupported")
        name = node.func.id
        args = [_Expr.emit(em, frame, a, lines) for a in node.args]
        arr = em.tmp()
        init = ", ".join(args) if args else "NULL"
        lines.append(f"    pval *{arr}[{max(1, len(args))}] = {{{init}}};")
        t = em.tmp()
        r = frame.resolve(name)
        if r is not None:
            fv = _Expr.frame_read(frame, r)
            lines.append(f"    pval *{t} = p_call(S, {fv}, {len(args)}, {arr});")
        elif name in BUILTINS:
            lines.append(f"    pval *{t} = p_builtin_call(S, {c_str(name)}, {len(args)}, {arr});")
        else:
            raise UnsupportedError(node, f"function or builtin '{name}' not defined")
        lines.append(f"    if (!{t}) return NULL;")
        return t


def compile_source(source: str) -> str:
    """Compile a Python source string to a full C99 translation unit."""
    return Emitter(source).compile_module()
