"""End-to-end semantic equivalence engine: generated C99 vs NumPy.

For every kernel that the pipeline emits from a real-world source, this
engine *executes* the compiled C and compares its output against a NumPy
reference that reproduces the intended high-level semantics of the operation.

Unlike ``purce.verifier.fuzzer`` (which exercises the hand-written
``MATH_KERNEL_BODIES`` in isolation), this module drives the *generated*
kernels for an entire source module: pipeline -> C99 -> gcc -shared ->
ctypes -> execute -> compare.

The result is a reproducible per-node / per-source / per-algorithm report
that quantifies exactly how much of the "high-level" Python intent is
realised by the "low-level" generated C99 — the breakthrough metric.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from purce.backend.c99_generator import (
    BODY_PARAM_MAP,
    C99Generator,
    _sanitize_name,
)
from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import MathIRNode

DELIM = "// ==== KERNEL ==== "

# ── classification of kernels whose generated C is not intended to match
#    a plain NumPy op (structural stubs, RNG streams, or pure allocation).
STRUCTURAL_GAP = {
    "linalg_qr",       # documented simplified stub
    "linalg_svd",      # documented simplified stub
    "array_split",     # documented stub
    "linalg_eig",      # Gershgorin center-only approximation (intentional)
    "alloc_random",    # LCG stream, not numpy.random distribution
    "array_permutation",  # LCG-backed shuffle
    "noop_seed",       # state-only
    "element_finfo",   # constant, not an array op
}


class _UnsupportedAlgorithm(Exception):
    pass


class _NoReference(Exception):
    pass


# ── signature parsing ────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    name: str
    ctype: str
    is_pointer: bool
    is_const: bool


_CTYPE_MAP = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "int": ctypes.c_int,
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
    "uint32_t": ctypes.c_uint32,
}


def parse_kernel_signature(c_content: str) -> tuple[str, list[ParamSpec]]:
    """Parse ``void name(type a, const double * b)`` from generated C."""
    m = re.search(r"^void\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", c_content, re.MULTILINE)
    if not m:
        raise _UnsupportedAlgorithm("no kernel signature in generated C")
    name, params_str = m.groups()
    params: list[ParamSpec] = []
    for raw in params_str.split(","):
        raw = raw.strip()
        if not raw:
            continue
        is_const = "const" in raw
        is_pointer = "*" in raw
        tokens = raw.replace("const", " ").replace("*", " ").split()
        ctype = tokens[0] if tokens else "double"
        pname = tokens[-1] if tokens else "p"
        params.append(ParamSpec(name=pname, ctype=ctype, is_pointer=is_pointer,
                                is_const=is_const))
    return name, params


def ctype_of(spec: ParamSpec):
    return _CTYPE_MAP.get(spec.ctype, ctypes.c_double)


# ── input generation workloads (shapes/ranges per algorithm) ──────────

def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(lo, hi)


def _mat_shape(rng: random.Random):
    return _rand_int(rng, 2, 6)


# ── NumPy references: exact intended high-level semantics ────────────

def _ref_element_binary(arrays, scalars, op) -> np.ndarray:
    a = arrays["A"]
    b = arrays["B"]
    if op == "div":
        return np.where(b != 0.0, a / b, 0.0)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "max":
        return np.maximum(a, b)
    if op == "min":
        return np.minimum(a, b)
    if op == "power":
        return np.power(a, b)
    if op == "where":
        return np.where(arrays["cond"] != 0.0, a, b)
    if op == "greater":
        return (a > b).astype(np.float64)
    if op == "less":
        return (a < b).astype(np.float64)
    if op == "logaddexp":
        return np.logaddexp(a, b)
    if op == "isclose":
        return (np.abs(a - b) <= 1e-8).astype(np.float64)
    raise _NoReference(op)


def _ref_element_unary(arrays, scalars, op) -> np.ndarray:
    x = arrays["x"]
    if op == "abs":
        return np.abs(x)
    if op == "sqrt":
        return np.sqrt(x)
    if op == "exp":
        return np.exp(x)
    if op == "log":
        return np.log(x)
    if op == "sin":
        return np.sin(x)
    if op == "cos":
        return np.cos(x)
    if op == "tan":
        return np.tan(x)
    if op == "tanh":
        return np.tanh(x)
    if op == "neg":
        return -x
    if op == "sign":
        return np.sign(x).astype(np.float64)
    if op == "floor":
        return np.floor(x)
    if op == "log10":
        return np.log10(x)
    if op == "conj":
        return x.copy()
    if op == "angle":
        return np.where(x < 0.0, math.pi, 0.0)
    if op == "real":
        return x.copy()
    if op == "imag":
        return np.zeros_like(x)
    if op == "copy":
        return x.copy()
    if op == "round":
        return np.floor(x + 0.5)
    if op == "ceil":
        return np.ceil(x)
    if op == "trunc":
        return np.trunc(x)
    if op == "isnan":
        return np.isnan(x).astype(np.float64)
    if op == "isinf":
        return (np.abs(x) > 1e308).astype(np.float64)
    raise _NoReference(op)


def _ref_clip(arrays, scalars) -> np.ndarray:
    return np.clip(arrays["x"], arrays["lo"], arrays["hi"])


def _ref_reduce(arrays, scalars, op) -> float:
    x = arrays["x"]
    if op in ("sum", "mean", "max", "min", "prod"):
        fn = {"sum": np.sum, "mean": np.mean, "max": np.max,
              "min": np.min, "prod": np.prod}[op]
        return float(fn(x))
    if op == "var":
        return float(np.mean((x - np.mean(x)) ** 2))
    if op == "argmax":
        return float(np.argmax(x))
    if op == "argmin":
        return float(np.argmin(x))
    if op == "any":
        return 1.0 if np.any(x != 0.0) else 0.0
    if op == "all":
        return 1.0 if np.all(x != 0.0) else 0.0
    if op == "cumsum":
        return np.cumsum(x)
    if op == "diff":
        out = np.empty_like(x)
        out[0] = x[0]
        out[1:] = np.diff(x)
        return out
    raise _NoReference(op)


def _ref_matmul(arrays, scalars) -> np.ndarray:
    m = int(scalars["m"])
    k = int(scalars["k"])
    p = int(scalars["p"])
    A = arrays["A"].reshape(m, k)
    B = arrays["B"].reshape(k, p)
    return (A @ B).ravel()


def _ref_alloc(arrays, scalars, op) -> np.ndarray:
    n = int(scalars["n"])
    if op == "zeros":
        return np.zeros(n)
    if op == "ones":
        return np.ones(n)
    if op == "eye":
        return np.eye(n).ravel()
    if op == "arange":
        return np.arange(n, dtype=np.float64)
    if op == "linspace":
        if n <= 1:
            return np.zeros(n)
        return np.linspace(0.0, 1.0, n)
    if op == "full":
        return np.ones(n)
    raise _NoReference(op)


def _ref_transpose(arrays, scalars) -> np.ndarray:
    rows = int(scalars["rows"])
    cols = int(scalars["cols"])
    return arrays["x"].reshape(rows, cols).T.ravel()


def _ref_outer(arrays, scalars) -> np.ndarray:
    return np.outer(arrays["A"], arrays["B"]).ravel()


def _ref_matrix_diag(arrays, scalars, op) -> np.ndarray:
    n = int(scalars["n"])
    if op == "diag":
        return np.diag(arrays["x"].reshape(n, n)).copy()
    if op == "diag_from":
        out = np.zeros(n * n)
        out[:: n + 1] = arrays["x"]
        return out
    if op == "tril":
        return np.tril(arrays["x"].reshape(n, n)).ravel()
    if op == "triu":
        return np.triu(arrays["x"].reshape(n, n)).ravel()
    raise _NoReference(op)


def _ref_linalg(arrays, scalars, op) -> np.ndarray:
    n = int(scalars["n"])
    A = arrays["A"].reshape(n, n)
    if op == "solve":
        b = arrays["b"][:n]
        return np.linalg.solve(A, b)
    if op == "inv":
        return np.linalg.inv(A).ravel()
    if op == "cholesky":
        return np.linalg.cholesky(A).ravel()
    if op == "det":
        return np.array([np.linalg.det(A)])
    if op == "norm":
        return np.array([np.linalg.norm(arrays["x"])])
    raise _NoReference(op)


def _ref_fft(arrays, scalars, op) -> tuple[np.ndarray, np.ndarray]:
    n = int(scalars["n"])
    sig = arrays["real"] + 1j * arrays["imag"]
    if op == "fft":
        f = np.fft.fft(sig)
    else:
        f = np.fft.ifft(sig)
    return f.real, f.imag


def _ref_array_ops(arrays, scalars, op) -> np.ndarray:
    x = arrays["x"]
    if op == "concat":
        return np.concatenate([arrays["A"], arrays["B"]])
    if op == "flip":
        return x[::-1].copy()
    if op == "reshape" or op == "flatten" or op == "squeeze" or op == "expand_dims":
        return x.copy()
    if op == "sort":
        return np.sort(x)
    if op == "argsort":
        return np.argsort(x).astype(np.float64)
    if op == "roll":
        shift = int(arrays["shift"][0])
        return np.roll(x, shift)
    if op == "tile":
        reps = int(arrays["reps"][0])
        return np.tile(x, reps)
    if op == "repeat":
        reps = int(arrays["reps"][0])
        return np.repeat(x, reps)
    if op == "searchsorted":
        return np.array([np.searchsorted(x, arrays["v"][0], side="left")])
    if op == "take":
        idx = arrays["idx"].astype(np.int64)
        out = np.zeros_like(x)
        valid = (idx >= 0) & (idx < int(scalars["k"]))
        out[valid] = x[idx[valid]]
        return out
    if op == "unique":
        out, counts = np.unique(x, return_counts=True)
        return out, counts.astype(np.float64)
    raise _NoReference(op)


# ── comparison (IEEE-aware, mirroring fuzzer._all_close) ─────────────

def _all_close(a, b, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return False
    nan_both = np.isnan(a) & np.isnan(b)
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    inf_a = np.isinf(a)
    inf_b = np.isinf(b)
    bad = a_nan ^ b_nan
    bad = bad | ((inf_a | inf_b) & (inf_a != inf_b))
    ok_sign = (a > 0) == (b > 0)
    bad = bad | ((inf_a & inf_b) & ~ok_sign)
    finite = ~(a_nan | b_nan | inf_a | inf_b)
    bad = bad | finite & (np.abs(a - b) > atol + rtol * np.abs(b))
    return bool(np.all(~bad))


def _max_error(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    nan_mismatch = (np.isnan(a) != np.isnan(b)) | (np.isinf(a) != np.isinf(b))
    if np.any(nan_mismatch):
        return float("inf")
    d = np.abs(a - b)
    d[np.isnan(d)] = 0.0
    return float(np.max(d)) if d.size else 0.0


# ── per-kernel execution ─────────────────────────────────────────────

@dataclass
class KernelResult:
    node_id: str
    algorithm: str
    iterations: int
    passed: int
    failed: int
    max_error: float
    mean_error: float
    error: str = ""
    tested: bool = False


def _expected_outputs(node: MathIRNode) -> list[str]:
    return [o[0] for o in node.outputs]


class GeneratedKernelExecutor:
    """Compile a module's generated C once, then execute each kernel."""

    def __init__(self, c_files: list[str], algo_by_node: dict[str, str],
                 keep_dir: str | None = None):
        self._c_files = c_files
        self._algo_by_node = algo_by_node
        self._gcc = self._find_gcc()
        self._lib = None
        self._keep_dir = keep_dir

    @staticmethod
    def _find_gcc() -> str:
        for name in ("gcc", "cc"):
            path = shutil.which(name)
            if path:
                return path
        raise RuntimeError("no C compiler on PATH")

    def build(self) -> None:
        tmp = self._keep_dir or tempfile.mkdtemp(prefix="purce_eq_")
        c_path = os.path.join(tmp, "kernels.c")
        with open(c_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(self._c_files))
        ext = ".dll" if platform.system() == "Windows" else ".so"
        lib_path = os.path.join(tmp, "kernels" + ext)
        cmd = [self._gcc, "-shared", "-fPIC", "-O2", "-std=c99"]
        if platform.system() == "Windows":
            cmd = [self._gcc, "-shared", "-O2", "-std=c99"]
        cmd += [c_path, "-o", lib_path, "-lm"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(
                f"gcc failed ({res.returncode}):\n{res.stderr[:800]}"
            )
        self._lib = ctypes.CDLL(lib_path)

    def close(self) -> None:
        self._lib = None

    def run_kernel(self, node: MathIRNode, c_content: str, seed: int,
                   iterations: int = 25,
                   rtol: float = 1e-5, atol: float = 1e-8) -> KernelResult:
        algo = node.algorithm
        result = KernelResult(node_id=node.node_id, algorithm=algo,
                              iterations=iterations, passed=0, failed=0,
                              max_error=0.0, mean_error=0.0)
        try:
            fname, params = parse_kernel_signature(c_content)
            driver = _make_driver(algo, node, self._lib, fname, params)
        except (_UnsupportedAlgorithm, _NoReference) as e:
            result.error = str(e)
            return result

        rng = random.Random(seed)
        errors: list[float] = []
        for _ in range(iterations):
            try:
                inputs, out_meta = driver.make_case(rng)
                expected = driver.reference(inputs)
                actual = driver.call(inputs, out_meta)
                if _all_close(actual, expected, rtol, atol):
                    result.passed += 1
                else:
                    result.failed += 1
                    result.failed = result.failed
                    errors.append(-1.0)
                err = _max_error(actual, expected)
                if err is not None and err != float("inf"):
                    errors.append(err)
            except Exception as e:  # any infra failure on the case
                result.failed += 1
                result.error = f"{result.error}; {e}"[:400]
        if errors:
            ferr = [e for e in errors if e >= 0.0]
            result.max_error = max(ferr) if ferr else 0.0
            result.mean_error = sum(ferr) / len(ferr) if ferr else 0.0
        result.tested = result.failed == 0 and result.error == ""
        return result