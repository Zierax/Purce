"""C99 compilation and ctypes bridge for differential fuzzing.

Compiles generated C99 kernel code into a shared library,
loads it via ctypes, and provides typed call interfaces for each operation.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from purce.backend.c99_generator import MATH_KERNEL_BODIES


@dataclass
class CompiledKernels:
    """Handle to a compiled shared library of C99 kernels."""
    lib: ctypes.CDLL
    lib_path: str
    temp_dir: str

    def close(self) -> None:
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass


_KERNEL_BODIES = MATH_KERNEL_BODIES


_C_SOURCE_HEADER = r"""
#define _USE_MATH_DEFINES
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

"""


def _make_kernel_wrapper(name: str) -> str:
    """Generate a static kernel wrapper with consistent parameter names."""
    body = _KERNEL_BODIES[name]

    if name in ("element_add", "element_sub", "element_mul", "element_div"):
        sig = f"static void kernel_{name}(const double *A, const double *B, double *C, int n)"
    elif name in ("reduce_sum", "reduce_mean", "reduce_max", "reduce_min"):
        sig = f"static void kernel_{name}(const double *x, int n, double *result_ptr)"
    elif name in ("alloc_zeros", "alloc_ones", "alloc_eye"):
        sig = f"static void kernel_{name}(double *out, int n)"
    elif name.startswith("element_"):
        sig = f"static void kernel_{name}(const double *x, double *out, int n)"
    elif name == "matmul":
        sig = f"static void kernel_{name}(const double *A, const double *B, double *C, int m, int n, int p)"
    elif name == "linalg_solve":
        sig = f"static void kernel_{name}(const double *A, const double *b, double *x, int n)"
    elif name == "linalg_inv":
        sig = f"static void kernel_{name}(const double *A, double *inv, int n)"
    elif name == "linalg_cholesky":
        sig = f"static void kernel_{name}(const double *A, double *L, int n)"
    elif name == "linalg_eig":
        sig = f"static void kernel_{name}(const double *A, double *eigenvalues, int n)"
    elif name in ("fft", "ifft"):
        sig = f"static void kernel_{name}(const double *real, const double *imag, double *out_real, double *out_imag, int n, int log_n)"
    else:
        return ""

    indented = "\n".join("    " + line for line in body.split("\n"))
    return f"{sig} {{\n{indented}\n}}\n"


_C_SOURCE_WRAPPERS = r"""

/* ══════════════════════════════════════════════════════════════
 * WRAPPER FUNCTIONS — consistent ctypes-callable interfaces
 * ══════════════════════════════════════════════════════════════ */

void purce_add(const double *A, const double *B, double *C, int n) { kernel_element_add(A, B, C, n); }
void purce_sub(const double *A, const double *B, double *C, int n) { kernel_element_sub(A, B, C, n); }
void purce_mul(const double *A, const double *B, double *C, int n) { kernel_element_mul(A, B, C, n); }
void purce_div(const double *A, const double *B, double *C, int n) { kernel_element_div(A, B, C, n); }

void purce_reduce_sum(const double *x, int n, double *r) { kernel_reduce_sum(x, n, r); }
void purce_reduce_mean(const double *x, int n, double *r) { kernel_reduce_mean(x, n, r); }
void purce_reduce_max(const double *x, int n, double *r) { kernel_reduce_max(x, n, r); }
void purce_reduce_min(const double *x, int n, double *r) { kernel_reduce_min(x, n, r); }

void purce_abs(const double *x, double *o, int n) { kernel_element_abs(x, o, n); }
void purce_sqrt(const double *x, double *o, int n) { kernel_element_sqrt(x, o, n); }
void purce_exp(const double *x, double *o, int n) { kernel_element_exp(x, o, n); }
void purce_log(const double *x, double *o, int n) { kernel_element_log(x, o, n); }
void purce_sin(const double *x, double *o, int n) { kernel_element_sin(x, o, n); }
void purce_cos(const double *x, double *o, int n) { kernel_element_cos(x, o, n); }
void purce_tan(const double *x, double *o, int n) { kernel_element_tan(x, o, n); }

void purce_matmul(const double *A, const double *B, double *C, int m, int n, int p) { kernel_matmul(A, B, C, m, n, p); }

void purce_linalg_solve(const double *A, const double *b, double *x, int n) { kernel_linalg_solve(A, b, x, n); }
void purce_linalg_inv(const double *A, double *inv, int n) { kernel_linalg_inv(A, inv, n); }
void purce_linalg_cholesky(const double *A, double *L, int n) { kernel_linalg_cholesky(A, L, n); }
void purce_linalg_eig(const double *A, double *e, int n) { kernel_linalg_eig(A, e, n); }

void purce_fft(const double *re, const double *im, double *ore, double *oim, int n, int l) { kernel_fft(re, im, ore, oim, n, l); }
void purce_ifft(const double *re, const double *im, double *ore, double *oim, int n, int l) { kernel_ifft(re, im, ore, oim, n, l); }

void purce_zeros(double *o, int n) { kernel_alloc_zeros(o, n); }
void purce_ones(double *o, int n) { kernel_alloc_ones(o, n); }
void purce_eye(double *o, int n) { kernel_alloc_eye(o, n); }
"""


def _generate_c_source() -> str:
    """Generate complete C source with all kernels and wrappers."""
    parts = [_C_SOURCE_HEADER]
    for name in sorted(_KERNEL_BODIES.keys()):
        parts.append(_make_kernel_wrapper(name))
        parts.append("")
    parts.append(_C_SOURCE_WRAPPERS)
    return "\n".join(parts)


def _find_gcc() -> str:
    """Locate gcc compiler."""
    for name in ("gcc", "cc", "x86_64-w64-mingw32-gcc", "mingw32-gcc"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "gcc not found. Install MinGW-w64 or add gcc to PATH."
    )


def compile_kernels() -> CompiledKernels:
    """Compile all C99 kernels into a shared library and load via ctypes."""
    temp_dir = tempfile.mkdtemp(prefix="purce_fuzz_")
    c_path = os.path.join(temp_dir, "kernels.c")
    sysname = platform.system()

    if sysname == "Windows":
        lib_name = "kernels.dll"
        extra_flags = []
    elif sysname == "Darwin":
        lib_name = "libkernels.so"
        extra_flags = ["-fPIC"]
    else:
        lib_name = "libkernels.so"
        extra_flags = ["-fPIC"]

    lib_path = os.path.join(temp_dir, lib_name)
    source = _generate_c_source()
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(source)

    gcc = _find_gcc()
    cmd = [gcc, "-shared", "-O2", "-std=c99"] + extra_flags + ["-o", lib_path, c_path, "-lm"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"gcc failed (exit {result.returncode}):\n{result.stderr}")

    if not os.path.exists(lib_path):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Shared library not found at {lib_path}")

    lib = ctypes.CDLL(lib_path)
    _configure_signatures(lib)
    return CompiledKernels(lib=lib, lib_path=lib_path, temp_dir=temp_dir)


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restypes for all wrapper functions."""
    arr = ctypes.POINTER(ctypes.c_double)
    i32 = ctypes.c_int
    dbl_ptr = ctypes.POINTER(ctypes.c_double)

    binary_fns = ["purce_add", "purce_sub", "purce_mul", "purce_div"]
    for fn_name in binary_fns:
        getattr(lib, fn_name).restype = None
        getattr(lib, fn_name).argtypes = [arr, arr, arr, i32]

    reduce_fns = ["purce_reduce_sum", "purce_reduce_mean", "purce_reduce_max", "purce_reduce_min"]
    for fn_name in reduce_fns:
        getattr(lib, fn_name).restype = None
        getattr(lib, fn_name).argtypes = [arr, i32, dbl_ptr]

    unary_fns = ["purce_abs", "purce_sqrt", "purce_exp", "purce_log", "purce_sin", "purce_cos", "purce_tan"]
    for fn_name in unary_fns:
        getattr(lib, fn_name).restype = None
        getattr(lib, fn_name).argtypes = [arr, arr, i32]

    lib.purce_matmul.restype = None
    lib.purce_matmul.argtypes = [arr, arr, arr, i32, i32, i32]

    lib.purce_linalg_solve.restype = None
    lib.purce_linalg_solve.argtypes = [arr, arr, arr, i32]
    lib.purce_linalg_inv.restype = None
    lib.purce_linalg_inv.argtypes = [arr, arr, i32]
    lib.purce_linalg_cholesky.restype = None
    lib.purce_linalg_cholesky.argtypes = [arr, arr, i32]
    lib.purce_linalg_eig.restype = None
    lib.purce_linalg_eig.argtypes = [arr, arr, i32]

    lib.purce_fft.restype = None
    lib.purce_fft.argtypes = [arr, arr, arr, arr, i32, i32]
    lib.purce_ifft.restype = None
    lib.purce_ifft.argtypes = [arr, arr, arr, arr, i32, i32]

    alloc_fns = ["purce_zeros", "purce_ones", "purce_eye"]
    for fn_name in alloc_fns:
        getattr(lib, fn_name).restype = None
        getattr(lib, fn_name).argtypes = [arr, i32]


def _to_ptr(data):
    """Convert Python list to ctypes double pointer."""
    arr = (ctypes.c_double * len(data))(*data)
    return ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))


def _alloc_out(n):
    """Allocate output array of n doubles, return (array, pointer)."""
    arr = (ctypes.c_double * n)(0.0)
    ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
    return arr, ptr


class CKernelCaller:
    """Typed Python interface to call compiled C kernels."""

    def __init__(self, compiled: CompiledKernels):
        self._lib = compiled.lib

    def element_add(self, a, b, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_add(_to_ptr(a), _to_ptr(b), ptr, n)
        return [out[i] for i in range(n)]

    def element_sub(self, a, b, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_sub(_to_ptr(a), _to_ptr(b), ptr, n)
        return [out[i] for i in range(n)]

    def element_mul(self, a, b, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_mul(_to_ptr(a), _to_ptr(b), ptr, n)
        return [out[i] for i in range(n)]

    def element_div(self, a, b, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_div(_to_ptr(a), _to_ptr(b), ptr, n)
        return [out[i] for i in range(n)]

    def reduce_sum(self, x, n):
        r = ctypes.c_double(0.0)
        self._lib.purce_reduce_sum(_to_ptr(x), n, ctypes.byref(r))
        return r.value

    def reduce_mean(self, x, n):
        r = ctypes.c_double(0.0)
        self._lib.purce_reduce_mean(_to_ptr(x), n, ctypes.byref(r))
        return r.value

    def reduce_max(self, x, n):
        r = ctypes.c_double(0.0)
        self._lib.purce_reduce_max(_to_ptr(x), n, ctypes.byref(r))
        return r.value

    def reduce_min(self, x, n):
        r = ctypes.c_double(0.0)
        self._lib.purce_reduce_min(_to_ptr(x), n, ctypes.byref(r))
        return r.value

    def element_abs(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_abs(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_sqrt(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_sqrt(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_exp(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_exp(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_log(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_log(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_sin(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_sin(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_cos(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_cos(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def element_tan(self, x, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_tan(_to_ptr(x), ptr, n)
        return [out[i] for i in range(n)]

    def matmul(self, a, b, m, n, p):
        out, ptr = _alloc_out(m * n)
        self._lib.purce_matmul(_to_ptr(a), _to_ptr(b), ptr, m, n, p)
        return [out[i] for i in range(m * n)]

    def linalg_solve(self, A, b, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_linalg_solve(_to_ptr(A), _to_ptr(b), ptr, n)
        return [out[i] for i in range(n)]

    def linalg_inv(self, A, n):
        out, ptr = _alloc_out(n * n)
        self._lib.purce_linalg_inv(_to_ptr(A), ptr, n)
        return [out[i] for i in range(n * n)]

    def linalg_cholesky(self, A, n):
        out, ptr = _alloc_out(n * n)
        self._lib.purce_linalg_cholesky(_to_ptr(A), ptr, n)
        return [out[i] for i in range(n * n)]

    def linalg_eig(self, A, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_linalg_eig(_to_ptr(A), ptr, n)
        return [out[i] for i in range(n)]

    def fft(self, real, imag, n):
        log_n = 0
        tmp = n
        while tmp > 1:
            tmp >>= 1
            log_n += 1
        ore, ore_ptr = _alloc_out(n)
        oim, oim_ptr = _alloc_out(n)
        self._lib.purce_fft(_to_ptr(real), _to_ptr(imag), ore_ptr, oim_ptr, n, log_n)
        return [ore[i] for i in range(n)], [oim[i] for i in range(n)]

    def ifft(self, real, imag, n):
        log_n = 0
        tmp = n
        while tmp > 1:
            tmp >>= 1
            log_n += 1
        ore, ore_ptr = _alloc_out(n)
        oim, oim_ptr = _alloc_out(n)
        self._lib.purce_ifft(_to_ptr(real), _to_ptr(imag), ore_ptr, oim_ptr, n, log_n)
        return [ore[i] for i in range(n)], [oim[i] for i in range(n)]

    def alloc_zeros(self, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_zeros(ptr, n)
        return [out[i] for i in range(n)]

    def alloc_ones(self, n):
        out, ptr = _alloc_out(n)
        self._lib.purce_ones(ptr, n)
        return [out[i] for i in range(n)]

    def alloc_eye(self, n):
        out, ptr = _alloc_out(n * n)
        self._lib.purce_eye(ptr, n)
        return [out[i] for i in range(n * n)]
