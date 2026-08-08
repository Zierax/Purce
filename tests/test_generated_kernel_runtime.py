"""Runtime correctness gate for the generated C kernels.

Proves that the fixed kernels compute correct values when actually executed,
not merely that they compile. Each test:

* builds the IR and generates C for a fix-relevant source fragment;
* compiles the generated C into a shared library with gcc;
* binds the kernel signature via ctypes (parsed from the generated C);
* executes the kernel and compares against the numpy reference semantics.

Also covers RNG determinism: the shared LCG must be seeded by noop_seed and
produce identical streams for equal seeds and distinct streams for different
seeds.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess

import numpy as np
import pytest

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder

_HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")


def _generate_c_files(source: str, module_name: str = "rt_mod") -> list[str]:
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    generator = C99Generator(target_profile="generic-c99")
    result = generator.generate(graph, module_name=module_name)
    return [f.content for f in result.files if f.file_type == "c"]


def _compile_shared(c_content: str, tmp_path: object, name: str):
    c_file = tmp_path / f"{name}.c"
    so_file = tmp_path / f"{name}.so"
    c_file.write_text(c_content)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-std=c99", c_file, "-o", so_file, "-lm"],
        check=True,
        capture_output=True,
        text=True,
    )
    return ctypes.CDLL(str(so_file))


def _bind_kernel(lib, c_content: str) -> ctypes._CFuncPtr:
    m = re.search(r"\bvoid\s+(\w+)\s*\((.*?)\)", c_content)
    assert m, f"no kernel signature found in:\n{c_content[:400]}"
    name, params = m.group(1), m.group(2)
    argtypes = []
    for raw in params.split(","):
        p = raw.strip()
        if not p:
            continue
        if re.search(r"\bint\b", p):
            argtypes.append(ctypes.c_int)
        elif "*" in p:
            argtypes.append(ctypes.POINTER(ctypes.c_double))
        else:
            argtypes.append(ctypes.c_double)
    kernel = getattr(lib, name)
    kernel.argtypes = argtypes
    kernel.restype = None
    return kernel


def _double_array(values: list[float]):
    arr = (ctypes.c_double * max(len(values), 1))(*values)
    return arr, ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))


@RequiresGcc
class TestGeneratedKernelRuntime:
    def test_alloc_eye_identity(self, tmp_path) -> None:
        c = _generate_c_files("import numpy as np\ndef f(n: int):\n    return np.eye(n)")[0]
        kernel = _bind_kernel(_compile_shared(c, tmp_path, "eye"), c)
        out, ptr = _double_array([0.0] * 16)
        kernel(4, ptr)
        assert list(out) == np.eye(4).flatten().tolist()

    def test_matrix_diag_from_diagonal_matrix(self, tmp_path) -> None:
        c = _generate_c_files(
            "import numpy as np\ndef f():\n    return np.diag([1.0, 2.0, 3.0])"
        )[0]
        kernel = _bind_kernel(_compile_shared(c, tmp_path, "diag"), c)
        out, ptr = _double_array([0.0] * 9)
        kernel(3, ptr)
        assert list(out) == np.diag([1.0, 2.0, 3.0]).flatten().tolist()

    def test_matrix_diag_extract_diagonal_2d(self, tmp_path) -> None:
        c = _generate_c_files("import numpy as np\ndef f(x):\n    return np.diag(x)")[0]
        kernel = _bind_kernel(_compile_shared(c, tmp_path, "diagext"), c)
        matrix = np.arange(9, dtype=np.float64)
        x, xptr = _double_array(matrix.tolist())
        out, optr = _double_array([0.0] * 3)
        kernel(3, xptr, optr)
        assert list(out) == np.diag(matrix.reshape(3, 3)).tolist()

    def test_array_literal_values(self, tmp_path) -> None:
        c = _generate_c_files(
            "import numpy as np\ndef f():\n    return np.array([1.0, 2.0, 3.0])"
        )[0]
        kernel = _bind_kernel(_compile_shared(c, tmp_path, "arr"), c)
        out, ptr = _double_array([0.0] * 3)
        kernel(3, ptr)
        assert list(out) == [1.0, 2.0, 3.0]

    def test_array_literal_mixed_named_inputs(self, tmp_path) -> None:
        c = _generate_c_files(
            "import numpy as np\ndef f(x, y):\n    return np.array([x, y, 2.5])"
        )[0]
        kernel = _bind_kernel(_compile_shared(c, tmp_path, "arrmix"), c)
        x, xptr = _double_array([1.5, 9.0, 9.0])
        y, yptr = _double_array([3.5, 9.0, 9.0])
        out, optr = _double_array([0.0] * 3)
        kernel(3, xptr, yptr, optr)
        assert list(out) == [1.5, 3.5, 2.5]

    def test_information_gain_scalar_kernels(self, tmp_path) -> None:
        source = (
            "import numpy as np\n"
            "def information_gain(labels, mask):\n"
            "    n_total = labels.shape[0]\n"
            "    n_left = np.sum(mask)\n"
            "    n_right = np.subtract(n_total, n_left)\n"
            "    w_left = np.divide(n_left, n_total)\n"
            "    return np.add(n_right, w_left)\n"
        )
        libs = [_compile_shared(c, tmp_path, f"ig{i}") for i, c in enumerate(_generate_c_files(source))]
        for i, c in enumerate(_generate_c_files(source)):
            kernel = _bind_kernel(libs[i], c)
            if "element_sub" in c:
                out, optr = _double_array([0.0])
                kernel(1, ctypes.c_double(10.0), ctypes.c_double(4.0), optr)
                assert out[0] == pytest.approx(6.0)
            elif "element_div" in c:
                out, optr = _double_array([0.0])
                kernel(1, ctypes.c_double(4.0), ctypes.c_double(10.0), optr)
                assert out[0] == pytest.approx(0.4)

    def test_rng_seeded_determinism(self, tmp_path) -> None:
        source = (
            "import numpy as np\n"
            "def f(x, s: float):\n"
            "    np.random.seed(s)\n"
            "    return np.random.randn(x)\n"
        )
        lib = _compile_shared("\n\n".join(_generate_c_files(source)), tmp_path, "rng")
        files = _generate_c_files(source)
        seed_kernel = None
        rand_kernel = None
        for c in files:
            kernel = _bind_kernel(lib, c)
            if "purce_rng_state = (uint32_t)" in c:
                seed_kernel = kernel
            elif "1103515245u" in c:
                rand_kernel = kernel
        assert seed_kernel is not None and rand_kernel is not None

        n = 8
        x, xptr = _double_array([0.0] * n)
        out1, optr1 = _double_array([0.0] * n)
        out2, optr2 = _double_array([0.0] * n)
        out3, optr3 = _double_array([0.0] * n)

        seed_kernel(ctypes.c_double(42.0), optr1)
        rand_kernel(n, xptr, optr1)
        seed_kernel(ctypes.c_double(42.0), optr2)
        rand_kernel(n, xptr, optr2)
        seed_kernel(ctypes.c_double(7.0), optr3)
        rand_kernel(n, xptr, optr3)

        assert list(out1) == list(out2), "same seed must reproduce identical stream"
        assert list(out1) != list(out3), "different seed must change the stream"
        assert all(0.0 <= v < 1.0 for v in out1), "LCG must stay in [0, 1)"
