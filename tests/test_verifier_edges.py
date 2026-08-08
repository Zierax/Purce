"""Edge-case tests for the correctness-verification layer.

Covers missed branches in the differential fuzzer (failure/exception paths in
each ``_fuzz_*_op`` iteration loop, ``_random_*``/``_all_close``/``_max_error``
helpers, singular-pivot and ``n <= 1`` FFT branches), the ctypes bridge
(kernel-wrapper lookups, gcc-missing/compile-failure paths, platform-specific
build config), and the Z3 verifier (remaining elementwise algorithms,
fallback verification, dataclass equality).
"""

from __future__ import annotations

import builtins
import importlib
import random
import shutil

import pytest

import purce.verifier.ctypes_bridge as _ct
import purce.verifier.z3_verifier as _zv

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import DepKind, Dtype, Effect, MathIRGraph, MathIRNode
from purce.slicer.semantic_slicer import SemanticSlicer, SliceResult
from purce.verifier.fuzzer import (
    DifferentialFuzzer,
    FuzzResult,
    _all_close,
    _max_error,
    _random_float,
    _random_int,
)
from purce.verifier.z3_verifier import VerificationCondition, VerificationReport

try:
    import z3 as _z3  # noqa: F401

    _HAS_Z3 = True
except ImportError:  # pragma: no cover - depends on environment
    _HAS_Z3 = False

_HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")
RequiresZ3 = pytest.mark.skipif(not _HAS_Z3, reason="z3-solver not installed")


def _make_node(algorithm: str) -> MathIRNode:
    return MathIRNode(
        node_id=f"edge_{algorithm}",
        origin_symbol=f"mod.{algorithm}",
        origin_file="test.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="double -> double",
        math_intent=f"Edge {algorithm}",
        inputs=[("x", Dtype.FLOAT64, "array")],
        outputs=[("result", Dtype.FLOAT64, "array")],
        effects=[Effect.PURE],
        algorithm=algorithm,
        stack_usage=64,
    )


def _run_pipeline(source: str, module_name: str = "edge_mod"):
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    slice_result = slicer.slice(graph, list(graph.nodes.keys()))
    generator = C99Generator(target_profile="generic-c99")
    gen_result = generator.generate(slice_result.graph, module_name=module_name)
    return gen_result, graph, slice_result


class _WrongBackend:
    """Fake C backend returning deterministically wrong outputs."""

    def __getattr__(self, name: str):
        def _call(*args):
            if name in ("fft", "ifft"):
                return [float(v) + 7.0 for v in args[0]], [float(v) + 7.0 for v in args[1]]
            if name in ("reduce_sum", "reduce_mean", "reduce_max", "reduce_min"):
                return sum(args[0]) + 0.5
            if name == "matmul" or name.startswith("linalg"):
                return [float(v) + 9.0 for v in args[0]]
            if isinstance(args[0], int):
                return [7.0, 7.0, 7.0]
            return [float(v) + 1.0 for v in args[0]]

        return _call


class _RaisingBackend:
    """Fake C backend whose kernels always raise."""

    def __getattr__(self, name: str):
        def _call(*args):
            raise TypeError("synthetic backend failure")

        return _call


# One public fuzz method per distinct `_fuzz_*_op` implementation.
_FUZZ_METHODS = [
    "fuzz_matmul",
    "fuzz_element_add",
    "fuzz_reduce_sum",
    "fuzz_linalg_solve",
    "fuzz_fft",
    "fuzz_element_sin",
    "fuzz_alloc_zeros",
]


class TestFuzzerFailureBranches:
    """Exercise the failed!=0 / failures.append branches of each loop."""

    @pytest.mark.parametrize("method", _FUZZ_METHODS)
    def test_wrong_backend_records_mismatch(self, method: str) -> None:
        fuzzer = DifferentialFuzzer(c_caller=_WrongBackend())
        result = getattr(fuzzer, method)(iterations=4)
        assert result.passed == 0
        assert result.failed == 4
        assert result.success_rate == 0.0
        assert not result.all_passed
        assert result.tested_c
        assert len(result.failures) == 4
        assert all(isinstance(d, dict) for d in result.failures)

    @pytest.mark.parametrize("method", _FUZZ_METHODS)
    def test_raising_backend_records_errors(self, method: str) -> None:
        fuzzer = DifferentialFuzzer(c_caller=_RaisingBackend())
        result = getattr(fuzzer, method)(iterations=3)
        assert result.failed == 3
        assert result.passed == 0
        assert not result.all_passed
        assert result.failures
        assert all(d.get("error") == "synthetic backend failure" for d in result.failures)

    def test_python_only_mode_reports_no_tested_c(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_element_add(iterations=3)
        assert result.tested_c is False
        assert result.all_passed

    def test_fuzz_all_small_iterations(self) -> None:
        fuzzer = DifferentialFuzzer()
        results = fuzzer.fuzz_all(iterations=2)
        assert len(results) == 25
        assert all(r.all_passed for r in results.values())


class TestFuzzerHelpers:
    def test_random_float_deterministic_with_seed(self) -> None:
        random.seed(7)
        first = [_random_float(-5.0, 5.0) for _ in range(5)]
        random.seed(7)
        second = [_random_float(-5.0, 5.0) for _ in range(5)]
        assert first == second
        assert all(-5.0 <= v <= 5.0 for v in first)

    def test_random_float_spans_range(self) -> None:
        random.seed(99)
        values = [_random_float(1.0, 2.0) for _ in range(200)]
        assert all(1.0 <= v <= 2.0 for v in values)
        assert min(values) < 1.5 < max(values)

    def test_random_int_deterministic_with_seed(self) -> None:
        random.seed(3)
        a = [_random_int(1, 16) for _ in range(10)]
        random.seed(3)
        b = [_random_int(1, 16) for _ in range(10)]
        assert a == b

    def test_random_int_inclusive_bounds(self) -> None:
        random.seed(11)
        assert [_random_int(2, 2) for _ in range(5)] == [2, 2, 2, 2, 2]

    def test_all_close_equal_arrays(self) -> None:
        assert _all_close([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1e-5, 1e-8)

    def test_all_close_length_mismatch(self) -> None:
        assert not _all_close([1.0, 2.0], [1.0, 2.0, 3.0], 1e-5, 1e-8)

    def test_all_close_within_tolerance(self) -> None:
        assert _all_close([1.0], [1.0 + 1e-9], 1e-5, 0.0)

    def test_all_close_exceeds_tolerance(self) -> None:
        assert not _all_close([1.0], [2.0], 1e-5, 1e-8)

    def test_all_close_empty_arrays(self) -> None:
        assert _all_close([], [], 1e-5, 1e-8)

    def test_max_error_empty(self) -> None:
        assert _max_error([], []) == 0.0

    def test_max_error_positive(self) -> None:
        assert _max_error([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == 2.0

    def test_max_error_large_relative(self) -> None:
        assert _max_error([1e-6], [100.0]) == pytest.approx(100.0)


class TestFuzzResultProperties:
    def test_eq_and_repr(self) -> None:
        a = FuzzResult(operation="x", iterations=1, passed=1, failed=0)
        b = FuzzResult(operation="x", iterations=1, passed=1, failed=0)
        assert a == b
        assert isinstance(repr(a), str)
        b.failed = 1
        assert a != b

    def test_success_rate_and_all_passed(self) -> None:
        r = FuzzResult(operation="x", iterations=4, passed=3, failed=1)
        assert r.success_rate == pytest.approx(0.75)
        assert not r.all_passed
        assert r.max_error == 0.0
        assert r.mean_error == 0.0


class TestFuzzerPythonReferenceEdges:
    def test_linalg_solve_singular_pivot(self) -> None:
        """Column with a zero pivot walks the `continue` path (line 596)."""
        x = DifferentialFuzzer._linalg_solve_python([[0.0, 1.0], [0.0, 1.0]], [1.0, 2.0], 2)
        assert len(x) == 2

    def test_linalg_inv_singular_pivot(self) -> None:
        """Column with a zero pivot walks the `continue` path (line 618)."""
        inv = DifferentialFuzzer._linalg_inv_python([[0.0, 1.0], [0.0, 1.0]], None, 2)
        assert len(inv) == 4

    def test_fft_trivial_input(self) -> None:
        """n <= 1 short-circuit in `_fft_python` (line 658)."""
        r, i = DifferentialFuzzer._fft_python([1.0], [2.0], 1)
        assert (r, i) == ([1.0], [2.0])

    def test_reduce_python_helpers_shape(self) -> None:
        exp, x = DifferentialFuzzer._reduce_mean_python()
        assert isinstance(exp, float)
        assert len(x) >= 1
        exp, x = DifferentialFuzzer._reduce_max_python()
        assert exp == max(x)


class TestCtypesBridgeHelpers:
    def test_make_kernel_wrapper_unknown_body(self) -> None:
        assert _ct._make_kernel_wrapper("definitely_not_a_kernel") == ""

    def test_make_kernel_wrapper_body_without_paramspec(self) -> None:
        """array_diff has a kernel body but no BODY_PARAM_MAP entry (line 56)."""
        assert _ct._make_kernel_wrapper("array_diff") == ""

    def test_make_kernel_wrapper_valid(self) -> None:
        wrapper = _ct._make_kernel_wrapper("element_add")
        assert "kernel_element_add" in wrapper
        assert wrapper.lstrip().startswith("static void")
        assert "{" in wrapper

    def test_find_gcc_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(_ct.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="gcc not found"):
            _ct._find_gcc()

    def test_compiled_kernels_close_swallows_errors(self, monkeypatch) -> None:
        def _boom(path, **kwargs):
            raise OSError("cannot delete")

        monkeypatch.setattr(shutil, "rmtree", _boom)
        compiled = _ct.CompiledKernels(lib=object(), lib_path="/tmp/x", temp_dir="/tmp/purce_edge_test")
        compiled.close()  # must not raise

    def test_compile_kernels_gcc_error(self, monkeypatch) -> None:
        """gcc exits non-zero -> RuntimeError (lines 172-174)."""

        class _Fail:
            returncode = 1
            stderr = "syntax error\n"

        monkeypatch.setattr(_ct.platform, "system", lambda: "Linux")
        monkeypatch.setattr(_ct.shutil, "which", lambda name: "/fake/gcc")
        monkeypatch.setattr(_ct.shutil, "rmtree", lambda path, **kw: None)
        monkeypatch.setattr(_ct.subprocess, "run", lambda cmd, **kw: _Fail())
        with pytest.raises(RuntimeError, match="gcc failed"):
            _ct.compile_kernels()

    def test_compile_kernels_missing_library(self, monkeypatch) -> None:
        """gcc 'succeeds' but no shared library appears (lines 177-178)."""

        class _Ok:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(_ct.platform, "system", lambda: "Linux")
        monkeypatch.setattr(_ct.shutil, "which", lambda name: "/fake/gcc")
        monkeypatch.setattr(_ct.shutil, "rmtree", lambda path, **kw: None)
        monkeypatch.setattr(_ct.subprocess, "run", lambda cmd, **kw: _Ok())
        with pytest.raises(RuntimeError, match="Shared library not found"):
            _ct.compile_kernels()


class TestCompileKernelsPlatformConfig:
    """Mock gcc/subprocess to exercise platform-specific build branches."""

    def _monkeypatch_compile(self, monkeypatch, sysname: str, calls: dict):
        class _Ok:
            returncode = 0
            stderr = ""

        def fake_run(cmd, **kw):
            calls["cmd"] = list(cmd)
            with open(cmd[-3], "wb") as fh:
                fh.write(b"fake shared lib")
            return _Ok()

        monkeypatch.setattr(_ct.platform, "system", lambda: sysname)
        monkeypatch.setattr(_ct.shutil, "which", lambda name: "/fake/gcc")
        monkeypatch.setattr(_ct.subprocess, "run", fake_run)
        monkeypatch.setattr(_ct, "_configure_signatures", lambda lib: None)
        monkeypatch.setattr(_ct.ctypes, "CDLL", lambda path: object())

    def test_windows_build_config(self, monkeypatch) -> None:
        calls: dict = {}
        self._monkeypatch_compile(monkeypatch, "Windows", calls)
        compiled = _ct.compile_kernels()
        try:
            assert compiled.lib_path.endswith("kernels.dll")
            assert "-fPIC" not in calls["cmd"]
        finally:
            compiled.close()

    def test_darwin_build_config(self, monkeypatch) -> None:
        calls: dict = {}
        self._monkeypatch_compile(monkeypatch, "Darwin", calls)
        compiled = _ct.compile_kernels()
        try:
            assert compiled.lib_path.endswith("libkernels.so")
            assert "-fPIC" in calls["cmd"]
        finally:
            compiled.close()


@RequiresGcc
class TestRealCTypeKernels:
    """Real gcc compile + ctypes round-trip against a tiny kernel set."""

    def test_compile_and_call_tiny_kernels(self) -> None:
        compiled = _ct.compile_kernels()
        try:
            caller = _ct.CKernelCaller(compiled)
            assert caller.element_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 3) == [5.0, 7.0, 9.0]
            assert caller.element_sub([5.0, 5.0], [1.0, 2.0], 2) == [4.0, 3.0]
            assert caller.reduce_sum([1.0, 2.0, 3.0], 3) == pytest.approx(6.0)
            assert caller.reduce_mean([2.0, 4.0, 6.0], 3) == pytest.approx(4.0)
            assert caller.alloc_zeros(3) == [0.0, 0.0, 0.0]
            assert caller.alloc_ones(3) == [1.0, 1.0, 1.0]
            assert caller.alloc_eye(2) == [1.0, 0.0, 0.0, 1.0]
            mm = caller.matmul([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 2, 2, 2)
            assert mm == [19.0, 22.0, 43.0, 50.0]
        finally:
            compiled.close()


@RequiresZ3
class TestZ3ElementwiseAlgorithms:
    def test_element_sub(self) -> None:
        report = _zv.Z3Verifier().verify_node(_make_node("element_sub"))
        assert {c.name for c in report.conditions} == {
            "element_sub_overflow_float64",
            "element_sub_overflow_float32",
        }
        assert all(c.result in ("SAT", "UNSAT") for c in report.conditions)

    def test_element_mul(self) -> None:
        report = _zv.Z3Verifier().verify_node(_make_node("element_mul"))
        assert {c.name for c in report.conditions} == {
            "element_mul_overflow_float64",
            "element_mul_overflow_float32",
        }

    def test_element_div(self) -> None:
        report = _zv.Z3Verifier().verify_node(_make_node("element_div"))
        assert {c.name for c in report.conditions} == {
            "element_div_zero_float64",
            "element_div_zero_float32",
        }
        assert all(c.result in ("UNSAT", "UNKNOWN") for c in report.conditions)

    def test_verify_graph_returns_reports(self) -> None:
        graph = MathIRGraph()
        graph.add_node(_make_node("element_sub"))
        report = _zv.Z3Verifier().verify_graph(graph)
        assert list(report) == ["edge_element_sub"]


class TestZ3DataClasses:
    def test_condition_equality(self) -> None:
        a = VerificationCondition(name="n", description="d", formula="f", result="SAT")
        b = VerificationCondition(name="n", description="d", formula="f", result="SAT")
        assert a == b
        b.result = "UNSAT"
        assert a != b

    def test_report_equality_and_counts(self) -> None:
        r1 = VerificationReport(node_id="n1", all_verified=True)
        r1.conditions.append(VerificationCondition("a", "d", "f", "UNSAT"))
        r1.conditions.append(VerificationCondition("b", "d", "f", "SAT"))
        r2 = VerificationReport(node_id="n1", all_verified=True)
        r2.conditions.append(VerificationCondition("a", "d", "f", "UNSAT"))
        r2.conditions.append(VerificationCondition("b", "d", "f", "SAT"))
        assert r1 == r2
        assert r1.verified_count == 1
        assert r1.violated_count == 1
        c = VerificationCondition("c", "d", "f", "UNKNOWN")
        assert c.result == "UNKNOWN"


class TestZ3Fallback:
    def test_fallback_when_import_fails(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "z3" or (isinstance(name, str) and name.startswith("z3.")):
                raise ImportError("z3 disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            importlib.reload(_zv)
            assert _zv.Z3_AVAILABLE is False
            verifier = _zv.Z3Verifier()
            assert verifier.available is False
            report = verifier.verify_node(_make_node("element_add"))
            assert report.all_verified is False
            assert len(report.conditions) == 1
            cond = report.conditions[0]
            assert cond.name == "z3_unavailable"
            assert cond.result == "UNKNOWN"
        finally:
            monkeypatch.undo()
            importlib.reload(_zv)
            assert _zv.Z3_AVAILABLE is True


class TestGraphNodeEdges:
    def test_reachable_from_skips_missing_entries(self) -> None:
        graph = MathIRGraph()
        graph.add_node(_make_node("a"))
        sub = graph.reachable_from(["missing"])
        assert len(sub.nodes) == 0

    def test_reachable_from_tracks_known_entries(self) -> None:
        graph = MathIRGraph()
        base = _make_node("a")
        follower = _make_node("b")
        follower.nested_deps = [base.node_id]
        graph.add_node(base)
        graph.add_node(follower)
        sub = graph.reachable_from([follower.node_id])
        assert set(sub.nodes) == {base.node_id, follower.node_id}


class TestSlicerEdgeLines:
    def test_classify_and_add_skips_missing_node(self) -> None:
        slicer = SemanticSlicer()
        out = SliceResult(graph=MathIRGraph())
        slicer._classify_and_add(MathIRGraph(), {"ghost"}, out)
        assert len(out.graph.nodes) == 0
        assert out.pal_stubs == []
        assert out.data_assets == []

    def test_classify_impure_kernel_becomes_system_pal(self) -> None:
        node = _make_node("kernel")
        node.effects = [Effect.RANDOM]
        slicer = SemanticSlicer()
        assert slicer._classify(node) == DepKind.SYSTEM_PAL


@RequiresGcc
class TestPipelineVerificationSmoke:
    def test_pipeline_against_small_kernel(self) -> None:
        source = "import numpy as np\ndef add_arrays(a, b):\n    return np.add(a, b)"
        gen, graph, slice_result = _run_pipeline(source, "edge_mod")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert c_files, "no C emitted for np.add pipeline"
        assert any(n.algorithm == "element_add" for n in slice_result.graph.nodes.values())


class TestSemanticSoundnessFixes:
    """Regression tests for the five known semantic-soundness defects.

    Each test inspects the generated C to prove the fix is actually emitted,
    not just that the pipeline runs.
    """

    @staticmethod
    def _generate(source: str, module_name: str = "fix_mod"):
        builder = MathIRBuilder(origin_file=module_name)
        graph = builder.build_from_source(source, module=module_name)
        generator = C99Generator(target_profile="generic-c99")
        gen_result = generator.generate(graph, module_name=module_name)
        return graph, gen_result

    @staticmethod
    def _c_files(gen_result) -> list:
        return [f for f in gen_result.files if f.file_type == "c"]

    def test_alloc_eye_zero_fills_before_diagonal(self) -> None:
        """np.eye must fully zero-fill n*n cells, then set the diagonal."""
        _, gen = self._generate("import numpy as np\ndef f(n: int):\n    return np.eye(n)")
        c = self._c_files(gen)[0].content
        zero_fill = c.index("for (int i = 0; i < n * n; i++)")
        diag_set = c.index("= 1.0;")
        assert zero_fill < diag_set
        assert "result[i] = 0.0;" in c

    def test_array_literal_emits_element_assignments(self) -> None:
        """np.array([1.0, 2.0, 3.0]) must copy its literal elements into out."""
        _, gen = self._generate("import numpy as np\ndef f():\n    return np.array([1.0, 2.0, 3.0])")
        c = self._c_files(gen)[0].content
        assert "result[0] = 1;" in c
        assert "result[1] = 2;" in c
        assert "result[2] = 3;" in c

    def test_matrix_diag_from_constructs_diagonal_matrix(self) -> None:
        """np.diag([...]) must build an n x n diagonal matrix, zero-filled first."""
        graph, gen = self._generate(
            "import numpy as np\ndef f():\n    return np.diag([1.0, 2.0, 3.0])"
        )
        node = next(n for n in graph.nodes.values())
        assert node.algorithm == "matrix_diag_from"
        c = self._c_files(gen)[0].content
        zero_fill = c.index("for (int i = 0; i < n * n; i++)")
        diag_set = c.index("result[1 * n + 1] = 2;")
        assert zero_fill < diag_set
        assert "result[0 * n + 0] = 1;" in c

    def test_information_gain_scalar_ops_are_not_indexed(self) -> None:
        """Scalar intermediates must not be array-indexed in generated kernels."""
        source = (
            "import numpy as np\n"
            "def information_gain(labels, mask):\n"
            "    n_total = labels.shape[0]\n"
            "    n_left = np.sum(mask)\n"
            "    n_right = np.subtract(n_total, n_left)\n"
            "    w_left = np.divide(n_left, n_total)\n"
            "    return np.add(n_right, w_left)\n"
        )
        graph, gen = self._generate(source)
        for node in graph.nodes.values():
            if node.algorithm in ("element_sub", "element_div", "element_mul"):
                assert all(s == "scalar" for _, _, s in node.inputs)
        for f in gen.files:
            if f.file_type != "c":
                continue
            c = f.content
            for scalar_name in ("n_total", "_inter_n_left", "_inter_n_right"):
                assert f"{scalar_name}[i]" not in c
                assert f"{scalar_name}[0]" not in c

    def test_random_seed_wires_shared_rng_state(self) -> None:
        """np.random.seed(s) must feed the same LCG state the randn kernel uses."""
        source = (
            "import numpy as np\n"
            "def f(x, s: float):\n"
            "    np.random.seed(s)\n"
            "    return np.random.randn(x)\n"
        )
        graph, gen = self._generate(source)
        algos = {n.algorithm for n in graph.nodes.values()}
        assert {"noop_seed", "alloc_random"} <= algos
        seed_file = next(f for f in gen.files if f.file_type == "c"
                         and "purce_rng_state = (uint32_t)" in f.content)
        rand_file = next(f for f in gen.files if f.file_type == "c"
                         and "purce_rng_state * 1103515245u" in f.content)
        assert "purce_rng_state = (uint32_t)s;" in seed_file.content
        assert "static uint32_t purce_rng_state" in seed_file.content
        assert "static uint32_t purce_rng_state" in rand_file.content


@RequiresGcc
class TestGeneratedKernelsCompileClean:
    """Every generated kernel from the fixed code paths must pass gcc -Wall -Werror."""

    _CASES = {
        "eye": "import numpy as np\ndef f(n: int):\n    return np.eye(n)",
        "diag_vec": "import numpy as np\ndef f(x):\n    return np.diag(x)",
        "diag_lit": "import numpy as np\ndef f():\n    return np.diag([1.0, 2.0, 3.0])",
        "array_lit": "import numpy as np\ndef f():\n    return np.array([1.0, 2.0, 3.0])",
        "array_mixed": "import numpy as np\ndef f(x, y):\n    return np.array([x, y, 2.5])",
        "seed_rng": (
            "import numpy as np\n"
            "def f(x, s: float):\n"
            "    np.random.seed(s)\n"
            "    return np.random.randn(x)"
        ),
        "info_gain": (
            "import numpy as np\n"
            "def information_gain(labels, mask):\n"
            "    n_total = labels.shape[0]\n"
            "    n_left = np.sum(mask)\n"
            "    n_right = np.subtract(n_total, n_left)\n"
            "    return n_right"
        ),
    }

    def test_all_generated_c_files_compile(self, tmp_path) -> None:
        import subprocess

        for name, source in self._CASES.items():
            _, gen = TestSemanticSoundnessFixes._generate(source, module_name=f"fix_{name}")
            for cf in (f for f in gen.files if f.file_type == "c"):
                path = tmp_path / f"{name}_{cf.path}"
                path.write_text(cf.content)
                result = subprocess.run(
                    ["gcc", "-std=c99", "-Wall", "-Werror", "-fsyntax-only", "-x", "c", str(path)],
                    capture_output=True,
                    text=True,
                )
                assert result.returncode == 0, (
                    f"{name} :: {cf.path} failed:\n{result.stderr}"
                )