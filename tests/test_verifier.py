"""Tests for the differential fuzzer — now actually tests C code via ctypes."""
import pytest

from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode
from purce.verifier.fuzzer import DifferentialFuzzer, FuzzResult
from purce.verifier.z3_verifier import Z3Verifier

import shutil as _shutil
_HAS_GCC = _shutil.which("gcc") is not None or _shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")


def _make_node(
    algorithm: str,
    inputs: list[tuple[str, Dtype, str]] | None = None,
    outputs: list[tuple[str, Dtype, str]] | None = None,
) -> MathIRNode:
    return MathIRNode(
        node_id=f"test_{algorithm}",
        origin_symbol=f"mod.{algorithm}",
        origin_file="test.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="double -> double",
        math_intent=f"Test {algorithm}",
        inputs=inputs or [("x", Dtype.FLOAT64, "array")],
        outputs=outputs or [("result", Dtype.FLOAT64, "array")],
        effects=[Effect.PURE],
        algorithm=algorithm,
        stack_usage=64,
    )


# ── Python-only fuzzing (backward compat) ──

class TestFuzzerPythonOnly:
    def test_element_add_python(self) -> None:
        f = DifferentialFuzzer()
        r = f.fuzz_element_add(iterations=100)
        assert r.all_passed
        assert not r.tested_c

    def test_fuzz_all_python(self) -> None:
        f = DifferentialFuzzer()
        results = f.fuzz_all(iterations=20)
        assert len(results) == 25
        for r in results.values():
            assert r.all_passed


# ── C-backed fuzzing (actual verification) ──

@RequiresGcc
class TestFuzzerCBackendElementwise:
    def test_element_add(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_element_add(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-10
        finally:
            h.close()

    def test_element_sub(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_element_sub(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-10
        finally:
            h.close()

    def test_element_mul(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_element_mul(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-10
        finally:
            h.close()

    def test_element_div(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_element_div(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-10
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendReductions:
    def test_reduce_sum(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_reduce_sum(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-6
        finally:
            h.close()

    def test_reduce_mean(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_reduce_mean(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_reduce_max(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_reduce_max(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_reduce_min(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_reduce_min(iterations=200)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendUnary:
    @pytest.mark.parametrize("op", ["element_sin", "element_cos", "element_tan",
                                     "element_sqrt", "element_exp", "element_log", "element_abs"])
    def test_unary_ops(self, op: str) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            fn = getattr(f, f"fuzz_{op}")
            r = fn(iterations=200)
            assert r.all_passed, f"{op} failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendMatmul:
    def test_matmul(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_matmul(iterations=100)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
            assert r.max_error < 1e-6
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendFFT:
    def test_fft(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_fft(iterations=100)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_ifft(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_ifft(iterations=100)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendLinalg:
    def test_linalg_solve(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_linalg_solve(iterations=50)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_linalg_inv(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_linalg_inv(iterations=50)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_linalg_cholesky(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_linalg_cholesky(iterations=50)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()

    def test_linalg_eig(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            r = f.fuzz_linalg_eig(iterations=50)
            assert r.all_passed, f"Failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendAlloc:
    @pytest.mark.parametrize("op", ["alloc_zeros", "alloc_ones", "alloc_eye"])
    def test_alloc_ops(self, op: str) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            fn = getattr(f, f"fuzz_{op}")
            r = fn(iterations=100)
            assert r.all_passed, f"{op} failures: {r.failures[:3]}"
            assert r.tested_c
        finally:
            h.close()


@RequiresGcc
class TestFuzzerCBackendFuzzAll:
    def test_fuzz_all_with_c(self) -> None:
        f, h = DifferentialFuzzer.with_c_backend(seed=42)
        try:
            results = f.fuzz_all(iterations=50)
            assert len(results) == 25
            failed_ops = []
            for name, r in results.items():
                if not r.all_passed:
                    failed_ops.append(f"{name}: {r.passed}/{r.iterations}")
                assert r.tested_c, f"{name} did not test C code"
            assert not failed_ops, f"Operations failed: {failed_ops}"
        finally:
            h.close()


# ── FuzzResult data class ──

class TestFuzzResult:
    def test_success_rate(self) -> None:
        r = FuzzResult(operation="test", iterations=100, passed=95, failed=5)
        assert r.success_rate == pytest.approx(0.95)
        assert not r.all_passed

    def test_all_passed(self) -> None:
        r = FuzzResult(operation="test", iterations=100, passed=100, failed=0)
        assert r.all_passed

    def test_success_rate_zero_iterations(self) -> None:
        r = FuzzResult(operation="test", iterations=0, passed=0, failed=0)
        assert r.success_rate == 0.0

    def test_error_tracking(self) -> None:
        r = FuzzResult(
            operation="test", iterations=10, passed=8, failed=2,
            max_error=0.05, mean_error=0.01
        )
        assert r.max_error == 0.05
        assert r.mean_error == 0.01


# ── Z3 Verifier ──

class TestZ3Verifier:
    def test_element_add(self) -> None:
        verifier = Z3Verifier()
        node = _make_node("element_add")
        report = verifier.verify_node(node)
        assert report.node_id == "test_element_add"
        assert len(report.conditions) >= 1
        assert report.violated_count + report.verified_count == len(report.conditions)

    def test_matmul(self) -> None:
        verifier = Z3Verifier()
        node = _make_node(
            "matmul",
            inputs=[("A", Dtype.FLOAT64, "(m,k)"), ("B", Dtype.FLOAT64, "(k,n)")],
            outputs=[("C", Dtype.FLOAT64, "(m,n)")],
        )
        report = verifier.verify_node(node)
        assert report.node_id == "test_matmul"
        assert len(report.conditions) >= 1
        assert report.violated_count + report.verified_count == len(report.conditions)

    def test_linalg_solve(self) -> None:
        verifier = Z3Verifier()
        node = _make_node(
            "linalg_solve",
            inputs=[("A", Dtype.FLOAT64, "(n,n)"), ("b", Dtype.FLOAT64, "(n,)")],
            outputs=[("x", Dtype.FLOAT64, "(n,)")],
        )
        report = verifier.verify_node(node)
        assert len(report.conditions) >= 2
        known = report.violated_count + report.verified_count
        unknown = sum(1 for c in report.conditions if c.result == "UNKNOWN")
        assert known + unknown == len(report.conditions)

    def test_fft(self) -> None:
        verifier = Z3Verifier()
        node = _make_node(
            "fft",
            inputs=[("real", Dtype.FLOAT64, "array"), ("imag", Dtype.FLOAT64, "array")],
            outputs=[("out_real", Dtype.FLOAT64, "array"), ("out_imag", Dtype.FLOAT64, "array")],
        )
        report = verifier.verify_node(node)
        assert len(report.conditions) >= 1
        # Power-of-two sizing is enforced by the runtime guard in the generated
        # kernel, which the SMT layer cannot see; the honest verdict is UNKNOWN
        # (with the rejecting counterexample documented), verified via
        # differential fuzzing over the power-of-two domain instead.
        assert report.conditions[0].result == "UNKNOWN"
        assert report.conditions[0].counterexample is not None
        assert not report.all_verified

    def test_verify_graph(self) -> None:
        verifier = Z3Verifier()
        g = MathIRGraph()
        n1 = _make_node("element_add")
        n2 = _make_node("reduce_sum")
        g.add_node(n1)
        g.add_node(n2)
        reports = verifier.verify_graph(g)
        assert len(reports) == 2
        assert "test_element_add" in reports
        assert "test_reduce_sum" in reports
        for report in reports.values():
            assert len(report.conditions) >= 1
            assert report.violated_count + report.verified_count == len(report.conditions)

    def test_z3_available(self) -> None:
        verifier = Z3Verifier()
        assert verifier.available is True
