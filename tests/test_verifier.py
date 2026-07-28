import pytest

from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode
from purce.verifier.fuzzer import DifferentialFuzzer, FuzzResult
from purce.verifier.z3_verifier import Z3Verifier


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


class TestDifferentialFuzzerElementwise:
    def test_element_add(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_element_add(iterations=100)
        assert result.operation == "element_add"
        assert result.iterations == 100
        assert result.passed > 0

    def test_element_sub(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_element_sub(iterations=100)
        assert result.operation == "element_sub"
        assert result.passed > 0

    def test_element_mul(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_element_mul(iterations=100)
        assert result.operation == "element_mul"
        assert result.passed > 0

    def test_element_div(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_element_div(iterations=100)
        assert result.operation == "element_div"
        assert result.passed > 0


class TestDifferentialFuzzerReductions:
    def test_reduce_sum(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_reduce_sum(iterations=100)
        assert result.operation == "reduce_sum"
        assert result.passed > 0

    def test_reduce_mean(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_reduce_mean(iterations=100)
        assert result.operation == "reduce_mean"
        assert result.passed > 0

    def test_reduce_max(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_reduce_max(iterations=100)
        assert result.operation == "reduce_max"
        assert result.passed > 0

    def test_reduce_min(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_reduce_min(iterations=100)
        assert result.operation == "reduce_min"
        assert result.passed > 0


class TestDifferentialFuzzerLinearAlgebra:
    def test_linalg_solve(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_linalg_solve(iterations=50)
        assert result.operation == "linalg_solve"
        assert result.passed > 0

    def test_linalg_inv(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_linalg_inv(iterations=50)
        assert result.operation == "linalg_inv"
        assert result.passed > 0


class TestDifferentialFuzzerFFT:
    def test_fft(self) -> None:
        fuzzer = DifferentialFuzzer()
        result = fuzzer.fuzz_fft(iterations=50)
        assert result.operation == "fft"
        assert result.passed > 0


class TestDifferentialFuzzerFuzzAll:
    def test_fuzz_all(self) -> None:
        fuzzer = DifferentialFuzzer()
        results = fuzzer.fuzz_all(iterations=50)
        assert len(results) == 25
        for result in results.values():
            assert result.iterations == 50 or result.iterations > 0


class TestDifferentialFuzzerFuzzResult:
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


class TestZ3Verifier:
    def test_element_add(self) -> None:
        verifier = Z3Verifier()
        node = _make_node("element_add")
        report = verifier.verify_node(node)
        assert report.node_id == "test_element_add"
        assert len(report.conditions) > 0

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

    def test_linalg_solve(self) -> None:
        verifier = Z3Verifier()
        node = _make_node(
            "linalg_solve",
            inputs=[("A", Dtype.FLOAT64, "(n,n)"), ("b", Dtype.FLOAT64, "(n,)")],
            outputs=[("x", Dtype.FLOAT64, "(n,)")],
        )
        report = verifier.verify_node(node)
        assert len(report.conditions) >= 2

    def test_fft(self) -> None:
        verifier = Z3Verifier()
        node = _make_node(
            "fft",
            inputs=[("real", Dtype.FLOAT64, "array"), ("imag", Dtype.FLOAT64, "array")],
            outputs=[("out_real", Dtype.FLOAT64, "array"), ("out_imag", Dtype.FLOAT64, "array")],
        )
        report = verifier.verify_node(node)
        assert len(report.conditions) >= 1

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

    def test_z3_available(self) -> None:
        verifier = Z3Verifier()
        assert verifier.available is True
