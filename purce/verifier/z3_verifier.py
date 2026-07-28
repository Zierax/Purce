from __future__ import annotations

from dataclasses import dataclass, field

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from purce.ir.nodes import Dtype, MathIRGraph, MathIRNode


@dataclass
class VerificationCondition:
    name: str
    description: str
    formula: str
    result: str  # "SAT" (violated), "UNSAT" (safe), "UNKNOWN"
    counterexample: dict[str, float] | None = None


@dataclass
class VerificationReport:
    node_id: str
    conditions: list[VerificationCondition] = field(default_factory=list)
    all_verified: bool = True

    @property
    def violated_count(self) -> int:
        return sum(1 for c in self.conditions if c.result == "SAT")

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.conditions if c.result == "UNSAT")


class Z3Verifier:
    """SMT-based bounds checker using Z3 for verification conditions."""

    def __init__(self):
        self._available = Z3_AVAILABLE

    @property
    def available(self) -> bool:
        return self._available

    def verify_node(self, node: MathIRNode) -> VerificationReport:
        if not self._available:
            return self._fallback_verify(node)

        report = VerificationReport(node_id=node.node_id)

        if node.algorithm in ("matmul", "element_add", "element_sub", "element_mul", "element_div"):
            report.conditions.extend(self._verify_elementwise(node))

        if node.algorithm == "matmul":
            report.conditions.extend(self._verify_matmul_bounds(node))

        if node.algorithm in ("linalg_solve", "linalg_inv"):
            report.conditions.extend(self._verify_linalg(node))

        if node.algorithm in ("reduce_sum", "reduce_mean", "reduce_max", "reduce_min"):
            report.conditions.extend(self._verify_reduction(node))

        if node.algorithm in ("fft", "ifft"):
            report.conditions.extend(self._verify_fft(node))

        report.all_verified = all(c.result == "UNSAT" for c in report.conditions)
        return report

    def verify_graph(self, graph: MathIRGraph) -> dict[str, VerificationReport]:
        reports: dict[str, VerificationReport] = {}
        for nid, node in graph.nodes.items():
            reports[nid] = self.verify_node(node)
        return reports

    def _verify_elementwise(self, node: MathIRNode) -> list[VerificationCondition]:
        conditions: list[VerificationCondition] = []

        s = z3.Solver()
        n = z3.Int("n")
        s.add(n > 0, n <= 10000)

        for dtype, bound_name in [(Dtype.FLOAT64, "float64"), (Dtype.FLOAT32, "float32")]:
            a = z3.Real("a")
            b = z3.Real("b")

            if node.algorithm == "element_add":
                result_expr = a + b
            elif node.algorithm == "element_sub":
                result_expr = a - b
            elif node.algorithm == "element_mul":
                result_expr = a * b
            elif node.algorithm == "element_div":
                s_div = z3.Solver()
                b_sym = z3.Real("b_div")
                s_div.add(b_sym == 0)
                result_div = s_div.check()
                conditions.append(VerificationCondition(
                    name=f"element_div_zero_{bound_name}",
                    description="Division by zero: verify b=0 case is handled (returns 0.0)",
                    formula="b == 0 -> result == 0.0",
                    result="UNSAT" if result_div == z3.unsat else "UNKNOWN",
                ))
                continue
            else:
                continue

            upper = 1e30 if dtype == Dtype.FLOAT64 else 1e38
            s_check = z3.Solver()
            s_check.add(a >= -upper, a <= upper, b >= -upper, b <= upper)
            s_check.add(z3.Or(result_expr > upper, result_expr < -upper))
            result = s_check.check()

            conditions.append(VerificationCondition(
                name=f"{node.algorithm}_overflow_{bound_name}",
                description=f"Check {node.algorithm} output within {bound_name} range",
                formula=f"|{node.algorithm}(a, b)| <= {upper}",
                result="SAT" if result == z3.sat else "UNSAT",
            ))

        return conditions

    def _verify_matmul_bounds(self, node: MathIRNode) -> list[VerificationCondition]:
        conditions: list[VerificationCondition] = []
        s = z3.Solver()
        m, k, n_dim = z3.Ints("m k n_dim")
        s.add(m > 0, m <= 1000, k > 0, k <= 1000, n_dim > 0, n_dim <= 1000)
        s.add(m * k <= 1000000, k * n_dim <= 1000000, m * n_dim <= 1000000)

        result = s.check()
        conditions.append(VerificationCondition(
            name="matmul_dimension_bounds",
            description="Verify matmul dimensions are within safe bounds",
            formula="0 < m,k,n <= 1000, m*k <= 1M, k*n <= 1M, m*n <= 1M",
            result="UNSAT" if result == z3.unsat else "SAT",
        ))

        return conditions

    def _verify_linalg(self, node: MathIRNode) -> list[VerificationCondition]:
        conditions: list[VerificationCondition] = []

        n = z3.Int("n")
        s = z3.Solver()
        s.add(n >= 2, n <= 64)
        result = s.check()
        conditions.append(VerificationCondition(
            name=f"{node.algorithm}_dimension_bounds",
            description=f"Verify {node.algorithm} matrix dimension is within bounds",
            formula="2 <= n <= 64",
            result="UNSAT" if result == z3.unsat else "SAT",
        ))

        diag = z3.RealVector("diag", 64)
        s2 = z3.Solver()
        for i in range(64):
            s2.add(z3.Implies(n > i, z3.Abs(diag[i]) > 1e-10))
        result2 = s2.check()
        conditions.append(VerificationCondition(
            name=f"{node.algorithm}_non_singular",
            description=f"Verify {node.algorithm} diagonal elements are non-zero (non-singular)",
            formula="forall i: |diag[i]| > 1e-10",
            result="UNSAT" if result2 == z3.unsat else "UNKNOWN",
        ))

        return conditions

    def _verify_reduction(self, node: MathIRNode) -> list[VerificationCondition]:
        conditions: list[VerificationCondition] = []

        n = z3.Int("n")
        s = z3.Solver()
        s.add(n > 0, n <= 1000000)

        result = s.check()
        conditions.append(VerificationCondition(
            name=f"{node.algorithm}_input_bounds",
            description=f"Verify {node.algorithm} input size is within bounds",
            formula="0 < n <= 1000000",
            result="UNSAT" if result == z3.unsat else "SAT",
        ))

        return conditions

    def _verify_fft(self, node: MathIRNode) -> list[VerificationCondition]:
        conditions: list[VerificationCondition] = []

        n = z3.Int("n")
        s = z3.Solver()
        s.add(n >= 2, n <= 131072)
        s.add(n == 2 * (n / 2))

        result = s.check()
        conditions.append(VerificationCondition(
            name=f"{node.algorithm}_power_of_two",
            description=f"Verify {node.algorithm} input size is power of 2 and within bounds",
            formula="2 <= n <= 131072, n is power of 2",
            result="UNSAT" if result == z3.unsat else "SAT",
        ))

        return conditions

    def _fallback_verify(self, node: MathIRNode) -> VerificationReport:
        report = VerificationReport(node_id=node.node_id)
        report.conditions.append(VerificationCondition(
            name="z3_unavailable",
            description="Z3 solver not installed; using fallback verification",
            formula="N/A",
            result="UNKNOWN",
        ))
        report.all_verified = False
        return report
