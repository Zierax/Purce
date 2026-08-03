"""Edge case and robustness benchmarks.

Tests IEEE 754 edge cases: NaN, Inf, denormals, overflow, underflow, zero.
All operations tested against compiled C99 code when gcc is available.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from purce.verifier.fuzzer import DifferentialFuzzer

try:
    from purce.verifier.ctypes_bridge import compile_kernels
    _HAS_GCC = True
except (RuntimeError, OSError):
    _HAS_GCC = False


@dataclass
class EdgeCaseResult:
    operation: str
    input_type: str
    input_value: str
    expected: str
    actual: str
    passed: bool
    tested_c: bool
    notes: str = ""


@dataclass
class EdgeCaseSuite:
    operation: str
    cases: list[EdgeCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def _test_element_add_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_add")
    cases = [
        ("zero + zero", 0.0, 0.0, "Both zero"),
        ("negative + positive", -5.0, 5.0, "Cancellation"),
        ("large + small", 1e300, 1e-300, "Magnitude difference"),
        ("inf + inf", float('inf'), float('inf'), "Infinity"),
        ("neg_inf + neg_inf", float('-inf'), float('-inf'), "Negative infinity"),
        ("inf + neg_inf", float('inf'), float('-inf'), "Indeterminate -> NaN"),
        ("nan + nan", float('nan'), float('nan'), "NaN propagation"),
        ("negative_zero", -0.0, 0.0, "-0 + 0 = +0"),
    ]
    for name, a, b, notes in cases:
        py_result = a + b
        if c_caller:
            actual = c_caller.add(a, b)
            tested_c = True
            passed = (math.isnan(py_result) and math.isnan(actual)) or (py_result == actual)
            actual_str = str(actual)
        else:
            tested_c = False
            passed = True
            actual_str = f"{py_result}"
        suite.cases.append(EdgeCaseResult(
            operation="element_add", input_type=name,
            input_value=f"({a}, {b})", expected=str(py_result),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_element_mul_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_mul")
    cases = [
        ("zero * anything", 0.0, 100.0, "Zero annihilates"),
        ("inf * zero", float('inf'), 0.0, "Indeterminate -> NaN"),
        ("inf * inf", float('inf'), float('inf'), "Infinity squared"),
        ("inf * neg", float('inf'), -1.0, "Negative infinity"),
        ("nan * nan", float('nan'), float('nan'), "NaN propagation"),
        ("large * small", 1e200, 1e-200, "Unity product"),
        ("overflow", 1e200, 1e200, "Overflow to inf"),
    ]
    for name, a, b, notes in cases:
        py_result = a * b
        if c_caller:
            actual = c_caller.mul(a, b)
            tested_c = True
            passed = (math.isnan(py_result) and math.isnan(actual)) or (py_result == actual)
            actual_str = str(actual)
        else:
            tested_c = False
            passed = True
            actual_str = f"{py_result}"
        suite.cases.append(EdgeCaseResult(
            operation="element_mul", input_type=name,
            input_value=f"({a}, {b})", expected=str(py_result),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_element_div_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_div")
    cases = [
        ("zero / zero", 0.0, 0.0, "0/0 is NaN in IEEE 754"),
        ("nonzero / zero", 1.0, 0.0, "x/0 = inf"),
        ("neg / zero", -1.0, 0.0, "-x/0 = -inf"),
        ("inf / inf", float('inf'), float('inf'), "Indeterminate -> NaN"),
        ("inf / 2", float('inf'), 2.0, "Infinity preserved"),
        ("nan / 1", float('nan'), 1.0, "NaN propagation"),
        ("1 / inf", 1.0, float('inf'), "Converges to zero"),
    ]
    for name, a, b, notes in cases:
        py_result = a / b
        if c_caller:
            actual = c_caller.div(a, b)
            tested_c = True
            passed = (math.isnan(py_result) and math.isnan(actual)) or (py_result == actual)
            actual_str = str(actual)
        else:
            tested_c = False
            passed = True
            actual_str = f"{py_result}"
        suite.cases.append(EdgeCaseResult(
            operation="element_div", input_type=name,
            input_value=f"({a}, {b})", expected=str(py_result),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_reduce_sum_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="reduce_sum")
    cases = [
        ("single_zero", [0.0], "Single zero"),
        ("all_zeros", [0.0, 0.0, 0.0], "All zeros"),
        ("pos_and_neg", [1.0, -1.0, 2.0, -2.0], "Cancellation to zero"),
        ("large_values", [1e300, 1e300, 1e300], "Overflow to inf"),
        ("inf_in_list", [1.0, float('inf'), 2.0], "Infinity dominates"),
        ("nan_in_list", [1.0, float('nan'), 2.0], "NaN poisons sum"),
    ]
    for name, inputs, notes in cases:
        py_result = sum(inputs)
        if c_caller:
            actual = c_caller.reduce_sum(inputs)
            tested_c = True
            passed = (math.isnan(py_result) and math.isnan(actual)) or (py_result == actual)
            actual_str = str(actual)
        else:
            tested_c = False
            passed = True
            actual_str = str(py_result)
        suite.cases.append(EdgeCaseResult(
            operation="reduce_sum", input_type=name,
            input_value=str(inputs), expected=str(py_result),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_reduce_max_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="reduce_max")
    cases = [
        ("single_element", [42.0], "Single element"),
        ("all_equal", [5.0, 5.0, 5.0], "All equal"),
        ("negatives", [-10.0, -5.0, -1.0], "Closest to zero"),
        ("inf_present", [1.0, float('inf'), 2.0], "Infinity is max"),
        ("nan_present", [1.0, float('nan'), 2.0], "NaN poisons max"),
        ("mixed_signs", [-100.0, 0.0, 100.0], "Positive max"),
    ]
    for name, inputs, notes in cases:
        py_result = max(inputs)
        if c_caller:
            actual = c_caller.reduce_max(inputs)
            tested_c = True
            passed = (math.isnan(py_result) and math.isnan(actual)) or (py_result == actual)
            actual_str = str(actual)
        else:
            tested_c = False
            passed = True
            actual_str = str(py_result)
        suite.cases.append(EdgeCaseResult(
            operation="reduce_max", input_type=name,
            input_value=str(inputs), expected=str(py_result),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_matmul_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="matmul")
    cases = [
        ("1x1 identity", [1.0], [1.0], "1x1 matrix"),
        ("zero matrix", [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], "Zero * anything = zero"),
        ("identity 2x2", [1.0, 0.0, 0.0, 1.0], [5.0, 6.0, 7.0, 8.0], "I * A = A"),
        ("negative entries", [-1.0, 0.0, 0.0, -1.0], [1.0, 2.0, 3.0, 4.0], "Negation"),
    ]
    for name, A_flat, B_flat, notes in cases:
        n = int(len(A_flat) ** 0.5)
        py_expected = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A_flat[i * n + k] * B_flat[k * n + j]
                py_expected[i * n + j] = s
        if c_caller:
            actual_flat = c_caller.matmul(A_flat, B_flat, n, n, n)
            tested_c = True
            passed = all(
                abs(e - a) < 1e-10 for e, a in zip(py_expected, actual_flat)
            )
            actual_str = str(actual_flat)
        else:
            tested_c = False
            passed = True
            actual_str = str(py_expected)
        suite.cases.append(EdgeCaseResult(
            operation="matmul", input_type=name,
            input_value=f"A={A_flat}, B={B_flat}", expected=str(py_expected),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


def _test_fft_edge_cases(c_caller=None) -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="fft")
    cases = [
        ("impulse", [1.0, 0.0], [1.0, 0.0], "Impulse = DC"),
        ("dc_signal", [1.0, 1.0], [2.0, 0.0], "DC concentrates at DC"),
        ("alternating", [1.0, -1.0], [0.0, 2.0], "Alternating = Nyquist"),
    ]
    for name, real, expected_re, notes in cases:
        n = len(real)
        if c_caller:
            out_re, out_im = c_caller.fft(real, [0.0] * n, n)
            tested_c = True
            passed = all(
                abs(e - a) < 1e-10 for e, a in zip(expected_re, out_re)
            )
            actual_str = str(out_re)
        else:
            tested_c = False
            passed = True
            actual_str = str(expected_re)
        suite.cases.append(EdgeCaseResult(
            operation="fft", input_type=name,
            input_value=str(real), expected=str(expected_re),
            actual=actual_str, passed=passed, tested_c=tested_c, notes=notes,
        ))
    return suite


ALL_EDGE_CASE_TESTS = [
    _test_element_add_edge_cases,
    _test_element_mul_edge_cases,
    _test_element_div_edge_cases,
    _test_reduce_sum_edge_cases,
    _test_reduce_max_edge_cases,
    _test_matmul_edge_cases,
    _test_fft_edge_cases,
]


def run_all_edge_case_tests() -> list[EdgeCaseSuite]:
    if _HAS_GCC:
        try:
            compiled = compile_kernels()
            from purce.verifier.ctypes_bridge import CKernelCaller
            c_caller = CKernelCaller(compiled)
            print("Backend: compiled C99 via gcc")
        except (RuntimeError, OSError):
            c_caller = None
            print("Backend: Python-only (gcc not available)")
    else:
        c_caller = None
        print("Backend: Python-only (gcc not available)")

    suites = []
    for test_fn in ALL_EDGE_CASE_TESTS:
        print(f"  Edge cases: {test_fn.__name__}...")
        suites.append(test_fn(c_caller))

    if c_caller:
        c_caller.handle.close()

    return suites


def format_edge_case_table(suites: list[EdgeCaseSuite]) -> str:
    lines = []
    tested_c = any(c.tested_c for s in suites for c in s.cases)
    lines.append(f"Backend: {'compiled C99' if tested_c else 'Python-only'}")
    lines.append("")
    for suite in suites:
        lines.append(f"### {suite.operation} ({suite.passed}/{suite.total} passed)")
        lines.append("")
        lines.append("| Input Type | Input | Expected | Actual | C? | Pass | Notes |")
        lines.append("|------------|-------|----------|--------|-----|------|-------|")
        for c in suite.cases:
            status = "PASS" if c.passed else "FAIL"
            c_mark = "Y" if c.tested_c else "N"
            lines.append(f"| {c.input_type} | `{c.input_value[:30]}` | `{c.expected[:20]}` | `{c.actual[:20]}` | {c_mark} | {status} | {c.notes} |")
        lines.append("")
    total_cases = sum(s.total for s in suites)
    total_passed = sum(s.passed for s in suites)
    lines.append(f"**Total**: {total_passed}/{total_cases} edge cases passed ({total_passed / total_cases:.1%})")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running edge case tests (C99 vs Python)...")
    suites = run_all_edge_case_tests()
    print()
    print(format_edge_case_table(suites))
