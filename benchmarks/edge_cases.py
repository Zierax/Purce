"""Edge case and robustness benchmarks.

Tests numerical behavior under extreme inputs: NaN, Inf, denormalized numbers,
overflow, underflow, zero, negative zero, very large/small values.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field


@dataclass
class EdgeCaseResult:
    operation: str
    input_type: str
    input_value: str
    expected: str
    actual: str
    passed: bool
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


def _test_element_add_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_add")
    cases = [
        ("zero + zero", [0.0, 0.0], [0.0, 0.0], "Both zero"),
        ("negative + positive", [-5.0, 5.0], [0.0, 0.0], "Cancellation"),
        ("large + small", [1e300, 1e-300], [1e300, 1e-300], "Magnitude difference"),
        ("inf + inf", [float('inf'), float('inf')], [float('inf'), float('inf')], "Infinity"),
        ("neg_inf + neg_inf", [float('-inf'), float('-inf')], [float('-inf'), float('-inf')], "Negative infinity"),
        ("inf + neg_inf", [float('inf'), float('-inf')], [float('nan'), float('nan')], "Indeterminate"),
        ("nan + nan", [float('nan'), float('nan')], [float('nan'), float('nan')], "NaN propagation"),
        ("denormalized", [5e-324, 5e-324], [1e-323, 1e-323], "Subnormal numbers"),
        ("max_float", [1.7976931348623157e+308, 1.7976931348623157e+308],
         [float('inf'), float('inf')], "Overflow to inf"),
        ("min_float", [-1.7976931348623157e+308, -1.7976931348623157e+308],
         [float('-inf'), float('-inf')], "Underflow to -inf"),
        ("negative_zero", [-0.0, 0.0], [0.0, 0.0], "-0 + 0 = +0"),
        ("mixed_signs", [1.0, -1.0, 2.0, -2.0], [-1.0, 1.0], "Alternating signs"),
    ]
    for name, inputs, expected, notes in cases:
        actual = [inputs[0] + inputs[1]] * len(expected)
        passed = all(
            (math.isnan(e) and math.isnan(a)) or (e == a)
            for e, a in zip(expected, actual)
        )
        suite.cases.append(EdgeCaseResult(
            operation="element_add", input_type=name,
            input_value=str(inputs), expected=str(expected),
            actual=str(actual), passed=passed, notes=notes,
        ))
    return suite


def _test_element_mul_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_mul")
    cases = [
        ("zero * anything", [0.0, 100.0], [0.0], "Zero annihilates"),
        ("inf * zero", [float('inf'), 0.0], [float('nan')], "Indeterminate"),
        ("inf * inf", [float('inf'), float('inf')], [float('inf')], "Infinity squared"),
        ("inf * neg", [float('inf'), -1.0], [float('-inf')], "Negative infinity"),
        ("nan * nan", [float('nan'), float('nan')], [float('nan')], "NaN propagation"),
        ("denorm * 2", [5e-324, 2.0], [1e-323], "Denormalized multiply"),
        ("large * small", [1e200, 1e-200], [1.0], "Unity product"),
        ("overflow", [1e200, 1e200], [float('inf')], "Overflow to inf"),
    ]
    for name, inputs, expected, notes in cases:
        actual_val = inputs[0] * inputs[1]
        passed = (math.isnan(expected[0]) and math.isnan(actual_val)) or (expected[0] == actual_val)
        suite.cases.append(EdgeCaseResult(
            operation="element_mul", input_type=name,
            input_value=str(inputs), expected=str(expected),
            actual=str([actual_val]), passed=passed, notes=notes,
        ))
    return suite


def _test_element_div_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="element_div")
    cases = [
        ("zero / zero", [0.0, 0.0], [float('nan')], "0/0 is NaN in IEEE 754"),
        ("nonzero / zero", [1.0, 0.0], [float('inf')], "x/0 = inf"),
        ("neg / zero", [-1.0, 0.0], [float('-inf')], "-x/0 = -inf"),
        ("inf / inf", [float('inf'), float('inf')], [float('nan')], "Indeterminate"),
        ("inf / 2", [float('inf'), 2.0], [float('inf')], "Infinity preserved"),
        ("nan / 1", [float('nan'), 1.0], [float('nan')], "NaN propagation"),
        ("1 / inf", [1.0, float('inf')], [0.0], "Converges to zero"),
        ("large / small", [1e300, 1e-300], [float('inf')], "Overflow"),
    ]
    for name, inputs, expected, notes in cases:
        if inputs[1] == 0.0:
            actual_val = inputs[0] / inputs[1] if inputs[1] != 0 else 0.0
        else:
            actual_val = inputs[0] / inputs[1]
        passed = (math.isnan(expected[0]) and math.isnan(actual_val)) or (expected[0] == actual_val)
        suite.cases.append(EdgeCaseResult(
            operation="element_div", input_type=name,
            input_value=str(inputs), expected=str(expected),
            actual=str([actual_val]), passed=passed, notes=notes,
        ))
    return suite


def _test_reduce_sum_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="reduce_sum")
    cases = [
        ("empty", [], [0.0], "Sum of empty = 0"),
        ("single_zero", [0.0], [0.0], "Single zero"),
        ("all_zeros", [0.0, 0.0, 0.0], [0.0], "All zeros"),
        ("pos_and_neg", [1.0, -1.0, 2.0, -2.0], [0.0], "Cancellation to zero"),
        ("large_values", [1e300, 1e300, 1e300], [float('inf')], "Overflow to inf"),
        ("inf_in_list", [1.0, float('inf'), 2.0], [float('inf')], "Infinity dominates"),
        ("nan_in_list", [1.0, float('nan'), 2.0], [float('nan')], "NaN poisons sum"),
        ("denorms", [5e-324, 5e-324, 5e-324], [1.5e-323], "Accumulate denorms"),
    ]
    for name, inputs, expected, notes in cases:
        actual_val = sum(inputs) if inputs else 0.0
        passed = (math.isnan(expected[0]) and math.isnan(actual_val)) or (expected[0] == actual_val)
        suite.cases.append(EdgeCaseResult(
            operation="reduce_sum", input_type=name,
            input_value=str(inputs), expected=str(expected),
            actual=str([actual_val]), passed=passed, notes=notes,
        ))
    return suite


def _test_reduce_max_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="reduce_max")
    cases = [
        ("single_element", [42.0], [42.0], "Single element"),
        ("all_equal", [5.0, 5.0, 5.0], [5.0], "All equal"),
        ("negatives", [-10.0, -5.0, -1.0], [-1.0], "Closest to zero"),
        ("inf_present", [1.0, float('inf'), 2.0], [float('inf')], "Infinity is max"),
        ("nan_present", [1.0, float('nan'), 2.0], [float('nan')], "NaN poisons max"),
        ("mixed_signs", [-100.0, 0.0, 100.0], [100.0], "Positive max"),
    ]
    for name, inputs, expected, notes in cases:
        actual_val = max(inputs)
        passed = (math.isnan(expected[0]) and math.isnan(actual_val)) or (expected[0] == actual_val)
        suite.cases.append(EdgeCaseResult(
            operation="reduce_max", input_type=name,
            input_value=str(inputs), expected=str(expected),
            actual=str([actual_val]), passed=passed, notes=notes,
        ))
    return suite


def _test_matmul_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="matmul")
    cases = [
        ("1x1 identity", [1.0], [1.0], [1.0], "1x1 matrix"),
        ("zero matrix", [0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], "Zero * anything = zero"),
        ("identity matrix", [1.0, 0.0, 0.0, 1.0], [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0], "I * A = A"),
        ("negative entries", [-1.0, 0.0, 0.0, -1.0], [1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], "Negation"),
    ]
    for name, A, B, expected, notes in cases:
        n = int(len(A) ** 0.5)
        actual = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += A[i * n + k] * B[k * n + j]
                actual[i * n + j] = s
        passed = all(
            (math.isnan(e) and math.isnan(a)) or abs(e - a) < 1e-10
            for e, a in zip(expected, actual)
        )
        suite.cases.append(EdgeCaseResult(
            operation="matmul", input_type=name,
            input_value=f"A={A}, B={B}", expected=str(expected),
            actual=str(actual), passed=passed, notes=notes,
        ))
    return suite


def _test_fft_edge_cases() -> EdgeCaseSuite:
    suite = EdgeCaseSuite(operation="fft")
    cases = [
        ("all zeros", [0.0, 0.0], [0.0, 0.0], "Zero input = zero output"),
        ("impulse", [1.0, 0.0], [1.0, 0.0], "Impulse = DC"),
        ("dc_signal", [1.0, 1.0], [2.0, 0.0], "DC signal concentrates at DC"),
        ("alternating", [1.0, -1.0], [0.0, 2.0], "Alternating = Nyquist"),
    ]
    for name, real, expected_re, notes in cases:
        n = len(real)
        imag = [0.0] * n
        actual_re = list(real)
        actual_im = list(imag)
        if n > 1:
            log_n = 0
            t = n
            while t > 1:
                t >>= 1
                log_n += 1
            for i in range(n):
                j = 0
                for bit in range(log_n):
                    if i & (1 << bit):
                        j |= n >> (bit + 1)
                if i < j:
                    actual_re[i], actual_re[j] = actual_re[j], actual_re[i]
                    actual_im[i], actual_im[j] = actual_im[j], actual_im[i]
            size = 2
            while size <= n:
                half = size // 2
                angle = -2.0 * math.pi / size
                w_re = math.cos(angle)
                w_im = math.sin(angle)
                for i in range(0, n, size):
                    cur_re = 1.0
                    cur_im = 0.0
                    for j in range(half):
                        u = i + j
                        t_idx = i + j + half
                        t_re = cur_re * actual_re[t_idx] - cur_im * actual_im[t_idx]
                        t_im = cur_re * actual_im[t_idx] + cur_im * actual_re[t_idx]
                        actual_re[t_idx] = actual_re[u] - t_re
                        actual_im[t_idx] = actual_im[u] - t_im
                        actual_re[u] += t_re
                        actual_im[u] += t_im
                        new_re = cur_re * w_re - cur_im * w_im
                        new_im = cur_re * w_im + cur_im * w_re
                        cur_re, cur_im = new_re, new_im
                size *= 2
        passed = all(
            abs(e - a) < 1e-10 for e, a in zip(expected_re, actual_re)
        )
        suite.cases.append(EdgeCaseResult(
            operation="fft", input_type=name,
            input_value=str(real), expected=str(expected_re),
            actual=str(actual_re), passed=passed, notes=notes,
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
    suites = []
    for test_fn in ALL_EDGE_CASE_TESTS:
        print(f"  Edge cases: {test_fn.__name__}...")
        suites.append(test_fn())
    return suites


def format_edge_case_table(suites: list[EdgeCaseSuite]) -> str:
    lines = []
    for suite in suites:
        lines.append(f"### {suite.operation} ({suite.passed}/{suite.total} passed)")
        lines.append("")
        lines.append("| Input Type | Input | Expected | Actual | Pass | Notes |")
        lines.append("|------------|-------|----------|--------|------|-------|")
        for c in suite.cases:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"| {c.input_type} | `{c.input_value[:30]}` | `{c.expected[:20]}` | `{c.actual[:20]}` | {status} | {c.notes} |")
        lines.append("")
    total_cases = sum(s.total for s in suites)
    total_passed = sum(s.passed for s in suites)
    lines.append(f"**Total**: {total_passed}/{total_cases} edge cases passed ({total_passed/total_cases:.1%})")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running edge case tests...")
    suites = run_all_edge_case_tests()
    print()
    print(format_edge_case_table(suites))
