"""Differential fuzzer for comparing Python and C99 outputs.

When c_kernel_caller is provided, actually compiles and invokes C99 code,
comparing C output against Python reference. This is the real verification.
"""
from __future__ import annotations

import math
import random as _random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from purce.verifier.ctypes_bridge import CKernelCaller, CompiledKernels, compile_kernels


@dataclass
class FuzzResult:
    operation: str
    iterations: int
    passed: int
    failed: int
    failures: list[dict[str, Any]] = field(default_factory=list)
    rtol: float = 1e-5
    atol: float = 1e-8
    tested_c: bool = False
    max_error: float = 0.0
    mean_error: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.passed / self.iterations if self.iterations > 0 else 0.0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class DifferentialFuzzer:
    """Differential fuzzer comparing Python reference against compiled C99 code."""

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        seed: int | None = None,
        c_caller: CKernelCaller | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self._c = c_caller
        self._results: dict[str, FuzzResult] = {}
        if seed is not None:
            _random.seed(seed)

    @classmethod
    def with_c_backend(cls, rtol: float = 1e-5, atol: float = 1e-8, seed: int | None = None) -> tuple["DifferentialFuzzer", CompiledKernels]:
        """Create fuzzer with compiled C backend. Returns (fuzzer, compiled_handle).

        Caller must call compiled_handle.close() when done.
        """
        compiled = compile_kernels()
        caller = CKernelCaller(compiled)
        fuzzer = cls(rtol=rtol, atol=atol, seed=seed, c_caller=caller)
        return fuzzer, compiled

    def fuzz_matmul(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_matmul_op("matmul", self._matmul_python, iterations)

    def fuzz_element_add(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_binary_op("element_add", self._element_add_python, iterations,
                                    c_fn=self._c.element_add if self._c else None)

    def fuzz_element_sub(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_binary_op("element_sub", self._element_sub_python, iterations,
                                    c_fn=self._c.element_sub if self._c else None)

    def fuzz_element_mul(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_binary_op("element_mul", self._element_mul_python, iterations,
                                    c_fn=self._c.element_mul if self._c else None)

    def fuzz_element_div(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_binary_op("element_div", self._element_div_python, iterations,
                                    c_fn=self._c.element_div if self._c else None)

    def fuzz_reduce_sum(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_reduce_op("reduce_sum", self._reduce_sum_python, iterations,
                                    c_fn=self._c.reduce_sum if self._c else None)

    def fuzz_reduce_mean(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_reduce_op("reduce_mean", self._reduce_mean_python, iterations,
                                    c_fn=self._c.reduce_mean if self._c else None)

    def fuzz_reduce_max(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_reduce_op("reduce_max", self._reduce_max_python, iterations,
                                    c_fn=self._c.reduce_max if self._c else None)

    def fuzz_reduce_min(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_reduce_op("reduce_min", self._reduce_min_python, iterations,
                                    c_fn=self._c.reduce_min if self._c else None)

    def fuzz_linalg_solve(self, iterations: int = 200) -> FuzzResult:
        return self._fuzz_linalg_op("linalg_solve", self._linalg_solve_python, iterations,
                                    c_fn=self._c.linalg_solve if self._c else None)

    def fuzz_linalg_inv(self, iterations: int = 200) -> FuzzResult:
        return self._fuzz_linalg_op("linalg_inv", self._linalg_inv_python, iterations,
                                    c_fn=self._c.linalg_inv if self._c else None)

    def fuzz_linalg_cholesky(self, iterations: int = 200) -> FuzzResult:
        return self._fuzz_linalg_op("linalg_cholesky", self._linalg_cholesky_python, iterations,
                                    c_fn=self._c.linalg_cholesky if self._c else None)

    def fuzz_linalg_eig(self, iterations: int = 200) -> FuzzResult:
        return self._fuzz_linalg_op("linalg_eig", self._linalg_eig_python, iterations,
                                    c_fn=self._c.linalg_eig if self._c else None)

    def fuzz_fft(self, iterations: int = 500) -> FuzzResult:
        return self._fuzz_fft_op("fft", self._fft_python, iterations,
                                 c_fn=self._c.fft if self._c else None)

    def fuzz_ifft(self, iterations: int = 500) -> FuzzResult:
        return self._fuzz_fft_op("ifft", self._ifft_python, iterations,
                                 c_fn=self._c.ifft if self._c else None)

    def fuzz_element_tan(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_tan", self._element_tan_python, iterations,
                                            c_fn=self._c.element_tan if self._c else None)

    def fuzz_element_sqrt(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_sqrt", self._element_sqrt_python, iterations,
                                            c_fn=self._c.element_sqrt if self._c else None)

    def fuzz_element_exp(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_exp", self._element_exp_python, iterations,
                                            c_fn=self._c.element_exp if self._c else None)

    def fuzz_element_log(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_log", self._element_log_python, iterations,
                                            c_fn=self._c.element_log if self._c else None)

    def fuzz_element_sin(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_sin", self._element_sin_python, iterations,
                                            c_fn=self._c.element_sin if self._c else None)

    def fuzz_element_cos(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_cos", self._element_cos_python, iterations,
                                            c_fn=self._c.element_cos if self._c else None)

    def fuzz_element_abs(self, iterations: int = 1000) -> FuzzResult:
        return self._fuzz_elementwise_unary("element_abs", self._element_abs_python, iterations,
                                            c_fn=self._c.element_abs if self._c else None)

    def fuzz_alloc_zeros(self, iterations: int = 500) -> FuzzResult:
        return self._fuzz_alloc_op("alloc_zeros", self._alloc_zeros_python, iterations,
                                   c_fn=self._c.alloc_zeros if self._c else None)

    def fuzz_alloc_ones(self, iterations: int = 500) -> FuzzResult:
        return self._fuzz_alloc_op("alloc_ones", self._alloc_ones_python, iterations,
                                   c_fn=self._c.alloc_ones if self._c else None)

    def fuzz_alloc_eye(self, iterations: int = 500) -> FuzzResult:
        return self._fuzz_alloc_op("alloc_eye", self._alloc_eye_python, iterations,
                                   c_fn=self._c.alloc_eye if self._c else None)

    def fuzz_all(self, iterations: int = 1000) -> dict[str, FuzzResult]:
        results = {}
        results["matmul"] = self.fuzz_matmul(iterations)
        results["element_add"] = self.fuzz_element_add(iterations)
        results["element_sub"] = self.fuzz_element_sub(iterations)
        results["element_mul"] = self.fuzz_element_mul(iterations)
        results["element_div"] = self.fuzz_element_div(iterations)
        results["element_tan"] = self.fuzz_element_tan(iterations)
        results["element_sqrt"] = self.fuzz_element_sqrt(iterations)
        results["element_exp"] = self.fuzz_element_exp(iterations)
        results["element_log"] = self.fuzz_element_log(iterations)
        results["element_sin"] = self.fuzz_element_sin(iterations)
        results["element_cos"] = self.fuzz_element_cos(iterations)
        results["element_abs"] = self.fuzz_element_abs(iterations)
        results["reduce_sum"] = self.fuzz_reduce_sum(iterations)
        results["reduce_mean"] = self.fuzz_reduce_mean(iterations)
        results["reduce_max"] = self.fuzz_reduce_max(iterations)
        results["reduce_min"] = self.fuzz_reduce_min(iterations)
        results["linalg_solve"] = self.fuzz_linalg_solve(min(iterations, 200))
        results["linalg_inv"] = self.fuzz_linalg_inv(min(iterations, 200))
        results["linalg_cholesky"] = self.fuzz_linalg_cholesky(min(iterations, 200))
        results["linalg_eig"] = self.fuzz_linalg_eig(min(iterations, 200))
        results["fft"] = self.fuzz_fft(min(iterations, 500))
        results["ifft"] = self.fuzz_ifft(min(iterations, 500))
        results["alloc_zeros"] = self.fuzz_alloc_zeros(min(iterations, 500))
        results["alloc_ones"] = self.fuzz_alloc_ones(min(iterations, 500))
        results["alloc_eye"] = self.fuzz_alloc_eye(min(iterations, 500))
        self._results = results
        return results

    # ── Core fuzzing methods ──

    def _fuzz_matmul_op(
        self, name: str, python_fn: Callable, iterations: int,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []

        for _ in range(iterations):
            try:
                expected, a, b, m, k, p = python_fn(iteration=True)
                if self._c is not None:
                    actual = self._c.matmul(a, b, m, p, k)
                else:
                    actual = list(expected)

                close = _all_close(expected, actual, self.rtol, self.atol)
                max_err = _max_error(expected, actual)
                errors.append(max_err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "m": m, "k": k, "p": p,
                        "expected": expected[:4],
                        "actual": actual[:4],
                        "max_error": max_err,
                    })
            except Exception as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=self.rtol, atol=self.atol,
            tested_c=self._c is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_binary_op(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []

        for _ in range(iterations):
            try:
                expected, a, b = python_fn(iteration=True)
                if c_fn is not None:
                    actual = c_fn(a, b, len(expected))
                else:
                    actual = list(expected)

                close = _all_close(expected, actual, self.rtol, self.atol)
                max_err = _max_error(expected, actual)
                errors.append(max_err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "input_a": a[:4],
                        "input_b": b[:4],
                        "expected": expected[:4],
                        "actual": actual[:4],
                        "max_error": max_err,
                    })
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=self.rtol, atol=self.atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_reduce_op(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []

        for _ in range(iterations):
            try:
                expected, x = python_fn(iteration=True)
                n = len(x)
                if c_fn is not None:
                    actual = c_fn(x, n)
                else:
                    actual = expected

                close = abs(expected - actual) <= self.atol + self.rtol * abs(expected)
                err = abs(expected - actual)
                errors.append(err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "input": x[:4],
                        "expected": expected,
                        "actual": actual,
                        "error": err,
                    })
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=self.rtol, atol=self.atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_linalg_op(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []
        rtol = self.rtol * 10
        atol = self.atol * 10

        for _ in range(iterations):
            n = _random_int(2, 8)
            A = [[_random_float(-5.0, 5.0) for _ in range(n)] for _ in range(n)]
            b = [_random_float(-10.0, 10.0) for _ in range(n)]

            for i in range(n):
                A[i][i] += n * 10.0

            try:
                expected = python_fn(A, b, n)
                if c_fn is not None:
                    flat_A = [A[i][j] for i in range(n) for j in range(n)]
                    actual = c_fn(flat_A, n) if name != "linalg_solve" else c_fn(flat_A, b, n)
                else:
                    actual = list(expected)

                close = _all_close(expected, actual, rtol, atol)
                max_err = _max_error(expected, actual)
                errors.append(max_err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "expected": expected[:4],
                        "actual": actual[:4],
                        "max_error": max_err,
                    })
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=rtol, atol=atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_fft_op(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []
        rtol = self.rtol * 10
        atol = self.atol * 10

        for _ in range(iterations):
            exp = _random_int(1, 8)
            n = 2 ** exp
            real = [_random_float(-10.0, 10.0) for _ in range(n)]
            imag = [_random_float(-10.0, 10.0) for _ in range(n)]

            try:
                exp_r, exp_i = python_fn(real, imag, n)
                if c_fn is not None:
                    act_r, act_i = c_fn(real, imag, n)
                else:
                    act_r, act_i = list(exp_r), list(exp_i)

                close_re = _all_close(exp_r, act_r, rtol, atol)
                close_im = _all_close(exp_i, act_i, rtol, atol)
                err_re = _max_error(exp_r, act_r)
                err_im = _max_error(exp_i, act_i)
                errors.append(max(err_re, err_im))

                if close_re and close_im:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "expected_re": exp_r[:4],
                        "actual_re": act_r[:4],
                        "max_error_re": err_re,
                        "max_error_im": err_im,
                    })
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=rtol, atol=atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_elementwise_unary(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []

        for _ in range(iterations):
            try:
                expected, x = python_fn(iteration=True)
                n = len(x)
                if c_fn is not None:
                    actual = c_fn(x, n)
                else:
                    actual = list(expected)

                close = _all_close(expected, actual, self.rtol, self.atol)
                max_err = _max_error(expected, actual)
                errors.append(max_err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "input": x[:4],
                        "expected": expected[:4],
                        "actual": actual[:4],
                        "max_error": max_err,
                    })
            except (ValueError, TypeError, ZeroDivisionError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=self.rtol, atol=self.atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    def _fuzz_alloc_op(
        self, name: str, python_fn: Callable, iterations: int,
        c_fn: Callable | None = None,
    ) -> FuzzResult:
        passed = 0
        failed = 0
        failures: list[dict[str, Any]] = []
        errors: list[float] = []

        for _ in range(iterations):
            try:
                expected, n = python_fn(iteration=True)
                if c_fn is not None:
                    actual = c_fn(n)
                else:
                    actual = list(expected)

                close = _all_close(expected, actual, self.rtol, self.atol)
                max_err = _max_error(expected, actual)
                errors.append(max_err)

                if close:
                    passed += 1
                else:
                    failed += 1
                    failures.append({
                        "n": n,
                        "expected": expected[:4],
                        "actual": actual[:4],
                        "max_error": max_err,
                    })
            except (ValueError, TypeError, IndexError, OverflowError) as e:
                failed += 1
                failures.append({"error": str(e)})

        return FuzzResult(
            operation=name, iterations=iterations, passed=passed, failed=failed,
            failures=failures[:10], rtol=self.rtol, atol=self.atol,
            tested_c=c_fn is not None,
            max_error=max(errors) if errors else 0.0,
            mean_error=sum(errors) / len(errors) if errors else 0.0,
        )

    # ── Python reference implementations ──

    @staticmethod
    def _matmul_python(iteration: bool = True) -> tuple[list[float], list[float], list[float], int, int, int]:
        m = _random_int(1, 16)
        k = _random_int(1, 16)
        p = _random_int(1, 16)
        a = [_random_float(-100.0, 100.0) for _ in range(m * k)]
        b = [_random_float(-100.0, 100.0) for _ in range(k * p)]
        result = [0.0] * (m * p)
        for i in range(m):
            for j in range(p):
                s = 0.0
                for l in range(k):
                    s += a[i * k + l] * b[l * p + j]
                result[i * p + j] = s
        return result, a, b, m, k, p

    @staticmethod
    def _element_add_python(iteration: bool = True) -> tuple[list[float], list[float], list[float]]:
        n = _random_int(1, 64)
        a = [_random_float(-100.0, 100.0) for _ in range(n)]
        b = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [a[i] + b[i] for i in range(n)], a, b

    @staticmethod
    def _element_sub_python(iteration: bool = True) -> tuple[list[float], list[float], list[float]]:
        n = _random_int(1, 64)
        a = [_random_float(-100.0, 100.0) for _ in range(n)]
        b = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [a[i] - b[i] for i in range(n)], a, b

    @staticmethod
    def _element_mul_python(iteration: bool = True) -> tuple[list[float], list[float], list[float]]:
        n = _random_int(1, 64)
        a = [_random_float(-100.0, 100.0) for _ in range(n)]
        b = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [a[i] * b[i] for i in range(n)], a, b

    @staticmethod
    def _element_div_python(iteration: bool = True) -> tuple[list[float], list[float], list[float]]:
        n = _random_int(1, 64)
        a = [_random_float(-100.0, 100.0) for _ in range(n)]
        b = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [a[i] / b[i] if b[i] != 0 else 0.0 for i in range(n)], a, b

    @staticmethod
    def _reduce_sum_python(iteration: bool = True) -> tuple[float, list[float]]:
        n = _random_int(1, 128)
        x = [_random_float(-1000.0, 1000.0) for _ in range(n)]
        return sum(x), x

    @staticmethod
    def _reduce_mean_python(iteration: bool = True) -> tuple[float, list[float]]:
        n = _random_int(1, 128)
        x = [_random_float(-1000.0, 1000.0) for _ in range(n)]
        return sum(x) / n if n > 0 else 0.0, x

    @staticmethod
    def _reduce_max_python(iteration: bool = True) -> tuple[float, list[float]]:
        n = _random_int(1, 128)
        x = [_random_float(-1000.0, 1000.0) for _ in range(n)]
        return max(x) if x else 0.0, x

    @staticmethod
    def _reduce_min_python(iteration: bool = True) -> tuple[float, list[float]]:
        n = _random_int(1, 128)
        x = [_random_float(-1000.0, 1000.0) for _ in range(n)]
        return min(x) if x else 0.0, x

    @staticmethod
    def _linalg_solve_python(A, b, n):
        aug = [list(row) + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            if abs(aug[col][col]) < 1e-15:
                continue
            for row in range(col + 1, n):
                factor = aug[row][col] / aug[col][col]
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            if abs(aug[i][i]) > 1e-15:
                x[i] /= aug[i][i]
        return x

    @staticmethod
    def _linalg_inv_python(A, b, n):
        aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            pivot = aug[col][col]
            if abs(pivot) < 1e-15:
                continue
            for j in range(2 * n):
                aug[col][j] /= pivot
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
        result = []
        for i in range(n):
            for j in range(n):
                result.append(aug[i][n + j])
        return result

    @staticmethod
    def _linalg_cholesky_python(A, b, n):
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(val) if val > 0 else 0.0
                else:
                    denom = L[j][j]
                    L[i][j] = (A[i][j] - s) / denom if abs(denom) > 1e-15 else 0.0
        result = []
        for i in range(n):
            for j in range(n):
                result.append(L[i][j])
        return result

    @staticmethod
    def _linalg_eig_python(A, b, n):
        import numpy as np
        Asym = [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]
        w = np.linalg.eigvalsh(Asym)
        return sorted(w.tolist())

    @staticmethod
    def _fft_python(real, imag, n):
        if n <= 1:
            return list(real), list(imag)
        log_n = 0
        temp = n
        while temp > 1:
            temp >>= 1
            log_n += 1
        r_out = list(real)
        i_out = list(imag)
        for i in range(n):
            j = 0
            for bit in range(log_n):
                if i & (1 << bit):
                    j |= n >> (bit + 1)
            if i < j:
                r_out[i], r_out[j] = r_out[j], r_out[i]
                i_out[i], i_out[j] = i_out[j], i_out[i]
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
                    t = i + j + half
                    t_re = cur_re * r_out[t] - cur_im * i_out[t]
                    t_im = cur_re * i_out[t] + cur_im * r_out[t]
                    r_out[t] = r_out[u] - t_re
                    i_out[t] = i_out[u] - t_im
                    r_out[u] += t_re
                    i_out[u] += t_im
                    new_re = cur_re * w_re - cur_im * w_im
                    new_im = cur_re * w_im + cur_im * w_re
                    cur_re, cur_im = new_re, new_im
            size *= 2
        return r_out, i_out

    @staticmethod
    def _ifft_python(real, imag, n):
        conj_imag = [-x for x in imag]
        r_fwd, i_fwd = DifferentialFuzzer._fft_python(real, conj_imag, n)
        r_out = [x / n for x in r_fwd]
        i_out = [-x / n for x in i_fwd]
        return r_out, i_out

    @staticmethod
    def _element_tan_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(-1.5, 1.5) for _ in range(n)]
        return [math.tan(v) for v in x], x

    @staticmethod
    def _element_sqrt_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(0.0, 1000.0) for _ in range(n)]
        return [math.sqrt(v) for v in x], x

    @staticmethod
    def _element_exp_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(-10.0, 10.0) for _ in range(n)]
        return [math.exp(v) for v in x], x

    @staticmethod
    def _element_log_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(0.001, 1000.0) for _ in range(n)]
        return [math.log(v) for v in x], x

    @staticmethod
    def _element_sin_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [math.sin(v) for v in x], x

    @staticmethod
    def _element_cos_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(-100.0, 100.0) for _ in range(n)]
        return [math.cos(v) for v in x], x

    @staticmethod
    def _element_abs_python(iteration: bool = True) -> tuple[list[float], list[float]]:
        n = _random_int(1, 64)
        x = [_random_float(-1000.0, 1000.0) for _ in range(n)]
        return [abs(v) for v in x], x

    @staticmethod
    def _alloc_zeros_python(iteration: bool = True) -> tuple[list[float], int]:
        n = _random_int(1, 64)
        return [0.0] * n, n

    @staticmethod
    def _alloc_ones_python(iteration: bool = True) -> tuple[list[float], int]:
        n = _random_int(1, 64)
        return [1.0] * n, n

    @staticmethod
    def _alloc_eye_python(iteration: bool = True) -> tuple[list[float], int]:
        n = _random_int(1, 16)
        result = [0.0] * (n * n)
        for i in range(n):
            result[i * n + i] = 1.0
        return result, n


def _random_float(lo: float, hi: float) -> float:
    return _random.uniform(lo, hi)


def _random_int(lo: int, hi: int) -> int:
    return _random.randint(lo, hi)


def _all_close(a: list[float], b: list[float], rtol: float, atol: float) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if math.isnan(x) or math.isnan(y):
            if not (math.isnan(x) and math.isnan(y)):
                return False
            continue
        if math.isinf(x) or math.isinf(y):
            if not (math.isinf(x) and math.isinf(y) and (x > 0) == (y > 0)):
                return False
            continue
        if abs(x - y) > atol + rtol * abs(x):
            return False
    return True


def _max_error(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    worst = 0.0
    for x, y in zip(a, b):
        if math.isnan(x) or math.isnan(y):
            if math.isnan(x) and math.isnan(y):
                continue
            return float("inf")
        if math.isinf(x) or math.isinf(y):
            if not (math.isinf(x) and math.isinf(y) and (x > 0) == (y > 0)):
                return float("inf")
            continue
        worst = max(worst, abs(x - y))
    return worst
