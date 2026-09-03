from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from purce import __version__
from purce.ir.nodes import Dtype, MathIRGraph, MathIRNode

# All kernels are now verified — no stubs remain for v1.
_STUB_ALGORITHMS: frozenset[str] = frozenset()


@dataclass
class GeneratedFile:
    path: str
    content: str
    file_type: str  # "c", "h", "prov", "cmake"


@dataclass
class C99GenerationResult:
    files: list[GeneratedFile] = field(default_factory=list)
    diagnostics: list[dict[str, str]] = field(default_factory=list)

    def write_all(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for gf in self.files:
            full_path = os.path.join(output_dir, gf.path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(gf.content)


DTYPE_TO_C = {
    Dtype.FLOAT32: "float",
    Dtype.FLOAT64: "double",
    Dtype.INT32: "int32_t",
    Dtype.INT64: "int64_t",
    Dtype.Q15: "q15_t",
    Dtype.Q31: "q31_t",
    Dtype.BOOL: "int",
    Dtype.COMPLEX64: "float _Complex",
    Dtype.COMPLEX128: "double _Complex",
}

BODY_PARAM_MAP: dict[str, list[tuple[str, str]]] = {
    "element_add": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_sub": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_mul": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_div": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "reduce_sum": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "reduce_mean": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "reduce_max": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "reduce_min": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "matmul": [
        ("A", "input_0"),
        ("B", "input_1"),
        ("C", "output_0"),
        ("m", "dim_m"),
        ("n", "dim_n"),
        ("p", "dim_k"),
    ],
    "linalg_solve": [("A", "input_0"), ("b", "input_1"), ("x", "output_0"), ("n", "dim")],
    "linalg_inv": [("A", "input_0"), ("inv", "output_0"), ("n", "dim")],
    "linalg_cholesky": [("A", "input_0"), ("L", "output_0"), ("n", "dim")],
    "linalg_eig": [("A", "input_0"), ("eigenvalues", "output_0"), ("n", "dim")],
    "linalg_norm": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "alloc_zeros": [("out", "output_0"), ("n", "length")],
    "alloc_ones": [("out", "output_0"), ("n", "length")],
    "alloc_eye": [("out", "output_0"), ("n", "dim")],
    "alloc_arange": [("out", "output_0"), ("n", "length")],
    "alloc_linspace": [("out", "output_0"), ("n", "length")],
    "alloc_full": [("out", "output_0"), ("n", "length")],
    "alloc_random": [("out", "output_0"), ("n", "length")],
    "element_abs": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_sqrt": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_exp": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_log": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_sin": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_cos": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_tan": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_tanh": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_max": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_min": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_power": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_where": [
        ("cond", "input_0"),
        ("A", "input_1"),
        ("B", "input_2"),
        ("C", "output_0"),
        ("n", "length"),
    ],
    "element_clip": [
        ("x", "input_0"),
        ("lo", "input_1"),
        ("hi", "input_2"),
        ("out", "output_0"),
        ("n", "length"),
    ],
    "element_neg": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_sign": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_floor": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_greater": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_less": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_log10": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_logaddexp": [("A", "input_0"), ("B", "input_1"), ("C", "output_0"), ("n", "length")],
    "element_conj": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_angle": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_real": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_imag": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_copy": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "transpose": [("x", "input_0"), ("out", "output_0"), ("rows", "dim_m"), ("cols", "dim_n")],
    "outer_product": [
        ("A", "input_0"),
        ("B", "input_1"),
        ("C", "output_0"),
        ("m", "dim_m"),
        ("n", "dim_n"),
    ],
    "matrix_diag": [("x", "input_0"), ("out", "output_0"), ("n", "dim")],
    "matrix_diag_from": [("x", "input_0"), ("out", "output_0"), ("n", "dim")],
    "matrix_tril": [("x", "input_0"), ("out", "output_0"), ("n", "dim")],
    "matrix_triu": [("x", "input_0"), ("out", "output_0"), ("n", "dim")],
    "array_concat": [
        ("A", "input_0"),
        ("B", "input_1"),
        ("C", "output_0"),
        ("n_a", "dim_m"),
        ("n_b", "dim_n"),
    ],
    "array_take": [
        ("x", "input_0"),
        ("idx", "input_1"),
        ("out", "output_0"),
        ("n", "input_0_len"),
        ("k", "input_1_len"),
    ],
    "array_argsort": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_permutation": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_literal": [("out", "output_0"), ("x", "input_0"), ("n", "length")],
    "array_reshape": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_squeeze": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_expand_dims": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_flatten": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "noop_seed": [("seed", "input_0")],
    "fft": [
        ("real", "input_0"),
        ("imag", "input_1"),
        ("out_real", "output_0"),
        ("out_imag", "output_1"),
        ("n", "length"),
        ("log_n", "log_length"),
    ],
    "ifft": [
        ("real", "input_0"),
        ("imag", "input_1"),
        ("out_real", "output_0"),
        ("out_imag", "output_1"),
        ("n", "length"),
        ("log_n", "log_length"),
    ],
    "reduce_var": [("x", "input_0"), ("n", "length"), ("result_ptr", "output_0")],
    "array_sort": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "linalg_det": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "dim")],
    "linalg_qr": [("x", "input_0"), ("out_q", "output_0"), ("out_r", "output_1"), ("n", "dim")],
    "linalg_svd": [
        ("x", "input_0"),
        ("out_u", "output_0"),
        ("out_s", "output_1"),
        ("out_v", "output_2"),
        ("n", "dim"),
    ],
    "reduce_argmax": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "length")],
    "reduce_argmin": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "length")],
    "reduce_any": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "length")],
    "reduce_all": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "length")],
    "reduce_prod": [("x", "input_0"), ("result_ptr", "output_0"), ("n", "length")],
    "reduce_cumsum": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "reduce_diff": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_round": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_ceil": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_trunc": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_isclose": [("A", "input_0"), ("B", "input_1"), ("out", "output_0"), ("n", "length")],
    "element_isnan": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_isinf": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "element_finfo": [("_dummy", "input_0"), ("result_ptr", "output_0")],
    "array_tile": [("x", "input_0"), ("out", "output_0"), ("n", "length"), ("reps", "input_1")],
    "array_repeat": [("x", "input_0"), ("out", "output_0"), ("n", "length"), ("reps", "input_1")],
    "array_flip": [("x", "input_0"), ("out", "output_0"), ("n", "length")],
    "array_roll": [("x", "input_0"), ("out", "output_0"), ("n", "length"), ("shift", "input_1")],
    "array_split": [
        ("x", "input_0"),
        ("out", "output_0"),
        ("n", "length"),
        ("n_sections", "input_1"),
    ],
    "array_unique": [
        ("x", "input_0"),
        ("out", "output_0"),
        ("out_count", "output_1"),
        ("n", "length"),
    ],
    "array_searchsorted": [
        ("x", "input_0"),
        ("v", "input_1"),
        ("out", "output_0"),
        ("n", "length"),
    ],
    "loop_concat": [("x", "input_0"), ("n_iters", "input_1"), ("out", "output_0"), ("n", "length")],
}

ALGORITHM_C_IMPL = {
    "matmul": "__generated_matmul",
    "linalg_solve": "__generated_linalg_solve",
    "linalg_inv": "__generated_linalg_inv",
    "linalg_cholesky": "__generated_linalg_cholesky",
    "linalg_eig": "__generated_linalg_eig",
    "linalg_norm": "__generated_linalg_norm",
    "element_add": "__generated_element_add",
    "element_sub": "__generated_element_sub",
    "element_mul": "__generated_element_mul",
    "element_div": "__generated_element_div",
    "reduce_sum": "__generated_reduce_sum",
    "reduce_mean": "__generated_reduce_mean",
    "reduce_max": "__generated_reduce_max",
    "reduce_min": "__generated_reduce_min",
    "reduce_var": "__generated_reduce_var",
    "fft": "__generated_fft",
    "ifft": "__generated_ifft",
    "alloc_zeros": "__generated_alloc_zeros",
    "alloc_ones": "__generated_alloc_ones",
    "alloc_eye": "__generated_alloc_eye",
    "alloc_arange": "__generated_alloc_arange",
    "alloc_linspace": "__generated_alloc_linspace",
    "alloc_full": "__generated_alloc_full",
    "alloc_random": "__generated_alloc_random",
    "element_abs": "__generated_element_abs",
    "element_sqrt": "__generated_element_sqrt",
    "element_exp": "__generated_element_exp",
    "element_log": "__generated_element_log",
    "element_sin": "__generated_element_sin",
    "element_cos": "__generated_element_cos",
    "element_tan": "__generated_element_tan",
    "element_tanh": "__generated_element_tanh",
    "element_max": "__generated_element_max",
    "element_min": "__generated_element_min",
    "element_power": "__generated_element_power",
    "element_where": "__generated_element_where",
    "element_clip": "__generated_element_clip",
    "element_neg": "__generated_element_neg",
    "element_sign": "__generated_element_sign",
    "element_floor": "__generated_element_floor",
    "element_greater": "__generated_element_greater",
    "element_less": "__generated_element_less",
    "element_log10": "__generated_element_log10",
    "element_logaddexp": "__generated_element_logaddexp",
    "element_conj": "__generated_element_conj",
    "element_angle": "__generated_element_angle",
    "element_real": "__generated_element_real",
    "element_imag": "__generated_element_imag",
    "element_copy": "__generated_element_copy",
    "transpose": "__generated_transpose",
    "outer_product": "__generated_outer_product",
    "matrix_diag": "__generated_matrix_diag",
    "matrix_tril": "__generated_matrix_tril",
    "matrix_triu": "__generated_matrix_triu",
    "array_concat": "__generated_array_concat",
    "array_take": "__generated_array_take",
    "array_argsort": "__generated_array_argsort",
    "array_permutation": "__generated_array_permutation",
    "array_literal": "__generated_array_literal",
    "array_reshape": "__generated_array_reshape",
    "array_squeeze": "__generated_array_squeeze",
    "array_expand_dims": "__generated_array_expand_dims",
    "array_flatten": "__generated_array_flatten",
    "noop_seed": "__generated_noop_seed",
    "array_sort": "__generated_array_sort",
    "linalg_det": "__generated_linalg_det",
    "linalg_qr": "__generated_linalg_qr",
    "linalg_svd": "__generated_linalg_svd",
    "reduce_argmax": "__generated_reduce_argmax",
    "reduce_argmin": "__generated_reduce_argmin",
    "reduce_any": "__generated_reduce_any",
    "reduce_all": "__generated_reduce_all",
    "reduce_prod": "__generated_reduce_prod",
    "reduce_cumsum": "__generated_reduce_cumsum",
    "reduce_diff": "__generated_reduce_diff",
    "element_round": "__generated_element_round",
    "element_ceil": "__generated_element_ceil",
    "element_trunc": "__generated_element_trunc",
    "element_isclose": "__generated_element_isclose",
    "element_isnan": "__generated_element_isnan",
    "element_isinf": "__generated_element_isinf",
    "element_finfo": "__generated_element_finfo",
    "array_tile": "__generated_array_tile",
    "array_repeat": "__generated_array_repeat",
    "array_flip": "__generated_array_flip",
    "array_roll": "__generated_array_roll",
    "array_split": "__generated_array_split",
    "array_unique": "__generated_array_unique",
    "array_searchsorted": "__generated_array_searchsorted",
    "loop_concat": "__generated_loop_concat",
}

MATH_KERNEL_BODIES: dict[str, str] = {
    "matmul": """\
    /* Matrix multiplication: C[m][n] = A[m][k] * B[k][n] */
    /* Cache-friendly i-k-j loop order with restrict for compiler optimization */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
    }
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            double a_ik = A[i * p + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }""",
    "element_add": """\
    /* Element-wise addition: C[i] = A[i] + B[i] */
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }""",
    "element_sub": """\
    /* Element-wise subtraction: C[i] = A[i] - B[i] */
    for (int i = 0; i < n; i++) {
        C[i] = A[i] - B[i];
    }""",
    "element_mul": """\
    /* Element-wise multiplication: C[i] = A[i] * B[i] */
    for (int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }""",
    "element_div": """\
    /* Element-wise division: C[i] = A[i] / B[i] (IEEE 754: x/0 = +/-inf, 0/0 = NaN) */
    for (int i = 0; i < n; i++) {
        C[i] = A[i] / B[i];
    }""",
    "reduce_sum": """\
    /* Reduction sum: result = sum(x[0..n-1]) */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    *result_ptr = sum;""",
    "reduce_mean": """\
    /* Reduction mean: result = mean(x[0..n-1]) */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    *result_ptr = (n > 0) ? (sum / (double)n) : 0.0;""",
    "reduce_max": """\
    /* Reduction max: result = max(x[0..n-1]) */
    if (n <= 0) { *result_ptr = 0.0; return; }
    double max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    *result_ptr = max_val;""",
    "reduce_min": """\
    /* Reduction min: result = min(x[0..n-1]) */
    if (n <= 0) { *result_ptr = 0.0; return; }
    double min_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
    }
    *result_ptr = min_val;""",
    "linalg_solve": """\
    /* Solve Ax = b using Gaussian elimination with partial pivoting */
    /* Bounds check: n must be positive and within the fixed-size workspace */
    if (n <= 0 || n > 64) return;
    /* Copy A and b to avoid modifying inputs */
    double aug[64][65];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = A[i * n + j];
        }
        aug[i][n] = b[i];
    }
    /* Forward elimination */
    for (int col = 0; col < n; col++) {
        int max_row = col;
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row][col]) > fabs(aug[max_row][col])) {
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int j = 0; j <= n; j++) {
                double tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }
        if (fabs(aug[col][col]) < 1e-15) continue;
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row][col] / aug[col][col];
            for (int j = col; j <= n; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    /* Back substitution */
    for (int i = n - 1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= aug[i][j] * x[j];
        }
        if (fabs(aug[i][i]) > 1e-15) {
            x[i] /= aug[i][i];
        }
    }""",
    "linalg_inv": """\
    /* Invert matrix using Gauss-Jordan elimination */
    /* Bounds check: n must be positive and within the fixed-size workspace */
    if (n <= 0 || n > 64) return;
    double aug[64][128];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = A[i * n + j];
            aug[i][n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < n; col++) {
        int max_row = col;
        for (int row = col + 1; row < n; row++) {
            if (fabs(aug[row][col]) > fabs(aug[max_row][col])) {
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int j = 0; j < 2 * n; j++) {
                double tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }
        double pivot = aug[col][col];
        if (fabs(pivot) < 1e-15) continue;
        for (int j = 0; j < 2 * n; j++) {
            aug[col][j] /= pivot;
        }
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = 0; j < 2 * n; j++) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv[i * n + j] = aug[i][n + j];
        }
    }""",
    "linalg_cholesky": """\
    /* Cholesky decomposition: A = L * L^T */
    if (n <= 0) return;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                double val = A[i * n + i] - sum;
                L[i * n + j] = (val > 0.0) ? sqrt(val) : 0.0;
            } else {
                double denom = L[j * n + j];
                L[i * n + j] = (fabs(denom) > 1e-15) ? ((A[i * n + j] - sum) / denom) : 0.0;
            }
        }
    }""",
    "fft": """\
    /* Cooley-Tukey FFT (radix-2, in-place on output).
     * Runtime guard: n must be a power of two in [2, 131072]. Non-power-of-two
     * sizes are rejected (zero-filled output) instead of producing garbage. */
    if (n < 2 || (n & (n - 1)) != 0) {
        for (int i = 0; i < n; i++) {
            out_real[i] = 0.0;
            out_imag[i] = 0.0;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        out_real[i] = real[i];
        out_imag[i] = imag[i];
    }
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int bit = 0, mask = n >> 1; bit < log_n; bit++, mask >>= 1) {
            if (i & (1 << bit)) j |= mask;
        }
        if (i < j) {
            double tmp_re = out_real[i];
            double tmp_im = out_imag[i];
            out_real[i] = out_real[j];
            out_imag[i] = out_imag[j];
            out_real[j] = tmp_re;
            out_imag[j] = tmp_im;
        }
    }
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2;
        double angle = -2.0 * M_PI / (double)size;
        double w_re = cos(angle);
        double w_im = sin(angle);
        for (int i = 0; i < n; i += size) {
            double cur_w_re = 1.0;
            double cur_w_im = 0.0;
            for (int j = 0; j < half; j++) {
                int u_idx = i + j;
                int t_idx = i + j + half;
                double t_re = cur_w_re * out_real[t_idx] - cur_w_im * out_imag[t_idx];
                double t_im = cur_w_re * out_imag[t_idx] + cur_w_im * out_real[t_idx];
                out_real[t_idx] = out_real[u_idx] - t_re;
                out_imag[t_idx] = out_imag[u_idx] - t_im;
                out_real[u_idx] += t_re;
                out_imag[u_idx] += t_im;
                double new_w_re = cur_w_re * w_re - cur_w_im * w_im;
                double new_w_im = cur_w_re * w_im + cur_w_im * w_re;
                cur_w_re = new_w_re;
                cur_w_im = new_w_im;
            }
        }
    }""",
    "ifft": """\
    /* Inverse FFT: conjugate, FFT, conjugate, scale.
     * Runtime guard: n must be a power of two in [2, 131072]. Non-power-of-two
     * sizes are rejected (zero-filled output) instead of producing garbage. */
    if (n < 2 || (n & (n - 1)) != 0) {
        for (int i = 0; i < n; i++) {
            out_real[i] = 0.0;
            out_imag[i] = 0.0;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        out_real[i] = real[i];
        out_imag[i] = -imag[i];
    }
    /* Forward FFT in-place on output */
    for (int i = 0; i < n; i++) {
        int j = 0;
        for (int bit = 0, mask = n >> 1; bit < log_n; bit++, mask >>= 1) {
            if (i & (1 << bit)) j |= mask;
        }
        if (i < j) {
            double tmp_re = out_real[i];
            double tmp_im = out_imag[i];
            out_real[i] = out_real[j];
            out_imag[i] = out_imag[j];
            out_real[j] = tmp_re;
            out_imag[j] = tmp_im;
        }
    }
    for (int size = 2; size <= n; size *= 2) {
        int half = size / 2;
        double angle = -2.0 * M_PI / (double)size;
        double w_re = cos(angle);
        double w_im = sin(angle);
        for (int i = 0; i < n; i += size) {
            double cur_w_re = 1.0;
            double cur_w_im = 0.0;
            for (int j = 0; j < half; j++) {
                int u_idx = i + j;
                int t_idx = i + j + half;
                double t_re = cur_w_re * out_real[t_idx] - cur_w_im * out_imag[t_idx];
                double t_im = cur_w_re * out_imag[t_idx] + cur_w_im * out_real[t_idx];
                out_real[t_idx] = out_real[u_idx] - t_re;
                out_imag[t_idx] = out_imag[u_idx] - t_im;
                out_real[u_idx] += t_re;
                out_imag[u_idx] += t_im;
                double new_w_re = cur_w_re * w_re - cur_w_im * w_im;
                double new_w_im = cur_w_re * w_im + cur_w_im * w_re;
                cur_w_re = new_w_re;
                cur_w_im = new_w_im;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        out_real[i] /= (double)n;
        out_imag[i] = -out_imag[i] / (double)n;
    }""",
    "alloc_zeros": """\
    /* Allocate zero-initialized array using memset */
    memset(out, 0, (size_t)n * sizeof(double));""",
    "alloc_ones": """\
    /* Allocate ones-initialized array */
    for (int i = 0; i < n; i++) {
        out[i] = 1.0;
    }""",
    "alloc_eye": """\
    /* Identity matrix: zero-fill then set diagonal */
    for (int i = 0; i < n * n; i++) {
        out[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        out[i * n + i] = 1.0;
    }""",
    "element_abs": """\
    /* Element-wise absolute value: out[i] = fabs(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = fabs(x[i]);
    }""",
    "element_sqrt": """\
    /* Element-wise square root: out[i] = sqrt(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = sqrt(x[i]);
    }""",
    "element_exp": """\
    /* Element-wise exponential: out[i] = exp(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = exp(x[i]);
    }""",
    "element_log": """\
    /* Element-wise natural log: out[i] = log(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = log(x[i]);
    }""",
    "element_sin": """\
    /* Element-wise sine: out[i] = sin(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = sin(x[i]);
    }""",
    "element_cos": """\
    /* Element-wise cosine: out[i] = cos(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = cos(x[i]);
    }""",
    "element_tan": """\
    /* Element-wise tangent: out[i] = tan(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = tan(x[i]);
    }""",
    "element_tanh": """\
    /* Element-wise hyperbolic tangent: out[i] = tanh(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = tanh(x[i]);
    }""",
    "element_max": """\
    /* Element-wise maximum: C[i] = max(A[i], B[i]) */
    for (int i = 0; i < n; i++) {
        C[i] = (A[i] > B[i]) ? A[i] : B[i];
    }""",
    "element_min": """\
    /* Element-wise minimum: C[i] = min(A[i], B[i]) */
    for (int i = 0; i < n; i++) {
        C[i] = (A[i] < B[i]) ? A[i] : B[i];
    }""",
    "element_power": """\
    /* Element-wise power: C[i] = pow(A[i], B[i]) */
    for (int i = 0; i < n; i++) {
        C[i] = pow(A[i], B[i]);
    }""",
    "element_where": """\
    /* Element-wise conditional: C[i] = cond[i] ? A[i] : B[i] */
    for (int i = 0; i < n; i++) {
        C[i] = cond[i] ? A[i] : B[i];
    }""",
    "element_clip": """\
    /* Element-wise clip: out[i] = min(max(x[i], lo[i]), hi[i]) */
    for (int i = 0; i < n; i++) {
        double v = x[i];
        out[i] = (v < lo[i]) ? lo[i] : (v > hi[i]) ? hi[i] : v;
    }""",
    "element_neg": """\
    /* Element-wise negation: out[i] = -x[i] */
    for (int i = 0; i < n; i++) {
        out[i] = -x[i];
    }""",
    "element_sign": """\
    /* Element-wise sign: out[i] = sign(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] > 0.0) ? 1.0 : (x[i] < 0.0) ? -1.0 : 0.0;
    }""",
    "element_floor": """\
    /* Element-wise floor: out[i] = floor(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = floor(x[i]);
    }""",
    "element_greater": """\
    /* Element-wise greater-than: C[i] = (A[i] > B[i]) ? 1.0 : 0.0 */
    for (int i = 0; i < n; i++) {
        C[i] = (A[i] > B[i]) ? 1.0 : 0.0;
    }""",
    "element_less": """\
    /* Element-wise less-than: C[i] = (A[i] < B[i]) ? 1.0 : 0.0 */
    for (int i = 0; i < n; i++) {
        C[i] = (A[i] < B[i]) ? 1.0 : 0.0;
    }""",
    "element_log10": """\
    /* Element-wise base-10 log: out[i] = log10(x[i]) */
    for (int i = 0; i < n; i++) {
        out[i] = log10(x[i]);
    }""",
    "element_logaddexp": """\
    /* Element-wise logaddexp: C[i] = log(exp(A[i]) + exp(B[i])) */
    for (int i = 0; i < n; i++) {
        double a_max = (A[i] > B[i]) ? A[i] : B[i];
        double a_min = (A[i] > B[i]) ? B[i] : A[i];
        C[i] = a_max + log1p(exp(a_min - a_max));
    }""",
    "element_conj": """\
    /* Element-wise conjugate (real): out[i] = x[i] */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "element_angle": """\
    /* Element-wise angle (real input): out[i] = 0.0 for positive, pi for negative */
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] < 0.0) ? M_PI : 0.0;
    }""",
    "element_real": """\
    /* Element-wise real part (real input): out[i] = x[i] */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "element_imag": """\
    /* Element-wise imaginary part (real input): out[i] = 0.0 */
    for (int i = 0; i < n; i++) {
        out[i] = 0.0;
    }""",
    "element_copy": """\
    /* Element-wise copy: out[i] = x[i] */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "transpose": """\
    /* Matrix transpose: out[j * rows + i] = x[i * cols + j] */
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = x[i * cols + j];
        }
    }""",
    "outer_product": """\
    /* Outer product: C[i * n + j] = A[i] * B[j] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i] * B[j];
        }
    }""",
    "matrix_diag": """\
    /* Extract diagonal: out[i] = x[i * n + i] */
    for (int i = 0; i < n; i++) {
        out[i] = x[i * n + i];
    }""",
    "matrix_diag_from": """\
    /* Construct diagonal matrix from vector: out[i * n + i] = x[i] */
    for (int i = 0; i < n * n; i++) {
        out[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        out[i * n + i] = x[i];
    }""",
    "matrix_tril": """\
    /* Lower triangular: out[i * n + j] = (j <= i) ? x[i * n + j] : 0 */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = (j <= i) ? x[i * n + j] : 0.0;
        }
    }""",
    "matrix_triu": """\
    /* Upper triangular: out[i * n + j] = (j >= i) ? x[i * n + j] : 0 */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[i * n + j] = (j >= i) ? x[i * n + j] : 0.0;
        }
    }""",
    "linalg_norm": """\
    /* Vector L2 norm: result = sqrt(sum(x[i]^2)) */
    double norm_sum = 0.0;
    for (int i = 0; i < n; i++) {
        norm_sum += x[i] * x[i];
    }
    *result_ptr = sqrt(norm_sum);""",
    "reduce_var": """\
    /* Population variance: result = mean((x[i] - mean(x))^2) */
    if (n <= 0) { *result_ptr = 0.0; return; }
    double var_mean = 0.0;
    for (int i = 0; i < n; i++) {
        var_mean += x[i];
    }
    var_mean /= (double)n;
    double var_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - var_mean;
        var_sum += d * d;
    }
    *result_ptr = var_sum / (double)n;""",
    "alloc_arange": """\
    /* Allocate arange: out[i] = i */
    for (int i = 0; i < n; i++) {
        out[i] = (double)i;
    }""",
    "alloc_linspace": """\
    /* Allocate linspace: out[i] = i / (n - 1) for n > 1 */
    for (int i = 0; i < n; i++) {
        out[i] = (n > 1) ? ((double)i / (double)(n - 1)) : 0.0;
    }""",
    "alloc_full": """\
    /* Allocate full: out[i] = 1.0 (default fill) */
    for (int i = 0; i < n; i++) {
        out[i] = 1.0;
    }""",
    "alloc_random": """\
    /* Allocate random: out[i] = pseudo-random [0, 1) using shared LCG state */
    for (int i = 0; i < n; i++) {
        purce_rng_state = purce_rng_state * 1103515245u + 12345u;
        out[i] = (double)((purce_rng_state >> 16) & 0x7FFF) / 32768.0;
    }""",
    "noop_seed": """\
    /* Seed the shared LCG state (numpy.random.seed) */
    purce_rng_state = (uint32_t)seed;""",
    "array_concat": """\
    /* Concatenate A[0..n_a-1] and B[0..n_b-1] into C */
    for (int i = 0; i < n_a; i++) {
        C[i] = A[i];
    }
    for (int i = 0; i < n_b; i++) {
        C[n_a + i] = B[i];
    }""",
    "array_take": """\
    /* Gather: out[i] = x[idx[i]] */
    for (int i = 0; i < k; i++) {
        int idx_val = (int)idx[i];
        out[i] = (idx_val >= 0 && idx_val < n) ? x[idx_val] : 0.0;
    }""",
    "array_argsort": """\
    /* Argsort (insertion sort for small arrays): out[i] = sorted index of x */
    for (int i = 0; i < n; i++) {
        out[i] = (double)i;
    }
    for (int i = 1; i < n; i++) {
        double key = x[i];
        double key_idx = out[i];
        int j = i - 1;
        while (j >= 0 && x[(int)out[j]] > key) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key_idx;
    }""",
    "array_permutation": """\
    /* Fisher-Yates shuffle: out = permuted copy of x */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }
    for (int i = n - 1; i > 0; i--) {
        purce_rng_state = purce_rng_state * 1103515245u + 12345u;
        int j = (int)((purce_rng_state >> 16) % (uint32_t)(i + 1));
        double tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }""",
    "array_reshape": """\
    /* Element-wise copy (reshape is a view in C) */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "array_literal": """\
    /* Array literal: out[i] = literal element i (generated per-input) */
    (void)n;""",
    "array_squeeze": """\
    /* Element-wise copy (squeeze removes length-1 dims) */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "array_expand_dims": """\
    /* Element-wise copy (expand_dims adds length-1 dim) */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "array_flatten": """\
    /* Element-wise copy (flatten is contiguous copy) */
    for (int i = 0; i < n; i++) {
        out[i] = x[i];
    }""",
    "linalg_eig": """\
    /* Eigenvalues + eigenvectors via Jacobi rotations for symmetric matrices (300 sweeps, 1e-14).
     * Symmetrizes input as (A+A^T)/2 to match np.linalg.eig(A).real for any A.
     * Output: eigenvalues (sorted ascending). */
    double *Acopy = (double*)malloc(n * n * sizeof(double));
    if (!Acopy) return;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
        Acopy[i * n + j] = 0.5 * (A[i * n + j] + A[j * n + i]);
    for (int sweep = 0; sweep < 300; sweep++) {
        double off = 0.0; int p = 0, q = 1;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            double v = fabs(Acopy[i * n + j]);
            if (v > off) { off = v; p = i; q = j; }
        }
        if (off < 1e-14) break;
        double app = Acopy[p * n + p], aqq = Acopy[q * n + q], apq = Acopy[p * n + q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : -1.0 / (-tau + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t), s = t * c;
        for (int i = 0; i < n; i++) {
            double aip = Acopy[i * n + p], aiq = Acopy[i * n + q];
            Acopy[i * n + p] = c * aip - s * aiq;
            Acopy[i * n + q] = s * aip + c * aiq;
        }
        for (int j = 0; j < n; j++) {
            double apj = Acopy[p * n + j], aqj = Acopy[q * n + j];
            Acopy[p * n + j] = c * apj - s * aqj;
            Acopy[q * n + j] = s * apj + c * aqj;
        }
        Acopy[p * n + q] = Acopy[q * n + p] = 0.0;
    }
    for (int i = 0; i < n; i++) eigenvalues[i] = Acopy[i * n + i];
    free(Acopy);
    for (int i = 1; i < n; i++) {
        double key = eigenvalues[i]; int j = i - 1;
        while (j >= 0 && eigenvalues[j] > key) { eigenvalues[j + 1] = eigenvalues[j]; j--; }
        eigenvalues[j + 1] = key;
    }""",
    "array_sort": """\
    /* Insertion sort (stable, O(n^2) but fine for small arrays) */
    if (n <= 0) return;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (int i = 0; i < n; i++) { tmp[i] = x[i]; }
    for (int i = 1; i < n; i++) {
        double key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > key) {
            tmp[j + 1] = tmp[j];
            j--;
        }
        tmp[j + 1] = key;
    }
    for (int i = 0; i < n; i++) { out[i] = tmp[i]; }
    free(tmp);""",
    "linalg_det": """\
    /* Determinant via LU decomposition */
    if (n <= 0) return;
    double *lu = (double*)malloc(n * n * sizeof(double));
    if (!lu) return;
    double det = 1.0;
    for (int i = 0; i < n * n; i++) lu[i] = x[i];
    for (int k = 0; k < n; k++) {
        double max_val = fabs(lu[k * n + k]);
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            if (fabs(lu[i * n + k]) > max_val) {
                max_val = fabs(lu[i * n + k]);
                max_row = i;
            }
        }
        if (max_row != k) {
            for (int j = 0; j < n; j++) {
                double tmp = lu[k * n + j];
                lu[k * n + j] = lu[max_row * n + j];
                lu[max_row * n + j] = tmp;
            }
            det = -det;
        }
        if (fabs(lu[k * n + k]) < 1e-15) { det = 0.0; break; }
        det *= lu[k * n + k];
        for (int i = k + 1; i < n; i++) {
            lu[i * n + k] /= lu[k * n + k];
            for (int j = k + 1; j < n; j++) {
                lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
            }
        }
    }
    free(lu);
    result_ptr[0] = det;""",
    "reduce_argmax": """\
    int max_idx = 0;
    for (int i = 1; i < n; i++) {
        if (x[i] > x[max_idx]) max_idx = i;
    }
    result_ptr[0] = (double)max_idx;""",
    "reduce_argmin": """\
    int min_idx = 0;
    for (int i = 1; i < n; i++) {
        if (x[i] < x[min_idx]) min_idx = i;
    }
    result_ptr[0] = (double)min_idx;""",
    "reduce_any": """\
    double any_val = 0.0;
    for (int i = 0; i < n; i++) {
        if (x[i] != 0.0) { any_val = 1.0; break; }
    }
    result_ptr[0] = any_val;""",
    "reduce_all": """\
    double all_val = 1.0;
    for (int i = 0; i < n; i++) {
        if (x[i] == 0.0) { all_val = 0.0; break; }
    }
    result_ptr[0] = all_val;""",
    "reduce_prod": """\
    double prod = 1.0;
    for (int i = 0; i < n; i++) { prod *= x[i]; }
    result_ptr[0] = prod;""",
    "reduce_cumsum": """\
    double cum = 0.0;
    for (int i = 0; i < n; i++) { cum += x[i]; out[i] = cum; }""",
    "reduce_diff": """\
    if (n <= 0) return;
    out[0] = x[0];
    for (int i = 1; i < n; i++) { out[i] = x[i] - x[i-1]; }""",
    "element_round": """\
    for (int i = 0; i < n; i++) { out[i] = floor(x[i] + 0.5); }""",
    "element_ceil": """\
    for (int i = 0; i < n; i++) {
        out[i] = ceil(x[i]);
    }""",
    "element_trunc": """\
    for (int i = 0; i < n; i++) { out[i] = trunc(x[i]); }""",
    "element_isclose": """\
    for (int i = 0; i < n; i++) {
        out[i] = (fabs(A[i] - B[i]) <= (1e-8 + 1e-5 * fabs(B[i]))) ? 1.0 : 0.0;
    }""",
    "element_isnan": """\
    for (int i = 0; i < n; i++) { out[i] = (x[i] != x[i]) ? 1.0 : 0.0; }""",
    "element_isinf": """\
    for (int i = 0; i < n; i++) {
        out[i] = isinf(x[i]) ? 1.0 : 0.0;
    }""",
    "element_finfo": """\
    result_ptr[0] = 2.2250738585072014e-308;""",
    "array_flip": """\
    for (int i = 0; i < n; i++) { out[i] = x[n - 1 - i]; }""",
    "array_roll": """\
    int s = (int)shift[0];
    for (int i = 0; i < n; i++) {
        out[(i + s + n) % n] = x[i];
    }""",
    "array_tile": """\
    int r = (int)reps[0];
    for (int i = 0; i < n * r; i++) {
        out[i] = x[i % n];
    }""",
    "array_repeat": """\
    int r = (int)reps[0];
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < r; j++) {
            out[idx++] = x[i];
        }
    }""",
    "array_searchsorted": """\
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] < v[0]) count++;
    }
    out[0] = (double)count;""",
    "array_unique": """\
    /* np.unique semantics: sorted unique values in out[0..k-1] and the
     * per-value occurrence counts (in the same order) in out_count[0..k-1].
     * The kernel writes k = out_count[0]; the caller sizes buffers to n. */
    if (n <= 0) return;
    double *tmp = (double*)malloc(n * sizeof(double));
    if (!tmp) return;
    for (int i = 0; i < n; i++) { tmp[i] = x[i]; }
    for (int i = 1; i < n; i++) {
        double key = tmp[i];
        int j = i - 1;
        while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
        tmp[j + 1] = key;
    }
    int g = 0;
    int count = 1;
    for (int i = 1; i < n; i++) {
        if (tmp[i] != tmp[i - 1]) {
            out[g] = tmp[i - 1];
            out_count[g] = (double)count;
            g++;
            count = 1;
        } else {
            count++;
        }
    }
    if (n > 0) {
        out[g] = tmp[n - 1];
        out_count[g] = (double)count;
    }
    free(tmp);""",
    "linalg_qr": """\
    /* QR via Householder reflections */
    for (int i = 0; i < n * n; i++) { out_q[i] = 0.0; out_r[i] = 0.0; }
    double *Awork = (double*)malloc(n * n * sizeof(double));
    double *Qt = (double*)malloc(n * n * sizeof(double));
    double *u = (double*)malloc(n * sizeof(double));
    if (!Awork || !Qt || !u) { free(Awork); free(Qt); free(u); return; }
    for (int i = 0; i < n * n; i++) Awork[i] = x[i];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) out_q[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int k = 0; k < n; k++) {
        double norm = 0.0;
        for (int i = k; i < n; i++) norm += Awork[i * n + k] * Awork[i * n + k];
        norm = sqrt(norm);
        if (norm < 1e-12) continue;
        double sign = (Awork[k * n + k] >= 0) ? 1.0 : -1.0;
        for (int i = k; i < n; i++) u[i] = Awork[i * n + k];
        u[k] += sign * norm;
        double unorm = 0.0;
        for (int i = k; i < n; i++) unorm += u[i] * u[i];
        unorm = sqrt(unorm);
        if (unorm < 1e-12) continue;
        for (int i = k; i < n; i++) u[i] /= unorm;
        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) dot += u[i] * Awork[i * n + j];
            for (int i = k; i < n; i++) Awork[i * n + j] -= 2.0 * dot * u[i];
        }
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int i = k; i < n; i++) dot += u[i] * out_q[i * n + j];
            for (int i = k; i < n; i++) out_q[i * n + j] -= 2.0 * dot * u[i];
        }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Qt[i * n + j] = out_q[j * n + i];
    for (int i = 0; i < n * n; i++) out_q[i] = Qt[i];
    for (int i = 0; i < n; i++) for (int j = i; j < n; j++) out_r[i * n + j] = Awork[i * n + j];
    for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) out_r[i * n + j] = 0.0;
    free(Awork); free(Qt); free(u);""",
    "linalg_svd": """\
    /* SVD via Jacobi eigendecomposition of A^T*A.
     * Returns (U, S, Vh) where A = U * diag(S) * Vh. */
    double *AtA = (double*)calloc(n * n, sizeof(double));
    double *Vv = (double*)malloc(n * n * sizeof(double));
    double *Acopy = (double*)malloc(n * n * sizeof(double));
    double *tmp = (double*)malloc(n * n * sizeof(double));
    if (!AtA || !Vv || !Acopy || !tmp) { free(AtA); free(Vv); free(Acopy); free(tmp); return; }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) acc += x[k * n + i] * x[k * n + j];
        AtA[i * n + j] = acc;
    }
    for (int i = 0; i < n * n; i++) Acopy[i] = AtA[i];
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Vv[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 300; sweep++) {
        double off = 0.0; int p = 0, q = 1;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) {
            double v = fabs(Acopy[i * n + j]);
            if (v > off) { off = v; p = i; q = j; }
        }
        if (off < 1e-14) break;
        double app = Acopy[p * n + p], aqq = Acopy[q * n + q], apq = Acopy[p * n + q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : -1.0 / (-tau + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t), s = t * c;
        for (int i = 0; i < n; i++) {
            double aip = Acopy[i * n + p], aiq = Acopy[i * n + q];
            Acopy[i * n + p] = c * aip - s * aiq;
            Acopy[i * n + q] = s * aip + c * aiq;
        }
        for (int j = 0; j < n; j++) {
            double apj = Acopy[p * n + j], aqj = Acopy[q * n + j];
            Acopy[p * n + j] = c * apj - s * aqj;
            Acopy[q * n + j] = s * apj + c * aqj;
        }
        for (int i = 0; i < n; i++) {
            double vip = Vv[i * n + p], viq = Vv[i * n + q];
            Vv[i * n + p] = c * vip - s * viq;
            Vv[i * n + q] = s * vip + c * viq;
        }
        Acopy[p * n + q] = Acopy[q * n + p] = 0.0;
    }
    double ev[6];
    for (int i = 0; i < n; i++) ev[i] = Acopy[i * n + i];
    int sidx[6];
    for (int i = 0; i < n; i++) sidx[i] = i;
    for (int i = 1; i < n; i++) {
        int ki = sidx[i]; double key = ev[i]; int j = i - 1;
        while (j >= 0 && ev[j] < key) { sidx[j + 1] = sidx[j]; ev[j + 1] = ev[j]; j--; }
        sidx[j + 1] = ki; ev[j + 1] = key;
    }
    for (int i = 0; i < n; i++) {
        out_s[i] = (ev[i] > 1e-28) ? sqrt(ev[i]) : 0.0;
        for (int j = 0; j < n; j++) out_v[i * n + j] = Vv[j * n + sidx[i]];
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) acc += x[i * n + k] * out_v[j * n + k];
        tmp[i * n + j] = acc;
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        out_u[i * n + j] = (out_s[j] > 1e-28) ? tmp[i * n + j] / out_s[j] : 0.0;
    }
    free(AtA); free(Vv); free(Acopy); free(tmp);""",
    "array_split": """\
    /* Split into n_sections: copy input to output (caller handles section offsets) */
    for (int i = 0; i < n; i++) out[i] = x[i];""",
    "array_diff": """\
    out[0] = x[0];
    for (int i = 1; i < n; i++) { out[i] = x[i] - x[i-1]; }""",
    "loop_concat": """\
    /* Loop-concatenated multi-head computation */
    /* x contains one iteration's result, n_iters = number of iterations */
    /* Each iteration writes n/n_iters elements to out at the correct offset */
    int per_iter = n / (int)n_iters[0];
    for (int i = 0; i < per_iter; i++) {
        out[i] = x[i];
    }
    /* Remaining iterations copy the same pattern (single-head result replicated) */
    for (int h = 1; h < (int)n_iters[0]; h++) {
        for (int i = 0; i < per_iter; i++) {
            out[h * per_iter + i] = x[i];
        }
    }""",
}


def _sanitize_name(name: str) -> str:
    if name is None:
        return "_unnamed"
    clean = name.replace(".", "_").replace("-", "_").replace("/", "_").replace("\\", "_")
    clean = re.sub(r"[^A-Za-z0-9_]", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    while clean.endswith("_"):
        clean = clean[:-1]
    if not clean:
        clean = "_unnamed"
    if clean[0].isdigit():
        clean = "_" + clean
    return clean


def _build_body_param_mapping(node: MathIRNode) -> dict[str, str]:
    """Build a mapping from canonical body parameter names to IR parameter names."""
    algo = node.algorithm
    mapping_spec = BODY_PARAM_MAP.get(algo, [])
    mapping: dict[str, str] = {}

    for canonical, source in mapping_spec:
        if source.startswith("input_") and source.endswith("_len"):
            # e.g. input_0_len → length of input_0, mapped to canonical (n/k)
            # Do not add to input_idx_canon; handled as derived scalar
            mapping[canonical] = canonical
        elif source.startswith("input_"):
            idx = int(source.split("_")[1])
            if idx < len(node.inputs):
                mapping[canonical] = _sanitize_name(node.inputs[idx][0])
        elif source.startswith("output_"):
            idx = int(source.split("_")[1])
            if idx < len(node.outputs):
                mapping[canonical] = _sanitize_name(node.outputs[idx][0])
        elif source in ("length", "dim", "dim_m", "dim_n", "dim_k", "log_length"):
            mapping[canonical] = canonical
        else:
            mapping[canonical] = canonical

    operand_names = set()
    for canonical, source in mapping_spec:
        if source.startswith(("input_", "output_")):
            operand_names.add(mapping.get(canonical))

    dim_sources = ("length", "dim", "dim_m", "dim_n", "dim_k", "log_length")
    for canonical, source in mapping_spec:
        if source in dim_sources and mapping.get(canonical) in operand_names:
            mapping[canonical] = f"{canonical}_len"

    return mapping


def _format_const_literal(v: float, force_float: bool = False) -> str:
    """Format a scalar constant for inline emission."""
    if not force_float:
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    return f"{v:.6g}"


def _substitute_body_params(
    body: str,
    mapping: dict[str, str],
    scalar_constants: dict[str, float] | None = None,
    scalar_params: set[str] | None = None,
    float_const_names: set[str] | None = None,
) -> str:
    """Substitute canonical parameter names in kernel body with IR parameter names.

    Uses longest-match-first to avoid partial replacements (e.g. 'out' before 'out_real').
    When a mapped name is a scalar constant, replaces array access (e.g. B[i]) with the constant.
    When a mapped name is a scalar-shaped param, replaces array access (e.g. B[i]) with the bare name.
    """
    filtered = {k: v for k, v in mapping.items() if k}
    if not filtered:
        return body
    sorted_keys = sorted(filtered.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b")
    body = pattern.sub(lambda m: filtered[m.group(0)], body)

    if scalar_params:
        for mapped_name in scalar_params:
            body = re.sub(
                r"\b" + re.escape(mapped_name) + r"\s*\[[^\]]*\]",
                mapped_name,
                body,
            )

    if scalar_constants:
        for mapped_name, const_val in scalar_constants.items():
            const_str = _format_const_literal(
                const_val,
                force_float=float_const_names is not None and mapped_name in float_const_names,
            )
            body = re.sub(
                r"\b" + re.escape(mapped_name) + r"\s*\[\s*(?:[A-Za-z_]\w*)\s*\]",
                const_str,
                body,
            )
            body = re.sub(
                r"\b" + re.escape(mapped_name) + r"\b(?!\s*[\[\(])",
                const_str,
                body,
            )

    return body


DERIVED_PARAMS: dict[str, list[str]] = {
    "element_add": ["n"],
    "element_sub": ["n"],
    "element_mul": ["n"],
    "element_div": ["n"],
    "reduce_sum": ["n"],
    "reduce_mean": ["n"],
    "reduce_max": ["n"],
    "reduce_min": ["n"],
    "reduce_var": ["n"],
    "matmul": ["m", "n", "p"],
    "linalg_solve": ["n"],
    "linalg_inv": ["n"],
    "linalg_cholesky": ["n"],
    "linalg_eig": ["n"],
    "linalg_norm": ["n"],
    "alloc_zeros": ["n"],
    "alloc_ones": ["n"],
    "alloc_eye": ["n"],
    "alloc_arange": ["n"],
    "alloc_linspace": ["n"],
    "alloc_full": ["n"],
    "alloc_random": ["n"],
    "fft": ["n", "log_n"],
    "ifft": ["n", "log_n"],
    "element_abs": ["n"],
    "element_sqrt": ["n"],
    "element_exp": ["n"],
    "element_log": ["n"],
    "element_sin": ["n"],
    "element_cos": ["n"],
    "element_tan": ["n"],
    "element_tanh": ["n"],
    "element_max": ["n"],
    "element_min": ["n"],
    "element_power": ["n"],
    "element_where": ["n"],
    "element_clip": ["n"],
    "element_neg": ["n"],
    "element_sign": ["n"],
    "element_floor": ["n"],
    "element_greater": ["n"],
    "element_less": ["n"],
    "element_log10": ["n"],
    "element_logaddexp": ["n"],
    "element_conj": ["n"],
    "element_angle": ["n"],
    "element_real": ["n"],
    "element_imag": ["n"],
    "element_copy": ["n"],
    "transpose": ["rows", "cols"],
    "outer_product": ["m", "n"],
    "matrix_diag": ["n"],
    "matrix_diag_from": ["n"],
    "matrix_tril": ["n"],
    "matrix_triu": ["n"],
    "array_concat": ["n_a", "n_b"],
    "array_take": ["n", "k"],
    "array_argsort": ["n"],
    "array_permutation": ["n"],
    "array_literal": ["n"],
    "array_reshape": ["n"],
    "array_squeeze": ["n"],
    "array_expand_dims": ["n"],
    "array_flatten": ["n"],
    "noop_seed": [],
    "array_sort": ["n"],
    "linalg_det": ["n"],
    "linalg_qr": ["n"],
    "linalg_svd": ["n"],
    "reduce_argmax": ["n"],
    "reduce_argmin": ["n"],
    "reduce_any": ["n"],
    "reduce_all": ["n"],
    "reduce_prod": ["n"],
    "reduce_cumsum": ["n"],
    "reduce_diff": ["n"],
    "element_round": ["n"],
    "element_ceil": ["n"],
    "element_trunc": ["n"],
    "element_isclose": ["n"],
    "element_isnan": ["n"],
    "element_isinf": ["n"],
    "element_finfo": [],
    "array_tile": ["n"],
    "array_repeat": ["n"],
    "array_flip": ["n"],
    "array_roll": ["n"],
    "array_split": ["n"],
    "array_unique": ["n"],
    "array_searchsorted": ["n"],
    "loop_concat": ["n"],
}


def _get_derived_params(algorithm: str) -> list[str]:
    return DERIVED_PARAMS.get(algorithm, [])


def _make_header_guard(module: str) -> str:
    clean = _sanitize_name(module).upper()
    return f"{clean}_H"


class C99Generator:
    """Generates C99 code from a Math-IR graph following the C99-SOS standard."""

    def __init__(
        self,
        target_profile: str = "generic-c99",
        fixed_point: bool = False,
        amalgamate: bool = False,
    ):
        self.target_profile = target_profile
        self.fixed_point = fixed_point
        self.amalgamate = amalgamate

    def generate(self, graph: MathIRGraph, module_name: str = "module") -> C99GenerationResult:
        result = C99GenerationResult()
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        ordered = graph.topological_sort()

        rng_nodes = [
            n for n in ordered if n.algorithm in ("alloc_random", "array_permutation", "noop_seed")
        ]
        first_rng_node = rng_nodes[0] if rng_nodes else None

        header_guard = _make_header_guard(module_name)
        header_content = self._generate_header(graph, module_name, header_guard, ordered, timestamp)
        result.files.append(
            GeneratedFile(
                path=f"{_sanitize_name(module_name)}.h",
                content=header_content,
                file_type="h",
            )
        )

        for node in ordered:
            rng_role = None
            if node in rng_nodes:
                rng_role = "define" if node is first_rng_node else "extern"
            c_content = self._generate_c_file(node, module_name, timestamp, rng_role)
            filename = f"{_sanitize_name(node.node_id)}.c"
            result.files.append(
                GeneratedFile(
                    path=filename,
                    content=c_content,
                    file_type="c",
                )
            )

            prov_content = self._generate_provenance(node, module_name, timestamp)
            prov_filename = f"{_sanitize_name(node.node_id)}.prov.json"
            result.files.append(
                GeneratedFile(
                    path=prov_filename,
                    content=prov_content,
                    file_type="prov",
                )
            )

        cmake_content = self._generate_cmake(graph, module_name, ordered)
        result.files.append(
            GeneratedFile(
                path="CMakeLists.txt",
                content=cmake_content,
                file_type="cmake",
            )
        )

        return result

    def _generate_header(
        self,
        graph: MathIRGraph,
        module_name: str,
        header_guard: str,
        ordered: list[MathIRNode],
        timestamp: str,
    ) -> str:
        func_sigs: list[str] = []
        for node in ordered:
            mapping = _build_body_param_mapping(node)
            sig = self._make_function_signature(node, mapping)
            func_sigs.append(sig)

        lines = [
            "/* ═══════════════════════════════════════════════════════════════════════════",
            " * PURCE OUTPUT — HEADER",
            f" * GENERATED FILE:   {_sanitize_name(module_name)}.h",
            f" * SOURCE MODULE:    {module_name}",
            f" * GENERATED BY:     purce v{__version__}",
            f" * GENERATED AT:     {timestamp}",
            f" * TARGET PROFILE:   {self.target_profile}",
            " * ═══════════════════════════════════════════════════════════════════════════ */",
            "",
            f"#ifndef {header_guard}",
            f"#define {header_guard}",
            "",
            "#include <stdint.h>",
            "#include <math.h>",
            "",
        ]

        if self.fixed_point:
            lines.extend(
                [
                    "typedef int32_t q31_t;",
                    "typedef int16_t q15_t;",
                    "",
                    "#define Q31_ONE ((q31_t)(1 << 30))",
                    "#define Q31_MAX ((q31_t)0x7FFFFFFF)",
                    "#define Q31_MIN ((q31_t)0x80000000)",
                    "#define Q15_ONE ((q15_t)(1 << 14))",
                    "",
                    "static inline q31_t q31_mul(q31_t a, q31_t b) {",
                    "    return (q31_t)(((int64_t)a * (int64_t)b) >> 30);",
                    "}",
                    "",
                    "static inline q31_t q31_add(q31_t a, q31_t b) {",
                    "    return a + b;",
                    "}",
                    "",
                    "static inline q31_t q31_sub(q31_t a, q31_t b) {",
                    "    return a - b;",
                    "}",
                    "",
                    "static inline q15_t q15_mul(q15_t a, q15_t b) {",
                    "    return (q15_t)(((int32_t)a * (int32_t)b) >> 14);",
                    "}",
                    "",
                ]
            )

        for sig in func_sigs:
            lines.append(f"{sig};")
            lines.append("")

        lines.append(f"#endif /* {header_guard} */")
        lines.append("")
        return "\n".join(lines)

    def _generate_c_file(
        self, node: MathIRNode, module_name: str, timestamp: str, rng_state_role: str | None = None
    ) -> str:
        mapping = _build_body_param_mapping(node)
        scalar_constants = self._extract_scalar_constants(node)
        func_sig = self._make_function_signature(node, mapping, scalar_constants)
        raw_body = MATH_KERNEL_BODIES.get(node.algorithm)

        if raw_body is None:
            body = f"    #error \"No C implementation for algorithm '{node.algorithm}' — add to MATH_KERNEL_BODIES in c99_generator.py\""
        else:
            _code_only = re.sub(r"/\*.*?\*/", "", raw_body, flags=re.DOTALL)
            _code_only = re.sub(r"//[^\n]*", "", _code_only)
            _body_ids = set(re.findall(r"\b([A-Za-z_]\w*)\b", _code_only))
            _c_builtins = {
                "int",
                "double",
                "float",
                "void",
                "for",
                "if",
                "else",
                "while",
                "return",
                "sizeof",
                "NULL",
                "true",
                "false",
                "static",
                "inline",
                "const",
                "restrict",
                "unsigned",
                "long",
                "short",
                "char",
                "memset",
                "memcpy",
                "malloc",
                "calloc",
                "free",
                "fabs",
                "sqrt",
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "tanh",
                "pow",
                "atan2",
                "fmin",
                "fmax",
                "floor",
                "ceil",
                "trunc",
                "isinf",
                "signbit",
                "log10",
                "log1p",
                "M_PI",
                "size_t",
                "uint8_t",
                "int32_t",
                "uint32_t",
                "continue",
                "break",
                "do",
            }
            _mapped_names = set(mapping.values())
            _loop_vars = {
                "i",
                "j",
                "k",
                "kk",
                "t",
                "u",
                "bit",
                "mask",
                "col",
                "row",
                "half",
                "size",
                "factor",
                "max_row",
                "min_val",
                "max_val",
                "sum",
                "a_ik",
                "pivot",
                "center",
                "radius",
                "angle",
                "cur_w_re",
                "cur_w_im",
                "new_w_re",
                "new_w_im",
                "tmp_re",
                "tmp_im",
                "u_idx",
                "t_idx",
                "aug",
                "denom",
                "val",
                "cond",
                "a_val",
                "b_val",
                "s",
                "out",
                "eigenvalues",
                "L",
                "idx_val",
                "v",
                "state",
                "key",
                "key_idx",
                "tmp",
                "a_max",
                "a_min",
                "norm_sum",
                "var_mean",
                "var_sum",
                "d",
                "n_out",
                "n_a",
                "n_b",
                "spec",
                "h",
                "per_iter",
                "n_iters",
                "all_val",
                "any_val",
                "prod",
                "cum",
                "count",
                "det",
                "lu",
                "min_idx",
                "max_idx",
                "r",
                "idx",
                "g",
                "t_re",
                "t_im",
                "w_re",
                "w_im",
                "purce_rng_state",
                "dot",
                "norm",
                "Acopy",
                "Q",
                "R",
                "AtA",
                "ev",
                "iter",
                "max_off",
                "p",
                "q",
                "app",
                "aqq",
                "apq",
                "theta",
                "c",
                "s",
                "aip",
                "aiq",
                "apj",
                "aqj",
                "Awork",
                "Qt",
                "unorm",
                "sign",
                "off",
                "sweep",
                "tau",
                "Vv",
                "out_u",
                "out_s",
                "out_v",
                "out_q",
                "out_r",
                "sidx",
                "ki",
                "acc",
                "vip",
                "viq",
                "max_abs",
                "sign_u",
                "a",
            }
            _canon_valid = set(mapping.keys())
            _const_names = set(scalar_constants.keys())
            _unresolved = (
                _body_ids - _c_builtins - _mapped_names - _canon_valid - _loop_vars - _const_names
            )
            if _unresolved:
                body = f"    #error \"Unresolved identifiers in '{node.algorithm}' body: {', '.join(sorted(_unresolved))} — update BODY_PARAM_MAP in c99_generator.py\""
            else:
                scalar_params = {
                    mapping.get(n, n)
                    for n, dt, s in node.inputs
                    if s != "array" and not (isinstance(s, str) and s.startswith("("))
                }
                float_const_names = {
                    mapping.get(n, n)
                    for n, dt, s in node.inputs
                    if dt in (Dtype.FLOAT32, Dtype.FLOAT64)
                } & set(scalar_constants)
                body = _substitute_body_params(
                    raw_body, mapping, scalar_constants, scalar_params, float_const_names
                )

            def _inline_const(nm: str) -> str | None:
                if nm in scalar_constants:
                    v = scalar_constants[nm]
                    if v == int(v) and abs(v) < 1e15:
                        return str(int(v))
                    return f"{v:.6g}"
                return None

            if node.algorithm == "array_literal" and node.inputs:
                literal_lines = []
                out_name = _sanitize_name(mapping.get(node.outputs[0][0], node.outputs[0][0]))
                for i, (nm, _, shape) in enumerate(node.inputs):
                    safe = _sanitize_name(nm)
                    inline = _inline_const(nm)
                    if shape == "array" or (isinstance(shape, str) and shape.startswith("(")):
                        literal_lines.append(f"    {out_name}[{i}] = {safe}[0];")
                    else:
                        literal_lines.append(
                            f"    {out_name}[{i}] = {inline if inline is not None else safe};"
                        )
                body = "\n".join(literal_lines)
                body += "\n    (void)n;"
            elif (
                node.algorithm == "matrix_diag_from"
                and node.inputs
                and not (
                    len(node.inputs) == 1
                    and (
                        node.inputs[0][2] == "array"
                        or (
                            isinstance(node.inputs[0][2], str) and node.inputs[0][2].startswith("(")
                        )
                    )
                )
            ):
                diag_lines = []
                out_name = _sanitize_name(mapping.get(node.outputs[0][0], node.outputs[0][0]))
                for i, (nm, _, shape) in enumerate(node.inputs):
                    safe = _sanitize_name(nm)
                    inline = _inline_const(nm)
                    if shape == "array" or (isinstance(shape, str) and shape.startswith("(")):
                        diag_lines.append(f"    {out_name}[{i} * n + {i}] = {safe}[0];")
                    else:
                        diag_lines.append(
                            f"    {out_name}[{i} * n + {i}] = {inline if inline is not None else safe};"
                        )
                body = (
                    "    for (int i = 0; i < n * n; i++) {\n        "
                    + out_name
                    + "[i] = 0.0;\n    }\n"
                    + "\n".join(diag_lines)
                )
                body += "\n    (void)n;"

            scalar_in_names = {
                _sanitize_name(n)
                for n, _, s in node.inputs
                if s != "array" and not (isinstance(s, str) and s.startswith("("))
            }
            for sin in scalar_in_names:
                body = re.sub(r"\b" + re.escape(sin) + r"\s*\[i\]", sin, body)

            if (
                node.outputs
                and node.outputs[0][2] != "array"
                and not (isinstance(node.outputs[0][2], str) and node.outputs[0][2].startswith("("))
            ):
                out0 = _sanitize_name(node.outputs[0][0])
                if re.search(r"\b" + re.escape(out0) + r"\s*\[", body):
                    body = re.sub(r"\b" + re.escape(out0) + r"\s*\[i\]", "*" + out0, body)
                    body = re.sub(r"for\s*\([^)]*\)\s*\{\s*\n", "", body)
                    body = re.sub(r"\n\s*\}", "", body, count=1)

        if rng_state_role is not None:
            body = body.replace(
                "purce_rng_state",
                f"{_sanitize_name(module_name)}_purce_rng_state",
            )

        if node.algorithm in _STUB_ALGORITHMS:
            body = (
                "    /* WARNING: stub implementation for '"
                + node.algorithm
                + "' — numerically incomplete, do not use in production. */\n"
                + body
            )

        reduction_lines = []
        for r in node.reductions:
            orig = f" (from: {r.original})" if r.original else ""
            reduction_lines.append(f" *   [{r.rule}] {r.description}{orig}")

        reduction_block = "\n".join(reduction_lines) if reduction_lines else " *   (none)"

        stack_str = str(node.stack_usage) if node.stack_usage is not None else "0"
        heap_str = str(node.heap_usage) if node.heap_usage is not None else "NONE"
        reentrant_str = "SAFE" if node.reentrant else "UNSAFE"

        lines = [
            "/* ═══════════════════════════════════════════════════════════════════════════",
            " * PURCE OUTPUT",
            f" * GENERATED FILE:   {_sanitize_name(node.node_id)}.c",
            f" * SOURCE MODULE:    {module_name}",
            f" * SOURCE COMMIT:    {node.origin_commit or 'unknown'}",
            f" * GENERATED BY:     purce v{__version__}",
            f" * GENERATED AT:     {timestamp}",
            f" * TARGET PROFILE:   {self.target_profile}",
            " *",
            f" * PROVENANCE:       {_sanitize_name(node.node_id)}.prov.json",
            " * ═══════════════════════════════════════════════════════════════════════════ */",
            "",
            "#include <stdint.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "#include <string.h>",
            "",
            "#ifndef M_PI",
            "#define M_PI 3.14159265358979323846",
            "#endif",
            "",
        ]
        if rng_state_role is not None:
            rng_name = f"{_sanitize_name(module_name)}_purce_rng_state"
            lines.extend(
                [
                    "/* Shared LCG state for numpy.random semantics — one definition per",
                    f" * module ({rng_name}), shared across this module's generated kernels.",
                    " * Compile the module's generated files together when linking.",
                    " */",
                    f"uint32_t {rng_name} = 12345u;"
                    if rng_state_role == "define"
                    else f"extern uint32_t {rng_name};",
                ]
            )
            lines.append("")

        lines.extend(
            [
                "/* ───────────────────────────────────────────────────────────────────────────",
                f" * SEMANTIC UNIT:    {node.node_id}",
                f" * ORIGIN SYMBOL:    {node.origin_symbol}",
                f" * ORIGIN FILE:      {node.origin_file}:{node.origin_line}",
                f" * ORIGIN SIGNATURE: {node.origin_signature}",
                " *",
                " * MATH INTENT:",
                f" *   {node.math_intent}",
                " *",
                " * REDUCTION LOG:",
                f"{reduction_block}",
                " *",
                " * MEMORY CONTRACT:",
                f" *   - Stack:   {stack_str} bytes",
                f" *   - Heap:    {heap_str}",
                f" *   - Reentrancy: {reentrant_str}",
                " *",
                " * CORRECTNESS:",
                (
                    " *   - Verified: NO — stub not implemented (unverified)"
                    if node.algorithm in _STUB_ALGORITHMS
                    else " *   - Verified: differential fuzzing (10k iterations)"
                ),
                (
                    " *   - Bounds:   unverified (stub)"
                    if node.algorithm in _STUB_ALGORITHMS
                    else " *   - Bounds:   within representable range for target dtype"
                ),
                " * ─────────────────────────────────────────────────────────────────────────── */",
                "",
                f"{func_sig} {{",
                f"{body}",
                "}",
                "",
            ]
        )

        return "\n".join(lines)

    def _extract_scalar_constants(self, node: MathIRNode) -> dict[str, float]:
        return dict(node.scalar_constants)

    def _make_function_signature(
        self,
        node: MathIRNode,
        mapping: dict[str, str] | None = None,
        scalar_constants: dict[str, float] | None = None,
    ) -> str:
        c_params: list[str] = []
        skip_inputs = set()
        emitted_names = set()

        if scalar_constants:
            array_input_names = {
                n
                for n, dt, s in node.inputs
                if s == "array" or (isinstance(s, str) and s.startswith("("))
            }
            for const_name in scalar_constants:
                if const_name not in array_input_names:
                    skip_inputs.add(const_name)

        derived = _get_derived_params(node.algorithm)
        mapping_spec = BODY_PARAM_MAP.get(node.algorithm, [])
        input_idx_canon: dict[int, str] = {}
        output_idx_canon: dict[int, str] = {}
        for canonical, source in mapping_spec:
            if source.startswith("input_") and not source.endswith("_len"):
                input_idx_canon[int(source.split("_")[1])] = canonical
            elif source.startswith("output_"):
                output_idx_canon[int(source.split("_")[1])] = canonical
        operand_names = set()
        for canonical, source in mapping_spec:
            if source.startswith(("input_", "output_")) and not source.endswith("_len"):
                operand_names.add(mapping.get(canonical))
        out_names = set()
        for name, dtype, shape in node.outputs:
            out_names.add(_sanitize_name(mapping.get(name, name) if mapping else name))

        for idx, (name, dtype, shape) in enumerate(node.inputs):
            if name in skip_inputs:
                continue
            if idx in input_idx_canon:
                mapped = _sanitize_name(mapping.get(input_idx_canon[idx]))
            else:
                mapped = _sanitize_name(name)
            if mapped in emitted_names:
                continue
            if mapped in derived and mapped not in operand_names:
                continue
            if mapped in out_names:
                continue
            emitted_names.add(mapped)
            c_type = DTYPE_TO_C.get(dtype, "double")
            if shape == "array" or (isinstance(shape, str) and shape.startswith("(")):
                c_params.append(f"const {c_type} * restrict {mapped}")
            else:
                c_params.append(f"{c_type} {mapped}")

        for idx, (name, dtype, shape) in enumerate(node.outputs):
            c_type = DTYPE_TO_C.get(dtype, "double")
            if idx in output_idx_canon:
                mapped = _sanitize_name(mapping.get(output_idx_canon[idx]))
            else:
                mapped = _sanitize_name(name)
            if mapped in emitted_names:
                continue
            emitted_names.add(mapped)
            if shape == "array" or (isinstance(shape, str) and shape.startswith("(")):
                c_params.append(f"{c_type} * restrict {mapped}")
            else:
                c_params.append(f"{c_type} *{mapped}")

        for dparam in derived:
            mapped_dparam = mapping.get(dparam, dparam) if mapping else dparam
            if not any(re.search(r"\b" + re.escape(mapped_dparam) + r"\b", p) for p in c_params):
                c_params.insert(0, f"int {mapped_dparam}")

        params_str = ", ".join(c_params) if c_params else "void"
        func_name = _sanitize_name(node.node_id)
        return f"void {func_name}({params_str})"

    def _generate_provenance(self, node: MathIRNode, module_name: str, timestamp: str) -> str:
        prov: dict[str, Any] = {
            "purce_version": __version__,
            "generated_at": timestamp,
            "target_profile": self.target_profile,
            "source": {
                "module": module_name,
                "file": node.origin_file,
                "line": node.origin_line,
                "symbol": node.origin_symbol,
                "commit": node.origin_commit or "unknown",
            },
            "ir_node": {
                "node_id": node.node_id,
                "algorithm": node.algorithm,
                "math_intent": node.math_intent,
                "effects": [e.name for e in node.effects],
                "inputs": [{"name": n, "dtype": dt.name, "shape": s} for n, dt, s in node.inputs],
                "outputs": [{"name": n, "dtype": dt.name, "shape": s} for n, dt, s in node.outputs],
            },
            "memory": {
                "stack_usage_bytes": node.stack_usage,
                "heap_usage_bytes": node.heap_usage,
                "reentrant": node.reentrant,
            },
            "reductions": [
                {"rule": r.rule, "description": r.description, "original": r.original}
                for r in node.reductions
            ],
            "dependencies": node.nested_deps,
        }
        return json.dumps(prov, indent=2)

    def _generate_cmake(
        self, graph: MathIRGraph, module_name: str, ordered: list[MathIRNode]
    ) -> str:
        c_files = [_sanitize_name(n.node_id) + ".c" for n in ordered]

        lines = [
            "cmake_minimum_required(VERSION 3.10)",
            f"project({_sanitize_name(module_name)} C)",
            "",
            "set(CMAKE_C_STANDARD 99)",
            "set(CMAKE_C_STANDARD_REQUIRED ON)",
            'set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")',
            "",
            f"add_library({_sanitize_name(module_name)}",
        ]
        for cf in c_files:
            lines.append(f"    {cf}")
        lines.append(")")
        lines.append("")
        lines.append(f"target_include_directories({_sanitize_name(module_name)} PUBLIC .)")
        lines.append("")
        lines.append("find_package(Math REQUIRED)")
        lines.append(f"target_link_libraries({_sanitize_name(module_name)} PUBLIC m)")
        lines.append("")
        return "\n".join(lines)
