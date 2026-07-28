"""Real-world integration tests: parse real ML/scientific code through purce pipeline.

Tests that purce can parse, build IR, slice, and generate C99 from
real-world code patterns found in JAX, PyTorch, SciPy, and NumPy.
"""

from __future__ import annotations

import os

import pytest

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_pipeline(source: str, module_name: str = "test_mod") -> tuple:
    parser = PythonParser(target_profile="generic-c99")
    parsed = parser.parse_source(source, f"{module_name}.py")
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)
    generator = C99Generator(target_profile="generic-c99")
    gen_result = generator.generate(slice_result.graph, module_name=module_name)
    return gen_result, graph, slice_result, parsed


def _load_realworld_file(filename: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "realworld", filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── JAX ops ──────────────────────────────────────────────────────────────────


class TestJaxOps:
    def test_jax_softmax_parses(self) -> None:
        src = _load_realworld_file("jax_ops.py")
        gen, graph, _, _ = _run_pipeline(src, "jax_ops")
        assert len(graph.nodes) > 0

    def test_jax_softmax_generates_c(self) -> None:
        src = _load_realworld_file("jax_ops.py")
        gen, _, _, _ = _run_pipeline(src, "jax_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) > 0

    def test_jax_layer_norm_parses(self) -> None:
        src = """\
import numpy as np
def layer_norm(x, gamma, beta):
    mean = np.mean(x)
    variance = np.mean(np.power(np.subtract(x, mean), 2.0))
    normalized = np.divide(np.subtract(x, mean), np.sqrt(np.add(variance, 1e-5)))
    return np.add(np.multiply(gamma, normalized), beta)
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_ln")
        assert len(graph.nodes) == 1

    def test_jax_attention_parses(self) -> None:
        src = """\
import numpy as np
def attention(Q, K, V):
    scores = np.divide(np.matmul(Q, np.transpose(K)), np.sqrt(Q.shape[-1]))
    weights = np.divide(np.exp(scores), np.sum(np.exp(scores)))
    return np.matmul(weights, V)
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_attn")
        assert len(graph.nodes) == 1

    def test_jax_cross_entropy_parses(self) -> None:
        src = """\
import numpy as np
def cross_entropy(logits, labels):
    log_probs = np.log(np.add(np.exp(logits), 1e-7))
    return np.negative(np.sum(np.multiply(labels, log_probs)))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_ce")
        assert len(graph.nodes) == 1

    def test_jax_mse_loss_parses(self) -> None:
        src = """\
import numpy as np
def mse_loss(pred, target):
    diff = np.subtract(pred, target)
    return np.mean(np.power(diff, 2.0))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_mse")
        assert len(graph.nodes) == 1

    def test_jax_gelu_parses(self) -> None:
        src = """\
import numpy as np
def gelu(x):
    return np.multiply(x, np.multiply(0.5, np.add(1.0, np.tanh(np.multiply(np.sqrt(np.divide(2.0, np.pi)), np.add(x, np.multiply(0.044715, np.power(x, 3.0))))))))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_gelu")
        assert len(graph.nodes) == 1

    def test_jax_all_ops_have_provenance(self) -> None:
        src = _load_realworld_file("jax_ops.py")
        gen, _, _, _ = _run_pipeline(src, "jax_ops")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── PyTorch ops ──────────────────────────────────────────────────────────────


class TestPytorchOps:
    def test_pytorch_ops_parses(self) -> None:
        src = _load_realworld_file("pytorch_ops.py")
        gen, graph, _, _ = _run_pipeline(src, "pytorch_ops")
        assert len(graph.nodes) > 0

    def test_pytorch_ops_generates_c(self) -> None:
        src = _load_realworld_file("pytorch_ops.py")
        gen, _, _, _ = _run_pipeline(src, "pytorch_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) > 0

    def test_pytorch_relu_parses(self) -> None:
        src = """\
import numpy as np
def relu(x):
    return np.maximum(x, 0.0)
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_relu")
        assert len(graph.nodes) >= 1

    def test_pytorch_adam_parses(self) -> None:
        src = """\
import numpy as np
def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m_new = np.add(np.multiply(beta1, m), np.multiply(1.0 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1.0 - beta2, np.power(grads, 2.0)))
    m_hat = np.divide(m_new, 1.0 - np.power(beta1, t))
    v_hat = np.divide(v_new, 1.0 - np.power(beta2, t))
    params_new = np.subtract(params, np.multiply(lr, np.divide(m_hat, np.add(np.sqrt(v_hat), eps))))
    return params_new, m_new, v_new
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_adam")
        assert len(graph.nodes) == 1

    def test_pytorch_linear_parses(self) -> None:
        src = """\
import numpy as np
def linear(x, weight, bias):
    return np.add(np.matmul(x, np.transpose(weight)), bias)
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_linear")
        assert len(graph.nodes) == 1

    def test_pytorch_huber_loss_parses(self) -> None:
        src = """\
import numpy as np
def huber(pred, target, delta=1.0):
    diff = np.subtract(pred, target)
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = np.subtract(abs_diff, quadratic)
    return np.mean(np.add(np.multiply(0.5, np.power(quadratic, 2.0)), linear))
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_huber")
        assert len(graph.nodes) == 1

    def test_pytorch_no_malloc(self) -> None:
        src = _load_realworld_file("pytorch_ops.py")
        gen, _, _, _ = _run_pipeline(src, "pytorch_ops")
        for f in gen.files:
            if f.file_type == "c":
                assert "malloc" not in f.content
                assert "calloc" not in f.content


# ── SciPy ops ────────────────────────────────────────────────────────────────


class TestScipyOps:
    def test_scipy_ops_parses(self) -> None:
        src = _load_realworld_file("scipy_ops.py")
        gen, graph, _, _ = _run_pipeline(src, "scipy_ops")
        assert len(graph.nodes) > 0

    def test_scipy_ops_generates_c(self) -> None:
        src = _load_realworld_file("scipy_ops.py")
        gen, _, _, _ = _run_pipeline(src, "scipy_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) > 0

    def test_scipy_cholesky_parses(self) -> None:
        src = """\
import numpy as np
def cholesky(A, n):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                val = A[i, i] - s
                L[i, j] = np.sqrt(val) if val > 0 else 0.0
            else:
                denom = L[j, j]
                L[i, j] = (A[i, j] - s) / denom if abs(denom) > 1e-15 else 0.0
    return L
"""
        gen, graph, _, _ = _run_pipeline(src, "sp_cholesky")
        assert len(graph.nodes) >= 1

    def test_scipy_convolve_parses(self) -> None:
        src = """\
import numpy as np
def convolve(signal, kernel):
    n = len(signal) + len(kernel) - 1
    result = np.zeros(n)
    for i in range(len(signal)):
        for j in range(len(kernel)):
            result[i + j] += signal[i] * kernel[j]
    return result
"""
        gen, graph, _, _ = _run_pipeline(src, "sp_conv")
        assert len(graph.nodes) >= 1

    def test_scipy_all_ops_have_header(self) -> None:
        src = _load_realworld_file("scipy_ops.py")
        gen, _, _, _ = _run_pipeline(src, "scipy_ops")
        h_files = [f for f in gen.files if f.file_type == "h"]
        assert len(h_files) == 1
        assert "#ifndef" in h_files[0].content


# ── Stress tests ─────────────────────────────────────────────────────────────


class TestStress:
    def test_100_random_sources_parse_without_crash(self) -> None:
        import random
        random.seed(42)
        ops = ["np.add", "np.subtract", "np.multiply", "np.divide",
               "np.sum", "np.mean", "np.sqrt", "np.exp", "np.log",
               "np.sin", "np.cos", "np.abs", "np.tan"]
        for _ in range(100):
            op = random.choice(ops)
            n_inputs = 2 if op in ("np.add", "np.subtract", "np.multiply", "np.divide") else 1
            params = ", ".join(f"x{i}" for i in range(n_inputs))
            args = ", ".join(f"x{i}" for i in range(n_inputs))
            src = f"import numpy as np\ndef f({params}):\n    return {op}({args})"
            builder = MathIRBuilder(origin_file="stress")
            graph = builder.build_from_source(src, module="stress")
            assert isinstance(graph.nodes, dict)

    def test_large_multi_op_source_parses(self) -> None:
        lines = ["import numpy as np"]
        for i in range(50):
            lines.append(f"def func_{i}(a, b):")
            lines.append(f"    return np.add(a, b)")
        src = "\n".join(lines)
        builder = MathIRBuilder(origin_file="large")
        graph = builder.build_from_source(src, module="large")
        assert len(graph.nodes) == 50
