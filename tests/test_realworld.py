"""Real-world integration tests: parse real ML/scientific code through purce pipeline.

Tests that purce can parse, build IR, slice, and generate C99 from
real-world code patterns found in JAX, PyTorch, SciPy, and NumPy.
"""

from __future__ import annotations

import os
import re


from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer


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


def _has_c_function(content: str) -> bool:
    return bool(re.search(r"\b(void|int|double|float)\s+\w+\s*\(", content))


# ── JAX ops ──────────────────────────────────────────────────────────────────


class TestJaxOps:
    def test_jax_softmax_parses(self) -> None:
        src = _load_realworld_file("jax_ops.py")
        gen, graph, _, _ = _run_pipeline(src, "jax_ops")
        assert len(graph.nodes) >= 1

    def test_jax_softmax_generates_c(self) -> None:
        src = _load_realworld_file("jax_ops.py")
        gen, _, _, _ = _run_pipeline(src, "jax_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

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
        assert len(graph.nodes) >= 1

    def test_jax_attention_parses(self) -> None:
        src = """\
import numpy as np
def attention(Q, K, V):
    scores = np.divide(np.matmul(Q, np.transpose(K)), np.sqrt(Q.shape[-1]))
    weights = np.divide(np.exp(scores), np.sum(np.exp(scores)))
    return np.matmul(weights, V)
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_attn")
        assert len(graph.nodes) >= 1

    def test_jax_cross_entropy_parses(self) -> None:
        src = """\
import numpy as np
def cross_entropy(logits, labels):
    log_probs = np.log(np.add(np.exp(logits), 1e-7))
    return np.negative(np.sum(np.multiply(labels, log_probs)))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_ce")
        assert len(graph.nodes) >= 1

    def test_jax_mse_loss_parses(self) -> None:
        src = """\
import numpy as np
def mse_loss(pred, target):
    diff = np.subtract(pred, target)
    return np.mean(np.power(diff, 2.0))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_mse")
        assert len(graph.nodes) >= 1

    def test_jax_gelu_parses(self) -> None:
        src = """\
import numpy as np
def gelu(x):
    return np.multiply(x, np.multiply(0.5, np.add(1.0, np.tanh(np.multiply(np.sqrt(np.divide(2.0, np.pi)), np.add(x, np.multiply(0.044715, np.power(x, 3.0))))))))
"""
        gen, graph, _, _ = _run_pipeline(src, "jax_gelu")
        assert len(graph.nodes) >= 1

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
        assert len(graph.nodes) >= 1

    def test_pytorch_ops_generates_c(self) -> None:
        src = _load_realworld_file("pytorch_ops.py")
        gen, _, _, _ = _run_pipeline(src, "pytorch_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_pytorch_relu_parses(self) -> None:
        src = """\
import numpy as np
def relu(x):
    return np.maximum(x, 0.0)
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_relu")
        assert len(graph.nodes) == 1

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
        assert len(graph.nodes) >= 1

    def test_pytorch_linear_parses(self) -> None:
        src = """\
import numpy as np
def linear(x, weight, bias):
    return np.add(np.matmul(x, np.transpose(weight)), bias)
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_linear")
        assert len(graph.nodes) >= 1

    def test_pytorch_huber_loss_parses(self) -> None:
        src = """\
import numpy as np
def huber(pred, target, delta=1.0):
    diff = np.subtract(pred, target)
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear_part = np.subtract(abs_diff, quadratic)
    return np.mean(np.add(np.multiply(0.5, np.power(quadratic, 2.0)), linear_part))
"""
        gen, graph, _, _ = _run_pipeline(src, "pt_huber")
        assert len(graph.nodes) >= 1

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
        assert len(graph.nodes) >= 1

    def test_scipy_ops_generates_c(self) -> None:
        src = _load_realworld_file("scipy_ops.py")
        gen, _, _, _ = _run_pipeline(src, "scipy_ops")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

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


# ── Activations ──────────────────────────────────────────────────────────────


class TestActivations:
    def test_activations_parses(self) -> None:
        src = _load_realworld_file("activations.py")
        gen, graph, _, _ = _run_pipeline(src, "activations")
        assert len(graph.nodes) >= 1

    def test_activations_generates_c(self) -> None:
        src = _load_realworld_file("activations.py")
        gen, _, _, _ = _run_pipeline(src, "activations")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_activations_no_malloc(self) -> None:
        src = _load_realworld_file("activations.py")
        gen, _, _, _ = _run_pipeline(src, "activations")
        for f in gen.files:
            if f.file_type == "c":
                assert "malloc" not in f.content
                assert "calloc" not in f.content

    def test_activations_has_provenance(self) -> None:
        src = _load_realworld_file("activations.py")
        gen, _, _, _ = _run_pipeline(src, "activations")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Losses ───────────────────────────────────────────────────────────────────


class TestLosses:
    def test_losses_parses(self) -> None:
        src = _load_realworld_file("losses.py")
        gen, graph, _, _ = _run_pipeline(src, "losses")
        assert len(graph.nodes) >= 1

    def test_losses_generates_c(self) -> None:
        src = _load_realworld_file("losses.py")
        gen, _, _, _ = _run_pipeline(src, "losses")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Normalization ────────────────────────────────────────────────────────────


class TestNormalization:
    def test_normalization_parses(self) -> None:
        src = _load_realworld_file("normalization.py")
        gen, graph, _, _ = _run_pipeline(src, "normalization")
        assert len(graph.nodes) >= 1

    def test_normalization_generates_c(self) -> None:
        src = _load_realworld_file("normalization.py")
        gen, _, _, _ = _run_pipeline(src, "normalization")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Layers ───────────────────────────────────────────────────────────────────


class TestLayers:
    def test_layers_parses(self) -> None:
        src = _load_realworld_file("layers.py")
        gen, graph, _, _ = _run_pipeline(src, "layers")
        assert len(graph.nodes) >= 1

    def test_layers_generates_c(self) -> None:
        src = _load_realworld_file("layers.py")
        gen, _, _, _ = _run_pipeline(src, "layers")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Optimizers ───────────────────────────────────────────────────────────────


class TestOptimizers:
    def test_optimizers_parses(self) -> None:
        src = _load_realworld_file("optimizers.py")
        gen, graph, _, _ = _run_pipeline(src, "optimizers")
        assert len(graph.nodes) >= 1

    def test_optimizers_generates_c(self) -> None:
        src = _load_realworld_file("optimizers.py")
        gen, _, _, _ = _run_pipeline(src, "optimizers")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Linear algebra ───────────────────────────────────────────────────────────


class TestLinearAlgebra:
    def test_linear_algebra_parses(self) -> None:
        src = _load_realworld_file("linear_algebra.py")
        gen, graph, _, _ = _run_pipeline(src, "linear_algebra")
        assert len(graph.nodes) >= 1

    def test_linear_algebra_generates_c(self) -> None:
        src = _load_realworld_file("linear_algebra.py")
        gen, _, _, _ = _run_pipeline(src, "linear_algebra")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Attention ────────────────────────────────────────────────────────────────


class TestAttention:
    def test_attention_parses(self) -> None:
        src = _load_realworld_file("attention.py")
        gen, graph, _, _ = _run_pipeline(src, "attention")
        assert len(graph.nodes) >= 1

    def test_attention_generates_c(self) -> None:
        src = _load_realworld_file("attention.py")
        gen, _, _, _ = _run_pipeline(src, "attention")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Convolution ──────────────────────────────────────────────────────────────


class TestConvolution:
    def test_convolution_parses(self) -> None:
        src = _load_realworld_file("convolution.py")
        gen, graph, _, _ = _run_pipeline(src, "convolution")
        assert len(graph.nodes) >= 1

    def test_convolution_generates_c(self) -> None:
        src = _load_realworld_file("convolution.py")
        gen, _, _, _ = _run_pipeline(src, "convolution")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Signal processing ────────────────────────────────────────────────────────


class TestSignalProcessing:
    def test_signal_processing_parses(self) -> None:
        src = _load_realworld_file("signal_processing.py")
        gen, graph, _, _ = _run_pipeline(src, "signal_processing")
        assert len(graph.nodes) >= 1

    def test_signal_processing_generates_c(self) -> None:
        src = _load_realworld_file("signal_processing.py")
        gen, _, _, _ = _run_pipeline(src, "signal_processing")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"


# ── Extra patterns ───────────────────────────────────────────────────────────


class TestExtraPatterns:
    def test_extra_patterns_source_strings_parse(self) -> None:
        import importlib.util

        path = os.path.join(os.path.dirname(__file__), "realworld", "extra_patterns.py")
        spec = importlib.util.spec_from_file_location("extra_patterns", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        patterns = [
            ("TRANSFORMER_SELF_ATTENTION", mod.TRANSFORMER_SELF_ATTENTION),
            ("RNN_CELL", mod.RNN_CELL),
            ("CONV_1D", mod.CONV_1D),
        ]
        for name, src in patterns:
            gen, graph, _, _ = _run_pipeline(src, f"extra_{name}")
            assert len(graph.nodes) >= 1, f"Pattern {name} produced 0 nodes"
            c_files = [f for f in gen.files if f.file_type == "c"]
            assert len(c_files) >= 1, f"Pattern {name} produced no C files"

    def test_extra_patterns_have_provenance(self) -> None:
        import importlib.util

        path = os.path.join(os.path.dirname(__file__), "realworld", "extra_patterns.py")
        spec = importlib.util.spec_from_file_location("extra_patterns", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        src = mod.TRANSFORMER_SELF_ATTENTION
        gen, _, _, _ = _run_pipeline(src, "extra_attn")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Transformer patterns ─────────────────────────────────────────────────────


class TestTransformers:
    def test_transformers_parses(self) -> None:
        src = _load_realworld_file("transformers.py")
        gen, graph, _, _ = _run_pipeline(src, "transformers")
        assert len(graph.nodes) >= 1

    def test_transformers_generates_c(self) -> None:
        src = _load_realworld_file("transformers.py")
        gen, _, _, _ = _run_pipeline(src, "transformers")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_transformers_have_provenance(self) -> None:
        src = _load_realworld_file("transformers.py")
        gen, _, _, _ = _run_pipeline(src, "transformers")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)

    def test_transformers_no_malloc(self) -> None:
        src = _load_realworld_file("transformers.py")
        gen, _, _, _ = _run_pipeline(src, "transformers")
        for f in gen.files:
            if f.file_type == "c":
                assert "malloc" not in f.content
                assert "calloc" not in f.content


# ── Generative model patterns ────────────────────────────────────────────────


class TestGenerative:
    def test_generative_parses(self) -> None:
        src = _load_realworld_file("generative.py")
        gen, graph, _, _ = _run_pipeline(src, "generative")
        assert len(graph.nodes) >= 1

    def test_generative_generates_c(self) -> None:
        src = _load_realworld_file("generative.py")
        gen, _, _, _ = _run_pipeline(src, "generative")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_generative_have_provenance(self) -> None:
        src = _load_realworld_file("generative.py")
        gen, _, _, _ = _run_pipeline(src, "generative")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Reinforcement learning patterns ──────────────────────────────────────────


class TestReinforcementLearning:
    def test_rl_parses(self) -> None:
        src = _load_realworld_file("reinforcement_learning.py")
        gen, graph, _, _ = _run_pipeline(src, "rl")
        assert len(graph.nodes) >= 1

    def test_rl_generates_c(self) -> None:
        src = _load_realworld_file("reinforcement_learning.py")
        gen, _, _, _ = _run_pipeline(src, "rl")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_rl_have_provenance(self) -> None:
        src = _load_realworld_file("reinforcement_learning.py")
        gen, _, _, _ = _run_pipeline(src, "rl")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Graph neural network patterns ────────────────────────────────────────────


class TestGraphNeuralNetworks:
    def test_gnn_parses(self) -> None:
        src = _load_realworld_file("graph_neural_networks.py")
        gen, graph, _, _ = _run_pipeline(src, "gnn")
        assert len(graph.nodes) >= 1

    def test_gnn_generates_c(self) -> None:
        src = _load_realworld_file("graph_neural_networks.py")
        gen, _, _, _ = _run_pipeline(src, "gnn")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_gnn_have_provenance(self) -> None:
        src = _load_realworld_file("graph_neural_networks.py")
        gen, _, _, _ = _run_pipeline(src, "gnn")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Time series patterns ─────────────────────────────────────────────────────


class TestTimeSeries:
    def test_timeseries_parses(self) -> None:
        src = _load_realworld_file("time_series.py")
        gen, graph, _, _ = _run_pipeline(src, "ts")
        assert len(graph.nodes) >= 1

    def test_timeseries_generates_c(self) -> None:
        src = _load_realworld_file("time_series.py")
        gen, _, _, _ = _run_pipeline(src, "ts")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_timeseries_have_provenance(self) -> None:
        src = _load_realworld_file("time_series.py")
        gen, _, _, _ = _run_pipeline(src, "ts")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Computer vision patterns ─────────────────────────────────────────────────


class TestComputerVision:
    def test_cv_parses(self) -> None:
        src = _load_realworld_file("computer_vision.py")
        gen, graph, _, _ = _run_pipeline(src, "cv")
        assert len(graph.nodes) >= 1

    def test_cv_generates_c(self) -> None:
        src = _load_realworld_file("computer_vision.py")
        gen, _, _, _ = _run_pipeline(src, "cv")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_cv_have_provenance(self) -> None:
        src = _load_realworld_file("computer_vision.py")
        gen, _, _, _ = _run_pipeline(src, "cv")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Recommendation patterns ──────────────────────────────────────────────────


class TestRecommendation:
    def test_recsys_parses(self) -> None:
        src = _load_realworld_file("recommendation.py")
        gen, graph, _, _ = _run_pipeline(src, "recsys")
        assert len(graph.nodes) >= 1

    def test_recsys_generates_c(self) -> None:
        src = _load_realworld_file("recommendation.py")
        gen, _, _, _ = _run_pipeline(src, "recsys")
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(c_files) >= 1
        for cf in c_files:
            assert _has_c_function(cf.content), f"No function signature in {cf.path}"

    def test_recsys_have_provenance(self) -> None:
        src = _load_realworld_file("recommendation.py")
        gen, _, _, _ = _run_pipeline(src, "recsys")
        prov_files = [f for f in gen.files if f.file_type == "prov"]
        c_files = [f for f in gen.files if f.file_type == "c"]
        assert len(prov_files) == len(c_files)


# ── Stress tests ─────────────────────────────────────────────────────────────


class TestStress:
    def test_100_random_sources_parse_without_crash(self) -> None:
        import random

        random.seed(42)
        ops = [
            "np.add",
            "np.subtract",
            "np.multiply",
            "np.divide",
            "np.sum",
            "np.mean",
            "np.sqrt",
            "np.exp",
            "np.log",
            "np.sin",
            "np.cos",
            "np.abs",
            "np.tan",
        ]
        for i in range(100):
            op = random.choice(ops)
            n_inputs = 2 if op in ("np.add", "np.subtract", "np.multiply", "np.divide") else 1
            params = ", ".join(f"x{j}" for j in range(n_inputs))
            args = ", ".join(f"x{j}" for j in range(n_inputs))
            src = f"import numpy as np\ndef f({params}):\n    return {op}({args})"
            builder = MathIRBuilder(origin_file="stress")
            graph = builder.build_from_source(src, module="stress")
            assert isinstance(graph.nodes, dict)
            assert len(graph.nodes) == 1, (
                f"Iteration {i}: expected 1 node for {op}, got {len(graph.nodes)}"
            )

    def test_large_multi_op_source_parses(self) -> None:
        lines = ["import numpy as np"]
        for i in range(50):
            lines.append(f"def func_{i}(a, b):")
            lines.append("    return np.add(a, b)")
        src = "\n".join(lines)
        builder = MathIRBuilder(origin_file="large")
        graph = builder.build_from_source(src, module="large")
        assert len(graph.nodes) == 50
