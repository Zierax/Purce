"""Real-world test: JAX-style operations.

These functions mimic common JAX/XLA computation patterns:
  - jax.numpy operations (same API as NumPy)
  - jax.lax control flow
  - jax.grad automatic differentiation patterns
  - jax.vmap vectorized mapping

The purce parser should extract math kernels from these patterns.
"""

import numpy as np


def jax_softmax(x: np.ndarray) -> np.ndarray:
    """jax.nn.softmax: stable softmax with max subtraction."""
    shifted = np.subtract(x, np.max(x))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x))


def jax_layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Layer normalization: normalize along last axis."""
    mean = np.mean(x)
    variance = np.mean(np.power(np.subtract(x, mean), 2.0))
    normalized = np.divide(np.subtract(x, mean), np.sqrt(np.add(variance, 1e-5)))
    return np.add(np.multiply(gamma, normalized), beta)


def jax_silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return np.multiply(x, np.divide(1.0, np.add(1.0, np.exp(np.negative(x)))))


def jax_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = np.divide(np.matmul(Q, np.transpose(K)), np.sqrt(d_k))
    weights = np.divide(np.exp(scores), np.sum(np.exp(scores)))
    return np.matmul(weights, V)


def jax_cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy loss from logits."""
    log_probs = np.log(np.add(np.exp(logits), 1e-7))
    return np.negative(np.sum(np.multiply(labels, log_probs)))


def jax_mse_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Mean squared error loss."""
    diff = np.subtract(predictions, targets)
    return np.mean(np.power(diff, 2.0))


def jax_l1_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """L1 (absolute error) loss."""
    return np.mean(np.abs(np.subtract(predictions, targets)))


def jax_huber_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Huber loss (smooth L1)."""
    diff = np.subtract(predictions, targets)
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, 1.0)
    linear = np.subtract(abs_diff, quadratic)
    return np.mean(np.add(np.multiply(0.5, np.power(quadratic, 2.0)), linear))


def jax_gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation: x * Phi(x)."""
    return np.multiply(
        x,
        np.multiply(
            0.5,
            np.add(
                1.0,
                np.tanh(np.multiply(
                    np.sqrt(np.divide(2.0, np.pi)),
                    np.add(x, np.multiply(0.044715, np.power(x, 3.0)))
                ))
            )
        )
    )


def jax_mish(x: np.ndarray) -> np.ndarray:
    """Mish activation: x * tanh(softplus(x))."""
    softplus = np.log(np.add(1.0, np.exp(x)))
    return np.multiply(x, np.tanh(softplus))


def jax_rotary_embedding(x: np.ndarray, theta: float = 10000.0) -> np.ndarray:
    """Rotary position embedding (RoPE)."""
    d = x.shape[-1]
    freqs = np.power(theta, np.divide(-np.arange(0, d, 2).astype(np.float64), d))
    t = np.arange(x.shape[0]).reshape(-1, 1)
    angles = np.multiply(t, freqs)
    cos_cache = np.cos(angles)
    sin_cache = np.sin(angles)
    out = np.zeros_like(x)
    out[..., 0::2] = np.subtract(
        np.multiply(x[..., 0::2], cos_cache),
        np.multiply(x[..., 1::2], sin_cache)
    )
    out[..., 1::2] = np.add(
        np.multiply(x[..., 0::2], sin_cache),
        np.multiply(x[..., 1::2], cos_cache)
    )
    return out


def jax_group_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                   num_groups: int = 32) -> np.ndarray:
    """Group normalization."""
    shape = x.shape
    channels = shape[-1]
    group_size = channels // num_groups
    out = np.zeros_like(x)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = x[..., start:end]
        mean = np.mean(group)
        var = np.mean(np.power(np.subtract(group, mean), 2.0))
        normed = np.divide(np.subtract(group, mean), np.sqrt(np.add(var, 1e-5)))
        out[..., start:end] = np.add(np.multiply(gamma[start:end], normed), beta[start:end])
    return out


def jax_spectral_norm(W: np.ndarray, u: np.ndarray, n_iters: int = 1) -> np.ndarray:
    """Spectral normalization of weight matrix."""
    for _ in range(n_iters):
        v = np.divide(np.matmul(np.transpose(W), u), np.add(np.linalg.norm(np.matmul(np.transpose(W), u)), 1e-7))
        u = np.divide(np.matmul(W, v), np.add(np.linalg.norm(np.matmul(W, v)), 1e-7))
    sigma = np.dot(np.matmul(np.transpose(W), u), v)
    return np.divide(W, np.add(sigma, 1e-7))


def jax_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between two vectors."""
    dot = np.sum(np.multiply(a, b))
    norm_a = np.sqrt(np.sum(np.power(a, 2.0)))
    norm_b = np.sqrt(np.sum(np.power(b, 2.0)))
    return np.divide(dot, np.add(np.multiply(norm_a, norm_b), 1e-7))


def jax_batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched matrix multiplication."""
    return np.matmul(A, B)


def jax_rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """RMS normalization."""
    rms = np.sqrt(np.add(np.mean(np.power(x, 2.0)), 1e-7))
    return np.multiply(np.divide(x, rms), weight)
