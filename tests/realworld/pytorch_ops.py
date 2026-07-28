"""Real-world test: PyTorch-style operations.

These functions mimic common PyTorch computation patterns:
  - torch.nn.functional operations
  - torch.optim optimizers
  - torch.nn modules (linear, conv, norm)
  - torch.autograd gradient patterns

The purce parser should extract math kernels from these patterns.
"""

import numpy as np


def torch_relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation: max(0, x)."""
    return np.maximum(x, 0.0)


def torch_leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    """LeakyReLU: x if x > 0 else negative_slope * x."""
    return np.where(np.greater(x, 0.0), x, np.multiply(negative_slope, x))


def torch_elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU: x if x > 0 else alpha * (exp(x) - 1)."""
    return np.where(np.greater(x, 0.0), x, np.multiply(alpha, np.subtract(np.exp(x), 1.0)))


def torch_selu(x: np.ndarray) -> np.ndarray:
    """SELU: lambda * (x if x > 0 else alpha * (exp(x) - 1))."""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return np.multiply(scale, np.where(np.greater(x, 0.0), x, np.multiply(alpha, np.subtract(np.exp(x), 1.0))))


def torch_celu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """CELU: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))."""
    return np.maximum(x, 0.0) + np.minimum(0.0, np.multiply(alpha, np.subtract(np.exp(np.divide(x, alpha)), 1.0)))


def torch_gelu_approx(x: np.ndarray) -> np.ndarray:
    """GELU approximation (tanh)."""
    return np.multiply(
        0.5,
        np.multiply(
            np.add(x, np.multiply(0.044715, np.power(x, 3.0))),
            np.tanh(np.multiply(0.7978845608, np.add(x, np.multiply(0.044715, np.power(x, 3.0)))))
        )
    )


def torch_softmax_2d(x: np.ndarray) -> np.ndarray:
    """Softmax along last axis."""
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x, axis=-1, keepdims=True))


def torch_log_softmax(x: np.ndarray) -> np.ndarray:
    """Log-softmax: log(softmax(x))."""
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return np.subtract(shifted, np.log(sum_exp))


def torch_nll_loss(log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Negative log-likelihood loss."""
    batch_size = log_probs.shape[0]
    losses = np.zeros(batch_size)
    for i in range(batch_size):
        losses[i] = np.negative(log_probs[i, int(targets[i])])
    return np.mean(losses)


def torch_adam_step(params: np.ndarray, grads: np.ndarray,
                    m: np.ndarray, v: np.ndarray,
                    t: int, lr: float = 0.001,
                    beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> tuple:
    """Adam optimizer single step."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1.0 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1.0 - beta2, np.power(grads, 2.0)))
    m_hat = np.divide(m_new, 1.0 - np.power(beta1, t))
    v_hat = np.divide(v_new, 1.0 - np.power(beta2, t))
    params_new = np.subtract(params, np.multiply(lr, np.divide(m_hat, np.add(np.sqrt(v_hat), eps))))
    return params_new, m_new, v_new


def torch_linear_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Linear layer: x @ W^T + b."""
    return np.add(np.matmul(x, np.transpose(weight)), bias)


def torch_group_norm_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                             num_groups: int = 32, eps: float = 1e-5) -> np.ndarray:
    """Group normalization."""
    N, C, H, W = x.shape
    x_flat = x.reshape(N, num_groups, C // num_groups, H, W)
    mean = np.mean(x_flat, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_flat, axis=(2, 3, 4), keepdims=True)
    x_norm = np.divide(np.subtract(x_flat, mean), np.sqrt(np.add(var, eps)))
    x_norm = x_norm.reshape(N, C, H, W)
    return np.add(np.multiply(weight.reshape(1, C, 1, 1), x_norm), bias.reshape(1, C, 1, 1))


def torch_l1_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L1 (Manhattan) distance."""
    return np.sum(np.abs(np.subtract(a, b)))


def torch_cosine_embedding_loss(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine embedding loss."""
    cos_sim = np.divide(
        np.sum(np.multiply(x1, x2)),
        np.add(np.multiply(np.sqrt(np.sum(np.power(x1, 2.0))),
                            np.sqrt(np.sum(np.power(x2, 2.0)))), 1e-7)
    )
    return np.mean(np.maximum(np.subtract(0.0, cos_sim), 0.0))


def torch_huber_loss(pred: np.ndarray, target: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Smooth L1 (Huber) loss."""
    diff = np.subtract(pred, target)
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = np.subtract(abs_diff, quadratic)
    return np.mean(np.add(np.multiply(0.5, np.power(quadratic, 2.0)), linear))


def torch_instance_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Instance normalization."""
    mean = np.mean(x, axis=(1, 2), keepdims=True)
    var = np.var(x, axis=(1, 2), keepdims=True)
    return np.divide(np.subtract(x, mean), np.sqrt(np.add(var, eps)))
