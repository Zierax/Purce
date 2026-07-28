"""Neural network layers — dense, conv2d, multi-head attention, GNN message passing."""

import numpy as np


def dense_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fully connected layer: y = x @ W + b"""
    return np.add(np.matmul(x, W), b)


def dense_backward(d_out: np.ndarray, x: np.ndarray, W: np.ndarray) -> tuple:
    """Backward pass for dense layer."""
    dW = np.matmul(x.T, d_out)
    db = np.sum(d_out, axis=0)
    d_x = np.matmul(d_out, W.T)
    return dW, db, d_x


def conv2d_forward(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray,
                   stride: int = 1, padding: int = 0) -> np.ndarray:
    """2D convolution: output[i,j] = sum over (k,l) of x[i+k,j+l] * kernel[k,l] + bias."""
    batch, in_h, in_w, in_c = x.shape
    k_h, k_w, in_c2, out_c = kernel.shape
    assert in_c == in_c2

    out_h = (in_h + 2 * padding - k_h) // stride + 1
    out_w = (in_w + 2 * padding - k_w) // stride + 1

    if padding > 0:
        x_padded = np.zeros((batch, in_h + 2 * padding, in_w + 2 * padding, in_c))
        x_padded[:, padding:padding + in_h, padding:padding + in_w, :] = x
        x = x_padded

    output = np.zeros((batch, out_h, out_w, out_c))
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride
            w_start = j * stride
            patch = x[:, h_start:h_start + k_h, w_start:w_start + k_w, :]
            for co in range(out_c):
                val = np.sum(np.multiply(patch, kernel[:, :, :, co]))
                output[:, i, j, co] = val + bias[co]
    return output


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         n_heads: int) -> tuple:
    """Multi-head scaled dot-product attention.

    Returns (output, attention_weights).
    """
    batch, seq_len, d_model = Q.shape
    d_head = d_model // n_heads

    outputs = []
    all_weights = []

    for h in range(n_heads):
        q_h = Q[:, :, h * d_head:(h + 1) * d_head]
        k_h = K[:, :, h * d_head:(h + 1) * d_head]
        v_h = V[:, :, h * d_head:(h + 1) * d_head]

        scores = np.matmul(q_h, k_h.transpose(0, 2, 1))
        scale = np.sqrt(np.float64(d_head))
        scores = np.divide(scores, scale)

        weights = _softmax_3d(scores)
        attn_out = np.matmul(weights, v_h)
        outputs.append(attn_out)
        all_weights.append(weights)

    concatenated = np.concatenate(outputs, axis=-1)
    return concatenated, all_weights


def _softmax_3d(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x, axis=-1, keepdims=True))


def graph_message_passing(node_features: np.ndarray,
                          edge_index: np.ndarray,
                          edge_weights: np.ndarray,
                          W_message: np.ndarray,
                          W_update: np.ndarray) -> np.ndarray:
    """Graph Neural Network message passing layer.

    node_features: (num_nodes, feature_dim)
    edge_index: (num_edges, 2) — [src, dst] pairs
    edge_weights: (num_edges,)
    """
    num_nodes, feat_dim = node_features.shape
    messages = np.zeros((num_nodes, feat_dim))

    for e in range(edge_index.shape[0]):
        src = edge_index[e, 0]
        dst = edge_index[e, 1]
        msg = np.multiply(node_features[src], edge_weights[e])
        messages[dst] = np.add(messages[dst], msg)

    messages = np.matmul(messages, W_message)
    updated = np.add(node_features, messages)
    updated = np.matmul(updated, W_update)
    return updated


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                       eps: float = 1e-5) -> np.ndarray:
    """Layer normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    normalized = np.divide(np.subtract(x, mean), np.sqrt(np.add(variance, eps)))
    return np.add(np.multiply(normalized, gamma), beta)


def batch_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                       running_mean: np.ndarray, running_var: np.ndarray,
                       eps: float = 1e-5, momentum: float = 0.1) -> np.ndarray:
    """Batch normalization across batch dimension."""
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    running_mean_new = np.add(np.multiply(running_mean, 1 - momentum),
                              np.multiply(mean, momentum))
    running_var_new = np.add(np.multiply(running_var, 1 - momentum),
                             np.multiply(var, momentum))
    normalized = np.divide(np.subtract(x, mean), np.sqrt(np.add(var, eps)))
    output = np.add(np.multiply(normalized, gamma), beta)
    return output


def dropout_forward(x: np.ndarray, rate: float = 0.1) -> tuple:
    """Dropout: zero out random elements, scale survivors."""
    mask = np.greater(np.random.random(x.shape), rate).astype(np.float64)
    scale = 1.0 / (1.0 - rate)
    output = np.multiply(np.multiply(x, mask), scale)
    return output, mask


def residual_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Residual connection: output = x + W2 @ relu(W1 @ x + b1) + b2."""
    h = dense_forward(x, W1, b1)
    h = np.maximum(h, 0.0)
    h = dense_forward(h, W2, b2)
    return np.add(x, h)
