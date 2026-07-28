"""Advanced attention mechanisms — flash attention, sparse attention, linear attention, ALiBi."""

import numpy as np


def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           block_size: int = 64) -> np.ndarray:
    """Flash Attention: memory-efficient attention via tiling."""
    batch, seq_len, d_head = Q.shape
    O = np.zeros_like(Q)
    l = np.zeros((batch, seq_len, 1))
    m = np.full((batch, seq_len, 1), -1e30)

    num_blocks = (seq_len + block_size - 1) // block_size
    scale = np.sqrt(np.float64(d_head))

    for j in range(num_blocks):
        j_start = j * block_size
        j_end = min(j_start + block_size, seq_len)
        Kj = K[:, j_start:j_end, :]
        Vj = V[:, j_start:j_end, :]

        S_block = np.matmul(Q, Kj.transpose(0, 2, 1))
        S_block = np.divide(S_block, scale)

        for i in range(num_blocks):
            i_start = i * block_size
            i_end = min(i_start + block_size, seq_len)
            Qi = Q[:, i_start:i_end, :]
            Si = S_block[:, i_start:i_end, :]

            m_new = np.maximum(m[:, i_start:i_end], np.max(Si, axis=-1, keepdims=True))
            P = np.exp(np.subtract(Si, m_new))
            l_new = np.add(l[:, i_start:i_end], np.sum(P, axis=-1, keepdims=True))
            scale_factor = np.exp(np.subtract(m[:, i_start:i_end], m_new))
            O[:, i_start:i_end] = np.add(
                np.multiply(O[:, i_start:i_end], scale_factor),
                np.matmul(P, Vj))
            m[:, i_start:i_end] = m_new
            l[:, i_start:i_end] = l_new

    O = np.divide(O, l)
    return O


def sparse_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Sparse attention with arbitrary boolean mask."""
    batch, seq_len, d_head = Q.shape
    scale = np.sqrt(np.float64(d_head))
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    scores = np.divide(scores, scale)
    scores = np.where(mask, scores, -1e30)
    weights = _softmax_3d(scores)
    return np.matmul(weights, V)


def linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     kernel_fn: str = "elu") -> np.ndarray:
    """Linear attention: O(n*d^2) instead of O(n^2*d)."""
    if kernel_fn == "elu":
        Q_k = np.add(_elu(Q), 1.0)
        K_k = np.add(_elu(K), 1.0)
    else:
        Q_k = np.maximum(Q, 0.0)
        K_k = np.maximum(K, 0.0)

    KV = np.matmul(K_k.transpose(0, 2, 1), V)
    Z = np.matmul(K_k.transpose(0, 2, 1), np.ones_like(V[:, :1, :]))
    numerator = np.matmul(Q_k, KV)
    denominator = np.add(np.matmul(Q_k, Z), 1e-6)
    return np.divide(numerator, denominator)


def alibi_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    num_heads: int, max_seq_len: int = 2048) -> np.ndarray:
    """ALiBi: Attention with Linear Biases — no learned positional embeddings."""
    batch, seq_len, d_model = Q.shape
    d_head = d_model // num_heads
    slopes = _get_alibi_slopes(num_heads)
    outputs = []

    for h in range(num_heads):
        q_h = Q[:, :, h * d_head:(h + 1) * d_head]
        k_h = K[:, :, h * d_head:(h + 1) * d_head]
        v_h = V[:, :, h * d_head:(h + 1) * d_head]
        scores = np.matmul(q_h, k_h.transpose(0, 2, 1))
        bias = np.multiply(slopes[h], _causal_bias(seq_len))
        scores = np.add(scores, bias)
        weights = _softmax_3d(scores)
        outputs.append(np.matmul(weights, v_h))

    return np.concatenate(outputs, axis=-1)


def _get_alibi_slopes(num_heads: int) -> list:
    ratio = 2.0 ** (-8.0 / num_heads)
    return [ratio ** (i + 1) for i in range(num_heads)]


def _causal_bias(seq_len: int) -> np.ndarray:
    bias = np.zeros((1, seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            bias[0, i, j] = -(j - i)
    return bias


def _softmax_3d(x: np.ndarray) -> np.ndarray:
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x, axis=-1, keepdims=True))


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(np.greater(x, 0), x, np.subtract(np.exp(x), 1.0))
