"""Transformer architecture patterns: attention, positional encoding, FFN, layer norm."""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.transpose(K)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(weights, V)


def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[-1]
    d_k = d_model // n_heads
    head_dim = Q.shape[0] * n_heads
    Q_reshaped = np.reshape(Q, (n_heads, -1, d_k))
    K_reshaped = np.reshape(K, (n_heads, -1, d_k))
    V_reshaped = np.reshape(V, (n_heads, -1, d_k))
    scores = np.matmul(Q_reshaped, np.transpose(K_reshaped)) / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    attended = np.matmul(weights, V_reshaped)
    return np.reshape(attended, (head_dim, -1))


def sinusoidal_positional_encoding(max_len, d_model):
    positions = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def rotary_embedding(positions, d_model):
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    angles = positions.reshape(-1, 1) * div_term
    cos_part = np.cos(angles)
    sin_part = np.sin(angles)
    return np.concatenate([cos_part, sin_part], axis=-1)


def feedforward_network(x, d_model, d_ff):
    w1 = np.ones((d_model, d_ff))
    w2 = np.ones((d_ff, d_model))
    hidden = np.maximum(0, np.matmul(x, w1))
    return np.matmul(hidden, w2)


def transformer_encoder_block(x, n_heads, d_ff):
    d_model = x.shape[-1]
    Q = np.ones((d_model, d_model))
    K = np.ones((d_model, d_model))
    V = np.ones((d_model, d_model))
    attn_out = scaled_dot_product_attention(
        np.matmul(x, Q), np.matmul(x, K), np.matmul(x, V)
    )
    x = np.add(x, attn_out)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = np.divide(np.subtract(x, mean), np.sqrt(np.add(var, 1e-5)))
    ffn_out = feedforward_network(x_norm, d_model, d_ff)
    return np.add(x_norm, ffn_out)


def gptcausal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def cross_entropy_attention_loss(logits, labels):
    log_probs = np.log(np.add(np.exp(logits), 1e-7))
    return np.negative(np.sum(np.multiply(labels, log_probs))) / logits.shape[0]
