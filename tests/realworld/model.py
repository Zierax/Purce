"""Transformer model — complete forward pass tying all modules together."""

import numpy as np
from tests.realworld.layers import (
    dense_forward, multi_head_attention, layer_norm_forward,
    residual_block, dropout_forward, batch_norm_forward
)
from tests.realworld.activations import gelu, swish
from tests.realworld.losses import softmax_cross_entropy, label_smoothing_cross_entropy
from tests.realworld.optimizers import adam_update, adamw_update


def transformer_encoder_block(x: np.ndarray, W_qkv: np.ndarray, W_out: np.ndarray,
                              W_ff1: np.ndarray, b_ff1: np.ndarray,
                              W_ff2: np.ndarray, b_ff2: np.ndarray,
                              ln1_gamma: np.ndarray, ln1_beta: np.ndarray,
                              ln2_gamma: np.ndarray, ln2_beta: np.ndarray,
                              n_heads: int) -> np.ndarray:
    """Single transformer encoder block: attention + FFN with residual connections."""
    batch, seq_len, d_model = x.shape

    QKV = dense_forward(x, W_qkv, np.zeros((W_qkv.shape[1],)))
    d_head = d_model // n_heads
    Q = QKV[:, :, :d_model]
    K = QKV[:, :, d_model:2 * d_model]
    V = QKV[:, :, 2 * d_model:]

    attn_out, _ = multi_head_attention(Q, K, V, n_heads)
    attn_out = dense_forward(attn_out, W_out, np.zeros((W_out.shape[1],)))

    h1 = layer_norm_forward(np.add(x, attn_out), ln1_gamma, ln1_beta)

    ffn = dense_forward(h1, W_ff1, b_ff1)
    ffn = gelu(ffn)
    ffn = dense_forward(ffn, W_ff2, b_ff2)

    out = layer_norm_forward(np.add(h1, ffn), ln2_gamma, ln2_beta)
    return out


def transformer_decoder_block(x: np.ndarray, memory: np.ndarray,
                              W_qkv_self: np.ndarray, W_out_self: np.ndarray,
                              W_qkv_cross: np.ndarray, W_out_cross: np.ndarray,
                              W_ff1: np.ndarray, b_ff1: np.ndarray,
                              W_ff2: np.ndarray, b_ff2: np.ndarray,
                              ln1_gamma: np.ndarray, ln1_beta: np.ndarray,
                              ln2_gamma: np.ndarray, ln2_beta: np.ndarray,
                              ln3_gamma: np.ndarray, ln3_beta: np.ndarray,
                              n_heads: int) -> np.ndarray:
    """Transformer decoder block with self-attention + cross-attention."""
    batch, seq_len, d_model = x.shape
    d_head = d_model // n_heads

    QKV = dense_forward(x, W_qkv_self, np.zeros((W_qkv_self.shape[1],)))
    Q = QKV[:, :, :d_model]
    K = QKV[:, :, d_model:2 * d_model]
    V = QKV[:, :, 2 * d_model:]
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    K_masked = np.where(causal_mask[np.newaxis, :, :], K, 0.0)
    V_masked = np.where(causal_mask[np.newaxis, :, :], V, 0.0)
    attn1, _ = multi_head_attention(Q, K_masked, V_masked, n_heads)
    attn1 = dense_forward(attn1, W_out_self, np.zeros((W_out_self.shape[1],)))
    h1 = layer_norm_forward(np.add(x, attn1), ln1_gamma, ln1_beta)

    Q2 = dense_forward(h1, W_qkv_cross[:, :d_model], np.zeros((d_model,)))
    cross_attn, _ = multi_head_attention(Q2, memory, memory, n_heads)
    cross_attn = dense_forward(cross_attn, W_out_cross, np.zeros((W_out_cross.shape[1],)))
    h2 = layer_norm_forward(np.add(h1, cross_attn), ln2_gamma, ln2_beta)

    ffn = dense_forward(h2, W_ff1, b_ff1)
    ffn = gelu(ffn)
    ffn = dense_forward(ffn, W_ff2, b_ff2)
    return layer_norm_forward(np.add(h2, ffn), ln3_gamma, ln3_beta)


def mlp_mixer_block(x: np.ndarray, W_token: np.ndarray, W_channel: np.ndarray) -> np.ndarray:
    """MLP-Mixer: alternate token-mixing and channel-mixing."""
    batch, seq_len, channels = x.shape

    token_mixed = np.transpose(x, (0, 2, 1))
    token_mixed = dense_forward(token_mixed, W_token, np.zeros((seq_len,)))
    token_mixed = np.transpose(token_mixed, (0, 2, 1))
    h = layer_norm_forward(np.add(x, token_mixed), np.ones(channels), np.zeros(channels))

    channel_mixed = dense_forward(h, W_channel, np.zeros((W_channel.shape[1],)))
    channel_mixed = gelu(channel_mixed)
    return layer_norm_forward(np.add(h, channel_mixed), np.ones(channels), np.zeros(channels))


def mixture_of_experts(x: np.ndarray, expert_weights: list, gating_weights: np.ndarray,
                       top_k: int = 2) -> np.ndarray:
    """Mixture of Experts: route inputs to top-k experts."""
    batch, seq_len, d_model = x.shape
    n_experts = len(expert_weights)

    gate_logits = dense_forward(x, gating_weights, np.zeros((n_experts,)))
    gate_probs = _softmax(gate_logits)
    top_indices = np.argsort(gate_probs, axis=-1)[:, :, -top_k:]
    top_vals = np.take_along_axis(gate_probs, top_indices, axis=-1)

    output = np.zeros_like(x)
    flat_x = x.reshape(-1, d_model)
    flat_output = np.zeros((batch * seq_len, d_model))

    for e_idx in range(n_experts):
        mask = (top_indices == e_idx).any(axis=-1).flatten()
        if mask.any():
            expert_input = flat_x[mask]
            expert_out = dense_forward(expert_input, expert_weights[e_idx][0],
                                       expert_weights[e_idx][1])
            expert_gate = gate_probs.reshape(-1, n_experts)[:, e_idx:e_idx+1][mask]
            flat_output[mask] += np.multiply(expert_out, expert_gate)

    return flat_output.reshape(batch, seq_len, d_model)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x, axis=-1, keepdims=True))


def training_step(x: np.ndarray, targets: np.ndarray,
                  params: dict, lr: float = 0.001, t: int = 1) -> tuple:
    """Full training step: forward + backward + optimizer update."""
    logits = dense_forward(x, params['W_out'], params['b_out'])
    loss = softmax_cross_entropy(logits, targets)
    grad = np.subtract(_softmax(logits), targets)

    dW_out = dense_forward(x.T, grad, np.zeros((grad.shape[1],)))
    db_out = np.sum(grad, axis=0)

    params['W_out'], params['m_W'], params['v_W'] = adam_update(
        params['W_out'], dW_out, params['m_W'], params['v_W'], t, lr)
    params['b_out'], params['m_b'], params['v_b'] = adam_update(
        params['b_out'], db_out, params['m_b'], params['v_b'], t, lr)

    return params, float(np.mean(loss))
