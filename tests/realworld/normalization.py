"""Normalization layers — RMSNorm, GroupNorm, InstanceNorm, SpectralNorm."""

import numpy as np


def rmsnorm_forward(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x / sqrt(mean(x^2) + eps) * gamma."""
    rms = np.sqrt(np.add(np.mean(np.power(x, 2), axis=-1, keepdims=True), eps))
    return np.multiply(np.divide(x, rms), gamma)


def groupnorm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, eps: float = 1e-5
) -> np.ndarray:
    """Group normalization: normalize within groups of channels."""
    batch, channels, h, w = x.shape
    channels_per_group = channels // num_groups
    output = np.zeros_like(x)
    for g in range(num_groups):
        start = g * channels_per_group
        end = start + channels_per_group
        group = x[:, start:end, :, :]
        mean = np.mean(group, axis=(1, 2, 3), keepdims=True)
        var = np.var(group, axis=(1, 2, 3), keepdims=True)
        norm = np.divide(np.subtract(group, mean), np.sqrt(np.add(var, eps)))
        output[:, start:end, :, :] = np.add(np.multiply(norm, gamma[start:end]), beta[start:end])
    return output


def instancenorm_forward(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Instance normalization: normalize each sample independently."""
    mean = np.mean(x, axis=(2, 3), keepdims=True)
    var = np.var(x, axis=(2, 3), keepdims=True)
    norm = np.divide(np.subtract(x, mean), np.sqrt(np.add(var, eps)))
    return np.add(np.multiply(norm, gamma), beta)


def spectral_norm_forward(W: np.ndarray, u: np.ndarray, n_power_iterations: int = 1) -> tuple:
    """Spectral normalization: W / sigma(W) using power iteration."""
    for _ in range(n_power_iterations):
        v = np.matmul(W.T, u)
        v = np.divide(v, np.add(np.sqrt(np.sum(np.power(v, 2))), 1e-7))
        u = np.matmul(W, v)
        u = np.divide(u, np.add(np.sqrt(np.sum(np.power(u, 2))), 1e-7))
    sigma = np.dot(u, np.matmul(W, v))
    W_sn = np.divide(W, np.add(sigma, 1e-7))
    return W_sn, u


def weight_standardization(W: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Weight standardization: (W - mean) / sqrt(var + eps)."""
    mean = np.mean(W, axis=(0, 1), keepdims=True)
    var = np.var(W, axis=(0, 1), keepdims=True)
    return np.divide(np.subtract(W, mean), np.sqrt(np.add(var, eps)))
