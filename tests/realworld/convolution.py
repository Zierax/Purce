"""Advanced convolution operations — depthwise, separable, transposed, dilated."""

import numpy as np


def depthwise_conv2d(x: np.ndarray, kernel: np.ndarray,
                     stride: int = 1, padding: int = 0) -> np.ndarray:
    """Depthwise convolution: each channel has its own filter."""
    batch, in_h, in_w, channels = x.shape
    k_h, k_w, _, _ = kernel.shape
    out_h = (in_h + 2 * padding - k_h) // stride + 1
    out_w = (in_w + 2 * padding - k_w) // stride + 1

    if padding > 0:
        x_padded = np.zeros((batch, in_h + 2 * padding, in_w + 2 * padding, channels))
        x_padded[:, padding:padding+in_h, padding:padding+in_w, :] = x
        x = x_padded

    output = np.zeros((batch, out_h, out_w, channels))
    for i in range(out_h):
        for j in range(out_w):
            h_s = i * stride
            w_s = j * stride
            patch = x[:, h_s:h_s+k_h, w_s:w_s+k_w, :]
            for c in range(channels):
                output[:, i, j, c] = np.sum(np.multiply(patch[:, :, :, c], kernel[:, :, 0, c]))
    return output


def separable_conv2d(x: np.ndarray, depthwise_kernel: np.ndarray,
                     pointwise_kernel: np.ndarray) -> np.ndarray:
    """Separable conv: depthwise conv followed by 1x1 pointwise conv."""
    dw_out = depthwise_conv2d(x, depthwise_kernel)
    batch, h, w, channels = dw_out.shape
    pw_channels = pointwise_kernel.shape[-1]
    output = np.zeros((batch, h, w, pw_channels))
    flat = dw_out.reshape(batch * h * w, channels)
    result = np.matmul(flat, pointwise_kernel.reshape(channels, pw_channels))
    output = result.reshape(batch, h, w, pw_channels)
    return output


def transposed_conv2d(x: np.ndarray, kernel: np.ndarray,
                      stride: int = 2, padding: int = 0) -> np.ndarray:
    """Transposed (fractionally strided) convolution."""
    batch, in_h, in_w, in_channels = x.shape
    k_h, k_w, out_channels, _ = kernel.shape
    out_h = (in_h - 1) * stride - 2 * padding + k_h
    out_w = (in_w - 1) * stride - 2 * padding + k_w
    output = np.zeros((batch, out_h, out_w, out_channels))

    for i in range(in_h):
        for j in range(in_w):
            for c_in in range(in_channels):
                h_s = i * stride - padding
                w_s = j * stride - padding
                for kh in range(k_h):
                    for kw in range(k_w):
                        oh = h_s + kh
                        ow = w_s + kw
                        if 0 <= oh < out_h and 0 <= ow < out_w:
                            for c_out in range(out_channels):
                                output[:, oh, ow, c_out] += np.multiply(
                                    x[:, i, j, c_in],
                                    kernel[kh, kw, c_out, c_in])
    return output


def dilated_conv2d(x: np.ndarray, kernel: np.ndarray, dilation: int = 2,
                   padding: int = 0) -> np.ndarray:
    """Dilated (atrous) convolution with holes."""
    batch, in_h, in_w, in_channels = x.shape
    k_h, k_w, _, out_channels = kernel.shape
    effective_h = k_h + (k_h - 1) * (dilation - 1)
    effective_w = k_w + (k_w - 1) * (dilation - 1)
    out_h = (in_h + 2 * padding - effective_h) // 1 + 1
    out_w = (in_w + 2 * padding - effective_w) // 1 + 1

    if padding > 0:
        x_padded = np.zeros((batch, in_h + 2*padding, in_w + 2*padding, in_channels))
        x_padded[:, padding:padding+in_h, padding:padding+in_w, :] = x
        x = x_padded

    output = np.zeros((batch, out_h, out_w, out_channels))
    for i in range(out_h):
        for j in range(out_w):
            for co in range(out_channels):
                val = 0.0
                for kh in range(k_h):
                    for kw in range(k_w):
                        ih = i + kh * dilation
                        iw = j + kw * dilation
                        if ih < x.shape[1] and iw < x.shape[2]:
                            for ci in range(in_channels):
                                val += x[:, ih, iw, ci] * kernel[kh, kw, ci, co]
                output[:, i, j, co] = val
    return output


def grouped_conv2d(x: np.ndarray, kernel: np.ndarray, groups: int = 2) -> np.ndarray:
    """Grouped convolution: channels split into groups, convolved independently."""
    batch, in_h, in_w, in_channels = x.shape
    k_h, k_w, _, out_channels = kernel.shape
    channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    out_h = in_h - k_h + 1
    out_w = in_w - k_w + 1
    output = np.zeros((batch, out_h, out_w, out_channels))

    for g in range(groups):
        ci_start = g * channels_per_group
        co_start = g * out_channels_per_group
        for i in range(out_h):
            for j in range(out_w):
                for co in range(out_channels_per_group):
                    val = 0.0
                    for kh in range(k_h):
                        for kw in range(k_w):
                            for ci in range(channels_per_group):
                                val += x[:, i+kh, j+kw, ci_start+ci] * kernel[kh, kw, ci, co]
                    output[:, i, j, co_start+co] = val
    return output
