"""Image processing — convolution, filtering, edge detection, denoising.

Harder patterns: separable kernels, nested convolution loops, border
handling, and multi-channel pooling reductions.
"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D correlation convolution with zero padding (valid border)."""
    h, w = image.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            acc = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    acc = np.add(acc, np.multiply(image[i + ki, j + kj], kernel[ki, kj]))
            result[i, j] = acc
    return result


def separable_conv2d(image: np.ndarray, row_k: np.ndarray,
                     col_k: np.ndarray) -> np.ndarray:
    """Convolve with separable kernel: first rows, then columns."""
    h, w = image.shape
    mid = np.zeros((h, w))
    k_len = row_k.shape[0]
    half = k_len // 2
    for i in range(h):
        for j in range(w):
            acc = 0.0
            for k in range(k_len):
                jj = j + k - half
                if jj >= 0 and jj < w:
                    acc = np.add(acc, np.multiply(image[i, jj], row_k[k]))
            mid[i, j] = acc
    result = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            acc = 0.0
            for k in range(k_len):
                ii = i + k - half
                if ii >= 0 and ii < h:
                    acc = np.add(acc, np.multiply(mid[ii, j], col_k[k]))
            result[i, j] = acc
    return result


def sobel_gradient(image: np.ndarray) -> tuple:
    """Sobel edge detection: horizontal and vertical gradients."""
    Gx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    Gy = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    dx = convolve2d(image, Gx)
    dy = convolve2d(image, Gy)
    return dx, dy


def gaussian_blur(image: np.ndarray, sigma: float, radius: int = 2) -> np.ndarray:
    """Gaussian blur with a precomputed normalized kernel."""
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    total = 0.0
    for i in range(size):
        for j in range(size):
            dx = i - radius
            dy = j - radius
            val = np.exp(np.negative(np.divide(np.add(dx * dx, dy * dy),
                                               np.multiply(2.0, np.multiply(sigma, sigma)))))
            kernel[i, j] = val
            total = np.add(total, val)
    kernel = np.divide(kernel, total)
    return convolve2d(image, kernel)


def non_max_suppression(mag: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Non-maximum suppression along gradient direction."""
    h, w = mag.shape
    result = np.zeros((h, w))
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = np.mod(np.degrees(angle[i, j]), 180.0)
            if (a >= 0.0 and a < 22.5) or (a >= 157.5 and a < 180.0):
                left = mag[i, j - 1]
                right = mag[i, j + 1]
            elif a >= 22.5 and a < 67.5:
                left = mag[i - 1, j + 1]
                right = mag[i + 1, j - 1]
            elif a >= 67.5 and a < 112.5:
                left = mag[i - 1, j]
                right = mag[i + 1, j]
            else:
                left = mag[i - 1, j - 1]
                right = mag[i + 1, j + 1]
            if mag[i, j] >= left and mag[i, j] >= right:
                result[i, j] = mag[i, j]
    return result


def max_pool(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """Max pooling with stride (no overlap when pool_size == stride)."""
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    result = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            best = -1e30
            for pi in range(pool_size):
                for pj in range(pool_size):
                    v = image[i * stride + pi, j * stride + pj]
                    best = np.maximum(best, v)
            result[i, j] = best
    return result


def average_pool(image: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """Average pooling over non-overlapping windows."""
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    result = np.zeros((out_h, out_w))
    area = pool_size * pool_size
    for i in range(out_h):
        for j in range(out_w):
            acc = 0.0
            for pi in range(pool_size):
                for pj in range(pool_size):
                    acc = np.add(acc, image[i * stride + pi, j * stride + pj])
            result[i, j] = np.divide(acc, area)
    return result


def box_blur(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fast box blur via cumulative row and column sums."""
    h, w = image.shape
    result = np.zeros((h, w))
    for i in range(h):
        acc = 0.0
        for j in range(w + radius):
            j_in = j
            j_out = j - radius
            if j_in < w:
                acc = np.add(acc, image[i, j_in])
            if j_out >= 0 and j_out < w:
                if i < h:
                    result[i, j_out] = acc
    return result
