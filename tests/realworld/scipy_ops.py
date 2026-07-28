"""Real-world test: SciPy-style operations.

These functions mimic common SciPy computation patterns:
  - scipy.signal (convolution, filtering, spectral analysis)
  - scipy.linalg (decompositions, solvers)
  - scipy.optimize (gradient descent patterns)
  - scipy.special (mathematical functions)

The purce parser should extract math kernels from these patterns.
"""

import numpy as np


def scipy_convolve1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1D convolution (full mode)."""
    n = len(signal) + len(kernel) - 1
    result = np.zeros(n)
    for i in range(len(signal)):
        for j in range(len(kernel)):
            result[i + j] += signal[i] * kernel[j]
    return result


def scipy_correlate1d(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1D cross-correlation."""
    n = len(signal) + len(kernel) - 1
    result = np.zeros(n)
    for i in range(len(signal)):
        for j in range(len(kernel)):
            result[i + j] += signal[i] * kernel[j]
    return result


def scipy_decimate(signal: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample by factor."""
    n = len(signal) // factor
    result = np.zeros(n)
    for i in range(n):
        result[i] = signal[i * factor]
    return result


def scipy_resample(signal: np.ndarray, num: int) -> np.ndarray:
    """Resample signal to num points using linear interpolation."""
    n = len(signal)
    result = np.zeros(num)
    for i in range(num):
        pos = i * (n - 1) / (num - 1) if num > 1 else 0
        idx = int(pos)
        frac = pos - idx
        if idx + 1 < n:
            result[i] = np.add(np.multiply(signal[idx], 1.0 - frac),
                               np.multiply(signal[idx + 1], frac))
        else:
            result[i] = signal[idx]
    return result


def scipy_savgol_filter(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing (simplified moving average)."""
    n = len(signal)
    result = np.zeros(n)
    half = window // 2
    for i in range(n):
        s = 0.0
        count = 0
        for j in range(max(0, i - half), min(n, i + half + 1)):
            s += signal[j]
            count += 1
        result[i] = s / count
    return result


def scipy_median_filter(signal: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter."""
    n = len(signal)
    result = np.zeros(n)
    half = kernel_size // 2
    for i in range(n):
        window = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            window.append(signal[j])
        window.sort()
        result[i] = window[len(window) // 2]
    return result


def scipy_lu_decompose(A: np.ndarray, n: int) -> tuple:
    """LU decomposition with partial pivoting."""
    L = np.eye(n)
    U = np.copy(A)
    P = np.eye(n)
    for col in range(n):
        max_row = col
        for row in range(col + 1, n):
            if abs(U[row, col]) > abs(U[max_row, col]):
                max_row = row
        if max_row != col:
            U[[col, max_row]] = U[[max_row, col]]
            P[[col, max_row]] = P[[max_row, col]]
            L[[col, max_row]] = L[[max_row, col]]
        for row in range(col + 1, n):
            factor = U[row, col] / U[col, col]
            L[row, col] = factor
            for j in range(col, n):
                U[row, j] -= factor * U[col, j]
    return L, U, P


def scipy_cholesky(A: np.ndarray, n: int) -> np.ndarray:
    """Cholesky decomposition: A = L * L^T."""
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                val = A[i, i] - s
                L[i, j] = np.sqrt(val) if val > 0 else 0.0
            else:
                denom = L[j, j]
                L[i, j] = (A[i, j] - s) / denom if abs(denom) > 1e-15 else 0.0
    return L


def scipy_qr(A: np.ndarray, n: int) -> tuple:
    """QR decomposition using Gram-Schmidt."""
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = np.subtract(v, np.multiply(R[i, j], Q[:, i]))
        R[j, j] = np.sqrt(np.sum(np.power(v, 2.0)))
        if R[j, j] > 1e-15:
            Q[:, j] = np.divide(v, R[j, j])
    return Q, R


def scipy_svd_power_iteration(A: np.ndarray, n: int, num_iters: int = 100) -> np.ndarray:
    """Singular value estimation via power iteration."""
    singular_values = np.zeros(n)
    B = np.copy(A)
    for _ in range(num_iters):
        for col in range(n):
            v = B[:, col].copy()
            norm = np.sqrt(np.sum(np.power(v, 2.0)))
            if norm > 1e-15:
                v = np.divide(v, norm)
            singular_values[col] = norm
    return singular_values


def scipy_gradient_descent(f_grad, x0: np.ndarray, lr: float = 0.01,
                           n_steps: int = 100) -> np.ndarray:
    """Simple gradient descent pattern."""
    x = np.copy(x0)
    for _ in range(n_steps):
        grad = f_grad(x)
        x = np.subtract(x, np.multiply(lr, grad))
    return x


def scipy_newton_solve(f, f_prime, x0: float, n_iters: int = 10) -> float:
    """Newton's method root finding."""
    x = x0
    for _ in range(n_iters):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-15:
            break
        x = np.subtract(x, np.divide(fx, fpx))
    return x


def scipy_bisect(f, a: float, b: float, n_iters: int = 50) -> float:
    """Bisection method root finding."""
    for _ in range(n_iters):
        mid = np.add(a, np.multiply(0.5, np.subtract(b, a)))
        if np.multiply(np.subtract(f(mid), 0.0), np.subtract(f(a), 0.0)) < 0:
            b = mid
        else:
            a = mid
    return np.add(a, np.multiply(0.5, np.subtract(b, a)))


def scipy_stft(signal: np.ndarray, window_size: int = 256,
               hop_size: int = 128) -> np.ndarray:
    """Short-time Fourier Transform (magnitude only)."""
    num_frames = (len(signal) - window_size) // hop_size + 1
    spec = np.zeros((num_frames, window_size // 2 + 1))
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start:start + window_size]
        fft_result = np.zeros(window_size, dtype=np.complex128)
        for k in range(window_size):
            s = 0.0 + 0.0j
            for j in range(window_size):
                angle = -2.0 * np.pi * k * j / window_size
                s += frame[j] * np.exp(np.multiply(1j, angle))
            fft_result[k] = s
        spec[i, :] = np.abs(fft_result[:window_size // 2 + 1])
    return spec


def scipy_welch_psd(signal: np.ndarray, nperseg: int = 256) -> np.ndarray:
    """Welch power spectral density estimation."""
    n_segments = max(1, (len(signal) - nperseg) // nperseg + 1)
    psd = np.zeros(nperseg // 2 + 1)
    for i in range(n_segments):
        start = i * nperseg
        segment = signal[start:start + nperseg]
        fft_mag = np.abs(np.fft.fft(segment))
        psd += np.power(fft_mag[:nperseg // 2 + 1], 2.0)
    psd = np.divide(psd, n_segments)
    return psd
