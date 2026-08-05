"""Time series — ARIMA residuals, exponential smoothing, spectral analysis.

Harder patterns: rolling windows with strided updates, recursive
filter state, trend-seasonal decomposition, and ACF/PACF kernels.
"""

import numpy as np


def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially weighted moving average (recursive filter)."""
    n = x.shape[0]
    out = np.zeros(n)
    acc = x[0]
    out[0] = acc
    for t in range(1, n):
        acc = np.add(np.multiply(alpha, x[t]), np.multiply(np.subtract(1.0, alpha), acc))
        out[t] = acc
    return out


def ewma_centered(x: np.ndarray, span: int) -> np.ndarray:
    """Centered EWMA used for seasonal adjustment."""
    alpha = np.divide(2.0, np.add(span, 1.0))
    fwd = ewma(x, alpha)
    rev = ewma(np.flip(x), alpha)
    return np.add(np.multiply(0.5, fwd), np.multiply(0.5, np.flip(rev)))


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with edge truncation."""
    n = x.shape[0]
    out = np.zeros(n)
    for t in range(n):
        lo = max(0, t - window + 1)
        hi = t + 1
        out[t] = np.divide(np.sum(x[lo:hi]), hi - lo)
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with Welford-style accumulation."""
    n = x.shape[0]
    out = np.zeros(n)
    for t in range(1, n):
        lo = max(0, t - window + 1)
        hi = t + 1
        block = x[lo:hi]
        mean = np.mean(block)
        var = np.mean(np.power(np.subtract(block, mean), 2.0))
        out[t] = np.sqrt(var)
    out[0] = 0.0
    return out


def zscore_anomaly(x: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Z-score anomaly detection with rolling statistics."""
    mu = rolling_mean(x, window)
    sigma = np.add(rolling_std(x, window), 1e-7)
    z = np.divide(np.subtract(x, mu), sigma)
    flags = np.abs(z) > threshold
    return np.where(flags, 1.0, 0.0)


def seasonal_decompose(x: np.ndarray, period: int) -> tuple:
    """Classical seasonal decomposition: trend, seasonal, residual."""
    n = x.shape[0]
    trend = np.zeros(n)
    for t in range(n):
        lo = max(0, t - period)
        hi = min(n, t + period + 1)
        trend[t] = np.mean(x[lo:hi])
    detrended = np.subtract(x, trend)
    seasonal = np.zeros(n)
    for p in range(period):
        idx = list(range(p, n, period))
        seasonal[idx] = np.mean(detrended[idx])
    seasonal = np.subtract(seasonal, np.mean(seasonal))
    residual = np.subtract(np.subtract(x, trend), seasonal)
    return trend, seasonal, residual


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation function (ACF) for lags 0..max_lag."""
    n = x.shape[0]
    mu = np.mean(x)
    centered = np.subtract(x, mu)
    var = np.add(np.sum(np.power(centered, 2.0)), 1e-7)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        num = np.sum(np.multiply(centered[:n - lag], centered[lag:]))
        acf[lag] = np.divide(num, var)
    return acf


def partial_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Partial autocorrelation via Levinson-Durbin recursion."""
    acf = autocorrelation(x, max_lag)
    pacf = np.zeros(max_lag + 1)
    pacf[0] = 1.0
    phi = np.zeros((max_lag + 1, max_lag + 1))
    for i in range(1, max_lag + 1):
        num = acf[i]
        for j in range(1, i):
            num = np.subtract(num, np.multiply(phi[i - 1, j], acf[i - j]))
        denom = np.subtract(1.0, np.sum(np.multiply(phi[i - 1, 1:i], acf[1:i])))
        phi[i, i] = np.divide(num, np.add(denom, 1e-7))
        for j in range(1, i):
            phi[i, j] = np.subtract(phi[i - 1, j],
                                    np.multiply(phi[i, i], phi[i - 1, i - j]))
        pacf[i] = phi[i, i]
    return pacf


def hmm_forward(logits: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Forward algorithm for a discrete HMM (log-space)."""
    T, n_states = logits.shape
    alpha = np.zeros((T, n_states))
    alpha[0] = np.add(logits[0], np.log(np.divide(1.0, n_states)))
    for t in range(1, T):
        for s in range(n_states):
            acc = -1e30
            for prev in range(n_states):
                cand = np.add(np.add(alpha[t - 1, prev],
                                     np.log(np.add(A[prev, s], 1e-7))),
                              logits[t, s])
                acc = np.maximum(acc, cand)
            alpha[t, s] = acc
    return alpha


def kalman_scalar_update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
                         H: np.ndarray, R: np.ndarray) -> tuple:
    """Scalar measurement Kalman update (vectorized over channels)."""
    K = np.divide(np.multiply(P, H), np.add(np.multiply(np.multiply(H, P), H), R))
    x_new = np.add(x, np.multiply(K, np.subtract(z, np.multiply(H, x))))
    P_new = np.multiply(np.subtract(1.0, np.multiply(K, H)), P)
    return x_new, P_new


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping distance (O(n*m)) with a banded matrix."""
    n = a.shape[0]
    m = b.shape[0]
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            d = np.power(np.subtract(a[i], b[j]), 2.0)
            if i == 0 and j == 0:
                cost[i, j] = d
            elif i == 0:
                cost[i, j] = np.add(d, cost[i, j - 1])
            elif j == 0:
                cost[i, j] = np.add(d, cost[i - 1, j])
            else:
                cost[i, j] = np.add(d, np.minimum(np.minimum(cost[i - 1, j],
                                                             cost[i, j - 1]),
                                                  cost[i - 1, j - 1]))
    return np.sqrt(cost[n - 1, m - 1])


def fourier_features(t: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Fourier basis features for seasonal regression."""
    n = t.shape[0]
    feat = np.zeros((n, 2 * n_harmonics))
    for k in range(n_harmonics):
        phase = np.multiply(np.multiply(2.0, np.pi), np.multiply(k + 1, t))
        feat[:, 2 * k] = np.cos(phase)
        feat[:, 2 * k + 1] = np.sin(phase)
    return feat
