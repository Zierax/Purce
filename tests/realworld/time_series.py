"""Time series analysis patterns: forecasting, anomaly detection, seasonality."""

import numpy as np


def exponential_moving_average(data, alpha=0.3):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = np.add(np.multiply(alpha, data[t]), np.multiply(np.subtract(1.0, alpha), ema[t - 1]))
    return ema


def simple_moving_average(data, window):
    result = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result[i] = np.mean(data[start:i + 1])
    return result


def seasonal_decomposition(data, period):
    n = len(data)
    seasonal = np.zeros(n)
    trend = np.zeros(n)
    for i in range(n):
        window_start = max(0, i - period // 2)
        window_end = min(n, i + period // 2 + 1)
        trend[i] = np.mean(data[window_start:window_end])
    detrended = np.subtract(data, trend)
    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal[i] = np.mean(detrended[indices])
    for i in range(n):
        seasonal[i] = seasonal[i % period]
    residual = np.subtract(np.subtract(data, trend), seasonal)
    return trend, seasonal, residual


def autocorrelation(data, max_lag):
    mean = np.mean(data)
    var = np.var(data)
    result = np.zeros(max_lag)
    for lag in range(max_lag):
        cov = np.mean(np.multiply(np.subtract(data[:-lag] if lag > 0 else data, mean),
                                  np.subtract(data[lag:], mean)))
        result[lag] = np.divide(cov, np.add(var, 1e-8))
    return result


def detect_anomalies(data, threshold=3.0):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs(np.divide(np.subtract(data, mean), np.add(std, 1e-8)))
    return np.greater(z_scores, threshold)


def fourier_features(data, n_components):
    n = len(data)
    t = np.arange(n).reshape(-1, 1)
    features = np.zeros((n, n_components * 2))
    for k in range(n_components):
        freq = np.divide(np.multiply(2.0, np.pi, np.add(k, 1)), n)
        features[:, 2 * k] = np.sin(np.multiply(freq, t)).flatten()
        features[:, 2 * k + 1] = np.cos(np.multiply(freq, t)).flatten()
    return features


def dtw_distance(s1, s2):
    n = len(s1)
    m = len(s2)
    dtw = np.full((n + 1, m + 1), 1e9)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.abs(np.subtract(s1[i - 1], s2[j - 1]))
            dtw[i, j] = np.add(cost, np.minimum(
                np.minimum(dtw[i - 1, j], dtw[i, j - 1]),
                dtw[i - 1, j - 1]
            ))
    return dtw[n, m]


def kalman_predict(x_est, P_est, F, Q):
    x_pred = np.matmul(F, x_est)
    P_pred = np.add(np.matmul(np.matmul(F, P_est), np.transpose(F)), Q)
    return x_pred, P_pred


def kalman_update(x_pred, P_pred, z, H, R):
    y = np.subtract(z, np.matmul(H, x_pred))
    S = np.add(np.matmul(np.matmul(H, P_pred), np.transpose(H)), R)
    K = np.matmul(np.matmul(P_pred, np.transpose(H)), np.linalg.inv(S))
    x_est = np.add(x_pred, np.matmul(K, y))
    P_est = np.matmul(np.subtract(np.eye(P_pred.shape[0]), np.matmul(K, H)), P_pred)
    return x_est, P_est
