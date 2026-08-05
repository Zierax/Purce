"""Kalman filter — state-space estimation with predict/update cycles.

Harder patterns: matrix inverses, transposes, outer products, and
multi-branch conditional updates in a tight loop.
"""

import numpy as np


def kalman_predict(x: np.ndarray, P: np.ndarray, F: np.ndarray,
                   Q: np.ndarray) -> tuple:
    """Predict step: x' = Fx, P' = FPF^T + Q."""
    x_new = np.matmul(F, x)
    P_new = np.add(np.matmul(F, np.matmul(P, np.transpose(F))), Q)
    return x_new, P_new


def kalman_update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
                  H: np.ndarray, R: np.ndarray) -> tuple:
    """Update step: Kalman gain, state correction, covariance update."""
    S = np.add(np.matmul(H, np.matmul(P, np.transpose(H))), R)
    S_inv = np.linalg.inv(S)
    K = np.matmul(np.matmul(P, np.transpose(H)), S_inv)
    innovation = np.subtract(z, np.matmul(H, x))
    x_new = np.add(x, np.matmul(K, innovation))
    I = np.eye(P.shape[0])
    KH = np.matmul(K, H)
    P_new = np.matmul(np.subtract(I, KH), P)
    return x_new, P_new


def kalman_filter_loop(x: np.ndarray, P: np.ndarray, F: np.ndarray,
                       H: np.ndarray, Q: np.ndarray, R: np.ndarray,
                       measurements: np.ndarray, steps: int) -> tuple:
    """Full Kalman filtering over a measurement sequence."""
    n = x.shape[0]
    m = measurements.shape[0]
    x_hist = np.zeros((steps, n))
    P_hist = np.zeros((steps, n, n))
    for t in range(steps):
        x, P = kalman_predict(x, P, F, Q)
        if t < m:
            z = measurements[t]
            x, P = kalman_update(x, P, z, H, R)
        x_hist[t] = x
        P_hist[t] = P
    return x_hist, P_hist


def extended_kalman_predict(x: np.ndarray, P: np.ndarray,
                            F: np.ndarray, Q: np.ndarray) -> tuple:
    """EKF predict with a nonlinearity folded into the state."""
    x_new = np.tanh(np.matmul(F, x))
    F_jac = np.multiply(np.subtract(1.0, np.power(x_new, 2)), F)
    P_new = np.add(np.matmul(F_jac, np.matmul(P, np.transpose(F_jac))), Q)
    return x_new, P_new


def information_filter_update(x: np.ndarray, P: np.ndarray, z: np.ndarray,
                              H: np.ndarray, R: np.ndarray) -> tuple:
    """Information-form update using the inverse covariance."""
    P_inv = np.linalg.inv(P)
    R_inv = np.linalg.inv(R)
    Y = np.add(P_inv, np.matmul(np.transpose(H), np.matmul(R_inv, H)))
    y = np.add(np.matmul(P_inv, x),
               np.matmul(np.transpose(H), np.matmul(R_inv, z)))
    x_new = np.matmul(np.linalg.inv(Y), y)
    return x_new, Y


def unscented_transform(X: np.ndarray, w_m: np.ndarray,
                        w_c: np.ndarray, R: np.ndarray) -> tuple:
    """Unscented transform: sigma-point mean and covariance."""
    n = X.shape[1]
    x_mean = np.matmul(w_m, X)
    diff = np.subtract(X, x_mean)
    P = np.add(np.matmul(np.multiply(w_c[:, np.newaxis], diff),
                         np.transpose(diff)), R)
    return x_mean, P


def smoother_rauch(x_fwd: np.ndarray, P_fwd: np.ndarray,
                   F: np.ndarray, Q: np.ndarray) -> tuple:
    """Rauch-Tung-Striebel smoother (backward pass)."""
    T, n = x_fwd.shape
    x_sm = np.copy(x_fwd)
    P_sm = np.copy(P_fwd)
    for t in range(T - 2, -1, -1):
        P_pred = np.add(np.matmul(F, np.matmul(P_fwd[t], np.transpose(F))), Q)
        P_pred_inv = np.linalg.inv(P_pred)
        G = np.matmul(np.matmul(P_fwd[t], np.transpose(F)), P_pred_inv)
        residual = np.subtract(x_sm[t + 1], np.matmul(F, x_fwd[t]))
        x_sm[t] = np.add(x_fwd[t], np.matmul(G, residual))
        P_sm[t] = np.add(P_fwd[t],
                         np.matmul(G, np.matmul(np.subtract(P_sm[t + 1], P_pred),
                                                np.transpose(G))))
    return x_sm, P_sm
