"""Optimization / auto-diff style — linear solvers, least squares, projection.

Harder patterns: conjugate gradient, Gram-Schmidt, Cholesky factor
updates, and nested loop solvers with pivot logic.
"""

import numpy as np


def conjugate_gradient(A: np.ndarray, b: np.ndarray,
                       max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Conjugate gradient solver for symmetric positive-definite A."""
    n = A.shape[0]
    x = np.zeros(n)
    r = b
    p = r
    rs_old = np.dot(r, r)
    for _ in range(max_iter):
        Ap = np.matmul(A, p)
        alpha = np.divide(rs_old, np.add(np.dot(p, Ap), 1e-12))
        x = np.add(x, np.multiply(alpha, p))
        r = np.subtract(r, np.multiply(alpha, Ap))
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break
        beta = np.divide(rs_new, np.add(rs_old, 1e-12))
        p = np.add(r, np.multiply(beta, p))
        rs_old = rs_new
    return x


def gauss_seidel(A: np.ndarray, b: np.ndarray,
                 max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Gauss-Seidel iteration for linear systems."""
    n = A.shape[0]
    x = np.zeros(n)
    for _ in range(max_iter):
        for i in range(n):
            acc = b[i]
            for j in range(n):
                if i != j:
                    acc = np.subtract(acc, np.multiply(A[i, j], x[j]))
            x[i] = np.divide(acc, np.add(A[i, i], 1e-9))
        residual = np.subtract(b, np.matmul(A, x))
        if np.sqrt(np.sum(np.power(residual, 2.0))) < tol:
            break
    return x


def modified_gram_schmidt(A: np.ndarray) -> tuple:
    """Modified Gram-Schmidt orthogonalization."""
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    V = np.copy(A)
    for j in range(m):
        norm = np.sqrt(np.sum(np.power(V[:, j], 2.0)))
        Q[:, j] = np.divide(V[:, j], np.add(norm, 1e-9))
        for i in range(m):
            R[j, i] = np.dot(Q[:, j], V[:, i])
        for i in range(m):
            V[:, i] = np.subtract(V[:, i], np.multiply(R[j, i], Q[:, j]))
    return Q, R


def householder_qr(A: np.ndarray) -> tuple:
    """QR via Householder reflections (loop over columns)."""
    n, m = A.shape
    R = np.copy(A)
    Q = np.eye(n)
    for k in range(min(n, m)):
        norm_x = np.sqrt(np.sum(np.power(R[k:, k], 2.0)))
        v = np.copy(R[k:, k])
        v[0] = np.subtract(v[0], np.multiply(np.sign(v[0]), norm_x))
        v_norm = np.sqrt(np.sum(np.power(v, 2.0)))
        if v_norm > 1e-12:
            v = np.divide(v, v_norm)
            for j in range(k, m):
                proj = np.multiply(2.0, np.dot(v, R[k:, j]))
                R[k:, j] = np.subtract(R[k:, j], np.multiply(proj, v))
            for j in range(n):
                proj = np.multiply(2.0, np.dot(v, Q[j, k:]))
                Q[j, k:] = np.subtract(Q[j, k:], np.multiply(proj, v))
    return np.transpose(Q), R


def cholesky_update(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank-1 Cholesky update: chol(L L^T + x x^T)."""
    n = L.shape[0]
    L_new = np.copy(L)
    x_new = np.copy(x)
    for k in range(n):
        r = np.sqrt(np.add(np.power(L_new[k, k], 2.0), np.power(x_new[k], 2.0)))
        c = np.divide(r, L_new[k, k])
        s = np.divide(x_new[k], L_new[k, k])
        L_new[k, k] = r
        for i in range(k + 1, n):
            L_new[i, k] = np.divide(np.add(L_new[i, k],
                                           np.multiply(x_new[i], s)), c)
            x_new[i] = np.subtract(np.multiply(x_new[i], c),
                                   np.multiply(L_new[i, k], s))
    return L_new


def total_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Total least squares: regularized inverse-power solution of [X y]."""
    n, d = X.shape
    M = np.zeros((d + 1, d + 1))
    for i in range(d + 1):
        for j in range(d + 1):
            acc = 0.0
            for k in range(n):
                if i < d and j < d:
                    acc = np.add(acc, np.multiply(X[k, i], X[k, j]))
                elif i < d and j == d:
                    acc = np.add(acc, np.multiply(X[k, i], y[k]))
                elif i == d and j < d:
                    acc = np.add(acc, np.multiply(y[k], X[k, j]))
                else:
                    acc = np.add(acc, np.multiply(y[k], y[k]))
            M[i, j] = acc
    reg = np.multiply(1e-6, np.eye(d + 1))
    return np.linalg.solve(np.add(M, reg), np.ones(d + 1))


def admm_prox(x: np.ndarray, rho: float, lam: float) -> np.ndarray:
    """Soft-threshold (shrinkage) proximal operator."""
    return np.sign(x) * np.maximum(np.subtract(np.abs(x), np.divide(lam, rho)), 0.0)


def admm_update(z: np.ndarray, u: np.ndarray, A: np.ndarray,
                b: np.ndarray, rho: float, lam: float) -> np.ndarray:
    """One ADMM iteration: x-step (linear solve) + z-step (prox)."""
    n = A.shape[0]
    AtA = np.matmul(np.transpose(A), A)
    reg = np.multiply(rho, np.eye(n))
    x = np.linalg.solve(np.add(AtA, reg),
                        np.add(np.matmul(np.transpose(A), b),
                               np.multiply(rho, np.subtract(z, u))))
    z_new = admm_prox(np.add(x, u), rho, lam)
    u = np.add(u, np.subtract(x, z_new))
    return z_new, u


def projected_gradient(x0: np.ndarray, grad0: np.ndarray,
                       lr: float, radius: float, n_steps: int) -> np.ndarray:
    """Projected gradient descent onto an L2 ball."""
    x = x0
    for _ in range(n_steps):
        x = np.subtract(x, np.multiply(lr, grad0))
        norm = np.sqrt(np.sum(np.power(x, 2.0)))
        if norm > radius:
            x = np.multiply(np.divide(radius, np.add(norm, 1e-9)), x)
    return x


def coordinate_descent(A: np.ndarray, b: np.ndarray, lam: float,
                       n_steps: int) -> np.ndarray:
    """Cyclic coordinate descent for L1-regularized least squares."""
    n, d = A.shape
    x = np.zeros(d)
    residual = np.copy(b)
    for _ in range(n_steps):
        for j in range(d):
            col = A[:, j]
            rho = np.add(np.dot(col, residual), np.multiply(np.dot(col, col), x[j]))
            z = np.subtract(rho, lam)
            if z <= 0.0:
                z = 0.0
            x_new = np.divide(z, np.add(np.dot(col, col), 1e-9))
            residual = np.add(residual, np.multiply(np.subtract(x[j], x_new), col))
            x[j] = x_new
    return x


def least_squares_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via normal equations with a small ridge."""
    n, d = X.shape
    XtX = np.matmul(np.transpose(X), X)
    XtX = np.add(XtX, np.multiply(1e-6, np.eye(d)))
    XtX_inv = np.linalg.inv(XtX)
    return np.matmul(XtX_inv, np.matmul(np.transpose(X), y))


def qr_least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QR-based least squares (numerically stable)."""
    Q, R = householder_qr(X)
    Qt_y = np.matmul(np.transpose(Q), y)
    n = R.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        acc = Qt_y[i]
        for j in range(i + 1, n):
            acc = np.subtract(acc, np.multiply(R[i, j], x[j]))
        x[i] = np.divide(acc, np.add(R[i, i], 1e-9))
    return x
