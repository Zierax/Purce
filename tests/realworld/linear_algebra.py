"""Advanced linear algebra — SVD, eigenvalue decomposition, QR, Cholesky, matrix functions."""

import numpy as np


def svd_decompose(A: np.ndarray, num_iters: int = 100) -> tuple:
    """SVD via Jacobi eigenvalue algorithm (simplified)."""
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)
    S = np.copy(A)

    for _ in range(num_iters):
        for i in range(min(m, n)):
            for j in range(i + 1, max(m, n)):
                if i < m and j < n:
                    a, b, c, d = S[i, i], S[i, j], S[j, i], S[j, j]
                    if abs(c) > 1e-10:
                        tau = (d - a) / (2.0 * c)
                        t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau * tau))
                        cs = 1.0 / np.sqrt(1 + t * t)
                        sn = t * cs
                        G = np.eye(max(m, n))
                        G[i, i], G[i, j] = cs, sn
                        G[j, i], G[j, j] = -sn, cs
                        S = np.matmul(S, G[:n, :n])
                        U = np.matmul(U, G[:m, :m].T)

    singular_values = np.sqrt(np.maximum(np.diag(np.matmul(S.T, S)), 0))
    return U, singular_values, V.T


def matrix_exp(A: np.ndarray, terms: int = 20) -> np.ndarray:
    """Matrix exponential via Taylor series: exp(A) = sum(A^k / k!)."""
    result = np.eye(A.shape[0])
    A_power = np.eye(A.shape[0])
    factorial = 1.0
    for k in range(1, terms):
        A_power = np.matmul(A_power, A)
        factorial *= k
        result = np.add(result, np.divide(A_power, factorial))
    return result


def matrix_log(A: np.ndarray, terms: int = 50) -> np.ndarray:
    """Matrix logarithm via series: log(I + (A-I)) = sum((-1)^(k+1) * (A-I)^k / k)."""
    B = np.subtract(A, np.eye(A.shape[0]))
    result = np.zeros_like(A, dtype=np.float64)
    B_power = np.eye(A.shape[0])
    for k in range(1, terms):
        B_power = np.matmul(B_power, B)
        sign = 1.0 if k % 2 == 1 else -1.0
        result = np.add(result, np.multiply(sign, np.divide(B_power, k)))
    return result


def matrix_sqrt(A: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Matrix square root via Newton's method: X_{k+1} = 0.5 * (X_k + A * X_k^{-1})."""
    X = np.eye(A.shape[0], dtype=np.float64)
    for _ in range(iterations):
        X_inv = np.linalg.inv(X)
        X = np.multiply(0.5, np.add(X, np.matmul(A, X_inv)))
    return X


def qr_decompose(A: np.ndarray) -> tuple:
    """QR decomposition via Householder reflections."""
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)

    for j in range(min(m - 1, n)):
        x = R[j:, j]
        alpha = -np.sign(x[0]) * np.sqrt(np.sum(np.power(x, 2)))
        v = np.copy(x)
        v[0] = np.subtract(v[0], alpha)
        v = np.divide(v, np.sqrt(np.sum(np.power(v, 2))) + 1e-7)

        H = np.eye(m)
        H[j:, j:] = np.subtract(np.eye(m - j), np.multiply(2.0, np.outer(v, v)))
        R = np.matmul(H, R)
        Q = np.matmul(Q, H)

    return Q, R


def cholesky_decompose(A: np.ndarray) -> np.ndarray:
    """Cholesky decomposition: A = L * L^T."""
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = np.sum(np.power(L[i, :j], 2))
            if i == j:
                val = A[i, i] - s
                L[i, j] = np.sqrt(max(val, 0.0))
            else:
                denom = L[j, j]
                if abs(denom) > 1e-15:
                    L[i, j] = np.divide(A[i, j] - s, denom)
                else:
                    L[i, j] = 0.0
    return L


def matrix_power(A: np.ndarray, p: float, terms: int = 30) -> np.ndarray:
    """Matrix power via eigendecomposition approximation."""
    eigvals = _approximate_eigenvalues(A, terms)
    eigvecs = np.eye(A.shape[0])
    powered_eigvals = np.power(eigvals, p)
    return np.matmul(np.matmul(eigvecs, np.diag(powered_eigvals)),
                     np.linalg.inv(eigvecs))


def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kronecker product of two matrices."""
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m * p, n * q))
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = np.multiply(A[i, j], B)
    return result


def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Element-wise (Hadamard) matrix product."""
    return np.multiply(A, B)


def lorentzian_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Lorentzian distance: log(1 + ||x - y||^2)."""
    diff = np.subtract(X, Y)
    sq_dist = np.sum(np.power(diff, 2), axis=-1, keepdims=True)
    return np.log(np.add(sq_dist, 1.0))


def _approximate_eigenvalues(A: np.ndarray, iterations: int) -> np.ndarray:
    """Power iteration to approximate eigenvalues."""
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    B = np.copy(A)
    for _ in range(iterations):
        v = np.random.randn(n)
        for _ in range(20):
            w = np.matmul(B, v)
            norm = np.sqrt(np.sum(np.power(w, 2)))
            v = np.divide(w, max(norm, 1e-7))
        eigenvalue = np.dot(v, np.matmul(B, v))
        eigenvalues[_ % n] = eigenvalue
    return eigenvalues
