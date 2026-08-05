"""Numerical methods — root finding, optimization, quadrature, splines.

Harder patterns: iterative loops with convergence tests, nested
evaluation, bracketing logic, and cubic-spline systems.
"""

import numpy as np


def bisection_root(fa: float, fb: float, a: float, b: float,
                   tol: float = 1e-8, max_iter: int = 100) -> float:
    """Bisection on a monotone function evaluated at endpoints only."""
    lo = a
    hi = b
    f_lo = fa
    f_hi = fb
    for _ in range(max_iter):
        mid = np.multiply(0.5, np.add(lo, hi))
        f_mid = _evaluate_monotone(mid)
        if np.multiply(f_lo, f_mid) < 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
        if np.abs(np.subtract(hi, lo)) < tol:
            break
    return np.multiply(0.5, np.add(lo, hi))


def _evaluate_monotone(x: float) -> float:
    """Monotone target: f(x) = x^3 - 2x - 5 (single real root)."""
    return np.subtract(np.subtract(np.power(x, 3.0), np.multiply(2.0, x)), 5.0)


def newton_raphson(x0: float, tol: float = 1e-8, max_iter: int = 50) -> tuple:
    """Newton-Raphson with finite-difference derivative."""
    x = x0
    iters = 0
    for _ in range(max_iter):
        f = _evaluate_monotone(x)
        df = _finite_diff(x, 1e-6)
        x_new = np.subtract(x, np.divide(f, df))
        if np.abs(np.subtract(x_new, x)) < tol:
            x = x_new
            iters += 1
            break
        x = x_new
        iters += 1
    return x, iters


def _finite_diff(x: float, h: float) -> float:
    """Central finite-difference derivative."""
    fp = _evaluate_monotone(np.add(x, h))
    fm = _evaluate_monotone(np.subtract(x, h))
    return np.divide(np.subtract(fp, fm), np.multiply(2.0, h))


def secant_method(x0: float, x1: float, tol: float = 1e-8,
                  max_iter: int = 50) -> float:
    """Secant method (no derivative needed)."""
    x_prev = x0
    x = x1
    for _ in range(max_iter):
        f_prev = _evaluate_monotone(x_prev)
        f_cur = _evaluate_monotone(x)
        denom = np.subtract(f_cur, f_prev)
        if np.abs(denom) < 1e-12:
            break
        x_new = np.subtract(x, np.divide(np.multiply(f_cur, np.subtract(x, x_prev)), denom))
        if np.abs(np.subtract(x_new, x)) < tol:
            x = x_new
            break
        x_prev = x
        x = x_new
    return x


def gradient_descent(grad0: np.ndarray, lr: float, n_steps: int) -> np.ndarray:
    """Plain gradient descent on a quadratic potential."""
    x = grad0
    for _ in range(n_steps):
        g = _quadratic_grad(x)
        x = np.subtract(x, np.multiply(lr, g))
    return x


def _quadratic_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of 0.5*x^T Q x with Q = 2I (stiff but solvable)."""
    return np.multiply(2.0, x)


def momentum_descent(grad0: np.ndarray, lr: float, momentum: float,
                     n_steps: int) -> np.ndarray:
    """Gradient descent with momentum (heavy-ball method)."""
    x = grad0
    v = np.zeros_like(grad0)
    for _ in range(n_steps):
        g = _quadratic_grad(x)
        v = np.add(np.multiply(momentum, v), np.multiply(lr, g))
        x = np.subtract(x, v)
    return x


def trapezoidal_integral(values: np.ndarray, h: float) -> float:
    """Composite trapezoidal rule on uniformly spaced samples."""
    n = values.shape[0]
    total = np.multiply(np.add(values[0], values[n - 1]), 0.5)
    total = np.add(total, np.sum(values[1:n - 1]))
    return np.multiply(total, h)


def simpson_integral(values: np.ndarray, h: float) -> float:
    """Composite Simpson's rule (n must be odd number of samples)."""
    n = values.shape[0]
    even_sum = 0.0
    odd_sum = 0.0
    for i in range(1, n - 1):
        if i % 2 == 0:
            even_sum = np.add(even_sum, values[i])
        else:
            odd_sum = np.add(odd_sum, values[i])
    total = np.add(np.add(values[0], values[n - 1]),
                   np.add(np.multiply(2.0, even_sum), np.multiply(4.0, odd_sum)))
    return np.multiply(np.divide(h, 3.0), total)


def natural_cubic_spline(x: np.ndarray, y: np.ndarray) -> tuple:
    """Natural cubic spline coefficients via tridiagonal solve."""
    n = x.shape[0]
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = np.multiply(3.0, np.subtract(
            np.divide(np.subtract(y[i + 1], y[i]), h[i]),
            np.divide(np.subtract(y[i], y[i - 1]), h[i - 1])))
    c = np.zeros(n)
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        l[i] = np.subtract(np.multiply(2.0, np.subtract(x[i + 1], x[i - 1])),
                           np.multiply(h[i - 1], mu[i - 1]))
        mu[i] = np.divide(h[i], l[i])
        z[i] = np.divide(np.subtract(alpha[i], np.multiply(h[i - 1], z[i - 1])), l[i])
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 2, -1, -1):
        c[j] = np.subtract(z[j], np.multiply(mu[j], c[j + 1]))
        b[j] = np.divide(np.subtract(np.subtract(y[j + 1], y[j]), np.multiply(h[j],
                        np.add(c[j + 1], np.multiply(2.0, c[j])))), h[j])
        d[j] = np.divide(np.subtract(c[j + 1], c[j]), np.multiply(3.0, h[j]))
    return b, c, d


def richardson_extrapolation(rh: float, r_half: float) -> float:
    """Richardson extrapolation: R = (4*R_{h/2} - R_h)/3."""
    return np.divide(np.subtract(np.multiply(4.0, r_half), rh), 3.0)


def golden_section_search(flo: float, fhi: float, a: float, b: float,
                          tol: float = 1e-6) -> float:
    """Golden-section minimization of a unimodal quadratic (endpoint values)."""
    phi = np.divide(np.subtract(np.sqrt(5.0), 1.0), 2.0)
    c = np.subtract(b, np.multiply(phi, np.subtract(b, a)))
    d = np.add(a, np.multiply(phi, np.subtract(b, a)))
    fc = _evaluate_quadratic(c)
    fd = _evaluate_quadratic(d)
    for _ in range(100):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = np.subtract(b, np.multiply(phi, np.subtract(b, a)))
            fc = _evaluate_quadratic(c)
        else:
            a = c
            c = d
            fc = fd
            d = np.add(a, np.multiply(phi, np.subtract(b, a)))
            fd = _evaluate_quadratic(d)
        if np.abs(np.subtract(b, a)) < tol:
            break
    return np.multiply(0.5, np.add(a, b))


def _evaluate_quadratic(x: float) -> float:
    """Unimodal quadratic: f(x) = (x - 2)^2 + 1."""
    d = np.subtract(x, 2.0)
    return np.add(np.multiply(d, d), 1.0)
