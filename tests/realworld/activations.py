"""Advanced activation functions — swish, gelu, mish, lisht, serf, soiu."""

import numpy as np


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Swish: x * sigmoid(beta * x)"""
    return np.multiply(x, _sigmoid(np.multiply(x, beta)))


def swish_derivative(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """d/dx swish(x) = swish(x) + sigmoid(beta*x) * (1 - swish(x)) * beta"""
    s = _sigmoid(np.multiply(x, beta))
    sw = np.multiply(x, s)
    return np.add(sw, np.multiply(np.multiply(s, np.subtract(1.0, sw)), beta))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU: x * Phi(x) where Phi is the CDF of standard normal."""
    inner = np.multiply(0.7978845608, np.add(x, np.multiply(0.044715, np.power(x, 3))))
    return np.multiply(x, _sigmoid(inner))


def mish(x: np.ndarray) -> np.ndarray:
    """Mish: x * tanh(softplus(x))"""
    return np.multiply(x, np.tanh(_softplus(x)))


def lisht(x: np.ndarray) -> np.ndarray:
    """LiSHT: x * tanh(x)"""
    return np.multiply(x, np.tanh(x))


def serf(x: np.ndarray) -> np.ndarray:
    """SERF: x * erf(softplus(x))"""
    return np.multiply(x, _erf(_softplus(x)))


def soiu(x: np.ndarray, alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """SoIU activation: sigmoid(alpha * x) * tanh(softplus(beta * x))"""
    return np.multiply(_sigmoid(np.multiply(alpha, x)),
                       np.tanh(_softplus(np.multiply(beta, x))))


def logsig(x: np.ndarray) -> np.ndarray:
    """LogSigmoid: log(sigmoid(x))"""
    return np.log(np.add(_sigmoid(x), 1e-7))


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU: x if x > 0, alpha * (exp(x) - 1) otherwise."""
    return np.where(np.greater(x, 0), x, np.multiply(alpha, np.subtract(np.exp(x), 1.0)))


def selu(x: np.ndarray) -> np.ndarray:
    """SELU: lambda * elu(x, alpha) with lambda=1.0507, alpha=1.6733."""
    alpha = 1.6733
    lam = 1.0507
    return np.multiply(lam, elu(x, alpha))


def glu(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Gated Linear Unit: split x into (a, b), return a * sigmoid(b)."""
    mid = x.shape[axis] // 2
    a = np.take(x, range(mid), axis=axis)
    b = np.take(x, range(mid, x.shape[axis]), axis=axis)
    return np.multiply(a, _sigmoid(b))


def celu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Continuously differentiable ELU."""
    return np.where(np.greater(x, 0), x, np.multiply(alpha, np.subtract(np.exp(np.divide(x, alpha)), 1.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(np.greater(x, 20), x, np.log(np.add(np.exp(x), 1.0)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(np.greater(x, 0),
                    np.divide(1.0, np.add(1.0, np.exp(-x))),
                    np.divide(np.exp(x), np.add(np.exp(x), 1.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Approximation of erf(x) using Horner form."""
    x_abs = np.abs(x)
    t = np.divide(1.0, np.add(1.0, np.multiply(0.3275911, x_abs)))
    t2 = np.multiply(t, t)
    t3 = np.multiply(t2, t)
    t4 = np.multiply(t3, t)
    t5 = np.multiply(t4, t)
    poly = np.add(np.multiply(0.254829592, t),
                  np.add(np.multiply(-0.284496736, t2),
                         np.add(np.multiply(1.421413741, t3),
                                np.add(np.multiply(-1.453152027, t4),
                                       np.multiply(1.061405429, t5)))))
    result = np.subtract(1.0, np.multiply(poly, np.exp(np.multiply(-x_abs, x_abs))))
    return np.where(np.greater(x, 0), result, np.negative(result))
