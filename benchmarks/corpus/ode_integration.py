"""ODE integration — RK4, adaptive step control, stiff system solvers.

Harder patterns: nested loop integration kernels, error-controlled
step halving, and complex-arithmetic-backed wavefunctions.
"""

import numpy as np


def rk4_step(f: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Single RK4 step on the right-hand side already sampled at four points.

    The caller provides f = [f(t0, y0), f(t0+h/2, y1), f(t0+h/2, y2), f(t0+h, y3)].
    """
    k1 = f[0]
    k2 = f[1]
    k3 = f[2]
    k4 = f[3]
    return np.add(y, np.multiply(np.divide(h, 6.0),
                                 np.add(np.add(k1, np.multiply(2.0, k2)),
                                        np.add(np.multiply(2.0, k3), k4))))


def heun_step(f0: np.ndarray, f1: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Heun (trapezoidal predictor-corrector) step."""
    return np.add(y, np.multiply(np.divide(h, 2.0), np.add(f0, f1)))


def explicit_euler_step(f: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    """Forward Euler step."""
    return np.add(y, np.multiply(h, f))


def adaptive_rk4(y0: np.ndarray, t_end: float, h0: float, tol: float) -> tuple:
    """Adaptive step-size RK4 with step doubling error control."""
    t = 0.0
    h = h0
    y = y0
    steps = 0
    while t < t_end and steps < 10000:
        y_half = _sample_dynamics(y, t, h)
        y_full = _sample_dynamics_half(y, t, h)
        err = np.max(np.abs(np.subtract(y_full, y_half)))
        if err < tol:
            y = y_full
            t = np.add(t, h)
        h = np.multiply(h, np.multiply(0.9, np.power(np.divide(tol, np.add(err, 1e-12)),
                                                     0.2)))
        steps += 1
    return y, t


def _sample_dynamics(y: np.ndarray, t: float, h: float) -> np.ndarray:
    """Sample dynamics at four RK nodes (harmonic oscillator RHS)."""
    k1 = _rhs(y, t)
    k2 = _rhs(np.add(y, np.multiply(k1, np.multiply(0.5, h))), np.add(t, np.multiply(0.5, h)))
    k3 = _rhs(np.add(y, np.multiply(k2, np.multiply(0.5, h))), np.add(t, np.multiply(0.5, h)))
    k4 = _rhs(np.add(y, np.multiply(k3, h)), np.add(t, h))
    return rk4_step(np.array([k1, k2, k3, k4]), y, h)


def _sample_dynamics_half(y: np.ndarray, t: float, h: float) -> np.ndarray:
    """Dynamics sampled at h/2 for error estimation (embedded pair)."""
    hm = np.multiply(0.5, h)
    return _sample_dynamics(y, t, hm)


def _rhs(y: np.ndarray, t: float) -> np.ndarray:
    """Linear damped-oscillator RHS: dy/dt = [y1, -w^2 y0 - 2*z*w*y1]."""
    w = 1.0
    z = 0.1
    return np.array([y[1], np.subtract(np.negative(np.multiply(w * w, y[0])),
                                       np.multiply(np.multiply(2.0 * z * w, y[1]), 1.0))])


def velocity_verlet(acc0: np.ndarray, vel: np.ndarray, pos: np.ndarray,
                    acc1: np.ndarray, dt: float) -> tuple:
    """Velocity-Verlet: pos update with old accel, then vel with new accel."""
    pos_new = np.add(pos, np.add(np.multiply(vel, dt),
                                 np.multiply(np.multiply(0.5, acc0), dt * dt)))
    vel_new = np.add(vel, np.multiply(np.multiply(0.5, np.add(acc0, acc1)), dt))
    return pos_new, vel_new


def backward_euler_linear(y: np.ndarray, A: np.ndarray, b: np.ndarray,
                          h: float) -> np.ndarray:
    """Backward Euler for the linear system dy/dt = Ay + b."""
    I = np.eye(A.shape[0])
    M = np.subtract(I, np.multiply(h, A))
    rhs = np.add(y, np.multiply(h, b))
    return np.linalg.solve(M, rhs)


def semi_implicit_symplectic(pos: np.ndarray, vel: np.ndarray,
                             acc: np.ndarray, dt: float) -> tuple:
    """Semi-implicit (symplectic) Euler step."""
    vel_new = np.add(vel, np.multiply(acc, dt))
    pos_new = np.add(pos, np.multiply(vel_new, dt))
    return pos_new, vel_new


def integrate_damped_wave(y0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Integrate the damped wave equation with semi-implicit Euler."""
    dim = y0.shape[0]
    history = np.zeros((n_steps, dim))
    y = y0
    v = np.zeros(dim)
    for t in range(n_steps):
        a = np.multiply(-1.0, np.multiply(y, 0.5))
        y, v = semi_implicit_symplectic(y, v, a, dt)
        history[t] = y
    return history
