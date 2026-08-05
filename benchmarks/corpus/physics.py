"""Physics simulation — spring-damper systems, pendulum, circuits, fluids.

Harder patterns: coupled ODE integration, energy invariants, and
nonlinear feedback with clamping.
"""

import numpy as np


def spring_damper_step(pos: np.ndarray, vel: np.ndarray, k: float,
                       c: float, m: float, dt: float) -> tuple:
    """Single step of a damped harmonic oscillator."""
    force = np.add(np.multiply(-k, pos), np.multiply(-c, vel))
    acc = np.divide(force, m)
    vel_new = np.add(vel, np.multiply(acc, dt))
    pos_new = np.add(pos, np.multiply(vel_new, dt))
    return pos_new, vel_new


def pendulum_step(theta: np.ndarray, omega: np.ndarray, g: float,
                  L: float, damping: float, dt: float) -> tuple:
    """Pendulum dynamics with numerical damping."""
    alpha = np.add(np.multiply(np.divide(-g, L), np.sin(theta)),
                   np.multiply(-damping, omega))
    omega_new = np.add(omega, np.multiply(alpha, dt))
    theta_new = np.add(theta, np.multiply(omega_new, dt))
    return theta_new, omega_new


def rc_circuit_step(v_cap: np.ndarray, v_in: np.ndarray, r: float,
                    c: float, dt: float) -> np.ndarray:
    """RC low-pass filter discretized with backward Euler."""
    tau = np.multiply(r, c)
    return np.divide(np.add(np.multiply(dt, v_in), np.multiply(tau, v_cap)),
                     np.add(tau, dt))


def rlc_circuit_energy(v_cap: np.ndarray, i_inductor: np.ndarray,
                       c: float, l: float) -> np.ndarray:
    """Total stored energy in an RLC circuit over time."""
    e_cap = np.multiply(0.5, np.multiply(c, np.power(v_cap, 2.0)))
    e_ind = np.multiply(0.5, np.multiply(l, np.power(i_inductor, 2.0)))
    return np.add(e_cap, e_ind)


def heat_diffusion_step(grid: np.ndarray, alpha: float, dt: float,
                        dx: float) -> np.ndarray:
    """Explicit 1D heat equation update (stable CFL assumed)."""
    n = grid.shape[0]
    new_grid = np.copy(grid)
    r = np.divide(np.multiply(alpha, dt), np.multiply(dx, dx))
    for i in range(1, n - 1):
        lap = np.subtract(np.add(grid[i - 1], grid[i + 1]),
                          np.multiply(2.0, grid[i]))
        new_grid[i] = np.add(grid[i], np.multiply(r, lap))
    return new_grid


def wave_equation_step(prev: np.ndarray, curr: np.ndarray, c: float,
                       dt: float, dx: float) -> np.ndarray:
    """Explicit 1D wave equation update (leapfrog in time)."""
    n = prev.shape[0]
    next_grid = np.copy(curr)
    r = np.multiply(np.multiply(c, dt), np.divide(c, dx))
    for i in range(1, n - 1):
        lap = np.subtract(np.add(prev[i - 1], prev[i + 1]),
                          np.multiply(2.0, prev[i]))
        next_grid[i] = np.add(np.multiply(2.0, curr[i]), np.subtract(np.negative(prev[i]),
                                                                     np.multiply(r, lap)))
    return next_grid


def diffusion_reaction_step(u: np.ndarray, v: np.ndarray, du: float,
                            dv: float, f: float, k: float, dt: float) -> tuple:
    """Gray-Scott reaction-diffusion update (two coupled fields)."""
    n = u.shape[0]
    u_new = np.copy(u)
    v_new = np.copy(v)
    for i in range(1, n - 1):
        lap_u = np.subtract(np.add(u[i - 1], u[i + 1]), np.multiply(2.0, u[i]))
        lap_v = np.subtract(np.add(v[i - 1], v[i + 1]), np.multiply(2.0, v[i]))
        reaction = np.multiply(u[i], np.multiply(v[i], v[i]))
        u_new[i] = np.add(u[i], np.multiply(np.add(np.multiply(du, lap_u),
                                                   np.subtract(f - k, reaction)), dt))
        v_new[i] = np.add(v[i], np.multiply(np.add(np.multiply(dv, lap_v),
                                                   np.subtract(reaction, np.multiply(k, v[i]))), dt))
    return u_new, v_new


def projectile_trajectory(v0: float, angle_deg: float, g: float,
                          dt: float) -> np.ndarray:
    """2D projectile motion with drag removed (vacuum parabola)."""
    theta = np.multiply(np.divide(np.pi, 180.0), angle_deg)
    vx = np.multiply(v0, np.cos(theta))
    vy = np.multiply(v0, np.sin(theta))
    t = 0.0
    x = 0.0
    y = 0.0
    path = []
    while y >= 0.0:
        path.append([x, y])
        x = np.add(x, np.multiply(vx, dt))
        y = np.add(y, np.multiply(vy, dt))
        vy = np.subtract(vy, np.multiply(g, dt))
        t = np.add(t, dt)
    return np.array(path)


def drag_coefficient(reynolds: np.ndarray) -> np.ndarray:
    """Sphere drag coefficient as a piecewise function of Re."""
    cd = np.zeros_like(reynolds)
    for i in range(reynolds.shape[0]):
        re = reynolds[i]
        if re < 0.1:
            cd[i] = np.divide(24.0, re)
        elif re < 1000.0:
            cd[i] = np.add(np.divide(24.0, re), np.multiply(0.4, np.divide(1.0, np.sqrt(re))))
        else:
            cd[i] = 0.4
    return cd


def lorenz_step(state: np.ndarray, sigma: float, rho: float,
                beta: float, dt: float) -> np.ndarray:
    """One RK2 step of the Lorenz attractor."""
    k1 = _lorenz_rhs(state, sigma, rho, beta)
    mid = np.add(state, np.multiply(np.multiply(0.5, dt), k1))
    k2 = _lorenz_rhs(mid, sigma, rho, beta)
    return np.add(state, np.multiply(dt, k2))


def _lorenz_rhs(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    x = state[0]
    y = state[1]
    z = state[2]
    return np.array([np.multiply(sigma, np.subtract(y, x)),
                     np.subtract(np.multiply(x, np.subtract(rho, z)), y),
                     np.subtract(np.multiply(x, y), np.multiply(beta, z))])


def kepler_orbit_state(energy: float, angular_momentum: float,
                       mu: float) -> np.ndarray:
    """Semi-major axis and eccentricity from orbital elements."""
    a = np.negative(np.divide(mu, np.multiply(2.0, energy)))
    e2 = np.subtract(1.0, np.divide(np.multiply(2.0, np.multiply(energy, np.power(angular_momentum, 2.0))),
                                    np.power(mu, 2.0)))
    e = np.sqrt(np.maximum(e2, 0.0))
    return np.array([a, e])


def spring_energy(pos: np.ndarray, vel: np.ndarray, k: float, m: float) -> np.ndarray:
    """Total mechanical energy of an oscillator system (invariant check)."""
    pe = np.multiply(0.5, np.multiply(k, np.power(pos, 2.0)))
    ke = np.multiply(0.5, np.multiply(m, np.power(vel, 2.0)))
    return np.add(pe, ke)
