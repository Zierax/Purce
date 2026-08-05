"""Physics / control systems — transfer functions, PID, state space.

Harder patterns: polynomial arithmetic, error accumulation, saturation
logic, and closed-loop simulation loops.
"""

import numpy as np


def pid_controller(setpoint: np.ndarray, measured: np.ndarray,
                   kp: float, ki: float, kd: float,
                   dt: float, out_min: float = -1.0,
                   out_max: float = 1.0) -> tuple:
    """PID controller with clamped output and anti-windup."""
    n = setpoint.shape[0]
    output = np.zeros(n)
    integral = 0.0
    prev_error = 0.0
    for t in range(n):
        error = np.subtract(setpoint[t], measured[t])
        integral = np.add(integral, np.multiply(error, dt))
        derivative = np.divide(np.subtract(error, prev_error), np.add(dt, 1e-9))
        u = np.add(np.add(np.multiply(kp, error),
                          np.multiply(ki, integral)),
                   np.multiply(kd, derivative))
        if u > out_max:
            u = out_max
        if u < out_min:
            u = out_min
        output[t] = u
        prev_error = error
    return output, integral


def transfer_function_response(b: np.ndarray, a: np.ndarray,
                               u: np.ndarray) -> np.ndarray:
    """Direct-form II digital filter (difference equation)."""
    nb = b.shape[0]
    na = a.shape[0]
    n = u.shape[0]
    y = np.zeros(n)
    for t in range(n):
        acc = np.multiply(b[0], u[t])
        for k in range(1, nb):
            if t >= k:
                acc = np.add(acc, np.multiply(b[k], u[t - k]))
        for k in range(1, na):
            if t >= k:
                acc = np.subtract(acc, np.multiply(a[k], y[t - k]))
        y[t] = acc
    return y


def state_space_step(x: np.ndarray, u: np.ndarray, A: np.ndarray,
                     B: np.ndarray, C: np.ndarray, D: np.ndarray) -> tuple:
    """Single state-space step: x' = Ax + Bu, y = Cx + Du."""
    x_new = np.add(np.matmul(A, x), np.matmul(B, u))
    y = np.add(np.matmul(C, x), np.matmul(D, u))
    return x_new, y


def simulate_state_space(x0: np.ndarray, inputs: np.ndarray,
                         A: np.ndarray, B: np.ndarray,
                         C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Simulate a discrete-time LTI system over a control input sequence."""
    T, nu = inputs.shape
    n = x0.shape[0]
    ny = C.shape[0]
    y_hist = np.zeros((T, ny))
    x = x0
    for t in range(T):
        x, y = state_space_step(x, inputs[t], A, B, C, D)
        y_hist[t] = y
    return y_hist


def root_mean_square_error(reference: np.ndarray, actual: np.ndarray) -> float:
    """Root mean square error between two signals."""
    diff = np.subtract(reference, actual)
    return np.sqrt(np.mean(np.power(diff, 2.0)))


def settling_time(step_response: np.ndarray, setpoint: float,
                  band: float = 0.02) -> int:
    """Time (samples) until the step response stays within a band."""
    n = step_response.shape[0]
    settled = 0
    for t in range(n):
        if np.abs(np.subtract(step_response[t], setpoint)) < band:
            settled = t
    return settled


def phase_margin(crossover: float, phase_at: float) -> float:
    """Phase margin (degrees) at the gain crossover frequency."""
    return np.add(180.0, phase_at)


def gain_crossover(phase: np.ndarray, mag_db: np.ndarray) -> float:
    """Linear interpolation of the frequency where magnitude crosses 0 dB."""
    n = phase.shape[0]
    idx = 0
    for i in range(1, n):
        if mag_db[i - 1] >= 0.0 and mag_db[i] < 0.0:
            idx = i
            break
    if idx == 0:
        return 0.0
    t = np.divide(mag_db[idx - 1], np.subtract(mag_db[idx - 1], mag_db[idx]))
    return np.add(np.subtract(idx, 1.0), t)


def saturation_energy(u: np.ndarray, limit: float) -> float:
    """Cumulative saturation energy (how much signal was clipped)."""
    clipped = np.abs(np.subtract(u, np.clip(u, -limit, limit)))
    return np.sum(clipped)


def control_allocation(f_x: np.ndarray, f_y: np.ndarray,
                       max_thrust: float) -> np.ndarray:
    """Simple control allocation mixing x/y thrust with saturation."""
    mag = np.sqrt(np.add(np.power(f_x, 2.0), np.power(f_y, 2.0)))
    scale = np.minimum(np.divide(max_thrust, np.add(mag, 1e-9)), 1.0)
    return np.array([np.multiply(f_x, scale), np.multiply(f_y, scale)])


def mechanical_filter(x: np.ndarray, m: float, c: float,
                      k: float, dt: float) -> np.ndarray:
    """Second-order mechanical system response to force input."""
    n = x.shape[0]
    pos = np.zeros(n)
    vel = 0.0
    accel = 0.0
    for t in range(n):
        accel = np.divide(np.subtract(x[t], np.add(np.multiply(c, vel),
                                                   np.multiply(k, pos))),
                          m)
        vel = np.add(vel, np.multiply(accel, dt))
        pos = np.add(pos, np.multiply(vel, dt))
        pos_out = pos
        if t == 0:
            pos_out = x[0]
        pos[t] = pos_out
    return pos


def thermal_response(power: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """First-order thermal model: T_{t+1} = T_t + (P - T_t)/tau * dt."""
    n = power.shape[0]
    temp = np.zeros(n)
    for t in range(1, n):
        rate = np.divide(np.subtract(power[t], temp[t - 1]), tau)
        temp[t] = np.add(temp[t - 1], np.multiply(rate, dt))
    return temp
