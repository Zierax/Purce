"""Option pricing — Black-Scholes, binomial tree, Monte Carlo paths.

Harder patterns: erf/erfcinv-style transforms, cumulative loops,
ternary branches, and payoff aggregation with reductions.
"""

import numpy as np


def black_scholes(S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0.0) -> np.ndarray:
    """Black-Scholes European option price (call and put)."""
    d1 = np.divide(np.add(np.log(np.divide(S, K)),
                          np.multiply(np.add(np.subtract(r, q),
                                             np.divide(np.power(sigma, 2.0), 2.0)), T)),
                   np.multiply(sigma, np.sqrt(T)))
    d2 = np.subtract(d1, np.multiply(sigma, np.sqrt(T)))
    neg_d2 = np.negative(d2)
    call = np.subtract(np.multiply(S, np.multiply(np.exp(np.multiply(-q, T)),
                                                  _norm_cdf(d1))),
                       np.multiply(np.multiply(K, np.exp(np.multiply(-r, T))),
                                   _norm_cdf(d2)))
    put = np.add(np.multiply(np.multiply(K, np.exp(np.multiply(-r, T))),
                             _norm_cdf(neg_d2)),
                 np.multiply(np.multiply(S, np.exp(np.multiply(-q, T))),
                             np.subtract(_norm_cdf(d1), 1.0)))
    return np.array([call, put])


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via Abramowitz-Stegun erf approximation."""
    t = np.divide(1.0, np.add(1.0, np.multiply(0.3275911, np.abs(x))))
    poly = np.add(np.multiply(1.061405429,
                              np.add(np.multiply(-1.453152027,
                                                 np.add(np.multiply(1.421413741,
                                                                    np.add(np.multiply(-0.284496736,
                                                                                       np.add(np.multiply(0.254829592, t), -1.0)),
                                                                           t)),
                                                       t)),
                                     t)),
                  t)
    erf_approx = np.subtract(1.0,
                             np.multiply(np.multiply(poly, t),
                                         np.exp(np.multiply(-np.multiply(x, x), np.subtract(1.273240, 1.0)))))
    erf_approx = np.where(np.less(x, 0.0), np.negative(erf_approx), erf_approx)
    return np.multiply(0.5, np.add(1.0, erf_approx))


def binomial_tree(S: float, K: float, T: float, r: float,
                  sigma: float, n_steps: int) -> float:
    """Cox-Ross-Rubinstein binomial tree for European options."""
    dt = np.divide(T, n_steps)
    u = np.exp(np.multiply(sigma, np.sqrt(dt)))
    d = np.exp(np.multiply(-sigma, np.sqrt(dt)))
    p = np.divide(np.subtract(np.exp(np.multiply(r, dt)), d),
                  np.subtract(u, d))
    discount = np.exp(np.multiply(-r, dt))
    n = n_steps + 1
    values = np.zeros(n)
    for i in range(n):
        st = np.multiply(S, np.power(u, np.subtract(n_steps, i)))
        st = np.multiply(st, np.power(d, i))
        values[i] = np.maximum(np.subtract(st, K), 0.0)
    for step in range(n_steps - 1, -1, -1):
        for i in range(step + 1):
            up = np.multiply(discount, values[i])
            down = np.multiply(discount, values[i + 1])
            values[i] = np.add(np.multiply(p, up), np.multiply(np.subtract(1.0, p), down))
    return values[0]


def monte_carlo_payoff(S0: float, K: float, T: float, r: float,
                       sigma: float, n_paths: int, n_steps: int) -> float:
    """Monte Carlo European call via geometric Brownian motion."""
    dt = np.divide(T, n_steps)
    drift = np.multiply(np.subtract(r, np.divide(np.power(sigma, 2.0), 2.0)), dt)
    vol = np.multiply(sigma, np.sqrt(dt))
    S = np.full(n_paths, S0)
    for t in range(n_steps):
        shock = _gauss_noise(n_paths)
        S = np.multiply(S, np.exp(np.add(drift, np.multiply(vol, shock))))
    payoff = np.maximum(np.subtract(S, K), 0.0)
    mean = np.mean(payoff)
    return np.multiply(np.exp(np.multiply(-r, T)), mean)


def _gauss_noise(n: int) -> np.ndarray:
    """Deterministic pseudo-Gaussian noise via Box-Muller on a ramp."""
    u1 = np.linspace(0.1, 0.9, n)
    u2 = np.linspace(0.2, 0.8, n)
    r = np.sqrt(np.multiply(-2.0, np.log(u1)))
    theta = np.multiply(2.0 * np.pi, u2)
    return np.multiply(r, np.cos(theta))


def delta_hedge_pnl(S_path: np.ndarray, K: float, r: float,
                    sigma: float, dt: float) -> float:
    """Cumulative PnL of a delta-hedged short call position."""
    T, n = S_path.shape
    pnl = 0.0
    for t in range(n - 1):
        tau = np.multiply(np.subtract(n - 1 - t, 1), dt)
        d1 = np.divide(np.add(np.log(np.divide(S_path[t], K)),
                              np.add(r, np.divide(np.power(sigma, 2.0), 2.0)) * tau),
                       np.multiply(sigma, np.sqrt(tau)))
        delta = _norm_cdf(d1)
        dS = np.subtract(S_path[t + 1], S_path[t])
        pnl = np.add(pnl, np.multiply(np.subtract(delta, _norm_cdf(d1)), dS))
    return pnl


def implied_vol_vega(S: float, K: float, T: float, r: float,
                     sigma0: float, market_price: float) -> float:
    """Vega (price sensitivity to volatility) at a given sigma."""
    d1 = np.divide(np.add(np.log(np.divide(S, K)),
                          np.multiply(np.add(r, np.divide(np.power(sigma0, 2.0), 2.0)), T)),
                   np.multiply(sigma0, np.sqrt(T)))
    pdf = _norm_pdf(d1)
    vega = np.multiply(np.multiply(np.multiply(S, np.sqrt(T)), pdf),
                       np.exp(np.multiply(-r, T)))
    return vega


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.divide(1.0, np.sqrt(np.multiply(2.0, np.pi))),
                       np.exp(np.multiply(-0.5, np.power(x, 2.0))))
