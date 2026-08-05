"""N-body simulation — gravitational dynamics with softening.

Harder patterns: pairwise distance matrices via broadcasting, nested
loop accumulation, and update-with-damping in a single pass.
"""

import numpy as np


def gravitational_accel(pos: np.ndarray, masses: np.ndarray,
                        softening: float = 1e-2) -> np.ndarray:
    """Acceleration on every body from all others (O(N^2) direct sum)."""
    n, dim = pos.shape
    acc = np.zeros((n, dim))
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = np.subtract(pos[i], pos[j])
                dist2 = np.add(np.sum(np.power(diff, 2.0)), softening * softening)
                inv = np.divide(1.0, np.multiply(dist2, np.sqrt(dist2)))
                acc[i] = np.add(acc[i], np.multiply(masses[j], np.multiply(diff, inv)))
    return acc


def leapfrog_step(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray,
                  dt: float, softening: float = 1e-2) -> tuple:
    """Leapfrog integrator: half-kick, drift, kick."""
    acc = gravitational_accel(pos, masses, softening)
    vel = np.add(vel, np.multiply(acc, np.multiply(0.5, dt)))
    pos = np.add(pos, np.multiply(vel, dt))
    acc_new = gravitational_accel(pos, masses, softening)
    vel = np.add(vel, np.multiply(acc_new, np.multiply(0.5, dt)))
    return pos, vel


def total_energy(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray,
                 softening: float = 1e-2) -> float:
    """Total mechanical energy: kinetic + gravitational potential."""
    kin = np.multiply(0.5, np.sum(np.multiply(masses, np.sum(np.power(vel, 2.0), axis=-1))))
    n = pos.shape[0]
    pot = 0.0
    for i in range(n):
        for j in range(n):
            if i < j:
                diff = np.subtract(pos[i], pos[j])
                dist = np.sqrt(np.add(np.sum(np.power(diff, 2.0)), softening * softening))
                pot = np.subtract(pot, np.divide(np.multiply(masses[i], masses[j]), dist))
    return np.add(kin, pot)


def center_of_mass(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Mass-weighted centroid of the system."""
    total = np.sum(masses)
    weighted = np.sum(np.multiply(masses[:, np.newaxis], pos), axis=0)
    return np.divide(weighted, total)


def binding_energy(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray) -> float:
    """Specific binding energy of a two-body configuration."""
    diff = np.subtract(pos[0], pos[1])
    dist = np.sqrt(np.sum(np.power(diff, 2.0)))
    mu = np.divide(np.multiply(masses[0], masses[1]),
                   np.add(masses[0], masses[1]))
    v_rel = np.subtract(vel[0], vel[1])
    kinetic = np.multiply(0.5, np.multiply(mu, np.sum(np.power(v_rel, 2.0))))
    potential = np.negative(np.divide(np.multiply(masses[0], masses[1]), dist))
    return np.divide(np.add(kinetic, potential), mu)


def orbit_circular_velocity(mass: float, radius: float) -> float:
    """Circular orbital speed: v = sqrt(G*M/r) with G absorbed."""
    return np.sqrt(np.divide(mass, radius))


def perihelion_advance(a: float, e: float, m_central: float) -> float:
    """Relativistic perihelion precession per orbit (quadrupole term)."""
    return np.multiply(np.divide(6.0 * np.pi * m_central,
                                 np.multiply(np.multiply(a, 1.0 - e * e), 1.0 - e * e)),
                       1.0 / 3.0)


def simulate_trajectory(pos0: np.ndarray, vel0: np.ndarray,
                        masses: np.ndarray, n_steps: int,
                        dt: float) -> np.ndarray:
    """Run a leapfrog simulation, returning positions at each step."""
    n = pos0.shape[0]
    history = np.zeros((n_steps, n, pos0.shape[1]))
    pos = pos0
    vel = vel0
    for t in range(n_steps):
        pos, vel = leapfrog_step(pos, vel, masses, dt)
        history[t] = pos
    return history


def gravitational_softening_energy(pos: np.ndarray, masses: np.ndarray,
                                   softening: float) -> float:
    """Softened pairwise potential summed over all pairs."""
    n = pos.shape[0]
    pot = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.subtract(pos[i], pos[j])
            dist2 = np.add(np.sum(np.power(diff, 2.0)), softening * softening)
            pot = np.subtract(pot, np.divide(np.multiply(masses[i], masses[j]),
                                             np.sqrt(dist2)))
    return pot
