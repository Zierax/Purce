"""DSP / audio — filtering, spectral transforms, resampling, envelopes.

Harder patterns: multi-pass biquad filters, overlap-add convolution,
windowed spectral processing, and normalization loops.
"""

import numpy as np


def biquad_filter(x: np.ndarray, b0: float, b1: float, b2: float,
                  a1: float, a2: float) -> np.ndarray:
    """Direct-form I biquad (second-order IIR) filter."""
    n = x.shape[0]
    y = np.zeros(n)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for t in range(n):
        y[t] = np.add(np.add(np.multiply(b0, x[t]), np.multiply(b1, x1)),
                      np.add(np.multiply(b2, x2), np.negative(np.add(np.multiply(a1, y1),
                                                                     np.multiply(a2, y2)))))
        x2 = x1
        x1 = x[t]
        y2 = y1
        y1 = y[t]
    return y


def cascade_biquad(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Cascade of biquad sections (coeffs: [sections, 5])."""
    n_sections = coeffs.shape[0]
    y = x
    for s in range(n_sections):
        b0 = coeffs[s, 0]
        b1 = coeffs[s, 1]
        b2 = coeffs[s, 2]
        a1 = coeffs[s, 3]
        a2 = coeffs[s, 4]
        y = biquad_filter(y, b0, b1, b2, a1, a2)
    return y


def hamming_window(size: int) -> np.ndarray:
    """Hamming window: w(n) = 0.54 - 0.46*cos(2*pi*n/(N-1))."""
    idx = np.arange(size)
    return np.subtract(0.54, np.multiply(0.46, np.cos(np.multiply(
        np.divide(np.multiply(2.0, np.pi), np.subtract(size, 1.0)), idx))))


def blackman_window(size: int) -> np.ndarray:
    """Blackman window with 0.42/0.5/0.08 coefficients."""
    idx = np.arange(size)
    a0 = 0.42
    a1 = 0.5
    a2 = 0.08
    t1 = np.multiply(np.divide(np.multiply(2.0, np.pi), np.subtract(size, 1.0)), idx)
    t2 = np.multiply(2.0, t1)
    return np.subtract(np.add(a0, np.multiply(a1, np.cos(t1))),
                       np.multiply(a2, np.cos(t2)))


def overlap_add_convolution(x: np.ndarray, h: np.ndarray,
                            block: int) -> np.ndarray:
    """Overlap-add linear convolution via segmented FFT-free blocks."""
    n = x.shape[0]
    m = h.shape[0]
    out_len = n + m - 1
    y = np.zeros(out_len)
    for start in range(0, n, block):
        end = min(start + block, n)
        seg_len = end - start
        for i in range(seg_len):
            for j in range(m):
                y[start + i + j] = np.add(y[start + i + j],
                                          np.multiply(x[start + i], h[j]))
    return y


def envelope_detector(x: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Fast attack / slow release envelope follower."""
    n = x.shape[0]
    env = np.zeros(n)
    peak = 0.0
    for t in range(n):
        abs_x = np.abs(x[t])
        if abs_x > peak:
            peak = np.add(np.multiply(attack, abs_x), np.multiply(np.subtract(1.0, attack), peak))
        else:
            peak = np.add(np.multiply(release, abs_x), np.multiply(np.subtract(1.0, release), peak))
        env[t] = peak
    return env


def decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by factor with a simple moving-average prefilter."""
    n = x.shape[0]
    n_out = n // factor
    y = np.zeros(n_out)
    for i in range(n_out):
        start = i * factor
        acc = 0.0
        for k in range(factor):
            acc = np.add(acc, x[start + k])
        y[i] = np.divide(acc, factor)
    return y


def interpolate_linear(x: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation by integer factor."""
    n = x.shape[0]
    n_out = (n - 1) * factor + 1
    y = np.zeros(n_out)
    for i in range(n - 1):
        for f in range(factor):
            t = np.divide(f, factor)
            y[i * factor + f] = np.add(np.multiply(np.subtract(1.0, t), x[i]),
                                       np.multiply(t, x[i + 1]))
    y[(n - 1) * factor] = x[n - 1]
    return y


def spectral_centroid(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    """Weighted mean frequency of a spectrum."""
    num = np.sum(np.multiply(freqs, magnitudes))
    den = np.add(np.sum(magnitudes), 1e-7)
    return np.divide(num, den)


def spectral_rolloff(magnitudes: np.ndarray, freqs: np.ndarray,
                     percent: float = 0.85) -> float:
    """Frequency below which `percent` of the total energy is concentrated."""
    total = np.sum(magnitudes)
    cum = 0.0
    rolloff = freqs[0]
    n = magnitudes.shape[0]
    for i in range(n):
        cum = np.add(cum, magnitudes[i])
        if cum >= np.multiply(percent, total):
            rolloff = freqs[i]
            break
    return rolloff


def zero_crossing_rate(x: np.ndarray) -> float:
    """Number of sign changes per sample."""
    n = x.shape[0]
    count = 0.0
    for i in range(1, n):
        if np.multiply(x[i], x[i - 1]) < 0.0:
            count = np.add(count, 1.0)
    return np.divide(count, np.subtract(n, 1))


def mfcc_from_filterbank(spectrogram: np.ndarray, n_coeffs: int = 13) -> np.ndarray:
    """MFCC via log-energy + DCT (discrete cosine transform)."""
    n_frames, n_bins = spectrogram.shape
    log_energy = np.log(np.add(spectrogram, 1e-7))
    mfcc = np.zeros((n_frames, n_coeffs))
    for f in range(n_frames):
        for c in range(n_coeffs):
            acc = 0.0
            for k in range(n_bins):
                acc = np.add(acc, np.multiply(log_energy[f, k],
                                              np.cos(np.multiply(np.pi,
                                                  np.divide(np.multiply(c, np.add(k, 0.5)), n_bins)))))
            mfcc[f, c] = acc
    return mfcc


def autocorr_energy(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased autocorrelation used in LP coefficient estimation."""
    n = x.shape[0]
    r = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acc = 0.0
        for i in range(n - lag):
            acc = np.add(acc, np.multiply(x[i], x[i + lag]))
        r[lag] = np.divide(acc, n)
    return r


def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Linear prediction coefficients via Levinson-Durbin recursion."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc = np.subtract(acc, np.multiply(a[j], r[i - j]))
        k = np.divide(acc, np.add(e, 1e-7))
        a_new = np.copy(a)
        for j in range(1, i):
            a_new[j] = np.subtract(a[j], np.multiply(k, a[i - j]))
        a_new[i] = k
        a = a_new
        e = np.multiply(np.subtract(1.0, np.multiply(k, k)), e)
    return a
