"""Signal processing — FFT-based convolution, spectrogram, mel filterbank, STFT."""

import numpy as np


def fft_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT-based convolution: conv(x, k) = IFFT(FFT(x) * FFT(k))."""
    n = len(signal) + len(kernel) - 1
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    sig_padded = np.zeros(fft_size)
    sig_padded[:len(signal)] = signal
    kern_padded = np.zeros(fft_size)
    kern_padded[:len(kernel)] = kernel

    sig_fft = _fft(sig_padded)
    kern_fft = _fft(kern_padded)
    product = np.multiply(sig_fft, kern_fft)
    result = np.real(_ifft(product))
    return result[:n]


def spectrogram(signal: np.ndarray, window_size: int = 256,
                hop_size: int = 128, n_fft: int = 256) -> np.ndarray:
    """Short-time Fourier Transform spectrogram."""
    num_frames = (len(signal) - window_size) // hop_size + 1
    window = _hann_window(window_size)
    spec = np.zeros((num_frames, n_fft // 2 + 1))

    for i in range(num_frames):
        start = i * hop_size
        frame = np.multiply(signal[start:start + window_size], window)
        spectrum = _fft_real(frame, n_fft)
        mag = np.abs(spectrum[:n_fft // 2 + 1])
        spec[i, :] = np.log(np.add(mag, 1e-7))
    return spec


def mel_filterbank(n_filters: int = 80, n_fft: int = 256,
                   sample_rate: int = 16000) -> np.ndarray:
    """Mel-scale triangular filterbank."""
    low_freq_mel = 0.0
    high_freq_mel = 2595.0 * np.log10(np.add(1.0, np.divide(sample_rate / 2.0, 700.0)))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = np.multiply(700.0, np.subtract(np.power(10.0, np.divide(mel_points, 2595.0)), 1.0))
    bin_points = np.floor(np.divide(np.multiply(n_fft + 1, hz_points), sample_rate)).astype(int)

    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[m - 1, k] = np.divide(np.subtract(k, f_left),
                                                  np.subtract(f_center, f_left))
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[m - 1, k] = np.divide(np.subtract(f_right, k),
                                                  np.subtract(f_right, f_center))
    return filterbank


def stft(signal: np.ndarray, window_size: int = 1024,
         hop_size: int = 512, n_fft: int = 1024) -> tuple:
    """Short-Time Fourier Transform: returns (magnitude, phase)."""
    num_frames = (len(signal) - window_size) // hop_size + 1
    window = _hann_window(window_size)
    magnitude = np.zeros((num_frames, n_fft // 2 + 1))
    phase = np.zeros((num_frames, n_fft // 2 + 1))

    for i in range(num_frames):
        start = i * hop_size
        frame = np.multiply(signal[start:start + window_size], window)
        spectrum = _fft_real(frame, n_fft)
        half = spectrum[:n_fft // 2 + 1]
        magnitude[i, :] = np.abs(half)
        phase[i, :] = np.angle(half)
    return magnitude, phase


def istft(magnitude: np.ndarray, phase: np.ndarray,
          window_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    """Inverse Short-Time Fourier Transform."""
    num_frames = magnitude.shape[0]
    output_len = (num_frames - 1) * hop_size + window_size
    output = np.zeros(output_len)
    window = _hann_window(window_size)
    window_sum = np.zeros(output_len)

    for i in range(num_frames):
        spectrum = np.multiply(magnitude[i], np.exp(np.multiply(1j, phase[i])))
        full_spectrum = np.zeros(window_size)
        full_spectrum[:window_size // 2 + 1] = spectrum
        full_spectrum[window_size // 2 + 1:] = np.conj(spectrum[1:window_size // 2])[::-1]
        frame = np.real(_ifft(full_spectrum))
        start = i * hop_size
        output[start:start + window_size] += np.multiply(frame, window)
        window_sum[start:start + window_size] += np.power(window, 2)

    window_sum = np.maximum(window_sum, 1e-7)
    return np.divide(output, window_sum)


def cepstral_coefficients(spectrogram: np.ndarray, n_coeffs: int = 13) -> np.ndarray:
    """MFCC: Mel-Frequency Cepstral Coefficients."""
    mel_spec = np.dot(spectrogram, mel_filterbank(spectrogram.shape[1], spectrogram.shape[1]).T)
    log_mel = np.log(np.add(mel_spec, 1e-7))
    n_frames = log_mel.shape[0]
    mfcc = np.zeros((n_frames, n_coeffs))
    for i in range(n_frames):
        for j in range(n_coeffs):
            s = 0.0
            for k in range(log_mel.shape[1]):
                s += log_mel[i, k] * np.cos(np.pi * j * (k + 0.5) / log_mel.shape[1])
            mfcc[i, j] = s
    return mfcc


def _fft(x: np.ndarray) -> np.ndarray:
    """FFT via Cooley-Tukey radix-2."""
    n = len(x)
    if n <= 1:
        return x.astype(np.complex128)
    if n % 2 != 0:
        return _dft(x)

    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    T = np.exp(np.multiply(-2j, np.pi * np.arange(n // 2) / n))
    result = np.zeros(n, dtype=np.complex128)
    result[:n // 2] = np.add(even, np.multiply(T, odd))
    result[n // 2:] = np.subtract(even, np.multiply(T, odd))
    return result


def _ifft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    conj = np.conj(x)
    result = _fft(conj)
    return np.divide(np.conj(result), n)


def _fft_real(x: np.ndarray, n: int = 0) -> np.ndarray:
    if n > len(x):
        padded = np.zeros(n)
        padded[:len(x)] = x
        return _fft(padded)
    return _fft(x)


def _dft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    result = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        for j in range(n):
            result[k] += x[j] * np.exp(-2j * np.pi * k * j / n)
    return result


def _hann_window(size: int) -> np.ndarray:
    return np.multiply(0.5, np.subtract(1.0, np.cos(np.multiply(
        2.0 * np.pi / size, np.arange(size)))))
