"""Data pipeline — tokenization, embedding, positional encoding, data augmentation."""

import numpy as np


def bpe_tokenize(text: str, vocab: dict, merges: list) -> list:
    """Byte Pair Encoding tokenization."""
    tokens = list(text.encode('utf-8'))
    for pair in merges:
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair:
                merged = vocab.get(pair, sum(pair))
                tokens = tokens[:i] + [merged] + tokens[i + 2:]
            else:
                i += 1
    return tokens


def wordpiece_tokenize(tokens: list, vocab: dict, max_len: int = 512) -> list:
    """WordPiece tokenization with ## prefix for subwords."""
    output = []
    for token in tokens[:max_len]:
        chars = list(token)
        if token in vocab:
            output.append(vocab[token])
        else:
            start = 0
            while start < len(chars):
                end = len(chars)
                found = False
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in vocab:
                        output.append(vocab[substr])
                        found = True
                        break
                    end -= 1
                if not found:
                    output.append(vocab.get('[UNK]', 0))
                    break
                start = end
    return output


def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding: PE(pos, 2i) = sin(pos/10000^(2i/d))."""
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.multiply(np.arange(0, d_model, 2), -(np.log(10000.0) / d_model)))
    pe[:, 0::2] = np.sin(np.multiply(position, div_term))
    pe[:, 1::2] = np.cos(np.multiply(position, div_term))
    return pe


def rotary_encoding(x: np.ndarray, seq_len: int) -> np.ndarray:
    """Rotary Position Embedding (RoPE)."""
    d = x.shape[-1]
    theta = np.power(10000.0, np.divide(-np.arange(0, d, 2).astype(np.float64), d))
    positions = np.arange(seq_len).reshape(-1, 1)
    angles = np.multiply(positions, theta)
    cos_cache = np.cos(angles)
    sin_cache = np.sin(angles)

    x_rotated = np.zeros_like(x)
    x_rotated[..., 0::2] = np.subtract(
        np.multiply(x[..., 0::2], cos_cache),
        np.multiply(x[..., 1::2], sin_cache))
    x_rotated[..., 1::2] = np.add(
        np.multiply(x[..., 0::2], sin_cache),
        np.multiply(x[..., 1::2], cos_cache))
    return x_rotated


def learned_embedding(tokens: np.ndarray, vocab_size: int, embed_dim: int) -> np.ndarray:
    """Simple lookup embedding."""
    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, embed_dim) * 0.02
    return embeddings[tokens]


def token_mixup(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Data augmentation: mixup at the token level."""
    batch_size = x.shape[0]
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(batch_size)
    return np.add(np.multiply(lam, x), np.multiply(1 - lam, x[indices]))


def random_erase(x: np.ndarray, prob: float = 0.15) -> np.ndarray:
    """Random erasing augmentation."""
    mask = np.greater(np.random.random(x.shape), prob).astype(np.float64)
    return np.multiply(x, mask)


def time_warp(signal: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
    """Time warping augmentation for 1D signals."""
    n = len(signal)
    center = n // 2
    shift = int(n * warp_factor * np.random.uniform(-1, 1))
    output = np.zeros(n)
    for i in range(n):
        src = i + int(shift * np.exp(-((i - center) ** 2) / (2 * (n / 4) ** 2)))
        src = max(0, min(n - 1, src))
        output[i] = signal[src]
    return output


def spec_augment(spec: np.ndarray, freq_mask: int = 10,
                 time_mask: int = 20, num_masks: int = 2) -> np.ndarray:
    """SpecAugment: time and frequency masking augmentation."""
    result = np.copy(spec)
    n_freq, n_time = result.shape

    for _ in range(num_masks):
        f = np.random.randint(0, max(freq_mask, 1))
        f_start = np.random.randint(0, max(n_freq - f, 1))
        result[f_start:f_start + f, :] = 0.0

        t = np.random.randint(0, max(time_mask, 1))
        t_start = np.random.randint(0, max(n_time - t, 1))
        result[:, t_start:t_start + t] = 0.0

    return result


def collate_sequences(sequences: list, pad_value: float = 0.0) -> np.ndarray:
    """Pad sequences to the same length and stack."""
    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded


def create_attention_mask(seq_len: int, causal: bool = True) -> np.ndarray:
    """Create attention mask: 1 for attended positions, 0 for masked."""
    mask = np.ones((seq_len, seq_len))
    if causal:
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                mask[i, j] = 0.0
    return mask


def sliding_window(data: np.ndarray, window_size: int,
                   stride: int = 1) -> np.ndarray:
    """Create sliding window view of 1D data."""
    n_windows = (len(data) - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        windows[i] = data[i * stride:i * stride + window_size]
    return windows
