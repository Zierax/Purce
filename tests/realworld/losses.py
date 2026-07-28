"""Loss functions — cross-entropy, focal, huber, contrastive, CTC, label smoothing."""

import numpy as np


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Numerically stable softmax cross-entropy loss per sample."""
    shifted = np.subtract(logits, np.max(logits, axis=-1, keepdims=True))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    neg_log_prob = np.add(shifted, log_sum_exp)
    return np.sum(np.multiply(labels, neg_log_prob), axis=-1)


def sparse_softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy with integer labels."""
    shifted = np.subtract(logits, np.max(logits, axis=-1, keepdims=True))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    batch_idx = np.arange(logits.shape[0])
    neg_log_prob = np.add(shifted[batch_idx, labels], log_sum_exp)
    return neg_log_prob


def focal_loss(logits: np.ndarray, labels: np.ndarray,
               gamma: float = 2.0, alpha: float = 0.25) -> np.ndarray:
    """Focal loss for class imbalance: FL = -alpha * (1-p)^gamma * log(p)."""
    probs = _softmax(logits)
    batch_idx = np.arange(logits.shape[0])
    p_t = probs[batch_idx, labels]
    focal_weight = np.multiply(alpha, np.power(np.subtract(1.0, p_t), gamma))
    ce = np.negative(np.log(np.add(p_t, 1e-7)))
    return np.multiply(focal_weight, ce)


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray,
               delta: float = 1.0) -> np.ndarray:
    """Huber loss: quadratic for small errors, linear for large."""
    diff = np.subtract(y_true, y_pred)
    abs_diff = np.abs(diff)
    quadratic = np.multiply(0.5, np.power(abs_diff, 2))
    linear = np.subtract(abs_diff, np.multiply(0.5, delta))
    return np.where(np.less(abs_diff, delta), quadratic, linear)


def contrastive_loss(embed1: np.ndarray, embed2: np.ndarray,
                     label: np.ndarray, margin: float = 1.0) -> np.ndarray:
    """Contrastive loss for siamese networks."""
    diff = np.subtract(embed1, embed2)
    dist = np.sqrt(np.add(np.sum(np.power(diff, 2), axis=-1), 1e-7))
    pos_loss = np.multiply(label, np.power(dist, 2))
    neg_dist = np.maximum(np.subtract(margin, dist), 0.0)
    neg_loss = np.multiply(np.subtract(1.0, label), np.power(neg_dist, 2))
    return np.multiply(0.5, np.add(pos_loss, neg_loss))


def ctc_loss(log_probs: np.ndarray, targets: np.ndarray,
             input_lengths: np.ndarray, target_lengths: np.ndarray) -> np.ndarray:
    """CTC loss for sequence-to-sequence models (simplified)."""
    batch_size = log_probs.shape[0]
    losses = np.zeros(batch_size)
    for b in range(batch_size):
        T = input_lengths[b]
        C = log_probs.shape[2]
        S = target_lengths[b]
        alpha = np.full((T, 2 * S + 1), -1e30)
        alpha[0, 0] = log_probs[b, 0, targets[b, 0]]
        if S > 1:
            alpha[0, 1] = log_probs[b, 0, targets[b, 1]]
        for t in range(1, T):
            for s in range(2 * S + 1):
                val = alpha[t - 1, s]
                if s > 0:
                    val = np.logaddexp(val, alpha[t - 1, s - 1])
                if s > 1:
                    val = np.logaddexp(val, alpha[t - 1, s - 2])
                char_idx = targets[b, s // 2] if s < 2 * S else 0
                alpha[t, s] = np.add(val, log_probs[b, t, char_idx])
        losses[b] = -np.logaddexp(alpha[T - 1, 2 * S - 1], alpha[T - 1, 2 * S - 2])
    return losses


def label_smoothing_cross_entropy(logits: np.ndarray, labels: np.ndarray,
                                   smoothing: float = 0.1) -> np.ndarray:
    """Cross-entropy with label smoothing."""
    n_classes = logits.shape[-1]
    smoothed = np.full_like(labels, smoothing / n_classes)
    indices = np.arange(logits.shape[0])
    smoothed[indices, labels] = np.subtract(1.0 - smoothing, smoothed[indices, labels])
    return softmax_cross_entropy(logits, smoothed)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) = sum(p * log(p/q))."""
    p_safe = np.clip(p, 1e-7, 1.0)
    q_safe = np.clip(q, 1e-7, 1.0)
    return np.sum(np.multiply(p_safe, np.log(np.divide(p_safe, q_safe))), axis=-1)


def cosine_similarity_loss(embed1: np.ndarray, embed2: np.ndarray,
                           label: np.ndarray) -> np.ndarray:
    """Cosine similarity-based loss."""
    dot = np.sum(np.multiply(embed1, embed2), axis=-1)
    norm1 = np.sqrt(np.add(np.sum(np.power(embed1, 2), axis=-1), 1e-7))
    norm2 = np.sqrt(np.add(np.sum(np.power(embed2, 2), axis=-1), 1e-7))
    cos_sim = np.divide(dot, np.multiply(norm1, norm2))
    return np.mean(np.power(np.subtract(label, cos_sim), 2))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.subtract(x, np.max(x, axis=-1, keepdims=True))
    exp_x = np.exp(shifted)
    return np.divide(exp_x, np.sum(exp_x, axis=-1, keepdims=True))
