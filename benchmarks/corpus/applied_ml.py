"""Applied ML / scientific computing — GP kernels, clustering, feature math.

Harder patterns: pair-sum kernels, determinant ratios, information
gains, and mixed integer/float index arithmetic.
"""

import numpy as np


def gaussian_process_mean(kernel: np.ndarray, y: np.ndarray,
                          alpha: np.ndarray) -> np.ndarray:
    """Posterior mean of a GP at training points: mu = K alpha."""
    return np.matmul(kernel, alpha)


def gaussian_process_var(kernel_self: np.ndarray, k_x: np.ndarray,
                         K_inv: np.ndarray) -> np.ndarray:
    """Predictive variance: sigma^2 = k(x,x) - k_x^T K^{-1} k_x."""
    kx = np.matmul(K_inv, k_x)
    return np.subtract(kernel_self, np.dot(k_x, kx))


def mll_gradient(K: np.ndarray, y: np.ndarray, alpha: np.ndarray,
                 K_inv: np.ndarray) -> np.ndarray:
    """Negative log marginal likelihood gradient w.r.t. kernel params."""
    K_inv_y = np.matmul(K_inv, y)
    gram = np.subtract(np.outer(alpha, alpha), K_inv)
    n = K.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc = np.add(acc, np.multiply(gram[i, j], K[i, j]))
        grad[i] = np.multiply(acc, 0.5)
    return grad


def mahalanobis_distance(x: np.ndarray, mu: np.ndarray,
                         cov_inv: np.ndarray) -> float:
    """Mahalanobis distance: sqrt((x-mu)^T C^{-1} (x-mu))."""
    diff = np.subtract(x, mu)
    return np.sqrt(np.dot(diff, np.matmul(cov_inv, diff)))


def gaussian_mixture_log_likelihood(x: np.ndarray, means: np.ndarray,
                                    variances: np.ndarray,
                                    weights: np.ndarray) -> np.ndarray:
    """Log-likelihood of samples under a diagonal Gaussian mixture."""
    n, d = x.shape
    k = means.shape[0]
    ll = np.zeros(n)
    for i in range(n):
        acc = -1e30
        for c in range(k):
            log_g = 0.0
            for j in range(d):
                var = np.add(variances[c, j], 1e-9)
                diff = np.subtract(x[i, j], means[c, j])
                log_g = np.subtract(log_g, np.add(np.divide(np.power(diff, 2.0),
                                                            np.multiply(2.0, var)),
                                                  np.multiply(0.5, np.log(var))))
            term = np.add(np.log(np.add(weights[c], 1e-12)), log_g)
            acc = np.maximum(acc, term)
        ll[i] = acc
    return ll


def information_gain(labels: np.ndarray, mask: np.ndarray) -> float:
    """Information gain of a split on a binary label vector."""
    n_total = labels.shape[0]
    n_left = np.sum(mask)
    n_right = np.subtract(n_total, n_left)
    def _entropy(subset: np.ndarray) -> float:
        size = subset.shape[0]
        if size == 0:
            return 0.0
        pos = np.sum(subset) / size
        if pos <= 0.0 or pos >= 1.0:
            return 0.0
        return np.negative(np.add(np.multiply(pos, np.log(pos)),
                                  np.multiply(np.subtract(1.0, pos), np.log(np.subtract(1.0, pos)))))
    h_parent = _entropy(labels)
    w_left = np.divide(n_left, n_total)
    w_right = np.divide(n_right, n_total)
    left_masked = np.multiply(labels, mask)
    right_masked = np.multiply(labels, np.subtract(1.0, mask))
    h_left = _entropy(left_masked)
    h_right = _entropy(right_masked)
    h_after = np.add(np.multiply(w_left, h_left),
                     np.multiply(w_right, h_right))
    return np.subtract(h_parent, h_after)


def gini_impurity(groups: np.ndarray, labels: np.ndarray) -> float:
    """Weighted Gini impurity across class-group partitions."""
    n = labels.shape[0]
    total = 0.0
    for g in range(groups.shape[1]):
        mask = groups[:, g]
        size = np.sum(mask)
        if size == 0:
            continue
        p = np.divide(np.sum(np.multiply(labels, mask)), size)
        gini = np.multiply(2.0, np.multiply(p, np.subtract(1.0, p)))
        total = np.add(total, np.multiply(np.divide(size, n), gini))
    return total


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix with L2 row normalization."""
    norms = np.sqrt(np.add(np.sum(np.power(X, 2.0), axis=1, keepdims=True), 1e-9))
    Xn = np.divide(X, norms)
    return np.matmul(Xn, np.transpose(Xn))


def rbf_kernel_grad(X: np.ndarray, gamma: float) -> np.ndarray:
    """Gradient of the RBF kernel matrix w.r.t. gamma."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d2 = np.sum(np.power(np.subtract(X[i], X[j]), 2.0))
            K[i, j] = np.exp(np.multiply(-gamma, d2))
    return np.multiply(K, -1.0)


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample softmax cross-entropy (numerically stable)."""
    max_l = np.max(logits, axis=-1, keepdims=True)
    shifted = np.subtract(logits, max_l)
    log_softmax = np.subtract(shifted, np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True)))
    return np.negative(np.sum(np.multiply(labels, log_softmax), axis=-1))


def focal_loss(probabilities: np.ndarray, labels: np.ndarray,
               gamma: float) -> np.ndarray:
    """Focal loss for imbalanced classification."""
    pt = np.where(np.equal(labels, 1.0), probabilities, np.subtract(1.0, probabilities))
    pt = np.clip(pt, 1e-7, 1.0 - 1e-7)
    return np.negative(np.multiply(np.multiply(np.power(np.subtract(1.0, pt), gamma),
                                               labels), np.log(pt)))


def spectral_clustering_affinity(X: np.ndarray, sigma: float) -> np.ndarray:
    """Affinity matrix for spectral clustering."""
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d2 = np.sum(np.power(np.subtract(X[i], X[j]), 2.0))
            W[i, j] = np.exp(np.negative(np.divide(d2, np.multiply(2.0, np.multiply(sigma, sigma)))))
    return W


def log_sum_exp_2d(M: np.ndarray) -> np.ndarray:
    """Log-sum-exp over the last axis (row-wise)."""
    m = np.max(M, axis=-1, keepdims=True)
    return np.add(m, np.log(np.sum(np.exp(np.subtract(M, m)), axis=-1)))


def soft_threshold_shrinkage(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding operator (lasso proximal)."""
    return np.multiply(np.sign(x),
                       np.maximum(np.subtract(np.abs(x), threshold), 0.0))


def group_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               n_groups: int, eps: float = 1e-5) -> np.ndarray:
    """Group normalization over the channel dimension."""
    n, c = x.shape
    g_size = c // n_groups
    out = np.zeros_like(x)
    for i in range(n):
        for g in range(n_groups):
            start = g * g_size
            end = start + g_size
            block = x[i, start:end]
            mean = np.mean(block)
            var = np.mean(np.power(np.subtract(block, mean), 2.0))
            out[i, start:end] = np.add(np.multiply(
                np.divide(np.subtract(block, mean), np.sqrt(np.add(var, eps))),
                gamma[start:end]), beta[start:end])
    return out
