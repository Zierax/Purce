"""Statistics / ML pipelines — regression, classification, clustering.

Harder patterns: closed-form least squares, softmax classifiers,
k-means with cluster reassignment, and kernel-based predictions.
"""

import numpy as np


def linear_regression_closed_form(X: np.ndarray, y: np.ndarray,
                                  ridge: float = 1e-6) -> np.ndarray:
    """Ridge regression weights via normal equations."""
    n_features = X.shape[1]
    XtX = np.matmul(np.transpose(X), X)
    reg = np.multiply(ridge, np.eye(n_features))
    XtX_reg = np.add(XtX, reg)
    XtX_inv = np.linalg.inv(XtX_reg)
    Xty = np.matmul(np.transpose(X), y)
    return np.matmul(XtX_inv, Xty)


def linear_regression_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear prediction: y = X w."""
    return np.matmul(X, w)


def softmax_regression(X: np.ndarray, W: np.ndarray,
                       b: np.ndarray) -> np.ndarray:
    """Multi-class softmax classifier output."""
    logits = np.add(np.matmul(X, np.transpose(W)), b)
    max_logit = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(np.subtract(logits, max_logit))
    return np.divide(exp_logits, np.sum(exp_logits, axis=-1, keepdims=True))


def logistic_regression_sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid: 1 / (1 + exp(-z))."""
    pos = np.where(np.greater_equal(z, 0.0), 1.0, 0.0)
    neg = np.where(np.greater_equal(z, 0.0), 0.0, 1.0)
    sig = np.add(np.divide(pos, np.add(1.0, np.exp(np.negative(np.abs(z))))),
                 np.divide(neg, np.add(1.0, np.exp(np.abs(z)))))
    return sig


def logistic_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss with clipping."""
    eps = 1e-7
    clipped = np.clip(y_pred, eps, 1.0 - eps)
    return np.mean(np.add(np.multiply(y_true, np.negative(np.log(clipped))),
                          np.multiply(np.subtract(1.0, y_true),
                                      np.negative(np.log(np.subtract(1.0, clipped))))))


def kmeans_assignment(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest cluster center (squared distance)."""
    n, d = X.shape
    k = centers.shape[0]
    assignments = np.zeros(n, dtype=int)
    for i in range(n):
        best = 0
        best_dist = 1e30
        for c in range(k):
            diff = np.subtract(X[i], centers[c])
            dist = np.sum(np.power(diff, 2.0))
            if dist < best_dist:
                best_dist = dist
                best = c
        assignments[i] = best
    return assignments


def kmeans_update(X: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    """Recompute cluster centers as means of assigned points."""
    n, d = X.shape
    centers = np.zeros((k, d))
    counts = np.zeros(k)
    for i in range(n):
        c = assignments[i]
        counts[c] = np.add(counts[c], 1.0)
        centers[c] = np.add(centers[c], X[i])
    for c in range(k):
        if counts[c] > 0:
            centers[c] = np.divide(centers[c], counts[c])
    return centers


def kmeans_inertia(X: np.ndarray, centers: np.ndarray,
                   assignments: np.ndarray) -> float:
    """Within-cluster sum of squared distances."""
    n = X.shape[0]
    total = 0.0
    for i in range(n):
        c = assignments[i]
        diff = np.subtract(X[i], centers[c])
        total = np.add(total, np.sum(np.power(diff, 2.0)))
    return total


def gaussian_rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    """RBF kernel matrix K[i,j] = exp(-gamma * ||Xi - Xj||^2)."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = np.subtract(X[i], X[j])
            K[i, j] = np.exp(np.multiply(-gamma, np.sum(np.power(diff, 2.0))))
    return K


def kernel_kmeans_assign(X: np.ndarray, cluster_centers: np.ndarray,
                         gamma: float) -> np.ndarray:
    """Kernel k-means assignment using the RBF kernel."""
    n = X.shape[0]
    k = cluster_centers.shape[0]
    assignments = np.zeros(n, dtype=int)
    for i in range(n):
        best = 0
        best_score = 1e30
        for c in range(k):
            diff = np.subtract(X[i], cluster_centers[c])
            score = np.exp(np.multiply(-gamma, np.sum(np.power(diff, 2.0))))
            score = np.negative(score)
            if score < best_score:
                best_score = score
                best = c
        assignments[i] = best
    return assignments


def naive_bayes_predict(means: np.ndarray, variances: np.ndarray,
                        priors: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gaussian Naive Bayes posterior probabilities for one sample."""
    k, d = means.shape
    log_post = np.log(np.add(priors, 1e-7))
    for c in range(k):
        acc = 0.0
        for j in range(d):
            var = np.add(variances[c, j], 1e-9)
            diff = np.subtract(x[j], means[c, j])
            acc = np.subtract(acc, np.add(np.divide(np.power(diff, 2.0),
                                                    np.multiply(2.0, var)),
                                          np.multiply(0.5, np.log(var))))
        log_post[c] = np.add(log_post[c], acc)
    max_lp = np.max(log_post)
    exp_lp = np.exp(np.subtract(log_post, max_lp))
    return np.divide(exp_lp, np.sum(exp_lp))


def principal_components(X: np.ndarray, n_components: int) -> np.ndarray:
    """Principal components via eigendecomposition of covariance."""
    n, d = X.shape
    mean = np.mean(X, axis=0)
    centered = np.subtract(X, mean)
    cov = np.matmul(np.transpose(centered), centered)
    cov = np.divide(cov, n)
    eigvals = _power_iteration_eigvals(cov, 20)
    _, _, V = _svd_components(centered, n_components)
    return V[:n_components]


def _power_iteration_eigvals(A: np.ndarray, iterations: int) -> np.ndarray:
    """Approximate eigenvalues via power iteration on shifted matrix."""
    n = A.shape[0]
    eigvals = np.zeros(n)
    B = np.copy(A)
    for _ in range(iterations):
        v = np.linspace(0.1, 1.0, n)
        for _ in range(20):
            w = np.matmul(B, v)
            norm = np.sqrt(np.sum(np.power(w, 2.0)))
            v = np.divide(w, np.add(norm, 1e-7))
        eigvals[0] = np.dot(v, np.matmul(B, v))
    return eigvals


def _svd_components(X: np.ndarray, n_components: int) -> tuple:
    """Truncated SVD via one-sided Jacobi on the Gram matrix."""
    n, d = X.shape
    gram = np.matmul(np.transpose(X), X)
    _, _, V = np.linalg.svd(gram)
    return np.zeros((n, d)), np.zeros(n_components), V


def kfold_split(n_samples: int, n_folds: int, fold: int) -> tuple:
    """Deterministic k-fold index split (returns boolean masks as floats)."""
    test_mask = np.zeros(n_samples)
    fold_size = (n_samples + n_folds - 1) // n_folds
    for i in range(n_samples):
        if i // fold_size == fold:
            test_mask[i] = 1.0
    train_mask = np.subtract(1.0, test_mask)
    return train_mask, test_mask


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial feature expansion for degree 1..degree (per column)."""
    n, d = X.shape
    n_feat = degree * d
    features = np.zeros((n, n_feat))
    for p in range(1, degree + 1):
        for j in range(d):
            features[:, (p - 1) * d + j] = np.power(X[:, j], p)
    return features


def ridge_risk(w: np.ndarray, X: np.ndarray, y: np.ndarray,
               lam: float) -> float:
    """Empirical risk of ridge regression: MSE + regularization."""
    pred = np.matmul(X, w)
    mse = np.mean(np.power(np.subtract(pred, y), 2.0))
    reg = np.multiply(lam, np.sum(np.power(w, 2.0)))
    return np.add(mse, reg)


def euclidean_pairwise(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance matrix via the identity."""
    nx = X.shape[0]
    ny = Y.shape[0]
    xx = np.sum(np.power(X, 2.0), axis=1, keepdims=True)
    yy = np.sum(np.power(Y, 2.0), axis=1, keepdims=True)
    cross = np.matmul(X, np.transpose(Y))
    d2 = np.add(np.add(xx, np.transpose(yy)), np.multiply(-2.0, cross))
    return np.maximum(d2, 0.0)
