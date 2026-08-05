"""Graph algorithms — adjacency transforms, PageRank, shortest paths.

Harder patterns: loop-based relaxation, normalization with degree
matrices, and iterative convergence with error tracking.
"""

import numpy as np


def adjacency_laplacian(A: np.ndarray) -> np.ndarray:
    """Graph Laplacian: L = D - A (D = degree matrix)."""
    n = A.shape[0]
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return np.subtract(D, A)


def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    """Symmetric normalized Laplacian: I - D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    degrees = np.sum(A, axis=1)
    d_inv_sqrt = np.divide(1.0, np.sqrt(np.add(degrees, 1e-7)))
    D = np.diag(d_inv_sqrt)
    L = np.subtract(np.eye(n),
                    np.matmul(np.matmul(D, A), D))
    return L


def pagerank(A: np.ndarray, damping: float = 0.85,
             max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank via power iteration on the column-stochastic matrix."""
    n = A.shape[0]
    out_deg = np.add(np.sum(A, axis=1), 1e-9)
    P = np.divide(A, out_deg[:, np.newaxis])
    rank = np.full(n, np.divide(1.0, n))
    teleport = np.divide(np.subtract(1.0, damping), n)
    for _ in range(max_iter):
        new_rank = np.add(np.multiply(damping, np.matmul(rank, P)),
                          np.full(n, teleport))
        delta = np.max(np.abs(np.subtract(new_rank, rank)))
        rank = new_rank
        if delta < tol:
            break
    return rank


def floyd_warshall(A: np.ndarray) -> np.ndarray:
    """All-pairs shortest paths via Floyd-Warshall."""
    n = A.shape[0]
    dist = np.copy(A)
    for i in range(n):
        for j in range(n):
            if dist[i, j] == 0.0 and i != j:
                dist[i, j] = 1e30
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cand = np.add(dist[i, k], dist[k, j])
                if cand < dist[i, j]:
                    dist[i, j] = cand
    return dist


def bellman_ford(A: np.ndarray, source: int) -> np.ndarray:
    """Bellman-Ford single-source shortest paths (loop relaxation)."""
    n = A.shape[0]
    dist = np.full(n, 1e30)
    dist[source] = 0.0
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if A[u, v] > 0.0 and A[u, v] < 1e30:
                    cand = np.add(dist[u], A[u, v])
                    if cand < dist[v]:
                        dist[v] = cand
    return dist


def graph_convolution(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """One-hop graph convolution: H = A_hat X W (A_hat normalized)."""
    n = A.shape[0]
    degrees = np.add(np.sum(A, axis=1), 1e-9)
    d_inv = np.divide(1.0, np.sqrt(degrees))
    A_hat = np.multiply(A, np.multiply(d_inv[:, np.newaxis], d_inv[np.newaxis, :]))
    W = np.eye(X.shape[1])
    return np.matmul(np.matmul(A_hat, X), W)


def graph_clustering_coefficient(A: np.ndarray) -> np.ndarray:
    """Local clustering coefficient per node: 2T / (d(d-1))."""
    n = A.shape[0]
    A2 = np.matmul(A, A)
    triangles = np.diag(A2)
    degrees = np.sum(A, axis=1)
    denom = np.multiply(degrees, np.subtract(degrees, 1.0))
    coeff = np.divide(np.multiply(2.0, triangles), np.add(denom, 1e-9))
    return coeff


def eigenvector_centrality(A: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Eigenvector centrality via power iteration."""
    n = A.shape[0]
    x = np.full(n, 1.0)
    for _ in range(max_iter):
        x_new = np.matmul(A, x)
        norm = np.sqrt(np.sum(np.power(x_new, 2.0)))
        x_new = np.divide(x_new, np.add(norm, 1e-7))
        if np.max(np.abs(np.subtract(x_new, x))) < 1e-8:
            x = x_new
            break
        x = x_new
    return x


def katz_centrality(A: np.ndarray, alpha: float, max_iter: int = 100) -> np.ndarray:
    """Katz centrality: x = (I - alpha A^T)^{-1} 1."""
    n = A.shape[0]
    I = np.eye(n)
    M = np.subtract(I, np.multiply(alpha, np.transpose(A)))
    ones = np.ones(n)
    return np.linalg.solve(M, ones)


def betweenness_approx(A: np.ndarray, n_samples: int) -> np.ndarray:
    """Approximate betweenness via sampled shortest paths (loop-heavy)."""
    n = A.shape[0]
    betweenness = np.zeros(n)
    for s in range(n_samples):
        dist = np.full(n, 1e30)
        dist[s] = 0.0
        for _ in range(n - 1):
            for u in range(n):
                for v in range(n):
                    if A[u, v] > 0.0:
                        cand = np.add(dist[u], A[u, v])
                        if cand < dist[v]:
                            dist[v] = cand
        for v in range(n):
            if v != s and dist[v] < 1e29:
                betweenness[v] = np.add(betweenness[v], 1.0)
    return np.divide(betweenness, n_samples)


def graph_diameter(A: np.ndarray) -> float:
    """Diameter of the graph from all-pairs shortest paths."""
    dist = floyd_warshall(A)
    n = A.shape[0]
    max_dist = 0.0
    for i in range(n):
        for j in range(n):
            if dist[i, j] < 1e29 and dist[i, j] > max_dist:
                max_dist = dist[i, j]
    return max_dist


def degree_distribution(A: np.ndarray, max_degree: int) -> np.ndarray:
    """Histogram of node degrees."""
    n = A.shape[0]
    degrees = np.sum(A, axis=1).astype(int)
    hist = np.zeros(max_degree + 1)
    for i in range(n):
        d = int(degrees[i])
        if d <= max_degree:
            hist[d] = np.add(hist[d], 1.0)
    return np.divide(hist, n)


def spectral_gap(A: np.ndarray) -> float:
    """Spectral gap from the two largest eigenvalues (power iteration)."""
    n = A.shape[0]
    x = np.full(n, 1.0)
    for _ in range(50):
        x = np.matmul(A, x)
        norm = np.sqrt(np.sum(np.power(x, 2.0)))
        x = np.divide(x, np.add(norm, 1e-7))
    lambda1 = np.dot(x, np.matmul(A, x))
    B = np.subtract(A, np.multiply(lambda1, np.outer(x, x)))
    y = np.full(n, 1.0)
    for _ in range(50):
        y = np.matmul(B, y)
        norm = np.sqrt(np.sum(np.power(y, 2.0)))
        y = np.divide(y, np.add(norm, 1e-7))
    lambda2 = np.dot(y, np.matmul(B, y))
    return np.subtract(lambda1, lambda2)
