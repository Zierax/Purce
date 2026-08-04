"""Graph neural network patterns: message passing, graph convolution, attention."""

import numpy as np


def graph_convolution(A_hat, X, W):
    return np.maximum(0, np.matmul(A_hat, np.matmul(X, W)))


def message_passing(messages, edge_index):
    aggregated = np.zeros((messages.shape[0], messages.shape[1]))
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        aggregated[dst] = np.add(aggregated[dst], messages[src])
    return aggregated


def graph_attention(A, X, W_query, W_key, W_value):
    Q = np.matmul(X, W_query)
    K = np.matmul(X, W_key)
    V = np.matmul(X, W_value)
    d_k = K.shape[-1]
    scores = np.divide(np.matmul(Q, np.transpose(K)), np.sqrt(d_k))
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.matmul(weights, V)


def graph_norm(X, epsilon=1e-6):
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return np.divide(np.subtract(X, mean), np.sqrt(np.add(var, epsilon)))


def positional_encoding_2d(H, W, d_model):
    pe = np.zeros((H * W, d_model))
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            for k in range(0, d_model, 2):
                div_term = np.exp(np.multiply(k, -(np.log(10000.0) / d_model)))
                pe[idx, k] = np.sin(np.multiply(np.add(i, j), div_term))
                pe[idx, k + 1] = np.cos(np.multiply(np.add(i, j), div_term))
    return pe


def edge_features(X, edge_index):
    src_features = X[edge_index[0]]
    dst_features = X[edge_index[1]]
    return np.subtract(src_features, dst_features)


def graph_pooling(X, clusters):
    n_clusters = np.max(clusters) + 1
    pooled = np.zeros((n_clusters, X.shape[1]))
    for i in range(X.shape[0]):
        pooled[clusters[i]] = np.add(pooled[clusters[i]], X[i])
    counts = np.zeros(n_clusters)
    for c in clusters:
        counts[c] += 1
    return np.divide(pooled, np.add(counts.reshape(-1, 1), 1e-8))


def graph_resnet(X, A_hat, W1, W2):
    h = np.maximum(0, np.matmul(A_hat, np.matmul(X, W1)))
    h2 = np.matmul(A_hat, np.matmul(h, W2))
    return np.maximum(0, np.add(X, h2))
