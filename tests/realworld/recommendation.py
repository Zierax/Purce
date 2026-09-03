"""Recommendation and ranking patterns: CF, embeddings, NCF, BPR."""

import numpy as np


def cosine_similarity_matrix(X):
    norms = np.sqrt(np.sum(np.power(X, 2), axis=1, keepdims=True))
    norms = np.maximum(norms, 1e-8)
    X_norm = np.divide(X, norms)
    return np.matmul(X_norm, np.transpose(X_norm))


def matrix_factorization_loss(R, P, Q, lambda_reg=0.01):
    mask = np.greater(R, 0).astype(float)
    predictions = np.matmul(P, np.transpose(Q))
    error = np.multiply(mask, np.subtract(R, predictions))
    loss = np.mean(np.power(error, 2))
    reg = np.multiply(lambda_reg, np.add(np.sum(np.power(P, 2)), np.sum(np.power(Q, 2))))
    return np.add(loss, reg)


def bpr_loss(pos_scores, neg_scores):
    return np.negative(
        np.mean(np.log(np.add(np.sigmoid(np.subtract(pos_scores, neg_scores)), 1e-8)))
    )


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = np.sum(np.power(np.subtract(anchor, positive), 2))
    neg_dist = np.sum(np.power(np.subtract(anchor, negative), 2))
    return np.maximum(0.0, np.add(np.subtract(pos_dist, neg_dist), margin))


def item_cf_scores(user_items, item_similarity):
    user_profile = np.matmul(user_items, item_similarity)
    return np.divide(user_profile, np.add(np.sum(np.abs(user_profile)), 1e-8))


def implicit_als_loss(R, X, Y, alpha=40):
    C = np.multiply(alpha, np.greater(R, 0).astype(float))
    P = np.greater(R, 0).astype(float)
    X_loss = np.sum(np.multiply(C, np.power(np.subtract(P, np.matmul(X, np.transpose(Y))), 2)))
    reg = np.add(np.sum(np.power(X, 2)), np.sum(np.power(Y, 2)))
    return np.add(X_loss, reg)


def listwise_loss(scores, labels):
    probs = np.exp(scores) / np.sum(np.exp(scores))
    return np.negative(np.sum(np.multiply(labels, np.log(np.add(probs, 1e-8)))))


def pointwise_loss(score, label, margin=1.0):
    return np.maximum(0.0, np.subtract(margin, np.multiply(label, score)))


def neural_cf_forward(user_emb, item_emb, W1, W2, b1, b2):
    x = np.concatenate([user_emb, item_emb])
    h = np.maximum(0, np.add(np.matmul(W1, x), b1))
    return np.add(np.matmul(W2, h), b2)
