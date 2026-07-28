"""Optimizers — Adam, AdamW, LAMB, Lion, SOAP, AdaFactor."""

import numpy as np


def adam_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8) -> tuple:
    """Adam optimizer: adaptive moment estimation."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1 - beta2, np.power(grads, 2)))
    m_hat = np.divide(m_new, np.subtract(1.0, np.power(beta1, t)))
    v_hat = np.divide(v_new, np.subtract(1.0, np.power(beta2, t)))
    params_new = np.subtract(params, np.multiply(lr, np.divide(m_hat, np.add(np.sqrt(v_hat), eps))))
    return params_new, m_new, v_new


def adamw_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                 t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01) -> tuple:
    """AdamW: Adam with decoupled weight decay."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1 - beta2, np.power(grads, 2)))
    m_hat = np.divide(m_new, np.subtract(1.0, np.power(beta1, t)))
    v_hat = np.divide(v_new, np.subtract(1.0, np.power(beta2, t)))
    params_new = np.subtract(params, np.multiply(lr,
        np.add(np.divide(m_hat, np.add(np.sqrt(v_hat), eps)),
               np.multiply(weight_decay, params))))
    return params_new, m_new, v_new


def lamb_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-6, weight_decay: float = 0.01) -> tuple:
    """LAMB: Layer-wise Adaptive Moments for batch training."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1 - beta2, np.power(grads, 2)))
    m_hat = np.divide(m_new, np.subtract(1.0, np.power(beta1, t)))
    v_hat = np.divide(v_new, np.subtract(1.0, np.power(beta2, t)))
    update = np.add(np.divide(m_hat, np.add(np.sqrt(v_hat), eps)),
                    np.multiply(weight_decay, params))
    params_norm = np.sqrt(np.add(np.sum(np.power(params, 2)), 1e-7))
    update_norm = np.sqrt(np.add(np.sum(np.power(update, 2)), 1e-7))
    params_new = np.subtract(params, np.multiply(lr,
        np.multiply(np.divide(params_norm, update_norm), update)))
    return params_new, m_new, v_new


def lion_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray,
                t: int, lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.99,
                weight_decay: float = 0.0) -> tuple:
    """Lion optimizer: sign-based with momentum."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1 - beta1, grads))
    update = np.sign(m_new)
    params_new = np.subtract(
        np.multiply(np.subtract(1.0, np.multiply(lr, weight_decay)), params),
        np.multiply(lr, update))
    m_decay = np.add(np.multiply(beta2, m), np.multiply(1 - beta2, grads))
    return params_new, m_decay


def soap_update(params: np.ndarray, grads: np.ndarray, m: np.ndarray, v: np.ndarray,
                t: int, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8, precondition_frequency: int = 10) -> tuple:
    """SOAP: Shampoo-like optimizer with Adam updates."""
    m_new = np.add(np.multiply(beta1, m), np.multiply(1 - beta1, grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(1 - beta2, np.power(grads, 2)))
    if t % precondition_frequency == 0:
        precond = np.sqrt(np.add(v_new, eps))
    else:
        precond = np.ones_like(params)
    m_hat = np.divide(m_new, np.subtract(1.0, np.power(beta1, t)))
    params_new = np.subtract(params, np.multiply(lr, np.divide(m_hat, precond)))
    return params_new, m_new, v_new


def adafactor_update(params: np.ndarray, grads: np.ndarray, r: np.ndarray, c: np.ndarray,
                     t: int, lr: float = 0.001, beta1: float = 0.0,
                     beta2: float = 0.999, eps: float = 1e-30,
                     decay_rate: float = -0.8) -> tuple:
    """AdaFactor: memory-efficient optimizer using factored second moments."""
    if beta1 > 0:
        r_new = np.add(np.multiply(beta1, r), np.multiply(1 - beta1, grads))
    else:
        r_new = grads

    v_row = np.mean(np.power(r_new, 2), axis=-1)
    v_col = np.mean(np.power(r_new, 2), axis=0)

    update = np.divide(r_new, np.add(np.sqrt(np.outer(v_row, v_col)), eps))

    if t > 1:
        factor = np.minimum(np.power(t, decay_rate), 0.1)
    else:
        factor = 1.0

    params_new = np.subtract(params, np.multiply(np.multiply(lr, factor), update))
    return params_new, r_new, c
