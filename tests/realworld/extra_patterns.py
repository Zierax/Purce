"""Additional real-world test sources: transformer, RNN, convolution patterns."""

TRANSFORMER_SELF_ATTENTION = """\
import numpy as np

def self_attention(Q, K, V, scale):
    scores = np.matmul(Q, np.transpose(K))
    scaled = np.divide(scores, scale)
    weights = np.exp(scaled)
    sum_weights = np.sum(weights)
    normalized = np.divide(weights, sum_weights)
    return np.matmul(normalized, V)
"""

RNN_CELL = """\
import numpy as np

def rnn_step(x, h_prev, W_xh, W_hh, b_h):
    z1 = np.matmul(x, W_xh)
    z2 = np.matmul(h_prev, W_hh)
    h_t = np.tanh(np.add(np.add(z1, z2), b_h))
    return h_t
"""

CONV_1D = """\
import numpy as np

def conv1d(x, w, b, stride):
    n_out = int(np.divide(np.subtract(len(x), len(w)), stride)) + 1
    out = np.zeros(n_out)
    for i in range(n_out):
        s = 0.0
        for j in range(len(w)):
            s += x[i * stride + j] * w[j]
        out[i] = s + b
    return out
"""

BATCH_NORM = """\
import numpy as np

def batch_norm(x, gamma, beta, eps):
    mean = np.mean(x)
    variance = np.mean(np.power(np.subtract(x, mean), 2.0))
    inv_std = np.divide(1.0, np.sqrt(np.add(variance, eps)))
    normalized = np.multiply(np.subtract(x, mean), inv_std)
    return np.add(np.multiply(gamma, normalized), beta)
"""

LABEL_SMOOTHING_CE = """\
import numpy as np

def label_smoothing_ce(logits, targets, num_classes, smoothing):
    confidence = np.subtract(1.0, smoothing)
    low_val = np.divide(smoothing, np.subtract(num_classes, 1.0))
    one_hot = np.multiply(np.ones(num_classes), low_val)
    one_hot[targets] = confidence
    log_probs = np.log(np.add(np.exp(logits), 1e-7))
    loss = np.negative(np.sum(np.multiply(one_hot, log_probs)))
    return loss
"""

COSINE_ANNEALING = """\
import numpy as np

def cosine_annealing(epoch, max_epochs, lr_min, lr_max):
    return np.add(lr_min, np.multiply(
        np.divide(np.subtract(lr_max, lr_min), 2.0),
        np.add(1.0, np.cos(np.divide(np.multiply(np.pi, epoch), max_epochs)))
    ))
"""

GROUP_NORM = """\
import numpy as np

def group_norm(x, gamma, beta, num_groups, eps):
    group_size = int(np.divide(len(x), num_groups))
    out = np.zeros_like(x)
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group = x[start:end]
        mean = np.mean(group)
        var = np.mean(np.power(np.subtract(group, mean), 2.0))
        normed = np.divide(np.subtract(group, mean), np.sqrt(np.add(var, eps)))
        out[start:end] = np.add(np.multiply(gamma[start:end], normed), beta[start:end])
    return out
"""

FOCAL_LOSS = """\
import numpy as np

def focal_loss(logits, targets, gamma, alpha):
    probs = np.exp(logits)
    p_t = np.where(targets == 1, probs, np.subtract(1.0, probs))
    alpha_t = np.where(targets == 1, alpha, np.subtract(1.0, alpha))
    focal_weight = np.power(np.subtract(1.0, p_t), gamma)
    loss = np.negative(np.multiply(alpha_t, np.multiply(focal_weight, np.log(np.add(p_t, 1e-7)))))
    return np.mean(loss)
"""

MIXUP_AUGMENTATION = """\
import numpy as np

def mixup(x1, x2, y1, y2, lam):
    mixed_x = np.add(np.multiply(lam, x1), np.multiply(np.subtract(1.0, lam), x2))
    mixed_y = np.add(np.multiply(lam, y1), np.multiply(np.subtract(1.0, lam), y2))
    return mixed_x, mixed_y
"""

ADAMW_STEP = """\
import numpy as np

def adamw_step(params, grads, m, v, t, lr, beta1, beta2, eps, weight_decay):
    m_new = np.add(np.multiply(beta1, m), np.multiply(np.subtract(1.0, beta1), grads))
    v_new = np.add(np.multiply(beta2, v), np.multiply(np.subtract(1.0, beta2), np.power(grads, 2.0)))
    m_hat = np.divide(m_new, np.subtract(1.0, np.power(beta1, t)))
    v_hat = np.divide(v_new, np.subtract(1.0, np.power(beta2, t)))
    update = np.add(np.multiply(weight_decay, params), np.divide(m_hat, np.add(np.sqrt(v_hat), eps)))
    return np.subtract(params, np.multiply(lr, update))
"""

ALL_EXTRA_SOURCES = [
    ("transformer_self_attention", TRANSFORMER_SELF_ATTENTION),
    ("rnn_cell", RNN_CELL),
    ("conv1d", CONV_1D),
    ("batch_norm", BATCH_NORM),
    ("label_smoothing_ce", LABEL_SMOOTHING_CE),
    ("cosine_annealing", COSINE_ANNEALING),
    ("group_norm", GROUP_NORM),
    ("focal_loss", FOCAL_LOSS),
    ("mixup_augmentation", MIXUP_AUGMENTATION),
    ("adamw_step", ADAMW_STEP),
]
