"""Generate 25 harsh real-world files — deterministic, reproducible."""
from __future__ import annotations

import os
import random
import textwrap

from .config import HARSH_CORPUS_SEED

# 25 harsh patterns covering the frontier: PDE, large graph, quantization, control ill-conditioned, etc.
HARSH_TEMPLATES = [
    ("harsh_pde_solver.py", '''
import numpy as np
def pde_step(u, dx, dt, alpha):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx*dx)
    return u + alpha * dt * lap
def pde_solve(u0, steps, dx, dt):
    u = np.copy(u0)
    for _ in range(steps):
        u = pde_step(u, dx, dt, 0.1)
    return u
'''),
    ("harsh_large_matmul_chain.py", '''
import numpy as np
def chain_matmul(A, B, C, D):
    t1 = np.matmul(A, B)
    t2 = np.matmul(t1, C)
    return np.matmul(t2, D)
def batch_chain(batch):
    s = np.zeros((64, 64))
    for m in batch:
        s = s + np.matmul(m, m.T)
    return s
'''),
    ("harsh_ill_conditioned_solve.py", '''
import numpy as np
def hilbert_solve(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    b = np.ones(n)
    return np.linalg.solve(H, b)
def cond_solve(A, b):
    return np.linalg.solve(A, b)
'''),
    ("harsh_quantization.py", '''
import numpy as np
def quantize(x, scale, zp):
    q = np.clip(np.round(x / scale + zp), -128, 127)
    return (q - zp) * scale
def dequant_matmul(A_q, B_q, s_a, s_b):
    A = A_q * s_a
    B = B_q * s_b
    return np.matmul(A, B)
'''),
    ("harsh_graph_large.py", '''
import numpy as np
def pagerank(adj, d, iters):
    n = adj.shape[0]
    r = np.ones(n) / n
    for _ in range(iters):
        r = (1-d)/n + d * np.matmul(adj, r)
    return r
def gnn_layer(X, A, W):
    AX = np.matmul(A, X)
    return np.tanh(np.matmul(AX, W))
def large_graph_conv(A, X, Ws):
    for W in Ws:
        X = gnn_layer(X, A, W)
    return X
'''),
    ("harsh_control_kalman_large.py", '''
import numpy as np
def kalman_predict(x, P, F, Q):
    x = np.matmul(F, x)
    P = np.matmul(np.matmul(F, P), F.T) + Q
    return x, P
def kalman_update(x, P, z, H, R):
    y = z - np.matmul(H, x)
    S = np.matmul(np.matmul(H, P), H.T) + R
    K = np.matmul(np.matmul(P, H.T), np.linalg.inv(S))
    x = x + np.matmul(K, y)
    P = P - np.matmul(np.matmul(K, H), P)
    return x, P
def kalman_large(steps, n):
    x = np.zeros(n)
    P = np.eye(n)
    F = np.eye(n) * 0.99
    Q = np.eye(n) * 0.01
    for _ in range(steps):
        x, P = kalman_predict(x, P, F, Q)
    return x
'''),
    ("harsh_transformer_deep.py", '''
import numpy as np
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
def attention(Q, K, V):
    d = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(float(d))
    w = softmax(scores)
    return np.matmul(w, V)
def transformer_block(X, Wq, Wk, Wv, Wo):
    Q = np.matmul(X, Wq)
    K = np.matmul(X, Wk)
    V = np.matmul(X, Wv)
    A = attention(Q, K, V)
    return np.matmul(A, Wo) + X
def deep_transformer(X, layers):
    for Wq, Wk, Wv, Wo in layers:
        X = transformer_block(X, Wq, Wk, Wv, Wo)
    return X
'''),
    ("harsh_fft_large.py", '''
import numpy as np
def fft_chain(x):
    X = np.fft.fft(x)
    Y = np.fft.ifft(X)
    return Y.real
def spectral_filter(x, filt):
    X = np.fft.fft(x)
    Y = X * filt
    return np.fft.ifft(Y).real
def large_fft_batch(batch):
    out = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        out[i] = np.fft.fft(batch[i]).real
    return out
'''),
    ("harsh_ode_stiff.py", '''
import numpy as np
def rk4_step(f, y, t, dt):
    k1 = f(t, y)
    k2 = f(t+dt/2, y + dt*k1/2)
    k3 = f(t+dt/2, y + dt*k2/2)
    k4 = f(t+dt, y + dt*k3)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
def stiff_ode(y0, dt, steps):
    def f(t, y):
        return -1000*y + np.sin(t)
    y = np.copy(y0)
    for i in range(steps):
        y = rk4_step(f, y, float(i)*dt, dt)
    return y
'''),
    ("harsh_sparse_ops.py", '''
import numpy as np
def sparse_matvec(rows, cols, vals, x):
    y = np.zeros_like(x)
    for r, c, v in zip(rows, cols, vals):
        y[int(r)] += v * x[int(c)]
    return y
def sparse_outer(a, b):
    return np.outer(a, b)
def sparse_chain(A, B, x):
    return np.matmul(A, np.matmul(B, x))
'''),
    ("harsh_reduction_stress.py", '''
import numpy as np
def reduction_stress(x):
    s = np.sum(x)
    m = np.mean(x)
    v = np.var(x)
    p = np.prod(x[:8])
    am = np.argmax(x)
    an = np.argmin(x)
    return s + m + v + p + float(am) + float(an)
def cumsum_chain(x):
    c = np.cumsum(x)
    return np.diff(c)
def any_all_chain(x):
    return float(np.any(x > 0)) + float(np.all(x > -1))
'''),
    ("harsh_elementwise_chain.py", '''
import numpy as np
def element_chain(a, b, c):
    t1 = np.add(a, b)
    t2 = np.multiply(t1, c)
    t3 = np.divide(t2, b + 1.0)
    t4 = np.sqrt(np.abs(t3))
    t5 = np.exp(np.clip(t4, -5, 5))
    t6 = np.log(t5 + 1.0)
    return np.tanh(t6)
def trig_chain(x):
    return np.sin(x) + np.cos(x) + np.tan(np.clip(x, -1, 1))
'''),
    ("harsh_linalg_large.py", '''
import numpy as np
def linalg_large(A):
    inv = np.linalg.inv(A)
    det = np.linalg.det(A)
    ch = np.linalg.cholesky(A + np.eye(A.shape[0])*0.1)
    qr = np.linalg.qr(A)
    svd = np.linalg.svd(A)
    return inv, det, ch, qr, svd
def norm_chain(A):
    return np.linalg.norm(A) + np.linalg.norm(A, axis=0).sum()
'''),
    ("harsh_array_ops_stress.py", '''
import numpy as np
def array_stress(x):
    y = np.flip(x)
    y = np.roll(y, 2)
    y = np.tile(y, 2)[:len(x)]
    y = np.repeat(y, 2)[:len(x)*2]
    y = np.concatenate([y, y])
    return np.unique(y)
def take_stress(x, idx):
    return np.take(x, idx)
def split_stress(x):
    parts = np.split(x, 4)
    return np.concatenate(parts)
'''),
    ("harsh_mixed_precision.py", '''
import numpy as np
def mixed_chain(a, b):
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    c = np.add(af, bf)
    return c.astype(np.float64)
def float_int_mix(x):
    y = x.astype(np.int32)
    z = y.astype(np.float64)
    return np.add(x, z)
'''),
    ("harsh_random_stress.py", '''
import numpy as np
def random_stress(n):
    a = np.random.randn(n)
    b = np.random.random(n)
    c = np.random.randint(0, 10, size=n).astype(np.float64)
    d = np.random.uniform(-1, 1, size=n)
    return np.add(a, b) + np.add(c, d)
def permute_stress(x):
    p = np.random.permutation(len(x))
    return np.take(x, p)
'''),
    ("harsh_broadcast_stress.py", '''
import numpy as np
def broadcast_add(A, b):
    return A + b
def broadcast_matmul_chain(A, B, C):
    t = np.matmul(A, B)
    return t + C
def outer_stress(a, b):
    return np.outer(a, b) + np.outer(b, a)
'''),
    ("harsh_cholesky_stress.py", '''
import numpy as np
def cholesky_stress(n):
    A = np.random.randn(n, n)
    S = np.matmul(A, A.T) + np.eye(n)*n
    L = np.linalg.cholesky(S)
    return np.matmul(L, L.T)
def inv_cholesky_chain(A):
    L = np.linalg.cholesky(A)
    inv = np.linalg.inv(A)
    return np.matmul(L, inv)
'''),
    ("harsh_statistics_stress.py", '''
import numpy as np
def stats_stress(x):
    m = np.mean(x)
    v = np.var(x)
    s = np.sum((x - m)**2)
    return s / (len(x) * v + 1e-9)
def moment_chain(x):
    return np.mean(x**2) - np.mean(x)**2
'''),
    ("harsh_signal_stress.py", '''
import numpy as np
def signal_chain(x):
    X = np.fft.fft(x)
    Y = np.fft.ifft(X)
    z = Y.real + Y.imag
    return np.convolve(z[:8], np.ones(3)/3)[:len(z)]
def filter_stress(x, h):
    return np.convolve(x, h)[:len(x)]
'''),
    ("harsh_nested_composition.py", '''
import numpy as np
def nested_deep(a, b, c, d):
    t1 = np.add(np.multiply(a, b), np.divide(c, d + 1.0))
    t2 = np.sqrt(np.abs(t1))
    t3 = np.exp(np.clip(t2, -3, 3))
    t4 = np.sin(t3) + np.cos(t3)
    return np.sum(t4)
def composed_linalg(A, x):
    y = np.linalg.solve(A, x)
    z = np.matmul(A, y)
    return np.linalg.norm(z - x)
'''),
    ("harsh_allocation_stress.py", '''
import numpy as np
def alloc_chain(n):
    a = np.zeros(n)
    b = np.ones(n)
    c = np.eye(n)
    d = np.arange(n, dtype=np.float64)
    e = np.linspace(0, 1, n)
    f = np.full(n, 3.14)
    return a + b + c[0] + d + e + f
def alloc_large(n):
    return np.zeros(n * n) + np.ones(n * n)
'''),
    ("harsh_eig_qr_svd_stub.py", '''
import numpy as np
def eig_qr_svd_stub(A):
    w = np.linalg.eig(A)[0]
    q, r = np.linalg.qr(A)
    u, s, vh = np.linalg.svd(A)
    return w + q[0] + r[0] + u[0]
'''),
    ("harsh_take_gather_stress.py", '''
import numpy as np
def gather_stress(X, idx):
    return np.take(X, idx)
def take_large(X):
    idx = np.argsort(X)
    return np.take(X, idx)
def gather_chain(a, b, idx):
    t = np.concatenate([a, b])
    return np.take(t, idx)
'''),
    ("harsh_sort_unique_stress.py", '''
import numpy as np
def sort_unique_chain(x):
    s = np.sort(x)
    u = np.unique(s)
    a = np.argsort(x)
    t = np.take(x, a)
    return s + u[:len(s)] + t
def searchsorted_stress(x, v):
    return np.searchsorted(np.sort(x), v)
'''),
]

def generate_harsh_corpus(output_dir: str, seed: int = HARSH_CORPUS_SEED) -> list[str]:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    # Deterministic shuffle then take first 25 (already 25)
    templates = sorted(HARSH_TEMPLATES, key=lambda x: x[0])
    written = []
    for fname, content in templates:
        fpath = os.path.join(output_dir, fname)
        # Dedent and ensure reproducible newline
        text = textwrap.dedent(content).lstrip() + "\n"
        with open(fpath, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        written.append(fpath)
    # Also write a manifest with seed
    with open(os.path.join(output_dir, "_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\ncount={len(written)}\n")
    return sorted(written)
