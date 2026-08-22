"""Generate 100 chaos programs — random Python AST that uses np.* in wild ways."""
from __future__ import annotations

import os
import random
import textwrap

from .config import CHAOS_CORPUS_SEED

# Pool of numpy ops to sample from — covers all 92 kernels plus some invalid
NUMPY_OPS = [
    "np.add(a, b)", "np.subtract(a, b)", "np.multiply(a, b)", "np.divide(a, b)",
    "np.sqrt(np.abs(a))", "np.exp(np.clip(a, -5, 5))", "np.log(np.abs(a)+1)",
    "np.sin(a)", "np.cos(a)", "np.tan(np.clip(a, -1, 1))", "np.tanh(a)",
    "np.maximum(a, b)", "np.minimum(a, b)", "np.power(a, b)", "np.where(a>0, a, b)",
    "np.clip(a, -1, 1)", "np.negative(a)", "np.sign(a)", "np.floor(a)", "np.ceil(a)",
    "np.trunc(a)", "np.round(a)", "np.greater(a, b)", "np.less(a, b)",
    "np.sum(a)", "np.mean(a)", "np.max(a)", "np.min(a)", "np.var(a)", "np.prod(a[:4])",
    "np.argmax(a)", "np.argmin(a)", "np.any(a>0)", "np.all(a>0)",
    "np.matmul(A, B)", "np.dot(A, B)", "np.linalg.solve(A, b)", "np.linalg.inv(A)",
    "np.linalg.cholesky(A+np.eye(4)*4)", "np.linalg.norm(a)", "np.linalg.det(A)",
    "np.fft.fft(a)", "np.fft.ifft(a)",
    "np.zeros(8)", "np.ones(8)", "np.eye(4)", "np.arange(8, dtype=np.float64)",
    "np.linspace(0, 1, 8)", "np.full(8, 2.0)", "np.outer(a[:4], b[:4])",
    "np.concatenate([a, b])", "np.take(a, np.argsort(a))", "np.argsort(a)",
    "np.flip(a)", "np.roll(a, 2)", "np.tile(a, 2)[:8]", "np.repeat(a, 2)[:8]",
    "np.sort(a)", "np.unique(a)", "np.searchsorted(np.sort(a), 0.5)",
    "np.isclose(a, b)", "np.isnan(a)", "np.isinf(a)",
]

INVALID_SNIPPETS = [
    "eval('np.add(a,b)')",
    "open('/tmp/x','w').write('hi')",
    "np.add(a, b) + undefined_var",
    "np.unknown_op(a, b)",
]

def _random_program(rng: random.Random, idx: int) -> str:
    n_ops = rng.randint(2, 6)
    ops = [rng.choice(NUMPY_OPS) for _ in range(n_ops)]
    # 10% chance to inject an invalid snippet
    if rng.random() < 0.1:
        ops[rng.randrange(len(ops))] = rng.choice(INVALID_SNIPPETS)
    # Build function with varying signatures
    sigs = ["(a, b)", "(x)", "(A, B)", "(a, b, c)", "(X, y)"]
    sig = rng.choice(sigs)
    body_lines = []
    for i, op in enumerate(ops):
        var = f"t{i}"
        # Some ops return scalar, some array — just assign
        body_lines.append(f"    {var} = {op}")
    # Return last var or sum of scalars
    body_lines.append(f"    return {', '.join([f't{i}' for i in range(len(ops))])}")
    # Occasionally make multi-function file
    extra = ""
    if rng.random() < 0.3:
        extra = f"\ndef helper_{idx}(x):\n    return np.sin(x) + np.cos(x)\n"
    return textwrap.dedent(f"""
import numpy as np
{extra}
def func_{idx}{sig}:
{chr(10).join(body_lines)}
""")


def generate_chaos_corpus(output_dir: str, n: int = 100, seed: int = CHAOS_CORPUS_SEED) -> list[str]:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for i in range(n):
        fname = f"chaos_{i:03d}.py"
        fpath = os.path.join(output_dir, fname)
        prog = _random_program(rng, i)
        with open(fpath, "w", encoding="utf-8", newline="\n") as f:
            f.write(prog)
        written.append(fpath)
    with open(os.path.join(output_dir, "_manifest.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={seed}\ncount={n}\n")
    return sorted(written)
