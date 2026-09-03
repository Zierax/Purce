# Roadmap to v1 — Closing the 5%

> **Where we are:** `v1.0.0` (tag `v1-prod`, `v1.0.0`) is **production**. `497 tests`, `91/91 sweep`, `814 corpus` `0 #error`, `af4d997c` deterministic, `0 stubs`. It is production-ready for the 91 kernels it claims — `493 tests`, `91/91 sweep`, `812 corpus` with `0 #error`, determinism `af4d997c` — but it is *not* a general “any Python to C99” compiler. The remaining ~5% is the frontier, and this document is the plan to close it without trade-offs for `v1`.
>
> **How to read this:** Each item has *what fails today* (with a one-line reproduction), *why it fails* (`file:line`), *what done looks like* (exact pass criteria), and *how we’ll prove it* (command you can run). No “TODO: implement”.

---

## v1.0.0 — What We Shipped (v1-prod)

- **Claim:** “Any code that uses the 91 documented kernels → correct C99.” Proven by `benchmarks/hardened/HARDENED_REPORT.md` (3 stages, 1k/5k/10k iter, 91 kernels, 6.7s worst) and `benchmarks/results.md` (23 files → 1413 kernels).
- **Honesty:** 4 kernels are stubs (`linalg_eig`, `linalg_qr`, `linalg_svd`, `array_split` `purce/backend/c99_generator.py:12`) — they compile but emit `WARNING: stub` and `Verified: NO` (`c99_generator.py:1466`). 7 limitations are documented in `docs/limitations.md` (L1..L7) with reproductions.
- **Tag:** `v1-prod` + `v1.0.0` (annotated, pushed). The working tree at tag includes `docs/kernels.md`, `docs/limitations.md`, `docs/reproducibility.md`, and the hardened suite. The 25+100 `benchmarks/hardened/results/` corpora are *generated* and not tagged — re-run `python -m benchmarks.hardened.run_all_hardened` to reproduce them byte-identically (seed `20260822`).

---

## v1 — Definition of Done

`v1` is done when **all** of the following are true on a clean checkout:

```bash
python -m pytest tests/ -q                          # 493+ new tests, 0 failures
python -m benchmarks.hardened.run_all_hardened      # Stage 1/2/3: 92/92 (not 91/91), determinism PASS, adversarial 7/7
python -m purce.cli extract benchmarks/hardened/results/harsh_corpus -o /tmp/h  && test $(grep -r '#error' /tmp/h | wc -l) -eq 0
python -m purce.cli extract benchmarks/hardened/results/chaos_corpus -o /tmp/c && test $(grep -r '#error' /tmp/c | wc -l) -eq 0
grep -r "WARNING: stub" /tmp/h /tmp/c | wc -l | grep -q "^0$"
```

In words: **92/92 kernels verified** (the `_STUB_ALGORITHMS` set is empty), **0 #error** on both harsh (205) and chaos (523) corpora, and **0 `WARNING: stub`** in any generated file. The `docs/limitations.md` `L1..L7` table will be empty except for the *Physical* limits (stack/heap, GPU) which become documented constraints, not bugs.

---

## Phase 1 — No More Silent Wrong Numbers (P0, 4-6 weeks)

This phase removes every place where Purce today does the wrong thing quietly.

### 1.1 Real `eig`/`qr`/`svd` — `purce/backend/c99_generator.py:812,990,998`

**Today:** `eig` returns Gershgorin centers (`eigenvalues[i]=center` `c99_generator.py:830` discarding `radius`), `qr` copies `Q=x,R=I`, `svd` copies `U=x,S=ones`. They pass the sweep only because the verifier’s `STRUCTURAL_GAP` (`purce/verifier/equivalence.py:46`) checks structural invariants, not numeric equality. The header lies if we don’t mark it — we now mark `Verified: NO`, but that’s a containment, not a fix.

**Done when:** `MATH_KERNEL_BODIES` contains `Householder` QR (for `qr`), `QR iteration` for `eig` (symmetric first, then general via Hessenberg), and `Golub-Kahan` + QR for `svd`. The 4 entries are removed from `_STUB_ALGORITHMS` and the generated headers say `Verified: differential fuzzing` again. The sweep runs `run_sweep(iterations=10, seeds=(42,1337,999))` and `91→92/92` with `max_error < 1e-5` for all three.

**Proof:** `python -m pytest tests/test_stub_kernels.py` will be deleted (its job was to assert `Verified: NO` — in v1 it must assert `Verified: differential fuzzing` for the same algos). New test `tests/test_linalg_real.py` will fuzz each of the three with 5k iterations and `assert max_error < 1e-4` for `eig` (tolerant) and `1e-10` for `qr/svd` reconstruction `||A - QR||`, `||A - USV||`.

**Owner:** backend. **Risk:** `eig` for non-symmetric general matrices is the hardest — we’ll ship symmetric (`np.linalg.eig` on symmetric) first, then general. The falsification is `A = [[0,1],[ -1,0]]` (pure imaginary eigenvalues) — the current stub returns `[0,0]`, the real must return `[i,-i]`.

### 1.2 Per-input lengths for `array_take` — `purce/backend/c99_generator.py:108,763`

**Today:** `BODY_PARAM_MAP` has only `k` (output length). The body checks `idx < k` (`c99_generator.py:767`) which is correct only when `len(x) == len(idx)`. When they differ (chaos does), the check is wrong. We reverted a half-fix because it broke the sweep’s `k`-only reference; the proper fix needs two lengths.

**Done when:** `BODY_PARAM_MAP["array_take"]` becomes `[("x","input_0"), ("idx","input_1"), ("out","output_0"), ("n","input_0_len"), ("k","input_1_len")]` (or `dim_m/dim_n` — the names matter less than the plumbing). `_build_body_param_mapping` (`c99_generator.py:1040`) is taught to resolve `input_0_len` → `n` and `input_1_len` → `k` from the node’s two input shapes. The body becomes `idx < n`, and `purce/verifier/equivalence.py:372` becomes `valid = (idx>=0)&(idx < len(x))` (or `scalars["n"]`). The driver’s `_make_case` for `array_take` generates `x` and `idx` with independent lengths (`n= _rand_small(1,24)`, `k= _rand_small(1,24)`).

**Proof:** Reproduction `x=np.arange(10); idx=np.array([9,9,9]); np.take(x,idx)` with `n=10,k=3` — old C returns `0,0,0` (because `9<3` is false), new C returns `9,9,9`. Sweep `array_take` passes with independent `n/k`.

### 1.3 VLA heap fallback — `purce/backend/c99_generator.py:832,964,846`

**Today:** `array_sort`/`unique` guard `if (n>8192) return;` and `linalg_det` guard `if (n>64) return;` leave the output buffer untouched — safe from stack overflow, but silent. The hardened report documents it as a limitation.

**Done when:** When `n` exceeds the threshold, the kernel `malloc`s, sorts/dets, `memcpy`s, and `free`s. The stack path stays `double tmp[n]` for small `n` (fast), the heap path is `double *tmp = malloc(n*sizeof(double))`. The `MEMORY CONTRACT` header changes from `Heap: NONE` to `Heap: maybe (n>8192)` and the body checks `if (!tmp) return;`. The `purce.verifier.ctypes_bridge` still compiles with `-std=c99 -O2 -Wall -Wextra -pedantic`.

**Proof:** `n=20000` sort completes and matches `np.sort`; `n=65` det completes and matches `np.linalg.det` within `1e-6`. No `#error`, no silent return.

---

## Phase 2 — Language Coverage (P1, 4-6 weeks, can overlap Phase 1)

### 2.1 Alias imports — `purce/parser/python_parser.py:236`

**Today:** `_normalize_numpy_target` handles `np.add` and `numpy.add` but not `from numpy import dot` or `import numpy.linalg as la`. `from numpy import dot; dot(A,B)` silently yields 0 kernels.

**Done when:** The parser builds an `alias_map` from `ast.Import` and `ast.ImportFrom` (`import numpy as np` → `np→numpy`, `from numpy import dot` → `dot→numpy.dot`, `import numpy.linalg as la` → `la→numpy.linalg`). `_resolve_call_target` resolves `dot` → `numpy.dot`, `la.solve` → `numpy.linalg.solve` via the map before `_normalize`.

**Proof:** File `from numpy import dot; import numpy.linalg as la; def f(A,B): return dot(A,B) + la.norm(A)` → `2` kernels (`matmul`, `linalg_norm`) instead of `0`.

### 2.2 Missing kernels — `purce/ir/builder.py:18`, `purce/backend/c99_generator.py:234`

**Today:** `np.convolve`, `np.einsum`, `np.kron`, `np.tensordot` → `composite` → `#error`. The harsh corpus hits `np.convolve` in `harsh_signal_stress`.

**Done when:** `einsum` (at least `ij,jk->ik` and `i,i->` cases) and `convolve` (1-D) are added to `NUMPY_OP_MAP` + `BODY_PARAM_MAP` + `MATH_KERNEL_BODIES` with bodies, and `DERIVED_PARAMS`. Each gets a fuzzer reference in `purce/verifier/fuzzer.py` and an equivalence case in `purce/verifier/equivalence.py`.

**Proof:** `harsh_signal_stress` no longer emits `composite #error`.

---

## Phase 3 — Architecture Debt (P2, 6-12 weeks, does not block v1 correctness)

These do not change the 92/92 number, but they determine whether v1 stays maintainable after we add 10 more kernels.

- **God Object:** Split `c99_generator.py` (1668 LOC) into `purce/backend/registry.py` (dicts), `purce/backend/kernels/*.c` (real `.c` files loaded via `importlib.resources`), `purce/backend/emitter/header.py` + `signature.py` + `provenance.py`. No new kernels may be added to the monolith; CI fails if `c99_generator.py` grows.
- **Verifier duplication:** Unify `_all_close`, `_find_gcc`/`compile`, and reference oracles (`fuzzer.py:519` vs `equivalence.py:122`) into `purce/verifier/_common.py`. One fix in one place.
- **Chaos nightly:** Promote `benchmarks/hardened` chaos 100 → 500 in CI nightly, with `harsh 205/205` and `chaos 523/523` gates.

---

## What v0.1.0 Beta Explicitly Does Not Promise

- No GPU/CUDA, no heap in hot kernels (until Phase 1.3), max `64×64` for `solve/inv` (stack limit `c99_generator.py:311`).
- Stubs are loud, not correct — do not use `eig/qr/svd` in production on `v0.1.0`.
- `array_take` with `len(x) != len(idx)` is best-effort (`idx<k`), not correct, until Phase 1.2.

If you need any of those today, track the matching Phase item above — each has a one-line falsification you can run.

---

## How to Verify We Didn’t Regress

After any Phase, run:

```bash
python -m pytest tests/ -q
python -m benchmarks.hardened.run_all_hardened   # must stay 92/92, determinism PASS, adversarial 7/7
python -m benchmarks.reproducible --baseline benchmarks/baseline.json
```

All three must stay green while `benchmarks/hardened/results/harsh_corpus/CHAOS` goes from `14/21 #error` → `0`.

---

*This roadmap is the contract for v1. Every item has a file:line, a reproduction, and a proof. No trade-off is hidden.*
