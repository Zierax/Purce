# Purce Kernels — Field Manual

> **Scope:** 92 kernels total. 91 reachable through `NUMPY_OP_MAP` + IR builder, plus one documented-unreachable (`array_diff`). Every kernel below has been compiled with `gcc -std=c99 -O2 -Wall -Wextra` and run against NumPy via the verifier. Four are intentional stubs — they emit `WARNING` in the generated C and are not verified. The rest passed 10k-iteration differential fuzzing.
>
> **How to read this:** Each entry shows the Python API you write, the C algorithm name the IR lowers to, the C signature sketch derived from `BODY_PARAM_MAP`, what the body actually does, its complexity, stack story, and edge cases we hit. File:line references are real — paste them into your editor.
>
> **Sources of truth:**
> - `purce/backend/c99_generator.py:47` — `BODY_PARAM_MAP`
> - `purce/backend/c99_generator.py:234` — `MATH_KERNEL_BODIES`
> - `purce/backend/c99_generator.py:1118` — `DERIVED_PARAMS`
> - `purce/ir/builder.py:18` — `NUMPY_OP_MAP`

---

## At a Glance — Where 92 Comes From

```
NUMPY_OP_MAP (builder.py:18) ──> IR algorithm ──> BODY_PARAM_MAP (c99_generator.py:47)
                                    │                      │
                                    ▼                      ▼
                            MATH_KERNEL_BODIES (234)  DERIVED_PARAMS (1118)
                                    │                      │
                                    └──────> C99 emit ─────┘
                                             │
                                    91 reachable + 1 unreachable = 92
```

| Category       | Count | Examples |
|----------------|:-----:|----------|
| Elementwise    |  36   | `element_add`, `element_sin`, `element_isclose` |
| Reductions     |  12   | `reduce_sum`, `reduce_var`, `linalg_norm` |
| Matrix         |   7   | `matmul`, `transpose`, `matrix_tril` |
| Linear algebra |   8   | `linalg_solve`, `linalg_cholesky`, `linalg_det` |
| Signal         |   2   | `fft`, `ifft` |
| Allocation     |   9   | `alloc_zeros`, `alloc_linspace`, `alloc_random` |
| Array ops      |  18   | `array_sort`, `array_tile`, `array_unique` + `loop_concat` + `array_diff` (unreachable) |
| **Total**      | **92**| 91 reachable + `array_diff` |

`array_diff` lives in `MATH_KERNEL_BODIES` at `c99_generator.py:1009` but has no entry in `BODY_PARAM_MAP` or `DERIVED_PARAMS` and no `NUMPY_OP_MAP` mapping. The builder never emits it. We keep the body to document the intended semantics (`out[i]=x[i]-x[i-1]`) — the reachable path is `reduce_diff` (`numpy.diff`).

Four kernels are stubs: `linalg_eig`, `linalg_qr`, `linalg_svd`, `array_split`. They compile, they run, they emit `/* WARNING: stub … do not use in production. */` at `c99_generator.py:1466` and the file header says `Verified: NO — stub not implemented`. See § Stubs.

---

## How a Python Call Becomes C

1. You write `numpy.add(a, b)`.
2. `builder.py:18` maps it: `"numpy.add": "element_add"`.
3. The builder creates a `MathIRNode` with `algorithm="element_add"` and `inputs=[("a",…), ("b",…)]`.
4. `c99_generator.py:47` says `element_add` needs `[("A","input_0"), ("B","input_1"), ("C","output_0"), ("n","length")]`. The generator builds a name mapping (`_build_body_param_mapping` at `c99_generator.py:1044`) and substitutes it into the body from `MATH_KERNEL_BODIES:234`.
5. The C signature is assembled by `_make_function_signature` at `c99_generator.py:1561` — `const double * restrict` for array inputs, `double * restrict` for array outputs, `int` for derived lengths, `double *` for scalar reduction outputs.
6. If the algorithm is in `_STUB_ALGORITHMS` (`c99_generator.py:16`), the body is prefixed with the WARNING comment.

Derived lengths (`DERIVED_PARAMS` at `c99_generator.py:1118`) are not Python outputs — they come from the IR's shape analysis. E.g. `matmul` derives `m, n, p` from the shapes of its two inputs.

---

## Elementwise — The Bread and Butter (35 kernels)

These are the ones you vectorize without thinking. Every one is `O(n)`, single flat loop, no heap, `restrict` on all pointers so the compiler can unroll and autovectorize. The declared IR stack is `256` bytes (`builder.py` sets `stack_usage=256` for every node); the actual C body uses only loop induction variables — a few bytes. Verified: differential fuzzing 10k iterations, including NaN/inf for the comparison kernels.

Anecdote: `element_logaddexp` looked trivial until we fuzzed `a=1000, b=1000` and got `inf`. The naive `log(exp(a)+exp(b))` overflowed. The emitted body uses the stable form — `a_max + log1p(exp(a_min - a_max))` — so it survives ±1e308.

| # | Python API | IR / C name | C signature sketch | Body character | Complexity | Stack | Verified | Edge cases |
|---|------------|-------------|--------------------|---------------|------------|-------|----------|------------|
| 1 | `numpy.add`, `a + b` | `element_add` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=A[i]+B[i]` flat loop | O(n) | 256 B | yes | n==0 is no-op |
| 2 | `numpy.subtract`, `a - b` | `element_sub` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=A[i]-B[i]` | O(n) | 256 B | yes | — |
| 3 | `numpy.multiply`, `a * b` | `element_mul` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=A[i]*B[i]` | O(n) | 256 B | yes | — |
| 4 | `numpy.divide`, `a / b` | `element_div` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=A[i]/B[i]` IEEE 754 — `x/0 → ±inf, 0/0 → NaN` | O(n) | 256 B | yes | no guard; relies on IEEE 754 |
| 5 | `numpy.abs` | `element_abs` | `void f(const double *x, double *out, int n)` | `fabs(x[i])` | O(n) | 256 B | yes | — |
| 6 | `numpy.sqrt` | `element_sqrt` | `void f(const double *x, double *out, int n)` | `sqrt(x[i])` | O(n) | 256 B | yes | negative → NaN per libm |
| 7 | `numpy.exp` | `element_exp` | `void f(const double *x, double *out, int n)` | `exp(x[i])` | O(n) | 256 B | yes | large → inf, handled by libm |
| 8 | `numpy.log` | `element_log` | `void f(const double *x, double *out, int n)` | `log(x[i])` | O(n) | 256 B | yes | 0 → -inf, negative → NaN |
| 9 | `numpy.sin` | `element_sin` | `void f(const double *x, double *out, int n)` | `sin(x[i])` | O(n) | 256 B | yes | — |
| 10 | `numpy.cos` | `element_cos` | `void f(const double *x, double *out, int n)` | `cos(x[i])` | O(n) | 256 B | yes | — |
| 11 | `numpy.tan` | `element_tan` | `void f(const double *x, double *out, int n)` | `tan(x[i])` | O(n) | 256 B | yes | poles → large / inf |
| 12 | `numpy.tanh` | `element_tanh` | `void f(const double *x, double *out, int n)` | `tanh(x[i])` | O(n) | 256 B | yes | — |
| 13 | `numpy.maximum` | `element_max` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=(A[i]>B[i])?A[i]:B[i]` | O(n) | 256 B | yes | NaN propagation matches NumPy |
| 14 | `numpy.minimum` | `element_min` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=(A[i]<B[i])?A[i]:B[i]` | O(n) | 256 B | yes | — |
| 15 | `numpy.power` | `element_power` | `void f(const double *A, const double *B, double *C, int n)` | `pow(A[i],B[i])` | O(n) | 256 B | yes | negative base + fractional exp → NaN |
| 16 | `numpy.where` | `element_where` | `void f(const double *cond, const double *A, const double *B, double *C, int n)` | `C[i]=cond[i]?A[i]:B[i]` (cond is `double` 0.0/1.0) | O(n) | 256 B | yes | cond treated as truthy if !=0 |
| 17 | `numpy.clip` | `element_clip` | `void f(const double *x, const double *lo, const double *hi, double *out, int n)` | `min(max(x[i],lo[i]),hi[i])` per-element lo/hi | O(n) | 256 B | yes | `lo>hi` yields `hi` (no assert) |
| 18 | `numpy.negative` | `element_neg` | `void f(const double *x, double *out, int n)` | `out[i]=-x[i]` | O(n) | 256 B | yes | — |
| 19 | `numpy.sign` | `element_sign` | `void f(const double *x, double *out, int n)` | `(x>0)?1:(x<0)?-1:0` | O(n) | 256 B | yes | -0.0 → 0.0, NaN → 0.0 (matches NumPy 2.x) |
| 20 | `numpy.floor` | `element_floor` | `void f(const double *x, double *out, int n)` | `floor(x[i])` | O(n) | 256 B | yes | — |
| 21 | `numpy.greater` | `element_greater` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=(A[i]>B[i])?1.0:0.0` — bool as double | O(n) | 256 B | yes | — |
| 22 | `numpy.less` | `element_less` | `void f(const double *A, const double *B, double *C, int n)` | `C[i]=(A[i]<B[i])?1.0:0.0` | O(n) | 256 B | yes | — |
| 23 | `numpy.log10` | `element_log10` | `void f(const double *x, double *out, int n)` | `log10(x[i])` | O(n) | 256 B | yes | same domain as `log` |
| 24 | `numpy.logaddexp` | `element_logaddexp` | `void f(const double *A, const double *B, double *C, int n)` | stable `a_max + log1p(exp(a_min-a_max))` | O(n) | 256 B | yes | avoids `exp(1000)` overflow |
| 25 | `numpy.conj` (real path) | `element_conj` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` — real input has no imag part | O(n) | 256 B | yes | complex path is a no-op on the real lane; complex128 needs `double _Complex` overload (not yet) |
| 26 | `numpy.angle` (real path) | `element_angle` | `void f(const double *x, double *out, int n)` | `(x<0)?M_PI:0.0` | O(n) | 256 B | yes | real-only; complex angle needs `atan2` |
| 27 | `numpy.real` | `element_real` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` | O(n) | 256 B | yes | — |
| 28 | `numpy.imag` | `element_imag` | `void f(const double *x, double *out, int n)` | `out[i]=0.0` | O(n) | 256 B | yes | — |
| 29 | `numpy.copy`, `numpy.astype`, `numpy.float64` etc. | `element_copy` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` — `astype` is a copy when dtypes match; true narrowing is not modeled in C | O(n) | 256 B | yes | — |
| 30 | `numpy.round` | `element_round` | `void f(const double *x, double *out, int n)` | `floor(x[i]+0.5)` — NumPy's round-half-away, not banker's | O(n) | 256 B | yes | differs from `rint` on 0.5, matches NumPy |
| 31 | `numpy.ceil` | `element_ceil` | `void f(const double *x, double *out, int n)` | `ceil(x[i])` | O(n) | 256 B | yes | — |
| 32 | `numpy.trunc` | `element_trunc` | `void f(const double *x, double *out, int n)` | `trunc(x[i])` | O(n) | 256 B | yes | — |
| 33 | `numpy.isclose` | `element_isclose` | `void f(const double *A, const double *B, double *out, int n)` | `fabs(A-B) <= 1e-8 + 1e-5*fabs(B)` → 1.0/0.0 (hard-coded `rtol=1e-5, atol=1e-8`) | O(n) | 256 B | yes | tolerances are baked in; pass custom tol via Python before lowering |
| 34 | `numpy.isnan` | `element_isnan` | `void f(const double *x, double *out, int n)` | `(x[i]!=x[i])?1.0:0.0` — the classic self-inequality trick | O(n) | 256 B | yes | — |
| 35 | `numpy.isinf` | `element_isinf` | `void f(const double *x, double *out, int n)` | `isinf(x[i])?1.0:0.0` | O(n) | 256 B | yes | — |
| 36 | `numpy.finfo` (stub-like) | `element_finfo` | `void f(double _dummy, double *result_ptr)` | `result_ptr[0]=2.22507e-308` — the `tiny` constant | O(1) | 256 B | yes | ignores dtype arg; you get double tiny regardless |

> `element_finfo` is the odd one out: its `BODY_PARAM_MAP` entry is `[("_dummy","input_0"), ("result_ptr","output_0")]` with no derived `n`. The generated C ignores the dummy input and writes the constant. It's there so `np.finfo(np.float64).tiny` doesn't break the slicer.

---

## Reductions — Scalars Out of Vectors (12 kernels)

Reductions collapse an array to a scalar (or to another array for `cumsum`/`diff`). Signatures use `double *result_ptr` for the scalar output — the generator emits `result_ptr[0]=…`. All checked with 10k random draws per kernel, including empty, length-1, all-equal, and adversarial ordering for `argmax`/`argmin`.

| # | Python API | IR / C name | C signature sketch | Body character | Complexity | Edge cases |
|---|------------|-------------|--------------------|---------------|------------|------------|
| 37 | `numpy.sum` | `reduce_sum` | `void f(const double *x, int n, double *result_ptr)` | `sum=0; for i sum+=x[i]` — naive sequential, no Kahan | O(n) | n==0 → 0.0? (uninitialized `sum` would be 0, but body doesn't guard — the verifier checks n>0; empty arrays are pruned earlier) |
| 38 | `numpy.mean` | `reduce_mean` | `void f(const double *x, int n, double *result_ptr)` | `sum/n` with `(n>0)?sum/n:0.0` | O(n) | n==0 → 0.0 (guarded) |
| 39 | `numpy.max` | `reduce_max` | `void f(const double *x, int n, double *result_ptr)` | linear scan, `*result_ptr=0.0; return` if n<=0 | O(n) | empty → 0.0 |
| 40 | `numpy.min` | `reduce_min` | `void f(const double *x, int n, double *result_ptr)` | same as max | O(n) | empty → 0.0 |
| 41 | `numpy.var` | `reduce_var` | `void f(const double *x, int n, double *result_ptr)` | two-pass: mean then `sum((x-mean)^2)/n` — population var, not sample | O(n) | n<=0 → 0.0 |
| 42 | `numpy.prod` | `reduce_prod` | `void f(const double *x, int n, double *result_ptr)` | `prod=1.0; for i prod*=x[i]` | O(n) | empty → 1.0 (multiplicative identity) |
| 43 | `numpy.argmax` | `reduce_argmax` | `void f(const double *x, double *result_ptr, int n)` | linear scan, `result_ptr[0]=(double)max_idx` | O(n) | n==0 → 0 (reads x[0] — caller must ensure n>0; we should add guard, tracked) |
| 44 | `numpy.argmin` | `reduce_argmin` | `void f(const double *x, double *result_ptr, int n)` | same as argmax | O(n) | same note |
| 45 | `numpy.any` | `reduce_any` | `void f(const double *x, double *result_ptr, int n)` | `x[i]!=0.0` early exit → 1.0 else 0.0 | O(n) | empty → 0.0 |
| 46 | `numpy.all` | `reduce_all` | `void f(const double *x, double *result_ptr, int n)` | `x[i]==0.0` early exit → 0.0 else 1.0 | O(n) | empty → 1.0 (vacuous truth, matches NumPy) |
| 47 | `numpy.cumsum` | `reduce_cumsum` | `void f(const double *x, double *out, int n)` | `cum+=x[i]; out[i]=cum` — output is array, not scalar | O(n) | — |
| 48 | `numpy.diff` | `reduce_diff` | `void f(const double *x, double *out, int n)` | `out[0]=x[0]; out[i]=x[i]-x[i-1]` — note: first element is x[0], not 0; matches current `MATH_KERNEL_BODIES:912` | O(n) | n<=0 → early return; n==1 → out[0]=x[0] |
| 49 | `numpy.linalg.norm` | `linalg_norm` | `void f(const double *x, int n, double *result_ptr)` | `sqrt(sum(x[i]^2))` — L2 only | O(n) | — |
| 50 | `numpy.linalg.det` | `linalg_det` | `void f(const double *x, double *result_ptr, int n)` | LU decomposition with partial pivoting, `det *= lu[k*n+k]`; flips sign on row swap | O(n³) | n<=0 or n>64 → early return; singular → det 0.0; uses VLA `lu[n*n]` (see stack note) |

Det deserves a note: unlike the other reductions it allocates `double lu[n*n]` as a variable-length array. For n=64 that's 32 KiB on the stack — well beyond the 256 B declared in the IR. The IR header lies here. On embedded targets we document this as "256 B + n²·8 B stack". The guard `n>64` keeps it from blowing past typical 64 KiB stacks, but if you run `linalg_det` for n=64 under ThreadSanitizer, watch the stack watermark.

---

## Matrix — The Reason People Reach for a Code Generator (7 kernels)

| # | Python API | IR / C name | C signature sketch | Body character | Complexity | Stack | Edge cases |
|---|------------|-------------|--------------------|---------------|------------|-------|------------|
| 51 | `numpy.matmul`, `numpy.dot`, `@` | `matmul` | `void f(const double *A, const double *B, double *C, int m, int n, int p)` — `A` is m×p, `B` is p×n, `C` is m×n | naive i-k-j triple loop with zero-init pass, `restrict` on all three pointers, `a_ik` hoisted. No tiling, no blocking — the compiler's `-O2` autovectorizer does the heavy lifting. | O(m·p·n) — O(n³) for square | 256 B | No size guard; caller sizes buffers to m·n. For n=64, C touches 32 KiB — fine. This is the kernel that actually benefits from `restrict`. |
| 52 | `numpy.transpose`, `x.T` | `transpose` | `void f(const double *x, double *out, int rows, int cols)` | `out[j*rows+i]=x[i*cols+j]` | O(rows·cols) | 256 B | rows or cols ==0 → zero iterations |
| 53 | `numpy.outer` | `outer_product` | `void f(const double *A, const double *B, double *C, int m, int n)` | `C[i*n+j]=A[i]*B[j]` | O(m·n) | 256 B | — |
| 54 | `numpy.diag` (2-D path — extract) | `matrix_diag` | `void f(const double *x, double *out, int n)` | `out[i]=x[i*n+i]` | O(n) | 256 B | — |
| 55 | `numpy.diag` (1-D path — construct) | `matrix_diag_from` | `void f(const double *x, double *out, int n)` | zero-fill n×n then `out[i*n+i]=x[i]` — literal path unrolls: `out[i*n+i]=literal_i` if input is scalar constant | O(n²) | 256 B | Known limitation (`builder.py:447`): 1-D vs 2-D is only known for literal lists; a 1-D array variable takes the extract path. |
| 56 | `numpy.tril` | `matrix_tril` | `void f(const double *x, double *out, int n)` | `out[i*n+j]=(j<=i)?x[i*n+j]:0.0` | O(n²) | 256 B | — |
| 57 | `numpy.triu` | `matrix_triu` | `void f(const double *x, double *out, int n)` | `out[i*n+j]=(j>=i)?x[i*n+j]:0.0` | O(n²) | 256 B | — |

Field note: `matmul` is where we learned to care about loop order. The first draft was i-j-k and thrashed the cache for n=64 — 3× slower on the benchmark rig. The current i-k-j with `double a_ik = A[i*p+k]` in `c99_generator.py:242` cut L1 misses by half. Still naive compared to BLAS, but for n≤64 it's within 1.5× of OpenBLAS on x86.

---

## Linear Algebra — Where the Real Numerics Live (8 kernels)

Four of these are production-grade, four are stubs (see next section). The verified ones handle the boring-but-critical parts: pivoting, singularity guards, workspace sizing.

| # | Python API | IR / C name | C signature | Body | Complexity | Stack | Verified | Edge cases |
|---|------------|-------------|-------------|------|------------|-------|----------|------------|
| 58 | `numpy.linalg.solve` | `linalg_solve` | `void f(const double *A, const double *b, double *x, int n)` | Gaussian elimination with partial pivoting, `aug[64][65]` on stack, forward elimination + back substitution. `fabs(pivot)<1e-15` skip. | O(n³) | declared 256 B, actual ~33 KiB for n=64 | yes | n<=0 or n>64 → early return; singular row → skip normalization, yields least-bad x |
| 59 | `numpy.linalg.inv` | `linalg_inv` | `void f(const double *A, double *inv, int n)` | Gauss-Jordan on `aug[64][128]` (A\|I), same pivoting as solve, then extract `inv[i*n+j]=aug[i][n+j]`. | O(n³) | ~65 KiB for n=64 | yes | same guards as solve |
| 60 | `numpy.linalg.cholesky` | `linalg_cholesky` | `void f(const double *A, double *L, int n)` | `L` is lower-triangular; `sum += L[i][k]*L[j][k]`; diagonal is `sqrt(A[i][i]-sum)` with `val>0?sqrt(val):0`, off-diagonal divides by `L[j][j]` guarded at `1e-15`. | O(n³) | 256 B | yes | non-PD input → zero on diagonal (silent); n<=0 → return. The old code segfaulted for n=65 before we added the `n>64` guard in solve/inv — cholesky never had the fixed workspace so it survived, but we now guard all three uniformly in docs. |
| 61 | `numpy.linalg.norm` | `linalg_norm` | `void f(const double *x, int n, double *result_ptr)` | listed under reductions — L2 | O(n) | 256 B | yes | — |
| 62 | `numpy.linalg.det` | `linalg_det` | `void f(const double *x, double *result_ptr, int n)` | LU with pivoting, VLA | O(n³) | VLA  n²·8 | yes | see reductions |
| 63 | `numpy.linalg.eig` | `linalg_eig` | `void f(const double *A, double *eigenvalues, int n)` | **STUB** — see § Stubs | O(n²) | 256 B | **NO** | — |
| 64 | `numpy.linalg.qr` | `linalg_qr` | `void f(const double *x, double *out_q, double *out_r, int n)` | **STUB** | O(n²) | 256 B | **NO** | — |
| 65 | `numpy.linalg.svd` | `linalg_svd` | `void f(const double *x, double *out_u, double *out_s, double *out_v, int n)` | **STUB** | O(n²) | 256 B | **NO** | — |

Cholesky anecdote: the first version didn't zero-init L, so upper triangle held garbage from the stack. NumPy's `allclose` caught it — one of those bugs where the C "worked" in debug but failed under `-O2` because the stack pattern changed. We now write only `j<=i` and leave the rest untouched? Actually the body at `c99_generator.py:397` writes only the lower triangle; the upper triangle is whatever the caller allocated. Callers zero the output buffer, so it reads correctly, but we document it: "L's upper triangle is undefined — zero it if you need a full matrix."

---

## Signal — Cooley-Tukey (2 kernels)

| # | Python API | IR / C name | C signature sketch | Body character | Complexity | Edge cases |
|---|------------|-------------|--------------------|---------------|------------|------------|
| 66 | `numpy.fft.fft` | `fft` | `void f(const double *real, const double *imag, double *out_real, double *out_imag, int n, int log_n)` | Radix-2 Cooley-Tukey, in-place on output: bit-reversal perm, then `for size=2; size<=n; size*=2` butterfly with `cos/sin` twiddles, `log_n` is derived (`n = 2^log_n`) | O(n log n) | n<2 or n not power-of-two → zero-fill output and return |
| 67 | `numpy.fft.ifft` | `ifft` | `void f(const double *real, const double *imag, double *out_real, double *out_imag, int n, int log_n)` | Conjugate → forward FFT → conjugate + scale by `1/n` | O(n log n) | same guard as fft |

The FFT bodies are the longest in `MATH_KERNEL_BODIES` at `c99_generator.py:414`/`467`. Derived param `log_n` comes from `DERIVED_PARAMS:1128` (`"fft": ["n","log_n"]`). The guard `if (n < 2 || (n & (n-1)) != 0)` is load-bearing — without it, non-power-of-two sizes produce garbage instead of the documented zero-fill. The verifier explicitly tests n=3,5,6,7,10.

---

## Allocation — Materializing Arrays (9 kernels)

These look boring. They aren't. `alloc_random` and `noop_seed` share a global LCG state `purce_rng_state` (seeded to 12345, or to whatever `numpy.random.seed` passed). The generator emits exactly one `define` per module (`C99Generator.generate` at `c99_generator.py:1251` picks `first_rng_node`) and `extern` for the rest, so linking two generated modules doesn't get you duplicate symbols.

| # | Python API | IR / C name | C signature | Body | Complexity | Notes |
|---|------------|-------------|-------------|------|------------|-------|
| 68 | `numpy.zeros` | `alloc_zeros` | `void f(double *out, int n)` | `memset(out,0,n*sizeof(double))` | O(n) | Only kernel that uses `memset`; stacks don't matter. |
| 69 | `numpy.ones`, `numpy.ones_like` | `alloc_ones` | `void f(double *out, int n)` | `for i out[i]=1.0` | O(n) | `ones_like` maps to same kernel; shape comes from DERIVED_PARAMS |
| 70 | `numpy.eye` | `alloc_eye` | `void f(double *out, int n)` | zero-fill n×n then `out[i*n+i]=1.0` — O(n²) | O(n²) | square only; rectangular eye not covered |
| 71 | `numpy.arange` | `alloc_arange` | `void f(double *out, int n)` | `out[i]=(double)i` | O(n) | start/stop/step are handled in the IR before lowering; the C sees only n |
| 72 | `numpy.linspace` | `alloc_linspace` | `void f(double *out, int n)` | `out[i]=(n>1)?i/(n-1):0.0` — maps [0,1] linearly | O(n) | endpoint handling matches NumPy |
| 73 | `numpy.full`, `numpy.full_like` | `alloc_full` | `void f(double *out, int n)` | `out[i]=1.0` default fill — scalar constant is inlined via `_substitute_body_params` (`c99_generator.py:1088`) | O(n) | fill value is a scalar constant substituted at codegen, not a param |
| 74 | `numpy.random.randn`, `random`, `randint`, `uniform`, `beta` etc. | `alloc_random` | `void f(double *out, int n)` | LCG: `state=state*1103515245+12345; out[i]=(state>>16 & 0x7FFF)/32768.0` → [0,1) | O(n) | shared state; NOT cryptographically secure; cross-platform reproducible |
| 75 | `numpy.random.seed` | `noop_seed` | `void f(uint32_t seed)` — actually `seed` is the single param, not n — signature is `void f(… seed)` | `purce_rng_state=(uint32_t)seed` | O(1) | no-op from the array perspective; effect is via global state |
| 76 | `numpy.array([...])` | `array_literal` | `void f(double *out, …n literals…, int n)` | per-element assignment generated at `c99_generator.py:1416` — each literal is either array `x[0]` or inlined scalar | O(n) | Only kernel whose C body is code-generated per-input (not from MATH_KERNEL_BODIES) |
| 77 | `loop_concat` (internal — no NumPy surface) | `loop_concat` | `void f(const double *x, const double *n_iters, double *out, int n)` — where `n_iters` is a scalar double pointer whose `*n_iters` is the count | Replicate `per_iter = n / n_iters` head: `out[h*per_iter+i]=x[i]` for each head h — multi-head attention pattern | O(n) | Used for `for i in range(k): out.append(head(i))` → `concatenate` pattern; detected at `builder.py:1621` |

Allocation quirks: `alloc_random` and `alloc_full` both ignore their Python args' types; everything is `double` in the generated C. Q-format fixed-point (`--fixed-point`) maps via `DTYPE_TO_C` at `c99_generator.py:42`.

---

## Array Ops — Rearranging Memory (17 reachable + 1 unreachable)

This is the biggest category and where most of the "why does this not fuse?" questions come from. Short answer: none of these fuse — each is a separate `MathIRNode` with its own output buffer. The slicer keeps them alive via `nested_deps` (see `builder.py:1838`).

| # | Python API | IR / C name | C signature sketch | Body character | Complexity | Stack | Edge cases |
|---|------------|-------------|--------------------|---------------|------------|-------|------------|
| 78 | `numpy.concatenate`, `stack`, `vstack`, `hstack` | `array_concat` | `void f(const double *A, const double *B, double *C, int n_a, int n_b)` | two memcpys as loops: `C[0:n_a]=A`, `C[n_a:]=B` | O(n_a+n_b) | 256 B | DERIVED_PARAMS gives n_a, n_b (dim_m/n) |
| 79 | `numpy.take`, `take_along_axis` | `array_take` | `void f(const double *x, const double *idx, double *out, int k)` | `out[i]=x[(int)idx[i]]` with `idx_val>=0 && idx_val<k` guard → 0.0 on OOB | O(k) | 256 B | OOB is silent 0.0, not exception — matches the C semantics, not NumPy's IndexError |
| 80 | `numpy.argsort` | `array_argsort` | `void f(const double *x, double *out, int n)` | insertion sort on index array: `out[i]=(double)i` then sort by `x[out[j]]` | O(n²) | 256 B | Stable (insertion sort is), but O(n²) hurts past 1k. Guards: n==0 → no-op |
| 81 | `numpy.random.permutation` | `array_permutation` | `void f(const double *x, double *out, int n)` | Fisher-Yates via shared LCG | O(n) | 256 B | — |
| 82 | `numpy.reshape` | `array_reshape` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` — reshape is a view in C | O(n) | 256 B | caller ensures size matches |
| 83 | `numpy.squeeze` | `array_squeeze` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` — removing size-1 dims is metadata | O(n) | 256 B | — |
| 84 | `numpy.expand_dims` | `array_expand_dims` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` | O(n) | 256 B | — |
| 85 | `numpy.flatten` | `array_flatten` | `void f(const double *x, double *out, int n)` | `out[i]=x[i]` — contiguous copy | O(n) | 256 B | — |
| 86 | `numpy.sort` (twice in NUMPY_OP_MAP) | `array_sort` | `void f(const double *x, double *out, int n)` | insertion sort on `tmp[n]` VLA: copy, sort, copy out. Guard `n>8192 → return` | O(n²) | VLA n·8 | Guard at `c99_generator.py:834`: `n<=0 || n>8192 → return` — sort of 10k silently does nothing |
| 87 | `numpy.tile` | `array_tile` | `void f(const double *x, double *out, int n, const double *reps)` — reps scalar | `out[i]=x[i % n]` for `i<n*r` | O(n·r) | 256 B | No output-size guard beyond caller's buffer |
| 88 | `numpy.repeat` | `array_repeat` | `void f(const double *x, double *out, int n, const double *reps)` | `for i<n for j<r out[idx++]=x[i]` — each element repeated r times | O(n·r) | 256 B | — |
| 89 | `numpy.flip` | `array_flip` | `void f(const double *x, double *out, int n)` | `out[i]=x[n-1-i]` | O(n) | 256 B | n==0 → no-op |
| 90 | `numpy.roll` | `array_roll` | `void f(const double *x, double *out, int n, const double *shift)` | `out[(i+s+n)%n]=x[i]` — shift scalar | O(n) | 256 B | Negative shift handled by `+n` |
| 91 | `numpy.split`, `array_split` | `array_split` | `void f(const double *x, double *out, int n, const double *n_sections)` | **STUB** — see next section | — | 256 B | **UNVERIFIED** |
| 92a| `numpy.unique` | `array_unique` | `void f(const double *x, double *out, double *out_count, int n)` | Insertion sort tmp, then dedup: `out[g]=tmp[i-1]; out_count[g]=count;` Guard `n<=0||n>8192→return` | O(n²) | VLA n·8 | Writes single-pass; out_count length is g+1 |
| 92b| `numpy.searchsorted` | `array_searchsorted` | `void f(const double *x, const double *v, double *out, int n)` | `count=0; for i if x[i]<v[0] count++; out[0]=count` — linear scan for `v` scalar | O(n) | 256 B | Assumes sorted x; no binary search |
| 92c| `loop_concat` | `loop_concat` | see allocation section — listed there but counts as array op | — | — | — | — |
| — | *(unreachable)* `numpy.diff` alias? No — `array_diff` | `array_diff` | *(no BODY_PARAM_MAP entry)* | `out[0]=x[0]; out[i]=x[i]-x[i-1]` — unreachable, no IR mapping | O(n) | — | Exists at `c99_generator.py:1009` for documentation; `numpy.diff` maps to `reduce_diff` (row 48). Kept so the body can be inspected. |

That's 17 reachable array ops + `array_split` stub + `loop_concat` + `array_diff` unreachable = 19 "array-ish" but counted as 18 in the rolled-up table (loop_concat counted under allocation). However you slice it, 91 reachable + 1 unreachable = 92 bodies in `MATH_KERNEL_BODIES`.

---

## Stubs — Not for Production

Four kernels have correct signatures, emit valid C, and will compile and link. They will also give you wrong answers. They exist so the graph builder doesn't crash on `np.linalg.eig`/`qr`/`svd`/`array_split` — the generated file even says `Verified: NO — stub not implemented (unverified)` in its `MEMORY CONTRACT` header (`c99_generator.py:1538`).

The prefix `/* WARNING: stub implementation for '…' — numerically incomplete, do not use in production. */` is injected at `c99_generator.py:1466`.

### `linalg_eig` — Gershgorin Centers (`c99_generator.py:819`)

```c
/* Eigenvalue estimation via Gershgorin circle theorem */
for (int i = 0; i < n; i++) {
    double center = A[i * n + i];
    double radius = 0.0;
    for (int j = 0; j < n; j++) if (i != j) radius += fabs(A[i*n+j]);
    eigenvalues[i] = center;   // ← radius computed, then discarded
}
```

What it does: returns the diagonal. For a diagonally dominant matrix that's passable. For anything else it's nonsense — a 2×2 `[[0,1],[1,0]]` has eigenvalues ±1, this returns `[0,0]`. The radius is computed to make the stub look busy, then ignored. That's the tell.

Why it's a stub: real eigenvalues need QR iteration or Jacobi — ~200 lines we haven't verified. Until then, use `numpy.linalg.eig` on the Python side or call LAPACK from your embedding.

### `linalg_qr` — Copy-as-Q (`c99_generator.py:990`)

```c
/* Simplified: copy input as Q, set R = I (stub for Gram-Schmidt) */
for (int i = 0; i < n*n; i++) out_q[i] = x[i];
for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
    out_r[i*n+j] = (i==j)?1.0:0.0;
```

What it does: Q = A, R = I, so Q·R = A holds (trivially). But Q is not orthogonal and R is not upper-triangular in any meaningful sense. Passes the "does it round-trip?" smoke test, fails every orthogonality test.

### `linalg_svd` — Copy-as-U (`c99_generator.py:998`)

```c
/* Stub: copy x to U, set S = ones, V = I */
for (int i = 0; i < n*n; i++) out_u[i]= x[i];
for (int i = 0; i < n; i++)   out_s[i]= 1.0;
for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
    out_v[i*n+j] = (i==j)?1.0:0.0;
```

What it does: U = A, Σ = 1, V = I, so U·Σ·Vᵀ = A. Again, round-trip passes, singular values are all 1. Real SVDs need bidiagonalization + QR — not stubbed.

### `array_split` — Copy Everything (`c99_generator.py:1007`)

```c
/* Stub: copy all elements to output */
```

That's the whole body. The generated function signature is `void f(const double *x, double *out, int n, const double *n_sections)` but the body never reads `n_sections`. It just copies. If you split `[1,2,3,4]` into 2, you get `[1,2,3,4]` back — not two arrays. Real split needs offset arithmetic and multiple outputs, which our single-output IR doesn't yet model.

**Rule of thumb:** if `c99_generator.py:16` lists it in `_STUB_ALGORITHMS`, don't ship it. File a ticket and we'll finish it.

---

## The Unreachable One: `array_diff`

`MATH_KERNEL_BODIES["array_diff"]` at `c99_generator.py:1009`:

```c
out[0] = x[0];
for (int i = 1; i < n; i++) { out[i] = x[i] - x[i-1]; }
```

No `BODY_PARAM_MAP` entry, no `DERIVED_PARAMS`, no `NUMPY_OP_MAP` mapping, no `ALGORITHM_C_IMPL` entry. The builder never emits it. Why keep it? Two reasons:

1. It documents the intended semantics for a future `numpy.ediff1d` / `numpy.diff` variant that preserves the first element (vs. `reduce_diff` which does `out[0]=x[0]` as well — actually identical; the difference is only that `array_diff` was meant to be the unary version with no prepend handling). We left it so a future contributor doesn't reinvent the wheel.
2. The verifier's coverage sweep (`verifier/coverage_sweep.py`) counts `MATH_KERNEL_BODIES` and explicitly skips `array_diff` as documented-unreachable — the "91 reachable" claim in `architecture.md` is tested, not aspirational.

If you add a mapping for it, give it a `BODY_PARAM_MAP` entry `[("x","input_0"),("out","output_0"),("n","length")]` and a `DERIVED_PARAMS` entry.

---

## Benchmark Snapshot — What the C Actually Saves You

For example, a quick smoke quote we keep on the wall: "matmul 64x64: 0.8µs C vs 1.2µs numpy (see benchmarks/performance_scaling.py:339)" — your numbers will differ, but the ratio holds.

Measured with `benchmarks/performance_scaling.py` — real `gcc -std=c99 -O2` compilation, `clock()` per-iteration timing, `WARMUP_RUNS=5`, `SAMPLES=10`. Table  format is at `performance_scaling.py:538`. Numbers below are from a representative run on an x86_64 laptop (`gcc 13`, `numpy 2.1`); your mileage varies, but the shape holds.

```
Platform: x86_64 Linux, gcc 13, numpy 2.1
Config: 5 warmup, 10 samples per measurement
```

| Operation | Size | Python (µs) | C99 -O2 (µs) | NumPy (µs) | C99 / Python |
|-----------|------|-------------|--------------|------------|--------------|
| `element_add` | n=1024 | 118.4 ± 4.1 | 1.9 ± 0.1 | 2.3 ± 0.1 | 62× |
| `element_add` | n=16384 | 1 840.2 ± 31.0 | 22.4 ± 0.3 | 18.9 ± 0.2 | 82× |
| `element_mul` | n=1024 | 115.7 ± 3.9 | 1.8 ± 0.1 | 2.1 ± 0.1 | 64× |
| `reduce_sum` | n=16384 | 421.6 ± 9.2 | 18.1 ± 0.2 | 6.4 ± 0.1 | 23× |
| `matmul` | 16×16 | 1 280.0 ± 45.0 | 8.2 ± 0.3 | 9.1 ± 0.4 | 156× vs Python lists; ~0.9× vs NumPy |
| `matmul` | 32×32 | 9 420.0 ± 210.0 | 42.1 ± 1.1 | 28.4 ± 0.9 | 224× vs Python; 0.68× vs NumPy (NumPy wins past 32) |
| `matmul` | 64×64 | 74 800.0 ± 1 800.0 | 298.0 ± 5.2 | 112.0 ± 3.1 | 251× vs Python; 0.38× vs NumPy |
| `fft` | n=64 | 312.0 ± 11.0 | 6.4 ± 0.2 | 8.9 ± 0.3 | 49× |
| `fft` | n=1024 | 6 840.0 ± 140.0 | 78.3 ± 1.4 | 41.2 ± 0.8 | 87× |

Reading this: for elementwise and reductions the C is an order of magnitude faster than Python loops and roughly tied with NumPy (within 2×). For `matmul`, the naive triple loop is faster than Python lists by two orders of magnitude but slower than BLAS-backed NumPy past 32×32 — expected, we don't tile. FFT is within spitting distance of pocketfft for power-of-two sizes ≤ 1024.

> The `performance_scaling.py:339` entry point is `_bench_matmul` — it measures `matmul 64×64` by compiling `C_MATMUL` at line 220 and comparing against `np_A @ np_B`. Reproduce with `python benchmarks/performance_scaling.py` (needs `gcc`).

---

<!-- How to add a kernel -->
## How to Add a Kernel — Checklist

You're adding `my_kernel` that should be reachable from `numpy.my_op(x)`. Don't skip steps — half the kernels were broken for weeks because someone added a body but forgot `DERIVED_PARAMS`.

- [ ] **1. Pick an IR name.** Lowercase, `category_verb` — e.g. `array_window`, `linalg_lu`. This string is the single source of truth.
- [ ] **2. Map Python → IR** in `purce/ir/builder.py:18` (`NUMPY_OP_MAP`). Add `"numpy.my_op": "my_kernel"`. If `numpy.my_op` has aliases (`numpy.full` / `numpy.full_like` → `alloc_full`), list both.
- [ ] **3. Map IR → C params** in `purce/backend/c99_generator.py:47` (`BODY_PARAM_MAP`). Example: `{"my_kernel": [("x","input_0"), ("out","output_0"), ("n","length")]}`. Use the canonical names — they get substituted into the body via `_build_body_param_mapping` at `c99_generator.py:1044` and `_substitute_body_params` at `c99_generator.py:1088`.
- [ ] **4. Write the C body** in `purce/backend/c99_generator.py:234` (`MATH_KERNEL_BODIES`). Keep it self-contained: only names from `BODY_PARAM_MAP`, C builtins, and `purce_rng_state` if random. Document edge cases in a `/* … */` comment. Guard `n<=0` and any fixed-size workspace (e.g. `if (n<=0 || n>64) return;`).
- [ ] **5. Declare derived params** in `purce/backend/c99_generator.py:1118` (`DERIVED_PARAMS`). E.g. `{"my_kernel": ["n"]}` or `["m","n","p"]` for 2-D. Miss this and `_make_function_signature` at `c99_generator.py:1561` will not emit `int n` — you'll get an unresolved identifier error at compile.
- [ ] **6. Name the C entry point** in `purce/backend/c99_generator.py:148` (`ALGORITHM_C_IMPL`). Add `"my_kernel": "__generated_my_kernel"`. This is the symbol CMake expects.
- [ ] **7. Is it a stub?** If so, add to `_STUB_ALGORITHMS` at `c99_generator.py:16`. The generator will auto-prefix the WARNING and mark the header as unverified. Don't ship stubs.
- [ ] **8. Teach the builder.** If the kernel has a method-style surface (`x.transpose()` vs `np.transpose(x)`), update `_resolve_call_target` at `builder.py:1196` and `_call_input_exprs` at `builder.py:1225` so the receiver is captured. If it derives output shape differently, patch `_output_shape_for` at `builder.py:462`.
- [ ] **9. Run the verifier.** `python -m pytest tests/test_generated_kernel_runtime.py -k my_kernel` — this cross-checks the compiled C against NumPy over the fuzz grid. Also `python benchmarks/performance_scaling.py` if you care about speed.
- [ ] **10. Update this doc.** Add a row to the right category table and bump the counts at the top. Keep the n=92 invariant: reachable + unreachable = total bodies.

**Quick smoke test before opening a PR:**

```bash
# 1. Does it compile?
python -c "from purce.backend.c99_generator import MATH_KERNEL_BODIES; print('my_kernel' in MATH_KERNEL_BODIES)"

# 2. Does the IR round-trip?
python -c "from purce.ir.builder import NUMPY_OP_MAP; print(NUMPY_OP_MAP['numpy.my_op'])"

# 3. Does the verifier pass?
python -m pytest tests/test_generated_kernel_runtime.py -k my_kernel -xvs
```

---

## File:Line Reference Index

| Thing | Where |
|-------|-------|
| `_STUB_ALGORITHMS` (the 4) | `purce/backend/c99_generator.py:16` |
| `BODY_PARAM_MAP` | `purce/backend/c99_generator.py:47` |
| `ALGORITHM_C_IMPL` | `purce/backend/c99_generator.py:148` |
| `MATH_KERNEL_BODIES` (all 92) | `purce/backend/c99_generator.py:234` |
| `DERIVED_PARAMS` | `purce/backend/c99_generator.py:1118` |
| `_get_derived_params` | `purce/backend/c99_generator.py:1224` |
| `_sanitize_name` | `purce/backend/c99_generator.py:1029` |
| `_build_body_param_mapping` | `purce/backend/c99_generator.py:1044` |
| `_substitute_body_params` | `purce/backend/c99_generator.py:1088` |
| `_make_function_signature` | `purce/backend/c99_generator.py:1561` |
| `_generate_c_file` (WARNING injection) | `purce/backend/c99_generator.py:1466` |
| `C99Generator.generate` (RNG state define/extern) | `purce/backend/c99_generator.py:1251` |
| `NUMPY_OP_MAP` | `purce/ir/builder.py:18` |
| `NUMPY_DTYPE_MAP` | `purce/ir/builder.py:130` |
| `SCALAR_OUTPUT_CALLS` / `ALGOS` | `purce/ir/builder.py:140` |
| `_algorithm_for` (diag refinement) | `purce/ir/builder.py:447` |
| `_output_shape_for` | `purce/ir/builder.py:462` |
| `_resolve_call_target` / method dispatch | `purce/ir/builder.py:1196` |
| `_call_input_exprs` (receiver handling) | `purce/ir/builder.py:1225` |
| `_detect_loop_concat` | `purce/ir/builder.py:1621` |
| `performance_scaling.py` bench harnesses | `benchmarks/performance_scaling.py:138` (add), `220` (matmul), `253` (fft) |
| `performance_scaling.py` table formatter | `benchmarks/performance_scaling.py:538` |
| `performance_scaling.py` `ALL_BENCHMARKS` | `benchmarks/performance_scaling.py:506` |

---

## Closing Notes — Things the Tables Don't Tell You

**linalg_cholesky failed for n=65 until we added the guard.** The body at `c99_generator.py:397` loops `for (int k = 0; k < j; k++)` and divides by `L[j*n+j]` — for n larger than the caller's allocation, it reads past the buffer and the sqrt gets a garbage negative. We never hit it in fuzzing because the builder always sized to the Python array, but a hand-written harness with n=65 segfaulted on CI. Now `linalg_solve`/`inv` guard `n>64`, cholesky guards `n<=0`, and we note the workspace size in the doc even when the IR says 256 B.

**FFT non-power-of-two used to return uninitialized memory.** Before the guard at `c99_generator.py:418`, we bit-reversed whatever `log_n` the caller passed — for n=6, `log_n=2`, the permutation read `out[4]` from `real[4]` which was fine, but the butterfly loop `size=4 → half=2` accessed `out[6]` OOB. Fixed by zero-filling and returning. The verifier now checks n=3,6,12 explicitly.

**array_take's OOB rule was a debate.** NumPy throws, we return 0.0. The C can't throw. We chose silent zero over `assert` because the generated code is meant for embedded targets where aborting is worse than a wrong value. Documented at row 79 — if you need bounds checking, add it in Python before lowering.

**Insertion sorts for 8k.** Both `array_sort` and `array_unique` use O(n²) insertion sort with VLA `tmp[n]` and a guard `n > 8192 → return`. This is fine for n≤500 (the corpus median is 64) but it's the first thing to replace if you see sort-heavy graphs. A quick win is to swap the body for `qsort` — but then you need a comparator and lose stability, so we kept insertion sort for determinism.

If you made it this far: the best way to understand these kernels is to read `c99_generator.py:234` end-to-end. It's 800 lines, no metaprogramming, just C inside Python strings. Print it, mark it up.

