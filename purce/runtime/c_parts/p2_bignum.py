"""Tier-R runtime, part 2: arbitrary-precision integer arithmetic."""

P2 = r"""
/* ══════════════════════════════════════════════════════════════════
 * bignum (base 2^30)
 * ══════════════════════════════════════════════════════════════════ */
static int big_resize(pst *S, pbig *a, uint32_t n) {
    if (n == a->n) return 1;
    if (n == 0) {
        free(a->d);
        a->d = NULL;
        a->n = 0;
        return 1;
    }
    uint32_t *np = (uint32_t *)p_alloc(a->d, (size_t)n * sizeof(uint32_t));
    if (!np) { p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
    if (n > a->n) memset(np + a->n, 0, (size_t)(n - a->n) * sizeof(uint32_t));
    a->d = np;
    a->n = n;
    return 1;
}

static void big_norm(pbig *a) {
    while (a->n && a->d[a->n - 1] == 0) a->n--;
    if (!a->n) a->sign = 0;
}

static void big_copy(pst *S, pbig *dst, const pbig *src) {
    dst->sign = src->sign;
    if (!big_resize(S, dst, src->n)) return;
    if (src->n) memcpy(dst->d, src->d, (size_t)src->n * sizeof(uint32_t));
}

/* magnitude compare: -1 / 0 / 1 */
static int big_ucmp(const pbig *a, const pbig *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (uint32_t i = a->n; i-- > 0;) {
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    }
    return 0;
}

static int big_cmp(const pbig *a, const pbig *b) {
    if (a->sign != b->sign) return a->sign ? -1 : 1;
    int r = big_ucmp(a, b);
    return a->sign ? -r : r;
}

/* dst = |a| + |b| */
static void big_add_mag(pst *S, pbig *dst, const pbig *a, const pbig *b) {
    uint32_t n = (a->n > b->n ? a->n : b->n) + 1;
    if (!big_resize(S, dst, n)) return;
    uint64_t carry = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t s = carry;
        if (i < a->n) s += a->d[i];
        if (i < b->n) s += b->d[i];
        dst->d[i] = (uint32_t)(s & 0x3FFFFFFFull);
        carry = s >> 30;
    }
    big_norm(dst);
}

/* dst = |a| - |b|, requires |a| >= |b| */
static void big_sub_mag(pst *S, pbig *dst, const pbig *a, const pbig *b) {
    if (!big_resize(S, dst, a->n ? a->n : 1)) return;
    int64_t borrow = 0;
    for (uint32_t i = 0; i < a->n; i++) {
        int64_t d = (int64_t)a->d[i] - borrow;
        if (i < b->n) d -= b->d[i];
        if (d < 0) { d += (int64_t)0x40000000u; borrow = 1; } else borrow = 0;
        dst->d[i] = (uint32_t)d;
    }
    dst->n = a->n;
    big_norm(dst);
}

/* dst = |a| * |b| (schoolbook) */
static void big_mul_mag(pst *S, pbig *dst, const pbig *a, const pbig *b) {
    if (!a->n || !b->n) { dst->n = 0; dst->sign = 0; return; }
    uint32_t n = a->n + b->n;
    if (!big_resize(S, dst, n)) return;
    memset(dst->d, 0, (size_t)n * sizeof(uint32_t));
    for (uint32_t i = 0; i < a->n; i++) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < b->n; j++) {
            uint64_t t = (uint64_t)a->d[i] * b->d[j] + dst->d[i + j] + carry;
            dst->d[i + j] = (uint32_t)(t & 0x3FFFFFFFull);
            carry = t >> 30;
        }
        uint32_t k = i + b->n;
        while (carry && k < n) {
            uint64_t t = (uint64_t)dst->d[k] + carry;
            dst->d[k] = (uint32_t)(t & 0x3FFFFFFFull);
            carry = t >> 30;
            k++;
        }
    }
    big_norm(dst);
}

static uint32_t big_bitlen(const pbig *a) {
    if (!a->n) return 0;
    uint32_t top = a->d[a->n - 1];
    uint32_t lz = 0;
    for (uint32_t m = 0x20000000u; m; m >>= 1) {
        if (top & m) break;
        lz++;
    }
    return a->n * 30 - lz;
}

static int big_getbit(const pbig *a, uint32_t i) {
    uint32_t limb = i / 30, off = i % 30;
    if (limb >= a->n) return 0;
    return (a->d[limb] >> off) & 1;
}

/* bitwise long division on magnitudes: q=|num|/|den|, r=|num|%|den|.
 * Requires den > 0; returns 1 on success. */
static int big_divmod_mag(pst *S, const pbig *num, const pbig *den, pbig *q, pbig *r) {
    uint32_t nbits = big_bitlen(num), dbits = big_bitlen(den);
    if (dbits == 0) return 0;
    if (nbits == 0 || dbits > nbits) {
        q->n = 0; q->sign = 0;
        big_copy(S, r, num);
        return 1;
    }
    uint32_t nlim = (nbits + 29) / 30 + 1;
    if (!big_resize(S, q, nlim)) return 0;
    if (!big_resize(S, r, nlim)) return 0;
    memset(q->d, 0, (size_t)q->n * sizeof(uint32_t));
    memset(r->d, 0, (size_t)r->n * sizeof(uint32_t));
    r->n = 0;
    r->sign = 0;
    for (uint32_t i = nbits; i-- > 0;) {
        uint32_t car = big_getbit(num, i);
        for (uint32_t k = 0; k < r->n; k++) {
            uint32_t nv = (uint32_t)(((uint64_t)r->d[k] << 1) | car);
            car = nv >> 30;
            r->d[k] = nv & 0x3FFFFFFFull;
        }
        if (car) {
            if (r->n >= nlim) return 0;
            r->d[r->n++] = 1;
        }
        if (big_ucmp(r, den) >= 0) {
            pbig tmp;
            tmp.n = 0; tmp.sign = 0; tmp.d = NULL;
            big_copy(S, &tmp, r);
            big_sub_mag(S, r, &tmp, den);
            free(tmp.d);
            q->d[i / 30] |= (uint32_t)1 << (i % 30);
        }
    }
    big_norm(q);
    big_norm(r);
    return 1;
}

/* q = a / d (magnitudes), remainder returned via rem; d < 2^30 */
static int big_divrem_small(pst *S, const pbig *a, uint64_t d, pbig *q, uint64_t *rem) {
    uint64_t r = 0;
    if (q && !big_resize(S, q, a->n)) return 0;
    for (uint32_t i = a->n; i-- > 0;) {
        uint64_t cur = (r << 30) | a->d[i];
        uint64_t qi = cur / d;
        r = cur % d;
        if (q) q->d[i] = (uint32_t)qi;
    }
    if (q) big_norm(q);
    *rem = r;
    return 1;
}

static int big_is_zero(const pbig *a) { return a->n == 0; }

static pval *p_newint(pst *S) {
    pval *v = p_newobj(S, P_INT);
    if (!v) return NULL;
    v->u.i.n = 0;
    v->u.i.sign = 0;
    v->u.i.d = NULL;
    return v;
}

static pval *p_int_from_u64(pst *S, uint64_t x) {
    pval *v = p_newint(S);
    if (!v) return NULL;
    if (x == 0) return v;
    uint32_t a = (uint32_t)(x & 0x3FFFFFFFull);
    uint32_t b = (uint32_t)((x >> 30) & 0x3FFFFFFFull);
    uint32_t c = (uint32_t)(x >> 60);
    uint32_t n = c ? 3 : (b ? 2 : 1);
    if (!big_resize(S, &v->u.i, n)) return NULL;
    v->u.i.d[0] = a;
    if (n > 1) v->u.i.d[1] = b;
    if (n > 2) v->u.i.d[2] = c;
    big_norm(&v->u.i);
    return v;
}

static pval *p_int_from_long(pst *S, long long x) {
    if (x < 0) {
        pval *v = p_int_from_u64(S, (uint64_t)(-(x + 1)) + 1);
        if (v) v->u.i.sign = 1;
        return v;
    }
    return p_int_from_u64(S, (uint64_t)x);
}

static pval *p_int_neg(pst *S, pval *v) {
    pval *r = p_newint(S);
    if (!r) return NULL;
    big_copy(S, &r->u.i, &v->u.i);
    if (r->u.i.n) r->u.i.sign = !v->u.i.sign;
    return r;
}

static pval *p_int_abs(pst *S, pval *v) {
    pval *r = p_newint(S);
    if (!r) return NULL;
    big_copy(S, &r->u.i, &v->u.i);
    r->u.i.sign = 0;
    return r;
}

static pval *p_int_add(pst *S, pval *a, pval *b) {
    pval *r = p_newint(S);
    if (!r) return NULL;
    if (a->u.i.sign == b->u.i.sign) {
        big_add_mag(S, &r->u.i, &a->u.i, &b->u.i);
        r->u.i.sign = a->u.i.sign;
    } else {
        int c = big_ucmp(&a->u.i, &b->u.i);
        if (c == 0) return r;
        if (c > 0) { big_sub_mag(S, &r->u.i, &a->u.i, &b->u.i); r->u.i.sign = a->u.i.sign; }
        else { big_sub_mag(S, &r->u.i, &b->u.i, &a->u.i); r->u.i.sign = b->u.i.sign; }
    }
    return r;
}

static pval *p_int_sub(pst *S, pval *a, pval *b) {
    pval *nb = p_int_neg(S, b);
    if (!nb) return NULL;
    return p_int_add(S, a, nb);
}

static pval *p_int_mul(pst *S, pval *a, pval *b) {
    pval *r = p_newint(S);
    if (!r) return NULL;
    big_mul_mag(S, &r->u.i, &a->u.i, &b->u.i);
    r->u.i.sign = a->u.i.sign ^ b->u.i.sign;
    return r;
}

/* floor division and modulo (Python semantics) */
static int p_int_divmod(pst *S, pval *a, pval *b, pval **qo, pval **ro) {
    if (b->u.i.n == 0) {
        p_seterr(S, PE_ZERO, "integer division or modulo by zero");
        return 0;
    }
    pbig qm, rm;
    qm.n = 0; qm.sign = 0; qm.d = NULL;
    rm.n = 0; rm.sign = 0; rm.d = NULL;
    if (!big_divmod_mag(S, &a->u.i, &b->u.i, &qm, &rm)) {
        free(qm.d);
        free(rm.d);
        p_seterr(S, PE_RUNTIME, "division failed");
        return 0;
    }
    int sa = a->u.i.sign, sb = b->u.i.sign;
    pval *q = NULL, *r = NULL;
    if (sa == sb) {
        q = p_newint(S);
        r = p_newint(S);
        if (!q || !r) { free(qm.d); free(rm.d); return 0; }
        big_copy(S, &q->u.i, &qm);
        big_copy(S, &r->u.i, &rm);
        r->u.i.sign = sb;
    } else if (rm.n == 0) {
        q = p_newint(S);
        r = p_newint(S);
        if (!q || !r) { free(qm.d); free(rm.d); return 0; }
        big_copy(S, &q->u.i, &qm);
        q->u.i.sign = 1;
    } else {
        pbig one, qa;
        one.n = 1; one.sign = 0;
        one.d = (uint32_t *)malloc(sizeof(uint32_t));
        qa.n = 0; qa.sign = 0; qa.d = NULL;
        if (!one.d) { free(qm.d); free(rm.d); p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
        one.d[0] = 1;
        big_add_mag(S, &qa, &qm, &one);
        free(one.d);
        q = p_newint(S);
        r = p_newint(S);
        if (!q || !r) { free(qm.d); free(rm.d); free(qa.d); return 0; }
        big_copy(S, &q->u.i, &qa);
        q->u.i.sign = 1;
        free(qa.d);
        big_sub_mag(S, &r->u.i, &b->u.i, &rm);
        r->u.i.sign = sb;
    }
    free(qm.d);
    free(rm.d);
    *qo = q;
    *ro = r;
    return 1;
}

/* exponentiation (exp >= 0) */
static pval *p_int_pow_nonneg(pst *S, pval *base, uint64_t exp) {
    if (exp == 0) return p_int_from_long(S, 1);
    pval *result = p_int_from_long(S, 1);
    pval *b = p_int_abs(S, base);
    if (!result || !b) return NULL;
    uint64_t e = exp;
    while (e) {
        if (e & 1) {
            pval *nr = p_int_mul(S, result, b);
            if (!nr) return NULL;
            result = nr;
        }
        e >>= 1;
        if (e) {
            pval *nb = p_int_mul(S, b, b);
            if (!nb) return NULL;
            b = nb;
        }
    }
    if (base->u.i.sign && (exp & 1)) return p_int_neg(S, result);
    return result;
}

static double p_int_to_double(pst *S, pval *v) {
    (void)S;
    if (!v->u.i.n) return 0.0;
    double x = 0.0;
    for (uint32_t i = v->u.i.n; i-- > 0;)
        x = x * 1073741824.0 + (double)v->u.i.d[i];
    return v->u.i.sign ? -x : x;
}

/* decimal digits (most-significant first) */
static void big_to_decimal(pst *S, const pbig *a, char *out, size_t outcap,
                           size_t *outlen) {
    pbig work;
    work.n = 0; work.sign = 0; work.d = NULL;
    big_copy(S, &work, a);
    /* upper bound: 10 decimal digits per 30-bit limb */
    size_t maxd = (size_t)work.n * 10 + 2;
    char *buf = (char *)malloc(maxd + 1);
    if (!buf) {
        free(work.d);
        if (outcap) out[0] = 0;
        if (outlen) *outlen = 0;
        return;
    }
    char *pos = buf + maxd;
    *pos = 0;
    /* chunks come out least-significant first: write each one from the
     * end of the buffer so the final string is most-significant first
     * with no per-chunk digit reversal. */
    while (work.n) {
        uint64_t rem;
        if (!big_divrem_small(S, &work, 1000000000ull, &work, &rem)) break;
        int top = work.n == 0;
        char c[16];
        int k = snprintf(c, sizeof(c), top ? "%llu" : "%09llu",
                         (unsigned long long)rem);
        pos -= (size_t)k;
        memcpy(pos, c, (size_t)k);
    }
    free(work.d);
    size_t len = (size_t)(buf + maxd - pos);
    size_t cap = len < outcap ? len : (outcap ? outcap - 1 : 0);
    if (cap) memcpy(out, pos, cap);
    if (outcap) out[cap] = 0;
    if (outlen) *outlen = len;
}

/* parse decimal string into pval int; returns 0 on failure */
static int p_int_from_str(pst *S, const char *s, size_t len, pval **out) {
    size_t i = 0;
    int neg = 0;
    if (i < len && (s[i] == '-' || s[i] == '+')) { neg = s[i] == '-'; i++; }
    if (i >= len) return 0;
    pval *acc = p_int_from_long(S, 0);
    if (!acc) return 0;
    for (; i < len; i++) {
        char c = s[i];
        if (c < '0' || c > '9') return 0;
        pval *ten = p_int_from_long(S, 10);
        pval *na = p_int_mul(S, acc, ten);
        pval *d = p_int_from_long(S, (long long)(c - '0'));
        pval *nb = p_int_add(S, na, d);
        if (!ten || !na || !d || !nb) return 0;
        acc = nb;
    }
    if (neg) {
        pval *r = p_int_neg(S, acc);
        *out = r;
        return r != NULL;
    }
    *out = acc;
    return 1;
}
"""
