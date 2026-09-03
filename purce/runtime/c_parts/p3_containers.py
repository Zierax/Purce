"""Tier-R runtime, part 3: strings, containers, equality/ordering, truth.

Numeric comparisons mix int/float/bool; pure int/int paths stay exact on
bignums, int-vs-float falls back to ``double`` (documented, matches speed).
"""

P3 = r"""
static pval *p_len_int(pst *S, pval *v);
static pval *p_range_len(pst *S, pval *r);

/* ══════════════════════════════════════════════════════════════════
 * strings
 * ══════════════════════════════════════════════════════════════════ */
static int p_str_append(pst *S, pstr *s, const char *t, size_t n) {
    size_t need = s->len + n + 1;
    if (need > s->cap) {
        size_t ncap = s->cap ? s->cap : 32;
        while (ncap < need) ncap *= 2;
        char *np = (char *)p_alloc(s->s, ncap);
        if (!np) { p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
        s->s = np;
        s->cap = ncap;
    }
    if (n) memcpy(s->s + s->len, t, n);
    s->len += n;
    s->s[s->len] = 0;
    return 1;
}

static pval *p_newstr(pst *S, const char *t, size_t len) {
    pval *v = p_newobj(S, P_STR);
    if (!v) return NULL;
    v->u.s.s = NULL;
    v->u.s.len = 0;
    v->u.s.cap = 0;
    if (len && !p_str_append(S, &v->u.s, t, len)) return NULL;
    return v;
}

static pval *p_newstrn(pst *S, const char *t) {
    return p_newstr(S, t, strlen(t));
}

static pval *p_newfloat(pst *S, double x) {
    pval *v = p_newobj(S, P_FLOAT);
    if (!v) return NULL;
    v->u.f = x;
    return v;
}

static pval *p_newbool(pst *S, int x) {
    pval *v = p_newobj(S, P_BOOL);
    if (!v) return NULL;
    v->u.b = x ? 1 : 0;
    return v;
}

/* one-character string at index i (caller guarantees bounds) */
static pval *p_str_chat(pst *S, pval *v, size_t i) {
    return p_newstr(S, v->u.s.s + i, 1);
}

static pval *p_str_concat(pst *S, pval *a, pval *b) {
    pval *v = p_newobj(S, P_STR);
    if (!v) return NULL;
    v->u.s.s = NULL; v->u.s.len = 0; v->u.s.cap = 0;
    if (!p_str_append(S, &v->u.s, a->u.s.s ? a->u.s.s : "", a->u.s.len)) return NULL;
    if (!p_str_append(S, &v->u.s, b->u.s.s ? b->u.s.s : "", b->u.s.len)) return NULL;
    return v;
}

/* substring search: returns index or -1 */
static int p_str_isub(const pstr *h, const pstr *n) {
    if (!n->len) return 0;
    if (n->len > h->len) return -1;
    for (size_t i = 0; i + n->len <= h->len; i++)
        if (!memcmp(h->s + i, n->s, n->len)) return (int)i;
    return -1;
}

/* ══════════════════════════════════════════════════════════════════
 * int <-> str (exact)
 * ══════════════════════════════════════════════════════════════════ */
static void p_int_into_str(pst *S, pval *v, pstr *out) {
    if (v->u.i.sign && v->u.i.n)
        p_str_append(S, out, "-", 1);
    if (!v->u.i.n) {
        p_str_append(S, out, "0", 1);
        return;
    }
    size_t cap = (size_t)v->u.i.n * 11 + 8;
    char *dig = (char *)malloc(cap);
    if (!dig) { p_seterr(S, PE_RUNTIME, "out of memory"); return; }
    size_t n = 0;
    big_to_decimal(S, &v->u.i, dig, cap, &n);
    p_str_append(S, out, dig, n);
    free(dig);
}

static pval *p_str_to_int(pst *S, pval *s) {
    pval *r = NULL;
    if (!p_int_from_str(S, s->u.s.s ? s->u.s.s : "", s->u.s.len, &r))
        return NULL;
    return r;
}

/* ══════════════════════════════════════════════════════════════════
 * sequences (list/tuple): index / concat / membership
 * ══════════════════════════════════════════════════════════════════ */
static int p_seq_append(pst *S, pval *seq, pval *x) {
    pseq *q = &seq->u.l;
    if (q->len >= q->cap) {
        size_t ncap = q->cap ? q->cap * 2 : 8;
        pval **np = (pval **)p_alloc(q->items, ncap * sizeof(pval *));
        if (!np) { p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
        q->items = np;
        q->cap = ncap;
    }
    q->items[q->len++] = x;
    return 1;
}

/* Python-style index: folds negatives; returns -1 when out of range,
 * -2 when the key is not an int (error set). */
static long p_seq_ndex(pst *S, pval *seq, pval *key) {
    long long idx;
    if (key->tag == P_BOOL) {
        idx = key->u.b;
    } else if (key->tag == P_INT) {
        if (key->u.i.n == 0) idx = 0;
        else if (key->u.i.n > 1) {
            p_seterr(S, PE_TYPE, "sequence index too large");
            return -2;
        } else {
            idx = (long long)key->u.i.d[0];
            if (key->u.i.sign) idx = -idx;
        }
    } else {
        p_seterr(S, PE_TYPE, "sequence indices must be integers");
        return -2;
    }
    long long i = idx < 0 ? idx + (long long)seq->u.l.len : idx;
    if (i < 0 || i >= (long long)seq->u.l.len) return -1;
    return (long)i;
}

static pval *p_seq_get(pst *S, pval *seq, pval *key) {
    long li = p_seq_ndex(S, seq, key);
    if (li == -1)
        p_seterr(S, PE_INDEX, "%s index out of range",
                 seq->tag == P_LIST ? "list" : "tuple");
    if (li < 0) return NULL;
    return seq->u.l.items[li];
}

static int p_seq_set(pst *S, pval *seq, pval *key, pval *val) {
    long li = p_seq_ndex(S, seq, key);
    if (li == -1)
        p_seterr(S, PE_INDEX, "%s index out of range",
                 seq->tag == P_LIST ? "list" : "tuple");
    if (li < 0) return 0;
    seq->u.l.items[li] = val;
    return 1;
}

static pval *p_seq_concat(pst *S, pval *a, pval *b) {
    pval *v = p_newobj(S, a->tag);
    if (!v) return NULL;
    for (size_t i = 0; i < a->u.l.len; i++)
        if (!p_seq_append(S, v, a->u.l.items[i])) return NULL;
    for (size_t i = 0; i < b->u.l.len; i++)
        if (!p_seq_append(S, v, b->u.l.items[i])) return NULL;
    return v;
}

static int p_seq_contain(pst *S, pval *seq, pval *x) {
    for (size_t i = 0; i < seq->u.l.len; i++)
        if (p_eq(S, seq->u.l.items[i], x)) return 1;
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * dict (insertion-order; equality probing)
 * ══════════════════════════════════════════════════════════════════ */
static long p_dict_find(pdict *d, pval *k) {
    for (size_t i = 0; i < d->len; i++)
        if (p_eq(NULL, d->k[i], k)) return (long)i;
    return -1;
}

static int p_dict_set(pst *S, pdict *d, pval *k, pval *v) {
    long i = p_dict_find(d, k);
    if (i >= 0) { d->v[i] = v; return 1; }
    if (d->len >= d->cap) {
        size_t ncap = d->cap ? d->cap * 2 : 8;
        pval **nk = (pval **)p_alloc(d->k, ncap * sizeof(pval *));
        if (!nk) { p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
        pval **nv = (pval **)p_alloc(d->v, ncap * sizeof(pval *));
        if (!nv) { free(nk); p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
        d->k = nk;
        d->v = nv;
        d->cap = ncap;
    }
    d->k[d->len] = k;
    d->v[d->len] = v;
    d->len++;
    return 1;
}

static pval *p_dict_get(pst *S, pdict *d, pval *k) {
    long i = p_dict_find(d, k);
    if (i < 0) { p_seterr(S, PE_KEY, "key not in dict"); return NULL; }
    return d->v[i];
}

static pval *p_dict_getdef(pst *S, pdict *d, pval *k, pval *def) {
    (void)S;
    long i = p_dict_find(d, k);
    return i < 0 ? def : d->v[i];
}

static int p_dict_contains(pdict *d, pval *k) {
    return p_dict_find(d, k) >= 0;
}

/* ══════════════════════════════════════════════════════════════════
 * numeric comparison bridge (int/bool/float)
 * ══════════════════════════════════════════════════════════════════ */
static double p_num_double(pst *S, pval *v) {
    switch (v->tag) {
        case P_BOOL: return v->u.b ? 1.0 : 0.0;
        case P_INT:  return p_int_to_double(S, v);
        default:     return v->u.f;
    }
}

/* both are int/bool → exact bigint compare; otherwise double. NaN: 0. */
static int p_num_cmpall(pst *S, pval *a, pval *b, long *out) {
    int at = a->tag == P_INT || a->tag == P_BOOL || a->tag == P_FLOAT;
    int bt = b->tag == P_INT || b->tag == P_BOOL || b->tag == P_FLOAT;
    if (!at || !bt) return 0;
    if ((a->tag == P_INT || a->tag == P_BOOL) &&
        (b->tag == P_INT || b->tag == P_BOOL)) {
        pbig ba, bb;
        uint32_t la[1], lb[1];
        if (a->tag == P_BOOL) {
            ba.n = a->u.b ? 1 : 0; ba.sign = 0; ba.d = la;
            if (ba.n) la[0] = 1;
        } else ba = a->u.i;
        if (b->tag == P_BOOL) {
            bb.n = b->u.b ? 1 : 0; bb.sign = 0; bb.d = lb;
            if (bb.n) lb[0] = 1;
        } else bb = b->u.i;
        *out = big_cmp(&ba, &bb);
        return 1;
    }
    double xa = p_num_double(S, a), xb = p_num_double(S, b);
    if (xa < xb) *out = -1;
    else if (xa > xb) *out = 1;
    else *out = 0;
    return 1;
}

/* ══════════════════════════════════════════════════════════════════
 * equality / ordering / truth
 * ══════════════════════════════════════════════════════════════════ */
static int p_eq(pst *S, pval *a, pval *b) {
    if (a == b) return 1;
    long c;
    if (p_num_cmpall(S, a, b, &c)) return c == 0;
    if (a->tag != b->tag) return 0;
    switch (a->tag) {
        case P_NONE:  return 1;
        case P_BOOL:  return a->u.b == b->u.b;
        case P_INT:   return big_cmp(&a->u.i, &b->u.i) == 0;
        case P_FLOAT: return a->u.f == b->u.f;
        case P_STR:
            return a->u.s.len == b->u.s.len &&
                   (a->u.s.len == 0 ||
                    !memcmp(a->u.s.s, b->u.s.s, a->u.s.len));
        case P_LIST: case P_TUPLE:
            if (a->u.l.len != b->u.l.len) return 0;
            for (size_t i = 0; i < a->u.l.len; i++)
                if (!p_eq(S, a->u.l.items[i], b->u.l.items[i])) return 0;
            return 1;
        case P_DICT:
            if (a->u.d.len != b->u.d.len) return 0;
            for (size_t i = 0; i < a->u.d.len; i++) {
                long j = p_dict_find(&b->u.d, a->u.d.k[i]);
                if (j < 0) return 0;
                if (!p_eq(S, a->u.d.v[i], b->u.d.v[j])) return 0;
            }
            return 1;
        case P_FUNC:  case P_FRAME: return a == b;
        case P_RANGE:
            return p_eq(S, a->u.abc.a, b->u.abc.a) &&
                   p_eq(S, a->u.abc.b, b->u.abc.b) &&
                   p_eq(S, a->u.abc.c, b->u.abc.c);
        default:      return a == b;
    }
}

static int p_truth(pst *S, pval *v) {
    switch (v->tag) {
        case P_NONE:  return 0;
        case P_BOOL:  return v->u.b;
        case P_INT:   return v->u.i.n != 0;
        case P_FLOAT: return v->u.f != 0.0;          /* NaN → true */
        case P_STR:   return v->u.s.len > 0;
        case P_LIST:  case P_TUPLE: return v->u.l.len > 0;
        case P_DICT:  return v->u.d.len > 0;
        case P_FUNC:  case P_FRAME: case P_ITER: return 1;
        case P_RANGE: return p_truth(S, p_len_int(S, v));
        default:      return 1;
    }
}

static int p_lt(pst *S, pval *a, pval *b) {
    long c;
    if (p_num_cmpall(S, a, b, &c)) return c < 0;
    if (a->tag != b->tag) {
        p_seterr(S, PE_TYPE, "unorderable types: '%s' and '%s'",
                 p_type_name(a), p_type_name(b));
        return 0;
    }
    switch (a->tag) {
        case P_INT:   return big_cmp(&a->u.i, &b->u.i) < 0;
        case P_FLOAT: return a->u.f < b->u.f;
        case P_STR: {
            size_t n = a->u.s.len < b->u.s.len ? a->u.s.len : b->u.s.len;
            int c2 = n ? memcmp(a->u.s.s, b->u.s.s, n) : 0;
            if (c2) return c2 < 0;
            return a->u.s.len < b->u.s.len;
        }
        case P_LIST: case P_TUPLE: {
            size_t n = a->u.l.len < b->u.l.len ? a->u.l.len : b->u.l.len;
            for (size_t i = 0; i < n; i++) {
                if (p_eq(S, a->u.l.items[i], b->u.l.items[i])) continue;
                return p_lt(S, a->u.l.items[i], b->u.l.items[i]);
            }
            return a->u.l.len < b->u.l.len;
        }
        default:
            p_seterr(S, PE_TYPE, "'%s' objects are not orderable",
                     p_type_name(a));
            return 0;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * len
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_len_int(pst *S, pval *v) {
    switch (v->tag) {
        case P_STR:   return p_int_from_long(S, (long long)v->u.s.len);
        case P_LIST:  case P_TUPLE: return p_int_from_long(S, (long long)v->u.l.len);
        case P_DICT:  return p_int_from_long(S, (long long)v->u.d.len);
        case P_RANGE: return p_range_len(S, v);
        default:
            p_seterr(S, PE_TYPE, "object of type '%s' has no len()",
                     p_type_name(v));
            return NULL;
    }
}

/* d = |step|; span = |stop-start|; len = ceil(span / d), 0 if sign mismatch */
static pval *p_range_len(pst *S, pval *r) {
    pval *start = r->u.abc.a, *stop = r->u.abc.b, *step = r->u.abc.c;
    pval *zero = p_int_from_long(S, 0);
    pval *d = p_int_abs(S, step);
    pval *span = p_int_sub(S, stop, start);
    if (step->u.i.sign) span = p_int_neg(S, span);
    if ((step->u.i.n == 0) ||
        (!step->u.i.sign && big_cmp(&span->u.i, &zero->u.i) < 0) ||
        (step->u.i.sign && big_cmp(&span->u.i, &zero->u.i) > 0))
        return zero;
    pval *one = p_int_from_long(S, 1);
    pval *q = NULL, *rm = NULL;
    pval *num = p_int_add(S, span, p_int_sub(S, d, one));
    if (!p_int_divmod(S, num, d, &q, &rm)) {
        p_seterr(S, PE_ZERO, "range() step must not be zero");
        return NULL;
    }
    return q;
}
"""
