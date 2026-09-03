"""Tier-R runtime, part 5: ranges, iteration, arithmetic dispatch, function
calls, builtins. All names used here match P1..P4 exactly (single TU).
"""

P5 = r"""
/* ══════════════════════════════════════════════════════════════════
 * small helpers
 * ══════════════════════════════════════════════════════════════════ */
static int p_isnumv(pval *v) {
    return v->tag == P_INT || v->tag == P_BOOL || v->tag == P_FLOAT;
}

static int p_isintv(pval *v) {
    return v->tag == P_INT || v->tag == P_BOOL;
}

static int p_int_signed(pst *S, pval *v) {
    (void)S;
    if (v->tag == P_BOOL) return v->u.b;
    return v->tag == P_INT && v->u.i.n ? (v->u.i.sign ? -1 : 1) : 0;
}

/* promote bool to int pval; returns input otherwise */
static pval *p_promote_int(pst *S, pval *v) {
    if (v->tag == P_BOOL)
        return p_int_from_long(S, v->u.b ? 1 : 0);
    return v;
}

static int p_ne(pst *S, pval *a, pval *b) { return !p_eq(S, a, b); }
static int p_le(pst *S, pval *a, pval *b) { return !p_lt(S, b, a); }
static int p_gt(pst *S, pval *a, pval *b) { return p_lt(S, b, a); }
static int p_ge(pst *S, pval *a, pval *b) { return !p_lt(S, a, b); }
/* ══════════════════════════════════════════════════════════════════
 * range
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_newrange(pst *S, int narg, pval **args) {
    if (narg < 1 || narg > 3) {
        p_seterr(S, PE_TYPE, "range() takes 1-3 arguments");
        return NULL;
    }
    for (int i = 0; i < narg; i++) {
        if (!p_isintv(args[i])) {
            p_seterr(S, PE_TYPE, "range() arguments must be ints");
            return NULL;
        }
    }
    pval *a, *b, *c;
    if (narg == 1) {
        a = p_int_from_long(S, 0);
        b = p_promote_int(S, args[0]);
        c = p_int_from_long(S, 1);
    } else if (narg == 2) {
        a = p_promote_int(S, args[0]);
        b = p_promote_int(S, args[1]);
        c = p_int_from_long(S, 1);
    } else {
        a = p_promote_int(S, args[0]);
        b = p_promote_int(S, args[1]);
        c = p_promote_int(S, args[2]);
    }
    if (c->u.i.n == 0) {
        p_seterr(S, PE_VALUE, "range() arg 3 must not be zero");
        return NULL;
    }
    pval *v = p_newobj(S, P_RANGE);
    if (!v) return NULL;
    v->u.abc.a = a;
    v->u.abc.b = b;
    v->u.abc.c = c;
    return v;
}

/* range[i] → a + i*c, with negative i folded */
static pval *p_range_at(pst *S, pval *r, pval *i) {
    pval *t = p_int_mul(S, i, r->u.abc.c);
    return p_int_add(S, r->u.abc.a, t);
}

static pval *p_range_get(pst *S, pval *r, pval *key) {
    if (!p_isintv(key)) {
        p_seterr(S, PE_TYPE, "range indices must be integers");
        return NULL;
    }
    pval *i = p_promote_int(S, key);
    pval *len = p_range_len(S, r);
    if (!len) return NULL;
    if (i->u.i.sign) i = p_int_add(S, i, len);
    long c;
    if (p_num_cmpall(S, i, p_int_from_long(S, 0), &c) && c < 0) {
        p_seterr(S, PE_INDEX, "range object index out of range");
        return NULL;
    }
    if (p_num_cmpall(S, i, len, &c) && c >= 0) {
        p_seterr(S, PE_INDEX, "range object index out of range");
        return NULL;
    }
    return p_range_at(S, r, i);
}

static int p_range_contains(pst *S, pval *r, pval *x) {
    if (!p_isintv(x)) return 0;
    pval *i = p_promote_int(S, x);
    pval *span = p_int_sub(S, i, r->u.abc.a);
    pval *q = NULL, *rem = NULL;
    if (!p_int_divmod(S, span, r->u.abc.c, &q, &rem)) return 0;
    if (rem->u.i.n != 0) return 0;
    pval *len = p_range_len(S, r);
    if (!len) return 0;
    long c;
    return !(p_num_cmpall(S, q, p_int_from_long(S, 0), &c) && c < 0) &&
           !(p_num_cmpall(S, q, len, &c) && c >= 0);
}

/* ══════════════════════════════════════════════════════════════════
 * iteration protocol
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_iter(pst *S, pval *seq) {
    if (seq->tag == P_ITER) return seq;
    switch (seq->tag) {
        case P_LIST: case P_TUPLE: case P_STR: case P_DICT: case P_RANGE:
            break;
        default:
            p_seterr(S, PE_TYPE, "'%s' object is not iterable",
                     p_type_name(seq));
            return NULL;
    }
    pval *it = p_newobj(S, P_ITER);
    if (!it) return NULL;
    it->u.it.kind = (int)seq->tag;
    it->u.it.seq = seq;
    it->u.it.idx = 0;
    it->u.it.cur = NULL;
    return it;
}

/* next item; returns 1 and stores in *out, or 0 on exhaustion.
 * On error, S->err is set (check p_err after loop). */
static int p_iter_next(pst *S, pval *it, pval **out) {
    pval *seq = it->u.it.seq;
    *out = NULL;
    switch (it->u.it.kind) {
        case P_LIST: case P_TUPLE:
            if (it->u.it.idx >= seq->u.l.len) return 0;
            *out = seq->u.l.items[it->u.it.idx];
            break;
        case P_STR:
            if (it->u.it.idx >= seq->u.s.len) return 0;
            *out = p_newstr(S, seq->u.s.s + it->u.it.idx, 1);
            break;
        case P_DICT:
            if (it->u.it.idx >= seq->u.d.len) return 0;
            *out = seq->u.d.k[it->u.it.idx];
            break;
        case P_RANGE: {
            pval *len = p_range_len(S, seq);
            if (!len) return 0;
            pval *i = p_int_from_long(S, (long long)it->u.it.idx);
            long c;
            if (p_num_cmpall(S, i, len, &c) && c >= 0) return 0;
            *out = p_range_at(S, seq, i);
            break;
        }
        default:
            p_seterr(S, PE_RUNTIME, "bad iterator state");
            return 0;
    }
    it->u.it.idx++;
    it->u.it.cur = *out;
    return 1;
}

/* ══════════════════════════════════════════════════════════════════
 * arithmetic dispatch
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_arith_err(pst *S, const char *op, pval *a, pval *b) {
    p_seterr(S, PE_TYPE, "unsupported operand type(s) for %s: '%s' and '%s'",
             op, p_type_name(a), p_type_name(b));
    return NULL;
}

static pval *p_add(pst *S, pval *a, pval *b) {
    if (p_isnumv(a) && p_isnumv(b)) {
        if (p_isintv(a) && p_isintv(b))
            return p_int_add(S, p_promote_int(S, a), p_promote_int(S, b));
        return p_newfloat(S, p_num_double(S, a) + p_num_double(S, b));
    }
    if (a->tag == P_STR && b->tag == P_STR)
        return p_str_concat(S, a, b);
    if (a->tag == b->tag && (a->tag == P_LIST || a->tag == P_TUPLE))
        return p_seq_concat(S, a, b);
    return p_arith_err(S, "+", a, b);
}

static pval *p_sub(pst *S, pval *a, pval *b) {
    if (!p_isnumv(a) || !p_isnumv(b)) return p_arith_err(S, "-", a, b);
    if (p_isintv(a) && p_isintv(b))
        return p_int_sub(S, p_promote_int(S, a), p_promote_int(S, b));
    return p_newfloat(S, p_num_double(S, a) - p_num_double(S, b));
}

static pval *p_repeat(pst *S, pval *seq, pval *cnt) {
    if (!p_isintv(cnt)) return p_arith_err(S, "*", seq, cnt);
    pval *k = p_promote_int(S, cnt);
    if (k->u.i.sign) {
        p_seterr(S, PE_VALUE, "cannot multiply sequence by negative int");
        return NULL;
    }
    long long n = 0;
    if (k->u.i.n == 1) n = (long long)k->u.i.d[0];
    if (seq->tag == P_STR) {
        pval *out = p_newobj(S, P_STR);
        if (!out) return NULL;
        out->u.s.s = NULL; out->u.s.len = 0; out->u.s.cap = 0;
        for (long long i = 0; i < n; i++)
            if (!p_str_append(S, &out->u.s, seq->u.s.s ? seq->u.s.s : "",
                              seq->u.s.len))
                return NULL;
        return out;
    }
    if (seq->tag == P_LIST || seq->tag == P_TUPLE) {
        pval *out = p_newobj(S, seq->tag);
        if (!out) return NULL;
        for (long long i = 0; i < n; i++)
            for (size_t j = 0; j < seq->u.l.len; j++)
                if (!p_seq_append(S, out, seq->u.l.items[j])) return NULL;
        return out;
    }
    return p_arith_err(S, "*", seq, cnt);
}

static pval *p_mul(pst *S, pval *a, pval *b) {
    if (p_isnumv(a) && p_isnumv(b)) {
        if (p_isintv(a) && p_isintv(b))
            return p_int_mul(S, p_promote_int(S, a), p_promote_int(S, b));
        return p_newfloat(S, p_num_double(S, a) * p_num_double(S, b));
    }
    if (a->tag == P_STR || a->tag == P_LIST || a->tag == P_TUPLE)
        return p_repeat(S, a, b);
    if (b->tag == P_STR || b->tag == P_LIST || b->tag == P_TUPLE)
        return p_repeat(S, b, a);
    return p_arith_err(S, "*", a, b);
}

static pval *p_truediv(pst *S, pval *a, pval *b) {
    if (!p_isnumv(a) || !p_isnumv(b)) return p_arith_err(S, "/", a, b);
    double y = p_num_double(S, b);
    if (y == 0.0) {
        p_seterr(S, PE_ZERO, "division by zero");
        return NULL;
    }
    return p_newfloat(S, p_num_double(S, a) / y);
}

static pval *p_floordiv(pst *S, pval *a, pval *b) {
    if (!p_isnumv(a) || !p_isnumv(b)) return p_arith_err(S, "//", a, b);
    if (p_isintv(a) && p_isintv(b)) {
        pval *q = NULL, *r = NULL;
        if (!p_int_divmod(S, p_promote_int(S, a), p_promote_int(S, b),
                          &q, &r))
            return NULL;
        return q;
    }
    double x = p_num_double(S, a), y = p_num_double(S, b);
    if (y == 0.0) {
        p_seterr(S, PE_ZERO, "float floor division by zero");
        return NULL;
    }
    double r = fmod(x, y);
    double q = (x - r) / y;
    if (r != 0.0 && (r > 0) != (y > 0)) q -= 1.0;
    return p_newfloat(S, q);
}

static pval *p_mod(pst *S, pval *a, pval *b) {
    if (!p_isnumv(a) || !p_isnumv(b)) return p_arith_err(S, "%", a, b);
    if (p_isintv(a) && p_isintv(b)) {
        pval *q = NULL, *r = NULL;
        if (!p_int_divmod(S, p_promote_int(S, a), p_promote_int(S, b),
                          &q, &r))
            return NULL;
        return r;
    }
    double x = p_num_double(S, a), y = p_num_double(S, b);
    if (y == 0.0) {
        p_seterr(S, PE_ZERO, "float modulo by zero");
        return NULL;
    }
    double r = fmod(x, y);
    if (r != 0.0 && (r > 0) != (y > 0)) r += y;
    return p_newfloat(S, r);
}

static pval *p_pow(pst *S, pval *a, pval *b) {
    if (!p_isnumv(a) || !p_isnumv(b)) return p_arith_err(S, "**", a, b);
    if (p_isintv(a) && p_isintv(b)) {
        pval *base = p_promote_int(S, a);
        pval *exp = p_promote_int(S, b);
        if (!exp->u.i.sign)
            return p_int_pow_nonneg(S, base, exp->u.i.n ? exp->u.i.d[0] : 0);
    }
    double y = p_num_double(S, b);
    if (y == 0.0) return p_newfloat(S, 1.0);
    return p_newfloat(S, pow(p_num_double(S, a), y));
}

static pval *p_neg(pst *S, pval *a) {
    if (a->tag == P_BOOL)
        return p_int_from_long(S, a->u.b ? -1 : 0);
    if (a->tag == P_INT) {
        pval *r = p_int_neg(S, a);
        return r;
    }
    if (a->tag == P_FLOAT) return p_newfloat(S, -a->u.f);
    p_seterr(S, PE_TYPE, "bad operand type for unary -: '%s'",
             p_type_name(a));
    return NULL;
}

static pval *p_abs(pst *S, pval *a) {
    if (a->tag == P_BOOL) return p_int_from_long(S, a->u.b ? 1 : 0);
    if (a->tag == P_INT) return p_int_abs(S, a);
    if (a->tag == P_FLOAT) return p_newfloat(S, fabs(a->u.f));
    p_seterr(S, PE_TYPE, "bad operand type for abs(): '%s'", p_type_name(a));
    return NULL;
}

/* ══════════════════════════════════════════════════════════════════
 * getitem / setitem / membership
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_getitem(pst *S, pval *obj, pval *key) {
    switch (obj->tag) {
        case P_STR: {
            if (!p_isintv(key)) {
                p_seterr(S, PE_TYPE, "string indices must be integers");
                return NULL;
            }
            long li = p_seq_ndex(S, obj, key);
            if (li < 0) {
                if (!p_err(S))
                    p_seterr(S, PE_INDEX, "string index out of range");
                return NULL;
            }
            return p_str_chat(S, obj, (size_t)li);
        }
        case P_LIST: case P_TUPLE: return p_seq_get(S, obj, key);
        case P_DICT:  return p_dict_get(S, &obj->u.d, key);
        case P_RANGE: return p_range_get(S, obj, key);
        default:
            p_seterr(S, PE_TYPE, "'%s' object is not subscriptable",
                     p_type_name(obj));
            return NULL;
    }
}

static int p_setitem(pst *S, pval *obj, pval *key, pval *val) {
    switch (obj->tag) {
        case P_LIST:  return p_seq_set(S, obj, key, val);
        case P_DICT:  return p_dict_set(S, &obj->u.d, key, val);
        default:
            p_seterr(S, PE_TYPE, "'%s' object does not support item "
                     "assignment", p_type_name(obj));
            return 0;
    }
}

static int p_contains(pst *S, pval *obj, pval *x) {
    switch (obj->tag) {
        case P_STR:
            if (x->tag != P_STR) {
                p_seterr(S, PE_TYPE, "string membership requires a string");
                return 0;
            }
            return p_str_isub(&obj->u.s, &x->u.s) >= 0;
        case P_LIST: case P_TUPLE: return p_seq_contain(S, obj, x);
        case P_DICT:  return p_dict_contains(&obj->u.d, x);
        case P_RANGE: return p_range_contains(S, obj, x);
        default:
            p_seterr(S, PE_TYPE, "argument of type '%s' is not iterable",
                     p_type_name(obj));
            return 0;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * function calls / closures
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_newframe(pst *S, size_t nslots) {
    pval *fr = p_newobj(S, P_FRAME);
    if (!fr) return NULL;
    if (nslots) {
        fr->u.l.items = (pval **)calloc(nslots, sizeof(pval *));
        if (!fr->u.l.items) {
            p_seterr(S, PE_RUNTIME, "out of memory");
            return NULL;
        }
    }
    fr->u.l.len = nslots;
    fr->u.l.cap = nslots;
    return fr;
}

static pval *p_frame_get(pst *S, pval *fr, size_t i) {
    (void)S;
    if (!fr || i >= fr->u.l.len || !fr->u.l.items[i])
        return p_none(NULL);
    return fr->u.l.items[i];
}

static int p_frame_set(pst *S, pval *fr, size_t i, pval *v) {
    if (!fr || i >= fr->u.l.len) {
        p_seterr(S, PE_RUNTIME, "frame slot out of range");
        return 0;
    }
    fr->u.l.items[i] = v;
    return 1;
}

static pval *p_mkfunc(pst *S, pfn_run run, pval *env, int arity,
                      const char *name) {
    pval *f = p_newobj(S, P_FUNC);
    if (!f) return NULL;
    f->u.g.run = run;
    f->u.g.env = env;
    f->u.g.arity = arity;
    f->u.g.name = name;
    return f;
}

static pval *p_call(pst *S, pval *f, int narg, pval **argv) {
    if (!f || f->tag != P_FUNC) {
        p_seterr(S, PE_TYPE, "'%s' object is not callable",
                 f ? p_type_name(f) : "None");
        return NULL;
    }
    if (f->u.g.arity >= 0 && narg != f->u.g.arity) {
        p_seterr(S, PE_TYPE, "%s() takes exactly %d arguments (%d given)",
                 f->u.g.name ? f->u.g.name : "function", f->u.g.arity, narg);
        return NULL;
    }
    if (S->depth >= 256) {
        p_seterr(S, PE_RECUR, "maximum recursion depth exceeded");
        return NULL;
    }
    S->depth++;
    pval *r = f->u.g.run(S, f, argv, narg);
    S->depth--;
    return r;
}

/* ══════════════════════════════════════════════════════════════════
 * builtins
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_badarity(pst *S, const char *name, int n, int want) {
    p_seterr(S, PE_TYPE, "%s() takes %d argument(s), %d given",
             name, want, n);
    return NULL;
}

static pval *p_bi_len(pst *S, int n, pval **a) {
    if (n != 1) return p_badarity(S, "len", n, 1);
    return p_len_int(S, a[0]);
}

static pval *p_bi_print(pst *S, int n, pval **a) {
    pstr acc = { NULL, 0, 0 };
    for (int i = 0; i < n; i++) {
        if (i) p_str_append(S, &acc, " ", 1);
        p_str_of_buf(S, a[i], &acc);
    }
    p_str_append(S, &acc, "\n", 1);
    if (acc.s) {
        fwrite(acc.s, 1, acc.len, stdout);
        fflush(stdout);
    }
    free(acc.s);
    return p_none(S);
}

static pval *p_bi_str(pst *S, int n, pval **a) {
    if (n != 1) return p_badarity(S, "str", n, 1);
    return p_str_of(S, a[0]);
}

static pval *p_bi_int(pst *S, int n, pval **a) {
    if (n != 1) return p_badarity(S, "int", n, 1);
    pval *v = a[0];
    if (p_isintv(v)) return p_promote_int(S, v);
    if (v->tag == P_FLOAT) {
        if (isnan(v->u.f) || isinf(v->u.f)) {
            p_seterr(S, PE_VALUE, "cannot convert float to integer");
            return NULL;
        }
        double t = trunc(v->u.f);
        if (t < -9.0e15 || t > 9.0e15) {
            p_seterr(S, PE_VALUE, "float too large for integer conversion");
            return NULL;
        }
        return p_int_from_long(S, (long long)t);
    }
    if (v->tag == P_STR) return p_str_to_int(S, v);
    p_seterr(S, PE_TYPE, "int() argument must be a number or string");
    return NULL;
}

static pval *p_bi_float(pst *S, int n, pval **a) {
    if (n != 1) return p_badarity(S, "float", n, 1);
    pval *v = a[0];
    if (v->tag == P_FLOAT) return v;
    if (p_isintv(v)) return p_newfloat(S, p_num_double(S, v));
    if (v->tag == P_STR) {
        const char *s = v->u.s.s ? v->u.s.s : "";
        char *end = NULL;
        double d = strtod(s, &end);
        if (end != s && *end == '\0') return p_newfloat(S, d);
        p_seterr(S, PE_VALUE, "could not convert string to float");
        return NULL;
    }
    p_seterr(S, PE_TYPE, "float() argument must be a number or string");
    return NULL;
}

static pval *p_bi_bool(pst *S, int n, pval **a) {
    if (n != 1) return p_badarity(S, "bool", n, 1);
    return p_newbool(S, p_truth(S, a[0]));
}

/* iterable → array; *cap grows. Caller owns buffer. */
static int p_iter_collect(pst *S, pval *obj, pval ***buf, size_t *n) {
    pval *it = p_iter(S, obj);
    if (!it) return 0;
    size_t cap = 0;
    *buf = NULL;
    *n = 0;
    pval *x;
    while (p_iter_next(S, it, &x)) {
        if (*n >= cap) {
            size_t nc = cap ? cap * 2 : 8;
            pval **nb = (pval **)p_alloc(*buf, nc * sizeof(pval *));
            if (!nb) { p_seterr(S, PE_RUNTIME, "out of memory"); return 0; }
            *buf = nb;
            cap = nc;
        }
        (*buf)[(*n)++] = x;
    }
    if (p_err(S)) { free(*buf); *buf = NULL; return 0; }
    return 1;
}

static pval *p_bi_sum(pst *S, int n, pval **a) {
    if (n < 1 || n > 2) return p_badarity(S, "sum", n, 1);
    pval **items = NULL;
    size_t ni = 0;
    if (!p_iter_collect(S, a[0], &items, &ni)) return NULL;
    pval *acc = n == 2 ? a[1] : p_int_from_long(S, 0);
    for (size_t i = 0; i < ni; i++) {
        acc = p_add(S, acc, items[i]);
        if (!acc) { free(items); return NULL; }
    }
    free(items);
    return acc;
}

static pval *p_bi_minmax(pst *S, int n, pval **a, int want_max) {
    pval **items = NULL;
    size_t ni = 0;
    if (n == 1) {
        if (!p_iter_collect(S, a[0], &items, &ni)) return NULL;
    } else {
        items = a;
        ni = (size_t)n;
    }
    if (ni == 0) {
        if (n == 1) free(items);
        p_seterr(S, PE_VALUE, "%s() arg is an empty sequence",
                 want_max ? "max" : "min");
        return NULL;
    }
    pval *best = items[0];
    for (size_t i = 1; i < ni; i++) {
        int better = want_max ? p_lt(S, best, items[i])
                              : p_lt(S, items[i], best);
        if (better) best = items[i];
    }
    if (n == 1) free(items);
    return best;
}

static pval *p_bi_range(pst *S, int n, pval **a) {
    return p_newrange(S, n, a);
}

static pval *p_bi_list(pst *S, int n, pval **a) {
    pval *out = p_newobj(S, P_LIST);
    if (!out) return NULL;
    if (n == 0) return out;
    if (n != 1) return p_badarity(S, "list", n, 1);
    pval *it = p_iter(S, a[0]);
    if (!it) return NULL;
    pval *x;
    while (p_iter_next(S, it, &x)) {
        if (!p_seq_append(S, out, x)) return NULL;
    }
    return p_err(S) ? NULL : out;
}

static pval *p_builtin_call(pst *S, const char *name, int narg, pval **args) {
    if (!strcmp(name, "len"))    return p_bi_len(S, narg, args);
    if (!strcmp(name, "print"))  return p_bi_print(S, narg, args);
    if (!strcmp(name, "str"))    return p_bi_str(S, narg, args);
    if (!strcmp(name, "int"))    return p_bi_int(S, narg, args);
    if (!strcmp(name, "float"))  return p_bi_float(S, narg, args);
    if (!strcmp(name, "bool"))   return p_bi_bool(S, narg, args);
    if (!strcmp(name, "sum"))    return p_bi_sum(S, narg, args);
    if (!strcmp(name, "range"))  return p_bi_range(S, narg, args);
    if (!strcmp(name, "list"))   return p_bi_list(S, narg, args);
    if (!strcmp(name, "min"))    return p_bi_minmax(S, narg, args, 0);
    if (!strcmp(name, "max"))    return p_bi_minmax(S, narg, args, 1);
    if (!strcmp(name, "abs")) {
        if (narg != 1) return p_badarity(S, "abs", narg, 1);
        return p_abs(S, args[0]);
    }
    p_seterr(S, PE_UNSUP, "unsupported builtin function '%s'", name);
    return NULL;
}
"""
