"""Tier-R runtime, part 4: repr / str formatting (Python semantics).

Float formatting follows Python's shortest-round-trip rule: find the fewest
significant digits that strtod back to the original double, then render fixed
or scientific notation per Python thresholds (1e-4 .. 1e16 fixed, outside
scientific).
"""

P4 = r"""
/* ══════════════════════════════════════════════════════════════════
 * float formatting (shortest round-trip)
 * ══════════════════════════════════════════════════════════════════ */
static void p_fmt_digit(pst *S, pstr *out, int d) {
    char c = (char)('0' + (d % 10));
    p_str_append(S, out, &c, 1);
}

/* digits[0..n-1] significant digits, e10 = exponent of first digit.
 * Finds smallest precision whose %.*e printing round-trips. */
static void p_digits_and_exp(double x, int *digits, int *ndig, int *e10) {
    char buf[64];
    int n = 1;
    for (; n <= 17; n++) {
        snprintf(buf, sizeof(buf), "%.*e", n - 1, x);
        if (strtod(buf, NULL) == x) break;
    }
    if (n > 17) { snprintf(buf, sizeof(buf), "%.17e", x); n = 17; }
    char *e = strchr(buf, 'e');
    if (!e) { *ndig = 1; digits[0] = 0; *e10 = 0; return; }
    *e10 = (int)strtol(e + 1, NULL, 10);
    digits[0] = buf[0] - '0';
    int k = 1;
    for (int i = 2; buf[i] && buf[i] != 'e' && k < n; i++, k++)
        digits[k] = buf[i] - '0';
    while (k < n) digits[k++] = 0;
    *ndig = n;
}

static void p_fmt_double(pst *S, pstr *out, double x) {
    if (isnan(x)) { p_str_append(S, out, "nan", 3); return; }
    if (isinf(x)) {
        p_str_append(S, out, x < 0 ? "-inf" : "inf", x < 0 ? 4 : 3);
        return;
    }
    if (x == 0.0) {
        p_str_append(S, out, signbit(x) ? "-0.0" : "0.0", signbit(x) ? 4 : 3);
        return;
    }
    int neg = x < 0;
    if (neg) { p_str_append(S, out, "-", 1); x = -x; }
    int digits[17], nd, e10;
    p_digits_and_exp(x, digits, &nd, &e10);
    if (e10 >= 16 || e10 < -4) {
        p_fmt_digit(S, out, digits[0]);
        if (nd > 1) {
            p_str_append(S, out, ".", 1);
            for (int i = 1; i < nd; i++) p_fmt_digit(S, out, digits[i]);
        }
        char eb[32];
        int el = snprintf(eb, sizeof(eb), "e%+03d", e10);
        p_str_append(S, out, eb, (size_t)el);
        return;
    }
    if (e10 >= 0) {
        p_fmt_digit(S, out, digits[0]);
        for (int i = 1; i <= e10; i++)
            p_fmt_digit(S, out, i < nd ? digits[i] : 0);
        if (e10 + 1 < nd) {
            p_str_append(S, out, ".", 1);
            for (int i = e10 + 1; i < nd; i++) p_fmt_digit(S, out, digits[i]);
        } else {
            p_str_append(S, out, ".0", 2);
        }
        return;
    }
    p_str_append(S, out, "0.", 2);
    for (int i = 0; i < -e10 - 1; i++) p_str_append(S, out, "0", 1);
    for (int i = 0; i < nd; i++) p_fmt_digit(S, out, digits[i]);
}

/* ══════════════════════════════════════════════════════════════════
 * string escaping for repr
 * ══════════════════════════════════════════════════════════════════ */
static void p_quot_into(pst *S, pstr *out, const char *s, size_t len) {
    int has_sq = 0, has_dq = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\'') has_sq = 1;
        else if (s[i] == '"') has_dq = 1;
    }
    char q = (has_sq && !has_dq) ? '"' : '\'';
    p_str_append(S, out, &q, 1);
    for (size_t i = 0; i < len; i++) {
        char c = s[i];
        switch (c) {
            case '\n': p_str_append(S, out, "\\n", 2); break;
            case '\t': p_str_append(S, out, "\\t", 2); break;
            case '\r': p_str_append(S, out, "\\r", 2); break;
            case '\\': p_str_append(S, out, "\\\\", 2); break;
            case '\'': p_str_append(S, out, "\\'", 2); break;
            default:
                if (c == q) {
                    p_str_append(S, out, "\\\"", 2);
                } else {
                    p_str_append(S, out, &c, 1);
                }
        }
    }
    p_str_append(S, out, &q, 1);
}

/* ══════════════════════════════════════════════════════════════════
 * repr — recursive render into a pstr buffer
 * ══════════════════════════════════════════════════════════════════ */
static void p_repr_buf(pst *S, pval *v, pstr *out) {
    switch (v->tag) {
        case P_NONE:  p_str_append(S, out, "None", 4); break;
        case P_BOOL:  p_str_append(S, out, v->u.b ? "True" : "False",
                                   v->u.b ? 4 : 5); break;
        case P_INT:   p_int_into_str(S, v, out); break;
        case P_FLOAT: p_fmt_double(S, out, v->u.f); break;
        case P_STR:   p_quot_into(S, out, v->u.s.s ? v->u.s.s : "",
                                  v->u.s.len); break;
        case P_LIST:  case P_TUPLE: {
            char ob = (v->tag == P_LIST) ? '[' : '(';
            char cb = (v->tag == P_LIST) ? ']' : ')';
            p_str_append(S, out, &ob, 1);
            for (size_t i = 0; i < v->u.l.len; i++) {
                if (i) p_str_append(S, out, ", ", 2);
                p_repr_buf(S, v->u.l.items[i], out);
            }
            if (v->tag == P_TUPLE && v->u.l.len == 1)
                p_str_append(S, out, ",", 1);
            p_str_append(S, out, &cb, 1);
            break;
        }
        case P_DICT: {
            p_str_append(S, out, "{", 1);
            for (size_t i = 0; i < v->u.d.len; i++) {
                if (i) p_str_append(S, out, ", ", 2);
                p_repr_buf(S, v->u.d.k[i], out);
                p_str_append(S, out, ": ", 2);
                p_repr_buf(S, v->u.d.v[i], out);
            }
            p_str_append(S, out, "}", 1);
            break;
        }
        case P_RANGE: {
            p_str_append(S, out, "range(", 6);
            p_repr_buf(S, v->u.abc.a, out);
            p_str_append(S, out, ", ", 2);
            p_repr_buf(S, v->u.abc.b, out);
            pval *one = p_int_from_long(S, 1);
            if (!p_eq(S, v->u.abc.c, one)) {
                p_str_append(S, out, ", ", 2);
                p_repr_buf(S, v->u.abc.c, out);
            }
            p_str_append(S, out, ")", 1);
            break;
        }
        case P_ITER:  p_str_append(S, out, "<iterator>", 10); break;
        case P_FUNC:  p_str_append(S, out, "<function>", 10); break;
        case P_FRAME: p_str_append(S, out, "<frame>", 7); break;
        default:      p_str_append(S, out, "<object>", 8); break;
    }
}

static void p_repr_into(pst *S, pval *v) {
    p_repr_buf(S, v, &S->out);
}

/* str(v): str is identity; int/float/bool → repr text; containers → repr */
static void p_str_of_buf(pst *S, pval *v, pstr *out) {
    if (v->tag == P_STR || v->tag == P_NONE) {
        p_str_append(S, out, v->tag == P_STR ? (v->u.s.s ? v->u.s.s : "") : "None",
                     v->tag == P_STR ? v->u.s.len : 4);
        return;
    }
    p_repr_buf(S, v, out);
}

static pval *p_str_of(pst *S, pval *v) {
    if (v->tag == P_STR || v->tag == P_NONE)
        return v;
    pstr tmp = { NULL, 0, 0 };
    p_str_of_buf(S, v, &tmp);
    pval *r = p_newstr(S, tmp.s ? tmp.s : "", tmp.len);
    free(tmp.s);
    return r;
}
"""
