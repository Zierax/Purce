"""Tier-R runtime, part 7: harness — ctypes-friendly exports.

The differential driver (Python, ctypes) talks to the compiled module
through exactly these non-static functions:
  purce_tir_state_new / purce_tir_state_free
  purce_tir_entry        (returns the generated module entry)
  purce_tir_parse        (text → pval, see grammar below)
  purce_tir_run          (generic run)
  purce_tir_run_list     (parse args, run, write result/error text)

Grammar accepted by purce_tir_parse (one value):
  None | True | False | <int> | <float> | "..." | [v, v, ...]
"""

P7 = r"""
/* ══════════════════════════════════════════════════════════════════
 * harness exports (non-static, ctypes-callable)
 * ══════════════════════════════════════════════════════════════════ */
void *purce_tir_state_new(void) {
    return p_state_new();
}

void purce_tir_state_free(void *S) {
    p_state_free((pst *)S);
}

/* generated module entry pointer (cast to pfn_run by purce_tir_run) */
void *purce_tir_entry(void) {
    extern pval *pf_mod_entry(pst *S, pval *c, pval **argv, int narg);
    return (void *)pf_mod_entry;
}

/* ══════════════════════════════════════════════════════════════════
 * minimal literal parser (recursive descent over one value)
 * ══════════════════════════════════════════════════════════════════ */
static pval *p_parse_val(pst *S, const char **p, const char *end);

static void p_skip_ws(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n')) (*p)++;
}

static pval *p_parse_num(pst *S, const char **p, const char *end) {
    const char *s = *p;
    const char *start = s;                    /* keep sign for ints */
    if (s < end && (*s == '-' || *s == '+')) s++;
    int is_int = 1;
    const char *t = s;
    while (t < end && ((*t >= '0' && *t <= '9') || *t == '.' || *t == 'e' ||
                       *t == 'E' || *t == '+' || *t == '-'))
        t++;
    for (const char *u = s; u < t; u++)
        if (*u == '.' || *u == 'e' || *u == 'E') { is_int = 0; break; }
    size_t len = (size_t)(t - s);
    if (!len) {
        p_seterr(S, PE_VALUE, "bad number literal");
        return NULL;
    }
    *p = t;
    if (is_int) {
        pval *v = NULL;
        if (!p_int_from_str(S, start, (size_t)(t - start), &v)) {
            p_seterr(S, PE_VALUE, "bad integer literal");
            return NULL;
        }
        return v;
    }
    char *buf = (char *)malloc(len + 1);
    if (!buf) { p_seterr(S, PE_RUNTIME, "out of memory"); return NULL; }
    memcpy(buf, s, len);
    buf[len] = 0;
    double d = strtod(buf, NULL);
    free(buf);
    pval *v = p_newfloat(S, d);
    return v;
}

static pval *p_parse_str(pst *S, const char **p, const char *end) {
    (void)end;
    const char *s = *p + 1;               /* skip opening quote */
    pstr acc = { NULL, 0, 0 };
    while (1) {
        if (s >= end) { free(acc.s); p_seterr(S, PE_VALUE, "unterminated string"); return NULL; }
        char c = *s++;
        if (c == '"') break;
        if (c == '\\' && s < end) {
            char e = *s++;
            switch (e) {
                case 'n': p_str_append(S, &acc, "\n", 1); break;
                case 't': p_str_append(S, &acc, "\t", 1); break;
                case 'r': p_str_append(S, &acc, "\r", 1); break;
                case '\\': p_str_append(S, &acc, "\\", 1); break;
                case '"': p_str_append(S, &acc, "\"", 1); break;
                default: p_str_append(S, &acc, &e, 1); break;
            }
        } else {
            p_str_append(S, &acc, &c, 1);
        }
    }
    *p = s;
    pval *v = p_newstr(S, acc.s ? acc.s : "", acc.len);
    free(acc.s);
    return v;
}

static pval *p_parse_list(pst *S, const char **p, const char *end) {
    (*p)++;                                   /* skip '[' */
    pval *lst = p_newobj(S, P_LIST);
    if (!lst) return NULL;
    for (;;) {
        p_skip_ws(p, end);
        if (*p >= end) { p_seterr(S, PE_VALUE, "unterminated list"); return NULL; }
        if (**p == ']') { (*p)++; break; }
        pval *v = p_parse_val(S, p, end);
        if (!v) return NULL;
        if (!p_seq_append(S, lst, v)) return NULL;
        p_skip_ws(p, end);
        if (*p >= end) { p_seterr(S, PE_VALUE, "unterminated list"); return NULL; }
        if (**p == ',') { (*p)++; continue; }
        if (**p == ']') { (*p)++; break; }
        p_seterr(S, PE_VALUE, "expected ',' or ']'");
        return NULL;
    }
    return lst;
}

static pval *p_parse_val(pst *S, const char **p, const char *end) {
    p_skip_ws(p, end);
    if (*p >= end) { p_seterr(S, PE_VALUE, "empty input"); return NULL; }
    char c = **p;
    if (c == 'N' && end - *p >= 4 && !memcmp(*p, "None", 4)) {
        *p += 4;
        return p_none(S);
    }
    if (c == 'T' && end - *p >= 4 && !memcmp(*p, "True", 4)) {
        *p += 4;
        return p_newbool(S, 1);
    }
    if (c == 'F' && end - *p >= 5 && !memcmp(*p, "False", 5)) {
        *p += 5;
        return p_newbool(S, 0);
    }
    if (c == '"') return p_parse_str(S, p, end);
    if (c == '[') return p_parse_list(S, p, end);
    if ((c >= '0' && c <= '9') || c == '-' || c == '+')
        return p_parse_num(S, p, end);
    p_seterr(S, PE_VALUE, "unexpected character '%c' in literal", c);
    return NULL;
}

/* ══════════════════════════════════════════════════════════════════
 * run with argument text: "arg1, arg2, ..." (comma separated values)
 * ══════════════════════════════════════════════════════════════════ */
/* one literal from text (grammar above); returns NULL with err set */
void *purce_tir_parse(void *Sptr, const char *text, size_t n) {
    pst *S = (pst *)Sptr;
    if (!S || !text) return NULL;
    const char *p = text, *end = text + n;
    pval *v = p_parse_val(S, &p, end);
    return v;
}

int purce_tir_run_list(void *Sptr, void *entry, const char *line,
                       size_t n, char *out, size_t cap) {
    pst *S = (pst *)Sptr;
    pfn_run fn = (pfn_run)entry;
    if (!S || !fn || !out || !cap) return 0;
    const char *p = line, *end = line + n;
    pval **args = NULL;
    size_t nargs = 0, capargs = 0;
    if (n) {
        for (;;) {
            pval *v = p_parse_val(S, &p, end);
            if (!v) break;
            if (nargs >= capargs) {
                size_t nc = capargs ? capargs * 2 : 8;
                pval **na = (pval **)realloc(args, nc * sizeof(pval *));
                if (!na) { p_seterr(S, PE_RUNTIME, "out of memory"); break; }
                args = na;
                capargs = nc;
            }
            args[nargs++] = v;
            p_skip_ws(&p, end);
            if (p >= end) break;
            if (*p == ',') { p++; continue; }
            break;
        }
    }
    S->err = PE_NONE;
    S->errmsg[0] = 0;
    S->out.len = 0;
    if (S->out.s) S->out.s[0] = 0;
    pval *r = fn(S, p_none(S), args, (int)nargs);
    free(args);
    if (p_err(S)) {
        snprintf(out, cap, "#ERROR:%s:%s", p_kind_name(S->err), S->errmsg);
        return 0;
    }
    p_repr_into(S, r);
    snprintf(out, cap, "#RESULT:%s", S->out.s ? S->out.s : "");
    return 1;
}

int purce_tir_run(void *Sptr, void *entry, int argc, pval **argv,
                  char *out, size_t cap) {
    pst *S = (pst *)Sptr;
    pfn_run fn = (pfn_run)entry;
    if (!S || !fn || !out || !cap) return 0;
    S->err = PE_NONE;
    S->errmsg[0] = 0;
    S->out.len = 0;
    if (S->out.s) S->out.s[0] = 0;
    pval *r = fn(S, p_none(S), argv, argc);
    if (p_err(S)) {
        snprintf(out, cap, "#ERROR:%s:%s", p_kind_name(S->err), S->errmsg);
        return 0;
    }
    p_repr_into(S, r);
    snprintf(out, cap, "#RESULT:%s", S->out.s ? S->out.s : "");
    return 1;
}

const char *purce_tir_version(void) {
    return "purce-tier-r-0.1";
}
"""
