"""Tier-R runtime, part 1: C header, types, error reporting, memory/GC.

The runtime is a single C99 source assembled from parts P1..P5 (see
runtime_c.py). GC contract: collection happens ONLY at ``p_maybe_gc`` sites
emitted by the emitter (statement/loop boundaries); no runtime operation
triggers collection, so unrooted C temporaries inside one operation are
never collected.
"""

P1 = r"""
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
/* ══════════════════════════════════════════════════════════════════
 * PURCE TIER-R RUNTIME — C99, self-contained, single-threaded.
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

typedef struct pst pst;
typedef struct pval pval;

typedef enum {
    P_NONE = 0, P_BOOL, P_INT, P_FLOAT, P_STR, P_LIST, P_TUPLE,
    P_DICT, P_FUNC, P_FRAME, P_RANGE, P_ITER
} ptag;

/* arbitrary-precision integer: sign {0,1} x little-endian limbs base 2^30 */
typedef struct {
    uint8_t sign;
    uint32_t n;
    uint32_t *d;
} pbig;

typedef pval *(*pfn_run)(pst *S, pval *c, pval **argv, int narg);

typedef struct {
    char *s;
    size_t len;
    size_t cap;
} pstr;

typedef struct {
    pval **items;
    size_t len;
    size_t cap;
} pseq;

typedef struct {
    pval **k;
    pval **v;
    size_t len;
    size_t cap;
} pdict;

typedef enum {
    PE_NONE = 0, PE_ZERO, PE_INDEX, PE_KEY, PE_VALUE, PE_TYPE,
    PE_ASSERT, PE_UNSUP, PE_RECUR, PE_RUNTIME, PE_STOP
} perrk;

struct pval {
    ptag tag;
    int marked;
    union {
        int b;
        pbig i;
        double f;
        pstr s;
        pseq l;      /* P_LIST, P_TUPLE, P_FRAME (slots) */
        pdict d;
        struct { pval *a; pval *b; pval *c; } abc;                    /* P_RANGE */
        struct { int kind; pval *seq; size_t idx; pval *cur; } it;    /* P_ITER */
        struct { pfn_run run; pval *env; int arity; const char *name; } g;
    } u;
};

struct pst {
    pval **objs;
    size_t nobjs, capobjs;
    pval **roots;
    size_t nroots, caproots;
    size_t nalloc;
    size_t gc_at;
    int depth;
    perrk err;
    char errmsg[512];
    pstr out;
    const char *text;
    int gc_enable;
};

static pval *p_none_v;

/* ── forward declarations ─────────────────────────────────────────── */
static pval *p_newobj(pst *S, ptag tag);
static int p_eq(pst *S, pval *a, pval *b);
static int p_truth(pst *S, pval *v);
static int p_lt(pst *S, pval *a, pval *b);
static void p_repr_into(pst *S, pval *v);
static pval *p_call(pst *S, pval *f, int narg, pval **argv);
static void p_collect(pst *S);
static pval *p_newint(pst *S);
static pval *p_int_from_u64(pst *S, uint64_t x);
static pval *p_int_from_long(pst *S, long long x);
static pval *p_int_neg(pst *S, pval *v);
static pval *p_int_abs(pst *S, pval *v);
static pval *p_int_add(pst *S, pval *a, pval *b);
static pval *p_int_sub(pst *S, pval *a, pval *b);
static pval *p_int_mul(pst *S, pval *a, pval *b);
static double p_int_to_double(pst *S, pval *v);
static pval *p_newstr(pst *S, const char *s, size_t len);
static pval *p_newstrn(pst *S, const char *s);
static pval *p_newfloat(pst *S, double x);
static pval *p_newbool(pst *S, int x);
static int p_int_divmod(pst *S, pval *a, pval *b, pval **qo, pval **ro);
static void p_seterr(pst *S, perrk k, const char *fmt, ...);
static pval *p_none(pst *S);

static void p_none_setup(void) {
    static int done = 0;
    if (!done) {
        p_none_v = (pval *)calloc(1, sizeof(pval));
        if (p_none_v) p_none_v->tag = P_NONE;
        done = 1;
    }
}

static pval *p_none(pst *S) { (void)S; return p_none_v; }

/* ══════════════════════════════════════════════════════════════════
 * errors
 * ══════════════════════════════════════════════════════════════════ */
static void p_seterr(pst *S, perrk k, const char *fmt, ...) {
    if (S->err != PE_NONE && k != PE_STOP) return;
    S->err = k;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(S->errmsg, sizeof(S->errmsg), fmt, ap);
    va_end(ap);
}

static int p_err(pst *S) { return S->err != PE_NONE; }

static const char *p_type_name(pval *v) {
    switch (v->tag) {
        case P_NONE:  return "NoneType";
        case P_BOOL:  return "bool";
        case P_INT:   return "int";
        case P_FLOAT: return "float";
        case P_STR:   return "str";
        case P_LIST:  return "list";
        case P_TUPLE: return "tuple";
        case P_DICT:  return "dict";
        case P_FUNC:  return "function";
        case P_RANGE: return "range";
        case P_ITER:  return "iterator";
        default:      return "object";
    }
}

static const char *p_kind_name(perrk k) {
    switch (k) {
        case PE_ZERO:   return "zero";
        case PE_INDEX:  return "index";
        case PE_KEY:    return "key";
        case PE_VALUE:  return "value";
        case PE_TYPE:   return "type";
        case PE_ASSERT: return "assert";
        case PE_UNSUP:  return "unsup";
        case PE_RECUR:  return "recur";
        default:        return "runtime";
    }
}

/* ══════════════════════════════════════════════════════════════════
 * memory / GC
 * ══════════════════════════════════════════════════════════════════ */
static void *p_alloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p && n) return NULL;
    return p;
}

static pval *p_newobj(pst *S, ptag tag) {
    if (S->nobjs >= S->capobjs) {
        size_t ncap = S->capobjs ? S->capobjs * 2 : 4096;
        pval **np = (pval **)p_alloc(S->objs, ncap * sizeof(pval *));
        if (!np) { p_seterr(S, PE_RUNTIME, "out of memory"); return NULL; }
        S->objs = np;
        S->capobjs = ncap;
    }
    pval *v = (pval *)calloc(1, sizeof(pval));
    if (!v) { p_seterr(S, PE_RUNTIME, "out of memory"); return NULL; }
    v->tag = tag;
    S->objs[S->nobjs++] = v;
    return v;
}

static void p_root(pst *S, pval *v) {
    if (S->nroots >= S->caproots) {
        size_t ncap = S->caproots ? S->caproots * 2 : 256;
        pval **np = (pval **)p_alloc(S->roots, ncap * sizeof(pval *));
        if (!np) { p_seterr(S, PE_RUNTIME, "out of memory"); return; }
        S->roots = np;
        S->caproots = ncap;
    }
    S->roots[S->nroots++] = v;
}

static void p_mark(pst *S, pval *v) {
    if (!v || v->marked) return;
    static pval **stk;
    static size_t cap;
    size_t top = 0;
    if (!stk) {
        cap = 4096;
        stk = (pval **)malloc(cap * sizeof(pval *));
        if (!stk) { p_seterr(S, PE_RUNTIME, "out of memory"); return; }
    }
    stk[top++] = v;
    while (top) {
        pval *w = stk[--top];
        if (w->marked) continue;
        w->marked = 1;
        switch (w->tag) {
            case P_LIST: case P_TUPLE: case P_FRAME:
                for (size_t i = 0; i < w->u.l.len; i++) {
                    pval *c = w->u.l.items[i];
                    if (c && !c->marked) {
                        if (top >= cap) {
                            size_t ncap = cap * 2;
                            pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                            if (!ns) break;
                            stk = ns;
                            cap = ncap;
                        }
                        stk[top++] = c;
                    }
                }
                break;
            case P_DICT:
                for (size_t i = 0; i < w->u.d.len; i++) {
                    pval *a = w->u.d.k[i], *b = w->u.d.v[i];
                    if (a && !a->marked) {
                        if (top >= cap) {
                            size_t ncap = cap * 2;
                            pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                            if (!ns) break;
                            stk = ns;
                            cap = ncap;
                        }
                        stk[top++] = a;
                    }
                    if (b && !b->marked) {
                        if (top >= cap) {
                            size_t ncap = cap * 2;
                            pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                            if (!ns) break;
                            stk = ns;
                            cap = ncap;
                        }
                        stk[top++] = b;
                    }
                }
                break;
            case P_FUNC:
                if (w->u.g.env && !w->u.g.env->marked) {
                    if (top >= cap) {
                        size_t ncap = cap * 2;
                        pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                        if (!ns) break;
                        stk = ns;
                        cap = ncap;
                    }
                    stk[top++] = w->u.g.env;
                }
                break;
            case P_RANGE: {
                pval *kids[3] = { w->u.abc.a, w->u.abc.b, w->u.abc.c };
                for (int i = 0; i < 3; i++) {
                    pval *c = kids[i];
                    if (c && !c->marked) {
                        if (top >= cap) {
                            size_t ncap = cap * 2;
                            pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                            if (!ns) break;
                            stk = ns;
                            cap = ncap;
                        }
                        stk[top++] = c;
                    }
                }
                break;
            }
            case P_ITER: {
                pval *kids[2] = { w->u.it.seq, w->u.it.cur };
                for (int i = 0; i < 2; i++) {
                    pval *c = kids[i];
                    if (c && !c->marked) {
                        if (top >= cap) {
                            size_t ncap = cap * 2;
                            pval **ns = (pval **)realloc(stk, ncap * sizeof(pval *));
                            if (!ns) break;
                            stk = ns;
                            cap = ncap;
                        }
                        stk[top++] = c;
                    }
                }
                break;
            }
            default:
                break;
        }
    }
}

static void p_collect(pst *S) {
    if (S->nalloc < S->gc_at) return;
    S->nalloc = 0;
    S->gc_at = S->gc_at > (size_t)1 << 30 ? (size_t)1 << 30 : S->gc_at * 2;
    for (size_t i = 0; i < S->nroots; i++) p_mark(S, S->roots[i]);
    for (size_t i = 0; i < S->nobjs; i++) {
        pval *v = S->objs[i];
        if (v->marked) { v->marked = 0; continue; }
        switch (v->tag) {
            case P_INT:  free(v->u.i.d); break;
            case P_STR:  free(v->u.s.s); break;
            case P_LIST: case P_TUPLE: case P_FRAME: free(v->u.l.items); break;
            case P_DICT: free(v->u.d.k); free(v->u.d.v); break;
            default: break;
        }
        free(v);
        S->objs[i] = S->objs[S->nobjs - 1];
        S->nobjs--;
        i--;
    }
}

static void p_maybe_gc(pst *S) {
    S->nalloc++;
    if (S->gc_enable && S->nalloc >= S->gc_at) p_collect(S);
}
"""
