"""Tier-R runtime, part 6: state lifecycle, harness entry point, teardown."""

P6 = r"""
/* ══════════════════════════════════════════════════════════════════
 * state lifecycle
 * ══════════════════════════════════════════════════════════════════ */
static pst *p_state_new(void) {
    pst *S = (pst *)calloc(1, sizeof(pst));
    if (!S) return NULL;
    p_none_setup();
    S->gc_at = 4096;
    S->out.s = NULL;
    S->out.len = 0;
    S->out.cap = 0;
    return S;
}

static void p_state_free(pst *S) {
    if (!S) return;
    for (size_t i = 0; i < S->nobjs; i++) {
        pval *v = S->objs[i];
        if (!v) continue;
        switch (v->tag) {
            case P_INT:  free(v->u.i.d); break;
            case P_STR:  free(v->u.s.s); break;
            case P_LIST: case P_TUPLE: case P_FRAME: free(v->u.l.items); break;
            case P_DICT: free(v->u.d.k); free(v->u.d.v); break;
            default: break;
        }
        free(v);
    }
    free(S->objs);
    free(S->roots);
    free(S->out.s);
    free(S);
}
"""

P7 = r"""
/* ══════════════════════════════════════════════════════════════════
 * harness entry: run; write "#result:<repr>" or "#error:<kind>:<msg>"
 * into out; returns 1 on success. stdout already holds print output.
 * ══════════════════════════════════════════════════════════════════ */
static int purce_tir_run(pst *S, pfn_run entry, int argc, pval **argv,
                         char *out, size_t cap) {
    if (!S || !entry || !out || !cap) return 0;
    S->err = PE_NONE;
    S->errmsg[0] = 0;
    S->out.len = 0;
    if (S->out.s) S->out.s[0] = 0;
    pval *r = entry(S, p_none(S), argc, argv);
    if (p_err(S)) {
        snprintf(out, cap, "#ERROR:%s:%s", p_kind_name(S->err), S->errmsg);
        return 0;
    }
    p_repr_into(S, r);
    p_str_append(S, &S->out, "\n", 1);
    snprintf(out, cap, "#RESULT:%s", S->out.s ? S->out.s : "");
    return 1;
}

/* version marker for the bridge / tests */
static const char *p_tir_version(void) {
    return "purce-tier-r-0.1";
}
"""
