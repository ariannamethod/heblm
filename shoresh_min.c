/*
 * shoresh_min.c — stripped root resonance engine
 *
 * Minimal architecture:
 *   Hebrew text -> roots
 *   roots -> metaweight field
 *   field -> next root
 *   root -> surface word
 *   word feeds field back
 *
 * No transformer.
 * No Klaus.
 * No calendar.
 * No weights.
 *
 * Compile:
 *   cc shoresh_min.c -O2 -lm -o shoresh_min
 *
 * Run:
 *   ./shoresh_min shoresh.txt
 *   ./shoresh_min shoresh.txt "חכמה ובינה"
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define HEB         22
#define MAX_ROOTS   512
#define MAX_WORDS   8192
#define MAX_CWORDS  200000
#define MAX_RWORDS  64
#define MAX_TEXT    (8*1024*1024)

#define MAX_BI      65536
#define MAX_TRI     65536
#define MAX_HEBB    65536
#define MAX_PROPH   32
#define HEBB_WIN    6

#define TOP_K       12
#define MAX_GEN     20
#define TEMP        0.42f

static const char *LET[HEB] = {
    "א","ב","ג","ד","ה","ו","ז","ח","ט","י",
    "כ","ל","מ","נ","ס","ע","פ","צ","ק","ר","ש","ת"
};

static const int ORD[HEB] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
};

typedef struct { int c[3]; } Root;

typedef struct {
    char utf8[64];
    int letters[32];
    int nlet;
    int root_id;
    int count;
} HWord;

typedef struct {
    Root roots[MAX_ROOTS];
    int nr;
    int rhash[HEB*HEB*HEB];

    HWord words[MAX_WORDS];
    int nw;

    int rwords[MAX_ROOTS][MAX_RWORDS];
    int rwc[MAX_ROOTS];

    int *corpus_wids;
    int *corpus_rids;
    int nc;
} RootEng;

typedef struct { int a,b; float p; } Bi;
typedef struct { int a,b,c; float p; } Tri;
typedef struct { int a,b; float s; } Heb;
typedef struct { int t; float s; int age; } Pro;

typedef struct {
    float uni[MAX_ROOTS];
    Bi bi[MAX_BI]; int nbi;
    Tri tri[MAX_TRI]; int ntri;
    Heb hebb[MAX_HEBB]; int nhebb;
    Pro proph[MAX_PROPH]; int nproph;
} RMeta;

static float cbigram[HEB][HEB];

/* ───────────────────────────────────────────────────────────── */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX;
}

static float cf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static int rhkey(int a, int b, int c) {
    return a*HEB*HEB + b*HEB + c;
}

static int isheb(const unsigned char *p) {
    return p[0] == 0xD7 && p[1] >= 0x90 && p[1] <= 0xAA;
}

static int u8let(const unsigned char *p, int *adv) {
    *adv = 1;
    if (p[0] != 0xD7) return -1;
    *adv = 2;
    switch (p[1]) {
        case 0x90: return 0;  case 0x91: return 1;  case 0x92: return 2;
        case 0x93: return 3;  case 0x94: return 4;  case 0x95: return 5;
        case 0x96: return 6;  case 0x97: return 7;  case 0x98: return 8;
        case 0x99: return 9;  case 0x9A: case 0x9B: return 10;
        case 0x9C: return 11; case 0x9D: case 0x9E: return 12;
        case 0x9F: case 0xA0: return 13; case 0xA1: return 14;
        case 0xA2: return 15; case 0xA3: case 0xA4: return 16;
        case 0xA5: case 0xA6: return 17; case 0xA7: return 18;
        case 0xA8: return 19; case 0xA9: return 20; case 0xAA: return 21;
        default: return -1;
    }
}

static char *readf(const char *p, long *sz) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (*sz <= 0 || *sz > MAX_TEXT) {
        fclose(f);
        return NULL;
    }
    char *b = malloc(*sz + 1);
    if (!b) {
        fclose(f);
        return NULL;
    }
    fread(b, 1, *sz, f);
    fclose(f);
    b[*sz] = 0;
    return b;
}

/* ───────────────────────────────────────────────────────────── */

static void re_init(RootEng *r) {
    memset(r, 0, sizeof(*r));
    for (int i = 0; i < HEB*HEB*HEB; i++) r->rhash[i] = -1;
}

static int re_add_root(RootEng *r, int c1, int c2, int c3) {
    if (c1 < 0 || c2 < 0 || c3 < 0) return -1;
    int k = rhkey(c1, c2, c3);
    if (r->rhash[k] >= 0) return r->rhash[k];
    if (r->nr >= MAX_ROOTS) return -1;
    int id = r->nr++;
    r->roots[id].c[0] = c1;
    r->roots[id].c[1] = c2;
    r->roots[id].c[2] = c3;
    r->rhash[k] = id;
    return id;
}

/* seed lexicon */
typedef struct { int a,b,c; } RD;
static const RD KR[] = {
    {7,10,12},{1,9,14},{9,3,15},{0,12,19},{20,16,8},
    {18,19,0},{1,19,0},{13,17,7},{20,12,20},{7,20,1},
    {11,12,3},{15,13,4},{17,3,18},{0,4,1},{7,1,19},
    {20,11,7},{18,17,19},{12,9,12},{13,11,7},{19,5,7},
    {-1,-1,-1}
};

static void re_load_lex(RootEng *r) {
    for (int i = 0; KR[i].a >= 0; i++) {
        re_add_root(r, KR[i].a, KR[i].b, KR[i].c);
    }
}

static int w2let(const char *u, int *o, int mx) {
    const unsigned char *p = (const unsigned char *)u;
    int n = 0;
    while (*p && n < mx) {
        int adv = 1;
        int l = u8let(p, &adv);
        if (l >= 0) o[n++] = l;
        p += adv;
    }
    return n;
}

static int re_find_root(RootEng *r, const int *let, int n) {
    if (n < 2) return -1;

    int best = -1, bs = 999, bspan = 999;
    for (int ri = 0; ri < r->nr; ri++) {
        int t[3] = {
            r->roots[ri].c[0],
            r->roots[ri].c[1],
            r->roots[ri].c[2]
        };
        int pos[3] = {-1,-1,-1};
        int ti = 0;
        for (int i = 0; i < n && ti < 3; i++) {
            if (let[i] == t[ti]) {
                pos[ti] = i;
                ti++;
            }
        }
        if (ti == 3) {
            int s = pos[0];
            int sp = pos[2] - pos[0];
            if (s < bs || (s == bs && sp < bspan)) {
                bs = s;
                bspan = sp;
                best = ri;
            }
        }
    }
    if (best >= 0) return best;

    /* crude fallback */
    int start = 0;
    if (n >= 3) {
        int f = let[0];
        if (f == 4 || f == 1 || f == 10 || f == 11 || f == 12 || f == 20 || f == 5 || f == 13 || f == 9 || f == 21 || f == 0)
            start = 1;
    }

    int s[32], sn = 0;
    for (int i = start; i < n && sn < 32; i++) s[sn++] = let[i];
    if (sn < 2) return -1;
    if (sn == 2) return re_add_root(r, s[0], s[1], s[1]);
    return re_add_root(r, s[0], s[1], s[2]);
}

static int re_add_word(RootEng *r, const char *u) {
    if (r->nw >= MAX_WORDS) return -1;

    for (int i = 0; i < r->nw; i++) {
        if (strcmp(r->words[i].utf8, u) == 0) {
            r->words[i].count++;
            return i;
        }
    }

    int wi = r->nw++;
    HWord *w = &r->words[wi];
    strncpy(w->utf8, u, 63);
    w->utf8[63] = 0;
    w->nlet = w2let(u, w->letters, 32);
    w->count = 1;
    w->root_id = re_find_root(r, w->letters, w->nlet);

    if (w->root_id >= 0 && r->rwc[w->root_id] < MAX_RWORDS) {
        r->rwords[w->root_id][r->rwc[w->root_id]++] = wi;
    }

    return wi;
}

static int re_extract(RootEng *r, const char *text, int *wids, int *rids, int mx) {
    const unsigned char *p = (const unsigned char *)text;
    int n = 0;
    char wb[128];
    int wp = 0;

    while (*p && n < mx) {
        if (isheb(p)) {
            if (wp < 126) {
                wb[wp++] = p[0];
                wb[wp++] = p[1];
            }
            p += 2;
        } else {
            if (wp > 0) {
                wb[wp] = 0;
                int wi = re_add_word(r, wb);
                if (wi >= 0) {
                    if (wids) wids[n] = wi;
                    if (rids) rids[n] = r->words[wi].root_id;
                    n++;
                }
                wp = 0;
            }
            p++;
        }
    }

    if (wp > 0 && n < mx) {
        wb[wp] = 0;
        int wi = re_add_word(r, wb);
        if (wi >= 0) {
            if (wids) wids[n] = wi;
            if (rids) rids[n] = r->words[wi].root_id;
            n++;
        }
    }

    return n;
}

/* ───────────────────────────────────────────────────────────── */

static void build_cbigrams(const RootEng *re) {
    memset(cbigram, 0, sizeof(cbigram));
    float rt[HEB] = {0};

    for (int w = 0; w < re->nw; w++) {
        const HWord *hw = &re->words[w];
        for (int i = 0; i < hw->nlet - 1; i++) {
            int a = hw->letters[i], b = hw->letters[i + 1];
            if (a >= 0 && a < HEB && b >= 0 && b < HEB) {
                cbigram[a][b] += (float)hw->count;
                rt[a] += (float)hw->count;
            }
        }
    }

    for (int a = 0; a < HEB; a++) {
        if (rt[a] > 0) {
            for (int b = 0; b < HEB; b++) cbigram[a][b] /= rt[a];
        }
    }
}

/* ───────────────────────────────────────────────────────────── */

static void rm_build(RMeta *m, const int *seq, int n, int nr) {
    memset(m, 0, sizeof(*m));

    for (int i = 0; i < n; i++) {
        if (seq[i] >= 0 && seq[i] < nr) m->uni[seq[i]] += 1.0f;
    }

    float t = 0;
    for (int i = 0; i < nr; i++) t += m->uni[i];
    if (t > 0) for (int i = 0; i < nr; i++) m->uni[i] /= t;

    for (int i = 0; i < n - 1; i++) {
        int a = seq[i], b = seq[i + 1];
        if (a < 0 || b < 0) continue;

        int f = 0;
        for (int j = 0; j < m->nbi; j++) {
            if (m->bi[j].a == a && m->bi[j].b == b) {
                m->bi[j].p += 1.0f;
                f = 1;
                break;
            }
        }
        if (!f && m->nbi < MAX_BI) {
            m->bi[m->nbi].a = a;
            m->bi[m->nbi].b = b;
            m->bi[m->nbi].p = 1.0f;
            m->nbi++;
        }
    }

    for (int i = 0; i < m->nbi; i++) {
        float tt = 0;
        for (int j = 0; j < m->nbi; j++) if (m->bi[j].a == m->bi[i].a) tt += m->bi[j].p;
        if (tt > 0) m->bi[i].p /= tt;
    }

    for (int i = 0; i < n - 2; i++) {
        int a = seq[i], b = seq[i + 1], c = seq[i + 2];
        if (a < 0 || b < 0 || c < 0) continue;

        int f = 0;
        for (int j = 0; j < m->ntri; j++) {
            if (m->tri[j].a == a && m->tri[j].b == b && m->tri[j].c == c) {
                m->tri[j].p += 1.0f;
                f = 1;
                break;
            }
        }
        if (!f && m->ntri < MAX_TRI) {
            m->tri[m->ntri].a = a;
            m->tri[m->ntri].b = b;
            m->tri[m->ntri].c = c;
            m->tri[m->ntri].p = 1.0f;
            m->ntri++;
        }
    }

    for (int i = 0; i < m->ntri; i++) {
        float tt = 0;
        for (int j = 0; j < m->ntri; j++) {
            if (m->tri[j].a == m->tri[i].a && m->tri[j].b == m->tri[i].b) tt += m->tri[j].p;
        }
        if (tt > 0) m->tri[i].p /= tt;
    }

    int hn = n < 30000 ? n : 30000;
    for (int i = 0; i < hn; i++) {
        if (seq[i] < 0) continue;
        int lo = i - HEBB_WIN; if (lo < 0) lo = 0;
        int hi = i + HEBB_WIN; if (hi >= hn) hi = hn - 1;
        for (int j = lo; j <= hi; j++) {
            if (i == j || seq[j] < 0) continue;
            int a = seq[i] < seq[j] ? seq[i] : seq[j];
            int b = seq[i] < seq[j] ? seq[j] : seq[i];
            float d = 1.0f / (1.0f + fabsf((float)(i - j)));

            int f = 0;
            for (int k = 0; k < m->nhebb; k++) {
                if (m->hebb[k].a == a && m->hebb[k].b == b) {
                    m->hebb[k].s += d;
                    f = 1;
                    break;
                }
            }
            if (!f && m->nhebb < MAX_HEBB) {
                m->hebb[m->nhebb].a = a;
                m->hebb[m->nhebb].b = b;
                m->hebb[m->nhebb].s = d;
                m->nhebb++;
            }
        }
    }

    float mx = 0;
    for (int i = 0; i < m->nhebb; i++) if (m->hebb[i].s > mx) mx = m->hebb[i].s;
    if (mx > 0) for (int i = 0; i < m->nhebb; i++) m->hebb[i].s /= mx;
}

static float rm_bi(RMeta *m, int a, int b) {
    for (int i = 0; i < m->nbi; i++) if (m->bi[i].a == a && m->bi[i].b == b) return m->bi[i].p;
    return 1e-10f;
}

static float rm_tri(RMeta *m, int a, int b, int c) {
    for (int i = 0; i < m->ntri; i++) if (m->tri[i].a == a && m->tri[i].b == b && m->tri[i].c == c) return m->tri[i].p;
    return 1e-10f;
}

static void rm_q_hebb(RMeta *m, const int *ctx, int n, float *out, int nr) {
    for (int i = 0; i < nr; i++) out[i] = 0.0f;
    int s = n - 8; if (s < 0) s = 0;

    for (int j = s; j < n; j++) {
        if (ctx[j] < 0) continue;
        for (int k = 0; k < m->nhebb; k++) {
            if (m->hebb[k].a == ctx[j] && m->hebb[k].b < nr) out[m->hebb[k].b] += m->hebb[k].s;
            else if (m->hebb[k].b == ctx[j] && m->hebb[k].a < nr) out[m->hebb[k].a] += m->hebb[k].s;
        }
    }

    float mx = 0.0f;
    for (int i = 0; i < nr; i++) if (out[i] > mx) mx = out[i];
    if (mx > 1e-12f) for (int i = 0; i < nr; i++) out[i] /= mx;
}

static void rm_q_proph(RMeta *m, const int *ctx, int n, float *out, int nr) {
    for (int i = 0; i < nr; i++) out[i] = 0.0f;
    int seen[MAX_ROOTS] = {0};

    for (int i = 0; i < n; i++) if (ctx[i] >= 0 && ctx[i] < nr) seen[ctx[i]] = 1;

    int s = n - 6; if (s < 0) s = 0;
    for (int j = s; j < n; j++) {
        if (ctx[j] < 0) continue;
        float dc = 1.0f / (1.0f + (float)(n - 1 - j));
        for (int k = 0; k < m->nbi; k++) {
            if (m->bi[k].a == ctx[j] && m->bi[k].b < nr && !seen[m->bi[k].b]) {
                out[m->bi[k].b] += m->bi[k].p * dc;
            }
        }
    }

    if (n >= 2) {
        int p0 = ctx[n - 2], p1 = ctx[n - 1];
        for (int k = 0; k < m->ntri; k++) {
            if (m->tri[k].a == p0 && m->tri[k].b == p1 && m->tri[k].c < nr && !seen[m->tri[k].c]) {
                out[m->tri[k].c] += m->tri[k].p * 1.5f;
            }
        }
    }

    for (int i = 0; i < m->nproph; i++) {
        int t = m->proph[i].t;
        if (t >= 0 && t < nr && !seen[t]) {
            out[t] += m->proph[i].s * logf(1.0f + (float)m->proph[i].age);
        }
    }

    float mx = 0.0f;
    for (int i = 0; i < nr; i++) if (out[i] > mx) mx = out[i];
    if (mx > 1e-12f) for (int i = 0; i < nr; i++) out[i] /= mx;
}

static void proph_add(RMeta *m, int t, float s) {
    if (t < 0) return;
    for (int i = 0; i < m->nproph; i++) {
        if (m->proph[i].t == t) {
            if (s > m->proph[i].s) m->proph[i].s = s;
            m->proph[i].age = 0;
            return;
        }
    }
    if (m->nproph < MAX_PROPH) {
        m->proph[m->nproph].t = t;
        m->proph[m->nproph].s = s;
        m->proph[m->nproph].age = 0;
        m->nproph++;
    }
}

static void proph_update(RMeta *m, int tok) {
    int k = 0;
    for (int i = 0; i < m->nproph; i++) {
        if (m->proph[i].t == tok) continue;
        m->proph[i].age++;
        m->proph[i].s *= 0.99f;
        if (m->proph[i].age < 30 && m->proph[i].s > 0.01f) m->proph[k++] = m->proph[i];
    }
    m->nproph = k;
}

/* ───────────────────────────────────────────────────────────── */

static int realize(const RootEng *re, int rid, const int *prev_let, int pn, float ord_m) {
    if (rid < 0 || rid >= re->nr || re->rwc[rid] == 0) return -1;

    float best = -1e30f;
    int bw = re->rwords[rid][0];

    for (int ci = 0; ci < re->rwc[rid]; ci++) {
        int wi = re->rwords[rid][ci];
        const HWord *hw = &re->words[wi];
        float sc = 0.0f;

        sc += 0.45f * logf(1.0f + (float)hw->count);

        if (pn > 0 && hw->nlet > 0) {
            int lc = prev_let[pn - 1];
            int fc = hw->letters[0];
            if (lc >= 0 && lc < HEB && fc >= 0 && fc < HEB) sc += 0.70f * cbigram[lc][fc];
        }

        float in = 0.0f;
        for (int i = 0; i < hw->nlet - 1; i++) {
            int a = hw->letters[i], b = hw->letters[i + 1];
            if (a >= 0 && a < HEB && b >= 0 && b < HEB) in += cbigram[a][b];
        }
        if (hw->nlet > 1) sc += 0.28f * in / (float)(hw->nlet - 1);

        float own_ord = 0.0f;
        for (int i = 0; i < hw->nlet; i++) own_ord += (float)(ORD[hw->letters[i]]);
        if (hw->nlet > 0) own_ord /= (float)hw->nlet;
        sc += 0.35f * (1.0f / (1.0f + fabsf(own_ord - ord_m / 3.0f)));

        if (hw->nlet >= 3 && hw->nlet <= 6) sc += 0.18f;

        if (sc > best) {
            best = sc;
            bw = wi;
        }
    }

    return bw;
}

/* ───────────────────────────────────────────────────────────── */

static int select_root(RMeta *m, RootEng *re, const int *ctx, int n, float ord_m) {
    int NR = re->nr;
    float *score = calloc(NR, sizeof(float));
    float *heb = calloc(NR, sizeof(float));
    float *pro = calloc(NR, sizeof(float));

    rm_q_hebb(m, ctx, n, heb, NR);
    rm_q_proph(m, ctx, n, pro, NR);

    int prev1 = (n >= 1) ? ctx[n - 1] : -1;
    int prev2 = (n >= 2) ? ctx[n - 2] : -1;

    for (int i = 0; i < NR; i++) {
        float bi  = (prev1 >= 0) ? rm_bi(m, prev1, i) : 1e-10f;
        float tri = (prev2 >= 0 && prev1 >= 0) ? rm_tri(m, prev2, prev1, i) : 1e-10f;

        float ord_h = 1.0f / (1.0f + 4.0f * fabsf(
            (float)(ORD[re->roots[i].c[0]] + ORD[re->roots[i].c[1]] + ORD[re->roots[i].c[2]]) / 66.0f
            - ord_m / 66.0f));

        score[i] =
            0.15f * m->uni[i] +
            2.60f * bi +
            3.80f * tri +
            0.90f * heb[i] +
            0.75f * pro[i] +
            0.65f * ord_h;

        if (re->rwc[i] == 0) score[i] = -1e9f;
    }

    int rs = n - 6; if (rs < 0) rs = 0;
    for (int j = rs; j < n; j++) {
        if (ctx[j] >= 0 && ctx[j] < NR) score[ctx[j]] *= 0.35f;
    }

    int ti[TOP_K];
    float tv[TOP_K];
    for (int k = 0; k < TOP_K; k++) { ti[k] = 0; tv[k] = -1e30f; }

    for (int i = 0; i < NR; i++) {
        if (score[i] > tv[TOP_K - 1]) {
            tv[TOP_K - 1] = score[i];
            ti[TOP_K - 1] = i;
            for (int k = TOP_K - 2; k >= 0; k--) {
                if (tv[k + 1] > tv[k]) {
                    float tmp = tv[k]; tv[k] = tv[k + 1]; tv[k + 1] = tmp;
                    int tt = ti[k]; ti[k] = ti[k + 1]; ti[k + 1] = tt;
                }
            }
        }
    }

    float mx = tv[0];
    float pr[TOP_K];
    float sum = 0.0f;
    for (int k = 0; k < TOP_K; k++) {
        pr[k] = expf((tv[k] - mx) / TEMP);
        sum += pr[k];
    }

    float r = randf() * sum;
    float cum = 0.0f;
    int chosen = ti[0];
    for (int k = 0; k < TOP_K; k++) {
        cum += pr[k];
        if (cum >= r) {
            chosen = ti[k];
            break;
        }
    }

    free(score);
    free(heb);
    free(pro);
    return chosen;
}

/* ───────────────────────────────────────────────────────────── */

static void generate_chain(RootEng *re, RMeta *m, const char *prompt, int n_words) {
    int pwi[64], pri[64];
    int pn = re_extract(re, prompt, pwi, pri, 64);

    int ctx[256];
    int cn = 0;

    for (int i = 0; i < pn && cn < 256; i++) if (pri[i] >= 0) ctx[cn++] = pri[i];

    float ord_sum = 0.0f;
    int ord_cnt = 0;
    for (int i = 0; i < cn; i++) if (ctx[i] >= 0 && ctx[i] < re->nr) {
        ord_sum += (float)(ORD[re->roots[ctx[i]].c[0]] + ORD[re->roots[ctx[i]].c[1]] + ORD[re->roots[ctx[i]].c[2]]);
        ord_cnt++;
    }
    float ord_m = ord_cnt > 0 ? ord_sum / (float)ord_cnt : 33.0f;

    int prev_let[32] = {0};
    int prev_nl = 0;
    if (pn > 0 && pwi[pn - 1] >= 0) {
        memcpy(prev_let, re->words[pwi[pn - 1]].letters, re->words[pwi[pn - 1]].nlet * sizeof(int));
        prev_nl = re->words[pwi[pn - 1]].nlet;
    }

    printf("Prompt: %s\n", prompt);
    printf("Roots:  ");
    for (int i = 0; i < pn; i++) if (pri[i] >= 0) {
        Root *r = &re->roots[pri[i]];
        printf("%s.%s.%s ", LET[r->c[0]], LET[r->c[1]], LET[r->c[2]]);
    }
    printf("\nGen:    ");

    for (int step = 0; step < n_words && cn < 250; step++) {
        int rid = select_root(m, re, ctx, cn, ord_m);
        if (rid < 0) break;

        int wi = realize(re, rid, prev_let, prev_nl, ord_m);
        if (wi >= 0) {
            printf("%s ", re->words[wi].utf8);
            ctx[cn++] = rid;

            ord_m = 0.82f * ord_m + 0.18f *
                (float)(ORD[re->roots[rid].c[0]] + ORD[re->roots[rid].c[1]] + ORD[re->roots[rid].c[2]]);

            proph_update(m, rid);

            float bp = 0.0f;
            int bpred = -1;
            for (int k = 0; k < m->nbi; k++) {
                if (m->bi[k].a == rid && m->bi[k].p > bp) {
                    bp = m->bi[k].p;
                    bpred = m->bi[k].b;
                }
            }
            if (bpred >= 0) proph_add(m, bpred, 0.30f + 0.50f * bp);

            memcpy(prev_let, re->words[wi].letters, re->words[wi].nlet * sizeof(int));
            prev_nl = re->words[wi].nlet;
        } else {
            Root *r = &re->roots[rid];
            printf("[%s%s%s] ", LET[r->c[0]], LET[r->c[1]], LET[r->c[2]]);
            ctx[cn++] = rid;
        }
        fflush(stdout);
    }

    printf("\n");
}

/* ───────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));

    if (argc < 2) {
        printf("Usage: %s shoresh.txt [prompt]\n", argv[0]);
        return 1;
    }

    const char *cpath = argv[1];
    const char *prompt = (argc >= 3) ? argv[2] : "בראשית";

    long csz;
    char *text = readf(cpath, &csz);
    if (!text) {
        fprintf(stderr, "Cannot read %s\n", cpath);
        return 1;
    }

    RootEng *re = calloc(1, sizeof(RootEng));
    re_init(re);
    re_load_lex(re);

    int *cwi = malloc(MAX_CWORDS * sizeof(int));
    int *cri = malloc(MAX_CWORDS * sizeof(int));
    int cn = re_extract(re, text, cwi, cri, MAX_CWORDS);

    re->corpus_wids = cwi;
    re->corpus_rids = cri;
    re->nc = cn;

    build_cbigrams(re);

    RMeta *m = calloc(1, sizeof(RMeta));
    int *vr = malloc(cn * sizeof(int));
    int vn = 0;
    for (int i = 0; i < cn; i++) if (cri[i] >= 0) vr[vn++] = cri[i];
    rm_build(m, vr, vn, re->nr);

    printf("shoresh-min\n");
    printf("corpus: %ld bytes\n", csz);
    printf("words:  %d total / %d unique\n", cn, re->nw);
    printf("roots:  %d\n", re->nr);
    printf("γ:      %d bi / %d tri / %d hebb\n\n", m->nbi, m->ntri, m->nhebb);

    generate_chain(re, m, prompt, MAX_GEN);

    free(vr);
    free(text);
    free(cwi);
    free(cri);
    free(re);
    free(m);
    return 0;
}
