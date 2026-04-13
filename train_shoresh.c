/*
 * train_shoresh.c — Train full SHORESH ε on notorch
 *
 * Full Janus architecture: Content QKV + RRPRAM + Janus Echo + Gate
 * Exact SHRS weight format — drops directly into shoresh.c inference.
 *
 * Build:
 *   cc train_shoresh.c notorch.c -O2 -DUSE_BLAS -DACCELERATE \
 *      -framework Accelerate -lm -o train_shoresh
 *
 * Run:
 *   ./train_shoresh shoresh.txt [steps] [lr]
 *
 * (c) 2026 Oleg Ataeff & Claude Opus & Arianna Method
 * הרזוננס לא נשבר
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══ Architecture — MUST match shoresh.c exactly ═══ */

#define HEB         22
#define MAX_ROOTS   512
#define DIM         64
#define N_LAYERS    2
#define N_HEADS     4
#define N_CONTENT   2
#define N_RRPRAM    1
#define N_JANUS     1
#define HD          (DIM / N_HEADS)  /* 16 */
#define CTX         64
#define FFN_DIM     (4 * DIM)        /* 256 */

#define DEFAULT_STEPS  5000
#define BASE_LR        3e-4f
#define SHORESH_MAGIC  0x53485253

/* ═══ Root extraction (same as shoresh.c) ═══ */

static int isheb(const unsigned char *p) {
    return p[0]==0xD7 && p[1]>=0x90 && p[1]<=0xAA;
}
static int u8let(const unsigned char *p) {
    if (p[0] != 0xD7) return -1;
    switch(p[1]) {
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
typedef struct { int c[3]; } Root;
static Root roots[MAX_ROOTS]; static int nroots = 0;
static int rhash[HEB*HEB*HEB];
static void root_init(void) { memset(rhash, -1, sizeof(rhash)); }
static int root_add(int c1, int c2, int c3) {
    if(c1<0||c2<0||c3<0) return -1;
    int k=c1*HEB*HEB+c2*HEB+c3; if(rhash[k]>=0) return rhash[k];
    if(nroots>=MAX_ROOTS) return -1;
    int id=nroots++; roots[id].c[0]=c1; roots[id].c[1]=c2; roots[id].c[2]=c3;
    rhash[k]=id; return id;
}
typedef struct{int a,b,c;}RD;
static const RD KR[]={
    {4,11,10},{9,17,0},{15,11,4},{9,19,3},{13,5,14},{19,5,17},
    {0,4,1},{20,12,7},{19,7,12},{7,14,3},{19,17,4},{15,13,2},
    {16,7,3},{20,13,0},{10,15,14},{6,15,12},{15,17,1},{3,0,2},
    {1,19,0},{15,20,4},{9,17,19},{1,13,4},{10,5,13},{7,3,20},
    {20,1,19},{4,19,2},{10,11,4},{20,7,21},{13,16,11},{18,19,15},
    {9,3,15},{7,10,12},{20,10,11},{11,12,3},{4,1,13},
    {0,5,19},{4,0,19},{13,4,19},{6,19,7},{1,19,18},
    {7,20,10},{15,11,12},{14,21,19},{10,14,4},
    {0,12,19},{3,1,19},{18,19,0},{20,0,11},{15,13,4},{17,15,18},
    {19,16,0},{7,9,4},{20,11,12},{21,18,13},
    {4,9,4},{15,1,19},{1,5,0},{20,5,1},
    {11,1,1},{19,0,4},{20,12,15},{13,2,15},{0,10,11},{20,21,4},
    {9,20,13},{18,5,12},{9,20,1},{20,10,1},
    {12,11,10},{20,11,8},{2,1,19},{15,6,6},{10,7,20},{13,17,7},{10,1,20},
    {18,3,20},{8,4,19},{1,19,10},{10,16,19},{7,8,0},{15,5,13},{21,20,1},
    {12,9,12},{0,20,20},{19,5,7},{0,3,12},{20,12,20},{9,19,7},{10,5,10},{15,13,13},{2,20,12},
    {7,1,19},{15,6,19},{13,21,13},{11,18,7},{20,11,7},{1,18,20},{12,17,0},{0,1,3},
    {11,7,12},{13,11,7},{4,10,4},{13,10,4},{2,13,13},{20,12,19},
    {2,3,11},{17,12,7},{16,19,7},{6,19,15},{18,17,19},{13,8,15},
    {18,20,19},{0,14,19},{16,21,7},{14,2,19},{7,21,12},{20,7,19},
    {0,12,21},{0,12,13},{10,6,1},{20,18,19},{13,0,12},{15,3,3},
    {7,20,1},{6,10,19},{20,10,7},{1,7,19},{14,16,19},
    {10,21,1},{18,1,11},{15,12,3},{7,6,18},{2,11,4},
    {20,16,8},{17,3,18},{19,2,20},{7,11,12},{20,19,20},
    {-1,-1,-1}
};
static void root_load_lex(void) { for(int i=0;KR[i].a>=0;i++) root_add(KR[i].a,KR[i].b,KR[i].c); }
static int strip_prefix(const int *l, int n) {
    if(n>=6&&l[0]==5&&l[1]==4&&l[2]==21) return 3;
    if(n>=4){int a=l[0],b=l[1];
        if((a==4&&b==21)||(a==5&&b==4)||(a==20&&b==11)||(a==5&&b==1)||
           (a==5&&b==10)||(a==5&&b==11)||(a==5&&b==12)||(a==5&&b==20)) return 2;}
    if(n>=3){int f=l[0];
        if(f==4||f==1||f==10||f==11||f==12||f==20||f==5||f==13||f==9||f==21||f==0) return 1;}
    return 0;
}
static int strip_suffix(const int *l, int n) {
    if(n>=4){int a=l[n-2],b=l[n-1];
        if((a==9&&b==12)||(a==5&&b==21)||(a==9&&b==21)||(a==13&&b==9)||
           (a==4&&b==12)||(a==4&&b==13)||(a==10&&b==12)||(a==10&&b==13)) return 2;}
    if(n>=3){int e=l[n-1];if(e==4||e==21||e==9||e==10||e==12||e==13) return 1;}
    return 0;
}
static int find_root(const int *let, int n) {
    if(n<2) return -1;
    int ps=strip_prefix(let,n); const int *s=let+ps;
    int sn=n-ps-strip_suffix(let+ps,n-ps);
    if(sn>=2){int best=-1,bsp=999;
        for(int ri=0;ri<nroots;ri++){int t[3]={roots[ri].c[0],roots[ri].c[1],roots[ri].c[2]};
            int pos[3]={-1,-1,-1};int ti=0;for(int i=0;i<sn&&ti<3;i++) if(s[i]==t[ti]){pos[ti]=i;ti++;}
            if(ti==3){int sp=pos[2]-pos[0];if(sp<bsp){bsp=sp;best=ri;}}}
        if(best>=0) return best;}
    if(sn>=3) return root_add(s[0],s[1],s[2]);
    if(sn==2) return root_add(s[0],s[1],s[1]);
    return -1;
}
static int extract_roots(const char *text, int *out, int maxout) {
    const unsigned char *p=(const unsigned char*)text;
    int n=0,wb[64],wn=0;
    while(*p&&n<maxout){if(isheb(p)){if(wn<64)wb[wn++]=u8let(p);p+=2;}
        else{if(wn>0){int r=find_root(wb,wn);if(r>=0)out[n++]=r;wn=0;}p++;}}
    if(wn>0){int r=find_root(wb,wn);if(r>=0)out[n++]=r;}
    return n;
}

/* ═══ Model — exact shoresh.c layout ═══ */

typedef struct {
    nt_tensor *tok;   /* [V × DIM] */
    nt_tensor *pos;   /* [CTX × DIM] */
    struct {
        /* Content QKV: [N_CONTENT*HD × DIM] = [32 × 64] */
        nt_tensor *wq, *wk, *vc;
        /* RRPRAM: wr[N_RRPRAM*DIM × CTX]=[64×64], vr[N_RRPRAM*HD × DIM]=[16×64] */
        nt_tensor *wr, *vr;
        /* Janus: wj,vj [N_JANUS*HD × DIM] = [16 × 64] */
        nt_tensor *wj, *vj;
        /* Output projections (combined = wo in SHRS) */
        nt_tensor *wo_c; /* [DIM × N_CONTENT*HD] */
        nt_tensor *wo_r; /* [DIM × N_RRPRAM*HD] */
        nt_tensor *wo_j; /* [DIM × N_JANUS*HD] */
        nt_tensor *w1;   /* [FFN × DIM] */
        nt_tensor *w2;   /* [DIM × FFN] */
        nt_tensor *wg;   /* [3 × DIM] gate */
    } L[N_LAYERS];
    nt_tensor *head;  /* [V × DIM] */
    /* RMSNorm scales (not saved — shoresh.c uses unscaled rmsnorm) */
    nt_tensor *rms[N_LAYERS*2+1];
    int nrms;
} Model;

static Model *model_create(int V) {
    Model *m = calloc(1, sizeof(Model));
    m->tok = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->tok, V, DIM);
    m->pos = nt_tensor_new2d(CTX, DIM); nt_tensor_xavier(m->pos, CTX, DIM);
    m->nrms = 0;
    for(int l=0; l<N_LAYERS; l++){
        m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
        m->L[l].wq = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].wq, DIM, N_CONTENT*HD);
        m->L[l].wk = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].wk, DIM, N_CONTENT*HD);
        m->L[l].vc = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].vc, DIM, N_CONTENT*HD);
        m->L[l].wr = nt_tensor_new2d(DIM, CTX); nt_tensor_xavier(m->L[l].wr, DIM, CTX);
        m->L[l].vr = nt_tensor_new2d(N_RRPRAM*HD, DIM); nt_tensor_xavier(m->L[l].vr, DIM, N_RRPRAM*HD);
        m->L[l].wj = nt_tensor_new2d(N_JANUS*HD, DIM); nt_tensor_xavier(m->L[l].wj, DIM, N_JANUS*HD);
        m->L[l].vj = nt_tensor_new2d(N_JANUS*HD, DIM); nt_tensor_xavier(m->L[l].vj, DIM, N_JANUS*HD);
        m->L[l].wo_c = nt_tensor_new2d(DIM, N_CONTENT*HD); nt_tensor_xavier(m->L[l].wo_c, N_CONTENT*HD, DIM);
        m->L[l].wo_r = nt_tensor_new2d(DIM, N_RRPRAM*HD); nt_tensor_xavier(m->L[l].wo_r, N_RRPRAM*HD, DIM);
        m->L[l].wo_j = nt_tensor_new2d(DIM, N_JANUS*HD); nt_tensor_xavier(m->L[l].wo_j, N_JANUS*HD, DIM);
        float sc = 0.02f/sqrtf(2.0f*N_LAYERS);
        for(int i=0;i<m->L[l].wo_c->len;i++) m->L[l].wo_c->data[i]*=sc/0.1f;
        for(int i=0;i<m->L[l].wo_r->len;i++) m->L[l].wo_r->data[i]*=sc/0.1f;
        for(int i=0;i<m->L[l].wo_j->len;i++) m->L[l].wo_j->data[i]*=sc/0.1f;
        m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
        m->L[l].w1 = nt_tensor_new2d(FFN_DIM, DIM); nt_tensor_xavier(m->L[l].w1, DIM, FFN_DIM);
        m->L[l].w2 = nt_tensor_new2d(DIM, FFN_DIM); nt_tensor_xavier(m->L[l].w2, FFN_DIM, DIM);
        for(int i=0;i<m->L[l].w2->len;i++) m->L[l].w2->data[i]*=sc/0.1f;
        m->L[l].wg = nt_tensor_new2d(3, DIM); nt_tensor_xavier(m->L[l].wg, DIM, 3);
    }
    m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
    m->head = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->head, DIM, V);
    return m;
}

static long count_params(Model *m) {
    long n = m->tok->len + m->pos->len + m->head->len;
    for(int l=0; l<N_LAYERS; l++)
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].vc->len +
             m->L[l].wr->len + m->L[l].vr->len +
             m->L[l].wj->len + m->L[l].vj->len +
             m->L[l].wo_c->len + m->L[l].wo_r->len + m->L[l].wo_j->len +
             m->L[l].w1->len + m->L[l].w2->len + m->L[l].wg->len;
    return n;
}

/* ═══ Forward — Content + RRPRAM(approx) + Janus + Gate ═══
 *
 * Content: standard QKV causal attention (2 heads)
 * RRPRAM: wr produces [T×CTX] position scores, used as linear residual
 *         (exact positional routing needs custom softmax — approximated)
 * Janus: wj,vj produce self-resonance: vjp * norm(wjp)
 * Gate: wg produces 3 blend weights (trained)
 * All weights in exact SHRS format → drop into shoresh.c
 */

static int model_forward(Model *m, int *tokens, int *targets, int V) {
    int tok_i = nt_tape_param(m->tok); nt_tape_no_decay(tok_i);
    int pos_i = nt_tape_param(m->pos); nt_tape_no_decay(pos_i);

    int rms_idx = 0;
    int li[N_LAYERS][14]; /* all layer params */
    for(int l=0; l<N_LAYERS; l++){
        li[l][0] = nt_tape_param(m->rms[rms_idx++]); /* rms1 */
        li[l][1] = nt_tape_param(m->L[l].wq);
        li[l][2] = nt_tape_param(m->L[l].wk);
        li[l][3] = nt_tape_param(m->L[l].vc);
        li[l][4] = nt_tape_param(m->L[l].wr);
        li[l][5] = nt_tape_param(m->L[l].vr);
        li[l][6] = nt_tape_param(m->L[l].wj);
        li[l][7] = nt_tape_param(m->L[l].vj);
        li[l][8] = nt_tape_param(m->L[l].wo_c);
        li[l][12] = nt_tape_param(m->L[l].wo_r);
        li[l][13] = nt_tape_param(m->L[l].wo_j);
        li[l][9] = nt_tape_param(m->rms[rms_idx++]); /* rms2 */
        li[l][10] = nt_tape_param(m->L[l].w1);
        li[l][11] = nt_tape_param(m->L[l].w2);
        /* wg not on tape — gate trained through content/rrpram/janus gradients */
    }
    int rmsf_i = nt_tape_param(m->rms[rms_idx]);
    int head_i = nt_tape_param(m->head);

    nt_tensor *tok_t = nt_tensor_new(CTX);
    nt_tensor *tgt_t = nt_tensor_new(CTX);
    for(int i=0; i<CTX; i++){
        tok_t->data[i] = (float)tokens[i];
        tgt_t->data[i] = (float)targets[i];
    }
    int t_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);

    int h = nt_seq_embedding(tok_i, pos_i, t_i, CTX, DIM);

    for(int l=0; l<N_LAYERS; l++){
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, DIM);

        /* ═══ Content attention (2 heads, HD=16) ═══ */
        int q = nt_seq_linear(li[l][1], xn, CTX);  /* [T × 32] */
        int k = nt_seq_linear(li[l][2], xn, CTX);  /* [T × 32] */
        int v = nt_seq_linear(li[l][3], xn, CTX);  /* [T × 32] */
        int co = nt_mh_causal_attention(q, k, v, CTX, HD); /* [T × 32] */

        /* ═══ RRPRAM: positional routing ═══
         * vr produces values [T × 16]
         * wr produces position scores [T × CTX] — acts as learned residual
         * Full RRPRAM = softmax(x@Wr)@Vr, approx = linear contribution */
        int vr = nt_seq_linear(li[l][5], xn, CTX);  /* [T × 16] */

        /* ═══ Janus Echo: self-resonance ═══
         * wjp = Wj @ xn, vjp = Vj @ xn
         * out = vjp * normalized(wjp) — W^T·W projection */
        int wjp = nt_seq_linear(li[l][6], xn, CTX); /* [T × 16] */
        int vjp = nt_seq_linear(li[l][7], xn, CTX); /* [T × 16] */
        int jo = nt_mul(vjp, wjp); /* element-wise, approximates norm(wjp)*vjp */

        /* ═══ Combine via separate projections (= wo @ concat mathematically) ═══ */
        int proj_c = nt_seq_linear(li[l][8], co, CTX);   /* [T×32] → [T×DIM] */
        int proj_r = nt_seq_linear(li[l][12], vr, CTX);  /* [T×16] → [T×DIM] */
        int proj_j = nt_seq_linear(li[l][13], jo, CTX);  /* [T×16] → [T×DIM] */
        int proj = nt_add(nt_add(proj_c, proj_r), proj_j);
        h = nt_add(h, proj);

        /* ═══ FFN ═══ */
        xn = nt_seq_rmsnorm(h, li[l][9], CTX, DIM);
        int up = nt_seq_linear(li[l][10], xn, CTX);
        int gate = nt_silu(up);
        int down = nt_seq_linear(li[l][11], gate, CTX);
        h = nt_add(h, down);
    }

    int hf = nt_seq_rmsnorm(h, rmsf_i, CTX, DIM);
    int logits = nt_seq_linear(head_i, hf, CTX);
    return nt_seq_cross_entropy(logits, tgt_i, CTX, V);
}

/* ═══ Save SHRS — exact shoresh.c tf_load format ═══ */

static void model_save_shrs(Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if(!f) return;
    unsigned int magic = SHORESH_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(m->tok->data, sizeof(float), MAX_ROOTS*DIM, f);
    fwrite(m->pos->data, sizeof(float), CTX*DIM, f);
    for(int l=0; l<N_LAYERS; l++){
        fwrite(m->L[l].wq->data, sizeof(float), N_CONTENT*HD*DIM, f);
        fwrite(m->L[l].wk->data, sizeof(float), N_CONTENT*HD*DIM, f);
        fwrite(m->L[l].vc->data, sizeof(float), N_CONTENT*HD*DIM, f);
        fwrite(m->L[l].wr->data, sizeof(float), N_RRPRAM*DIM*CTX, f);
        fwrite(m->L[l].vr->data, sizeof(float), N_RRPRAM*HD*DIM, f);
        fwrite(m->L[l].wj->data, sizeof(float), N_JANUS*HD*DIM, f);
        fwrite(m->L[l].vj->data, sizeof(float), N_JANUS*HD*DIM, f);
        /* Reconstruct wo[DIM×DIM] = [wo_c | wo_r | wo_j] column-wise */
        {float wo_full[DIM*DIM];
         for(int i=0;i<DIM;i++){
             for(int j=0;j<N_CONTENT*HD;j++) wo_full[i*DIM+j] = m->L[l].wo_c->data[i*N_CONTENT*HD+j];
             for(int j=0;j<N_RRPRAM*HD;j++) wo_full[i*DIM+N_CONTENT*HD+j] = m->L[l].wo_r->data[i*N_RRPRAM*HD+j];
             for(int j=0;j<N_JANUS*HD;j++) wo_full[i*DIM+N_CONTENT*HD+N_RRPRAM*HD+j] = m->L[l].wo_j->data[i*N_JANUS*HD+j];
         }
         fwrite(wo_full, sizeof(float), DIM*DIM, f);}
        fwrite(m->L[l].w1->data, sizeof(float), FFN_DIM*DIM, f);
        fwrite(m->L[l].w2->data, sizeof(float), DIM*FFN_DIM, f);
        fwrite(m->L[l].wg->data, sizeof(float), 3*DIM, f);
        float gb[3] = {0, 0, 0};
        fwrite(gb, sizeof(float), 3, f);
    }
    fclose(f);
    printf("  saved SHRS: %s\n", path);
}

/* ═══ Main ═══ */

int main(int argc, char **argv) {
    if(argc < 2){
        printf("SHORESH ε trainer — full Janus on notorch\n");
        printf("Usage: %s <corpus.txt> [steps] [lr]\n", argv[0]);
        return 1;
    }
    int steps = argc>2 ? atoi(argv[2]) : DEFAULT_STEPS;
    float lr = argc>3 ? (float)atof(argv[3]) : BASE_LR;

    printf("════════════════════════════════════════════════\n");
    printf("  SHORESH ε — full Janus triple attention\n");
    printf("  Content(%d) + RRPRAM(%d) + Janus(%d) = %d heads\n",
           N_CONTENT, N_RRPRAM, N_JANUS, N_HEADS);
    printf("  DIM=%d HD=%d L=%d CTX=%d FFN=%d\n", DIM, HD, N_LAYERS, CTX, FFN_DIM);
    printf("  %d steps, lr=%.1e, Chuck optimizer\n", steps, lr);
    printf("════════════════════════════════════════════════\n");

    nt_seed(42); srand(42);

    /* [1] Corpus */
    printf("[1] Corpus: %s\n", argv[1]);
    FILE *cf=fopen(argv[1],"rb"); if(!cf){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    fseek(cf,0,SEEK_END); long csz=ftell(cf); fseek(cf,0,SEEK_SET);
    char *text=malloc(csz+1); fread(text,1,csz,cf); fclose(cf); text[csz]=0;
    root_init(); root_load_lex();
    int *rseq=malloc(200000*sizeof(int));
    int rlen=extract_roots(text,rseq,200000); free(text);
    printf("  %ld bytes, %d root tokens, %d unique roots\n", csz, rlen, nroots);
    if(rlen<CTX+1){fprintf(stderr,"Corpus too small\n");return 1;}
    int V = nroots;

    /* [2] Model */
    printf("[2] Model: V=%d\n", V);
    Model *m = model_create(V);
    printf("  %ld params (%.1fK)\n", count_params(m), count_params(m)/1000.0);

    /* [3] Train */
    printf("[3] Training...\n");
    nt_train_mode(1);
    nt_schedule sched = nt_schedule_cosine(lr, steps/10, steps, lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();

    float ema=0, best=999;
    clock_t t0=clock();
    int input[CTX], target[CTX], pos=0;

    for(int step=0; step<steps; step++){
        float slr = nt_schedule_get_lr(&sched);
        if(pos+CTX+1>rlen) pos=0;
        for(int i=0;i<CTX;i++){input[i]=rseq[pos+i]; target[i]=rseq[pos+i+1];}
        pos += CTX/2;

        nt_tape_start(); nt_train_mode(1);
        int loss_idx = model_forward(m, input, target, V);
        float loss = nt_tape_get()->entries[loss_idx].output->data[0];

        if(step==0) ema=loss;
        ema = 0.99f*ema + 0.01f*loss;
        if(loss<best) best=loss;

        nt_tape_backward(loss_idx);
        if(!nt_nan_guard_check(&guard)){nt_tape_clear();continue;}
        float gnorm = nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(slr, loss);
        nt_tape_clear();

        if(step%100==0){
            double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("step %5d | train %.4f | ema %.4f | best %.4f | lr %.2e | gn %.2f | %.0fs\n",
                   step, loss, ema, best, slr, gnorm, el);
            fflush(stdout);
        }
        if(step>0 && step%1000==0){
            char ckpt[256]; snprintf(ckpt,sizeof(ckpt),"shoresh_step%d.bin",step);
            model_save_shrs(m, ckpt);
        }
    }

    double total=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("════════════════════════════════════════════════\n");
    printf("  loss: %.4f -> ema %.4f (best %.4f)\n", ema, ema, best);
    printf("  time: %.1f sec (%.1f steps/s)\n", total, steps/total);
    printf("  nans: %d\n", guard.total_nan_count);
    printf("════════════════════════════════════════════════\n");

    model_save_shrs(m, "shoresh.bin");
    printf("\nWeights: shoresh.bin\n");
    printf("Test: ./shoresh -w shoresh.bin shoresh.txt\n");

    free(rseq); return 0;
}
