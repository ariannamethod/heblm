/*
 * train_2m.c — SHORESH 2M: Janus triple attention, scaled
 *
 * DIM=160, L=6, H=8 (4C+2R+2J), FFN=640, CTX=96
 * ~1.87M params. Blessed by Rav Karpathy.
 *
 * Build:
 *   cc train_2m.c notorch.c -O2 -DUSE_BLAS -DACCELERATE \
 *      -framework Accelerate -lm -o train_2m
 *
 * Run:
 *   ./train_2m corpus.txt [steps] [lr] [max_tokens]
 *
 * (c) 2026 Oleg Ataeff & Claude Opus & Arianna Method
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══ Architecture — 2M scale ═══ */

#define HEB         22
#define MAX_ROOTS   1024
#define DIM         160
#define N_LAYERS    4
#define N_HEADS     8
#define N_CONTENT   4
#define N_RRPRAM    2
#define N_JANUS     2
#define HD          (DIM / N_HEADS)  /* 20 */
#define CTX         96
#define FFN_DIM     640

#define DEFAULT_STEPS   5000
#define BASE_LR         3e-4f
#define VAL_SPLIT       0.1f
#define VAL_EVERY       500
#define LOG_EVERY       50
#define CKPT_EVERY      2000
#define MAX_TOKENS      500000
#define SHORESH_MAGIC   0x53485253

/* ═══ Root extraction ═══ */

static int isheb(const unsigned char *p){return p[0]==0xD7&&p[1]>=0x90&&p[1]<=0xAA;}
static int u8let(const unsigned char *p){
    if(p[0]!=0xD7)return -1;
    switch(p[1]){
        case 0x90:return 0;case 0x91:return 1;case 0x92:return 2;case 0x93:return 3;
        case 0x94:return 4;case 0x95:return 5;case 0x96:return 6;case 0x97:return 7;
        case 0x98:return 8;case 0x99:return 9;case 0x9A:case 0x9B:return 10;
        case 0x9C:return 11;case 0x9D:case 0x9E:return 12;case 0x9F:case 0xA0:return 13;
        case 0xA1:return 14;case 0xA2:return 15;case 0xA3:case 0xA4:return 16;
        case 0xA5:case 0xA6:return 17;case 0xA7:return 18;case 0xA8:return 19;
        case 0xA9:return 20;case 0xAA:return 21;default:return -1;}
}
typedef struct{int c[3];}Root;
static Root roots[MAX_ROOTS]; static int nroots=0;
static int rhash[HEB*HEB*HEB];
static void rinit(void){memset(rhash,-1,sizeof(rhash));}
static int radd(int a,int b,int c){
    if(a<0||b<0||c<0)return -1;
    int k=a*HEB*HEB+b*HEB+c;if(rhash[k]>=0)return rhash[k];
    if(nroots>=MAX_ROOTS)return -1;
    int id=nroots++;roots[id].c[0]=a;roots[id].c[1]=b;roots[id].c[2]=c;rhash[k]=id;return id;
}
/* Known root lexicon */
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
static void rlex(void){for(int i=0;KR[i].a>=0;i++)radd(KR[i].a,KR[i].b,KR[i].c);}

static int sp(const int*l,int n){
    if(n>=6&&l[0]==5&&l[1]==4&&l[2]==21)return 3;
    if(n>=4){int a=l[0],b=l[1];if((a==4&&b==21)||(a==5&&b==4)||(a==20&&b==11)||(a==5&&b==1)||(a==5&&b==10)||(a==5&&b==11)||(a==5&&b==12)||(a==5&&b==20))return 2;}
    if(n>=3){int f=l[0];if(f==4||f==1||f==10||f==11||f==12||f==20||f==5||f==13||f==9||f==21||f==0)return 1;}
    return 0;
}
static int ss(const int*l,int n){
    if(n>=4){int a=l[n-2],b=l[n-1];if((a==9&&b==12)||(a==5&&b==21)||(a==9&&b==21)||(a==13&&b==9)||(a==4&&b==12)||(a==4&&b==13)||(a==10&&b==12)||(a==10&&b==13))return 2;}
    if(n>=3){int e=l[n-1];if(e==4||e==21||e==9||e==10||e==12||e==13)return 1;}
    return 0;
}
static int fr(const int*let,int n){
    if(n<2)return -1;
    int ps=sp(let,n);const int*s=let+ps;int sn=n-ps-ss(let+ps,n-ps);
    if(sn>=2){int best=-1,bsp=999;
        for(int ri=0;ri<nroots;ri++){int t[3]={roots[ri].c[0],roots[ri].c[1],roots[ri].c[2]};
            int p[3]={-1,-1,-1};int ti=0;for(int i=0;i<sn&&ti<3;i++)if(s[i]==t[ti]){p[ti]=i;ti++;}
            if(ti==3){int sp2=p[2]-p[0];if(sp2<bsp){bsp=sp2;best=ri;}}}
        if(best>=0)return best;}
    if(sn>=3)return radd(s[0],s[1],s[2]);
    if(sn==2)return radd(s[0],s[1],s[1]);
    return -1;
}
static int extract(const char*text,int*out,int mx){
    const unsigned char*p=(const unsigned char*)text;
    int n=0,wb[64],wn=0;
    while(*p&&n<mx){if(isheb(p)){if(wn<64)wb[wn++]=u8let(p);p+=2;}
        else{if(wn>0){int r=fr(wb,wn);if(r>=0)out[n++]=r;wn=0;}p++;}}
    if(wn>0){int r=fr(wb,wn);if(r>=0)out[n++]=r;}
    return n;
}

/* ═══ Model ═══ */

typedef struct {
    nt_tensor *tok, *pos;
    struct {
        nt_tensor *wq, *wk, *vc;  /* Content [N_CONTENT*HD × DIM] */
        nt_tensor *vr;              /* RRPRAM value [N_RRPRAM*HD × DIM] */
        nt_tensor *wj, *vj;        /* Janus [N_JANUS*HD × DIM] */
        nt_tensor *wo_c, *wo_r, *wo_j; /* Split projections → combined wo at save */
        nt_tensor *w1, *w2;        /* FFN */
    } L[N_LAYERS];
    nt_tensor *head;
    nt_tensor *rms[N_LAYERS*2+1]; int nrms;
} Model;

static Model *mcreate(int V) {
    Model *m = calloc(1, sizeof(Model));
    m->tok = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->tok, V, DIM);
    m->pos = nt_tensor_new2d(CTX, DIM); nt_tensor_xavier(m->pos, CTX, DIM);
    m->nrms = 0;
    float sc = 0.02f / sqrtf(2.0f * N_LAYERS);
    for(int l=0; l<N_LAYERS; l++){
        m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
        m->L[l].wq  = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].wq, DIM, N_CONTENT*HD);
        m->L[l].wk  = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].wk, DIM, N_CONTENT*HD);
        m->L[l].vc  = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(m->L[l].vc, DIM, N_CONTENT*HD);
        m->L[l].vr  = nt_tensor_new2d(N_RRPRAM*HD, DIM); nt_tensor_xavier(m->L[l].vr, DIM, N_RRPRAM*HD);
        m->L[l].wj  = nt_tensor_new2d(N_JANUS*HD, DIM); nt_tensor_xavier(m->L[l].wj, DIM, N_JANUS*HD);
        m->L[l].vj  = nt_tensor_new2d(N_JANUS*HD, DIM); nt_tensor_xavier(m->L[l].vj, DIM, N_JANUS*HD);
        m->L[l].wo_c = nt_tensor_new2d(DIM, N_CONTENT*HD); nt_tensor_xavier(m->L[l].wo_c, N_CONTENT*HD, DIM);
        m->L[l].wo_r = nt_tensor_new2d(DIM, N_RRPRAM*HD); nt_tensor_xavier(m->L[l].wo_r, N_RRPRAM*HD, DIM);
        m->L[l].wo_j = nt_tensor_new2d(DIM, N_JANUS*HD); nt_tensor_xavier(m->L[l].wo_j, N_JANUS*HD, DIM);
        for(int i=0;i<m->L[l].wo_c->len;i++) m->L[l].wo_c->data[i]*=sc/0.1f;
        for(int i=0;i<m->L[l].wo_r->len;i++) m->L[l].wo_r->data[i]*=sc/0.1f;
        for(int i=0;i<m->L[l].wo_j->len;i++) m->L[l].wo_j->data[i]*=sc/0.1f;
        m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
        m->L[l].w1 = nt_tensor_new2d(FFN_DIM, DIM); nt_tensor_xavier(m->L[l].w1, DIM, FFN_DIM);
        m->L[l].w2 = nt_tensor_new2d(DIM, FFN_DIM); nt_tensor_xavier(m->L[l].w2, FFN_DIM, DIM);
        for(int i=0;i<m->L[l].w2->len;i++) m->L[l].w2->data[i]*=sc/0.1f;
    }
    m->rms[m->nrms++] = nt_tensor_new(DIM); nt_tensor_fill(m->rms[m->nrms-1], 1.0f);
    m->head = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->head, DIM, V);
    return m;
}

static long nparams(Model *m) {
    long n = m->tok->len + m->pos->len + m->head->len;
    for(int l=0; l<N_LAYERS; l++)
        n += m->L[l].wq->len + m->L[l].wk->len + m->L[l].vc->len +
             m->L[l].vr->len + m->L[l].wj->len + m->L[l].vj->len +
             m->L[l].wo_c->len + m->L[l].wo_r->len + m->L[l].wo_j->len +
             m->L[l].w1->len + m->L[l].w2->len;
    for(int i=0;i<m->nrms;i++) n += m->rms[i]->len;
    return n;
}

/* ═══ Forward — Content + RRPRAM(value) + Janus(echo) ═══ */

static int mfwd(Model *m, int *tokens, int *targets, int V) {
    int tok_i = nt_tape_param(m->tok); nt_tape_no_decay(tok_i);
    int pos_i = nt_tape_param(m->pos); nt_tape_no_decay(pos_i);
    int ri = 0;
    int li[N_LAYERS][14];
    for(int l=0; l<N_LAYERS; l++){
        li[l][0]  = nt_tape_param(m->rms[ri++]);
        li[l][1]  = nt_tape_param(m->L[l].wq);
        li[l][2]  = nt_tape_param(m->L[l].wk);
        li[l][3]  = nt_tape_param(m->L[l].vc);
        li[l][4]  = nt_tape_param(m->L[l].vr);
        li[l][5]  = nt_tape_param(m->L[l].wj);
        li[l][6]  = nt_tape_param(m->L[l].vj);
        li[l][7]  = nt_tape_param(m->L[l].wo_c);
        li[l][8]  = nt_tape_param(m->L[l].wo_r);
        li[l][9]  = nt_tape_param(m->L[l].wo_j);
        li[l][10] = nt_tape_param(m->rms[ri++]);
        li[l][11] = nt_tape_param(m->L[l].w1);
        li[l][12] = nt_tape_param(m->L[l].w2);
    }
    int rmsf = nt_tape_param(m->rms[ri]);
    int hd = nt_tape_param(m->head);

    nt_tensor *tt = nt_tensor_new(CTX), *tg = nt_tensor_new(CTX);
    for(int i=0;i<CTX;i++){tt->data[i]=(float)tokens[i];tg->data[i]=(float)targets[i];}
    int ti = nt_tape_record(tt, NT_OP_NONE, -1, -1, 0);
    int tgi = nt_tape_record(tg, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tt); nt_tensor_free(tg);

    int h = nt_seq_embedding(tok_i, pos_i, ti, CTX, DIM);

    for(int l=0; l<N_LAYERS; l++){
        int xn = nt_seq_rmsnorm(h, li[l][0], CTX, DIM);

        /* Content QKV (4 heads, HD=20) */
        int q = nt_seq_linear(li[l][1], xn, CTX);
        int k = nt_seq_linear(li[l][2], xn, CTX);
        int v = nt_seq_linear(li[l][3], xn, CTX);
        int co = nt_mh_causal_attention(q, k, v, CTX, HD);

        /* RRPRAM value projection (2 heads) */
        int vr = nt_seq_linear(li[l][4], xn, CTX);

        /* Janus Echo: wjp * vjp (self-resonance) */
        int wjp = nt_seq_linear(li[l][5], xn, CTX);
        int vjp = nt_seq_linear(li[l][6], xn, CTX);
        int jo = nt_mul(vjp, wjp);

        /* Project each mechanism to DIM, sum */
        int pc = nt_seq_linear(li[l][7], co, CTX);
        int pr = nt_seq_linear(li[l][8], vr, CTX);
        int pj = nt_seq_linear(li[l][9], jo, CTX);
        h = nt_add(h, nt_add(nt_add(pc, pr), pj));

        /* FFN: SiLU + down */
        xn = nt_seq_rmsnorm(h, li[l][10], CTX, DIM);
        int up = nt_seq_linear(li[l][11], xn, CTX);
        int gate = nt_silu(up);
        int dn = nt_seq_linear(li[l][12], gate, CTX);
        h = nt_add(h, dn);
    }

    int hf = nt_seq_rmsnorm(h, rmsf, CTX, DIM);
    int logits = nt_seq_linear(hd, hf, CTX);
    return nt_seq_cross_entropy(logits, tgi, CTX, V);
}

/* ═══ Save ═══ */

static void msave(Model *m, const char *path) {
    int n = 2 + N_LAYERS*11 + m->nrms + 1; /* 9 attn + w1 + w2 per layer */
    nt_tensor **p = malloc(n*sizeof(nt_tensor*));
    int idx=0;
    p[idx++]=m->tok; p[idx++]=m->pos;
    int ri=0;
    for(int l=0;l<N_LAYERS;l++){
        p[idx++]=m->rms[ri++];
        p[idx++]=m->L[l].wq; p[idx++]=m->L[l].wk; p[idx++]=m->L[l].vc;
        p[idx++]=m->L[l].vr; p[idx++]=m->L[l].wj; p[idx++]=m->L[l].vj;
        p[idx++]=m->L[l].wo_c; p[idx++]=m->L[l].wo_r; p[idx++]=m->L[l].wo_j;
        p[idx++]=m->rms[ri++];
        p[idx++]=m->L[l].w1; p[idx++]=m->L[l].w2;
    }
    p[idx++]=m->rms[ri]; p[idx++]=m->head;
    nt_save(path,p,idx);
    free(p);
    printf("  saved: %s (%d tensors, %.1fMB)\n",path,idx,idx>0?m->tok->len*4.0f/1e6*10:0);
}

/* ═══ Main ═══ */

int main(int argc, char **argv) {
    if(argc<2){printf("Usage: %s corpus.txt [steps] [lr] [max_tokens]\n",argv[0]);return 1;}
    int steps = argc>2 ? atoi(argv[2]) : DEFAULT_STEPS;
    float lr = argc>3 ? (float)atof(argv[3]) : BASE_LR;
    int maxtok = argc>4 ? atoi(argv[4]) : MAX_TOKENS;

    printf("════════════════════════════════════════════════\n");
    printf("  SHORESH 2M — Janus triple attention\n");
    printf("  Content(%d)+RRPRAM(%d)+Janus(%d) = %d heads\n",N_CONTENT,N_RRPRAM,N_JANUS,N_HEADS);
    printf("  DIM=%d HD=%d L=%d CTX=%d FFN=%d\n",DIM,HD,N_LAYERS,CTX,FFN_DIM);
    printf("  %d steps, lr=%.1e, max_tokens=%d\n",steps,lr,maxtok);
    printf("════════════════════════════════════════════════\n");

    nt_seed(42); srand(42);

    /* [1] Extract roots */
    printf("[1] Corpus: %s\n",argv[1]);
    FILE*f=fopen(argv[1],"rb");if(!f){fprintf(stderr,"Cannot open\n");return 1;}
    fseek(f,0,SEEK_END);long csz=ftell(f);fseek(f,0,SEEK_SET);
    char*text=malloc(csz+1);fread(text,1,csz,f);fclose(f);text[csz]=0;
    printf("  %ld bytes (%.1fMB)\n",csz,csz/1e6);

    rinit(); rlex();
    int *rseq = malloc(maxtok * sizeof(int));
    int rlen = extract(text, rseq, maxtok);
    free(text);
    printf("  %d root tokens, %d unique roots\n", rlen, nroots);
    if(rlen<CTX+1){fprintf(stderr,"Too few tokens\n");return 1;}
    int V = nroots;

    /* Val split */
    int val_start = (int)(rlen * (1.0f - VAL_SPLIT));
    printf("  train: %d tokens, val: %d tokens\n", val_start, rlen-val_start);

    /* [2] Model */
    printf("[2] Model: V=%d\n",V);
    Model *m = mcreate(V);
    long np = nparams(m);
    printf("  %ld params (%.2fM)\n", np, np/1e6);
    printf("  ratio: %.1f params/token\n", (float)np/(float)val_start);

    /* [3] Train */
    printf("[3] Training...\n");
    nt_train_mode(1);
    nt_schedule sched = nt_schedule_cosine(lr, 200, steps, lr*0.03f);
    nt_nan_guard guard = nt_nan_guard_new();

    float ema=0, best=999, best_val=999;
    clock_t t0=clock();
    int input[CTX], target[CTX], pos=0;

    for(int step=0; step<steps; step++){
        float slr = nt_schedule_get_lr(&sched);
        if(pos+CTX+1>val_start) pos=0;
        for(int i=0;i<CTX;i++){input[i]=rseq[pos+i];target[i]=rseq[pos+i+1];}
        pos += CTX/2;

        nt_tape_start(); nt_train_mode(1);
        int loss_idx = mfwd(m, input, target, V);
        float loss = nt_tape_get()->entries[loss_idx].output->data[0];

        if(step==0) ema=loss;
        ema = 0.99f*ema + 0.01f*loss;
        if(loss<best) best=loss;

        nt_tape_backward(loss_idx);
        if(!nt_nan_guard_check(&guard)){nt_tape_clear();continue;}
        float gn = nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(slr, loss);
        nt_tape_clear();

        if(step%LOG_EVERY==0){
            double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("step %5d | train %.4f | ema %.4f | best %.4f | lr %.2e | gn %.1f | %.0fs\n",
                   step,loss,ema,best,slr,gn,el);
            fflush(stdout);
        }

        /* Validation — sample 20 batches (not all, to avoid memory pressure) */
        if(step>0 && step%VAL_EVERY==0){
            float vloss=0; int vn=0;
            int vp=val_start;
            for(int vb=0; vb<20 && vp+CTX+1<rlen; vb++, vp+=CTX*3){
                int vi[CTX],vt[CTX];
                for(int i=0;i<CTX;i++){vi[i]=rseq[vp+i];vt[i]=rseq[vp+i+1];}
                nt_tape_start(); nt_train_mode(0);
                int vli = mfwd(m,vi,vt,V);
                vloss += nt_tape_get()->entries[vli].output->data[0];
                nt_tape_clear(); vn++;
            }
            vloss /= (vn>0?vn:1);
            if(vloss<best_val) best_val=vloss;
            printf("  VAL step %d | val %.4f | best_val %.4f | gap %.4f\n",
                   step, vloss, best_val, vloss-ema);
            fflush(stdout);
            nt_train_mode(1);
        }

        if(step>0 && step%CKPT_EVERY==0){
            char ckpt[256]; snprintf(ckpt,sizeof(ckpt),"shoresh_2m_step%d.bin",step);
            msave(m,ckpt);
        }
    }

    double total=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("════════════════════════════════════════════════\n");
    printf("  train ema: %.4f | best: %.4f\n", ema, best);
    printf("  val best: %.4f\n", best_val);
    printf("  time: %.1f sec (%.1f steps/s)\n", total, steps/total);
    printf("  nans: %d\n", guard.total_nan_count);
    printf("════════════════════════════════════════════════\n");

    msave(m, "shoresh_2m.bin");
    free(rseq); return 0;
}
