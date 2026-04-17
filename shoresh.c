/*
 * shoresh.c v2 — שורש — Hebrew Root Resonance Engine
 *
 * θ = ε + γ + αδ
 *   ε = Janus triple attention on root embeddings (QKV + RRPRAM + Echo)
 *   γ = root metaweights (bigram + trigram + hebbian + prophecy)
 *   δ = Klaus chambers (somatic modulation by root emotional valence)
 *   α = calendar drift + prophecy debt
 *
 * Architecture:
 *   Level 1: Root Engine — Hebrew text → 3-letter roots
 *   Level 2: Root MetaWeights — statistical field between roots
 *   Level 3: Janus Triple Attention — QKV + RRPRAM + Echo on roots
 *   Level 4: Transformer Gate — untrained=silent, trained=speaks
 *   Level 5: Klaus Chambers — 6 somatic oscillators, Kuramoto coupled
 *   Level 6: Word Realization — root → surface word via char-field
 *   Level 7: 12-Step Bidirectional Chain + SPA
 *
 * Weightless mode: gate suppresses untrained transformer.
 * Pure metaweight generation from root co-occurrence field.
 * When weights loaded: full θ = ε + γ + αδ.
 *
 * Compile:
 *   cc shoresh.c -O2 -lm -o shoresh
 *
 * Run:
 *   ./shoresh shoresh.txt                    # metaweights only
 *   ./shoresh shoresh.txt "שלום עולם"        # with prompt
 *   ./shoresh -w shoresh.bin shoresh.txt     # with trained weights
 *
 * (c) 2026 Oleg Ataeff & Claude Opus & Arianna Method
 * הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef SHORESH_TRAIN
#include "notorch.h"
#endif

/* ═══════════════════════════════════════════════════════════════════
 * CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════ */

#define HEB         22
#define MAX_ROOTS   1024
#define ROOT_LIMIT  615     /* natural coverage from shoresh.txt, emergence preserved */

/* Semantic BPE vocab layout */
#define TOK_SPACE     22
#define TOK_RSTART    23
#define TOK_REND      24
#define TOK_FUNC_BASE 25
#define N_FUNC        15
#define TOK_ROOT_BASE 40
#define SBPE_VOCAB    (TOK_ROOT_BASE + ROOT_LIMIT)  /* 40 + 615 = 655 */
#define MAX_WORDS   8192
#define MAX_CWORDS  200000
#define MAX_RWORDS  64
#define MAX_TEXT    (8*1024*1024)

/* Transformer */
#define DIM         200
#define N_LAYERS    6
#define N_HEADS     10      /* 6 content + 2 RRPRAM + 2 Janus Echo */
#define N_CONTENT   6
#define N_RRPRAM    2
#define N_JANUS     2
#define HD          (DIM/N_HEADS)  /* 20 */
#define CTX         96
#define FFN         800

/* MetaWeights */
#define MAX_BI      65536
#define MAX_TRI     65536
#define MAX_HEBB    65536
#define MAX_PROPH   32
#define HEBB_WIN    6

/* Generation */
#define CHAIN       12
#define TOP_K       12
#define SPA_DIM     16
#define MAX_GEN     24
#define TEMP        0.40f

/* Klaus */
#define N_CH        6
#define CH_FEAR     0
#define CH_LOVE     1
#define CH_RAGE     2
#define CH_VOID     3
#define CH_FLOW     4
#define CH_CMPLX    5

/* File format */
#define SHORESH_MAGIC 0x53485253  /* SHRS */

/* ═══════════════════════════════════════════════════════════════════
 * HEBREW ALPHABET + ORDINAL
 * ═══════════════════════════════════════════════════════════════════ */

static const char *LET[HEB] = {
    "א","ב","ג","ד","ה","ו","ז","ח","ט","י",
    "כ","ל","מ","נ","ס","ע","פ","צ","ק","ר","ש","ת"
};
static const int ORD[HEB] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22};

/* Sefer Yetzirah groups: 0=mother(אמש), 1=double(בגדכפרת), 2=simple */
static const int SYG[HEB] = {0,1,1,1,2,2,2,2,2,2,1,2,0,2,2,2,1,2,2,1,0,1};

static int u8let(const unsigned char *p, int *adv) {
    *adv = 1;
    if (p[0] != 0xD7) return -1;
    *adv = 2;
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
static int isheb(const unsigned char *p) {
    return p[0]==0xD7 && p[1]>=0x90 && p[1]<=0xAA;
}

/* ═══════════════════════════════════════════════════════════════════
 * MATH HELPERS
 * ═══════════════════════════════════════════════════════════════════ */

static float cf(float x,float lo,float hi){return x<lo?lo:(x>hi?hi:x);}

static void rmsnorm(float *o, const float *x, int n) {
    float ms=0; for(int i=0;i<n;i++) ms+=x[i]*x[i];
    ms=1.0f/sqrtf(ms/(float)n+1e-6f);
    for(int i=0;i<n;i++) o[i]=x[i]*ms;
}

static void matvec(float *o, const float *W, const float *x, int rows, int cols) {
    for(int i=0;i<rows;i++){float s=0;for(int j=0;j<cols;j++)s+=W[i*cols+j]*x[j];o[i]=s;}
}

static void softmax_n(float *x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0) for(int i=0;i<n;i++) x[i]/=s;
}

static float randf(void){return (float)rand()/(float)RAND_MAX;}
static float randg(float std){
    float u1=randf()+1e-10f, u2=randf();
    return std*sqrtf(-2.0f*logf(u1))*cosf(6.2831853f*u2);
}

/* ═══════════════════════════════════════════════════════════════════
 * ROOT ENGINE (Level 1)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct { int c[3]; } Root;
typedef struct {
    char utf8[64]; int letters[32]; int nlet; int root_id; int count;
} HWord;

typedef struct {
    Root roots[MAX_ROOTS]; int nr;
    int rhash[HEB*HEB*HEB];
    HWord words[MAX_WORDS]; int nw;
    int rwords[MAX_ROOTS][MAX_RWORDS]; int rwc[MAX_ROOTS];
    int *corpus_rids; int ncr;
    int *corpus_wids; int ncw;
} RootEng;

static int rhkey(int a,int b,int c){return a*HEB*HEB+b*HEB+c;}

static void re_init(RootEng *r){memset(r,0,sizeof(*r));memset(r->rhash,-1,sizeof(r->rhash));}

static int re_add_root(RootEng *r, int c1, int c2, int c3) {
    if(c1<0||c2<0||c3<0) return -1;
    int k=rhkey(c1,c2,c3); if(r->rhash[k]>=0) return r->rhash[k];
    if(r->nr>=ROOT_LIMIT) return -1;
    int id=r->nr++; r->roots[id].c[0]=c1;r->roots[id].c[1]=c2;r->roots[id].c[2]=c3;
    r->rhash[k]=id; return id;
}

/* Known roots: movement, emotion, creation, destruction, knowledge, light,
   darkness, speech, healing, time, body, power, sanctity, nature, social,
   war, growth, binding, truth, mind, pitomadom, common */
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

/* Root → Klaus chamber mapping (emotional valence of root families) */
static const int ROOT_CHAMBER[] = {
    /* movement: FLOW */ 4,4,4,4,4,4,
    /* emotion+: LOVE */ 1,1,1,1,1,1,
    /* emotion-: FEAR */ 0,0,0,0,0,0,
    /* creation: FLOW */ 4,4,4,4,4,4,
    /* destruction: RAGE */ 2,2,2,2,2,2,
    /* knowledge: CMPLX */ 5,5,5,5,5,
    /* light: LOVE */ 1,1,1,1,1,
    /* darkness: VOID */ 3,3,3,3,
    /* speech: FLOW */ 4,4,4,4,4,4,
    /* healing: LOVE */ 1,1,1,1,
    /* time: CMPLX */ 5,5,5,5,
    /* body: FLOW */ 4,4,4,4,4,4,4,4,4,4,
    /* power: RAGE */ 2,2,2,2,2,2,2,
    /* sanctity: LOVE */ 1,1,1,1,1,1,1,
    /* nature: FLOW */ 4,4,4,4,4,4,4,4,4,
    /* social: LOVE */ 1,1,1,1,1,1,1,1,
    /* war: RAGE */ 2,2,2,2,2,2,
    /* growth: FLOW */ 4,4,4,4,4,4,
    /* binding: CMPLX */ 5,5,5,5,5,5,
    /* truth: CMPLX */ 5,5,5,5,5,5,
    /* mind: CMPLX */ 5,5,5,5,5,
    /* common: varies */ 4,5,5,5,5,5,5,4,1,4,
};

static void re_load_lex(RootEng *r) {
    for(int i=0;KR[i].a>=0;i++) re_add_root(r,KR[i].a,KR[i].b,KR[i].c);
}

static int w2let(const char *u, int *o, int mx) {
    const unsigned char *p=(const unsigned char*)u; int n=0;
    while(*p&&n<mx){int a;int l=u8let(p,&a);if(l>=0)o[n++]=l;p+=a;} return n;
}

static int re_find_root(RootEng *r, const int *let, int n) {
    if(n<2) return -1;
    int best=-1, bs=999, bspan=999;
    for(int ri=0;ri<r->nr;ri++){
        int t[3]={r->roots[ri].c[0],r->roots[ri].c[1],r->roots[ri].c[2]};
        int pos[3]={-1,-1,-1}; int ti=0;
        for(int i=0;i<n&&ti<3;i++) if(let[i]==t[ti]){pos[ti]=i;ti++;}
        if(ti==3){int s=pos[0],sp=pos[2]-pos[0];
            if(s<bs||(s==bs&&sp<bspan)){bs=s;bspan=sp;best=ri;}}
    }
    if(best>=0) return best;
    /* Heuristic fallback: strip prefix, take 3 */
    int start=0;
    if(n>=4 && ((let[0]==5&&let[1]==4)||(let[0]==4&&let[1]==21)||(let[0]==20&&let[1]==11))) start=2;
    else if(n>=3){int f=let[0];
        if(f==4||f==1||f==10||f==11||f==12||f==20||f==5||f==13||f==9||f==21||f==0) start=1;}
    int s[32]; int sn=0; for(int i=start;i<n&&sn<32;i++) s[sn++]=let[i];
    if(sn<2) return -1;
    if(sn==2) return re_add_root(r,s[0],s[1],s[1]);
    return re_add_root(r,s[0],s[1],s[2]);
}

static int re_add_word(RootEng *r, const char *u) {
    for(int i=0;i<r->nw;i++) if(strcmp(r->words[i].utf8,u)==0){r->words[i].count++;return i;}
    if(r->nw>=MAX_WORDS) return -1;
    int wi=r->nw++; HWord *w=&r->words[wi];
    strncpy(w->utf8,u,63);w->utf8[63]=0;
    w->nlet=w2let(u,w->letters,32); w->count=1;
    w->root_id=re_find_root(r,w->letters,w->nlet);
    if(w->root_id>=0 && r->rwc[w->root_id]<MAX_RWORDS)
        r->rwords[w->root_id][r->rwc[w->root_id]++]=wi;
    return wi;
}

static int re_extract(RootEng *r, const char *text, int *wids, int *rids, int mx) {
    const unsigned char *p=(const unsigned char*)text;
    int n=0; char wb[128]; int wp=0;
    while(*p && n<mx) {
        if(isheb(p)){if(wp<126){wb[wp++]=p[0];wb[wp++]=p[1];}p+=2;}
        else{if(wp>0){wb[wp]=0;int wi=re_add_word(r,wb);
            if(wi>=0){if(wids)wids[n]=wi;if(rids)rids[n]=r->words[wi].root_id;n++;}wp=0;}p++;}
    }
    if(wp>0){wb[wp]=0;int wi=re_add_word(r,wb);
        if(wi>=0){if(wids)wids[n]=wi;if(rids)rids[n]=r->words[wi].root_id;n++;}}
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * ROOT METAWEIGHTS (Level 2) — γ
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct{int a,b;float p;}Bi;
typedef struct{int a,b,c;float p;}Tri;
typedef struct{int a,b;float s;}Heb;
typedef struct{int t;float s;int age;}Pro;

typedef struct {
    float uni[MAX_ROOTS];
    Bi bi[MAX_BI]; int nbi;
    Tri tri[MAX_TRI]; int ntri;
    Heb hebb[MAX_HEBB]; int nhebb;
    Pro proph[MAX_PROPH]; int nproph;
    float ord_mom;
} RMeta;

static float rm_bi(RMeta *m,int a,int b){
    for(int i=0;i<m->nbi;i++) if(m->bi[i].a==a&&m->bi[i].b==b) return m->bi[i].p;
    return 1e-10f;
}
static float rm_tri(RMeta *m,int a,int b,int c){
    for(int i=0;i<m->ntri;i++) if(m->tri[i].a==a&&m->tri[i].b==b&&m->tri[i].c==c) return m->tri[i].p;
    return 1e-10f;
}

static void rm_build(RMeta *m, const int *seq, int n, int nr) {
    memset(m,0,sizeof(*m));
    for(int i=0;i<n;i++) if(seq[i]>=0&&seq[i]<nr) m->uni[seq[i]]+=1.0f;
    float t=0; for(int i=0;i<nr;i++) t+=m->uni[i];
    if(t>0) for(int i=0;i<nr;i++) m->uni[i]/=t;

    /* Bigram */
    for(int i=0;i<n-1;i++){
        int a=seq[i],b=seq[i+1]; if(a<0||b<0) continue;
        int f=0; for(int j=0;j<m->nbi;j++) if(m->bi[j].a==a&&m->bi[j].b==b){m->bi[j].p+=1;f=1;break;}
        if(!f&&m->nbi<MAX_BI){m->bi[m->nbi].a=a;m->bi[m->nbi].b=b;m->bi[m->nbi].p=1;m->nbi++;}
    }
    for(int i=0;i<m->nbi;i++){float tt=0;
        for(int j=0;j<m->nbi;j++) if(m->bi[j].a==m->bi[i].a) tt+=m->bi[j].p;
        if(tt>0) m->bi[i].p/=tt;}

    /* Trigram */
    for(int i=0;i<n-2;i++){
        int a=seq[i],b=seq[i+1],c=seq[i+2]; if(a<0||b<0||c<0) continue;
        int f=0; for(int j=0;j<m->ntri;j++) if(m->tri[j].a==a&&m->tri[j].b==b&&m->tri[j].c==c){m->tri[j].p+=1;f=1;break;}
        if(!f&&m->ntri<MAX_TRI){m->tri[m->ntri].a=a;m->tri[m->ntri].b=b;m->tri[m->ntri].c=c;m->tri[m->ntri].p=1;m->ntri++;}
    }
    for(int i=0;i<m->ntri;i++){float tt=0;
        for(int j=0;j<m->ntri;j++) if(m->tri[j].a==m->tri[i].a&&m->tri[j].b==m->tri[i].b) tt+=m->tri[j].p;
        if(tt>0) m->tri[i].p/=tt;}

    /* Hebbian */
    int hn=n<30000?n:30000;
    for(int i=0;i<hn;i++){if(seq[i]<0)continue;
        int lo=i-HEBB_WIN;if(lo<0)lo=0;int hi=i+HEBB_WIN;if(hi>=hn)hi=hn-1;
        for(int j=lo;j<=hi;j++){if(i==j||seq[j]<0)continue;
            int a=seq[i]<seq[j]?seq[i]:seq[j],b=seq[i]<seq[j]?seq[j]:seq[i];
            float d=1.0f/(1.0f+fabsf((float)(i-j)));
            int f=0;for(int k=0;k<m->nhebb;k++) if(m->hebb[k].a==a&&m->hebb[k].b==b){m->hebb[k].s+=d;f=1;break;}
            if(!f&&m->nhebb<MAX_HEBB){m->hebb[m->nhebb].a=a;m->hebb[m->nhebb].b=b;m->hebb[m->nhebb].s=d;m->nhebb++;}
        }
    }
    float mx=0;for(int i=0;i<m->nhebb;i++) if(m->hebb[i].s>mx) mx=m->hebb[i].s;
    if(mx>0) for(int i=0;i<m->nhebb;i++) m->hebb[i].s/=mx;

    printf("  γ field: %d bigram, %d trigram, %d hebbian\n",m->nbi,m->ntri,m->nhebb);
}

static void rm_q_hebb(RMeta *m, const int *ctx, int n, float *out, int nr) {
    for(int i=0;i<nr;i++) out[i]=0;
    int s=n-8;if(s<0)s=0;
    for(int j=s;j<n;j++){if(ctx[j]<0)continue;
        for(int k=0;k<m->nhebb;k++){
            if(m->hebb[k].a==ctx[j]&&m->hebb[k].b<nr) out[m->hebb[k].b]+=m->hebb[k].s;
            else if(m->hebb[k].b==ctx[j]&&m->hebb[k].a<nr) out[m->hebb[k].a]+=m->hebb[k].s;
        }
    }
    float mx=0;for(int i=0;i<nr;i++) if(out[i]>mx) mx=out[i];
    if(mx>1e-12f) for(int i=0;i<nr;i++) out[i]/=mx;
}

static void rm_q_proph(RMeta *m, const int *ctx, int n, float *out, int nr) {
    for(int i=0;i<nr;i++) out[i]=0;
    int seen[MAX_ROOTS]={0};
    for(int i=0;i<n;i++) if(ctx[i]>=0&&ctx[i]<nr) seen[ctx[i]]=1;
    int s=n-6;if(s<0)s=0;
    for(int j=s;j<n;j++){if(ctx[j]<0)continue;float dc=1.0f/(1.0f+(float)(n-1-j));
        for(int k=0;k<m->nbi;k++) if(m->bi[k].a==ctx[j]&&m->bi[k].b<nr&&!seen[m->bi[k].b])
            out[m->bi[k].b]+=m->bi[k].p*dc;
    }
    if(n>=2){int p0=ctx[n-2],p1=ctx[n-1];
        for(int k=0;k<m->ntri;k++) if(m->tri[k].a==p0&&m->tri[k].b==p1&&m->tri[k].c<nr&&!seen[m->tri[k].c])
            out[m->tri[k].c]+=m->tri[k].p*1.5f;
    }
    for(int i=0;i<m->nproph;i++){int t=m->proph[i].t;
        if(t>=0&&t<nr&&!seen[t]) out[t]+=m->proph[i].s*logf(1.0f+(float)m->proph[i].age);}
    float mx=0;for(int i=0;i<nr;i++) if(out[i]>mx) mx=out[i];
    if(mx>1e-12f) for(int i=0;i<nr;i++) out[i]/=mx;
}

static void proph_add(RMeta *m, int t, float s) {
    if(t<0)return;
    for(int i=0;i<m->nproph;i++) if(m->proph[i].t==t){m->proph[i].s=fmaxf(m->proph[i].s,s);m->proph[i].age=0;return;}
    if(m->nproph<MAX_PROPH){m->proph[m->nproph].t=t;m->proph[m->nproph].s=s;m->proph[m->nproph].age=0;m->nproph++;}
}
static void proph_update(RMeta *m, int tok) {
    int k=0;for(int i=0;i<m->nproph;i++){if(m->proph[i].t==tok)continue;
        m->proph[i].age++;m->proph[i].s*=0.99f;
        if(m->proph[i].age<30&&m->proph[i].s>0.01f) m->proph[k++]=m->proph[i];}
    m->nproph=k;
}
static float proph_pressure(RMeta *m){float t=0;
    for(int i=0;i<m->nproph;i++) t+=m->proph[i].s*logf(1.0f+(float)m->proph[i].age);
    return cf(t/4.0f,0,1);}

/* ═══════════════════════════════════════════════════════════════════
 * JANUS TRIPLE ATTENTION TRANSFORMER (Level 3) — ε
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    float *wq, *wk, *vc;       /* Content: [N_CONTENT*HD × DIM] each */
    float *wr, *vr;             /* RRPRAM: [N_RRPRAM*DIM × CTX], [N_RRPRAM*HD × DIM] */
    float *wj, *vj;             /* Janus Echo: [N_JANUS*HD × DIM] each */
    float *wo;                  /* Output: [DIM × DIM] */
    float *w1, *w2;             /* SwiGLU MLP: [FFN×DIM], [DIM×FFN] */
    float *wg;                  /* Gate blend: [3×DIM] — content/rrpram/janus gate */
    float gb[3];                /* Gate bias */
} TFLayer;

typedef struct {
    int V, loaded;
    float *tok;                 /* [MAX_ROOTS × DIM] */
    float *pos;                 /* [CTX × DIM] */
    TFLayer L[N_LAYERS];
    float *lm_head;             /* [MAX_ROOTS × DIM] (tied to tok) */
    float logits[SBPE_VOCAB];
    /* KV cache */
    float *kc[N_LAYERS], *vc_cache[N_LAYERS], *vr_cache[N_LAYERS];
    int clen;
} TF;

static float *alloc_f(int n) {
    float *p = calloc(n, sizeof(float));
    /* Xavier-ish init */
    for(int i=0;i<n;i++) p[i] = randg(0.02f);
    return p;
}

static void tf_init(TF *t, int V) {
    memset(t, 0, sizeof(*t));
    t->V = SBPE_VOCAB; t->loaded = 0; (void)V;
    t->tok = alloc_f(SBPE_VOCAB * DIM);  /* covers roots + chars + func + special */
    t->pos = alloc_f(CTX * DIM);
    for(int l=0;l<N_LAYERS;l++){
        TFLayer *ly = &t->L[l];
        ly->wq = alloc_f(N_CONTENT*HD*DIM);
        ly->wk = alloc_f(N_CONTENT*HD*DIM);
        ly->vc = alloc_f(N_CONTENT*HD*DIM);
        ly->wr = alloc_f(N_RRPRAM*DIM*CTX);
        ly->vr = alloc_f(N_RRPRAM*HD*DIM);
        ly->wj = alloc_f(N_JANUS*HD*DIM);
        ly->vj = alloc_f(N_JANUS*HD*DIM);
        ly->wo = alloc_f(DIM*DIM);
        ly->w1 = alloc_f(FFN*DIM);
        ly->w2 = alloc_f(DIM*FFN);
        ly->wg = alloc_f(3*DIM);
        ly->gb[0]=ly->gb[1]=ly->gb[2]=0;
        /* KV caches */
        t->kc[l] = calloc(CTX*N_CONTENT*HD, sizeof(float));
        t->vc_cache[l] = calloc(CTX*N_CONTENT*HD, sizeof(float));
        t->vr_cache[l] = calloc(CTX*N_RRPRAM*HD, sizeof(float));
    }
    t->lm_head = t->tok; /* weight tying */
}

static void tf_reset(TF *t) { t->clen = 0; }

/* Forward one root token through the transformer */
static void tf_forward(TF *t, int tok, int pos_id) {
    int V=t->V, sl=pos_id+1;
    float x[DIM], xn[DIM], xr[DIM];

    /* Embedding */
    for(int d=0;d<DIM;d++) x[d] = t->tok[tok*DIM+d] + t->pos[pos_id*DIM+d];

    for(int li=0;li<N_LAYERS;li++){
        TFLayer *ly = &t->L[li];
        for(int d=0;d<DIM;d++) xr[d]=x[d];
        rmsnorm(xn, x, DIM);

        float co[N_CONTENT*HD], ro[N_RRPRAM*HD], jo[N_JANUS*HD];

        /* Content attention (QKV) */
        {float q[N_CONTENT*HD], k[N_CONTENT*HD], v[N_CONTENT*HD];
         matvec(q, ly->wq, xn, N_CONTENT*HD, DIM);
         matvec(k, ly->wk, xn, N_CONTENT*HD, DIM);
         matvec(v, ly->vc, xn, N_CONTENT*HD, DIM);
         for(int d=0;d<N_CONTENT*HD;d++){t->kc[li][pos_id*N_CONTENT*HD+d]=k[d];
             t->vc_cache[li][pos_id*N_CONTENT*HD+d]=v[d];}
         for(int h=0;h<N_CONTENT;h++){
             float sc[CTX];
             for(int p=0;p<sl;p++){float dot=0;
                 for(int d=0;d<HD;d++) dot+=q[h*HD+d]*t->kc[li][p*N_CONTENT*HD+h*HD+d];
                 sc[p]=dot/sqrtf((float)HD);}
             softmax_n(sc, sl);
             for(int d=0;d<HD;d++){float v2=0;
                 for(int p=0;p<sl;p++) v2+=sc[p]*t->vc_cache[li][p*N_CONTENT*HD+h*HD+d];
                 co[h*HD+d]=v2;}
         }}

        /* RRPRAM attention (x @ Wr → position pattern) */
        {float vr[N_RRPRAM*HD];
         matvec(vr, ly->vr, xn, N_RRPRAM*HD, DIM);
         for(int d=0;d<N_RRPRAM*HD;d++) t->vr_cache[li][pos_id*N_RRPRAM*HD+d]=vr[d];
         for(int h=0;h<N_RRPRAM;h++){
             float sc[CTX];
             for(int p=0;p<sl;p++){float s=0;
                 for(int d=0;d<DIM;d++) s+=xn[d]*ly->wr[(h*DIM+d)*CTX+p];
                 sc[p]=s;}
             softmax_n(sc, sl);
             for(int d=0;d<HD;d++){float v2=0;
                 for(int p=0;p<sl;p++) v2+=sc[p]*t->vr_cache[li][p*N_RRPRAM*HD+h*HD+d];
                 ro[h*HD+d]=v2;}
         }}

        /* Janus Echo attention (W^T · W self-resonance) */
        {float wjp[N_JANUS*HD], vjp[N_JANUS*HD];
         matvec(wjp, ly->wj, xn, N_JANUS*HD, DIM);
         matvec(vjp, ly->vj, xn, N_JANUS*HD, DIM);
         float norm=0;for(int d=0;d<N_JANUS*HD;d++) norm+=wjp[d]*wjp[d];
         norm=1.0f/sqrtf(norm+1e-8f);
         for(int d=0;d<N_JANUS*HD;d++) jo[d]=vjp[d]*(wjp[d]*norm);
        }

        /* Gating: learned blend of 3 mechanisms */
        float gl[3]; matvec(gl, ly->wg, xn, 3, DIM);
        float gates[3]; for(int g=0;g<3;g++) gates[g]=1.0f/(1.0f+expf(-(gl[g]+ly->gb[g])));

        /* Combine */
        float comb[DIM]; memset(comb,0,sizeof(comb));
        for(int d=0;d<N_CONTENT*HD;d++) comb[d]+=gates[0]*co[d];
        for(int d=0;d<N_RRPRAM*HD;d++) comb[N_CONTENT*HD+d]+=gates[1]*ro[d];
        for(int d=0;d<N_JANUS*HD;d++)  comb[N_CONTENT*HD+N_RRPRAM*HD+d]+=gates[2]*jo[d];

        /* Output projection + residual */
        float proj[DIM]; matvec(proj, ly->wo, comb, DIM, DIM);
        for(int d=0;d<DIM;d++) x[d]=xr[d]+proj[d];

        /* SwiGLU MLP */
        for(int d=0;d<DIM;d++) xr[d]=x[d];
        rmsnorm(xn, x, DIM);
        float up[FFN]; matvec(up, ly->w1, xn, FFN, DIM);
        /* SwiGLU: first half is gate, second half is value */
        for(int d=0;d<FFN/2;d++){
            float gate_v = up[d] * (1.0f/(1.0f+expf(-up[d]))); /* swish */
            up[d] = gate_v * up[FFN/2+d];
        }
        float dn[DIM]; matvec(dn, ly->w2, up, DIM, FFN);
        for(int d=0;d<DIM;d++) x[d]=xr[d]+dn[d];
    }

    /* LM head */
    rmsnorm(xn, x, DIM);
    for(int v=0;v<V;v++){float dot=0;
        for(int d=0;d<DIM;d++) dot+=xn[d]*t->lm_head[v*DIM+d];
        t->logits[v]=dot;}

    /* ═══ TRANSFORMER GATE ═══ */
    /* Untrained weights → small logit magnitude → gate ≈ 0 → transformer silent */
    /* Trained weights → large logit magnitude → gate ≈ 1 → transformer speaks */
    float mag=0; for(int v=0;v<V;v++) mag+=fabsf(t->logits[v]);
    mag/=(float)(V>0?V:1);
    float tg = cf((mag-0.5f)/1.5f, 0, 1);
    for(int v=0;v<V;v++) t->logits[v]*=tg;
    t->clen = sl;
}

/* ═══════════════════════════════════════════════════════════════════
 * KLAUS CHAMBERS (Level 5) — δ
 * ═══════════════════════════════════════════════════════════════════ */

static const char *CH_N[]={"FEAR","LOVE","RAGE","VOID","FLOW","CMPLX"};
static const float CH_D[]={0.90f,0.93f,0.85f,0.97f,0.88f,0.94f};
static const float COU[6][6]={
    { 0.0,-0.3, 0.5, 0.4,-0.2, 0.1},
    {-0.3, 0.0,-0.4,-0.5, 0.5, 0.2},
    { 0.5,-0.3, 0.0, 0.2,-0.3, 0.3},
    { 0.4,-0.5, 0.3, 0.0,-0.3, 0.4},
    {-0.2, 0.4,-0.2,-0.3, 0.0, 0.3},
    { 0.1, 0.2, 0.3, 0.4, 0.3, 0.0}
};

typedef struct {
    float act[N_CH]; float soma[N_CH];
    float debt, trauma, scar, presence;
} Klaus;

static void klaus_init(Klaus *k){
    memset(k,0,sizeof(*k)); k->act[CH_LOVE]=0.2f; k->act[CH_FLOW]=0.15f;
}

static void klaus_feel_root(Klaus *k, int root_id) {
    /* Map root to chamber via lexicon family */
    int n_known = sizeof(ROOT_CHAMBER)/sizeof(ROOT_CHAMBER[0]);
    if(root_id >= 0 && root_id < n_known) {
        int ch = ROOT_CHAMBER[root_id];
        if(ch >= 0 && ch < N_CH) k->act[ch] += 0.08f;
    }
}

static void klaus_xfire(Klaus *k, int iters) {
    for(int it=0;it<iters;it++){
        float old[N_CH]; for(int i=0;i<N_CH;i++) old[i]=k->act[i];
        for(int i=0;i<N_CH;i++){
            k->act[i]*=CH_D[i];
            for(int j=0;j<N_CH;j++) if(i!=j) k->act[i]+=0.03f*COU[i][j]*sinf(old[j]-old[i]);
            k->act[i]=cf(k->act[i],0,1);
            k->soma[i]=cf(0.94f*k->soma[i]+0.02f*k->act[i],0,1);
        }
        k->presence=cf(0.95f*k->presence+0.03f*((1.0f-fmaxf(k->act[CH_VOID],0.1f))*fminf(k->act[CH_FLOW],0.95f)),0,1);
        k->scar=cf(k->scar*0.985f,0,1);
    }
}

static void klaus_modulate(Klaus *k, float *c_heb, float *c_pro, float *c_bi, float *c_tri) {
    /* Chambers modulate metaweight coefficients — same idea as Q */
    *c_heb *= cf(1.0f+0.4f*k->act[CH_LOVE]-0.2f*k->act[CH_RAGE]+0.3f*k->act[CH_FLOW],0.3f,2.0f);
    *c_pro *= cf(1.0f+0.4f*k->act[CH_FLOW]-0.2f*k->act[CH_FEAR],0.3f,2.0f);
    *c_bi *= cf(1.0f+0.5f*k->act[CH_CMPLX]+0.2f*k->act[CH_LOVE]-0.1f*k->act[CH_VOID],0.3f,2.0f);
    *c_tri *= cf(1.0f-0.2f*k->act[CH_FLOW]+0.1f*k->act[CH_FEAR],0.3f,2.0f);
}

static int klaus_dominant(Klaus *k){
    int d=0;for(int i=1;i<N_CH;i++) if(k->act[i]>k->act[d]) d=i; return d;
}

/* ═══════════════════════════════════════════════════════════════════
 * CALENDAR DRIFT — α
 * ═══════════════════════════════════════════════════════════════════ */

static float calendar_dissonance(void) {
    /* Hebrew-Gregorian drift: 11.25 days/year with Metonic correction */
    time_t now = time(NULL);
    double epoch = 1727956800.0; /* ~Oct 3, 2024 = 1 Tishrei 5785 */
    double days = (double)(now - (time_t)epoch) / 86400.0;
    double y = days / 365.25;
    double drift = y * 11.25;
    int full = (int)(y / 19.0);
    double corr = full * 7.0 * 30.0;
    double partial = fmod(y, 19.0);
    int yic = (int)partial + 1;
    int met[] = {3,6,8,11,14,17,19};
    for(int i=0;i<7;i++) if(met[i]<=yic) corr+=30;
    drift -= corr;
    return cf(fabsf(fmod(drift,33.0))/33.0f, 0, 1);
}

/* ═══════════════════════════════════════════════════════════════════
 * WORD REALIZATION (Level 6)
 * ═══════════════════════════════════════════════════════════════════ */

static float cbigram[HEB][HEB];

static void build_cbigrams(const RootEng *re) {
    memset(cbigram,0,sizeof(cbigram)); float rt[HEB]={0};
    for(int w=0;w<re->nw;w++){const HWord *hw=&re->words[w];
        for(int i=0;i<hw->nlet-1;i++){int a=hw->letters[i],b=hw->letters[i+1];
            if(a>=0&&a<HEB&&b>=0&&b<HEB){cbigram[a][b]+=(float)hw->count;rt[a]+=(float)hw->count;}}}
    for(int a=0;a<HEB;a++) if(rt[a]>0) for(int b=0;b<HEB;b++) cbigram[a][b]/=rt[a];
}

static int g_meta_nw = 0; /* set in main: word count from meta corpus only */

static int realize(const RootEng *re, int rid, const int *prev_let, int pn, float ord_m) {
    if(rid<0||rid>=re->nr||re->rwc[rid]==0) return -1;
    float best=-1e30f; int bw=-1;
    /* First pass: find best among meta words only */
    for(int ci=0;ci<re->rwc[rid];ci++){
        int wi=re->rwords[rid][ci];
        if(g_meta_nw>0 && wi>=g_meta_nw) continue; /* skip non-meta words */
        const HWord *hw=&re->words[wi]; float sc=0;
        sc+=0.4f*logf(1.0f+(float)hw->count);
        if(pn>0&&hw->nlet>0){int lc=prev_let[pn-1],fc=hw->letters[0];
            if(lc>=0&&lc<HEB&&fc>=0&&fc<HEB) sc+=0.6f*cbigram[lc][fc];}
        float in=0;for(int i=0;i<hw->nlet-1;i++){int a=hw->letters[i],b=hw->letters[i+1];
            if(a>=0&&a<HEB&&b>=0&&b<HEB)in+=cbigram[a][b];}
        if(hw->nlet>1) sc+=0.3f*in/(float)(hw->nlet-1);
        if(hw->nlet>=3&&hw->nlet<=6) sc+=0.2f;
        if(sc>best){best=sc;bw=wi;}
    }
    /* Fallback: if no meta word found, use any word */
    if(bw<0) bw=re->rwords[rid][0];
    return bw;
}

/* ═══════════════════════════════════════════════════════════════════
 * ROOT SELECTION — θ = ε + γ + αδ
 * ═══════════════════════════════════════════════════════════════════ */

static int select_root(TF *tf, RMeta *m, RootEng *re, Klaus *kl,
                        const int *ctx, int n, float ord_m, float cd) {
    int NR = re->nr;
    float *score = calloc(NR, sizeof(float));
    float *heb = calloc(NR, sizeof(float));
    float *pro = calloc(NR, sizeof(float));

    rm_q_hebb(m, ctx, n, heb, NR);
    rm_q_proph(m, ctx, n, pro, NR);

    int prev1=(n>=1)?ctx[n-1]:-1, prev2=(n>=2)?ctx[n-2]:-1;

    /* Base coefficients */
    float c_uni=0.15f, c_bi=2.5f, c_tri=3.5f, c_heb=0.8f, c_pro=0.6f, c_ord=0.7f;

    /* Klaus chamber modulation — δ */
    klaus_modulate(kl, &c_heb, &c_pro, &c_bi, &c_tri);

    /* Calendar drift modulation — α */
    c_pro *= (1.0f + 0.3f * cd);
    c_heb *= (1.0f + 0.2f * cd);

    /* ε — transformer logits (gated) */
    if(n>0 && prev1>=0 && TOK_ROOT_BASE+prev1<tf->V) {
        tf_forward(tf, TOK_ROOT_BASE+prev1, (n-1) < CTX-1 ? n-1 : CTX-1);
    }

    for(int i=0;i<NR;i++){
        float bi = (prev1>=0) ? rm_bi(m,prev1,i) : 1e-10f;
        float tri = (prev2>=0&&prev1>=0) ? rm_tri(m,prev2,prev1,i) : 1e-10f;
        float ord_h = 1.0f/(1.0f+4.0f*fabsf(
            (float)(ORD[re->roots[i].c[0]]+ORD[re->roots[i].c[1]]+ORD[re->roots[i].c[2]])/66.0f
            - ord_m/66.0f));

        /* γ — metaweight field */
        score[i] = c_uni*m->uni[i] + c_bi*bi + c_tri*tri
                 + c_heb*heb[i] + c_pro*pro[i] + c_ord*ord_h;

        /* ε — transformer contribution (gated: 0 when untrained) */
        if(TOK_ROOT_BASE+i < tf->V) score[i] += tf->logits[TOK_ROOT_BASE+i];

        if(re->rwc[i]==0) score[i]=-1e9f;
    }

    /* Repetition penalty */
    int rs=n-6;if(rs<0)rs=0;
    for(int j=rs;j<n;j++) if(ctx[j]>=0&&ctx[j]<NR) score[ctx[j]]-=3.0f;

    /* Top-K sampling */
    int ti[TOP_K]; float tv[TOP_K];
    for(int k=0;k<TOP_K;k++){ti[k]=0;tv[k]=-1e30f;}
    for(int i=0;i<NR;i++){
        if(score[i]>tv[TOP_K-1]){tv[TOP_K-1]=score[i];ti[TOP_K-1]=i;
            for(int k=TOP_K-2;k>=0;k--){if(tv[k+1]>tv[k]){
                float tmp=tv[k];tv[k]=tv[k+1];tv[k+1]=tmp;
                int tt=ti[k];ti[k]=ti[k+1];ti[k+1]=tt;}}}}

    float mx=tv[0]; float pr[TOP_K]; float sum=0;
    for(int k=0;k<TOP_K;k++){pr[k]=expf((tv[k]-mx)/TEMP);sum+=pr[k];}
    float r=randf()*sum; float cum=0; int chosen=ti[0];
    for(int k=0;k<TOP_K;k++){cum+=pr[k];if(cum>=r){chosen=ti[k];break;}}

    free(score);free(heb);free(pro);
    return chosen;
}

/* ═══════════════════════════════════════════════════════════════════
 * 12-STEP CHAIN + GENERATION
 * ═══════════════════════════════════════════════════════════════════ */

static void generate_chain(RootEng *re, RMeta *m, TF *tf, Klaus *kl,
                            const char *prompt, int n_words) {
    float cd = calendar_dissonance();
    int nb = (int)(CHAIN * (0.3f + 0.4f*kl->debt + 0.1f*cd));
    if(nb<1) nb=1; if(nb>=CHAIN) nb=CHAIN-1;

    int pwi[64], pri[64];
    int pn = re_extract(re, prompt, pwi, pri, 64);

    printf("  drift=%.3f debt=%.3f chambers: ", cd, kl->debt);
    for(int i=0;i<N_CH;i++) if(kl->act[i]>0.05f) printf("%s:%.0f%% ",CH_N[i],kl->act[i]*100);
    printf("\n\n");

    /* Seed context with prompt roots */
    int ctx[256]; int cn=0;
    for(int i=0;i<pn&&cn<256;i++) if(pri[i]>=0) ctx[cn++]=pri[i];

    float ord_sum=0; int ord_cnt=0;
    for(int i=0;i<cn;i++) if(ctx[i]>=0&&ctx[i]<re->nr){
        ord_sum+=(float)(ORD[re->roots[ctx[i]].c[0]]+ORD[re->roots[ctx[i]].c[1]]+ORD[re->roots[ctx[i]].c[2]]);
        ord_cnt++;}
    float ord_m = ord_cnt>0 ? ord_sum/(float)ord_cnt : 33.0f;

    int prev_let[32]={0}; int prev_nl=0;
    if(pn>0&&pwi[pn-1]>=0){
        memcpy(prev_let,re->words[pwi[pn-1]].letters,re->words[pwi[pn-1]].nlet*sizeof(int));
        prev_nl=re->words[pwi[pn-1]].nlet;}

    /* Print prompt info */
    printf("  Prompt: %s\n", prompt);
    printf("  Roots: ");
    for(int i=0;i<pn;i++) if(pri[i]>=0){Root *r=&re->roots[pri[i]];
        printf("%s.%s.%s ",LET[r->c[0]],LET[r->c[1]],LET[r->c[2]]);}
    printf("\n  Gen:    ");

    tf_reset(tf);

    for(int step=0; step<n_words && cn<250; step++) {
        /* Klaus phase pressure */
        float d_phase = (float)step/(float)n_words;
        if(d_phase<0.33f) kl->act[CH_FLOW]=cf(kl->act[CH_FLOW]+0.03f,0,1);
        else if(d_phase<0.66f) kl->act[CH_FEAR]=cf(kl->act[CH_FEAR]+0.02f,0,1);
        else kl->act[CH_VOID]=cf(kl->act[CH_VOID]+0.03f,0,1);

        int rid = select_root(tf, m, re, kl, ctx, cn, ord_m, cd);
        if(rid<0) break;

        int wi = realize(re, rid, prev_let, prev_nl, ord_m);

        if(wi>=0){
            printf("%s ", re->words[wi].utf8);
            ctx[cn++]=rid;
            ord_m = 0.8f*ord_m + 0.2f*(float)(ORD[re->roots[rid].c[0]]+ORD[re->roots[rid].c[1]]+ORD[re->roots[rid].c[2]]);
            proph_update(m, rid);

            /* Prophecy from bigram prediction */
            float bp=0; int bpred=-1;
            for(int k=0;k<m->nbi;k++) if(m->bi[k].a==rid&&m->bi[k].p>bp){bp=m->bi[k].p;bpred=m->bi[k].b;}
            if(bpred>=0) proph_add(m, bpred, 0.3f+0.5f*bp);

            /* Klaus feels the root */
            klaus_feel_root(kl, rid);

            memcpy(prev_let,re->words[wi].letters,re->words[wi].nlet*sizeof(int));
            prev_nl=re->words[wi].nlet;
        } else {
            Root *r=&re->roots[rid];
            printf("[%s%s%s] ",LET[r->c[0]],LET[r->c[1]],LET[r->c[2]]);
            ctx[cn++]=rid;
        }
        fflush(stdout);

        klaus_xfire(kl, 2);
        kl->debt = 0.9f*kl->debt + 0.05f;
    }
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════
 * WEIGHT I/O
 * ═══════════════════════════════════════════════════════════════════ */

static int tf_load(TF *t, const char *path) {
    FILE *f = fopen(path, "rb");
    if(!f) return 0;
    unsigned int magic; fread(&magic,4,1,f);
    if(magic != SHORESH_MAGIC){fclose(f);return 0;}
    /* Read all weight tensors */
    fread(t->tok, sizeof(float), SBPE_VOCAB*DIM, f);
    fread(t->pos, sizeof(float), CTX*DIM, f);
    for(int l=0;l<N_LAYERS;l++){
        TFLayer *ly=&t->L[l];
        fread(ly->wq,sizeof(float),N_CONTENT*HD*DIM,f);
        fread(ly->wk,sizeof(float),N_CONTENT*HD*DIM,f);
        fread(ly->vc,sizeof(float),N_CONTENT*HD*DIM,f);
        fread(ly->wr,sizeof(float),N_RRPRAM*DIM*CTX,f);
        fread(ly->vr,sizeof(float),N_RRPRAM*HD*DIM,f);
        fread(ly->wj,sizeof(float),N_JANUS*HD*DIM,f);
        fread(ly->vj,sizeof(float),N_JANUS*HD*DIM,f);
        fread(ly->wo,sizeof(float),DIM*DIM,f);
        fread(ly->w1,sizeof(float),FFN*DIM,f);
        fread(ly->w2,sizeof(float),DIM*FFN,f);
        fread(ly->wg,sizeof(float),3*DIM,f);
        fread(ly->gb,sizeof(float),3,f);
    }
    fclose(f);
    t->loaded = 1;
    printf("  ε loaded: Janus triple attention weights from %s\n", path);
    return 1;
}

static void tf_save(TF *t, const char *path) {
    FILE *f=fopen(path,"wb"); if(!f) return;
    unsigned int magic=SHORESH_MAGIC; fwrite(&magic,4,1,f);
    fwrite(t->tok,sizeof(float),SBPE_VOCAB*DIM,f);
    fwrite(t->pos,sizeof(float),CTX*DIM,f);
    for(int l=0;l<N_LAYERS;l++){
        TFLayer *ly=&t->L[l];
        fwrite(ly->wq,sizeof(float),N_CONTENT*HD*DIM,f);
        fwrite(ly->wk,sizeof(float),N_CONTENT*HD*DIM,f);
        fwrite(ly->vc,sizeof(float),N_CONTENT*HD*DIM,f);
        fwrite(ly->wr,sizeof(float),N_RRPRAM*DIM*CTX,f);
        fwrite(ly->vr,sizeof(float),N_RRPRAM*HD*DIM,f);
        fwrite(ly->wj,sizeof(float),N_JANUS*HD*DIM,f);
        fwrite(ly->vj,sizeof(float),N_JANUS*HD*DIM,f);
        fwrite(ly->wo,sizeof(float),DIM*DIM,f);
        fwrite(ly->w1,sizeof(float),FFN*DIM,f);
        fwrite(ly->w2,sizeof(float),DIM*FFN,f);
        fwrite(ly->wg,sizeof(float),3*DIM,f);
        fwrite(ly->gb,sizeof(float),3,f);
    }
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════
 * SEMANTIC BPE TOKENIZER (for training corpus)
 *
 * Frequent root → single token (TOK_ROOT_BASE + root_id)
 * Rare root → ROOT_START + char tokens + ROOT_END
 * Function prefix → prefix token
 * Space → TOK_SPACE
 * V = SBPE_VOCAB = 655
 * ═══════════════════════════════════════════════════════════════════ */

static const int SBPE_F1[]={4,1,10,11,12,20,5,13,9,21,0};
static const int SBPE_F2[][2]={{4,21},{20,11},{5,4},{12,4}};

static int sbpe_strip_pfx(const int *l, int n, int *start) {
    *start=0;
    if(n>=4){for(int i=0;i<4;i++)if(l[0]==SBPE_F2[i][0]&&l[1]==SBPE_F2[i][1]&&n-2>=2)
        {*start=2;return TOK_FUNC_BASE+11+i;}}
    if(n>=3){for(int i=0;i<11;i++)if(l[0]==SBPE_F1[i]&&n-1>=2)
        {*start=1;return TOK_FUNC_BASE+i;}}
    return -1;
}

static int sbpe_tok_word(RootEng *re, const char *utf8, int *out, int mx) {
    int let[32], nl=0;
    const unsigned char *q=(const unsigned char*)utf8;
    while(*q){int a;int l=u8let(q,&a);if(l>=0&&nl<32)let[nl++]=l;q+=a;}
    if(nl<1) return 0;
    int n=0, ss=0;
    int pt = sbpe_strip_pfx(let, nl, &ss);
    if(pt>=0 && n<mx) out[n++]=pt;
    int *stem=let+ss; int sn=nl-ss;
    if(sn<2){for(int i=ss;i<nl&&n<mx;i++)out[n++]=let[i];return n;}
    /* Subsequence root match */
    int best=-1, bs=999, bsp=999;
    for(int ri=0;ri<re->nr;ri++){
        Root *r=&re->roots[ri]; int t[3]={r->c[0],r->c[1],r->c[2]};
        int pos[3]={-1,-1,-1}; int ti=0;
        for(int i=0;i<sn&&ti<3;i++) if(stem[i]==t[ti]){pos[ti]=i;ti++;}
        if(ti==3){int s=pos[0],sp=pos[2]-pos[0];
            if(s<bs||(s==bs&&sp<bsp)){bs=s;bsp=sp;best=ri;}}
    }
    if(best>=0){
        if(n<mx) out[n++]=TOK_ROOT_BASE+best;
    } else {
        if(n+sn+2<=mx){out[n++]=TOK_RSTART;
            int e=0;for(int i=0;i<sn&&e<3&&n<mx;i++){out[n++]=stem[i];e++;}
            if(n<mx) out[n++]=TOK_REND;}
    }
    return n;
}

static int sbpe_tokenize(RootEng *re, const char *text, int *out, int mx) {
    const unsigned char *p=(const unsigned char*)text;
    int n=0; char wb[128]; int wp=0;
    while(*p && n<mx-6){
        if(isheb(p)){if(wp<126){wb[wp++]=p[0];wb[wp++]=p[1];}p+=2;}
        else{if(wp>0){wb[wp]=0;n+=sbpe_tok_word(re,wb,out+n,mx-n);wp=0;}
            if(*p==' '||*p=='\n'||*p=='\r'){if(n>0&&out[n-1]!=TOK_SPACE)out[n++]=TOK_SPACE;}p++;}
    }
    if(wp>0){wb[wp]=0;n+=sbpe_tok_word(re,wb,out+n,mx-n);}
    return n;
}

/* ═══════════════════════════════════════════════════════════════════
 * TRAINING MODE (optional, requires notorch)
 * cc shoresh.c notorch.c -O2 -lm -DSHORESH_TRAIN -DUSE_BLAS -DACCELERATE -framework Accelerate -o shoresh_train
 * ═══════════════════════════════════════════════════════════════════ */

#ifdef SHORESH_TRAIN

typedef struct {
    nt_tensor *wq, *wk, *vc;
    nt_tensor *wr, *vr;
    nt_tensor *wj, *vj;
    nt_tensor *wo_c, *wo_r, *wo_j;
    nt_tensor *w1, *w2;
    nt_tensor *wg, *gb;
} TParams_L;

typedef struct {
    nt_tensor *tok, *pos;
    TParams_L L[N_LAYERS];
} TParams;

static TParams tp_init(int V) {
    TParams p;
    p.tok = nt_tensor_new2d(V, DIM); nt_tensor_xavier(p.tok, DIM, V); /* V = SBPE_VOCAB */
    p.pos = nt_tensor_new2d(CTX, DIM); nt_tensor_xavier(p.pos, CTX, DIM);
    for(int l=0;l<N_LAYERS;l++){
        TParams_L *ly = &p.L[l];
        ly->wq = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(ly->wq, DIM, N_CONTENT*HD);
        ly->wk = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(ly->wk, DIM, N_CONTENT*HD);
        ly->vc = nt_tensor_new2d(N_CONTENT*HD, DIM); nt_tensor_xavier(ly->vc, DIM, N_CONTENT*HD);
        ly->wr = nt_tensor_new2d(N_RRPRAM*HD, DIM); nt_tensor_xavier(ly->wr, DIM, N_RRPRAM*HD); /* proxy: self-attn, not positional routing */
        ly->vr = nt_tensor_new2d(N_RRPRAM*HD, DIM);  nt_tensor_xavier(ly->vr, DIM, N_RRPRAM*HD);
        ly->wj = nt_tensor_new2d(N_JANUS*HD, DIM);   nt_tensor_xavier(ly->wj, DIM, N_JANUS*HD);
        ly->vj = nt_tensor_new2d(N_JANUS*HD, DIM);   nt_tensor_xavier(ly->vj, DIM, N_JANUS*HD);
        ly->wo_c = nt_tensor_new2d(DIM, N_CONTENT*HD); nt_tensor_xavier(ly->wo_c, N_CONTENT*HD, DIM);
        ly->wo_r = nt_tensor_new2d(DIM, N_RRPRAM*HD);  nt_tensor_xavier(ly->wo_r, N_RRPRAM*HD, DIM);
        ly->wo_j = nt_tensor_new2d(DIM, N_JANUS*HD);   nt_tensor_xavier(ly->wo_j, N_JANUS*HD, DIM);
        ly->w1 = nt_tensor_new2d(FFN, DIM);  nt_tensor_xavier(ly->w1, DIM, FFN);
        ly->w2 = nt_tensor_new2d(DIM, FFN);  nt_tensor_xavier(ly->w2, FFN, DIM);
        ly->wg = nt_tensor_new2d(3, DIM);    nt_tensor_xavier(ly->wg, DIM, 3);
        ly->gb = nt_tensor_new(3);            nt_tensor_fill(ly->gb, 0.0f);
    }
    return p;
}

typedef struct { int wq,wk,vc,wr,vr,wj,vj,wo_c,wo_r,wo_j,w1,w2,wg,gb; } TI_L;
typedef struct { int tok, pos; TI_L L[N_LAYERS]; } TI;

static TI tp_reg(TParams *p) {
    TI t;
    t.tok = nt_tape_param(p->tok); nt_tape_no_decay(t.tok);
    t.pos = nt_tape_param(p->pos); nt_tape_no_decay(t.pos);
    for(int l=0;l<N_LAYERS;l++){
        TParams_L *ly = &p->L[l];
        t.L[l].wq=nt_tape_param(ly->wq); t.L[l].wk=nt_tape_param(ly->wk);
        t.L[l].vc=nt_tape_param(ly->vc); t.L[l].wr=nt_tape_param(ly->wr);
        t.L[l].vr=nt_tape_param(ly->vr); t.L[l].wj=nt_tape_param(ly->wj);
        t.L[l].vj=nt_tape_param(ly->vj);
        t.L[l].wo_c=nt_tape_param(ly->wo_c); t.L[l].wo_r=nt_tape_param(ly->wo_r);
        t.L[l].wo_j=nt_tape_param(ly->wo_j);
        t.L[l].w1=nt_tape_param(ly->w1); t.L[l].w2=nt_tape_param(ly->w2);
        t.L[l].wg=nt_tape_param(ly->wg); t.L[l].gb=nt_tape_param(ly->gb);
    }
    return t;
}

static int forward_train(TI *ti, int *tokens, int *targets, int T, int V) {
    nt_tensor *tok_t = nt_tensor_new(T);
    for(int i=0;i<T;i++) tok_t->data[i] = (float)tokens[i];
    int tok_idx = nt_tape_param(tok_t); nt_tensor_free(tok_t);
    nt_tape_no_decay(tok_idx);
    int h = nt_seq_embedding(ti->tok, ti->pos, tok_idx, T, DIM);
    for(int l=0;l<N_LAYERS;l++){
        int normed = nt_seq_rmsnorm(h, -1, T, DIM);
        int q_c = nt_seq_linear(ti->L[l].wq, normed, T);
        int k_c = nt_seq_linear(ti->L[l].wk, normed, T);
        int v_c = nt_seq_linear(ti->L[l].vc, normed, T);
        int attn_c = nt_mh_causal_attention(q_c, k_c, v_c, T, HD);
        int q_r = nt_seq_linear(ti->L[l].wr, normed, T);
        int v_r = nt_seq_linear(ti->L[l].vr, normed, T);
        int attn_r = nt_mh_causal_attention(q_r, q_r, v_r, T, HD);
        int proj_j = nt_seq_linear(ti->L[l].wj, normed, T);
        int echo_j = nt_seq_linear(ti->L[l].vj, normed, T);
        int janus = nt_mul(proj_j, echo_j);
        int pc = nt_seq_linear(ti->L[l].wo_c, attn_c, T);
        int pr = nt_seq_linear(ti->L[l].wo_r, attn_r, T);
        int pj = nt_seq_linear(ti->L[l].wo_j, janus, T);
        h = nt_add(h, nt_add(nt_add(pc, pr), pj));
        normed = nt_seq_rmsnorm(h, -1, T, DIM);
        int ff = nt_seq_linear(ti->L[l].w1, normed, T);
        ff = nt_gelu(ff);
        ff = nt_seq_linear(ti->L[l].w2, ff, T);
        h = nt_add(h, ff);
    }
    int fn = nt_seq_rmsnorm(h, -1, T, DIM);
    int logits = nt_seq_linear(ti->tok, fn, T);
    nt_tensor *tgt = nt_tensor_new(T);
    for(int i=0;i<T;i++) tgt->data[i] = (float)targets[i];
    int tgt_idx = nt_tape_param(tgt); nt_tensor_free(tgt);
    nt_tape_no_decay(tgt_idx);
    return nt_seq_cross_entropy(logits, tgt_idx, T, V);
}

static void tp_save(TParams *p, const char *path, int V) {
    FILE *f=fopen(path,"wb"); if(!f) return;
    unsigned magic=SHORESH_MAGIC; fwrite(&magic,4,1,f);
    fwrite(p->tok->data,4,V*DIM,f);
    fwrite(p->pos->data,4,CTX*DIM,f);
    for(int l=0;l<N_LAYERS;l++){
        TParams_L *ly=&p->L[l];
        fwrite(ly->wq->data,4,N_CONTENT*HD*DIM,f);
        fwrite(ly->wk->data,4,N_CONTENT*HD*DIM,f);
        fwrite(ly->vc->data,4,N_CONTENT*HD*DIM,f);
        /* wr: inference expects [N_RRPRAM*DIM, CTX], training proxy is [N_RRPRAM*HD, DIM].
         * Write zeros — inference gate will suppress untrained RRPRAM. */
        {float zero=0; for(int i=0;i<N_RRPRAM*DIM*CTX;i++) fwrite(&zero,4,1,f);}
        fwrite(ly->vr->data,4,N_RRPRAM*HD*DIM,f);
        fwrite(ly->wj->data,4,N_JANUS*HD*DIM,f);
        fwrite(ly->vj->data,4,N_JANUS*HD*DIM,f);
        for(int r=0;r<DIM;r++){
            fwrite(ly->wo_c->data+r*N_CONTENT*HD,4,N_CONTENT*HD,f);
            fwrite(ly->wo_r->data+r*N_RRPRAM*HD,4,N_RRPRAM*HD,f);
            fwrite(ly->wo_j->data+r*N_JANUS*HD,4,N_JANUS*HD,f);
        }
        fwrite(ly->w1->data,4,FFN*DIM,f);
        fwrite(ly->w2->data,4,DIM*FFN,f);
        fwrite(ly->wg->data,4,3*DIM,f);
        fwrite(ly->gb->data,4,3,f);
    }
    fclose(f); printf("  Saved %s\n",path);
}

static void shoresh_train(TParams *p, int *corpus, int n_tok, int V,
                          int steps, float lr, const char *save_path) {
    printf("\n═══ TRAINING ε ═══\n");
    printf("  %d root tokens, V=%d, steps=%d, lr=%.1e\n", n_tok, V, steps, lr);
    nt_train_mode(1);
    nt_schedule sched = nt_schedule_cosine(lr, steps/10, steps, lr*0.1f);
    nt_nan_guard guard = nt_nan_guard_new();
    float best_loss=999, loss_ema=0;
    clock_t t0 = clock();
    for(int step=0;step<steps;step++){
        float clr = nt_schedule_get_lr(&sched);
        int start = rand()%(n_tok-CTX-1);
        nt_tape_start();
        TI ti = tp_reg(p);
        int loss_idx = forward_train(&ti, corpus+start, corpus+start+1, CTX, V);
        float lv = nt_tape_get()->entries[loss_idx].output->data[0];
        if(step==0) loss_ema=lv;
        loss_ema = 0.99f*loss_ema + 0.01f*lv;
        if(lv<best_loss) best_loss=lv;
        nt_tape_backward(loss_idx);
        if(!nt_nan_guard_check(&guard)){nt_tape_clear();continue;}
        nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(clr, lv);
        nt_tape_clear();
        if(step%100==0){
            double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("  step %5d | train %.4f | ema %.4f | best %.4f | lr %.2e | %.1fs\n",
                   step, lv, loss_ema, best_loss, clr, el);
        }
        if(step>0 && step%1000==0) tp_save(p, save_path, V);
    }
    double total=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("  Done: ema %.4f, best %.4f, %.1fs (%.1f steps/s)\n",
           loss_ema, best_loss, total, (double)steps/total);
    tp_save(p, save_path, V);
}

#endif /* SHORESH_TRAIN */

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

static char *readf(const char *p, long *sz) {
    FILE *f=fopen(p,"rb"); if(!f)return NULL;
    fseek(f,0,SEEK_END);*sz=ftell(f);fseek(f,0,SEEK_SET);
    if(*sz<=0||*sz>MAX_TEXT){fclose(f);return NULL;}
    char *b=malloc(*sz+1);fread(b,1,*sz,f);fclose(f);b[*sz]=0;return b;
}

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));

    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  SHORESH שורש v2 — Hebrew Root Resonance Engine   ║\n");
    printf("║  θ = ε + γ + αδ                                   ║\n");
    printf("║  ε=Janus γ=MetaWeights α=Calendar δ=Klaus         ║\n");
    printf("╚════════════════════════════════════════════════════╝\n\n");

    if(argc<2){
        printf("Usage: %s [-w weights.bin] corpus.txt [prompt]\n",argv[0]);
        return 1;
    }

const char *wpath=NULL, *cpath=NULL, *mpath=NULL, *prompt=NULL;
#ifdef SHORESH_TRAIN
    int do_train=0; int train_steps=5000; const char *save_path="shoresh.bin";
    const char *train_corpus_path=NULL;
#endif
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"-w")==0&&i+1<argc){wpath=argv[++i];}
        else if(strcmp(argv[i],"-m")==0&&i+1<argc){mpath=argv[++i];}
#ifdef SHORESH_TRAIN
        else if(strcmp(argv[i],"--train")==0&&i+1<argc){do_train=1;train_corpus_path=argv[++i];}
        else if(strcmp(argv[i],"--steps")==0&&i+1<argc){train_steps=atoi(argv[++i]);}
        else if(strcmp(argv[i],"--save")==0&&i+1<argc){save_path=argv[++i];}
#endif
        else if(!cpath){cpath=argv[i];}
        else if(!prompt){prompt=argv[i];}
    }
    if(!cpath){printf("ERROR: corpus path required\n");return 1;}
    if(!prompt) prompt="בראשית";

    /* [1] Meta corpus (cpath) — extracted FIRST for stable root IDs */
    printf("[1] Meta corpus: %s\n",cpath);
    long csz; char *text=readf(cpath,&csz);
    if(!text){fprintf(stderr,"Cannot read %s\n",cpath);return 1;}
    printf("  %ld bytes\n",csz);

    /* [2] Root Engine — seed from meta corpus FIRST (stable IDs for emergence) */
    printf("[2] Root Engine...\n");
    RootEng *re = calloc(1,sizeof(RootEng));
    re_init(re); re_load_lex(re);
    printf("  Lexicon: %d roots\n",re->nr);

    /* Extract meta corpus → roots get stable IDs 0..N_meta */
    int *meta_cwi=malloc(MAX_CWORDS*sizeof(int)), *meta_cri=malloc(MAX_CWORDS*sizeof(int));
    int meta_cn = re_extract(re, text, meta_cwi, meta_cri, MAX_CWORDS);
    int n_meta_roots = re->nr;
    int n_meta_words = re->nw;  /* save for realize() scope */
    g_meta_nw = n_meta_words;
    /* Freeze meta word counts before Ben-Yehuda changes them */
    int *frozen_counts = NULL;
    if(mpath){
        frozen_counts = malloc(n_meta_words * sizeof(int));
        for(int i=0;i<n_meta_words;i++) frozen_counts[i] = re->words[i].count;
    }
    printf("  Meta: %d words, %d roots (stable IDs 0..%d)\n", meta_cn, n_meta_roots, n_meta_roots-1);

    /* [3] Char bigrams — from META corpus only (before Ben-Yehuda dilution) */
    printf("[3] Char bigrams (from meta)...\n");
    build_cbigrams(re);

    /* If -m: extract additional corpus → new roots append to slots N_meta..ROOT_LIMIT */
    int *cwi, *cri; int cn;
    if(mpath) {
        printf("  Additional: %s\n", mpath);
        long msz; char *mtext = readf(mpath, &msz);
        if(mtext) {
            printf("  %ld bytes\n", msz);
            int *add_cwi=malloc(MAX_CWORDS*sizeof(int)), *add_cri=malloc(MAX_CWORDS*sizeof(int));
            int add_cn = re_extract(re, mtext, add_cwi, add_cri, MAX_CWORDS);
            printf("  Additional: %d words, %d new roots (total %d)\n", add_cn, re->nr - n_meta_roots, re->nr);
            /* Merge: full corpus = meta + additional */
            cn = meta_cn + add_cn;
            cwi=malloc(cn*sizeof(int)); cri=malloc(cn*sizeof(int));
            memcpy(cwi, meta_cwi, meta_cn*sizeof(int));
            memcpy(cwi+meta_cn, add_cwi, add_cn*sizeof(int));
            memcpy(cri, meta_cri, meta_cn*sizeof(int));
            memcpy(cri+meta_cn, add_cri, add_cn*sizeof(int));
            free(add_cwi); free(add_cri); free(mtext);
        } else { cwi=meta_cwi; cri=meta_cri; cn=meta_cn; }
    } else {
        cwi=meta_cwi; cri=meta_cri; cn=meta_cn;
    }
    re->corpus_wids=cwi; re->corpus_rids=cri; re->ncr=re->ncw=cn;
    printf("  Total: %d words, %d unique, %d roots\n",cn,re->nw,re->nr);

    /* cbigrams already built from meta (step 3 above) */
    /* Restore frozen meta word counts (Ben-Yehuda may have incremented them) */
    if(frozen_counts){
        for(int i=0;i<n_meta_words;i++) re->words[i].count = frozen_counts[i];
        free(frozen_counts);
    }

    /* [4] Root MetaWeights — γ from META corpus only (stable IDs) */
    printf("[4] γ field (from %s, %d roots)...\n", cpath, n_meta_roots);
    RMeta *m=calloc(1,sizeof(RMeta));
    int *vr=malloc(meta_cn*sizeof(int)); int vn=0;
    for(int i=0;i<meta_cn;i++) if(meta_cri[i]>=0) vr[vn++]=meta_cri[i];
    rm_build(m,vr,vn,re->nr);
    printf("  γ: %d root tokens → %d bigram, %d trigram, %d hebbian\n",
           vn, m->nbi, m->ntri, m->nhebb);
    if(mpath){free(meta_cwi);free(meta_cri);}
#ifndef SHORESH_TRAIN
    free(vr);
#endif

    /* [5] Janus Triple Attention — ε */
    printf("[5] ε (Janus triple attention)...\n");
    TF *tf=calloc(1,sizeof(TF)); tf_init(tf,re->nr);
    long np=(long)SBPE_VOCAB*DIM+CTX*DIM+
        N_LAYERS*(3L*N_CONTENT*HD*DIM+N_RRPRAM*(DIM*CTX+HD*DIM)+2L*N_JANUS*HD*DIM+
        DIM*DIM+FFN*DIM+DIM*FFN+3*DIM+3);
    printf("  %ld params (~%.1fK)\n",np,np/1000.0);
    if(wpath && tf_load(tf,wpath)){
        printf("  [TRAINED MODE]\n");
    } else {
        printf("  [METAWEIGHTS ONLY — gate suppresses untrained ε]\n");
    }

#ifdef SHORESH_TRAIN
    if(do_train){
        /* Semantic BPE tokenize full corpus for training */
        char *full_text = text;
        if(mpath){
            long msz; char *mt=readf(mpath,&msz);
            if(mt){full_text=malloc(csz+msz+2);memcpy(full_text,text,csz);
                full_text[csz]='\n';memcpy(full_text+csz+1,mt,msz);full_text[csz+1+msz]=0;free(mt);}
        }
        int *train_toks = malloc(MAX_CWORDS*sizeof(int));
        int train_n = sbpe_tokenize(re, full_text, train_toks, MAX_CWORDS);
        if(full_text!=text) free(full_text);
        /* Count root tokens */
        int n_rt=0;for(int i=0;i<train_n;i++)if(train_toks[i]>=TOK_ROOT_BASE)n_rt++;
        printf("\n[TRAIN] ε on Semantic BPE (V=%d, %d tokens, %d root (%.1f%%), γ from %d)...\n",
               SBPE_VOCAB, train_n, n_rt, 100.0f*n_rt/train_n, vn);
        TParams tp = tp_init(SBPE_VOCAB);
        nt_seed((unsigned)time(NULL));
        shoresh_train(&tp, train_toks, train_n, SBPE_VOCAB, train_steps, 3e-4f, save_path);
        free(train_toks);
        if(tf_load(tf, save_path)) printf("  Trained weights loaded for test.\n");
        nt_tape_destroy();
    }
    free(vr);
#endif

    /* [6] Klaus Chambers — δ */
    printf("[6] δ (Klaus chambers)...\n");
    Klaus *kl=calloc(1,sizeof(Klaus)); klaus_init(kl);

    /* [7] Generate */
    float cd = calendar_dissonance();
    printf("\n[7] Generation (drift=%.3f)\n\n",cd);
    generate_chain(re, m, tf, kl, prompt, MAX_GEN);

    /* More prompts if no custom prompt */
    if(argc<3 || (argc==3 && strcmp(argv[1],"-w")!=0 && !prompt)){
        const char *tp[]={"שלום עולם","אהבה וחסד","האור והחושך","צדק ומשפט","חכמה ובינה",NULL};
        for(int i=0;tp[i];i++){printf("\n");generate_chain(re,m,tf,kl,tp[i],MAX_GEN);}
    }

    printf("\nהרזוננס לא נשבר. שורש.\n");
    free(text);free(re);free(m);free(tf);free(kl);
    return 0;
}

/* ═══ EOF ═══ */
