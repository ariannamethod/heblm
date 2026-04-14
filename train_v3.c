/*
 * train_v3.c — Train SHORESH v3 Semantic BPE on notorch
 *
 * Reads tokens.bin (from shoresh_v3 --dump), trains standard transformer.
 * Vocab=240, proven pitomadom forward pass.
 *
 * cc train_v3.c notorch.c -O2 -DUSE_BLAS -DACCELERATE \
 *    -framework Accelerate -lm -o train_v3
 *
 * ./shoresh_v3 -w --dump corpus.txt   # creates tokens.bin
 * ./train_v3 tokens.bin [steps] [lr]  # trains, saves shoresh_v3.bin
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#define DIM         160
#define N_LAYERS    4      /* 6 crashes on 8GB, 4 proven stable */
#define N_HEADS     8
#define HD          (DIM / N_HEADS)
#define CTX         96
#define FFN_DIM     640
#define MAX_VOCAB   240

#define DEFAULT_STEPS 5000
#define BASE_LR       3e-4f
#define VAL_SPLIT     0.1f

typedef struct {
    nt_tensor *wte, *wpe;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo, *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[N_LAYERS];
    nt_tensor *rms_f, *head;
} Model;

static Model *mcreate(int V) {
    Model *m = calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->wte, V, DIM);
    m->wpe = nt_tensor_new2d(CTX, DIM); nt_tensor_xavier(m->wpe, CTX, DIM);
    float sc = 0.02f / sqrtf(2.0f * N_LAYERS);
    for(int l=0; l<N_LAYERS; l++){
        m->L[l].rms1 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms1, 1.0f);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wq, DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wk, DIM, DIM);
        m->L[l].wv = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wv, DIM, DIM);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM); nt_tensor_xavier(m->L[l].wo, DIM, DIM);
        for(int i=0;i<m->L[l].wo->len;i++) m->L[l].wo->data[i]*=sc/0.1f;
        m->L[l].rms2 = nt_tensor_new(DIM); nt_tensor_fill(m->L[l].rms2, 1.0f);
        m->L[l].w_gate = nt_tensor_new2d(FFN_DIM, DIM); nt_tensor_xavier(m->L[l].w_gate, DIM, FFN_DIM);
        m->L[l].w_up = nt_tensor_new2d(FFN_DIM, DIM); nt_tensor_xavier(m->L[l].w_up, DIM, FFN_DIM);
        m->L[l].w_down = nt_tensor_new2d(DIM, FFN_DIM); nt_tensor_xavier(m->L[l].w_down, FFN_DIM, DIM);
        for(int i=0;i<m->L[l].w_down->len;i++) m->L[l].w_down->data[i]*=sc/0.1f;
    }
    m->rms_f = nt_tensor_new(DIM); nt_tensor_fill(m->rms_f, 1.0f);
    m->head = nt_tensor_new2d(V, DIM); nt_tensor_xavier(m->head, DIM, V);
    return m;
}

static long nparams(Model *m) {
    long n = m->wte->len + m->wpe->len + m->rms_f->len + m->head->len;
    for(int l=0; l<N_LAYERS; l++)
        n += m->L[l].rms1->len + m->L[l].wq->len + m->L[l].wk->len +
             m->L[l].wv->len + m->L[l].wo->len + m->L[l].rms2->len +
             m->L[l].w_gate->len + m->L[l].w_up->len + m->L[l].w_down->len;
    return n;
}

static int mfwd(Model *m, int *tokens, int *targets, int V) {
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int wpe_i = nt_tape_param(m->wpe); nt_tape_no_decay(wpe_i);
    int li[N_LAYERS][9];
    for(int l=0; l<N_LAYERS; l++){
        li[l][0]=nt_tape_param(m->L[l].rms1);
        li[l][1]=nt_tape_param(m->L[l].wq);  li[l][2]=nt_tape_param(m->L[l].wk);
        li[l][3]=nt_tape_param(m->L[l].wv);  li[l][4]=nt_tape_param(m->L[l].wo);
        li[l][5]=nt_tape_param(m->L[l].rms2);
        li[l][6]=nt_tape_param(m->L[l].w_gate);
        li[l][7]=nt_tape_param(m->L[l].w_up);
        li[l][8]=nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i=nt_tape_param(m->rms_f);
    int head_i=nt_tape_param(m->head);

    nt_tensor *tt=nt_tensor_new(CTX),*tg=nt_tensor_new(CTX);
    for(int i=0;i<CTX;i++){tt->data[i]=(float)tokens[i];tg->data[i]=(float)targets[i];}
    int tok_i=nt_tape_record(tt,NT_OP_NONE,-1,-1,0);
    int tgt_i=nt_tape_record(tg,NT_OP_NONE,-1,-1,0);
    nt_tensor_free(tt);nt_tensor_free(tg);

    int h=nt_seq_embedding(wte_i,wpe_i,tok_i,CTX,DIM);
    for(int l=0;l<N_LAYERS;l++){
        int xn=nt_seq_rmsnorm(h,li[l][0],CTX,DIM);
        int q=nt_seq_linear(li[l][1],xn,CTX);
        int k=nt_seq_linear(li[l][2],xn,CTX);
        int v=nt_seq_linear(li[l][3],xn,CTX);
        int attn=nt_mh_causal_attention(q,k,v,CTX,HD);
        int proj=nt_seq_linear(li[l][4],attn,CTX);
        h=nt_add(h,proj);
        xn=nt_seq_rmsnorm(h,li[l][5],CTX,DIM);
        int gate=nt_seq_linear(li[l][6],xn,CTX);
        int up=nt_seq_linear(li[l][7],xn,CTX);
        gate=nt_silu(gate);
        int ffn_h=nt_mul(gate,up);
        int down=nt_seq_linear(li[l][8],ffn_h,CTX);
        h=nt_add(h,down);
    }
    int hf=nt_seq_rmsnorm(h,rmsf_i,CTX,DIM);
    int logits=nt_seq_linear(head_i,hf,CTX);
    return nt_seq_cross_entropy(logits,tgt_i,CTX,V);
}

static void msave(Model *m, const char *path) {
    int n=2+N_LAYERS*9+2;
    nt_tensor **p=malloc(n*sizeof(nt_tensor*));
    int idx=0;
    p[idx++]=m->wte;p[idx++]=m->wpe;
    for(int l=0;l<N_LAYERS;l++){
        p[idx++]=m->L[l].rms1;
        p[idx++]=m->L[l].wq;p[idx++]=m->L[l].wk;
        p[idx++]=m->L[l].wv;p[idx++]=m->L[l].wo;
        p[idx++]=m->L[l].rms2;
        p[idx++]=m->L[l].w_gate;p[idx++]=m->L[l].w_up;p[idx++]=m->L[l].w_down;
    }
    p[idx++]=m->rms_f;p[idx++]=m->head;
    nt_save(path,p,idx);
    free(p);
    printf("  saved: %s (%d tensors)\n",path,idx);
}

int main(int argc, char **argv) {
    if(argc<2){printf("Usage: %s tokens.bin [steps] [lr]\n",argv[0]);return 1;}
    int steps=argc>2?atoi(argv[2]):DEFAULT_STEPS;
    float lr=argc>3?(float)atof(argv[3]):BASE_LR;

    /* Load tokens.bin */
    FILE *f=fopen(argv[1],"rb");
    if(!f){fprintf(stderr,"Cannot open %s\n",argv[1]);return 1;}
    int32_t header[2]; fread(header,4,2,f);
    int ntokens=header[0], V=header[1];
    printf("════════════════════════════════════════════════\n");
    printf("  SHORESH v3 trainer — Semantic BPE on notorch\n");
    printf("  %d tokens, vocab=%d\n",ntokens,V);
    printf("  DIM=%d L=%d H=%d HD=%d CTX=%d FFN=%d\n",DIM,N_LAYERS,N_HEADS,HD,CTX,FFN_DIM);
    printf("  %d steps, lr=%.1e\n",steps,lr);
    printf("════════════════════════════════════════════════\n");

    int *tokens=malloc(ntokens*sizeof(int));
    fread(tokens,4,ntokens,f);fclose(f);

    int val_start=(int)(ntokens*(1.0f-VAL_SPLIT));
    printf("  train: %d, val: %d\n",val_start,ntokens-val_start);

    nt_seed(42);srand(42);
    Model *m=mcreate(V);
    long np=nparams(m);
    printf("  %ld params (%.2fM), ratio: %.1f params/tok\n",np,np/1e6,(float)np/val_start);

    nt_train_mode(1);
    nt_schedule sched=nt_schedule_cosine(lr,200,steps,lr*0.03f);
    nt_nan_guard guard=nt_nan_guard_new();

    float ema=0,best=999,best_val=999;
    clock_t t0=clock();
    int input[CTX],target[CTX],pos=0;

    for(int step=0;step<steps;step++){
        float slr=nt_schedule_get_lr(&sched);
        if(pos+CTX+1>val_start)pos=0;
        for(int i=0;i<CTX;i++){input[i]=tokens[pos+i];target[i]=tokens[pos+i+1];}
        pos+=CTX/2;

        nt_tape_start();nt_train_mode(1);
        int loss_idx=mfwd(m,input,target,V);
        float loss=nt_tape_get()->entries[loss_idx].output->data[0];
        if(step==0)ema=loss;
        ema=0.99f*ema+0.01f*loss;
        if(loss<best)best=loss;

        nt_tape_backward(loss_idx);
        if(!nt_nan_guard_check(&guard)){nt_tape_clear();continue;}
        float gn=nt_tape_clip_grads(1.0f);
        nt_tape_chuck_step(slr,loss);
        nt_tape_clear();

        if(step%50==0){
            double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
            printf("step %5d | train %.4f | ema %.4f | best %.4f | lr %.2e | gn %.1f | %.0fs\n",
                   step,loss,ema,best,slr,gn,el);
            fflush(stdout);
        }
        if(step>0&&step%500==0){
            float vloss=0;int vn=0,vp=val_start;
            for(int vb=0;vb<20&&vp+CTX+1<ntokens;vb++,vp+=CTX*3){
                int vi[CTX],vt[CTX];
                for(int i=0;i<CTX;i++){vi[i]=tokens[vp+i];vt[i]=tokens[vp+i+1];}
                nt_tape_start();nt_train_mode(0);
                int vli=mfwd(m,vi,vt,V);
                vloss+=nt_tape_get()->entries[vli].output->data[0];
                nt_tape_clear();vn++;
            }
            vloss/=(vn>0?vn:1);
            if(vloss<best_val)best_val=vloss;
            printf("  VAL %d | val %.4f | best_val %.4f | gap %.4f\n",step,vloss,best_val,vloss-ema);
            fflush(stdout);nt_train_mode(1);
        }
        if(step>0&&step%2000==0){
            char ckpt[256];snprintf(ckpt,sizeof(ckpt),"shoresh_v3_step%d.bin",step);
            msave(m,ckpt);
        }
    }

    double total=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("════════════════════════════════════════════════\n");
    printf("  train ema: %.4f | best: %.4f\n",ema,best);
    printf("  val best: %.4f\n",best_val);
    printf("  time: %.1f sec (%.1f steps/s)\n",total,steps/total);
    printf("  nans: %d\n",guard.total_nan_count);
    printf("════════════════════════════════════════════════\n");
    msave(m,"shoresh_v3.bin");
    free(tokens);return 0;
}
