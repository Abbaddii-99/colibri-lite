#define _GNU_SOURCE
#include "model.h"
#include "pipeline.h"
#include "tensor.h"
#include "quant.h"
#include "attention.h"
#include "moe.h"
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

static inline float *falloc(int64_t n){ return (float*)calloc((size_t)n,sizeof(float)); }

static int cmp_edesc(const void *a, const void *b){
    float fa=*(const float*)a, fb=*(const float*)b;
    return (fa>fb)?-1:(fa<fb)?1:0;
}

static inline float sigmoidf(float x){
    if(x<-40) return 0; if(x>40) return 1;
    return 1.f/(1.f+expf(-x));
}

void la_predict(Model *m, int target, const float *h, int kind){
    Cfg *c=&m->c; if(!IS_GLM(*c)) return;
    Layer *l=&m->L[target]; int D=c->hidden, E=c->n_experts, K=c->topk;
    float *nrm=falloc(D), *ch=falloc(E);
    rmsnorm(nrm,h,l->post_ln,D,c->eps);
    matmul(ch,nrm,l->router,1,D,E);
    for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
    int *pred=la_pred[kind][target];
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++){ int tk=0; for(int j=0;j<kk;j++) if(pred[j]==e){tk=1;break;}
            if(!tk && ch[e]>bv){bv=ch[e];best=e;} }
        pred[kk]=best; }
    la_val[kind][target]=1;
    free(nrm); free(ch);
}

/* PILOT ring buffer (lock-free 1P/1C) */
static struct { int l,e; } pilot_q[4096];
static volatile unsigned pilot_w=0, pilot_r=0;
static Model *pilot_m=NULL;

void *pilot_worker(void *arg){
    (void)arg;
    for(;;){
        unsigned r=__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE);
        unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_ACQUIRE);
        if(r==w){ usleep(200); continue; }
        expert_prefetch(pilot_m, pilot_q[r&4095].l, pilot_q[r&4095].e);
        __atomic_store_n(&pilot_r,r+1,__ATOMIC_RELEASE);
    }
    return NULL;
}

void pilot_prefetch(Model *m, int lnext, const float *x, int S){
    Cfg *c=&m->c; if(!IS_GLM(*c)) return;
    Layer *l=&m->L[lnext]; int D=c->hidden, E=c->n_experts;
    int K = g_pilot_k<c->topk ? g_pilot_k : c->topk;
    if(!pilot_m){ pilot_m=m; pthread_t t; pthread_create(&t,NULL,pilot_worker,NULL); }
    float *nrm=falloc(D), *ch=falloc(E);
    for(int s=0;s<S;s++){
        rmsnorm(nrm, x+(int64_t)s*D, l->post_ln, D, c->eps);
        matmul(ch, nrm, l->router, 1, D, E);
        for(int e=0;e<E;e++) ch[e]=sigmoidf(ch[e])+l->router_bias[e];
        for(int kk=0;kk<K;kk++){
            int best=0; for(int e=1;e<E;e++) if(ch[e]>ch[best]) best=e;
            ch[best]=-2e30f;
            int found=0; ESlot *P=m->pin[lnext];
            for(int z=0;z<m->npin[lnext] && !found;z++) if(P[z].eid==best) found=1;
            ESlot *Sl=m->ecache[lnext];
            for(int z=0;z<m->ecn[lnext] && !found;z++) if(Sl[z].eid==best) found=1;
            if(!found){
                unsigned w=__atomic_load_n(&pilot_w,__ATOMIC_RELAXED);
                if(w-__atomic_load_n(&pilot_r,__ATOMIC_ACQUIRE)<4096){
                    pilot_q[w&4095].l=lnext; pilot_q[w&4095].e=best;
                    __atomic_store_n(&pilot_w,w+1,__ATOMIC_RELEASE);
                }
            }
        }
    }
    free(nrm); free(ch);
}

void layer_forward(Model *m, Layer *l, int li, float *x, int S, int pos_base, float *nrm, float *tmp){
    Cfg *c=&m->c; int D=c->hidden;
    if(g_spec && g_prefetch && l->sparse && m->enr[li]>0)
        for(int z=0;z<m->enr[li];z++) expert_prefetch(m,li,m->eroute[li][z]);
    if(g_looka && S==1 && li<c->n_layers && l->sparse) la_predict(m,li,x,0);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->in_ln, D, c->eps);
    attention(m,l,li,nrm,S,pos_base,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
    if(g_pilot && S<=8 && li+1<c->n_layers && m->L[li+1].sparse) pilot_prefetch(m,li+1,x,S);
    if(g_looka && S==1 && li+1<c->n_layers && m->L[li+1].sparse) la_predict(m,li+1,x,1);
    for(int s=0;s<S;s++) rmsnorm(nrm+(int64_t)s*D, x+(int64_t)s*D, l->post_ln, D, c->eps);
    if(l->sparse) moe(m,l,li,nrm,S,tmp); else dense_mlp(l,nrm,S,D,c->dense_inter,tmp);
    for(int64_t j=0;j<(int64_t)S*D;j++) x[j]+=tmp[j];
}

void layers_forward(Model *m, float *x, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *nrm=calloc((int64_t)S*D,4), *tmp=calloc((int64_t)S*D,4);
    for(int i=0;i<c->n_layers;i++){
        if(S>=8 && (i%4==0 || i==c->n_layers-1))
            fprintf(stderr,"[prefill] layer %d/%d · %d token\n", i+1, c->n_layers, S);
        layer_forward(m,&m->L[i],i,x,S,pos_base,nrm,tmp);
    }
    free(nrm); free(tmp);
}
