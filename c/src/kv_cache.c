#define _GNU_SOURCE
#include "model.h"
#include "kv_cache.h"
#include "load.h"
#include "tensor.h"
#include "quant.h"
#include "cache.h"
#include "pipeline.h"
#include "spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c;
    KVState *k=m->kv;
    if(k->Lc){ for(int i=0;i<c->n_layers+1;i++){ free(k->Lc[i]); free(k->Rc[i]); } free(k->Lc); free(k->Rc); }
    if(k->Ic){ for(int i=0;i<c->n_layers;i++) free(k->Ic[i]); free(k->Ic); k->Ic=NULL; }
    if(m->has_dsa){
        k->Ic=calloc(c->n_layers,sizeof(float*));
        if(!k->Ic){ fprintf(stderr,"OOM k->Ic\n"); exit(1); }
        for(int i=0;i<c->n_layers;i++) if(c->idx_type[i]) k->Ic[i]=calloc((int64_t)max_t*c->index_hd,4);
    }
    k->max_t=max_t;
    int NR=c->n_layers+1;
    int head_dim = c->hidden / c->n_heads;
    int kv_dim, rope_dim;
    if(IS_LLAMA(*c)){
        kv_dim = c->n_kv_heads * head_dim;
        rope_dim = kv_dim;  /* Rc stores V (same size as K) */
    } else {
        kv_dim = c->kv_lora;
        rope_dim = c->qk_rope;
    }
    k->Lc=calloc(NR,sizeof(float*)); k->Rc=calloc(NR,sizeof(float*));
    for(int i=0;i<NR;i++){
        k->Lc[i]=calloc((int64_t)max_t * (kv_dim > 0 ? kv_dim : 1), 4);
        k->Rc[i]=calloc((int64_t)max_t * (rope_dim > 0 ? rope_dim : 1), 4);
    }
    m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic; m->max_t=k->max_t; m->kv_start=k->kv_start;
}

void kv_bind(Model *m, KVState *k){
    m->kv=k; m->Lc=k->Lc; m->Rc=k->Rc; m->Ic=k->Ic;
    m->max_t=k->max_t; m->kv_start=k->kv_start;
}

float *step_all(Model *m, const int *ids, int S, int pos_base);

float *step(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=calloc((int64_t)S*D,4);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->hlast) memcpy(m->hlast, x+(int64_t)(S-1)*D, D*sizeof(float));
    if(m->has_mtp && S>=2 && g_draft>0) mtp_absorb(m, ids+1, x, S-1, pos_base);
    float *last=calloc(D,4); rmsnorm(last, x+(int64_t)(S-1)*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logit=calloc(c->vocab,4); matmul_qt(logit,last,&m->lm_head,1);
    m->t_head += now_s()-th0;
    free(x); free(last); return logit;
}

float *step_all(Model *m, const int *ids, int S, int pos_base){
    Cfg *c=&m->c; int D=c->hidden;
    float *x=calloc((int64_t)S*D,4);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,pos_base);
    if(m->h_all) memcpy(m->h_all, x, (int64_t)S*D*sizeof(float));
    float *norms=calloc((int64_t)S*D,4);
    for(int s=0;s<S;s++) rmsnorm(norms+(int64_t)s*D, x+(int64_t)s*D, m->final_norm, D, c->eps);
    double th0=now_s();
    float *logits=calloc((int64_t)S*c->vocab,4);
    for(int s=0;s<S;s++) matmul_qt(logits+(int64_t)s*c->vocab, norms+(int64_t)s*D, &m->lm_head, 1);
    m->t_head += now_s()-th0;
    free(x); free(norms); return logits;
}

void kv_disk_truncate(Model *m, int nrec){
    if(m->kv->disk_path[0]){
        FILE *f=fopen(m->kv->disk_path,"r+b"); if(!f) return;
        int NR=m->c.n_layers+1;
        int64_t pos=sizeof(int)+(int64_t)nrec*sizeof(int)+(int64_t)NR*sizeof(int32_t);
        if(ftruncate(fileno(f),pos)){} fclose(f);
    }
    m->kv->disk_nrec=nrec;
}

void kv_disk_reset(Model *m){ kv_disk_truncate(m,0); }

void kv_disk_append(Model *m, const int *hist, int len){
    if(!m->kv->disk_path[0]) return;
    FILE *f=fopen(m->kv->disk_path,"wb"); if(!f) return;
    int NR=m->c.n_layers+1;
    if(fwrite(&len,sizeof(int),1,f)!=1){ fclose(f); return; }
    if(len>0 && fwrite(hist,sizeof(int),len,f)!=(size_t)len){ fclose(f); return; }
    int32_t *starts=calloc(NR,sizeof(int32_t));
    for(int i=0;i<NR;i++) starts[i]=m->kv_start[i];
    if(fwrite(starts,sizeof(int32_t),NR,f)!=(size_t)NR){ free(starts); fclose(f); return; }
    free(starts);
    fclose(f);
    m->kv->disk_nrec=len;
}

int kv_disk_load(Model *m, int *hist, int maxctx){
    if(!m->kv->disk_path[0] || m->kv->disk_nrec<=0) return 0;
    int load=m->kv->disk_nrec;
    if(load>maxctx) load=maxctx;
    int NR=m->c.n_layers+1;
    FILE *f=fopen(m->kv->disk_path,"rb"); if(!f) return 0;
    int nrec=0; if(fread(&nrec,sizeof(int),1,f)!=1){ fclose(f); return 0; }
    if(nrec>load) nrec=load;
    if(nrec>0 && fread(hist,sizeof(int),(size_t)nrec,f)!=(size_t)nrec){ fclose(f); return 0; }
    int64_t off=sizeof(int)+(int64_t)nrec*sizeof(int)+(int64_t)NR*sizeof(int32_t);
    fseek(f,off,SEEK_SET);
    int32_t *starts=calloc(NR,sizeof(int32_t));
    if(fread(starts,sizeof(int32_t),NR,f)!=(size_t)NR){ free(starts); fclose(f); return 0; }
    for(int i=0;i<NR;i++) m->kv_start[i]=starts[i];
    free(starts);
    return nrec;
}

double kv_pool_bytes(Model *m, int max_ctx){
    Cfg *c=&m->c; int NR=c->n_layers+1;
    int head_dim = c->hidden / c->n_heads;
    int kv_slot = IS_LLAMA(*c) ? 2 * c->n_kv_heads * head_dim : c->kv_lora + c->qk_rope;
    double b=(double)NR*max_ctx*kv_slot*4;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        b+=(double)max_ctx*c->index_hd*4;
    return b;
}

