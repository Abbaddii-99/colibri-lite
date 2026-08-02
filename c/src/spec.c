#define _GNU_SOURCE
#include "model.h"
#include "spec.h"
#include "quant.h"
#include "tensor.h"
#include "kv_cache.h"
#include "pipeline.h"
#include "load.h"
#include "sampling.h"
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int ngram_draft(const int *ids, int len, int G, int *draft){
    if(len<4 || G<1) return 0;
    int a=ids[len-2], b=ids[len-1];
    for(int i=len-3;i>=1;i--)
        if(ids[i-1]==a && ids[i]==b){
            int n=0; for(int j=i+1;j<len && n<G;j++) draft[n++]=ids[j];
            return n;
        }
    return 0;
}

int mtp_argmax(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

int mtp_draft(Model *m, int next_tok, int kv, int G, int *draft){
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    int p=kv-1; if(p<0||G<1) return 0;
    if(m->kv_start[li]<0 || m->kv_start[li]>p) m->kv_start[li]=p;
    float *x=calloc(D,4), *cat=calloc(2*D,4), *hx=calloc(D,4);
    float *nrm=calloc(D,4), *tmp=calloc(D,4);
    float *row=calloc(D,4), *logit=calloc(c->vocab,4), *h=calloc(D,4);
    memcpy(h, m->hlast, D*sizeof(float));
    int tok=next_tok, n=0;
    for(int g=0; g<G; g++){
        int pos=p+g; if(pos+2>=m->max_t) break;
        embed_row(m, tok, x);
        rmsnorm(x, x, m->enorm, D, c->eps);
        rmsnorm(h, h, m->hnorm, D, c->eps);
        memcpy(cat, x, D*sizeof(float)); memcpy(cat+D, h, D*sizeof(float));
        matmul_qt(hx, cat, &m->eh_proj, 1);
        layer_forward(m, &m->mtpL, li, hx, 1, pos, nrm, tmp);
        rmsnorm(row, hx, m->mtp_norm, D, c->eps);
        matmul_qt(logit, row, &m->lm_head, 1);
        int t2=mtp_argmax(logit, c->vocab);
        draft[n++]=t2; tok=t2; memcpy(h, hx, D*sizeof(float));
    }
    free(x); free(cat); free(hx); free(nrm); free(tmp); free(row); free(logit); free(h);
    return n;
}

void mtp_absorb(Model *m, const int *next_ids, const float *x, int S, int pos_base){
    if(!m->has_mtp || S<1) return;
    Cfg *c=&m->c; int D=c->hidden, li=c->n_layers;
    if(m->kv_start[li]<0 || m->kv_start[li]>pos_base) m->kv_start[li]=pos_base;
    float *hx=calloc((int64_t)S*D,4), *cat=calloc(2*D,4), *e=calloc(D,4);
    float *hn=calloc(D,4), *hf=calloc(D,4);
    for(int i=0;i<S;i++){
        embed_row(m,next_ids[i],e);
        rmsnorm(e,e,m->enorm,D,c->eps);
        rmsnorm(hf,x+(int64_t)i*D,m->final_norm,D,c->eps);
        rmsnorm(hn,hf,m->hnorm,D,c->eps);
        memcpy(cat,e,D*sizeof(float)); memcpy(cat+D,hn,D*sizeof(float));
        matmul_qt(hx+(int64_t)i*D, cat, &m->eh_proj, 1);
    }
    float *nrm=calloc((int64_t)S*D,4), *tmp=calloc((int64_t)S*D,4);
    layer_forward(m,&m->mtpL,li,hx,S,pos_base,nrm,tmp);
    free(hx); free(cat); free(e); free(hn); free(hf); free(nrm); free(tmp);
}

void grammar_setup(void *T){
    const char *gf=getenv("GRAMMAR"); if(!gf||!*gf) return;
    FILE *f=fopen(gf,"rb");
    if(!f){ fprintf(stderr,"[GRAMMAR] cannot open %s\n",gf); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *txt=malloc((size_t)n+1);
    if(!txt || fread(txt,1,(size_t)n,f)!=(size_t)n){
        fprintf(stderr,"[GRAMMAR] failed to read %s\n",gf); fclose(f); free(txt); return; }
    fclose(f); txt[n]=0;
    if(gr_parse(&g_gram,txt)){ fprintf(stderr,"[GRAMMAR] %s: %s\n",gf,g_gram.err); free(txt); return; }
    free(txt);
    gr_state_init(&g_gst,&g_gram);
    if(!g_gst.alive){ fprintf(stderr,"[GRAMMAR] %s: grammar cannot be evaluated (left recursion?)\n",gf); return; }
    if(getenv("GRAMMAR_DRAFT")) g_gr_max=atoi(getenv("GRAMMAR_DRAFT"));
    if(g_gr_max<1) g_gr_max=1;
    if(g_gr_max>48) g_gr_max=48;
    g_gr_T=(Tok*)T; g_gr_on=1;
    fprintf(stderr,"[GRAMMAR] %s: %d rules, forced span capped at %d tokens/forward\n",gf,g_gram.n,g_gr_max);
}

void grammar_reset(void){
    if(!g_gr_on) return;
    gr_state_init(&g_gst,&g_gram); g_gr_armed=0;
    if(!g_gst.alive) g_gr_on=0;
}

void gr_feed(int t){
    if(!g_gr_on||!g_gr_T) return;
    char b[64]; int n=tok_decode(g_gr_T,&t,1,b,63);
    for(int i=0;i<n;i++){
        int r=gr_accept(&g_gst,(unsigned char)b[i]);
        if(r==1){ g_gr_armed=1; continue; }
        if(r<0){ g_gr_on=0; return; }
        if(!g_gr_armed) continue;
        gr_state_init(&g_gst,&g_gram); g_gr_armed=0;
        if(!g_gst.alive){ g_gr_on=0; return; }
        if(gr_accept(&g_gst,(unsigned char)b[i])==1) g_gr_armed=1;
    }
}

int grammar_draft(int *draft, int cap){
    if(!g_gr_on||!g_gr_armed||!g_gr_T||cap<1) return 0;
    if(g_gr_prop>=32 && g_gr_acc*2<g_gr_prop){
        g_gr_on=0;
        fprintf(stderr,"[GRAMMAR] %.0f%% acceptance after %llu proposals: grammar drafts disabled\n",
            100.0*g_gr_acc/g_gr_prop,(unsigned long long)g_gr_prop);
        return 0;
    }
    char fb[512]; int nb=gr_forced(&g_gst,fb,(int)sizeof fb-1);
    if(nb<=0) return 0;
    int g=tok_encode(g_gr_T,fb,nb,draft,cap);
    return g>0?g:0;
}

static inline int argmax_v(const float *lo, int V){
    int b=0; float bv=lo[0]; for(int i=1;i<V;i++) if(lo[i]>bv){bv=lo[i];b=i;} return b;
}

static inline int is_stop(int t){ for(int i=0;i<g_nstop;i++) if(t==g_stop[i]) return 1; return 0; }

static inline double rndu(void){
    g_rng^=g_rng<<13; g_rng^=g_rng>>7; g_rng^=g_rng<<17;
    return (double)(g_rng>>11)*(1.0/9007199254740992.0);
}

int spec_decode(Model *m, int *all, int kv, int n_new, int eos, float *logit,
                int *out, int *nout){
    Cfg *c=&m->c; int V=c->vocab; int emitted=0, done=0;
    int draft[64]; if(g_draft>63) g_draft=63;
    int carry_ban=-1;
    while(emitted<n_new && !done){
        int next=pick_tok(logit,V,carry_ban); carry_ban=-1; free(logit); logit=NULL;
        if((eos>=0 && next==eos) || is_stop(next)) break;
        out[emitted]=next; emitted++; all[kv]=next; m->n_emit++;
        gr_feed(next);
        if(emitted>=n_new) break;
        int g = 0, gsrc = 0;
        if(g_gr_on){
            g=grammar_draft(draft,g_gr_max);
            if(g>0) gsrc=1;
        }
        if(!g && g_draft>0){
            if(m->has_mtp && m->mtp_prop>=24 && m->mtp_acc*10 < m->mtp_prop){
                g_draft=0;
                fprintf(stderr,"[MTP] %.0f%% acceptance after %llu proposals: drafts disabled\n",
                    100.0*m->mtp_acc/m->mtp_prop, (unsigned long long)m->mtp_prop);
            }
        }
        if(!g && g_draft>0){
            if(m->has_mtp){ g=mtp_draft(m,next,kv,g_draft,draft); m->mtp_prop+=g; if(g)gsrc=2; }
            else { g=ngram_draft(all,kv+1,g_draft,draft); if(g)gsrc=2; }
        }
        if(g>n_new-emitted) g=n_new-emitted;
        if(kv+1+g+1>m->max_t) g=m->max_t-kv-2;
        if(g<0) g=0;
        if(gsrc==1) g_gr_prop+=(uint64_t)g;
        int S=1+g; int batch[64]; batch[0]=next; memcpy(batch+1,draft,g*sizeof(int));
        float *lo=step_all(m,batch,S,kv); m->n_fw++;
        int k=0;
        while(k<g && emitted<n_new){
            int accept;
            if(g_temp<=0) accept = (argmax_v(lo+(int64_t)k*V,V)==draft[k]);
            else { dist_build(lo+(int64_t)k*V,V);
                   accept = (rndu() < g_pbuf[draft[k]]); }
            if(!accept){ if(g_temp>0) carry_ban=draft[k]; break; }
            if((eos>=0 && draft[k]==eos) || is_stop(draft[k])){ done=1; break; }
            out[emitted]=draft[k]; emitted++; all[kv+1+k]=draft[k]; m->n_emit++;
            gr_feed(draft[k]); k++;
        }
        if(gsrc==1) g_gr_acc+=(uint64_t)k;
        else if(gsrc==2 && m->has_mtp) m->mtp_acc+=k;
        if(m->has_mtp && k>=1) mtp_absorb(m, all+kv+1, m->h_all, k, kv);
        if(m->h_all && k<S-1) memcpy(m->hlast, m->h_all+(int64_t)k*m->c.hidden, m->c.hidden*sizeof(float));
        kv += 1+k;
        logit=calloc(V,4); memcpy(logit, lo+(int64_t)k*V, V*sizeof(float)); free(lo);
    }
    if(logit) free(logit);
    *nout = kv;
    return emitted;
}

void emit_store(int t, void *ud){ EmitStore *e=(EmitStore*)ud; e->dst[e->n++]=t; }

void emit_stream(int t, void *ud){
    EmitStream *e=(EmitStream*)ud; char dec[64];
    int dn=tok_decode(e->T,&t,1,dec,63); dec[dn]=0; fputs(dec,stdout); fflush(stdout);
}
