#define _GNU_SOURCE
#include "model.h"
#include "load.h"
#include "cache.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Config parsing helpers ─────────────────────────── */
#include "json.h"
#include <string.h>
static jval* cfg_root(const char *snap, char **arena){
    char p[2048]; snprintf(p,sizeof(p),"%s/config.json",snap);
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); if(fread(b,1,n,f)!=(size_t)n){} b[n]=0; fclose(f);
    return json_parse(b,arena);
}
static int gi(jval*r,const char*k){ jval*v=json_get(r,k); return v?(int)v->num:0; }
static const char *gs(jval*r,const char*k){ jval*v=json_get(r,k); return v&&v->str?v->str:NULL; }

/* ── Architecture detection ─────────────────────────── */
static int detect_arch(jval *r){
    const char *mt = gs(r,"model_type");
    if(!mt) mt = gs(r,"arch");
    if(!mt) return ARCH_GLM;  /* legacy GLM has no arch field */
    if(!strcmp(mt,"glm")||!strcmp(mt,"glm-5.2")) return ARCH_GLM;
    if(!strcmp(mt,"llama")||!strcmp(mt,"llama2")||!strcmp(mt,"mistral")) return ARCH_LLAMA;
    if(!strcmp(mt,"qwen2")||!strcmp(mt,"qwen2_moe")) return ARCH_LLAMA;
    fprintf(stderr,"unknown architecture '%s', trying GLM path\n",mt);
    return ARCH_GLM;
}

/* ── Public API ─────────────────────────────────────── */
void load_cfg(Cfg *c, const char *snap){
    char *ar=NULL; jval *r=cfg_root(snap,&ar);
    memset(c,0,sizeof(*c));
    c->arch = detect_arch(r);
    c->hidden=gi(r,"hidden_size"); c->n_layers=gi(r,"num_hidden_layers");
    c->n_heads=gi(r,"num_attention_heads");
    c->n_kv_heads=gi(r,"num_key_value_heads");
    if(!c->n_kv_heads) c->n_kv_heads=c->n_heads;
    c->dense_inter=gi(r,"intermediate_size"); c->vocab=gi(r,"vocab_size");
    c->max_seq_len=gi(r,"max_position_embeddings");
    if(!c->max_seq_len||c->max_seq_len>131072) c->max_seq_len=131072;
    jval *ep=json_get(r,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    c->theta = gi(r,"rope_theta");
    if(!c->theta){ jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
                   c->theta=th?(float)th->num:10000.f; }
    c->n_stop=0;
    jval *eo=json_get(r,"eos_token_id");
    if(eo){ if(eo->t==J_NUM) c->stop_ids[c->n_stop++]=(int)eo->num;
            else if(eo->t==J_ARR) for(int i=0;i<eo->len && c->n_stop<8;i++)
                c->stop_ids[c->n_stop++]=(int)eo->kids[i]->num; }

    if(IS_GLM(*c)){
        c->n_experts=gi(r,"n_routed_experts"); c->topk=gi(r,"num_experts_per_tok");
        c->moe_inter=gi(r,"moe_intermediate_size");
        c->first_dense=gi(r,"first_k_dense_replace");
        c->q_lora=gi(r,"q_lora_rank"); c->kv_lora=gi(r,"kv_lora_rank");
        c->qk_nope=gi(r,"qk_nope_head_dim"); c->qk_rope=gi(r,"qk_rope_head_dim");
        c->v_head=gi(r,"v_head_dim"); c->n_shared=gi(r,"n_shared_experts");
        c->n_group=gi(r,"n_group"); c->topk_group=gi(r,"topk_group");
        jval *nt=json_get(r,"norm_topk_prob"); c->norm_topk=(nt&&nt->t==J_BOOL)?nt->boolean:0;
        jval *rs=json_get(r,"routed_scaling_factor"); c->routed_scale=rs?(float)rs->num:1.f;
        c->index_topk=gi(r,"index_topk"); c->index_nh=gi(r,"index_n_heads"); c->index_hd=gi(r,"index_head_dim");
        { jval *it=json_get(r,"indexer_types");
          int freq=gi(r,"index_topk_freq"); if(freq<1) freq=1;
          jval *of=json_get(r,"index_skip_topk_offset"); int off=of?(int)of->num:2;
          for(int i=0;i<c->n_layers && i<128;i++){
              if(it && it->t==J_ARR && i<it->len && it->kids[i]->str)
                  c->idx_type[i] = !strcmp(it->kids[i]->str,"full");
              else { int v=i-off+1; if(v<0) v=0; c->idx_type[i] = (v%freq)==0; }
          } }
        c->qk_head=c->qk_nope+c->qk_rope;
        if(c->n_group!=1){ fprintf(stderr,"GLM: this engine requires n_group=1\n"); exit(1); }
    } else {
        c->n_experts=0; c->topk=0; c->moe_inter=0;
        c->first_dense=0; c->q_lora=0; c->kv_lora=0;
        c->qk_nope=0; c->qk_rope=c->qk_head=gi(r,"head_dim")?gi(r,"head_dim"):c->hidden/c->n_heads;
        c->v_head=0; c->n_shared=0; c->n_group=0; c->topk_group=0; c->norm_topk=0;
        c->routed_scale=1.f; c->index_topk=0; c->index_nh=0; c->index_hd=0;
        memset(c->idx_type,0,sizeof(c->idx_type));
    }
    c->attn_scale = 1.f / sqrtf((float)(c->qk_head>0?c->qk_head:1));
    #define CKR(name,v,lo,hi) if((v)<(lo)||(v)>(hi)){ \
        fprintf(stderr,"config: %s=%d is outside [%d,%d]\n",name,(int)(v),(int)(lo),(int)(hi)); exit(1); }
    CKR("hidden_size",c->hidden,1,1<<20)         CKR("num_hidden_layers",c->n_layers,1,128)
    CKR("num_attention_heads",c->n_heads,1,1024) CKR("intermediate_size",c->dense_inter,1,1<<24)
    CKR("vocab_size",c->vocab,1,1<<24)
    if(IS_GLM(*c)){
        CKR("n_routed_experts",c->n_experts,1,4096) CKR("num_experts_per_tok",c->topk,1,64)
        CKR("moe_intermediate_size",c->moe_inter,1,1<<20) CKR("first_k_dense_replace",c->first_dense,0,c->n_layers)
        CKR("q_lora_rank",c->q_lora,0,1<<20)         CKR("kv_lora_rank",c->kv_lora,1,1<<20)
        CKR("qk_nope_head_dim",c->qk_nope,1,1<<16)   CKR("qk_rope_head_dim",c->qk_rope,1,1<<16)
        CKR("v_head_dim",c->v_head,1,1<<16)          CKR("n_shared_experts",c->n_shared,0,64)
        CKR("index_topk",c->index_topk,0,1<<20)      CKR("index_n_heads",c->index_nh,0,1024)
        CKR("index_head_dim",c->index_hd,0,1<<16)
    }
    #undef CKR
    /* Precompute RoPE base frequencies */
    c->rope_freq_n = 0;
    if(IS_LLAMA(*c)){
        int dim = c->qk_head; /* head_dim */
        c->rope_freq_n = dim / 2;
        if(c->rope_freq_n > 256) c->rope_freq_n = 256;
        for(int k = 0; k < c->rope_freq_n; k++)
            c->rope_freq[k] = powf(c->theta, -4.0f * k / dim);
    } else {
        c->rope_freq_n = c->qk_rope / 2;
        if(c->rope_freq_n > 256) c->rope_freq_n = 256;
        for(int k = 0; k < c->rope_freq_n; k++)
            c->rope_freq[k] = powf(c->theta, -2.0f * k / c->qk_rope);
    }
    free(ar);
}

void qt_from_disk(Model *m, const char *name, int O, int I, int bits, int drop, QT *t){
    char sn[300]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->S,sn)){
        int64_t nb=st_nbytes(&m->S,name);
        if(nb<0){ fprintf(stderr,"tensor '%s' not found\n",name); exit(1); }
        int fmt = (nb==(int64_t)O*I)?1 : (nb==(int64_t)O*((I+1)/2))?2 : 3;
        if(fmt==1){ if(t->fmt!=1||!t->q8){ t->fmt=1; t->O=O; t->I=I; t->q8=malloc(nb); t->s=calloc(O,4); } st_read_raw(&m->S,name,t->q8,drop); }
        else      { if(t->fmt!=fmt||!t->q4){ t->fmt=fmt; t->O=O; t->I=I; t->q4=malloc(nb); t->s=calloc(O,4); } st_read_raw(&m->S,name,t->q4,drop); }
        st_read_f32(&m->S,sn,t->s,drop);
    } else {
        if(!t->qf && !t->q8 && !t->q4) qt_alloc(t,O,I,bits);
        if(t->fmt==0) st_read_f32(&m->S,name,t->qf,drop);
        else { float *tmp=calloc((int64_t)O*I,4); st_read_f32(&m->S,name,tmp,drop); qt_fill(t,tmp,bits); free(tmp); }
    }
}

QT qt_load(Model *m, const char *name, int O, int I, int bits){
    QT t; memset(&t,0,sizeof(t)); qt_from_disk(m,name,O,I,bits,0,&t);
#ifdef COLI_CUDA
    if(g_cuda_enabled&&g_cuda_dense){
        t.cuda_eligible=1;
        int slot=g_cuda_rr++%g_cuda_ndev; t.cuda_device=g_cuda_devices[slot];
        g_cuda_dense_projected[slot]+=qt_bytes(&t);
    }
#endif
    return t;
}

float *ld(Model *m, const char *name){
    int64_t n=st_numel(&m->S,name); if(n<0){fprintf(stderr,"missing %s\n",name);exit(1);}
    float *p=calloc(n,4); st_read_f32(&m->S,name,p,0); return p;
}

void model_init(Model *m, const char *snap, int cap, int ebits, int dbits){
    memset(m,0,sizeof(*m)); m->ebits=ebits; m->dbits=dbits;
    load_cfg(&m->c,snap); st_init(&m->S,snap);
    Cfg *c=&m->c; char nm[256]; int H=c->n_heads, D=c->hidden;
    int io_bits = dbits>=8 ? 16 : dbits;
    m->embed   = qt_load(m,"model.embed_tokens.weight", c->vocab, D, io_bits);
    m->lm_head = qt_load(m,"lm_head.weight", c->vocab, D, io_bits);
    m->final_norm = ld(m,"model.norm.weight");
    m->L=calloc(c->n_layers,sizeof(Layer));
    int NR=c->n_layers+1;
    m->ecap=cap; m->ecache=calloc(NR,sizeof(ESlot*)); m->ecn=calloc(NR,sizeof(int));
    m->eroute=calloc(NR,sizeof(int*)); m->enr=calloc(NR,sizeof(int));
    m->pin=calloc(NR,sizeof(ESlot*)); m->npin=calloc(NR,sizeof(int));
    m->eusage=calloc(NR,sizeof(uint32_t*)); m->eheat=calloc(NR,sizeof(uint32_t*));
    m->kv=calloc(1,sizeof(KVState));
    m->kv_start=m->kv->kv_start=calloc(NR,sizeof(int));
    if(IS_LLAMA(*c)){
        int hd = D / H;
        for(int i=0;i<c->n_layers;i++){
            Layer *l=&m->L[i];
            #define PL(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PL("input_layernorm.weight"));
            l->post_ln=ld(m,PL("post_attention_layernorm.weight"));
            l->q_proj  = qt_load(m,PL("self_attn.q_proj.weight"), H*hd, D, dbits);
            l->k_proj  = qt_load(m,PL("self_attn.k_proj.weight"), c->n_kv_heads*hd, D, dbits);
            l->v_proj  = qt_load(m,PL("self_attn.v_proj.weight"), c->n_kv_heads*hd, D, dbits);
            l->o       = qt_load(m,PL("self_attn.o_proj.weight"), D, H*hd, dbits);
            l->sparse = 0;  /* LLaMA: no MoE */
            l->gate_proj = qt_load(m,PL("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
            l->up_proj   = qt_load(m,PL("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
            l->down_proj = qt_load(m,PL("mlp.down_proj.weight"), D, c->dense_inter, dbits);
            #undef PL
        }
    } else {
        for(int i=0;i<c->n_layers;i++){
            Layer *l=&m->L[i];
            #define P(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,P("input_layernorm.weight"));
            l->post_ln=ld(m,P("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,P("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,P("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,P("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,P("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse = (i >= c->first_dense);
            if(!l->sparse){
                l->gate_proj = qt_load(m,P("mlp.gate_proj.weight"), c->dense_inter, D, dbits);
                l->up_proj   = qt_load(m,P("mlp.up_proj.weight"),   c->dense_inter, D, dbits);
                l->down_proj = qt_load(m,P("mlp.down_proj.weight"), D, c->dense_inter, dbits);
            } else {
                l->router=ld(m,P("mlp.gate.weight"));
                l->router_bias=ld(m,P("mlp.gate.e_score_correction_bias"));
                int64_t sI64=(int64_t)c->moe_inter*c->n_shared; int sI=(int)sI64;
                l->sh_gate = qt_load(m,P("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
                l->sh_up   = qt_load(m,P("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
                l->sh_down = qt_load(m,P("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
                m->ecache[i]=calloc(cap,sizeof(ESlot));
                m->eroute[i]=calloc(c->topk,sizeof(int));
                m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
                m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            }
            #undef P
        }
    }
    if(IS_GLM(*c)){
        const char *req[]={"eh_proj.weight","enorm.weight","hnorm.weight","shared_head.norm.weight",
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_a_proj.weight","self_attn.q_b_proj.weight","self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight","self_attn.o_proj.weight","mlp.gate.weight",
            "mlp.shared_experts.gate_proj.weight","mlp.shared_experts.down_proj.weight",
            "mlp.experts.0.gate_proj.weight","mlp.experts.255.down_proj.weight"};
        char mn[256]; m->has_mtp=1;
        for(unsigned q=0;q<sizeof(req)/sizeof(req[0]);q++){
            snprintf(mn,sizeof(mn),"model.layers.%d.%s",c->n_layers,req[q]);
            if(!st_has(&m->S,mn)){ m->has_mtp=0; break; }
        }
        if(getenv("MTP") && atoi(getenv("MTP"))==0) m->has_mtp=0;
        if(m->has_mtp){
            int i=c->n_layers; Layer *l=&m->mtpL;
            #define PM(s) (snprintf(nm,sizeof(nm),"model.layers.%d." s,i),nm)
            l->in_ln=ld(m,PM("input_layernorm.weight"));
            l->post_ln=ld(m,PM("post_attention_layernorm.weight"));
            l->q_a   = qt_load(m,PM("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,PM("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,PM("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
            l->kv_a  = qt_load(m,PM("self_attn.kv_a_proj_with_mqa.weight"), c->kv_lora+c->qk_rope, D, dbits);
            l->kv_a_ln= ld(m,PM("self_attn.kv_a_layernorm.weight"));
            l->kv_b  = qt_load(m,PM("self_attn.kv_b_proj.weight"), H*(c->qk_nope+c->v_head), c->kv_lora, dbits);
            l->o     = qt_load(m,PM("self_attn.o_proj.weight"), D, H*c->v_head, dbits);
            l->sparse=1;
            l->router=ld(m,PM("mlp.gate.weight"));
            l->router_bias=ld(m,PM("mlp.gate.e_score_correction_bias"));
            int64_t sI64=(int64_t)c->moe_inter*c->n_shared; int sI=(int)sI64;
            l->sh_gate = qt_load(m,PM("mlp.shared_experts.gate_proj.weight"), sI, D, dbits);
            l->sh_up   = qt_load(m,PM("mlp.shared_experts.up_proj.weight"),   sI, D, dbits);
            l->sh_down = qt_load(m,PM("mlp.shared_experts.down_proj.weight"), D, sI, dbits);
            m->eh_proj = qt_load(m,PM("eh_proj.weight"), D, 2*D, dbits);
            m->enorm=ld(m,PM("enorm.weight")); m->hnorm=ld(m,PM("hnorm.weight"));
            m->mtp_norm=ld(m,PM("shared_head.norm.weight"));
            m->ecache[i]=calloc(cap,sizeof(ESlot));
            m->eroute[i]=calloc(c->topk,sizeof(int));
            m->eusage[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->eheat[i]=calloc(c->n_experts,sizeof(uint32_t));
            m->kv_start[i]=-1;
            #undef PM
        }
    }
    if(IS_GLM(*c)){
        m->has_dsa = (c->index_topk>0 && c->index_nh>0 && c->index_hd>0 && c->index_hd<=256);
        char inm[300];
        for(int i=0;i<c->n_layers && m->has_dsa;i++){
            if(!c->idx_type[i]) continue;
            snprintf(inm,sizeof(inm),"model.layers.%d.self_attn.indexer.wq_b.weight",i);
            if(!st_has(&m->S,inm)) m->has_dsa=0;
        }
        if(getenv("DSA") && atoi(getenv("DSA"))==0) m->has_dsa=0;
        if(m->has_dsa){
            m->ix_wq=calloc(c->n_layers,sizeof(QT)); m->ix_wk=calloc(c->n_layers,sizeof(QT));
            m->ix_wp=calloc(c->n_layers,sizeof(QT));
            m->ix_knw=calloc(c->n_layers,sizeof(float*)); m->ix_knb=calloc(c->n_layers,sizeof(float*));
            for(int i=0;i<c->n_layers;i++){
                if(!c->idx_type[i]) continue;
                #define PI(s) (snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.indexer." s,i),nm)
                m->ix_wq[i]=qt_load(m,PI("wq_b.weight"), c->index_nh*c->index_hd, c->q_lora, dbits);
                m->ix_wk[i]=qt_load(m,PI("wk.weight"), c->index_hd, D, dbits);
                m->ix_wp[i]=qt_load(m,PI("weights_proj.weight"), c->index_nh, D, dbits);
                m->ix_knw[i]=ld(m,PI("k_norm.weight")); m->ix_knb[i]=ld(m,PI("k_norm.bias"));
                #undef PI
            }
            fprintf(stderr,"[DSA] indexer active: top-%d sparse attention beyond %d context tokens\n",
                c->index_topk, c->index_topk);
        }
    }
    m->hlast=calloc(D,4); m->h_all=calloc((int64_t)64*D,4);
    int64_t rb=qt_bytes(&m->embed)+qt_bytes(&m->lm_head);
    for(int i=0;i<c->n_layers;i++){ Layer *l=&m->L[i];
        if(IS_LLAMA(*c)){
            rb+=qt_bytes(&l->q_proj)+qt_bytes(&l->k_proj)+qt_bytes(&l->v_proj)+qt_bytes(&l->o);
            rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
        } else {
            rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
            if(!l->sparse) rb+=qt_bytes(&l->gate_proj)+qt_bytes(&l->up_proj)+qt_bytes(&l->down_proj);
            else rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down);
        }
    }
    if(m->has_mtp){ Layer *l=&m->mtpL;
        rb+=qt_bytes(&l->q_a)+qt_bytes(&l->q_b)+qt_bytes(&l->kv_a)+qt_bytes(&l->kv_b)+qt_bytes(&l->o);
        rb+=qt_bytes(&l->sh_gate)+qt_bytes(&l->sh_up)+qt_bytes(&l->sh_down)+qt_bytes(&m->eh_proj);
    }
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        rb+=qt_bytes(&m->ix_wq[i])+qt_bytes(&m->ix_wk[i])+qt_bytes(&m->ix_wp[i]);
    m->resident_bytes=rb;
}

static void qt_free_fields(QT *t){ free(t->qf); free(t->q8); free(t->q4); free(t->s); }

void model_free(Model *m){
    for(int i=0;i<m->c.n_layers;i++){
        Layer *l=&m->L[i];
        free(l->in_ln); free(l->post_ln);
        if(IS_LLAMA(m->c)){
            qt_free_fields(&l->q_proj); qt_free_fields(&l->k_proj);
            qt_free_fields(&l->v_proj); qt_free_fields(&l->o);
            qt_free_fields(&l->gate_proj); qt_free_fields(&l->up_proj); qt_free_fields(&l->down_proj);
        } else {
            qt_free_fields(&l->q_a); qt_free_fields(&l->q_b);
            qt_free_fields(&l->kv_a); qt_free_fields(&l->kv_b); qt_free_fields(&l->o);
            free(l->q_a_ln); free(l->kv_a_ln);
            if(!l->sparse){
                qt_free_fields(&l->gate_proj); qt_free_fields(&l->up_proj); qt_free_fields(&l->down_proj);
            } else {
                free(l->router); free(l->router_bias);
                qt_free_fields(&l->sh_gate); qt_free_fields(&l->sh_up); qt_free_fields(&l->sh_down);
            }
        }
    }
    free(m->L);
    /* MTP */
    if(m->has_mtp){
        Layer *l=&m->mtpL;
        free(l->in_ln); free(l->post_ln);
        free(l->q_a.qf); free(l->q_a.q8); free(l->q_a.q4); free(l->q_a.s);
        free(l->q_b.qf); free(l->q_b.q8); free(l->q_b.q4); free(l->q_b.s);
        free(l->kv_a.qf); free(l->kv_a.q8); free(l->kv_a.q4); free(l->kv_a.s);
        free(l->kv_b.qf); free(l->kv_b.q8); free(l->kv_b.q4); free(l->kv_b.s);
        free(l->o.qf); free(l->o.q8); free(l->o.q4); free(l->o.s);
        free(l->q_a_ln); free(l->kv_a_ln);
        free(l->router); free(l->router_bias);
        free(l->sh_gate.qf); free(l->sh_gate.q8); free(l->sh_gate.q4); free(l->sh_gate.s);
        free(l->sh_up.qf); free(l->sh_up.q8); free(l->sh_up.q4); free(l->sh_up.s);
        free(l->sh_down.qf); free(l->sh_down.q8); free(l->sh_down.q4); free(l->sh_down.s);
        free(m->eh_proj.qf); free(m->eh_proj.q8); free(m->eh_proj.q4); free(m->eh_proj.s);
        free(m->enorm); free(m->hnorm); free(m->mtp_norm);
    }
    free(m->final_norm);
    free(m->embed.qf); free(m->embed.q8); free(m->embed.q4); free(m->embed.s);
    free(m->lm_head.qf); free(m->lm_head.q8); free(m->lm_head.q4); free(m->lm_head.s);
    free(m->hlast); free(m->h_all);
    /* expert caches — free slab/fslab before freeing arrays */
    if(m->ecache) for(int i=0;i<=m->c.n_layers;i++){
        if(m->ecache[i]){ for(int j=0;j<m->ecap;j++){ compat_aligned_free(m->ecache[i][j].slab); free(m->ecache[i][j].fslab); } }
        free(m->ecache[i]);
    }
    free(m->ecache); free(m->ecn);
    if(m->pin) for(int i=0;i<=m->c.n_layers;i++){
        if(m->pin[i]){ for(int j=0;j<m->npin[i];j++){ compat_aligned_free(m->pin[i][j].slab); free(m->pin[i][j].fslab); } }
        free(m->pin[i]);
    }
    free(m->pin); free(m->npin);
    /* workspace slots */
    for(int i=0;i<64;i++){ compat_aligned_free(m->ws[i].slab); free(m->ws[i].fslab); }
    if(m->eusage) for(int i=0;i<=m->c.n_layers;i++) free(m->eusage[i]);
    free(m->eusage);
    if(m->eheat) for(int i=0;i<=m->c.n_layers;i++) free(m->eheat[i]);
    free(m->eheat);
    if(m->eroute) for(int i=0;i<=m->c.n_layers;i++) free(m->eroute[i]);
    free(m->eroute); free(m->enr);
    /* DSA */
    if(m->has_dsa){
        for(int i=0;i<m->c.n_layers;i++){
            free(m->ix_wq[i].qf); free(m->ix_wq[i].q8); free(m->ix_wq[i].q4); free(m->ix_wq[i].s);
            free(m->ix_wk[i].qf); free(m->ix_wk[i].q8); free(m->ix_wk[i].q4); free(m->ix_wk[i].s);
            free(m->ix_wp[i].qf); free(m->ix_wp[i].q8); free(m->ix_wp[i].q4); free(m->ix_wp[i].s);
            free(m->ix_knw[i]); free(m->ix_knb[i]);
        }
        free(m->ix_wq); free(m->ix_wk); free(m->ix_wp);
        free(m->ix_knw); free(m->ix_knb);
        free(m->dsa_sel); free(m->dsa_nsel);
    }
    /* KV */
    free(m->kv);
    free(m->kv_start);
    /* shards */
    st_free(&m->S);
}

void embed_row(Model *m, int tok, float *x){
    int D=m->c.hidden; QT *e=&m->embed;
    if(e->fmt==0){ memcpy(x, e->qf+(int64_t)tok*D, D*sizeof(float)); return; }
    if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)tok*D; float s=e->s[tok];
        for(int i=0;i<D;i++) x[i]=(float)q[i]*s; return; }
    if(e->fmt==2){ const uint8_t *q=e->q4+(int64_t)tok*((D+1)/2); float s=e->s[tok];
        for(int i=0;i<D;i+=2){ uint8_t byte=q[i>>1]; x[i]=(float)((int)(byte&0xF)-8)*s;
            if(i+1<D) x[i+1]=(float)((int)(byte>>4)-8)*s; }
        return; }
    const uint8_t *q=e->q4+(int64_t)tok*((D+3)/4); float s=e->s[tok];
    for(int i=0;i<D;i++){ uint8_t byte=q[i>>2]; int sh=(i&3)*2; x[i]=(float)((int)((byte>>sh)&3)-2)*s; }
}
