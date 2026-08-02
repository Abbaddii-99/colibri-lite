#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "st.h"
#include "tier.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif

/* ── Architecture identifiers ──────────────────────────────── */
#define ARCH_GLM    0
#define ARCH_LLAMA  1
#define ARCH_NAME(c) ((c).arch==ARCH_GLM?"glm":(c).arch==ARCH_LLAMA?"llama":"?")
#define IS_GLM(c)   ((c).arch == ARCH_GLM)
#define IS_LLAMA(c) ((c).arch == ARCH_LLAMA)

/* ── Configuration ──────────────────────────────────────────── */
typedef struct {
    int arch;
    int hidden, n_layers, n_heads, n_kv_heads, n_experts, topk, moe_inter, dense_inter;
    int first_dense, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head, n_shared, vocab;
    int n_group, topk_group, norm_topk;
    int stop_ids[8], n_stop;
    int index_topk, index_nh, index_hd;
    int8_t idx_type[128];
    float eps, theta, attn_scale, routed_scale;
    int max_seq_len;
    float rope_freq[256];   /* precomputed cos/sin base freqs for RoPE */
    int rope_freq_n;
} Cfg;

/* ── Quantized Tensor (F32 / INT8 / INT4 / INT2) ─────── */
typedef struct {
    int fmt; float *qf; int8_t *q8; uint8_t *q4; float *s; int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
    int cuda_eligible, cuda_failed, cuda_device;
} QT;

static inline int64_t qt_bytes(const QT *t) {
    int64_t n = (int64_t)t->O * t->I;
    if (t->fmt == 0) return n * 4;
    if (t->fmt == 1) return n + (int64_t)t->O * 4;
    if (t->fmt == 3) return (int64_t)t->O * ((t->I + 3) / 4) + (int64_t)t->O * 4;
    return (int64_t)t->O * ((t->I + 1) / 2) + (int64_t)t->O * 4;
}

/* ── Layer (attention + MLP/MoE weights) ──────────────── */
typedef struct {
    float *in_ln, *post_ln;
    QT q_a, q_b, kv_a, kv_b, o; float *q_a_ln, *kv_a_ln;
    QT q_proj, k_proj, v_proj;   /* LLaMA direct Q/K/V projections (unused for GLM) */
    int sparse;
    QT gate_proj, up_proj, down_proj;
    float *router, *router_bias;
    QT sh_gate, sh_up, sh_down;
} Layer;

/* ── Expert slot (LRU cache entry) ───────────────────── */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used; } ESlot;

/* ── KV-cache state ─────────────────────────────────── */
typedef struct {
    float **Lc, **Rc, **Ic;
    int *kv_start, max_t;
    int disk_nrec;
    char disk_path[2048];
} KVState;

/* ── REPIN candidate (hot-store promotion) ──────────── */
typedef struct { long gain; int l, slot, eid; } RepinCand;

/* ── Top-level model ─────────────────────────────────── */
typedef struct {
    Cfg c; shards S;
    int ebits, dbits;
    QT embed, lm_head; float *final_norm;
    Layer *L;
    float **Lc, **Rc; int max_t;
    int *kv_start;
    KVState *kv;
    ESlot **ecache; int *ecn; int ecap;
    ESlot ws[64];
    ESlot **pin; int *npin;
    uint32_t **eusage;
    uint32_t **eheat;
    int has_dsa;
    QT *ix_wq, *ix_wk, *ix_wp;
    float **ix_knw, **ix_knb;
    float **Ic;
    int *dsa_sel, *dsa_nsel; int dsa_scap;
    int has_mtp; Layer mtpL; QT eh_proj;
    float *enorm, *hnorm, *mtp_norm;
    float *hlast, *h_all;
    uint64_t mtp_prop, mtp_acc;
    int **eroute; int *enr;
    uint64_t eclock, hits, miss, ereq;
    uint64_t gpu_expert_calls; int gpu_expert_count; int64_t gpu_expert_bytes;
    uint64_t n_fw, n_emit;
    double t_edisk, t_emm, t_attn, t_kvb, t_head;
    int64_t resident_bytes;
} Model;

/* ── Global tunables ─────────────────────────────────── */
extern int g_idot;
extern int g_i4s;
extern int g_nopack;
extern int g_prefetch;
extern int g_drop;
extern int g_direct;
extern float g_temp;
extern float g_nuc;
extern int g_topk;
extern float g_topp;
extern int g_absorb;
extern int g_dsa_force;
extern int g_pipe;
extern int g_pipe_nw;
extern int g_pilot;
extern int g_pilot_k;
extern int g_spec;
extern int g_looka;
extern int g_draft;
extern int g_grammar;
extern int g_gr_on;
extern int g_gr_armed;
extern int g_gr_max;
extern uint64_t g_gr_prop, g_gr_acc;
extern int g_verbose;
extern int g_usage;
extern int g_mlock;
extern int g_repin;
extern uint64_t g_last_repin;
extern int g_kvsave;
extern double g_mem_avail_boot;
extern int g_mtp;
extern char g_usage_path[2100];
extern uint64_t g_rng;
extern int g_stop[9], g_nstop;

/* Sampling buffers */
extern float *g_pbuf;
extern int *g_pidx;

/* Lookahead prediction tables */
extern int la_pred[2][130][16];
extern signed char la_val[2][130];
extern int64_t la_hit[3], la_tot[3];

#ifdef COLI_CUDA
extern int g_cuda_enabled;
extern double g_cuda_expert_gb;
extern int g_cuda_dense;
extern int g_cuda_devices[8], g_cuda_ndev, g_cuda_rr;
extern int64_t g_cuda_dense_projected[8];
#endif

/* Scratch buffer accessor */
void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx);

/* Utilities */
double now_s(void);
double rss_gb(void);

/* Grammar globals */
#include "grammar.h"
extern Grammar g_gram;
extern GrState g_gst;

/* Tokenizer */
#include "tok.h"
extern Tok *g_gr_T;

#endif /* MODEL_H */
