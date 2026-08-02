#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

/* Tunables */
int g_idot       = 1;
int g_i4s        = 1;
int g_nopack     = 0;
int g_drop       = 0;
int g_direct     = 0;
int g_prefetch   = 0;
float g_temp     = -1;
float g_nuc      = 0.95f;
int g_topk       = 0;
float g_topp     = 0;
int g_absorb     = -1;
int g_dsa_force  = 0;
int g_pipe       = 0;
int g_pipe_nw    = 8;
int g_pilot      = 0;
int g_pilot_k    = 8;
int g_spec       = 1;
int g_looka      = 0;
int g_draft      = 0;
int g_grammar    = 0;
int g_gr_on      = 0;
int g_gr_armed   = 0;
int g_gr_max     = 24;
uint64_t g_gr_prop = 0, g_gr_acc = 0;
int g_verbose    = 0;
int g_usage      = 0;
int g_mlock      = -1;
int g_repin      = 0;
uint64_t g_last_repin = 0;
int g_kvsave     = 1;
double g_mem_avail_boot = 0;
int g_mtp        = 0;
char g_usage_path[2100] = {0};
uint64_t g_rng   = 0x9E3779B97F4A7C15ULL;
int g_stop[9]    = {0};
int g_nstop      = 0;

float *g_pbuf = NULL;
int *g_pidx = NULL;

int la_pred[2][130][16] = {{{0}}};
signed char la_val[2][130] = {{0}};
int64_t la_hit[3] = {0}, la_tot[3] = {0};

#ifdef COLI_CUDA
int g_cuda_enabled = 0;
double g_cuda_expert_gb = 0;
int g_cuda_dense = 0;
int g_cuda_devices[8] = {0};
int g_cuda_ndev = 0;
int g_cuda_rr = 0;
int64_t g_cuda_dense_projected[8] = {0};
#endif

/* ── Utilities ───────────────────────────────────────────── */
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <sys/time.h>
double now_s(void){
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec + t.tv_nsec*1e-9;
}
double rss_gb(void){
    struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}

/* ── Grammar globals ─────────────────────────────────────── */
Grammar g_gram;
GrState g_gst;
Tok *g_gr_T = NULL;

/* ── Quant scratch buffers ──────────────────────────────── */
typedef struct { int8_t *xq; float *sx; size_t xq_cap, sx_cap; } QScratch;
static QScratch g_qscratch = {0};
static pthread_mutex_t g_qscratch_mtx = PTHREAD_MUTEX_INITIALIZER;

void quant_scratch(size_t xn, size_t sn, int8_t **xq, float **sx) {
    pthread_mutex_lock(&g_qscratch_mtx);
    if (xn > g_qscratch.xq_cap) {
        free(g_qscratch.xq);
        g_qscratch.xq = malloc(xn);
        if (!g_qscratch.xq) { fprintf(stderr, "OOM quant scratch\n"); exit(1); }
        g_qscratch.xq_cap = xn;
    }
    if (sn > g_qscratch.sx_cap) {
        free(g_qscratch.sx);
        g_qscratch.sx = malloc(sn * sizeof(float));
        if (!g_qscratch.sx) { fprintf(stderr, "OOM quant scales\n"); exit(1); }
        g_qscratch.sx_cap = sn;
    }
    *xq = g_qscratch.xq;
    *sx = g_qscratch.sx;
    pthread_mutex_unlock(&g_qscratch_mtx);
}
