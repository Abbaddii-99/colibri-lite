#define _GNU_SOURCE
#include "model.h"
#include "attention.h"
#include "tensor.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* ── Standard RoPE (LLaMA-style) ─────────────────────── */
static void rope_llama(float *qk, int pos, int dim, const float *freq_base){
    for(int i = 0; i < dim; i += 2){
        float ang = (float)pos * freq_base[i >> 1];
        float cs = cosf(ang), sn = sinf(ang);
        float a = qk[i], b = qk[i + 1];
        qk[i]     = a * cs - b * sn;
        qk[i + 1] = b * cs + a * sn;
    }
}

/* ── Standard MHA/GQA attention (LLaMA) ────────────── */
static void attention_llama(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    Cfg *c = &m->c;
    int H = c->n_heads, n_kv = c->n_kv_heads, D = c->hidden;
    int hd = D / H;
    double ta0 = now_s();
    float *Q = calloc((int64_t)S * H * hd, 4);
    float *K = calloc((int64_t)S * n_kv * hd, 4);
    float *V = calloc((int64_t)S * n_kv * hd, 4);
    for(int s = 0; s < S; s++){
        const float *xs = x + (int64_t)s * D;
        int pos = pos_base + s;
        matmul_qt(Q + (int64_t)s * H * hd, xs, &l->q_proj, 1);
        matmul_qt(K + (int64_t)s * n_kv * hd, xs, &l->k_proj, 1);
        matmul_qt(V + (int64_t)s * n_kv * hd, xs, &l->v_proj, 1);
        for(int h = 0; h < H; h++)
            rope_llama(Q + (int64_t)s * H * hd + (int64_t)h * hd, pos, hd, c->rope_freq);
        for(int h = 0; h < n_kv; h++)
            rope_llama(K + (int64_t)s * n_kv * hd + (int64_t)h * hd, pos, hd, c->rope_freq);
        memcpy(m->Lc[layer] + (int64_t)pos * n_kv * hd, K + (int64_t)s * n_kv * hd, (size_t)n_kv * hd * 4);
        memcpy(m->Rc[layer] + (int64_t)pos * n_kv * hd, V + (int64_t)s * n_kv * hd, (size_t)n_kv * hd * 4);
    }
    int st = m->kv_start[layer];
    float *ctx = calloc((int64_t)S * D, 4);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s = 0; s < S; s++)
    for(int h = 0; h < H; h++){
        int kv_h = h % n_kv;
        const float *q = Q + (int64_t)s * H * hd + (int64_t)h * hd;
        int nt = pos_base + s + 1 - st;
        float sc[8192];
        for(int t = st; t < st + nt; t++){
            const float *kp = m->Lc[layer] + (int64_t)t * n_kv * hd + (int64_t)kv_h * hd;
            float a = 0;
            for(int d = 0; d < hd; d++) a += q[d] * kp[d];
            sc[t - st] = a * c->attn_scale;
        }
        softmax(sc, nt);
        float *cx = ctx + (int64_t)s * D + (int64_t)h * hd;
        memset(cx, 0, (size_t)hd * 4);
        for(int t = st; t < st + nt; t++){
            const float *vp = m->Rc[layer] + (int64_t)t * n_kv * hd + (int64_t)kv_h * hd;
            float a = sc[t - st];
            for(int d = 0; d < hd; d++) cx[d] += a * vp[d];
        }
    }
    matmul_qt(out, ctx, &l->o, S);
    free(Q); free(K); free(V); free(ctx);
    m->t_attn += now_s() - ta0;
}

/* ── DSA score comparator ────────────────────────────── */
static int cmp_fdesc(const void *a, const void *b){
    float fa=*(const float*)a, fb=*(const float*)b;
    return (fa>fb)?-1:(fa<fb)?1:0;
}

void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out){
    if(IS_LLAMA(m->c)){ attention_llama(m,l,layer,x,S,pos_base,out); return; }
    Cfg *c=&m->c; int H=c->n_heads, D=c->hidden, qh=c->qk_head, vh=c->v_head;
    int kvb_dim=H*(c->qk_nope+vh), Tk=pos_base+S;
    double ta0=now_s();
    float *ctx=calloc((int64_t)S*H*vh,4);
    float *Q=calloc((int64_t)S*H*qh,4);
    float *QR=calloc((int64_t)S*c->q_lora,4), *comp=calloc(c->kv_lora+c->qk_rope,4);
    for(int s=0;s<S;s++){
        const float *xs=x+(int64_t)s*D; int pos=pos_base+s;
        float *qresid=QR+(int64_t)s*c->q_lora;
        matmul_qt(qresid, xs, &l->q_a, 1);
        rmsnorm(qresid, qresid, l->q_a_ln, c->q_lora, c->eps);
        float *qfull=Q+(int64_t)s*H*qh; matmul_qt(qfull, qresid, &l->q_b, 1);
        for(int h=0;h<H;h++) rope_interleave(qfull+(int64_t)h*qh+c->qk_nope, pos, c);
        matmul_qt(comp, xs, &l->kv_a, 1);
        float *Ldst=m->Lc[layer]+(int64_t)pos*c->kv_lora, *Rdst=m->Rc[layer]+(int64_t)pos*c->qk_rope;
        memcpy(Ldst, comp, c->kv_lora*sizeof(float));
        rmsnorm(Ldst, Ldst, l->kv_a_ln, c->kv_lora, c->eps);
        memcpy(Rdst, comp+c->kv_lora, c->qk_rope*sizeof(float));
        rope_interleave(Rdst, pos, c);
    }
    const int *dsel=NULL, *dnsel=NULL; int dtopk=0;
    if(m->has_dsa && layer<c->n_layers && m->kv_start[layer]==0){
        int nh=c->index_nh, hd=c->index_hd; dtopk=c->index_topk;
        if(c->idx_type[layer]){
            for(int s=0;s<S;s++){
                const float *xs=x+(int64_t)s*D; int pos=pos_base+s;
                float *kd=m->Ic[layer]+(int64_t)pos*hd;
                matmul_qt(kd, xs, &m->ix_wk[layer], 1);
                layernorm(kd, m->ix_knw[layer], m->ix_knb[layer], hd, 1e-6f);
                rope_interleave(kd, pos, c);
            }
            if((int64_t)S*dtopk > m->dsa_scap){
                free(m->dsa_sel); free(m->dsa_nsel);
                m->dsa_scap=(int64_t)S*dtopk;
                m->dsa_sel=malloc((size_t)m->dsa_scap*sizeof(int));
                m->dsa_nsel=malloc((size_t)S*sizeof(int));
            }
            #pragma omp parallel for schedule(dynamic,1)
            for(int s=0;s<S;s++){
                int pos=pos_base+s, nk=pos+1;
                if(nk<=dtopk && !g_dsa_force){ m->dsa_nsel[s]=0; continue; }
                int keep = nk<dtopk ? nk : dtopk;
                float *qi=calloc((int64_t)nh*hd,4);
                matmul_qt(qi, QR+(int64_t)s*c->q_lora, &m->ix_wq[layer], 1);
                for(int h=0;h<nh;h++) rope_interleave(qi+(int64_t)h*hd, pos, c);
                float w32[64];
                matmul_qt(w32, x+(int64_t)s*D, &m->ix_wp[layer], 1);
                float wsc=1.f/sqrtf((float)nh), rs=1.f/sqrtf((float)hd);
                float *isc=calloc(nk,4);
                for(int t=0;t<nk;t++){
                    const float *kt=m->Ic[layer]+(int64_t)t*hd;
                    float a=0;
                    for(int h=0;h<nh;h++){ const float *qhp=qi+(int64_t)h*hd;
                        float d0=0; for(int i=0;i<hd;i++) d0+=qhp[i]*kt[i];
                        d0*=rs; if(d0>0) a+=w32[h]*d0;
                    }
                    isc[t]=a*wsc;
                }
                float *tmp=calloc(nk,4); memcpy(tmp,isc,nk*sizeof(float));
                qsort(tmp,nk,sizeof(float),cmp_fdesc);
                float thr=tmp[keep-1];
                int *dst=m->dsa_sel+(int64_t)s*dtopk, nd=0;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]>thr) dst[nd++]=t;
                for(int t=0;t<nk && nd<keep;t++) if(isc[t]==thr) dst[nd++]=t;
                m->dsa_nsel[s]=nd;
                free(qi); free(isc); free(tmp);
            }
        }
        if(m->dsa_nsel){ dsel=m->dsa_sel; dnsel=m->dsa_nsel; }
    }
    int absorb = g_absorb==1 || (g_absorb<0 && S<=4);
    if(absorb && c->kv_lora<=512){
        int kvl=c->kv_lora, r0v=c->qk_nope;
        #pragma omp parallel for collapse(2) schedule(static)
        for(int s=0;s<S;s++) for(int h=0;h<H;h++){
            int pos=pos_base+s;
            const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
            const float *qr=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[512]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) qt_addrow(&l->kv_b, rbase+d, qp[d], qabs);
            float sc[8192];
            int st0=m->kv_start[layer];
            int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;
            const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
            int nt = ns ? ns : pos+1-st0;
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
                float a=0; for(int i=0;i<kvl;i++) a+=qabs[i]*Lt[i];
                for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
                sc[jj]=a*c->attn_scale;
            }
            softmax(sc,nt);
            float clat[512]; memset(clat,0,kvl*sizeof(float));
            for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
                const float *Lt=m->Lc[layer]+(int64_t)t*kvl;
                float a=sc[jj]; for(int i=0;i<kvl;i++) clat[i]+=a*Lt[i]; }
            qt_matvec_rows(&l->kv_b, rbase+r0v, vh, clat, ctx+((int64_t)s*H+h)*vh);
        }
        matmul_qt(out, ctx, &l->o, S);
        free(ctx); free(Q); free(QR); free(comp);
        m->t_attn += now_s()-ta0;
        return;
    }
    double tk0=now_s();
    int stL=m->kv_start[layer];
    float *kvb_all=calloc((int64_t)Tk*kvb_dim,4);
    matmul_qt(kvb_all+(int64_t)stL*kvb_dim, m->Lc[layer]+(int64_t)stL*c->kv_lora, &l->kv_b, Tk-stL);
    m->t_kvb += now_s()-tk0;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        int pos=pos_base+s;
        const float *qp=Q+(int64_t)s*H*qh+(int64_t)h*qh;
        const float *qr=qp+c->qk_nope;
        float sc[8192];
        int st0=m->kv_start[layer];
        int ns = (dnsel && dnsel[s]>0) ? dnsel[s] : 0;
        const int *tlist = ns ? dsel+(int64_t)s*dtopk : NULL;
        int nt = ns ? ns : pos+1-st0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *kn=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh);
            const float *kr=m->Rc[layer]+(int64_t)t*c->qk_rope;
            float a=0; for(int d=0;d<c->qk_nope;d++) a+=qp[d]*kn[d];
            for(int d=0;d<c->qk_rope;d++) a+=qr[d]*kr[d];
            sc[jj]=a*c->attn_scale;
        }
        softmax(sc,nt);
        float *cx=ctx+((int64_t)s*H+h)*vh; for(int d=0;d<vh;d++) cx[d]=0;
        for(int jj=0;jj<nt;jj++){ int t = tlist ? tlist[jj] : st0+jj;
            const float *vv=kvb_all+(int64_t)t*kvb_dim+(int64_t)h*(c->qk_nope+vh)+c->qk_nope;
            float a=sc[jj]; for(int d=0;d<vh;d++) cx[d]+=a*vv[d]; }
    }
    matmul_qt(out, ctx, &l->o, S);
    free(ctx); free(Q); free(QR); free(comp); free(kvb_all);
    m->t_attn += now_s()-ta0;
}
