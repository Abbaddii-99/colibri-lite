#define _GNU_SOURCE
#include "model.h"
#include "tensor.h"
#include <math.h>
#include <string.h>

void rmsnorm(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

void layernorm(float *v, const float *w, const float *b, int n, float eps){
    double mu=0; for(int i=0;i<n;i++) mu+=v[i]; mu/=n;
    double var=0; for(int i=0;i<n;i++){ double d=v[i]-mu; var+=d*d; } var/=n;
    float r=1.f/sqrtf((float)var+eps);
    for(int i=0;i<n;i++) v[i]=((float)(v[i]-mu))*r*w[i]+b[i];
}

void softmax(float *x,int n){ float m=-1e30f; for(int i=0;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-m);s+=x[i];} for(int i=0;i<n;i++) x[i]/=s; }

void rope_interleave(float *v, int pos, const Cfg *c){
    int sz=c->qk_rope; int half=sz/2;
    float stk[256]; float *in=sz<=256?stk:calloc(sz,4);
    memcpy(in,v,sz*4);
    for(int j=0;j<half;j++){
        float ang = pos*c->rope_freq[j], cs=cosf(ang), sn=sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        v[j]      = a*cs - b*sn;
        v[half+j] = b*cs + a*sn;
    }
    if(sz>256) free(in);
}

void qt_addrow(const QT *t, int row, float coef, float *acc){
    int I=t->I;
    if(t->fmt==0){ const float *w=t->qf+(int64_t)row*I; for(int i=0;i<I;i++) acc[i]+=coef*w[i]; return; }
    if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*I; float sc=t->s[row];
        for(int i=0;i<I;i++) acc[i]+=coef*sc*w[i]; return; }
    int rb=(I+1)/2;
    const uint8_t *w=t->q4+(int64_t)row*rb; float sc=t->s[row];
    for(int i=0;i<I;i++){ int v=(i&1)?(w[i>>1]>>4):(w[i>>1]&0xF); acc[i]+=coef*sc*(float)(v-8); }
}

void qt_matvec_rows(const QT *t, int r0, int n, const float *x, float *y){
    for(int r=0;r<n;r++){ int row=(r0+r)%t->O;
        if(t->fmt==0){ const float *w=t->qf+(int64_t)row*t->I; float a=0; for(int i=0;i<t->I;i++) a+=x[i]*w[i]; y[r]=a; }
        else if(t->fmt==1){ const int8_t *w=t->q8+(int64_t)row*t->I; float sc=t->s[row]; float a=0;
            for(int i=0;i<t->I;i++) a+=x[i]*w[i]; y[r]=a*sc; }
        else{ int rb=(t->I+1)/2; const uint8_t *w=t->q4+(int64_t)row*rb; float sc=t->s[row]; float a=0;
            for(int i=0;i<t->I;i++){ int v=(i&1)?(w[i>>1]>>4):(w[i>>1]&0xF); a+=x[i]*(float)(v-8); } y[r]=a*sc; }
    }
}
