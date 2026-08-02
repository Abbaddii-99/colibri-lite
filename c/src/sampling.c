#define _GNU_SOURCE
#include "model.h"
#include "sampling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int cmp_pdesc(const void *a, const void *b){
    int ia=*(const int*)a, ib=*(const int*)b;
    float fa=g_pbuf[ia], fb=g_pbuf[ib];
    return (fa>fb)?-1:(fa<fb)?1:0;
}

void dist_build(const float *lo, int V){
    float temp=g_temp;
    if(temp<0) temp=1.0f;
    free(g_pbuf); g_pbuf=malloc(V*sizeof(float));
    free(g_pidx); g_pidx=malloc(V*sizeof(int));
    float mx=-1e30f; for(int i=0;i<V;i++) if(lo[i]>mx) mx=lo[i];
    float sum=0; for(int i=0;i<V;i++){ float v=expf((lo[i]-mx)/temp); g_pbuf[i]=v; sum+=v; }
    float inv=1.f/sum; for(int i=0;i<V;i++) g_pbuf[i]*=inv;
    int nnuc=0;
    if(g_nuc<1.0f){
        for(int i=0;i<V;i++) g_pidx[nnuc++]=i;
        qsort(g_pidx,nnuc,sizeof(int),cmp_pdesc);
        float acc=0; int ncut=0;
        while(ncut<nnuc && acc<g_nuc){ acc+=g_pbuf[g_pidx[ncut]]; ncut++; }
        nnuc=ncut;
    } else {
        for(int i=0;i<V;i++) g_pidx[nnuc++]=i;
    }
    /* keep nnuc in g_pidx, rest discarded */
    for(int i=nnuc;i<V;i++) g_pidx[i]=-1;
}

int dist_sample(int V, int ban){
    int n=0; for(int i=0;i<V;i++) if(g_pidx[i]>=0 && g_pidx[i]!=ban) n++;
    if(n==0) return ban;
    float r=(float)(g_rng>>33)/8388608.0f; g_rng=g_rng*0x9E3779B97F4A7C15ULL+1;
    float cum=0; for(int i=0;i<V;i++){ if(g_pidx[i]<0 || g_pidx[i]==ban) continue;
        cum+=g_pbuf[g_pidx[i]]; if(r<=cum) return g_pidx[i]; }
    return g_pidx[0];
}

int pick_tok(const float *lo, int V, int ban){
    if(g_nuc<0.99f || g_temp>0){
        dist_build(lo, V);
        return dist_sample(V, ban);
    }
    float mx=-1e30f; int mid=0;
    for(int i=0;i<V;i++) if(lo[i]>mx){ mx=lo[i]; mid=i; }
    if(g_nstop){
        int banned=0; for(int i=0;i<g_nstop;i++) if(mid==g_stop[i]){ banned=1; break; }
        if(banned){
            dist_build(lo, V);
            return dist_sample(V, ban);
        }
    }
    return mid;
}

void stops_arm(const Cfg *c, int tok_eos){
    g_nstop=0; int seen=0;
    for(int i=0;i<c->n_stop;i++){
        int id=c->stop_ids[i];
        if(id==tok_eos && !seen){ seen=1; continue; }
        g_stop[g_nstop++]=id;
    }
}
