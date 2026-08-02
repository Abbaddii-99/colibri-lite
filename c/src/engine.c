#define _GNU_SOURCE
#include "model.h"
#include "engine.h"
#include "load.h"
#include "quant.h"
#include "tensor.h"
#include "cache.h"
#include "kv_cache.h"
#include "pipeline.h"
#include "spec.h"
#include "sampling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <omp.h>

/* ── forward_all: teacher-forcing ────────────────────── */
void forward_all(Model *m, const int *ids, int S, int *pred){
    Cfg *c=&m->c; int D=c->hidden;
    kv_alloc(m,S);
    float *x=calloc((int64_t)S*D,4);
    for(int s=0;s<S;s++) embed_row(m, ids[s], x+(int64_t)s*D);
    layers_forward(m,x,S,0);
    float *lo=calloc(c->vocab,4);
    for(int s=0;s<S;s++){
        float row[8192]; rmsnorm(row, x+(int64_t)s*D, m->final_norm, D, c->eps);
        matmul_qt(lo, row, &m->lm_head, 1);
        int best=0; float bv=lo[0]; for(int i=1;i<c->vocab;i++) if(lo[i]>bv){bv=lo[i];best=i;}
        pred[s]=best;
    }
    free(x); free(lo);
}

/* ── generate: prefill + speculative decode ──────────── */
void generate(Model *m, const int *prompt, int np, int n_new, int *out){
    kv_alloc(m,np+n_new+g_draft+2);
    for(int i=0;i<np;i++) out[i]=prompt[i];
    float *logit=step(m,prompt,np,0);
    int kv_out=0;
    spec_decode(m,out,np,n_new,-1,logit,out+np,&kv_out);
}

/* ── profile_print ───────────────────────────────────── */
void profile_print(Model *m, double elapsed){
    double accounted=m->t_edisk+m->t_emm+m->t_attn+m->t_head;
    printf("PROFILE: expert-disk %.3fs | expert-matmul %.3fs | attention %.3fs "
           "(including kvb %.3fs) | lm_head %.3fs | other %.3fs\n",
        m->t_edisk,m->t_emm,m->t_attn,m->t_kvb,m->t_head,elapsed-accounted);
}

/* ── run_replay: fixed-token decode benchmark ────────── */
void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0;
    m->t_edisk=m->t_emm=m->t_attn=m->t_kvb=m->t_head=0;
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i); free(logit); steps++;
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
}

/* ── run_text: real text generation ──────────────────── */
void run_text(Model *m, const char *snap, const char *prompt_text, int ngen){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);
    if(g_temp<0) g_temp=0.7f;
    int cap=(int)strlen(prompt_text)+16; int *pids=malloc(cap*sizeof(int));
    int np=tok_encode(&T,prompt_text,(int)strlen(prompt_text),pids,cap);
    if(np<1){ fprintf(stderr,"prompt is empty after tokenization\n"); return; }
    printf("prompt: %d tokens | generating up to %d (EOS stop=%d) | n-gram draft=%d\n", np, ngen, eos, g_draft);
    fputs(prompt_text,stdout); fflush(stdout);
    kv_alloc(m, np+ngen+g_draft+2);
    int *all=malloc((np+ngen+g_draft+2)*sizeof(int)); memcpy(all,pids,np*sizeof(int));
    double t=now_s();
    float *logit=step(m,pids,np,0);
    grammar_reset();
    int kv_dummy=0;
    int produced=spec_decode(m,all,np,ngen,eos,logit,all+np,&kv_dummy);
    double dt=now_s()-t;
    double tot=m->hits+m->miss;
    printf("\n---\n%d tokens in %.2fs (%.2f tok/s) | expert hit rate %.1f%% | RSS %.2f GB\n",
        produced, dt, produced/dt, tot?100.0*m->hits/tot:0.0, rss_gb());
    profile_print(m,dt);
    free(pids); free(all);
    usage_save(m);
}

/* ── run_score: log-likelihood scoring ────────────────── */
static double logprob_target(const float *lo, int V, int target, int *am){
    float mx=lo[0]; int best=0; for(int i=1;i<V;i++){ if(lo[i]>mx){mx=lo[i];best=i;} }
    double se=0; for(int i=0;i<V;i++) se+=exp((double)lo[i]-mx);
    if(am)*am=(best==target);
    return (double)(lo[target]-mx) - log(se);
}

void run_score(Model *m, const char *path){
    Cfg *c=&m->c; int D=c->hidden;
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    int64_t maxT=1; { char *ln=NULL; size_t cp=0;
        while(getline(&ln,&cp,f)>0){ int a,b; if(sscanf(ln,"%d %d",&a,&b)==2 && a>0 && b>0 && a<INT_MAX/2 && b<INT_MAX/2 && (int64_t)a+b>maxT) maxT=(int64_t)a+b; }
        free(ln); }
    if(maxT>INT_MAX/4){ fprintf(stderr,"maxT too large: %lld\n",(long long)maxT); exit(1); }
    kv_alloc(m,maxT);
    float *x=calloc((int64_t)maxT*D,4), *lo=calloc(c->vocab,4), *row=calloc(D,4);
    int *ids=malloc(maxT*sizeof(int));
    rewind(f); char *ln=NULL; size_t cp=0; int nreq=0; double t0=now_s();
    while(getline(&ln,&cp,f)>0){
        char *p=ln; int ctxlen=strtol(p,&p,10), contlen=strtol(p,&p,10), T=ctxlen+contlen;
        if(T<=0||ctxlen<1){ printf("0 0 0\n"); fflush(stdout); continue; }
        for(int i=0;i<T;i++) ids[i]=strtol(p,&p,10);
        for(int s=0;s<T;s++) embed_row(m, ids[s], x+(int64_t)s*D);
        layers_forward(m,x,T,0);
        double lp=0; int greedy=1;
        for(int pos=ctxlen-1; pos<T-1; pos++){
            rmsnorm(row, x+(int64_t)pos*D, m->final_norm, D, c->eps);
            matmul_qt(lo,row,&m->lm_head,1);
            int am; lp += logprob_target(lo,c->vocab,ids[pos+1],&am); if(!am) greedy=0;
        }
        printf("%.6f %d %d\n", lp, contlen, greedy); fflush(stdout);
        if(++nreq%5==0) fprintf(stderr,"[score %d req | %.1fs | RSS %.2f GB]\n",
            nreq, now_s()-t0, rss_gb());
    }
    free(ln); free(ids); free(x); free(lo); free(row); fclose(f);
}

/* ── run_chat: interactive chat ───────────────────────── */
void run_chat(Model *m, const char *snap){
    char tkp[2048]; snprintf(tkp,sizeof(tkp),"%s/tokenizer.json",snap);
    Tok T; tok_load(&T,tkp);
    int eos=tok_id_of(&T,"<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);
    if(g_temp<0) g_temp=0.7f;
    int ngen = getenv("NGEN")?atoi(getenv("NGEN")):512;
    printf("Chat ready (EOS=%d, stop=%d tokens, NGEN=%d). Type your message.\n", eos, g_nstop, ngen);
    int cap=4096; int *hist=malloc(cap*sizeof(int)); int kv=0; double t_run=0; int total=0;
    for(;;){
        char line[4096]; if(!fgets(line,sizeof(line),stdin)) break;
        if(line[0]=='\n') continue;
        if(!strncmp(line,":reset",6)){ kv=0; fprintf(stderr,"[CHAT] memory reset\n"); continue; }
        if(!strncmp(line,":quit",5)) break;
        int nlen=(int)strlen(line)-1; if(nlen>0 && line[nlen]=='\n') line[nlen]=0;
        int *pids=malloc((nlen+16)*sizeof(int));
        int np;
        if(kv==0){
            np=tok_encode(&T,"<|system|>",-1,pids,cap);
            np+=tok_encode(&T,line,strlen(line),pids+np,cap-np);
        } else {
            np=tok_encode(&T,line,strlen(line),pids,cap);
        }
        if(kv+np+ngen+g_draft+2 > cap){
            int ncap=kv+np+ngen+g_draft+2+1024; int *tmp=realloc(hist,(size_t)ncap*sizeof(int));
            if(!tmp){ fprintf(stderr,"OOM realloc hist\n"); exit(1); }
            hist=tmp; cap=ncap;
        }
        for(int i=0;i<np;i++) hist[kv+i]=pids[i];
        kv_alloc(m,kv+np+ngen+g_draft+2);
        double t0=now_s();
        float *logit=step(m,hist+kv,np,kv);
        kv+=np;
        grammar_reset();
        int produced=spec_decode(m,hist,kv,ngen,eos,logit,hist+kv,&kv);
        double dt=now_s()-t0; t_run+=dt; total+=produced;
        fprintf(stderr,"[CHAT] %d tokens in %.2fs (%.2f tok/s) | avg %.2f tok/s | RSS %.2f GB\n",
            produced, dt, produced/dt, total/t_run, rss_gb());
        usage_save(m);
        free(pids);
    }
    free(hist);
}

/* ── Serve context (multi-slot KV cache) ────────────── */
typedef struct {
    KVState kv;
    int *hist, len, first;
} ServeCtx;

static void serve_ctx_init(Model *m, ServeCtx *s, const char *snap, int slot, int maxctx){
    s->kv.kv_start = calloc(m->c.n_layers + 1, sizeof(int));
    if(m->has_mtp) s->kv.kv_start[m->c.n_layers] = -1;
    kv_bind(m, &s->kv);
    kv_alloc(m, maxctx);
    s->hist = malloc(maxctx * sizeof(int));
    s->len = 0;
    s->first = 1;
    if(g_kvsave && snap){
        if(slot == 0)
            snprintf(s->kv.disk_path, sizeof(s->kv.disk_path), "%s/.coli_kv", snap);
        else
            snprintf(s->kv.disk_path, sizeof(s->kv.disk_path), "%s/.coli_kv.%d", snap, slot);
        s->len = kv_disk_load(m, s->hist, maxctx);
        if(s->len > 0) s->first = 0;
    }
}

static void serve_ctx_free(Model *m, ServeCtx *s){
    KVState *k = &s->kv;
    int NR = m->c.n_layers + 1;
    if(k->Lc) for(int i = 0; i < NR; i++){ free(k->Lc[i]); free(k->Rc[i]); }
    if(k->Ic) for(int i = 0; i < m->c.n_layers; i++) free(k->Ic[i]);
    free(k->Lc); free(k->Rc); free(k->Ic); free(k->kv_start); free(s->hist);
}

/* ── run_serve: OpenAI-compatible server mode ──────────── */
void run_serve(Model *m, const char *snap){
    fprintf(stderr, "[SERVE] server mode (SNAP=%s)\n", snap);
    char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
    Tok T; tok_load(&T, tkp);
    int eos = tok_id_of(&T, "<|endoftext|>");
    stops_arm(&m->c, eos);
    grammar_setup(&T);
    if(g_temp < 0) g_temp = 0.7f;
    int ngen = getenv("NGEN") ? atoi(getenv("NGEN")) : 256;
    int maxctx = getenv("CTX") ? atoi(getenv("CTX")) : 4096;
    int nctx = getenv("KV_SLOTS") ? atoi(getenv("KV_SLOTS")) : 1;
    if(nctx < 1 || nctx > 16){ fprintf(stderr, "KV_SLOTS must be between 1 and 16\n"); exit(2); }

    KVState *initial = m->kv;
    free(initial->kv_start); m->kv_start = NULL;
    free(initial); m->kv = NULL;

    ServeCtx *ctx = calloc(nctx, sizeof(ServeCtx));
    for(int i = 0; i < nctx; i++) serve_ctx_init(m, &ctx[i], snap, i, maxctx);
    int active = 0;
    ServeCtx *sc = &ctx[0];
    kv_bind(m, &sc->kv);

    fprintf(stderr, "[KV] context slots: %d x %d tokens, projected pool %.2f GB\n",
        nctx, maxctx, kv_pool_bytes(m, maxctx) / 1e9);

    printf("\x01\x01READY\x01\x01\n");
    printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
    fflush(stdout);

    char *line = NULL; size_t cap = 0; ssize_t nr;
    char *buf = malloc(1 << 16);
    while((nr = getline(&line, &cap, stdin)) > 0){
        while(nr > 0 && (line[nr - 1] == '\n' || line[nr - 1] == '\r')) line[--nr] = 0;
        if(nr == 0) continue;

        /* ── RESET ── */
        if(!strcmp(line, "\x01\x01RESET") || !strcmp(line, "\x02RESET")){
            sc->len = 0; sc->first = 1;
            if(m->has_mtp) m->kv_start[m->c.n_layers] = -1;
            kv_disk_reset(m);
            printf("\x01\x01END\x01\x01\n");
            printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
            fflush(stdout);
            continue;
        }

        /* ── MORE (continue truncated response) ── */
        if(!strcmp(line, "\x02MORE")){
            if(sc->len < 1){
                printf("\x01\x01END\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f\n", rss_gb());
                fflush(stdout);
                continue;
            }
            int cur = ngen;
            if(sc->len + cur + g_draft + 2 >= maxctx) cur = maxctx - sc->len - g_draft - 2;
            uint64_t h0 = m->hits, ms0 = m->miss;
            double tt0 = now_s();
            float *logit = step(m, sc->hist + sc->len - 1, 1, sc->len - 1);
            int prod = 0;
            if(cur > 0){
                prod = spec_decode(m, sc->hist, sc->len, cur, eos, logit, sc->hist + sc->len, &sc->len);
                for(int i = sc->len - prod; i < sc->len; i++){
                    char dec[64]; int dn = tok_decode(&T, sc->hist + i, 1, dec, 63); dec[dn] = 0;
                    fputs(dec, stdout);
                }
            } else {
                free(logit);
            }
            double tdt = now_s() - tt0;
            if(tdt < 1e-6) tdt = 1e-6;
            double dh = (double)(m->hits - h0), dm = (double)(m->miss - ms0);
            printf("\n\x01\x01END\x01\x01\n");
            printf("STAT %d %.2f %.1f %.2f\n", prod, prod / tdt,
                (dh + dm) > 0 ? 100.0 * dh / (dh + dm) : 0.0, rss_gb());
            fflush(stdout);
            kv_disk_append(m, sc->hist, sc->len);
            usage_save(m);
            continue;
        }

        /* ── PROMPT (binary length-prefixed protocol) ── */
        char *raw = NULL, *input = line;
        int input_n = (int)nr, raw_mode = 0, req_ngen = ngen, prompt_tokens = 0;
        float base_temp = g_temp, base_nuc = g_nuc;

        if(!strncmp(line, "\x02PROMPT ", 8)){
            unsigned long long nb = 0; double rt = 0, rp = 0; int slot = 0;
            int nf = sscanf(line + 8, "%llu %d %lf %lf %d", &nb, &req_ngen, &rt, &rp, &slot);
            if(nf < 4 || nb > (16u << 20) || req_ngen < 1 || rt < 0 || rt > 2 ||
               rp <= 0 || rp > 1 || slot < 0 || slot >= nctx){
                printf("\x01\x01END\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb());
                fflush(stdout);
                continue;
            }
            active = slot;
            sc = &ctx[active];
            kv_bind(m, &sc->kv);
            raw = malloc((size_t)nb + 1);
            if(!raw){ fprintf(stderr, "OOM raw prompt\n"); exit(1); }
            if(fread(raw, 1, (size_t)nb, stdin) != (size_t)nb){ free(raw); break; }
            int delim = fgetc(stdin);
            if(delim != '\n' && delim != EOF) ungetc(delim, stdin);
            if(memchr(raw, 0, (size_t)nb)){
                free(raw);
                printf("\x01\x01END\x01\x01\n");
                printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb());
                fflush(stdout);
                continue;
            }
            raw[nb] = 0;
            input = raw;
            input_n = (int)nb;
            raw_mode = 1;
            if(req_ngen > ngen) req_ngen = ngen;
            g_temp = (float)rt;
            g_nuc = (float)rp;
        } else {
            active = 0;
            sc = &ctx[0];
            kv_bind(m, &sc->kv);
        }

        /* ── Tokenize ── */
        int k = 0;
        const char *tk = getenv("THINK") && atoi(getenv("THINK")) ? "<think>" : "<think></think>";

        if(raw_mode){
            int *tmp = malloc(maxctx * sizeof(int));
            if(!tmp){ fprintf(stderr, "OOM raw tokens\n"); exit(1); }
            prompt_tokens = tok_encode(&T, input, input_n, tmp, maxctx - 8 - g_draft);
            int old_len = sc->len, prefix = 0;
            while(prefix < old_len && prefix < prompt_tokens && sc->hist[prefix] == tmp[prefix])
                prefix++;
            if(prefix < old_len){
                sc->len = prefix;
                if(m->has_mtp) m->kv_start[m->c.n_layers] = -1;
                kv_disk_truncate(m, sc->len);
            }
            k = prompt_tokens - sc->len;
            if(k > 0) memcpy(sc->hist + sc->len, tmp + sc->len, k * sizeof(int));
            fprintf(stderr, "[API] KV slot %d prefix %d/%d token, prefill %d\n",
                active, sc->len, prompt_tokens, k);
            free(tmp);
        } else {
            int bl = 0;
            if(sc->first)
                bl += snprintf(buf, (1 << 16) - bl, "[gMASK]<sop>");
            bl += snprintf(buf + bl, (1 << 16) - bl, "<|user|>%s<|assistant|>%s", input, tk);
            k = tok_encode(&T, buf, bl, sc->hist + sc->len, maxctx - sc->len);
            prompt_tokens = k;
            if(sc->len + k + 8 + g_draft >= maxctx){
                sc->len = 0;
                sc->first = 1;
                kv_disk_reset(m);
                bl = 0;
                bl += snprintf(buf + bl, (1 << 16) - bl, "<|user|>%s<|assistant|>%s", input, tk);
                k = tok_encode(&T, buf, bl, sc->hist, maxctx);
                if(k > maxctx - 8 - g_draft) k = maxctx - 8 - g_draft;
                prompt_tokens = k;
            }
        }

        if(prompt_tokens < 1){
            free(raw);
            g_temp = base_temp;
            g_nuc = base_nuc;
            printf("\x01\x01END\x01\x01\n");
            printf("STAT 0 0.00 0.0 %.2f 0 0\n", rss_gb());
            fflush(stdout);
            continue;
        }

        sc->first = 0;
        int cur = req_ngen;
        if(sc->len + k + cur + g_draft + 2 >= maxctx)
            cur = maxctx - sc->len - k - g_draft - 2;
        uint64_t h0 = m->hits, ms0 = m->miss;
        double tt0 = now_s();
        float *logit;
        if(k > 0){ logit = step(m, sc->hist + sc->len, k, sc->len); sc->len += k; }
        else logit = step(m, sc->hist + sc->len - 1, 1, sc->len - 1);

        int prod = 0;
        grammar_reset();
        if(cur > 0){
            prod = spec_decode(m, sc->hist, sc->len, cur, eos, logit, sc->hist + sc->len, &sc->len);
            for(int i = sc->len - prod; i < sc->len; i++){
                char dec[64]; int dn = tok_decode(&T, sc->hist + i, 1, dec, 63); dec[dn] = 0;
                fputs(dec, stdout);
            }
        } else {
            free(logit);
        }

        double tdt = now_s() - tt0;
        if(tdt < 1e-6) tdt = 1e-6;
        double dh = (double)(m->hits - h0), dm = (double)(m->miss - ms0);
        printf("\x01\x01END\x01\x01\n");
        printf("STAT %d %.2f %.1f %.2f %d %d\n", prod, prod / tdt,
            (dh + dm) > 0 ? 100.0 * dh / (dh + dm) : 0.0, rss_gb(),
            prompt_tokens, prod >= cur);
        fflush(stdout);
        free(raw);
        g_temp = base_temp;
        g_nuc = base_nuc;
        usage_save(m);
        kv_disk_append(m, sc->hist, sc->len);
    }
    free(line); free(buf);
    usage_save(m);
    for(int i = 0; i < nctx; i++) serve_ctx_free(m, &ctx[i]);
    free(ctx); m->kv = NULL; m->Lc = m->Rc = m->Ic = NULL;
    m->kv_start = NULL; m->max_t = 0;
}

/* ── RE-PIN (hot-store promotion) ────────────────────── */
int repin_pick(Model *m, RepinCand *out, int maxc){
    Cfg *c=&m->c; int nb=0;
    for(int l=0;l<c->n_layers;l++){
        if(!m->npin || m->npin[l]<1 || !m->eheat[l]) continue;
        ESlot *P=m->pin[l]; int ids[4096], zp, eu; long g;
        int np=m->npin[l]; if(np>4096) np=4096;
        for(int z=0;z<np;z++) ids[z]=P[z].eid;
        if(!tier_pick_swap(m->eheat[l],c->n_experts,ids,np,&zp,&eu,&g)) continue;
        if(nb<maxc){ out[nb].gain=g; out[nb].l=l; out[nb].slot=zp; out[nb].eid=eu; nb++; }
        else { int w=0; for(int b=1;b<maxc;b++) if(out[b].gain<out[w].gain) w=b;
               if(g>out[w].gain){ out[w].gain=g; out[w].l=l; out[w].slot=zp; out[w].eid=eu; } }
    }
    return nb;
}

void repin_pass(Model *m){
    if(g_repin<=0) return;
    if(m->n_emit - g_last_repin < (uint64_t)g_repin) return;
    g_last_repin = m->n_emit;
    RepinCand cd[4]; int nb=repin_pick(m,cd,4);
    for(int b=0;b<nb;b++){
        ESlot *s=&m->pin[cd[b].l][cd[b].slot];
        int old=s->eid;
        uint32_t old_heat=m->eheat[cd[b].l][old], new_heat=m->eheat[cd[b].l][cd[b].eid];
        double t0=now_s();
        expert_load(m,cd[b].l,cd[b].eid,s);
        fprintf(stderr,"[REPIN] layer %d: evict %d (heat=%u) <- admit %d (heat=%u) in %.0f ms\n",
            cd[b].l,old,old_heat,cd[b].eid,new_heat,(now_s()-t0)*1e3);
    }
    for(int l=0;l<m->c.n_layers;l++) if(m->eheat[l]) tier_decay(m->eheat[l],m->c.n_experts);
}

/* ── Stats persistence ───────────────────────────────── */
void stats_dump(Model *m, const char *path){
    /* Writes expert usage histogram to path */
    if(!path||!*path) return;
    FILE *f=fopen(path,"wb"); if(!f){perror(path);return;}
    for(int i=0;i<m->c.n_layers;i++){
        if(!m->L[i].sparse) continue;
        fprintf(f,"[layer %d]\n", i);
        uint32_t total=0;
        for(int e=0;e<m->c.n_experts;e++) if(m->eusage[i][e]) total+=m->eusage[i][e];
        int n=0; for(int e=0;e<m->c.n_experts;e++) if(m->eusage[i][e]) n++;
        int *idx=malloc(n*sizeof(int)); for(int e=0;e<n;e++) idx[e]=e;
        for(int a=0;a<n;a++) for(int b=a+1;b<n;b++)
            if(m->eusage[i][idx[b]]>m->eusage[i][idx[a]]){ int t=idx[a]; idx[a]=idx[b]; idx[b]=t; }
        int top = n<32 ? n : 32;
        for(int j=0;j<top;j++){
            int e=idx[j];
            fprintf(f,"  expert %4d: %5.1f%% (%u uses)\n", e,
                100.0*m->eusage[i][e]/total, m->eusage[i][e]);
        }
        free(idx);
    }
    fclose(f);
}

void usage_save(Model *m){
    if(!g_usage_path[0]) return;
    FILE *f=fopen(g_usage_path,"ab"); if(!f) return;
    for(int i=0;i<m->c.n_layers;i++){
        if(!m->L[i].sparse) continue;
        for(int e=0;e<m->c.n_experts;e++){
            uint32_t v=m->eheat[i][e] ? m->eheat[i][e] : m->eusage[i][e];
            if(v==0) continue;
            fwrite(&i,sizeof(int),1,f); fwrite(&e,sizeof(int),1,f); fwrite(&v,sizeof(uint32_t),1,f);
        }
    }
    fclose(f);
}

/* ── Memory management helpers ────────────────────────── */
static double mem_available_gb(void){
#ifdef __APPLE__
    mach_msg_type_number_t cnt=HOST_VM_INFO64_COUNT; vm_statistics64_data_t vm;
    if(host_statistics64(mach_host_self(),HOST_VM_INFO64,(host_info64_t)&vm,&cnt)==KERN_SUCCESS)
        return (double)vm.free_count*4096/1e9;
    return 16.0;
#elif defined(__linux__)
    FILE *f=fopen("/proc/meminfo","rb"); if(!f) return 16.0;
    char buf[4096]; size_t n=fread(buf,1,sizeof(buf)-1,f); buf[n]=0; fclose(f);
    const char *p=strstr(buf,"MemAvailable:"); if(!p) return 16.0;
    return strtod(p+13,NULL)/1e6;
#else
    return 16.0;
#endif
}

static int64_t expert_avail(Model *m, double ram_gb, int ebits, int est_ctx){
    Cfg *c=&m->c; if(!IS_GLM(*c) || c->moe_inter==0) return 0;
    double overhead = ram_gb>0 ? ram_gb : mem_available_gb();
    overhead = overhead*0.85 - m->resident_bytes/1e9;
    if(m->has_dsa) for(int i=0;i<c->n_layers;i++) if(c->idx_type[i])
        overhead -= (double)est_ctx*c->index_hd*4/1e9;
    double expert_gb = (double)c->moe_inter*3*(ebits/8)/1e9;
    return (int64_t)(overhead/expert_gb*1.05);
}

static void cap_for_ram(Model *m, double ram_gb, int ebits, int est_ctx){
    int64_t avail=expert_avail(m,ram_gb,ebits,est_ctx);
    if(avail<1) avail=1;
    if(avail<m->ecap){
        fprintf(stderr,"[RAM] trimming expert cache from %d to %lld per layer (%.2f GB available)\n",
            m->ecap, (long long)avail, ram_gb>0?ram_gb:mem_available_gb());
        m->ecap = (int)avail;
    }
}

static void pin_load(Model *m, const char *path, double pin_gb){
    Cfg *c=&m->c; if(!IS_GLM(*c) || c->moe_inter==0) return;
    FILE *f=fopen(path,"rb"); if(!f) return;
    int64_t expert_bytes = (int64_t)c->moe_inter*3*(m->ebits/8);
    int max_pin = (int)(pin_gb*1e9/expert_bytes);
    if(max_pin<1) max_pin=1;
    fprintf(stderr,"[PIN] loading up to %d pinned experts (%.2f GB)\n", max_pin, pin_gb);
    int *cnt=calloc(c->n_layers,sizeof(int));
    for(;;){
        int l,e; uint32_t v; if(fread(&l,4,1,f)!=1) break;
        if(fread(&e,4,1,f)!=1) break; if(fread(&v,4,1,f)!=1) break;
        if(l<0||l>=c->n_layers||e<0||e>=c->n_experts||cnt[l]>=max_pin) continue;
        if(!m->L[l].sparse) continue;
        if(!m->pin[l]){ m->pin[l]=calloc(max_pin,sizeof(ESlot)); m->npin[l]=0; }
        if(m->npin[l]>=max_pin) continue;
        ESlot *s=&m->pin[l][m->npin[l]++];
        s->eid=-1; expert_load(m,l,e,s);
        fprintf(stderr,"[PIN] layer %d expert %d pinned\n", l, e);
        cnt[l]++;
    }
    free(cnt); fclose(f);
}

static int usage_load(Model *m, const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    int64_t n=0;
    for(;;){
        int l,e; uint32_t v; if(fread(&l,4,1,f)!=1) break;
        if(fread(&e,4,1,f)!=1) break; if(fread(&v,4,1,f)!=1) break;
        if(l>=0&&l<m->c.n_layers&&e>=0&&e<m->c.n_experts&&m->eheat[l])
            m->eheat[l][e]=v;
        n++;
    }
    fclose(f);
    return n;
}

/* ── JSON array reader (for ref_glm.json oracle) ────── */
static int *read_arr(jval*o,const char*k,int*n){ jval*a=json_get(o,k);
    int*r=malloc((size_t)a->len*sizeof(int)); if(!r){ fprintf(stderr,"OOM read_arr\n"); exit(1); }
    for(int i=0;i<a->len;i++) r[i]=(int)a->kids[i]->num; *n=a->len; return r; }

/* ── main ────────────────────────────────────────────── */
int main(int argc, char **argv){
    if(!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE")){
        setenv("OMP_WAIT_POLICY","active",0);
        setenv("GOMP_SPINCOUNT","200000",0);
        setenv("OMP_PROC_BIND","close",0);
        setenv("OMP_DYNAMIC","FALSE",0);
        setenv("COLI_OMP_TUNED","1",1);
#ifdef __linux__
        fprintf(stderr,"[OMP] hot-thread tuning: re-exec once (COLI_NO_OMP_TUNE=1 to skip)\n");
        execv("/proc/self/exe", argv);
        perror("[OMP] execv self-reexec failed, running untuned");
#endif
    }
    const char *snap=getenv("SNAP");
    if(argc>1 && !strcmp(argv[1],"--help")){   /* explicit --help → success */
        fprintf(stderr,"Colibri-Lite — modular inference engine\n");
        fprintf(stderr,"Usage: SNAP=<model-dir> %s [cache=%d] [ebits=%d] [dbits=%d]\n", argv[0], 64, 8, 8);
        fprintf(stderr,"Env vars: SNAP, TEMP, NUCLEUS, TOPK, TOPP, DRAFT, SPEC, LOOKA, PILOT, PIPE,\n");
        fprintf(stderr,"          DIRECT, DROP, IDOT, ABSORB, SEED, NGEN, PROMPT, PIN, REPIN,\n");
        fprintf(stderr,"          CTX (max ctx), KV_SLOTS (1-16), SERVE (server mode), KVSAVE,\n");
        fprintf(stderr,"          THINK, CHAT_TEMPLATE, CUDA_DEVICES (comma-sep, e.g. \"0,1\")\n");
        return 0;
    }
    if(!snap){
        fprintf(stderr,"Colibri-Lite — modular inference engine\n");
        fprintf(stderr,"Usage: SNAP=<model-dir> %s [cache=%d] [ebits=%d] [dbits=%d]\n", argv[0], 64, 8, 8);
        fprintf(stderr,"Env vars: SNAP, TEMP, NUCLEUS, TOPK, TOPP, DRAFT, SPEC, LOOKA, PILOT, PIPE,\n");
        fprintf(stderr,"          DIRECT, DROP, IDOT, ABSORB, SEED, NGEN, PROMPT, PIN, REPIN,\n");
        fprintf(stderr,"          CTX (max ctx), KV_SLOTS (1-16), SERVE (server mode), KVSAVE,\n");
        fprintf(stderr,"          THINK, CHAT_TEMPLATE, CUDA_DEVICES (comma-sep, e.g. \"0,1\")\n");
        return 1;
    }
    g_nopack  = getenv("NOPACK")?1:0;
    g_drop    = getenv("DROP")?1:0;
    g_prefetch= getenv("PREFETCH")?atoi(getenv("PREFETCH")):0;
    g_topk    = getenv("TOPK")?atoi(getenv("TOPK")):0;
    g_topp    = getenv("TOPP")?atof(getenv("TOPP")):0;
    g_mlock   = getenv("MLOCK")?atoi(getenv("MLOCK")):-1;
    g_spec    = getenv("SPEC")?atoi(getenv("SPEC")):1;
    g_draft   = getenv("DRAFT")?atoi(getenv("DRAFT")):-1;
    g_looka   = getenv("LOOKA")?atoi(getenv("LOOKA")):0;
    g_pilot   = getenv("PILOT")?atoi(getenv("PILOT")):0;
    g_pilot_k = getenv("PILOT_K")?atoi(getenv("PILOT_K")):8;
    if(g_pilot_k<1) g_pilot_k=1;
    g_pipe    = getenv("PIPE")?atoi(getenv("PIPE")):0;
    g_pipe_nw = getenv("PIPE_WORKERS")?atoi(getenv("PIPE_WORKERS")):8;
    if(g_pipe_nw<1) g_pipe_nw=1; if(g_pipe_nw>16) g_pipe_nw=16;
    g_direct  = getenv("DIRECT")?atoi(getenv("DIRECT")):0;
    g_idot    = getenv("IDOT")?atoi(getenv("IDOT")):1;
    g_repin   = getenv("REPIN")?atoi(getenv("REPIN")):0;
    g_kvsave  = getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    g_absorb  = getenv("ABSORB")?atoi(getenv("ABSORB")):-1;
    g_dsa_force = getenv("DSA_FORCE")?atoi(getenv("DSA_FORCE")):0;
    g_temp    = getenv("TEMP")?atof(getenv("TEMP")):-1;
    g_nuc     = getenv("NUCLEUS")?atof(getenv("NUCLEUS")):0.90f;
    if(getenv("SEED")) g_rng = (uint64_t)atoll(getenv("SEED"))*0x9E3779B97F4A7C15ULL+1;
    else { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); g_rng ^= (uint64_t)ts.tv_nsec<<20 ^ (uint64_t)getpid(); }
    if(g_draft>63) g_draft=63;
    int cap   = argc>1?atoi(argv[1]):64;
    int ebits = argc>2?atoi(argv[2]):8;
    int dbits = argc>3?atoi(argv[3]):ebits;
#ifdef COLI_CUDA
    { const char *cd=getenv("CUDA_DEVICES");
      int devs[16], nd=0;
      if(cd){ char tmp[128]; strncpy(tmp,cd,127); tmp[127]=0;
          char *p=strtok(tmp,","); while(p && nd<16){ devs[nd++]=atoi(p); p=strtok(NULL,","); } }
      else devs[nd++]=0;
      if(coli_cuda_init(devs,nd)){ g_cuda_enabled=1; fprintf(stderr,"[CUDA] enabled: %d device(s)\n",nd); }
      else fprintf(stderr,"[CUDA] init failed — running CPU-only\n");
    }
#endif
    printf("== GLM C engine, cache=%d experts/layer | experts@%d-bit dense@%d-bit ==\n", cap, ebits, dbits);
    g_mem_avail_boot = mem_available_gb();
    Model m; double t0=now_s(); model_init(&m,snap,cap,ebits,dbits);
    if(getenv("MTP") && atoi(getenv("MTP"))==0) m.has_mtp=0;
    if(getenv("DSA") && atoi(getenv("DSA"))==0) m.has_dsa=0;
    if(g_draft<0) g_draft = m.has_mtp ? 3 : 0;
    printf("loaded in %.2fs | resident dense: %.2f MB | layers=%d experts=%d | MTP %s (draft=%d)\n",
           now_s()-t0, m.resident_bytes/(1024.0*1024.0), m.c.n_layers, m.c.n_experts,
           m.has_mtp?"ACTIVE":"absent", g_draft);
    if(getenv("PIN")) pin_load(&m, getenv("PIN"), getenv("PIN_GB")?atof(getenv("PIN_GB")):10.0);
    { double ram_env = getenv("RAM_GB")?atof(getenv("RAM_GB")):0.0;
      int est_ctx = getenv("CTX")?atoi(getenv("CTX")):4096;
      snprintf(g_usage_path,sizeof(g_usage_path),"%s/.coli_usage",snap);
      int64_t hist = usage_load(&m,g_usage_path);
      if(hist>0) fprintf(stderr,"[USAGE] expert history: %lld selections (%s)\n",(long long)hist,g_usage_path);
      int autopin = getenv("AUTOPIN")?atoi(getenv("AUTOPIN")):1;
      if(!getenv("PIN") && autopin && hist>=5000){
          double conf = (double)hist/200000.0; if(conf>1) conf=1;
          double pin_gb = expert_avail(&m,ram_env,ebits,est_ctx)*0.5*conf/1e9;
          if(pin_gb>=0.5) pin_load(&m, g_usage_path, pin_gb);
      }
      cap_for_ram(&m, ram_env, ebits, est_ctx); }
    const char *stats=getenv("STATS");
    if(getenv("SCORE")){ run_score(&m, getenv("SCORE")); if(stats) stats_dump(&m,stats); return 0; }
    if(getenv("SERVE")){ run_serve(&m, snap); if(stats) stats_dump(&m,stats); return 0; }
    if(getenv("PROMPT")){
        int ngen=getenv("NGEN")?atoi(getenv("NGEN")):64;
        run_text(&m, snap, getenv("PROMPT"), ngen);
        if(stats) stats_dump(&m,stats);
        return 0;
    }
    const char *refpath=getenv("REF")?getenv("REF"):"ref_glm.json";
    FILE *f=fopen(refpath,"rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long nf=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(nf+1); if(fread(b,1,nf,f)!=(size_t)nf){} b[nf]=0; fclose(f);
    char *ar=NULL; jval *ref=json_parse(b,&ar);
    int np,nfull; int *prompt=read_arr(ref,"prompt_ids",&np); int *full=read_arr(ref,"full_ids",&nfull);
    int n_new=nfull-np;
    { int maxid=0; for(int i=0;i<nfull;i++) if(full[i]>maxid) maxid=full[i];
      if(m.c.vocab>1000 && maxid<1000 && !getenv("REF_FORCE")){
        fprintf(stderr,"ERRORE: ref_glm.json e' l'oracolo del modello TINY\n"); return 1; } }
    if(getenv("REPLAY")){ run_replay(&m,full,nfull,np); if(stats) stats_dump(&m,stats); return 0; }
    if(getenv("TF")){
        int *tf=read_arr(ref,"tf_pred",&(int){0});
        int *pred=malloc(nfull*sizeof(int)); double tt=now_s();
        forward_all(&m, full, nfull, pred); double tdt=now_s()-tt;
        int ok=0; for(int i=0;i<nfull;i++) if(pred[i]==tf[i]) ok++;
        printf("PREFILL (teacher-forcing) C vs oracle: %d/%d positions | %.1f pos/s\n",ok,nfull,nfull/tdt);
        return 0;
    }
    int *out=malloc((np+n_new)*sizeof(int));
    double t=now_s(); generate(&m,prompt,np,n_new,out); double dt=now_s()-t;
    int match=0;
    printf("\nReference (oracle): "); for(int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nGLM C engine      : "); for(int i=np;i<nfull;i++){ printf("%d ", out[i]); if(out[i]==full[i])match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu) | RSS: %.2f GB | %.1f tok/s\n",
        (m.hits+m.miss)?100.0*m.hits/(m.hits+m.miss):0.0, (unsigned long long)m.hits, (unsigned long long)m.miss, rss_gb(), n_new/dt);
    profile_print(&m,dt);
    if(stats) stats_dump(&m,stats);
    return 0;
}
