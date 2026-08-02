#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── rmsnorm ────────────────────────────────────────────── */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps){
    float s = 0.0f;
    for(int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / n + eps);
    for(int i = 0; i < n; i++) o[i] = w[i] * (s * x[i]);
}

/* ── F32 matmul ─────────────────────────────────────────── */
static void matmul_f32(float *out, const float *in, const float *w, int O, int I){
    #pragma omp parallel for
    for(int o = 0; o < O; o++){
        float sum = 0.0f;
        for(int i = 0; i < I; i++) sum += in[i] * w[(int64_t)o * I + i];
        out[o] = sum;
    }
}

/* ── INT8 matmul (weights pre-quantized) ───────────────── */
static void matmul_i8(float *out, const float *in, const int8_t *w, const float *s, int O, int I){
    #pragma omp parallel for
    for(int o = 0; o < O; o++){
        int sum = 0;
        for(int i = 0; i < I; i++) sum += (int)in[i] * w[(int64_t)o * I + i];
        out[o] = sum * s[o];
    }
}

/* ── INT8 dot-product (input also quantized) ─────────────── */
static inline float qrow_i8(const float *x, int8_t *q, int I){
    float amax=0; for(int i=0;i<I;i++){float a=fabsf(x[i]);if(a>amax)amax=a;}
    float s=amax/127.f; if(s<1e-12f)s=1e-12f; float inv=1.f/s;
    for(int i=0;i<I;i++) q[i]=(int8_t)lrintf(x[i]*inv);
    return s;
}
static void matmul_i8_idot(float *out, const float *in, const int8_t *w, const float *s, int O, int I){
    int8_t *xq=malloc(I); float sx=qrow_i8(in,xq,I); sx=1.f/(sx?sx:1.f);
    #pragma omp parallel for
    for(int o = 0; o < O; o++){
        int sum = 0;
        for(int i = 0; i < I; i++) sum += (int)xq[i] * w[(int64_t)o * I + i];
        out[o] = (float)sum * s[o] * sx;
    }
    free(xq);
}

/* ── INT4 matmul (packed) ────────────────────────────────── */
static void matmul_i4(float *out, const float *in, const uint8_t *w4, const float *s, int O, int I){
    int rb=(I+1)/2;
    #pragma omp parallel for
    for(int o = 0; o < O; o++){
        const uint8_t *w = w4 + (int64_t)o * rb;
        int sum = 0; int i = 0;
        for(; i+1 < I; i+=2){ uint8_t b=w[i>>1]; sum+=((int)(b&0xF)-8)*lrintf(in[i])+((int)(b>>4)-8)*lrintf(in[i+1]); }
        if(i<I){ uint8_t b=w[i>>1]; sum+=((int)(b&0xF)-8)*lrintf(in[i]); }
        out[o] = (float)sum * s[o];
    }
}

/* ── Bench: rmsnorm ─────────────────────────────────────── */
static void bench_rmsnorm(void){
    int dims[] = {1024, 2048, 4096, 8192, 16384};
    int ndims = sizeof(dims) / sizeof(dims[0]), reps = 50000;

    printf("\n-- rmsnorm throughput --\n");
    printf("%-8s %10s %15s %10s\n", "Dim", "Reps", "Time(ns/call)", "GB/s");
    printf("------------------------------------------------\n");

    for(int d = 0; d < ndims; d++){
        int D = dims[d];
        float *x = malloc(D * 4), *w = malloc(D * 4), *o = malloc(D * 4);
        for(int i = 0; i < D; i++){ x[i] = 1.0f; w[i] = 1.0f; }

        double t0 = now_s();
        for(int r = 0; r < reps; r++) rmsnorm(o, x, w, D, 1e-6f);
        double dt = now_s() - t0;
        volatile float sink = o[0]; (void)sink;

        double ns = dt / reps * 1e9;
        double gbs = (double)D * 4 * 3 / ns;
        printf("%-8d %10d %15.1f %10.1f\n", D, reps, ns, gbs);
        free(x); free(w); free(o);
    }
}

/* ── Bench: matmul variants ────────────────────────────── */
typedef struct { int O, I; const char *name; } task_t;
static const task_t tasks[] = {
    {4096, 4096,    "Q·K^T  4096×4096"},
    {4096, 14336,   "gate   4096×14336"},
    {14336, 4096,   "down   14336×4096"},
    {1024, 65536,   "embed  1024×65536"},
    {65536, 1024,   "unembed 65536×1024"},
};
static const int ntasks = sizeof(tasks) / sizeof(tasks[0]);
static const int reps = 100;

typedef void (*matmul_fn)(float*,const float*,const void*,const float*,int,int);
static void bench_one(const char *label, matmul_fn fn,
                      const void *w, const float *s, int O, int I){
    float *in = malloc(I * 4), *out = malloc(O * 4);
    for(int i = 0; i < I; i++) in[i] = (float)rand() / RAND_MAX * 2 - 1;

    double t0 = now_s();
    for(int r = 0; r < reps; r++) fn(out, in, w, s, O, I);
    double dt = now_s() - t0;
    volatile float sink = out[0]; (void)sink;

    double ms = dt / reps * 1000;
    double ops = 2.0 * O * I;
    double gflops = ops / (dt / reps) * 1e-9;
    double bytes_ = ((double)O * I + I + O) * 4;
    double gbs = bytes_ / (dt / reps) * 1e-9;
    printf("%-22s %10d %15.2f %12.1f %10.1f\n", label, reps, ms, gflops, gbs);

    free(in); free(out);
}

/* Wrappers to match matmul_fn signature */
static void wrap_f32(float *o, const float *x, const void *w, const float *s, int O, int I){
    (void)s; matmul_f32(o, x, (const float*)w, O, I);
}
static void wrap_i8(float *o, const float *x, const void *w, const float *s, int O, int I){
    matmul_i8(o, x, (const int8_t*)w, s, O, I);
}
static void wrap_i4(float *o, const float *x, const void *w, const float *s, int O, int I){
    matmul_i4(o, x, (const uint8_t*)w, s, O, I);
}
static void wrap_i8idot(float *o, const float *x, const void *w, const float *s, int O, int I){
    matmul_i8_idot(o, x, (const int8_t*)w, s, O, I);
}

static void bench_matmul(void){
    printf("\n-- F32 matmul throughput --\n");
    printf("%-22s %10s %15s %12s %10s\n", "Format/Operation", "Reps", "Time(ms/call)", "GFLOPS", "GB/s");
    printf("------------------------------------------------------------------\n");
    for(int t = 0; t < ntasks; t++){
        int O = tasks[t].O, I = tasks[t].I;
        float *wf = malloc((size_t)O * I * 4);
        for(int i = 0; i < O * I; i++) wf[i] = (float)rand() / RAND_MAX * 2 - 1;

        /* Quantize to int8 */
        int8_t *w8 = malloc((size_t)O * I); float *s8 = malloc(O * 4);
        for(int o = 0; o < O; o++){
            float amax = 0; for(int i = 0; i < I; i++){ float a = fabsf(wf[(int64_t)o*I+i]); if(a > amax) amax = a; }
            s8[o] = amax / 127.f; if(s8[o] < 1e-12f) s8[o] = 1e-12f;
            for(int i = 0; i < I; i++) w8[(int64_t)o*I+i] = (int8_t)lrintf(wf[(int64_t)o*I+i] / s8[o]);
        }
        /* Pack to int4 */
        int rb = (I+1)/2;
        uint8_t *w4 = malloc((size_t)O * rb); float *s4 = malloc(O * 4);
        for(int o = 0; o < O; o++){
            float amax = 0; for(int i = 0; i < I; i++){ float a = fabsf(wf[(int64_t)o*I+i]); if(a > amax) amax = a; }
            s4[o] = amax / 7.f; if(s4[o] < 1e-12f) s4[o] = 1e-12f;
            for(int i = 0; i < I; i+=2){
                int v0 = (int)lrintf(wf[(int64_t)o*I+i] / s4[o]); if(v0 > 7) v0 = 7; if(v0 < -8) v0 = -8;
                int v1 = (i+1 < I) ? (int)lrintf(wf[(int64_t)o*I+i+1] / s4[o]) : 0;
                if(v1 > 7) v1 = 7; if(v1 < -8) v1 = -8;
                w4[(int64_t)o*rb + (i>>1)] = (uint8_t)((v0+8) | ((v1+8)<<4));
            }
        }

        char label[64];
        snprintf(label, sizeof(label), "F32  %s", tasks[t].name);
        bench_one(label, wrap_f32, wf, NULL, O, I);

        snprintf(label, sizeof(label), "INT8 %s", tasks[t].name);
        bench_one(label, wrap_i8, w8, s8, O, I);

        snprintf(label, sizeof(label), "INT4 %s", tasks[t].name);
        bench_one(label, wrap_i4, w4, s4, O, I);

        snprintf(label, sizeof(label), "I8dot%s", tasks[t].name);
        bench_one(label, wrap_i8idot, w8, s8, O, I);

        free(wf); free(w8); free(s8); free(w4); free(s4);
    }
}

int main(int argc, char **argv){
    const char *mode = argc > 1 ? argv[1] : "all";

    printf("Colibri-Lite Math Benchmark\n");
    printf("============================\n");
#ifdef _OPENMP
    printf("OpenMP: %d threads\n", omp_get_max_threads());
#else
    printf("OpenMP: not available\n");
#endif
    printf("CPU: ");
#ifdef __x86_64__
    printf("x86_64");
#elif defined(__aarch64__)
    printf("ARM64");
#else
    printf("unknown");
#endif
    printf("\n");

    srand(42);

    int do_all = !strcmp(mode, "all");
    if(do_all || !strcmp(mode, "rmsnorm")) bench_rmsnorm();
    if(do_all || !strcmp(mode, "matmul")) bench_matmul();

    printf("\nDone.\n");
    return 0;
}
