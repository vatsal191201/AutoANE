// config.h — Model-agnostic structs, derived sizes, ANE init
// Model-specific dims come from models/*.h, selected via -DMODEL_HEADER
#pragma once
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <arm_neon.h>

// Include selected model config
// MODEL_HEADER is set by Makefile via -include models/xxx.h
#ifndef MODEL_NAME
#error "No model selected. Build with: make MODEL=qwen3_06b (or stories110m)"
#endif

// Derived weight sizes per layer (GQA-aware)
#define WQ_SZ (Q_DIM*DIM)
#define WK_SZ (KV_DIM*DIM)
#define WV_SZ (KV_DIM*DIM)
#define WO_SZ (DIM*Q_DIM)
#define W1_SZ (HIDDEN*DIM)
#define W2_SZ (DIM*HIDDEN)
#define W3_SZ (HIDDEN*DIM)
#define LAYER_PARAMS (WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)

// Attention score channels for SDPA backward
#define SCORE_CH (HEADS*SEQ)

// Per-layer weights
typedef struct {
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *W2, *W3;
    float *rms_att, *rms_ffn;
} LayerWeights;

// Adam optimizer state
typedef struct { float *m, *v; size_t n; } AdamState;
typedef struct {
    AdamState Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn;
} LayerAdam;

// Per-layer activations (saved for backward)
typedef struct {
    float *layer_in, *xnorm, *Q, *K, *V, *attn_out, *o_out;
    float *x2, *x2norm, *h1, *h3, *silu_out, *ffn_out;
} LayerActs;

// Per-layer gradients
typedef struct {
    float *Wq, *Wk, *Wv, *Wo, *W1, *W2, *W3, *rms_att, *rms_ffn;
} LayerGrads;

// ANE kernel handle
typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

// Per-layer IOSurfaces for pre-staged weights
typedef struct {
    // Fused kernels (legacy)
    IOSurfaceRef sdpaFwd_in, woFwd_in, ffnFused_in;
    // Unfused forward kernels (ANE matmul-only mode)
    IOSurfaceRef wqFwd_in, wkFwd_in, wvFwd_in;  // Wo uses woFwd_in
    IOSurfaceRef w1Fwd_in, w3Fwd_in, w2Fwd_in;
    // Backward kernels
    IOSurfaceRef ffnBwdW2t_in, ffnBwdW13t_in, wotBwd_in, qBwd_in, kvBwd_in;
} PerLayerSurfaces;

// Per-layer ANE requests (bound to per-layer IOSurfaces)
typedef struct {
    // Fused kernels (legacy)
    void *sdpaFwd, *woFwd, *ffnFused;
    // Unfused forward kernels (ANE matmul-only mode)
    void *wqFwd, *wkFwd, *wvFwd;  // Wo uses woFwd
    void *w1Fwd, *w3Fwd, *w2Fwd;
    // Backward kernels
    void *ffnBwdW2t, *ffnBwdW13t, *wotBwd, *qBwd, *kvBwd;
} PerLayerRequests;

// Checkpoint header
typedef struct {
    int magic, version, step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
    int kv_heads, head_dim, q_dim;  // GQA fields
    // Note: was int pad[3] in v3, now stores GQA info in v4+
} CkptHdr;

// Globals
static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Alloc helpers
static AdamState adam_alloc(size_t n) { AdamState s; s.m=(float*)calloc(n,4); s.v=(float*)calloc(n,4); s.n=n; return s; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }

static LayerWeights layer_weights_alloc(void) {
    LayerWeights w;
    w.Wq=(float*)malloc(WQ_SZ*4); w.Wk=(float*)malloc(WK_SZ*4);
    w.Wv=(float*)malloc(WV_SZ*4); w.Wo=(float*)malloc(WO_SZ*4);
    w.W1=(float*)malloc(W1_SZ*4); w.W2=(float*)malloc(W2_SZ*4); w.W3=(float*)malloc(W3_SZ*4);
    w.rms_att=(float*)malloc(DIM*4); w.rms_ffn=(float*)malloc(DIM*4);
    return w;
}
static void layer_weights_free(LayerWeights *w) {
    free(w->Wq);free(w->Wk);free(w->Wv);free(w->Wo);
    free(w->W1);free(w->W2);free(w->W3);free(w->rms_att);free(w->rms_ffn);
}
static LayerAdam layer_adam_alloc(void) {
    LayerAdam a;
    a.Wq=adam_alloc(WQ_SZ); a.Wk=adam_alloc(WK_SZ); a.Wv=adam_alloc(WV_SZ); a.Wo=adam_alloc(WO_SZ);
    a.W1=adam_alloc(W1_SZ); a.W2=adam_alloc(W2_SZ); a.W3=adam_alloc(W3_SZ);
    a.rms_att=adam_alloc(DIM); a.rms_ffn=adam_alloc(DIM);
    return a;
}
static void layer_adam_free(LayerAdam *a) {
    adam_free(&a->Wq);adam_free(&a->Wk);adam_free(&a->Wv);adam_free(&a->Wo);
    adam_free(&a->W1);adam_free(&a->W2);adam_free(&a->W3);
    adam_free(&a->rms_att);adam_free(&a->rms_ffn);
}
static LayerActs layer_acts_alloc(void) {
    LayerActs a;
    a.layer_in=(float*)malloc(SEQ*DIM*4);
    a.xnorm=(float*)malloc(SEQ*DIM*4);
    a.Q=(float*)malloc(SEQ*Q_DIM*4); a.K=(float*)malloc(SEQ*KV_DIM*4); a.V=(float*)malloc(SEQ*KV_DIM*4);
    a.attn_out=(float*)malloc(SEQ*Q_DIM*4); a.o_out=(float*)malloc(SEQ*DIM*4);
    a.x2=(float*)malloc(SEQ*DIM*4); a.x2norm=(float*)malloc(SEQ*DIM*4);
    a.h1=(float*)malloc(SEQ*HIDDEN*4); a.h3=(float*)malloc(SEQ*HIDDEN*4);
    a.silu_out=(float*)malloc(SEQ*HIDDEN*4); a.ffn_out=(float*)malloc(SEQ*DIM*4);
    return a;
}
static void layer_acts_free(LayerActs *a) {
    free(a->layer_in);free(a->xnorm);
    free(a->Q);free(a->K);free(a->V);
    free(a->attn_out);free(a->o_out);free(a->x2);free(a->x2norm);
    free(a->h1);free(a->h3);free(a->silu_out);free(a->ffn_out);
}
static LayerGrads layer_grads_alloc(void) {
    LayerGrads g;
    g.Wq=(float*)calloc(WQ_SZ,4); g.Wk=(float*)calloc(WK_SZ,4);
    g.Wv=(float*)calloc(WV_SZ,4); g.Wo=(float*)calloc(WO_SZ,4);
    g.W1=(float*)calloc(W1_SZ,4); g.W2=(float*)calloc(W2_SZ,4); g.W3=(float*)calloc(W3_SZ,4);
    g.rms_att=(float*)calloc(DIM,4); g.rms_ffn=(float*)calloc(DIM,4);
    return g;
}
static void layer_grads_zero(LayerGrads *g) {
    memset(g->Wq,0,WQ_SZ*4);memset(g->Wk,0,WK_SZ*4);
    memset(g->Wv,0,WV_SZ*4);memset(g->Wo,0,WO_SZ*4);
    memset(g->W1,0,W1_SZ*4);memset(g->W2,0,W2_SZ*4);memset(g->W3,0,W3_SZ*4);
    memset(g->rms_att,0,DIM*4);memset(g->rms_ffn,0,DIM*4);
}
static void layer_grads_free(LayerGrads *g) {
    free(g->Wq);free(g->Wk);free(g->Wv);free(g->Wo);
    free(g->W1);free(g->W2);free(g->W3);free(g->rms_att);free(g->rms_ffn);
}

// ===== LoRA adapter types =====
// Merge-based: W_eff = W_base + B@A, no forward/backward changes needed
typedef struct {
    int rank;
    float *Aq, *Bq;   // Wq: A[rank,DIM], B[Q_DIM,rank]
    float *Ak, *Bk;   // Wk: A[rank,DIM], B[KV_DIM,rank]
    float *Av, *Bv;   // Wv: A[rank,DIM], B[KV_DIM,rank]
    float *Ao, *Bo;   // Wo: A[rank,Q_DIM], B[DIM,rank]
    float *Wq_base, *Wk_base, *Wv_base, *Wo_base;
} LoRALayer;

typedef struct { AdamState Aq, Bq, Ak, Bk, Av, Bv, Ao, Bo; } LoRAAdam;
typedef struct { float *Aq, *Bq, *Ak, *Bk, *Av, *Bv, *Ao, *Bo; } LoRAGrads;

static LoRALayer lora_layer_alloc(int rank) {
    LoRALayer l; l.rank = rank;
    l.Aq=(float*)calloc((size_t)rank*DIM,4);     l.Bq=(float*)calloc((size_t)Q_DIM*rank,4);
    l.Ak=(float*)calloc((size_t)rank*DIM,4);     l.Bk=(float*)calloc((size_t)KV_DIM*rank,4);
    l.Av=(float*)calloc((size_t)rank*DIM,4);     l.Bv=(float*)calloc((size_t)KV_DIM*rank,4);
    l.Ao=(float*)calloc((size_t)rank*Q_DIM,4);   l.Bo=(float*)calloc((size_t)DIM*rank,4);
    l.Wq_base=(float*)malloc(WQ_SZ*4); l.Wk_base=(float*)malloc(WK_SZ*4);
    l.Wv_base=(float*)malloc(WV_SZ*4); l.Wo_base=(float*)malloc(WO_SZ*4);
    return l;
}
static void lora_layer_free(LoRALayer *l) {
    free(l->Aq);free(l->Bq);free(l->Ak);free(l->Bk);
    free(l->Av);free(l->Bv);free(l->Ao);free(l->Bo);
    free(l->Wq_base);free(l->Wk_base);free(l->Wv_base);free(l->Wo_base);
}
static LoRAAdam lora_adam_alloc(int rank) {
    LoRAAdam a;
    a.Aq=adam_alloc((size_t)rank*DIM);     a.Bq=adam_alloc((size_t)Q_DIM*rank);
    a.Ak=adam_alloc((size_t)rank*DIM);     a.Bk=adam_alloc((size_t)KV_DIM*rank);
    a.Av=adam_alloc((size_t)rank*DIM);     a.Bv=adam_alloc((size_t)KV_DIM*rank);
    a.Ao=adam_alloc((size_t)rank*Q_DIM);   a.Bo=adam_alloc((size_t)DIM*rank);
    return a;
}
static void lora_adam_free(LoRAAdam *a) {
    adam_free(&a->Aq);adam_free(&a->Bq);adam_free(&a->Ak);adam_free(&a->Bk);
    adam_free(&a->Av);adam_free(&a->Bv);adam_free(&a->Ao);adam_free(&a->Bo);
}
static LoRAGrads lora_grads_alloc(int rank) {
    LoRAGrads g;
    g.Aq=(float*)calloc((size_t)rank*DIM,4);     g.Bq=(float*)calloc((size_t)Q_DIM*rank,4);
    g.Ak=(float*)calloc((size_t)rank*DIM,4);     g.Bk=(float*)calloc((size_t)KV_DIM*rank,4);
    g.Av=(float*)calloc((size_t)rank*DIM,4);     g.Bv=(float*)calloc((size_t)KV_DIM*rank,4);
    g.Ao=(float*)calloc((size_t)rank*Q_DIM,4);   g.Bo=(float*)calloc((size_t)DIM*rank,4);
    return g;
}
static void lora_grads_zero(LoRAGrads *g, int rank) {
    memset(g->Aq,0,(size_t)rank*DIM*4);     memset(g->Bq,0,(size_t)Q_DIM*rank*4);
    memset(g->Ak,0,(size_t)rank*DIM*4);     memset(g->Bk,0,(size_t)KV_DIM*rank*4);
    memset(g->Av,0,(size_t)rank*DIM*4);     memset(g->Bv,0,(size_t)KV_DIM*rank*4);
    memset(g->Ao,0,(size_t)rank*Q_DIM*4);   memset(g->Bo,0,(size_t)DIM*rank*4);
}
static void lora_grads_free(LoRAGrads *g) {
    free(g->Aq);free(g->Bq);free(g->Ak);free(g->Bk);
    free(g->Av);free(g->Bv);free(g->Ao);free(g->Bo);
}

// Merge: W_eff[out,in] = W_base[out,in] + B[out,rank] @ A[rank,in]
static void lora_merge_weight(float *W_eff, const float *W_base, const float *B, const float *A,
                              int out_dim, int rank, int in_dim) {
    memcpy(W_eff, W_base, (size_t)out_dim*in_dim*4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, in_dim, rank, 1.0f, B, rank, A, in_dim, 1.0f, W_eff, in_dim);
}

// Project dW → LoRA grads: dB += dW @ A^T, dA += B^T @ dW
static void lora_grad_project(float *dA, float *dB, const float *dW, const float *A, const float *B,
                              int out_dim, int rank, int in_dim) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, in_dim, 1.0f, dW, in_dim, A, in_dim, 1.0f, dB, rank);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, in_dim, out_dim, 1.0f, B, rank, dW, in_dim, 1.0f, dA, in_dim);
}
