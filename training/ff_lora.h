// ff_lora.h — Forward-Forward + LoRA training algorithm
// Implements Hinton's Forward-Forward (arXiv:2212.13345) combined with LoRA
// for backprop-free LLM fine-tuning on forward-only hardware (Apple Neural Engine).
//
// Key idea: Two forward passes (positive/negative data) per step.
// Each projection's LoRA A/B matrices are updated using only local activations.
// No backward pass, no cross-layer gradient flow.
//
// Reference: docs/specs/2026-03-16-forward-forward-lora-design.md
#pragma once
#include "config.h"
#include "cpu_ops.h"

// ===== Per-projection FF state =====
// Stores intermediate activations from positive and negative forward passes
// needed for the FF-LoRA gradient computation.
typedef struct {
    float theta;       // Goodness threshold (learnable scalar)
    float *z_pos;      // [rank, SEQ] = A @ xnorm during positive pass
    float *z_neg;      // [rank, SEQ] = A @ xnorm during negative pass
    float *y_pos;      // [out_dim, SEQ] projection output during positive pass
    float *y_neg;      // [out_dim, SEQ] projection output during negative pass
    float *xnorm_pos;  // [in_dim, SEQ] layer-normed input during positive pass (shared across q,k,v)
    float *xnorm_neg;  // [in_dim, SEQ] layer-normed input during negative pass (shared across q,k,v)
    int out_dim;
    int in_dim;
} FFProjState;

// Per-layer FF state (4 attention projections)
typedef struct {
    FFProjState q, k, v, o;
    float G_layer_pos, G_layer_neg;  // Layer-level goodness (for logging)
} FFLayerState;

// ===== Allocation / deallocation =====

static FFProjState ff_proj_alloc(int out_dim, int in_dim, int rank) {
    FFProjState s;
    s.theta = 0.0f;
    s.out_dim = out_dim;
    s.in_dim = in_dim;
    s.z_pos = (float*)safe_calloc((size_t)rank * SEQ, 4);
    s.z_neg = (float*)safe_calloc((size_t)rank * SEQ, 4);
    s.y_pos = (float*)safe_calloc((size_t)out_dim * SEQ, 4);
    s.y_neg = (float*)safe_calloc((size_t)out_dim * SEQ, 4);
    // xnorm_pos/xnorm_neg are shared across projections in a layer,
    // so we allocate them at the layer level and point to them
    s.xnorm_pos = NULL;
    s.xnorm_neg = NULL;
    return s;
}

static void ff_proj_free(FFProjState *s) {
    free(s->z_pos); free(s->z_neg);
    free(s->y_pos); free(s->y_neg);
    // Don't free xnorm_pos/xnorm_neg — owned by layer
}

static FFLayerState ff_layer_alloc(int rank) {
    FFLayerState ls;
    ls.q = ff_proj_alloc(Q_DIM, DIM, rank);
    ls.k = ff_proj_alloc(KV_DIM, DIM, rank);
    ls.v = ff_proj_alloc(KV_DIM, DIM, rank);
    ls.o = ff_proj_alloc(DIM, Q_DIM, rank);
    ls.G_layer_pos = 0;
    ls.G_layer_neg = 0;

    // Allocate shared xnorm buffers for attention projections (q,k,v share input xnorm)
    // o projection has a different input (attn_out), so it gets its own xnorm
    ls.q.xnorm_pos = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    ls.q.xnorm_neg = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    ls.k.xnorm_pos = ls.q.xnorm_pos;  // share with q
    ls.k.xnorm_neg = ls.q.xnorm_neg;
    ls.v.xnorm_pos = ls.q.xnorm_pos;  // share with q
    ls.v.xnorm_neg = ls.q.xnorm_neg;
    // o projection input is attn_out [Q_DIM, SEQ]
    ls.o.xnorm_pos = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    ls.o.xnorm_neg = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);

    return ls;
}

static void ff_layer_free(FFLayerState *ls) {
    // Free shared xnorm for q (k,v share the same pointer)
    free(ls->q.xnorm_pos);
    free(ls->q.xnorm_neg);
    // Free o's xnorm (separate allocation)
    free(ls->o.xnorm_pos);
    free(ls->o.xnorm_neg);
    // Null out shared pointers before freeing projections
    ls->k.xnorm_pos = NULL; ls->k.xnorm_neg = NULL;
    ls->v.xnorm_pos = NULL; ls->v.xnorm_neg = NULL;
    ls->o.xnorm_pos = NULL; ls->o.xnorm_neg = NULL;
    ls->q.xnorm_pos = NULL; ls->q.xnorm_neg = NULL;
    ff_proj_free(&ls->q);
    ff_proj_free(&ls->k);
    ff_proj_free(&ls->v);
    ff_proj_free(&ls->o);
}

// ===== Modified LoRA forward: stores intermediate z = A @ xnorm =====
// out[out_dim, SEQ] += B[out_dim, rank] @ (A[rank, in_dim] @ x[in_dim, SEQ])
// Also saves: z_out[rank, SEQ] = A @ x (for FF gradient computation)
static void lora_addmm_ff(float *out, const float *A, const float *B,
                           const float *x, float *tmp_r, float *z_out,
                           int out_dim, int rank, int in_dim) {
    // z[rank, SEQ] = A[rank, in_dim] @ x[in_dim, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, SEQ, in_dim, 1.0f, A, in_dim, x, SEQ, 0.0f, tmp_r, SEQ);
    // Store z for gradient computation
    memcpy(z_out, tmp_r, (size_t)rank * SEQ * 4);
    // out[out_dim, SEQ] += B[out_dim, rank] @ z[rank, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, SEQ, rank, 1.0f, B, rank, tmp_r, SEQ, 1.0f, out, SEQ);
}

// ===== Negative data generation: token corruption =====
// Randomly replaces a fraction of input tokens with random tokens from vocab.
// neg_tokens must be pre-allocated to seq_len.
static void ff_generate_negative_tokens(uint16_t *neg_tokens, const uint16_t *pos_tokens,
                                         int seq_len, int vocab_size, float corrupt_rate,
                                         long seed) {
    srand48(seed);
    for (int i = 0; i < seq_len; i++) {
        float r = (float)drand48();
        if (r < corrupt_rate) {
            neg_tokens[i] = (uint16_t)((unsigned int)(drand48() * vocab_size) % vocab_size);
        } else {
            neg_tokens[i] = pos_tokens[i];
        }
    }
}

// ===== Compute goodness: G = (1/(out_dim * SEQ)) * ||y||_F^2 =====
// Uses vDSP for vectorized computation.
static float ff_compute_goodness(const float *y, int out_dim) {
    float sum_sq = 0;
    vDSP_dotpr(y, 1, y, 1, &sum_sq, (vDSP_Length)((size_t)out_dim * SEQ));
    return sum_sq / (float)((size_t)out_dim * SEQ);
}

// ===== Per-projection FF-LoRA gradient computation and accumulation =====
// Computes dB and dA from FF goodness objective and accumulates into gradient buffers.
//
// For the query projection (similarly for k,v,o):
//   dG/dB = (2/(out*SEQ)) * y @ z^T   where z = A @ xnorm (saved from forward)
//   dG/dA = (2/(out*SEQ)) * B^T @ y @ xnorm^T
//
// The FF update combines positive and negative passes:
//   dB_total = -s_pos * dG_pos/dB + s_neg * dG_neg/dB
//   dA_total = -s_pos * dG_pos/dA + s_neg * dG_neg/dA
// where s_pos = 1 - sigma(G_pos - theta), s_neg = 1 - sigma(theta - G_neg)
//
// Arguments:
//   dA, dB:       gradient accumulators [rank x in_dim], [out_dim x rank]
//   y_pos, z_pos: positive pass outputs [out_dim x SEQ], [rank x SEQ]
//   y_neg, z_neg: negative pass outputs
//   xnorm_pos, xnorm_neg: layer inputs [in_dim x SEQ]
//   A, B:         current LoRA weights
//   theta:        goodness threshold (updated in-place)
//   out_dim, rank, in_dim: dimensions
//   G_pos_out, G_neg_out: if non-NULL, store computed goodness values
static void ff_lora_compute_grads(
    float *dA, float *dB, float *theta,
    const float *y_pos, const float *z_pos, const float *xnorm_pos,
    const float *y_neg, const float *z_neg, const float *xnorm_neg,
    const float *A, const float *B,
    int out_dim, int rank, int in_dim,
    float *G_pos_out, float *G_neg_out)
{
    // Compute goodness for positive and negative passes
    float G_pos = ff_compute_goodness(y_pos, out_dim);
    float G_neg = ff_compute_goodness(y_neg, out_dim);

    if (G_pos_out) *G_pos_out = G_pos;
    if (G_neg_out) *G_neg_out = G_neg;

    // FF scaling factors (sigmoid)
    // s_pos = 1 - sigma(G_pos - theta): want to increase G_pos
    // s_neg = 1 - sigma(theta - G_neg): want to decrease G_neg
    float s_pos = 1.0f - 1.0f / (1.0f + expf(-(G_pos - *theta)));
    float s_neg = 1.0f - 1.0f / (1.0f + expf(-(*theta - G_neg)));

    float norm = 2.0f / (float)((size_t)out_dim * SEQ);
    float scale_pos = -s_pos * norm;  // negative: want to INCREASE G_pos (gradient ascent)
    float scale_neg = +s_neg * norm;  // positive: want to DECREASE G_neg (gradient descent)

    // --- dB accumulation ---
    // dB += scale * y @ z^T
    // [out_dim, rank] += [out_dim, SEQ] @ [SEQ, rank] (z^T is [SEQ, rank])
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, SEQ, scale_pos, y_pos, SEQ, z_pos, SEQ, 1.0f, dB, rank);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, SEQ, scale_neg, y_neg, SEQ, z_neg, SEQ, 1.0f, dB, rank);

    // --- dA accumulation ---
    // dA += scale * B^T @ y @ xnorm^T
    // Step 1: tmp[rank, SEQ] = B^T[rank, out_dim] @ y[out_dim, SEQ]
    // Step 2: dA[rank, in_dim] += scale * tmp[rank, SEQ] @ xnorm^T[SEQ, in_dim]
    float *tmp = (float*)safe_malloc((size_t)rank * SEQ * 4);

    // Positive pass
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, SEQ, out_dim, 1.0f, B, rank, y_pos, SEQ, 0.0f, tmp, SEQ);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, in_dim, SEQ, scale_pos, tmp, SEQ, xnorm_pos, SEQ, 1.0f, dA, in_dim);

    // Negative pass
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, SEQ, out_dim, 1.0f, B, rank, y_neg, SEQ, 0.0f, tmp, SEQ);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, in_dim, SEQ, scale_neg, tmp, SEQ, xnorm_neg, SEQ, 1.0f, dA, in_dim);

    free(tmp);

    // --- Update threshold ---
    // theta moves toward the midpoint of G_pos and G_neg
    // dt/dtheta_l = s_neg - s_pos (from the FF loss gradient w.r.t. theta)
    float lr_theta_scale = 0.1f;  // threshold learns 10x slower (stabilizer)
    *theta += lr_theta_scale * (s_neg - s_pos);
}

// ===== Initialize thresholds from pretrained model =====
// Runs one forward pass on real data and sets theta = 0.8 * mean_goodness per projection.
// The 0.8 factor ensures initial "surplus" goodness on positive data (see spec Risk R4).
static void ff_init_thresholds(FFLayerState *ff_states, int nlayers) {
    for (int L = 0; L < nlayers; L++) {
        // Thresholds will be set during the first positive forward pass.
        // Initialize to 0; they will be calibrated on step 0.
        ff_states[L].q.theta = 0.0f;
        ff_states[L].k.theta = 0.0f;
        ff_states[L].v.theta = 0.0f;
        ff_states[L].o.theta = 0.0f;
    }
}

// ===== Calibrate thresholds from a positive forward pass =====
// Sets theta = scale_factor * G_pos for each projection.
// Call after the first positive forward pass.
static void ff_calibrate_thresholds(FFLayerState *ff_states, int nlayers, float scale_factor) {
    for (int L = 0; L < nlayers; L++) {
        float Gq = ff_compute_goodness(ff_states[L].q.y_pos, ff_states[L].q.out_dim);
        float Gk = ff_compute_goodness(ff_states[L].k.y_pos, ff_states[L].k.out_dim);
        float Gv = ff_compute_goodness(ff_states[L].v.y_pos, ff_states[L].v.out_dim);
        float Go = ff_compute_goodness(ff_states[L].o.y_pos, ff_states[L].o.out_dim);
        ff_states[L].q.theta = scale_factor * Gq;
        ff_states[L].k.theta = scale_factor * Gk;
        ff_states[L].v.theta = scale_factor * Gv;
        ff_states[L].o.theta = scale_factor * Go;
    }
}

// ===== Print FF-LoRA diagnostics per layer =====
static void ff_print_layer_stats(const FFLayerState *ff, int layer,
                                  float Gq_pos, float Gq_neg,
                                  float Gk_pos, float Gk_neg,
                                  float Gv_pos, float Gv_neg,
                                  float Go_pos, float Go_neg) {
    printf("  L%02d: Gq[+%.3f -%.3f th%.3f] Gk[+%.3f -%.3f th%.3f] "
           "Gv[+%.3f -%.3f th%.3f] Go[+%.3f -%.3f th%.3f]\n",
           layer,
           Gq_pos, Gq_neg, ff->q.theta,
           Gk_pos, Gk_neg, ff->k.theta,
           Gv_pos, Gv_neg, ff->v.theta,
           Go_pos, Go_neg, ff->o.theta);
}
