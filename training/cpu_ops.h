// cpu_ops.h — CPU operations: RMSNorm, cross-entropy, Adam, embedding
#pragma once
#include "config.h"

static float *g_rms_tmp = NULL;

static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss);
}

static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = (float*)malloc(S*4);
    float *ss = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = (float*)malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    float *dot = (float*)calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot);
}

static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps, float wd) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * w[i]);
    }
}

// Cross-entropy loss: operates on logits[S, V] row-major (each row = one token, contiguous)
// Much faster than column-major: no strided gather/scatter, all ops on contiguous memory
static float cross_entropy_loss(float *dlogits, const float *logits, const uint16_t *targets, int V, int S) {
    float total_loss = 0;
    float invS = 1.0f / S;
    for (int t = 0; t < S; t++) {
        const float *row = logits + t * V;   // contiguous V-element row
        float *drow = dlogits + t * V;
        // Softmax on contiguous memory
        float maxv; vDSP_maxv(row, 1, &maxv, (vDSP_Length)V);
        float neg_max = -maxv;
        // Copy logits to dlogits, subtract max, exp, normalize — all in-place
        vDSP_vsadd(row, 1, &neg_max, drow, 1, (vDSP_Length)V);
        int n = V; vvexpf(drow, drow, &n);
        float sum; vDSP_sve(drow, 1, &sum, (vDSP_Length)V);
        float inv_sum = 1.0f / sum;
        vDSP_vsmul(drow, 1, &inv_sum, drow, 1, (vDSP_Length)V);
        // Loss + gradient
        int tgt = targets[t];
        total_loss -= logf(drow[tgt] + 1e-10f);
        drow[tgt] -= 1.0f;
        vDSP_vsmul(drow, 1, &invS, drow, 1, (vDSP_Length)V);
    }
    return total_loss / S;
}

// Vocab compaction: build mapping from full 32K vocab to compact vocab
typedef struct {
    int compact_vocab;          // number of active tokens
    int *full_to_compact;       // [VOCAB] → compact id (-1 if unused)
    int *compact_to_full;       // [compact_vocab] → full vocab id
} VocabMap;

static VocabMap vocab_map_build(const uint16_t *data, size_t n_tokens, int full_vocab) {
    VocabMap vm;
    vm.full_to_compact = (int*)malloc(full_vocab * sizeof(int));
    memset(vm.full_to_compact, -1, full_vocab * sizeof(int));
    // Scan for used tokens
    for (size_t i = 0; i < n_tokens; i++) {
        vm.full_to_compact[data[i]] = 0;  // mark as used
    }
    // Assign compact IDs
    int cid = 0;
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] == 0)
            vm.full_to_compact[v] = cid++;
        else
            vm.full_to_compact[v] = -1;
    }
    vm.compact_vocab = cid;
    vm.compact_to_full = (int*)malloc(cid * sizeof(int));
    for (int v = 0; v < full_vocab; v++) {
        if (vm.full_to_compact[v] >= 0)
            vm.compact_to_full[vm.full_to_compact[v]] = v;
    }
    return vm;
}

// Create compact embedding from full embedding
static float *vocab_compact_embed(const float *full_embed, const VocabMap *vm, int dim) {
    float *ce = (float*)malloc((size_t)vm->compact_vocab * dim * 4);
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(ce + c*dim, full_embed + vm->compact_to_full[c]*dim, dim*4);
    return ce;
}

// Scatter compact embed gradients back to full embed
static void vocab_scatter_grads(float *full_gembed, const float *compact_gembed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++) {
        int fv = vm->compact_to_full[c];
        for (int d = 0; d < dim; d++)
            full_gembed[fv*dim + d] += compact_gembed[c*dim + d];
    }
}

// Update full embed from compact embed (after adam)
static void vocab_update_full(float *full_embed, const float *compact_embed, const VocabMap *vm, int dim) {
    for (int c = 0; c < vm->compact_vocab; c++)
        memcpy(full_embed + vm->compact_to_full[c]*dim, compact_embed + c*dim, dim*4);
}

static void embed_lookup(float *x, const float *embed, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            x[d*seq + t] = embed[tok*dim + d];
    }
}

static void embed_backward(float *d_embed, const float *dx, const uint16_t *tokens, int dim, int seq) {
    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        for (int d = 0; d < dim; d++)
            d_embed[tok*dim + d] += dx[d*seq + t];
    }
}

// CPU fp32 SDPA backward — replaces ANE fp16 sdpaBwd1/sdpaBwd2 to avoid underflow
// Data layout: all tensors are [dim, SEQ] row-major (channel-first)
// Q, K_tiled, V_tiled, da: [Q_DIM, SEQ] where Q_DIM = HEADS * HD
// dq_out, dk_out, dv_out: [Q_DIM, SEQ] (caller handles GQA reduce)
static void cpu_sdpa_backward(
    const float *Q, const float *K_tiled, const float *V_tiled, const float *da,
    float *dq_out, float *dk_out, float *dv_out,
    int heads, int hd, int seq)
{
    float scale = 1.0f / sqrtf((float)hd);
    int q_dim = heads * hd;

    // Temp buffers (per head, processed sequentially)
    float *scores = (float*)malloc(seq * seq * sizeof(float));
    float *ds     = (float*)malloc(seq * seq * sizeof(float));

    memset(dq_out, 0, q_dim * seq * sizeof(float));
    memset(dk_out, 0, q_dim * seq * sizeof(float));
    memset(dv_out, 0, q_dim * seq * sizeof(float));

    for (int h = 0; h < heads; h++) {
        const float *Q_h  = Q + h * hd * seq;
        const float *K_h  = K_tiled + h * hd * seq;
        const float *V_h  = V_tiled + h * hd * seq;
        const float *da_h = da + h * hd * seq;
        float *dq_h = dq_out + h * hd * seq;
        float *dk_h = dk_out + h * hd * seq;
        float *dv_h = dv_out + h * hd * seq;

        // Step 1: Recompute attention scores = Q_h^T @ K_h / sqrt(hd) → [SEQ, SEQ]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, seq, hd, scale, Q_h, seq, K_h, seq, 0.0f, scores, seq);

        // Step 2: Causal mask + softmax (in-place on scores → probs)
        for (int s = 0; s < seq; s++) {
            float *row = scores + s * seq;
            // Mask future positions
            for (int j = s + 1; j < seq; j++) row[j] = -1e9f;
            // Stable softmax
            float maxv = -1e30f;
            for (int j = 0; j <= s; j++) maxv = fmaxf(maxv, row[j]);
            float sum = 0;
            for (int j = 0; j <= s; j++) {
                row[j] = expf(row[j] - maxv);
                sum += row[j];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int j = 0; j <= s; j++) row[j] *= inv_sum;
            for (int j = s + 1; j < seq; j++) row[j] = 0.0f;
        }
        // scores now contains probs [SEQ, SEQ]

        // Step 3: dV_h = da_h @ probs → [HD, SEQ] @ [SEQ, SEQ] → [HD, SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    hd, seq, seq, 1.0f, da_h, seq, scores, seq, 0.0f, dv_h, seq);

        // Step 4: dp = da_h^T @ V_h → [SEQ, HD] @ [HD, SEQ] → [SEQ, SEQ]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, seq, hd, 1.0f, da_h, seq, V_h, seq, 0.0f, ds, seq);
        // ds now contains dp [SEQ, SEQ]

        // Step 5: Softmax backward: ds = (dp - sum(dp * probs, axis=-1)) * probs * scale
        for (int s = 0; s < seq; s++) {
            float *dp_row = ds + s * seq;
            float *p_row  = scores + s * seq;
            float dot = 0;
            for (int j = 0; j < seq; j++) dot += dp_row[j] * p_row[j];
            for (int j = 0; j < seq; j++)
                dp_row[j] = (dp_row[j] - dot) * p_row[j] * scale;
        }
        // ds now contains the softmax backward result

        // Step 6: dQ_h = K_h @ ds^T → [HD, SEQ] @ [SEQ, SEQ]^T → [HD, SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    hd, seq, seq, 1.0f, K_h, seq, ds, seq, 0.0f, dq_h, seq);

        // Step 7: dK_h = Q_h @ ds → [HD, SEQ] @ [SEQ, SEQ] → [HD, SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    hd, seq, seq, 1.0f, Q_h, seq, ds, seq, 0.0f, dk_h, seq);
    }

    free(scores);
    free(ds);
}

// RoPE forward (in-place): apply rotation to Q/K
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_forward_inplace(float *x, int seq, int dim, int hd) {
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
            for (int p = 0; p < seq; p++) {
                float theta = p * freq;
                float cos_t = cosf(theta), sin_t = sinf(theta);
                int idx0 = (h * hd + 2 * i) * seq + p;
                int idx1 = (h * hd + 2 * i + 1) * seq + p;
                float v0 = x[idx0], v1 = x[idx1];
                x[idx0] = v0 * cos_t - v1 * sin_t;
                x[idx1] = v0 * sin_t + v1 * cos_t;
            }
        }
    }
}

// CPU fp32 SDPA forward — causal attention
// Data layout: all tensors [dim, SEQ] channel-first
// Q, K_tiled, V_tiled: [Q_DIM, SEQ] where Q_DIM = HEADS * HD
// attn_out: [Q_DIM, SEQ]
static void cpu_sdpa_forward(
    const float *Q, const float *K_tiled, const float *V_tiled, float *attn_out,
    int heads, int hd, int seq)
{
    float scale = 1.0f / sqrtf((float)hd);
    float *scores = (float*)malloc(seq * seq * sizeof(float));

    for (int h = 0; h < heads; h++) {
        const float *Q_h = Q + h * hd * seq;
        const float *K_h = K_tiled + h * hd * seq;
        const float *V_h = V_tiled + h * hd * seq;
        float *out_h = attn_out + h * hd * seq;

        // scores = Q_h^T @ K_h / sqrt(hd) → [seq, seq]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    seq, seq, hd, scale, Q_h, seq, K_h, seq, 0.0f, scores, seq);

        // Causal mask + softmax
        for (int s = 0; s < seq; s++) {
            float *row = scores + s * seq;
            for (int j = s + 1; j < seq; j++) row[j] = -1e9f;
            float maxv = -1e30f;
            for (int j = 0; j <= s; j++) maxv = fmaxf(maxv, row[j]);
            float sum = 0;
            for (int j = 0; j <= s; j++) {
                row[j] = expf(row[j] - maxv);
                sum += row[j];
            }
            float inv_sum = 1.0f / (sum + 1e-10f);
            for (int j = 0; j <= s; j++) row[j] *= inv_sum;
            for (int j = s + 1; j < seq; j++) row[j] = 0.0f;
        }

        // out_h[hd, seq] = V_h[hd, seq] @ probs^T[seq, seq]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    hd, seq, seq, 1.0f, V_h, seq, scores, seq, 0.0f, out_h, seq);
    }

    free(scores);
}

// RoPE backward (in-place): inverse rotation on dQ/dK gradients
// Data layout: [DIM, SEQ] channel-first, DIM = nheads * hd
static void rope_backward_inplace(float *dx, int seq, int dim, int hd) {
    int nheads = dim / hd;
    for (int h = 0; h < nheads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)hd);
            for (int p = 0; p < seq; p++) {
                float theta = p * freq;
                float cos_t = cosf(theta), sin_t = sinf(theta);
                int idx0 = (h * hd + 2 * i) * seq + p;
                int idx1 = (h * hd + 2 * i + 1) * seq + p;
                float v0 = dx[idx0], v1 = dx[idx1];
                dx[idx0] = v0 * cos_t + v1 * sin_t;
                dx[idx1] = -v0 * sin_t + v1 * cos_t;
            }
        }
    }
}
