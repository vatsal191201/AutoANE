// backprop_lora.h — P16 Hybrid: CPU fp32 backward pass for LoRA training
// Used with ANE conv-fused forward + CPU backward + LoRA gradient projection.
// All backward dx ops run on CPU via Accelerate BLAS. Only LoRA A/B + RMS norms are updated.
//
// Ported from train.m backward pass (lines 927-1192) with modifications:
// - No dW accumulation for frozen base weights (W1/W2/W3/Wq/Wk/Wv/Wo)
// - LoRA gradient projection via lora_grad_project() for Wq/Wk/Wv/Wo
// - No async GCD dispatch (simpler, sequential — async not needed for P16)
#pragma once
#include "config.h"
#include "cpu_ops.h"
#include "io.h"

// Per-layer activation storage for backward pass
typedef struct {
    float *x_pre;      // [DIM, SEQ] residual stream input to this layer
    float *xnorm;      // [DIM, SEQ] post-RMSNorm (pre-attention)
    float *Q;           // [Q_DIM, SEQ] query (post-LoRA, pre-RoPE)
    float *K;           // [KV_DIM, SEQ] key (post-LoRA, pre-RoPE)
    float *V;           // [KV_DIM, SEQ] value (post-LoRA)
    float *attn_out;    // [Q_DIM, SEQ] attention output (pre-Wo)
    float *x2;          // [DIM, SEQ] post-attention residual
    float *x2norm;      // [DIM, SEQ] post-RMSNorm (pre-FFN)
    float *h1;          // [HIDDEN, SEQ] gate projection output
    float *h3;          // [HIDDEN, SEQ] up projection output
    float *silu_out;    // [HIDDEN, SEQ] SiLU(h1) * h3
} BP_LayerActs;

static BP_LayerActs bp_layer_acts_alloc(void) {
    BP_LayerActs a;
    a.x_pre    = (float*)safe_malloc((size_t)DIM * SEQ * 4);
    a.xnorm    = (float*)safe_malloc((size_t)DIM * SEQ * 4);
    a.Q        = (float*)safe_malloc((size_t)Q_DIM * SEQ * 4);
    a.K        = (float*)safe_malloc((size_t)KV_DIM * SEQ * 4);
    a.V        = (float*)safe_malloc((size_t)KV_DIM * SEQ * 4);
    a.attn_out = (float*)safe_malloc((size_t)Q_DIM * SEQ * 4);
    a.x2       = (float*)safe_malloc((size_t)DIM * SEQ * 4);
    a.x2norm   = (float*)safe_malloc((size_t)DIM * SEQ * 4);
    a.h1       = (float*)safe_malloc((size_t)HIDDEN * SEQ * 4);
    a.h3       = (float*)safe_malloc((size_t)HIDDEN * SEQ * 4);
    a.silu_out = (float*)safe_malloc((size_t)HIDDEN * SEQ * 4);
    return a;
}

static void bp_layer_acts_free(BP_LayerActs *a) {
    free(a->x_pre); free(a->xnorm); free(a->Q); free(a->K); free(a->V);
    free(a->attn_out); free(a->x2); free(a->x2norm);
    free(a->h1); free(a->h3); free(a->silu_out);
}

// Backward working buffers (allocated once, reused across layers)
typedef struct {
    float *dy;          // [DIM, SEQ] gradient flowing backward
    float *dffn;        // [DIM, SEQ] scaled dy for FFN backward
    float *dsilu;       // [HIDDEN, SEQ] backward through W2
    float *dh1;         // [HIDDEN, SEQ] backward through SiLU (gate path)
    float *dh3;         // [HIDDEN, SEQ] backward through SiLU (up path)
    float *dx_ffn;      // [DIM, SEQ] backward through W1/W3
    float *dx2;         // [DIM, SEQ] backward through RMSNorm2 + residual
    float *da;          // [Q_DIM, SEQ] backward through Wo
    float *dx_attn;     // [DIM, SEQ] backward through Wq/Wk/Wv
    // SDPA backward outputs (all at Q_DIM = HEADS*HD, before GQA reduce)
    float *dq_full;     // [Q_DIM, SEQ]
    float *dk_full;     // [Q_DIM, SEQ] (HEADS-expanded, reduced to KV_DIM later)
    float *dv_full;     // [Q_DIM, SEQ] (HEADS-expanded, reduced to KV_DIM later)
    // Post-GQA-reduce
    float *dk;          // [KV_DIM, SEQ]
    float *dv;          // [KV_DIM, SEQ]
    // SiLU backward temporaries
    float *silu_tmp;    // [HIDDEN, SEQ]
    float *silu_tmp2;   // [HIDDEN, SEQ]
    // GQA tile buffers for SDPA backward (K_tiled, V_tiled at HEADS*HD)
    float *k_tiled;     // [Q_DIM, SEQ]
    float *v_tiled;     // [Q_DIM, SEQ]
    // Pre-allocated per-layer temporaries (eliminates malloc/free churn)
    float *dx2_scaled;  // [DIM, SEQ]
    float *dW_Wo;       // [DIM, Q_DIM]
    float *Q_rope;      // [Q_DIM, SEQ]
    float *K_rope;      // [KV_DIM, SEQ]
    float *dW_q;        // [Q_DIM, DIM]
    float *dW_kv;       // [KV_DIM, DIM] (reused for Wk and Wv)
    float *dx_rms;      // [DIM, SEQ]
    float *dx_rms_final;// [DIM, SEQ]
} BP_WorkBufs;

static BP_WorkBufs bp_work_alloc(void) {
    BP_WorkBufs b;
    b.dy        = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dffn      = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dsilu     = (float*)safe_calloc((size_t)HIDDEN * SEQ, 4);
    b.dh1       = (float*)safe_calloc((size_t)HIDDEN * SEQ, 4);
    b.dh3       = (float*)safe_calloc((size_t)HIDDEN * SEQ, 4);
    b.dx_ffn    = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dx2       = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.da        = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.dx_attn   = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dq_full   = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.dk_full   = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.dv_full   = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.dk        = (float*)safe_calloc((size_t)KV_DIM * SEQ, 4);
    b.dv        = (float*)safe_calloc((size_t)KV_DIM * SEQ, 4);
    b.silu_tmp  = (float*)safe_calloc((size_t)HIDDEN * SEQ, 4);
    b.silu_tmp2 = (float*)safe_calloc((size_t)HIDDEN * SEQ, 4);
    b.k_tiled   = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.v_tiled   = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.dx2_scaled    = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dW_Wo         = (float*)safe_calloc((size_t)DIM * Q_DIM, 4);
    b.Q_rope        = (float*)safe_calloc((size_t)Q_DIM * SEQ, 4);
    b.K_rope        = (float*)safe_calloc((size_t)KV_DIM * SEQ, 4);
    b.dW_q          = (float*)safe_calloc((size_t)Q_DIM * DIM, 4);
    b.dW_kv         = (float*)safe_calloc((size_t)KV_DIM * DIM, 4);
    b.dx_rms        = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    b.dx_rms_final  = (float*)safe_calloc((size_t)DIM * SEQ, 4);
    return b;
}

static void bp_work_free(BP_WorkBufs *b) {
    free(b->dy); free(b->dffn); free(b->dsilu); free(b->dh1); free(b->dh3);
    free(b->dx_ffn); free(b->dx2); free(b->da); free(b->dx_attn);
    free(b->dq_full); free(b->dk_full); free(b->dv_full);
    free(b->dk); free(b->dv);
    free(b->silu_tmp); free(b->silu_tmp2);
    free(b->k_tiled); free(b->v_tiled);
    free(b->dx2_scaled); free(b->dW_Wo); free(b->Q_rope); free(b->K_rope);
    free(b->dW_q); free(b->dW_kv); free(b->dx_rms); free(b->dx_rms_final);
}

// ===== P16 Backward Pass =====
// Computes LoRA gradients (dA, dB for Wq/Wk/Wv/Wo) + RMS norm gradients.
// dx chain flows through entire network (including frozen FFN layers) for residual connections.
//
// Arguments:
//   dlogits    — [SEQ, CV] gradient of loss w.r.t. logits (from cross_entropy_loss)
//   x_final    — [DIM, SEQ] final hidden state (post last layer, pre final RMSNorm)
//   cembed     — [CV, DIM] compact embedding matrix
//   rms_final  — [DIM] final RMSNorm weights
//   lw         — layer weights array
//   acts       — per-layer saved activations
//   lora_layers — LoRA adapter weights
//   lora_grads  — LoRA gradient accumulators (output)
//   grms_att   — per-layer RMSNorm attention gradient accumulator (output)
//   grms_ffn   — per-layer RMSNorm FFN gradient accumulator (output)
//   grms_final_out — final RMSNorm gradient accumulator (output)
//   res_alpha  — DeepNet residual scaling factor
//   work       — working buffers
//   lora_rank  — LoRA rank
static void bp_lora_backward(
    const float *dlogits, const float *x_final,
    const float *cembed, float *rms_final,
    LayerWeights *lw, BP_LayerActs *acts,
    LoRALayer *lora_layers, LoRAGrads *lora_grads,
    float **grms_att, float **grms_ffn, float *grms_final_out,
    float res_alpha, BP_WorkBufs *work, int lora_rank, int CV)
{

    // --- Classifier backward: dy = cembed^T @ dlogits^T ---
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, CV, 0.0f, work->dy, SEQ);

    // --- Final RMSNorm backward ---
    memset(work->dx_rms_final, 0, (size_t)SEQ * DIM * 4);
    rmsnorm_bwd(work->dx_rms_final, grms_final_out, work->dy, x_final, rms_final, DIM, SEQ);
    memcpy(work->dy, work->dx_rms_final, (size_t)SEQ * DIM * 4);

    // --- Per-layer backward (reversed) ---
    for (int L = NLAYERS - 1; L >= 0; L--) {
        LoRALayer *ll = &lora_layers[L];
        LoRAGrads *lg = &lora_grads[L];
        BP_LayerActs *ac = &acts[L];
        int r = lora_rank;

        // 1. FFN backward: dffn = alpha * dy
        vDSP_vsmul(work->dy, 1, &res_alpha, work->dffn, 1, (vDSP_Length)(SEQ * DIM));

        // 2. dffn @ W2^T -> dsilu [DIM->HIDDEN]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    HIDDEN, SEQ, DIM, 1.0f, lw[L].W2, HIDDEN, work->dffn, SEQ,
                    0.0f, work->dsilu, SEQ);

        // 3. SiLU derivative (vectorized vDSP)
        {
            int n = HIDDEN * SEQ;
            float minus1 = -1.0f, one = 1.0f;
            // sig = 1 / (1 + exp(-h1))
            vDSP_vsmul(ac->h1, 1, &minus1, work->silu_tmp, 1, (vDSP_Length)n);
            vvexpf(work->silu_tmp, work->silu_tmp, &n);
            vDSP_vsadd(work->silu_tmp, 1, &one, work->silu_tmp, 1, (vDSP_Length)n);
            vvrecf(work->silu_tmp, work->silu_tmp, &n);  // sig(h1)
            // dh3 = dsilu * sig(h1) * h1
            vDSP_vmul(ac->h1, 1, work->silu_tmp, 1, work->dh3, 1, (vDSP_Length)n);
            vDSP_vmul(work->dsilu, 1, work->dh3, 1, work->dh3, 1, (vDSP_Length)n);
            // dh1 = dsilu * h3 * (sig + h1*sig*(1-sig))
            vDSP_vsadd(work->silu_tmp, 1, &minus1, work->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vneg(work->silu_tmp2, 1, work->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(ac->h1, 1, work->silu_tmp2, 1, work->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vsadd(work->silu_tmp2, 1, &one, work->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(work->silu_tmp, 1, work->silu_tmp2, 1, work->silu_tmp2, 1, (vDSP_Length)n);
            vDSP_vmul(work->dsilu, 1, ac->h3, 1, work->dh1, 1, (vDSP_Length)n);
            vDSP_vmul(work->dh1, 1, work->silu_tmp2, 1, work->dh1, 1, (vDSP_Length)n);
        }

        // 4. dh1@W1^T + dh3@W3^T -> dx_ffn [HIDDEN->DIM]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, HIDDEN, 1.0f, lw[L].W1, DIM, work->dh1, SEQ,
                    0.0f, work->dx_ffn, SEQ);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, HIDDEN, 1.0f, lw[L].W3, DIM, work->dh3, SEQ,
                    1.0f, work->dx_ffn, SEQ);  // beta=1 to accumulate

        // (No dW for FFN — frozen weights, no LoRA on FFN)

        // 5. RMSNorm2 backward
        memset(work->dx2, 0, (size_t)SEQ * DIM * 4);
        rmsnorm_bwd(work->dx2, grms_ffn[L], work->dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
        // Add skip connection: dx2 += dy
        for (int i = 0; i < SEQ * DIM; i++) work->dx2[i] += work->dy[i];

        // 6. Wo backward: da = (alpha * dx2) @ Wo^T
        vDSP_vsmul(work->dx2, 1, &res_alpha, work->dx2_scaled, 1, (vDSP_Length)(SEQ * DIM));
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    Q_DIM, SEQ, DIM, 1.0f, lw[L].Wo, Q_DIM, work->dx2_scaled, SEQ,
                    0.0f, work->da, SEQ);

        // 7. dW Wo -> LoRA gradient
        memset(work->dW_Wo, 0, (size_t)DIM * Q_DIM * 4);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    DIM, Q_DIM, SEQ, 1.0f, work->dx2_scaled, SEQ, ac->attn_out, SEQ,
                    0.0f, work->dW_Wo, Q_DIM);
        lora_grad_project(lg->Ao, lg->Bo, work->dW_Wo, ll->Ao, ll->Bo, DIM, r, Q_DIM);

        // 8. SDPA backward: need Q,K,V with RoPE applied
        memcpy(work->Q_rope, ac->Q, (size_t)Q_DIM * SEQ * 4);
        memcpy(work->K_rope, ac->K, (size_t)KV_DIM * SEQ * 4);
        rope_forward_inplace(work->Q_rope, SEQ, Q_DIM, HD);
        rope_forward_inplace(work->K_rope, SEQ, KV_DIM, HD);

        // GQA tile K, V for SDPA backward
        gqa_tile_kv(work->k_tiled, work->K_rope, SEQ);
        gqa_tile_kv(work->v_tiled, ac->V, SEQ);

        // SDPA backward (CPU fp32)
        cpu_sdpa_backward(work->Q_rope, work->k_tiled, work->v_tiled, work->da,
                          work->dq_full, work->dk_full, work->dv_full, HEADS, HD, SEQ);

        // GQA reduce dK, dV from HEADS -> KV_HEADS
        gqa_reduce_kv(work->dk, work->dk_full, SEQ);
        gqa_reduce_kv(work->dv, work->dv_full, SEQ);
        // dQ stays at Q_DIM (no reduction)

        // RoPE backward
        rope_backward_inplace(work->dq_full, SEQ, Q_DIM, HD);
        rope_backward_inplace(work->dk, SEQ, KV_DIM, HD);

        // 9. dW Wq/Wk/Wv -> LoRA gradients (reuse pre-allocated dW buffers)
        // dW_q for Wq [Q_DIM, DIM]
        memset(work->dW_q, 0, (size_t)Q_DIM * DIM * 4);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    Q_DIM, DIM, SEQ, 1.0f, work->dq_full, SEQ, ac->xnorm, SEQ, 0.0f, work->dW_q, DIM);
        lora_grad_project(lg->Aq, lg->Bq, work->dW_q, ll->Aq, ll->Bq, Q_DIM, r, DIM);
        // dW_kv for Wk [KV_DIM, DIM] (reuse same buffer)
        memset(work->dW_kv, 0, (size_t)KV_DIM * DIM * 4);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    KV_DIM, DIM, SEQ, 1.0f, work->dk, SEQ, ac->xnorm, SEQ, 0.0f, work->dW_kv, DIM);
        lora_grad_project(lg->Ak, lg->Bk, work->dW_kv, ll->Ak, ll->Bk, KV_DIM, r, DIM);
        // dW_kv for Wv [KV_DIM, DIM] (reuse same buffer)
        memset(work->dW_kv, 0, (size_t)KV_DIM * DIM * 4);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    KV_DIM, DIM, SEQ, 1.0f, work->dv, SEQ, ac->xnorm, SEQ, 0.0f, work->dW_kv, DIM);
        lora_grad_project(lg->Av, lg->Bv, work->dW_kv, ll->Av, ll->Bv, KV_DIM, r, DIM);

        // 10. dx through Wq, Wk, Wv
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, Q_DIM, 1.0f, lw[L].Wq, DIM, work->dq_full, SEQ,
                    0.0f, work->dx_attn, SEQ);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, KV_DIM, 1.0f, lw[L].Wk, DIM, work->dk, SEQ,
                    1.0f, work->dx_attn, SEQ);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    DIM, SEQ, KV_DIM, 1.0f, lw[L].Wv, DIM, work->dv, SEQ,
                    1.0f, work->dx_attn, SEQ);

        // 11. RMSNorm1 backward + residual
        memset(work->dx_rms, 0, (size_t)SEQ * DIM * 4);
        rmsnorm_bwd(work->dx_rms, grms_att[L], work->dx_attn, ac->x_pre, lw[L].rms_att, DIM, SEQ);
        // dy for next (earlier) layer = dx_rms + dx2
        for (int i = 0; i < SEQ * DIM; i++) work->dy[i] = work->dx_rms[i] + work->dx2[i];
    }
}

// CV (compact vocab size) is passed as a parameter to bp_lora_backward().
