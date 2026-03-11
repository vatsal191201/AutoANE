// train.m — Dynamic weight ANE training (model-agnostic GQA support)
// Model selected at compile time via: make MODEL=qwen3_06b (or stories110m)
// Compile kernels ONCE at startup, update weights via IOSurface every step.
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include <math.h>

// === Gradient Sanitization (per Orion paper Bug #3 fix) ===
// Replaces NaN with 0, clips ±Inf to ±65504 (fp16 max)
// Returns count of sanitized values for monitoring
static int sanitize_gradients(float *buf, size_t n) {
    int fixed = 0;
    for (size_t i = 0; i < n; i++) {
        if (isnan(buf[i])) { buf[i] = 0.0f; fixed++; }
        else if (isinf(buf[i])) { buf[i] = (buf[i] > 0) ? 65504.0f : -65504.0f; fixed++; }
    }
    return fixed;
}
// Sanitize all layer gradients, return total count
static int sanitize_layer_grads(LayerGrads *g) {
    int n = 0;
    n += sanitize_gradients(g->Wq, WQ_SZ);
    n += sanitize_gradients(g->Wk, WK_SZ);
    n += sanitize_gradients(g->Wv, WV_SZ);
    n += sanitize_gradients(g->Wo, WO_SZ);
    n += sanitize_gradients(g->W1, W1_SZ);
    n += sanitize_gradients(g->W2, W2_SZ);
    n += sanitize_gradients(g->W3, W3_SZ);
    n += sanitize_gradients(g->rms_att, DIM);
    n += sanitize_gradients(g->rms_ffn, DIM);
    return n;
}
// Check weights for NaN/Inf after Adam update
static int validate_weights(float *buf, size_t n) {
    int bad = 0;
    for (size_t i = 0; i < n; i++) {
        if (isnan(buf[i]) || isinf(buf[i])) bad++;
    }
    return bad;
}

// Dynamic kernel set per layer
typedef struct {
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    Kern *ffnBwdW2t;   // dffn @ W2^T → dsilu_raw (DIM → HIDDEN)
    Kern *ffnBwdW13t;  // dh1@W1^T + dh3@W3^T → dx_ffn (HIDDEN → DIM)
    Kern *wotBwd;      // dx2 @ Wo → da (DIM → Q_DIM)
    Kern *sdpaBwd1;    // Q,K,V,da → dV_full,probs,dp (weight-free, has mask)
    Kern *sdpaBwd2;    // probs,dp,Q,K → dQ,dK_full (weight-free)
    Kern *qBwd;        // dq @ Wq → dx_q (Q_DIM → DIM)
    Kern *kvBwd;       // dk@Wk + dv@Wv → dx_kv (KV_DIM → DIM)
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// ===== Compile all dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk, float res_alpha) {
    NSDictionary *mask_w = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;

    // SDPA forward (no Wo): [1, DIM, 1, SDPA_FWD_SP] → [1, sdpa_out_ch, 1, SEQ]
    printf("  Compiling sdpaFwd (GQA)...\n");
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!dk->sdpaFwd) return false;

    // Wo forward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling woFwd...\n");
    dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!dk->woFwd) return false;

    // Fused FFN: [1, DIM, 1, FFN_FUSED_SP] → [1, DIM+3*HIDDEN, 1, SEQ]
    printf("  Compiling ffnFused...\n");
    int ffn_fused_och = DIM + 3*HIDDEN;
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic_alpha(res_alpha), @{},
        DIM*FFN_FUSED_SP*2, ffn_fused_och*SEQ*2);
    if (!dk->ffnFused) return false;

    // FFN backward W2^T: [1, DIM, 1, SEQ+HIDDEN] → [1, HIDDEN, 1, SEQ]
    printf("  Compiling ffnBwdW2t...\n");
    dk->ffnBwdW2t = compile_kern_mil_w(gen_ffn_bwd_w2t_dynamic(), @{},
        DIM*FFN_BWD_W2T_SP*2, HIDDEN*SEQ*2);
    if (!dk->ffnBwdW2t) return false;

    // FFN backward W1^T+W3^T: [1, HIDDEN, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling ffnBwdW13t...\n");
    dk->ffnBwdW13t = compile_kern_mil_w(gen_ffn_bwd_w13t_dynamic(), @{},
        HIDDEN*FFN_BWD_W13T_SP*2, DIM*SEQ*2);
    if (!dk->ffnBwdW13t) return false;

    // Wo^T backward: [1, DIM, 1, SEQ+Q_DIM] → [1, Q_DIM, 1, SEQ]
    printf("  Compiling wotBwd...\n");
    dk->wotBwd = compile_kern_mil_w(gen_wot_dynamic(), @{},
        DIM*WOT_BWD_SP*2, Q_DIM*SEQ*2);
    if (!dk->wotBwd) return false;

    // SDPA bwd1 (weight-free, has mask): [1, 4*Q_DIM, 1, SEQ] → [1, Q_DIM+2*SCORE_CH, 1, SEQ]
    printf("  Compiling sdpaBwd1 (GQA)...\n");
    dk->sdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1_noweight(), mask_w,
        4*Q_DIM*SEQ*2, (Q_DIM+2*SCORE_CH)*SEQ*2);
    if (!dk->sdpaBwd1) return false;

    // SDPA bwd2 (weight-free): [1, 2*SCORE_CH+2*Q_DIM, 1, SEQ] → [1, 2*Q_DIM, 1, SEQ]
    printf("  Compiling sdpaBwd2 (GQA)...\n");
    dk->sdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
        (2*SCORE_CH+2*Q_DIM)*SEQ*2, 2*Q_DIM*SEQ*2);
    if (!dk->sdpaBwd2) return false;

    // Q backward: [1, Q_DIM, 1, SEQ+DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling qBwd...\n");
    dk->qBwd = compile_kern_mil_w(gen_q_bwd_dynamic(), @{},
        Q_DIM*Q_BWD_SP*2, DIM*SEQ*2);
    if (!dk->qBwd) return false;

    // KV backward: [1, KV_DIM, 1, 2*SEQ+2*DIM] → [1, DIM, 1, SEQ]
    printf("  Compiling kvBwd...\n");
    dk->kvBwd = compile_kern_mil_w(gen_kv_bwd_dynamic(), @{},
        KV_DIM*KV_BWD_SP*2, DIM*SEQ*2);
    if (!dk->kvBwd) return false;

    return true;
}

// ===== Checkpoint =====
static void save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                            double ct, double cw, int cs, int adam_t,
                            LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                            float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "wb");
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM;
    h.hidden_dim = HIDDEN; h.n_heads = HEADS; h.seq_len = SEQ;
    h.lr = lr; h.loss = loss;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = adam_t;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        fwrite(la[L].Wq.m,4,WQ_SZ,f); fwrite(la[L].Wq.v,4,WQ_SZ,f);
        fwrite(la[L].Wk.m,4,WK_SZ,f); fwrite(la[L].Wk.v,4,WK_SZ,f);
        fwrite(la[L].Wv.m,4,WV_SZ,f); fwrite(la[L].Wv.v,4,WV_SZ,f);
        fwrite(la[L].Wo.m,4,WO_SZ,f); fwrite(la[L].Wo.v,4,WO_SZ,f);
        fwrite(la[L].W1.m,4,W1_SZ,f); fwrite(la[L].W1.v,4,W1_SZ,f);
        fwrite(la[L].W2.m,4,W2_SZ,f); fwrite(la[L].W2.v,4,W2_SZ,f);
        fwrite(la[L].W3.m,4,W3_SZ,f); fwrite(la[L].W3.v,4,W3_SZ,f);
        fwrite(la[L].rms_att.m,4,DIM,f); fwrite(la[L].rms_att.v,4,DIM,f);
        fwrite(la[L].rms_ffn.m,4,DIM,f); fwrite(la[L].rms_ffn.v,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f);
    fwrite(arms_final->m,4,DIM,f); fwrite(arms_final->v,4,DIM,f);
    fwrite(embed,4,VOCAB*DIM,f);
    fwrite(aembed->m,4,VOCAB*DIM,f); fwrite(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
}

static bool load_checkpoint(const char *path, int *step, int *total_steps, float *lr, float *loss,
                             double *ct, double *cw, int *cs, int *adam_t,
                             LayerWeights *lw, LayerAdam *la, float *rms_final, AdamState *arms_final,
                             float *embed, AdamState *aembed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    *step = h.step; *total_steps = h.total_steps; *lr = h.lr; *loss = h.loss;
    *ct = h.cum_train; *cw = h.cum_wall; *cs = h.cum_steps; *adam_t = h.adam_t;
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        fread(la[L].Wq.m,4,WQ_SZ,f); fread(la[L].Wq.v,4,WQ_SZ,f);
        fread(la[L].Wk.m,4,WK_SZ,f); fread(la[L].Wk.v,4,WK_SZ,f);
        fread(la[L].Wv.m,4,WV_SZ,f); fread(la[L].Wv.v,4,WV_SZ,f);
        fread(la[L].Wo.m,4,WO_SZ,f); fread(la[L].Wo.v,4,WO_SZ,f);
        fread(la[L].W1.m,4,W1_SZ,f); fread(la[L].W1.v,4,W1_SZ,f);
        fread(la[L].W2.m,4,W2_SZ,f); fread(la[L].W2.v,4,W2_SZ,f);
        fread(la[L].W3.m,4,W3_SZ,f); fread(la[L].W3.v,4,W3_SZ,f);
        fread(la[L].rms_att.m,4,DIM,f); fread(la[L].rms_att.v,4,DIM,f);
        fread(la[L].rms_ffn.m,4,DIM,f); fread(la[L].rms_ffn.v,4,DIM,f);
    }
    fread(rms_final,4,DIM,f);
    fread(arms_final->m,4,DIM,f); fread(arms_final->v,4,DIM,f);
    fread(embed,4,VOCAB*DIM,f);
    fread(aembed->m,4,VOCAB*DIM,f); fread(aembed->v,4,VOCAB*DIM,f);
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        int total_steps = 10000;
        float max_lr = 3e-4f;
        float adam_b1=0.9f, adam_b2=0.95f, adam_eps=1e-8f, wd=0.1f;
        int adam_t = 0, start_step = 0;
        int accum_steps = 10;
        int warmup_steps = 100;
        float grad_clip = 1.0f;
        float loss_scale = 256.0f;  // prevents fp16 gradient underflow in ANE backward
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);
        float min_lr_frac = 0.1f;
        double time_budget_sec = 0;  // 0 = unlimited (use --steps instead)

        bool do_resume = false, from_scratch = false;
        bool use_cpu_attn_bwd = false;  // CPU fp32 SDPA backward (fixes attention gradient underflow)
        bool cpu_only = false;  // Pure CPU training (no ANE, all cblas_sgemm)
        bool use_lora = false;
        int lora_rank = 8;
        float act_clamp = 0.0f;  // 0 = disabled, >0 = clamp activations to [-act_clamp, act_clamp]
        bool grad_sanitize = false;  // Enable gradient sanitization (NaN→0, ±Inf→±65504)
        int total_sanitized = 0;     // Running count of sanitized gradient values
        float adaptive_thresh = 0.0f;  // 0 = disabled, >0 = switch ANE→CPU when |x|_max exceeds threshold
        int adaptive_window = 5;       // Consecutive steps above threshold to trigger switch
        int adaptive_above_count = 0;  // Counter for consecutive steps above threshold
        int adaptive_switch_step = -1; // Step at which ANE→CPU switch occurred (-1 = no switch)
        const char *data_path = DEFAULT_DATA_PATH;
        for (int i=1; i<argc; i++) {
            if (strcmp(argv[i], "--resume") == 0) do_resume = true;
            else if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--cpu-attn-bwd") == 0) use_cpu_attn_bwd = true;
            else if (strcmp(argv[i], "--cpu-only") == 0) cpu_only = true;
            else if (strcmp(argv[i], "--lora") == 0) use_lora = true;
            else if (strcmp(argv[i], "--lora-rank") == 0 && i+1<argc) lora_rank = atoi(argv[++i]);
            else if (strcmp(argv[i], "--steps") == 0 && i+1<argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lr") == 0 && i+1<argc) max_lr = atof(argv[++i]);
            else if (strcmp(argv[i], "--accum") == 0 && i+1<argc) accum_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--warmup") == 0 && i+1<argc) warmup_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--clip") == 0 && i+1<argc) grad_clip = atof(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1<argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--wd") == 0 && i+1<argc) wd = atof(argv[++i]);
            else if (strcmp(argv[i], "--scale") == 0 && i+1<argc) loss_scale = atof(argv[++i]);
            else if (strcmp(argv[i], "--time") == 0 && i+1<argc) time_budget_sec = atof(argv[++i]);
            else if (strcmp(argv[i], "--clamp") == 0 && i+1<argc) act_clamp = atof(argv[++i]);
            else if (strcmp(argv[i], "--sanitize") == 0) grad_sanitize = true;
            else if (strcmp(argv[i], "--adaptive") == 0 && i+1<argc) adaptive_thresh = atof(argv[++i]);
            else if (strcmp(argv[i], "--adaptive-window") == 0 && i+1<argc) adaptive_window = atoi(argv[++i]);
        }
        if (cpu_only) use_cpu_attn_bwd = true;  // CPU-only implies CPU attention backward
        if (!cpu_only) ane_init();
        float lr = max_lr;

        // Allocate per-layer state
        LayerWeights lw[NLAYERS]; LayerAdam la[NLAYERS];
        LayerActs acts[NLAYERS]; LayerGrads grads[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            lw[L] = layer_weights_alloc(); la[L] = layer_adam_alloc();
            acts[L] = layer_acts_alloc(); grads[L] = layer_grads_alloc();
        }
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);
        float *grms_final = (float*)calloc(DIM, 4);
        float *gembed = (float*)calloc(VOCAB*DIM, 4);
        AdamState arms_final = adam_alloc(DIM);
        AdamState aembed = adam_alloc((size_t)VOCAB*DIM);

        double cum_train=0, cum_wall=0; int cum_steps=0;
        float resume_loss = 0;
        bool resuming = false;
        if (do_resume) {
            resuming = load_checkpoint(CKPT_PATH, &start_step, &total_steps, &lr, &resume_loss,
                &cum_train, &cum_wall, &cum_steps, &adam_t,
                lw, la, rms_final, &arms_final, embed, &aembed);
            if (resuming) printf("[RESUMED step %d, loss=%.4f]\n", start_step, resume_loss);
        }
        if (!resuming) {
            printf("=== %s Training: %s (%d layers, GQA %d/%d heads) ===\n",
                   cpu_only ? "CPU-Only" : "ANE Dynamic", MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
            printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
                   DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
            double xformer_m = (double)NLAYERS*(WQ_SZ + WK_SZ + WV_SZ + (double)WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2.0*DIM) / 1e6;
            double embed_m = (double)VOCAB*DIM / 1e6;
            printf("Params: %.1fM (transformer %.1fM + embed %.1fM)\n", xformer_m+embed_m, xformer_m, embed_m);
            if (cpu_only) printf("Mode: CPU-only (all matmuls via cblas_sgemm fp32)\n");
            else {
                printf("Kernels: 10 compiled (sdpaFwd+woFwd, ffnFused, ffnBwdW2t+W13t, wotBwd, sdpaBwd1+2, qBwd+kvBwd)\n");
                if (use_cpu_attn_bwd) printf("SDPA backward: CPU fp32 (accurate attention gradients)\n");
                else printf("SDPA backward: ANE fp16 (fast, may underflow)\n");
            }
            if (act_clamp > 0) printf("Activation clamp: [-%.1f, %.1f]\n", act_clamp, act_clamp);
            if (grad_sanitize) printf("Gradient sanitization: ENABLED (NaN→0, ±Inf→±65504)\n");
            if (adaptive_thresh > 0 && !cpu_only) printf("Adaptive ANE→CPU: threshold=%.0f, window=%d consecutive steps\n", adaptive_thresh, adaptive_window);
            printf("Accum %d steps, LR=%g\n", accum_steps, max_lr);
            double fwd_flops = 2.0*NLAYERS*((double)WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ) * SEQ;
            double total_flops = 3.0 * fwd_flops;
            printf("FLOPs/step: fwd=%.1fM total=%.1fM\n", fwd_flops/1e6, total_flops/1e6);
            printf("  Training from scratch (random init)\n");
            srand48(42);
            float scale_d=1.0f/sqrtf(DIM), scale_qd=1.0f/sqrtf(Q_DIM), scale_h=1.0f/sqrtf(HIDDEN);
            float res_scale = 1.0f/sqrtf(2.0f*NLAYERS);
            for (int L=0; L<NLAYERS; L++) {
                for(size_t i=0;i<WQ_SZ;i++) lw[L].Wq[i]=scale_d*(2*drand48()-1);
                for(size_t i=0;i<WK_SZ;i++) lw[L].Wk[i]=scale_d*(2*drand48()-1);
                for(size_t i=0;i<WV_SZ;i++) lw[L].Wv[i]=scale_d*(2*drand48()-1);
                for(size_t i=0;i<WO_SZ;i++) lw[L].Wo[i]=scale_qd*res_scale*(2*drand48()-1);
                for(size_t i=0;i<W1_SZ;i++) lw[L].W1[i]=scale_h*(2*drand48()-1);
                for(size_t i=0;i<W2_SZ;i++) lw[L].W2[i]=scale_d*res_scale*(2*drand48()-1);
                for(size_t i=0;i<W3_SZ;i++) lw[L].W3[i]=scale_h*(2*drand48()-1);
                for(int i=0;i<DIM;i++){lw[L].rms_att[i]=1.0f; lw[L].rms_ffn[i]=1.0f;}
            }
            for(int i=0;i<DIM;i++) rms_final[i]=1.0f;
            float escale = 0.02f;
            for(size_t i=0;i<(size_t)VOCAB*DIM;i++) embed[i]=escale*(2*drand48()-1);
        }

        // LoRA initialization
        LoRALayer lora_layers[NLAYERS];
        LoRAAdam lora_adam[NLAYERS];
        LoRAGrads lora_grads_arr[NLAYERS];
        if (use_lora) {
            int r = lora_rank;
            float a_scale = 1.0f / sqrtf((float)r);
            size_t lora_params = 0;
            for (int L = 0; L < NLAYERS; L++) {
                lora_layers[L] = lora_layer_alloc(r);
                lora_adam[L] = lora_adam_alloc(r);
                lora_grads_arr[L] = lora_grads_alloc(r);
                // Copy base weights (frozen)
                memcpy(lora_layers[L].Wq_base, lw[L].Wq, WQ_SZ*4);
                memcpy(lora_layers[L].Wk_base, lw[L].Wk, WK_SZ*4);
                memcpy(lora_layers[L].Wv_base, lw[L].Wv, WV_SZ*4);
                memcpy(lora_layers[L].Wo_base, lw[L].Wo, WO_SZ*4);
                // Init A with small random, B with zero (so LoRA starts as identity)
                for (size_t i = 0; i < (size_t)r*DIM; i++) lora_layers[L].Aq[i] = a_scale*(2*drand48()-1);
                for (size_t i = 0; i < (size_t)r*DIM; i++) lora_layers[L].Ak[i] = a_scale*(2*drand48()-1);
                for (size_t i = 0; i < (size_t)r*DIM; i++) lora_layers[L].Av[i] = a_scale*(2*drand48()-1);
                for (size_t i = 0; i < (size_t)r*Q_DIM; i++) lora_layers[L].Ao[i] = a_scale*(2*drand48()-1);
                // B matrices stay zero (calloc)
                lora_params += 2*((size_t)r*DIM) + (size_t)Q_DIM*r + (size_t)KV_DIM*r*2 + (size_t)KV_DIM*r + (size_t)r*Q_DIM + (size_t)DIM*r;
            }
            size_t rms_params = (size_t)NLAYERS * 2 * DIM + DIM;
            printf("LoRA: rank=%d, adapter params=%.1fK, trainable RMS params=%.1fK\n",
                   r, (float)lora_params/1e3, (float)rms_params/1e3);
            printf("  Adapters on: Wq, Wk, Wv, Wo (attention projections)\n");
            printf("  Frozen: W1, W2, W3, embed (FFN + embedding)\n");
        }

        // Precompute transposed weights for forward/backward kernels
        // Forward: sdpaFwd needs Wq^T[Q_DIM,DIM], Wk^T[KV_DIM,DIM], Wv^T[KV_DIM,DIM]
        //          woFwd needs Wo^T[DIM,Q_DIM]
        // Backward uses original (non-transposed) weights
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
            if (!cpu_only) {
                Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WK_SZ*4);
                Wvt_buf[L]=(float*)malloc(WV_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
                W1t_buf[L]=(float*)malloc(W1_SZ*4); W2t_buf[L]=(float*)malloc(W2_SZ*4);
                W3t_buf[L]=(float*)malloc(W3_SZ*4);
                transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
            } else {
                Wqt_buf[L]=NULL; Wkt_buf[L]=NULL; Wvt_buf[L]=NULL; Wot_buf[L]=NULL;
                W1t_buf[L]=NULL; W2t_buf[L]=NULL; W3t_buf[L]=NULL;
            }
        }

        // mmap token data
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { printf("Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { printf("mmap failed\n"); return 1; }
        size_t n_tokens = data_len / 2;
        printf("Token data: %zu tokens (%.1f MB)\n", n_tokens, data_len/1e6);

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d → %d active tokens (%.1fx reduction)\n", VOCAB, CV, (float)VOCAB/CV);

        float *cembed = vocab_compact_embed(embed, &vm, DIM);
        float *gcembed = (float*)calloc((size_t)CV*DIM, 4);
        AdamState acembed = adam_alloc((size_t)CV*DIM);

        // ===== Compile all kernels ONCE (skip for CPU-only mode) =====
        DynLayerKernels dk;
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        double compile_ms = 0;
        if (!cpu_only) {
            printf("Compiling 10 dynamic kernels (one-time)...\n");
            uint64_t tc = mach_absolute_time();
            if (!compile_dynamic_kernels(&dk, res_alpha)) {
                printf("Compilation failed!\n"); return 1;
            }
            compile_ms = tb_ms(mach_absolute_time() - tc);
            printf("Compiled 10 kernels in %.0fms (shared across all %d layers)\n", compile_ms, NLAYERS);

            // Allocate per-layer IOSurfaces + requests
            printf("Allocating per-layer IOSurfaces...\n");
            for (int L = 0; L < NLAYERS; L++) {
                pls[L].sdpaFwd_in    = make_surface(DIM*SDPA_FWD_SP*2);
                pls[L].woFwd_in      = make_surface(Q_DIM*WO_FWD_SP*2);
                pls[L].ffnFused_in   = make_surface(DIM*FFN_FUSED_SP*2);
                pls[L].ffnBwdW2t_in  = make_surface(DIM*FFN_BWD_W2T_SP*2);
                pls[L].ffnBwdW13t_in = make_surface(HIDDEN*FFN_BWD_W13T_SP*2);
                pls[L].wotBwd_in     = make_surface(DIM*WOT_BWD_SP*2);
                pls[L].qBwd_in       = make_surface(Q_DIM*Q_BWD_SP*2);
                pls[L].kvBwd_in      = make_surface(KV_DIM*KV_BWD_SP*2);

                plr[L].sdpaFwd   = make_request(dk.sdpaFwd,   pls[L].sdpaFwd_in);
                plr[L].woFwd     = make_request(dk.woFwd,     pls[L].woFwd_in);
                plr[L].ffnFused  = make_request(dk.ffnFused,  pls[L].ffnFused_in);
                plr[L].ffnBwdW2t = make_request(dk.ffnBwdW2t, pls[L].ffnBwdW2t_in);
                plr[L].ffnBwdW13t= make_request(dk.ffnBwdW13t,pls[L].ffnBwdW13t_in);
                plr[L].wotBwd    = make_request(dk.wotBwd,    pls[L].wotBwd_in);
                plr[L].qBwd      = make_request(dk.qBwd,      pls[L].qBwd_in);
                plr[L].kvBwd     = make_request(dk.kvBwd,     pls[L].kvBwd_in);
            }

            // Stage weights into per-layer surfaces
            for (int L = 0; L < NLAYERS; L++) {
                stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
                stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
                stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
                stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
                stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
                stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
                stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
                stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
            }
            printf("Per-layer weight staging complete\n\n");
        } else {
            memset(&dk, 0, sizeof(dk));
            memset(pls, 0, sizeof(pls));
            memset(plr, 0, sizeof(plr));
            printf("CPU-only mode: skipping ANE kernel compilation\n\n");
        }

        // Gradient + work buffers (GQA: Q has Q_DIM, K/V have KV_DIM)
        float *dy = (float*)malloc(SEQ*DIM*4);
        float *dffn = (float*)malloc(SEQ*DIM*4);
        float *dx_ffn = (float*)malloc(SEQ*DIM*4);
        float *dx2 = (float*)malloc(SEQ*DIM*4);
        float *dx_attn = (float*)malloc(SEQ*DIM*4);
        float *dq = (float*)malloc(SEQ*Q_DIM*4);     // Q_DIM for Q grads
        float *dk_buf = (float*)malloc(SEQ*KV_DIM*4); // KV_DIM for K grads
        float *dv = (float*)malloc(SEQ*KV_DIM*4);     // KV_DIM for V grads
        float *da_buf = (float*)malloc(SEQ*Q_DIM*4);  // Q_DIM for attn grads
        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(SEQ*CV*4);
        float *dlogits = (float*)malloc(SEQ*CV*4);
        float *gate_buf = (float*)malloc(SEQ*HIDDEN*4);
        float *dh1 = (float*)malloc(SEQ*HIDDEN*4);
        float *dh3 = (float*)malloc(SEQ*HIDDEN*4);
        float *dsilu = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp = (float*)malloc(SEQ*HIDDEN*4);
        float *silu_tmp2 = (float*)malloc(SEQ*HIDDEN*4);
        // GQA tile/reduce buffers
        float *k_tiled = (float*)malloc(SEQ*Q_DIM*4);  // KV_DIM → Q_DIM
        float *v_tiled = (float*)malloc(SEQ*Q_DIM*4);
        float *dq_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2
        float *dk_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd2 (needs reduce)
        float *dv_full = (float*)malloc(SEQ*Q_DIM*4);  // from sdpaBwd1 (needs reduce)

        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();
        srand48(42 + start_step);

        for (int step = start_step; step < total_steps; step++) {
            // Time budget check (skip first 5 steps for compile warmup)
            if (time_budget_sec > 0 && step > start_step + 5) {
                double elapsed_sec = tb_ms(mach_absolute_time() - t_wall_start) / 1000.0;
                if (elapsed_sec >= time_budget_sec) {
                    printf("Time budget %.0fs reached at step %d (%.1fs elapsed)\n", time_budget_sec, step, elapsed_sec);
                    total_steps = step;
                    break;
                }
            }
            uint64_t t0, t1, t_step = mach_absolute_time();

            // Sample data
            size_t max_pos = n_tokens - SEQ - 1;
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_tokens_raw = token_data + pos + 1;

            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_tokens_raw[t]];

            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ);

            double t_rms=0, t_ane_fwd=0, t_io_fwd=0, t_cblas_wait=0;
            double t_ane_bwd=0, t_io_bwd=0, t_silu=0, t_rms_bwd=0, t_cls=0, t_dw_copy=0;

            // ===== FORWARD (28 layers) =====
            for (int L=0; L<NLAYERS; L++) {
                LayerActs *ac = &acts[L];
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);

                // RMSNorm1 (CPU)
                t0 = mach_absolute_time();
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                t_rms += tb_ms(mach_absolute_time() - t0);

                // Wait for any pending dW cblas
                t0 = mach_absolute_time();
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t_cblas_wait += tb_ms(mach_absolute_time() - t0);

                if (cpu_only) {
                    // === CPU SDPA forward: QKV proj + RoPE + GQA tile + attention ===
                    t0 = mach_absolute_time();
                    // Q = Wq @ xnorm: [Q_DIM,DIM] @ [DIM,SEQ] → [Q_DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, ac->Q, SEQ);
                    // K = Wk @ xnorm: [KV_DIM,DIM] @ [DIM,SEQ] → [KV_DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, ac->K, SEQ);
                    // V = Wv @ xnorm: [KV_DIM,DIM] @ [DIM,SEQ] → [KV_DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, ac->V, SEQ);
                    // RoPE on Q and K
                    rope_forward_inplace(ac->Q, SEQ, Q_DIM, HD);
                    rope_forward_inplace(ac->K, SEQ, KV_DIM, HD);
                    // GQA tile K,V from KV_DIM → Q_DIM
                    gqa_tile_kv(k_tiled, ac->K, SEQ);
                    gqa_tile_kv(v_tiled, ac->V, SEQ);
                    // SDPA: causal attention
                    cpu_sdpa_forward(ac->Q, k_tiled, v_tiled, ac->attn_out, HEADS, HD, SEQ);
                    t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                    // === CPU Wo forward: o_out = Wo @ attn_out ===
                    t0 = mach_absolute_time();
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, ac->attn_out, SEQ, 0.0f, ac->o_out, SEQ);
                    t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                } else {
                // SDPA forward (ANE): xnorm + Wq,Wk,Wv → attn_out[Q_DIM], Q_rope[Q_DIM], K_rope[KV_DIM], V[KV_DIM], xnorm[DIM]
                t0 = mach_absolute_time();
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.sdpaFwd, plr[L].sdpaFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA output: [1, Q_DIM+Q_DIM+KV_DIM+KV_DIM+DIM, 1, SEQ] fp16
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                int off = 0;
                cvt_f16_f32(ac->attn_out, fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                cvt_f16_f32(ac->Q,        fwd_out + off, Q_DIM*SEQ); off += Q_DIM*SEQ;
                cvt_f16_f32(ac->K,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                cvt_f16_f32(ac->V,        fwd_out + off, KV_DIM*SEQ); off += KV_DIM*SEQ;
                // xnorm passthrough (DIM*SEQ) — not needed, already saved
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);

                // Wo forward (ANE): attn_out[Q_DIM] → o_out[DIM]
                t0 = mach_absolute_time();
                write_wo_fwd_acts(pls[L].woFwd_in, ac->attn_out);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.woFwd, plr[L].woFwd);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                io_read_dyn(dk.woFwd->ioOut, ac->o_out, DIM, SEQ);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                }

                // CPU: scaled residual + RMSNorm
                t0 = mach_absolute_time();
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                if (act_clamp > 0) {
                    float lo = -act_clamp, hi = act_clamp;
                    vDSP_vclip(ac->x2, 1, &lo, &hi, ac->x2, 1, (vDSP_Length)(SEQ*DIM));
                }
                rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                t_rms += tb_ms(mach_absolute_time() - t0);

                if (cpu_only) {
                    // === CPU FFN forward: W1/W3 + SiLU + W2 + residual ===
                    t0 = mach_absolute_time();
                    // h1 = W1 @ x2norm: [HIDDEN,DIM] @ [DIM,SEQ] → [HIDDEN,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, ac->x2norm, SEQ, 0.0f, ac->h1, SEQ);
                    // h3 = W3 @ x2norm: [HIDDEN,DIM] @ [DIM,SEQ] → [HIDDEN,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, ac->x2norm, SEQ, 0.0f, ac->h3, SEQ);
                    // SiLU(h1) * h3 → silu_out
                    {
                        int n = HIDDEN*SEQ;
                        float minus1 = -1.0f, one = 1.0f;
                        // sig = 1/(1+exp(-h1))
                        vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                        vvexpf(silu_tmp, silu_tmp, &n);
                        vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                        vvrecf(silu_tmp, silu_tmp, &n);  // sig
                        // silu_out = h1 * sig * h3
                        vDSP_vmul(ac->h1, 1, silu_tmp, 1, ac->silu_out, 1, (vDSP_Length)n);
                        vDSP_vmul(ac->silu_out, 1, ac->h3, 1, ac->silu_out, 1, (vDSP_Length)n);
                    }
                    // ffn_out = W2 @ silu_out: [DIM,HIDDEN] @ [HIDDEN,SEQ] → [DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, ac->silu_out, SEQ, 0.0f, ac->ffn_out, SEQ);
                    // x_cur = res_alpha * ffn_out + x2
                    vDSP_vsma(ac->ffn_out, 1, &res_alpha, ac->x2, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    if (act_clamp > 0) {
                        float lo = -act_clamp, hi = act_clamp;
                        vDSP_vclip(x_cur, 1, &lo, &hi, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    }
                    t_ane_fwd += tb_ms(mach_absolute_time() - t0);
                } else {
                // Fused FFN (ANE)
                t0 = mach_absolute_time();
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval_req(dk.ffnFused, plr[L].ffnFused);
                t_ane_fwd += tb_ms(mach_absolute_time() - t0);

                // Read fused output: [1, DIM+3*HIDDEN, 1, SEQ]
                t0 = mach_absolute_time();
                IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                int off = 0;
                cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
                IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                if (act_clamp > 0) {
                    float lo = -act_clamp, hi = act_clamp;
                    vDSP_vclip(x_cur, 1, &lo, &hi, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                }
                t_io_fwd += tb_ms(mach_absolute_time() - t0);
                }
            }

            // Final RMSNorm + classifier + loss (CPU)
            t0 = mach_absolute_time();
            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);
            t_rms += tb_ms(mach_absolute_time() - t0);
            // Classifier forward: logits[SEQ,CV] = x_final^T @ cembed^T (row-major, contiguous per-token)
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, x_final, SEQ, cembed, DIM, 0.0f, logits, CV);
            float loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);
            last_loss = loss;

            // ===== BACKWARD =====
            vDSP_vsmul(dlogits, 1, &loss_scale, dlogits, 1, (vDSP_Length)(SEQ*CV));

            // Classifier backward: dy[DIM,SEQ] = cembed^T @ dlogits^T
            // dlogits is [SEQ,CV], we need dy[d,t] = sum_v(cembed[v,d] * dlogits[t,v])
            t0 = mach_absolute_time();
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        DIM, SEQ, CV, 1.0f, cembed, DIM, dlogits, CV, 0.0f, dy, SEQ);
            t_cls += tb_ms(mach_absolute_time() - t0);

            // dEmbed async: gcembed[CV,DIM] += dlogits^T @ x_final^T
            dispatch_group_async(dw_grp, dw_q, ^{
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            CV, DIM, SEQ, 1.0f, dlogits, CV, x_final, SEQ, 1.0f, gcembed, DIM);
            });

            // Final RMSNorm backward
            float *dx_rms_final = (float*)calloc(SEQ*DIM, 4);
            rmsnorm_bwd(dx_rms_final, grms_final, dy, x_cur, rms_final, DIM, SEQ);
            memcpy(dy, dx_rms_final, SEQ*DIM*4);
            free(dx_rms_final);

            // ===== BACKWARD (28 layers, reverse) =====
            for (int L=NLAYERS-1; L>=0; L--) {
                LayerActs *ac = &acts[L];
                LayerGrads *gr = &grads[L];

                // dffn = alpha * dy
                vDSP_vsmul(dy, 1, &res_alpha, dffn, 1, (vDSP_Length)(SEQ*DIM));

                // FFN backward: dffn @ W2^T → dsilu_raw
                t0 = mach_absolute_time();
                if (cpu_only) {
                    // CPU: dsilu = W2^T @ dffn: W2 is [DIM,HIDDEN], W2^T is [HIDDEN,DIM]
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W2, HIDDEN, dffn, SEQ, 0.0f, dsilu, SEQ);
                } else {
                    write_ffn_bwd_w2t_acts(pls[L].ffnBwdW2t_in, dffn);
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    ane_eval_req(dk.ffnBwdW2t, plr[L].ffnBwdW2t);
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    io_read_dyn(dk.ffnBwdW2t->ioOut, dsilu, HIDDEN, SEQ);
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                }
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SiLU derivative (vectorized)
                t0 = mach_absolute_time();
                {
                    int n = HIDDEN*SEQ;
                    float minus1 = -1.0f, one = 1.0f;
                    vDSP_vsmul(ac->h1, 1, &minus1, silu_tmp, 1, (vDSP_Length)n);
                    vvexpf(silu_tmp, silu_tmp, &n);
                    vDSP_vsadd(silu_tmp, 1, &one, silu_tmp, 1, (vDSP_Length)n);
                    vvrecf(silu_tmp, silu_tmp, &n);  // sig
                    vDSP_vmul(ac->h1, 1, silu_tmp, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, dh3, 1, dh3, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp, 1, &minus1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vneg(silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(ac->h1, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vsadd(silu_tmp2, 1, &one, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(silu_tmp, 1, silu_tmp2, 1, silu_tmp2, 1, (vDSP_Length)n);
                    vDSP_vmul(dsilu, 1, ac->h3, 1, dh1, 1, (vDSP_Length)n);
                    vDSP_vmul(dh1, 1, silu_tmp2, 1, dh1, 1, (vDSP_Length)n);
                }
                t_silu += tb_ms(mach_absolute_time() - t0);

                // dh1@W1^T + dh3@W3^T → dx_ffn
                t0 = mach_absolute_time();
                if (cpu_only) {
                    // CPU: dx_ffn = W1^T @ dh1 + W3^T @ dh3
                    // W1 is [HIDDEN,DIM], W1^T is [DIM,HIDDEN]
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W1, DIM, dh1, SEQ, 0.0f, dx_ffn, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W3, DIM, dh3, SEQ, 1.0f, dx_ffn, SEQ);
                } else {
                    write_ffn_bwd_w13t_acts(pls[L].ffnBwdW13t_in, dh1, dh3);
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    ane_eval_req(dk.ffnBwdW13t, plr[L].ffnBwdW13t);
                    t_ane_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                    io_read_dyn(dk.ffnBwdW13t->ioOut, dx_ffn, DIM, SEQ);
                    t_io_bwd += tb_ms(mach_absolute_time() - t0);
                    t0 = mach_absolute_time();
                }
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // dW FFN async
                t0 = mach_absolute_time();
                float *capt_dffn = (float*)malloc(SEQ*DIM*4); memcpy(capt_dffn, dffn, SEQ*DIM*4);
                float *capt_silu = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_silu, ac->silu_out, SEQ*HIDDEN*4);
                float *capt_dh1 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh1, dh1, SEQ*HIDDEN*4);
                float *capt_dh3 = (float*)malloc(SEQ*HIDDEN*4); memcpy(capt_dh3, dh3, SEQ*HIDDEN*4);
                float *capt_x2n = (float*)malloc(SEQ*DIM*4); memcpy(capt_x2n, ac->x2norm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, capt_dffn, SEQ, capt_silu, SEQ, 1.0f, gr->W2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh1, SEQ, capt_x2n, SEQ, 1.0f, gr->W1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, capt_dh3, SEQ, capt_x2n, SEQ, 1.0f, gr->W3, DIM);
                    free(capt_dffn); free(capt_silu); free(capt_dh1); free(capt_dh3); free(capt_x2n);
                });

                // RMSNorm2 backward
                t0 = mach_absolute_time();
                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, gr->rms_ffn, dx_ffn, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);

                // Wo^T backward: alpha*dx2 @ Wo → da[Q_DIM]
                float *dx2_scaled = (float*)malloc(SEQ*DIM*4);
                vDSP_vsmul(dx2, 1, &res_alpha, dx2_scaled, 1, (vDSP_Length)(SEQ*DIM));
                t0 = mach_absolute_time();
                if (use_cpu_attn_bwd) {
                    // CPU fp32: da = Wo^T @ dx2_scaled  [Q_DIM,DIM]^T @ [DIM,SEQ] → [Q_DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wo, Q_DIM, dx2_scaled, SEQ,
                                0.0f, da_buf, SEQ);
                } else {
                    // ANE fp16 path
                    write_wot_bwd_acts(pls[L].wotBwd_in, dx2_scaled);
                    ane_eval_req(dk.wotBwd, plr[L].wotBwd);
                    io_read_dyn(dk.wotBwd->ioOut, da_buf, Q_DIM, SEQ);
                }
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // dWo async: gr->Wo[DIM,Q_DIM] += dx2_scaled[DIM,SEQ] @ attn_out^T[SEQ,Q_DIM]
                t0 = mach_absolute_time();
                float *capt_do = (float*)malloc(SEQ*DIM*4); memcpy(capt_do, dx2_scaled, SEQ*DIM*4);
                free(dx2_scaled);
                float *capt_attn = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_attn, ac->attn_out, SEQ*Q_DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, Q_DIM, SEQ,
                                1.0f, capt_do, SEQ, capt_attn, SEQ, 1.0f, gr->Wo, Q_DIM);
                    free(capt_do); free(capt_attn);
                });

                // GQA: tile K,V from KV_DIM → Q_DIM for SDPA backward
                t0 = mach_absolute_time();
                gqa_tile_kv(k_tiled, ac->K, SEQ);
                gqa_tile_kv(v_tiled, ac->V, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward: CPU fp32 path (accurate) or ANE fp16 path (fast but underflows)
                t0 = mach_absolute_time();
                if (use_cpu_attn_bwd) {
                    // CPU fp32 SDPA backward — full precision, no underflow
                    cpu_sdpa_backward(ac->Q, k_tiled, v_tiled, da_buf,
                                      dq_full, dk_full, dv_full, HEADS, HD, SEQ);
                } else {
                // ANE fp16 SDPA backward (matches original maderix/ANE — no extra scaling)
                t0 = mach_absolute_time();
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 0,       ac->Q,    Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, Q_DIM,   k_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 2*Q_DIM, v_tiled,  Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd1->ioIn, 3*Q_DIM, da_buf,   Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd1);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // SDPA backward part 2: probs,dp,Q,K_tiled → dQ,dK_full
                t0 = mach_absolute_time();
                io_copy(dk.sdpaBwd2->ioIn, 0, dk.sdpaBwd1->ioOut, Q_DIM, 2*SCORE_CH, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH,       ac->Q,   Q_DIM, SEQ);
                io_write_fp16_at(dk.sdpaBwd2->ioIn, 2*SCORE_CH+Q_DIM, k_tiled, Q_DIM, SEQ);
                t_io_bwd += tb_ms(mach_absolute_time() - t0);
                t0 = mach_absolute_time();
                ane_eval(dk.sdpaBwd2);
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // Read SDPA backward outputs
                t0 = mach_absolute_time();
                io_read_fp16(dk.sdpaBwd2->ioOut, dq_full, 0,     Q_DIM, SEQ);
                io_read_fp16(dk.sdpaBwd2->ioOut, dk_full, Q_DIM, Q_DIM, SEQ);
                io_read_fp16(dk.sdpaBwd1->ioOut, dv_full, 0,     Q_DIM, SEQ);
                } // end else (ANE path)
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // GQA: reduce dK, dV from Q_DIM (HEADS) → KV_DIM (KV_HEADS)
                gqa_reduce_kv(dk_buf, dk_full, SEQ);
                gqa_reduce_kv(dv, dv_full, SEQ);
                // dQ stays at Q_DIM — no reduction needed
                memcpy(dq, dq_full, SEQ*Q_DIM*4);

                // RoPE backward on dQ[Q_DIM] and dK[KV_DIM]
                rope_backward_inplace(dq, SEQ, Q_DIM, HD);
                rope_backward_inplace(dk_buf, SEQ, KV_DIM, HD);

                if (L == 0 && step % 10 == 0) {
                    float dqmx, dkmx, dvmx;
                    vDSP_maxmgv(dq, 1, &dqmx, (vDSP_Length)(SEQ*Q_DIM));
                    vDSP_maxmgv(dk_buf, 1, &dkmx, (vDSP_Length)(SEQ*KV_DIM));
                    vDSP_maxmgv(dv, 1, &dvmx, (vDSP_Length)(SEQ*KV_DIM));
                    printf("    L0 sdpa_bwd: |dq|=%.6f |dk|=%.6f |dv|=%.6f\n", dqmx, dkmx, dvmx);
                }

                // dWq/dWk/dWv async
                // dWq[Q_DIM,DIM] += dq[Q_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWk[KV_DIM,DIM] += dk[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                // dWv[KV_DIM,DIM] += dv[KV_DIM,SEQ] @ xnorm^T[SEQ,DIM]
                t0 = mach_absolute_time();
                float *capt_dq = (float*)malloc(SEQ*Q_DIM*4); memcpy(capt_dq, dq, SEQ*Q_DIM*4);
                float *capt_dk = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dk, dk_buf, SEQ*KV_DIM*4);
                float *capt_dv = (float*)malloc(SEQ*KV_DIM*4); memcpy(capt_dv, dv, SEQ*KV_DIM*4);
                float *capt_xn = (float*)malloc(SEQ*DIM*4); memcpy(capt_xn, ac->xnorm, SEQ*DIM*4);
                t_dw_copy += tb_ms(mach_absolute_time() - t0);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q_DIM, DIM, SEQ,
                                1.0f, capt_dq, SEQ, capt_xn, SEQ, 1.0f, gr->Wq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dk, SEQ, capt_xn, SEQ, 1.0f, gr->Wk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, KV_DIM, DIM, SEQ,
                                1.0f, capt_dv, SEQ, capt_xn, SEQ, 1.0f, gr->Wv, DIM);
                    free(capt_dq); free(capt_dk); free(capt_dv); free(capt_xn);
                });

                // Q backward: dq[Q_DIM] @ Wq^T → dx_q[DIM]
                // KV backward: dk[KV_DIM]@Wk^T + dv[KV_DIM]@Wv^T → dx_kv[DIM]
                float *dx_kv = (float*)malloc(SEQ*DIM*4);
                t0 = mach_absolute_time();
                if (use_cpu_attn_bwd) {
                    // CPU fp32: dx_q = Wq^T @ dq  [DIM,Q_DIM]^T is [DIM,Q_DIM] transposed
                    // Wq is [Q_DIM, DIM], so Wq^T is [DIM, Q_DIM]
                    // dx_attn[DIM,SEQ] = Wq^T[DIM,Q_DIM] @ dq[Q_DIM,SEQ]
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wq, DIM, dq, SEQ,
                                0.0f, dx_attn, SEQ);
                    // dx_kv[DIM,SEQ] = Wk^T @ dk + Wv^T @ dv
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, KV_DIM, 1.0f, lw[L].Wk, DIM, dk_buf, SEQ,
                                0.0f, dx_kv, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                DIM, SEQ, KV_DIM, 1.0f, lw[L].Wv, DIM, dv, SEQ,
                                1.0f, dx_kv, SEQ);  // beta=1 to accumulate
                } else {
                    // ANE fp16 path
                    write_q_bwd_acts(pls[L].qBwd_in, dq);
                    ane_eval_req(dk.qBwd, plr[L].qBwd);
                    io_read_dyn(dk.qBwd->ioOut, dx_attn, DIM, SEQ);
                    write_kv_bwd_acts(pls[L].kvBwd_in, dk_buf, dv);
                    ane_eval_req(dk.kvBwd, plr[L].kvBwd);
                    io_read_dyn(dk.kvBwd->ioOut, dx_kv, DIM, SEQ);
                }
                t_ane_bwd += tb_ms(mach_absolute_time() - t0);

                // dx_attn = dx_q + dx_kv
                for(int i=0; i<SEQ*DIM; i++) dx_attn[i] += dx_kv[i];
                free(dx_kv);

                // RMSNorm1 backward
                t0 = mach_absolute_time();
                float *dx_rms1 = (float*)calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms1, gr->rms_att, dx_attn, ac->layer_in, lw[L].rms_att, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dy[i] = dx_rms1[i] + dx2[i];
                free(dx_rms1);
                t_rms_bwd += tb_ms(mach_absolute_time() - t0);
            }

            // Embedding backward
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            embed_backward(gembed, dy, input_tokens, DIM, SEQ);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;

            // Adaptive ANE→CPU switch: check activation magnitude every step
            if (adaptive_thresh > 0 && !cpu_only) {
                float xmx_abs;
                vDSP_maxmgv(x_cur, 1, &xmx_abs, (vDSP_Length)(SEQ*DIM));
                if (xmx_abs > adaptive_thresh) {
                    adaptive_above_count++;
                    if (adaptive_above_count >= adaptive_window) {
                        printf("\n=== ADAPTIVE SWITCH: ANE → CPU at step %d ===\n", step);
                        printf("  |x|_max=%.1f exceeded threshold %.0f for %d consecutive steps\n",
                               xmx_abs, adaptive_thresh, adaptive_above_count);
                        cpu_only = true;
                        use_cpu_attn_bwd = true;
                        adaptive_switch_step = step;
                        printf("  Continuing training on CPU fp32 (weights already in fp32, no conversion needed)\n\n");
                    }
                } else {
                    adaptive_above_count = 0;  // Reset counter if below threshold
                }
            }

            if (step % 10 == 0 || step == start_step) {
                printf("  timing: ane_fwd=%.1f io_fwd=%.1f rms=%.1f ane_bwd=%.1f io_bwd=%.1f silu=%.1f rms_bwd=%.1f cls=%.1f cblas_wait=%.1f dw_copy=%.1f\n",
                       t_ane_fwd, t_io_fwd, t_rms, t_ane_bwd, t_io_bwd, t_silu, t_rms_bwd, t_cls, t_cblas_wait, t_dw_copy);
                float xmx, xmn;
                vDSP_maxv(x_cur,1,&xmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(x_cur,1,&xmn,(vDSP_Length)(SEQ*DIM));
                float dmx, dmn;
                vDSP_maxv(dy,1,&dmx,(vDSP_Length)(SEQ*DIM));
                vDSP_minv(dy,1,&dmn,(vDSP_Length)(SEQ*DIM));
                printf("step %-4d loss=%.4f  lr=%.2e  %.1fms/step  x[%.2f,%.2f] dy[%.3e,%.3e]\n",
                       step, loss, lr, step_ms, xmn, xmx, dmn, dmx);
            }

            // Adam update every accum_steps
            if ((step+1) % accum_steps == 0 || step == total_steps-1) {
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                float gsc = 1.0f / (accum_steps * loss_scale);
                adam_t++;

                // Scale gradients
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    for(size_t i=0;i<WQ_SZ;i++) g->Wq[i]*=gsc;
                    for(size_t i=0;i<WK_SZ;i++) g->Wk[i]*=gsc;
                    for(size_t i=0;i<WV_SZ;i++) g->Wv[i]*=gsc;
                    for(size_t i=0;i<WO_SZ;i++) g->Wo[i]*=gsc;
                    for(size_t i=0;i<W1_SZ;i++) g->W1[i]*=gsc;
                    for(size_t i=0;i<W2_SZ;i++) g->W2[i]*=gsc;
                    for(size_t i=0;i<W3_SZ;i++) g->W3[i]*=gsc;
                    for(int i=0;i<DIM;i++){g->rms_att[i]*=gsc; g->rms_ffn[i]*=gsc;}
                }
                for(int i=0;i<DIM;i++) grms_final[i]*=gsc;
                vocab_scatter_grads(gembed, gcembed, &vm, DIM);
                for(size_t i=0;i<(size_t)VOCAB*DIM;i++) gembed[i]*=gsc;

                // Gradient sanitization: NaN→0, ±Inf→±65504 (per Orion paper)
                if (grad_sanitize) {
                    int step_sanitized = 0;
                    for (int L=0; L<NLAYERS; L++) step_sanitized += sanitize_layer_grads(&grads[L]);
                    step_sanitized += sanitize_gradients(grms_final, DIM);
                    step_sanitized += sanitize_gradients(gembed, (size_t)VOCAB*DIM);
                    total_sanitized += step_sanitized;
                    if (step_sanitized > 0) {
                        printf("  [sanitize] step %d: fixed %d NaN/Inf gradient values (total: %d)\n",
                               step, step_sanitized, total_sanitized);
                    }
                }

                // Global gradient norm
                float grad_norm_sq = 0;
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    float s;
                    vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_att,1,g->rms_att,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                    vDSP_dotpr(g->rms_ffn,1,g->rms_ffn,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                }
                { float s;
                  vDSP_dotpr(grms_final,1,grms_final,1,&s,(vDSP_Length)DIM); grad_norm_sq+=s;
                  vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); grad_norm_sq+=s;
                }
                float grad_norm = sqrtf(grad_norm_sq);
                if ((step+1) % 10 == 0) {
                    float attn_sq=0, ffn_sq=0, embed_sq=0;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L]; float s;
                        vDSP_dotpr(g->Wq,1,g->Wq,1,&s,(vDSP_Length)WQ_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wk,1,g->Wk,1,&s,(vDSP_Length)WK_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wv,1,g->Wv,1,&s,(vDSP_Length)WV_SZ); attn_sq+=s;
                        vDSP_dotpr(g->Wo,1,g->Wo,1,&s,(vDSP_Length)WO_SZ); attn_sq+=s;
                        vDSP_dotpr(g->W1,1,g->W1,1,&s,(vDSP_Length)W1_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W2,1,g->W2,1,&s,(vDSP_Length)W2_SZ); ffn_sq+=s;
                        vDSP_dotpr(g->W3,1,g->W3,1,&s,(vDSP_Length)W3_SZ); ffn_sq+=s;
                    }
                    { float s;
                      vDSP_dotpr(gembed,1,gembed,1,&s,(vDSP_Length)(VOCAB*DIM)); embed_sq=s;
                    }
                    printf("  grad_norm=%.4f  attn=%.4f ffn=%.4f embed=%.4f\n",
                           grad_norm, sqrtf(attn_sq), sqrtf(ffn_sq), sqrtf(embed_sq));
                }

                // Gradient clipping
                if (grad_clip > 0 && grad_norm > grad_clip) {
                    float clip_scale = grad_clip / grad_norm;
                    for (int L=0; L<NLAYERS; L++) {
                        LayerGrads *g = &grads[L];
                        vDSP_vsmul(g->Wq,1,&clip_scale,g->Wq,1,(vDSP_Length)WQ_SZ);
                        vDSP_vsmul(g->Wk,1,&clip_scale,g->Wk,1,(vDSP_Length)WK_SZ);
                        vDSP_vsmul(g->Wv,1,&clip_scale,g->Wv,1,(vDSP_Length)WV_SZ);
                        vDSP_vsmul(g->Wo,1,&clip_scale,g->Wo,1,(vDSP_Length)WO_SZ);
                        vDSP_vsmul(g->W1,1,&clip_scale,g->W1,1,(vDSP_Length)W1_SZ);
                        vDSP_vsmul(g->W2,1,&clip_scale,g->W2,1,(vDSP_Length)W2_SZ);
                        vDSP_vsmul(g->W3,1,&clip_scale,g->W3,1,(vDSP_Length)W3_SZ);
                        vDSP_vsmul(g->rms_att,1,&clip_scale,g->rms_att,1,(vDSP_Length)DIM);
                        vDSP_vsmul(g->rms_ffn,1,&clip_scale,g->rms_ffn,1,(vDSP_Length)DIM);
                    }
                    vDSP_vsmul(grms_final,1,&clip_scale,grms_final,1,(vDSP_Length)DIM);
                    vDSP_vsmul(gembed,1,&clip_scale,gembed,1,(vDSP_Length)(VOCAB*DIM));
                }

                // Cosine LR schedule with warmup
                if (step < warmup_steps) {
                    lr = max_lr * ((float)(step + 1)) / warmup_steps;
                } else {
                    float decay_ratio = (float)(step - warmup_steps) / (float)(total_steps - warmup_steps);
                    float min_lr = max_lr * min_lr_frac;
                    lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_ratio)) * (max_lr - min_lr);
                }

                // Adam update
                for (int L=0; L<NLAYERS; L++) {
                    LayerGrads *g = &grads[L];
                    if (use_lora) {
                        // Project full weight grads → LoRA grads, then update adapters
                        LoRALayer *ll = &lora_layers[L];
                        LoRAGrads *lg = &lora_grads_arr[L];
                        LoRAAdam *la_l = &lora_adam[L];
                        int r = ll->rank;
                        lora_grad_project(lg->Aq, lg->Bq, g->Wq, ll->Aq, ll->Bq, Q_DIM, r, DIM);
                        lora_grad_project(lg->Ak, lg->Bk, g->Wk, ll->Ak, ll->Bk, KV_DIM, r, DIM);
                        lora_grad_project(lg->Av, lg->Bv, g->Wv, ll->Av, ll->Bv, KV_DIM, r, DIM);
                        lora_grad_project(lg->Ao, lg->Bo, g->Wo, ll->Ao, ll->Bo, DIM, r, Q_DIM);
                        adam_update(ll->Aq, lg->Aq, &la_l->Aq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Bq, lg->Bq, &la_l->Bq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Ak, lg->Ak, &la_l->Ak, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Bk, lg->Bk, &la_l->Bk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Av, lg->Av, &la_l->Av, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Bv, lg->Bv, &la_l->Bv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Ao, lg->Ao, &la_l->Ao, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        adam_update(ll->Bo, lg->Bo, &la_l->Bo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                        // Merge: W_eff = W_base + B @ A
                        lora_merge_weight(lw[L].Wq, ll->Wq_base, ll->Bq, ll->Aq, Q_DIM, r, DIM);
                        lora_merge_weight(lw[L].Wk, ll->Wk_base, ll->Bk, ll->Ak, KV_DIM, r, DIM);
                        lora_merge_weight(lw[L].Wv, ll->Wv_base, ll->Bv, ll->Av, KV_DIM, r, DIM);
                        lora_merge_weight(lw[L].Wo, ll->Wo_base, ll->Bo, ll->Ao, DIM, r, Q_DIM);
                        // RMS norms still trainable (small, no LoRA needed)
                        adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                        adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                        // FFN weights frozen — no update for W1, W2, W3
                    } else {
                    adam_update(lw[L].Wq, g->Wq, &la[L].Wq, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wk, g->Wk, &la[L].Wk, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wv, g->Wv, &la[L].Wv, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].Wo, g->Wo, &la[L].Wo, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W1, g->W1, &la[L].W1, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W2, g->W2, &la[L].W2, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].W3, g->W3, &la[L].W3, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(lw[L].rms_att, g->rms_att, &la[L].rms_att, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, g->rms_ffn, &la[L].rms_ffn, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    }

                    if (!cpu_only) {
                    // Update transposed weight buffers
                    transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
                    transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                    transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                    transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
                    transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
                    transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
                    transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);

                    // Re-stage weights
                    stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
                    stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
                    stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
                    stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
                    stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
                    stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
                    stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
                    stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
                    }
                }
                adam_update(rms_final, grms_final, &arms_final, adam_t, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                if (!use_lora) {
                    adam_update(embed, gembed, &aembed, adam_t, lr, adam_b1, adam_b2, adam_eps, wd);
                    free(cembed);
                    cembed = vocab_compact_embed(embed, &vm, DIM);
                }

                // Zero grads
                for (int L=0; L<NLAYERS; L++) layer_grads_zero(&grads[L]);
                if (use_lora) {
                    for (int L=0; L<NLAYERS; L++) lora_grads_zero(&lora_grads_arr[L], lora_rank);
                }
                memset(grms_final, 0, DIM*4);
                memset(gembed, 0, (size_t)VOCAB*DIM*4);
                memset(gcembed, 0, (size_t)CV*DIM*4);

                // Checkpoint — only save on best loss
                if ((step+1) % 100 == 0 && last_loss < best_loss) {
                    best_loss = last_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    save_checkpoint(CKPT_PATH, step+1, total_steps, lr, last_loss,
                        total_train_ms+cum_train, wall+cum_wall, total_steps_done+cum_steps, adam_t,
                        lw, la, rms_final, &arms_final, embed, &aembed);
                    printf("  [ckpt saved, best_loss=%.4f]\n", best_loss);
                }
            }
        }

        // Report
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        double train_sec = total_train_ms / 1000.0;
        double wall_sec = (wall + cum_wall) / 1000.0;
        double wq_sz=WQ_SZ, wk_sz=WK_SZ, wv_sz=WV_SZ, wo_sz=WO_SZ;
        double w1_sz=W1_SZ, w2_sz=W2_SZ, w3_sz=W3_SZ;
        double layer_p = wq_sz+wk_sz+wv_sz+wo_sz+w1_sz+w2_sz+w3_sz+2.0*DIM;
        double total_p = (double)NLAYERS*layer_p + DIM + (double)VOCAB*DIM;
        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        if (!cpu_only)
            printf("Compile:      %.0fms (one-time, %.1f%%)\n", compile_ms, 100*compile_ms/(wall+cum_wall));
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms/fmax(1,total_steps_done));
        printf("Wall time:    %.1fs\n", wall_sec);
        // Machine-parseable output for autoresearch
        printf("\n---\n");
        printf("final_loss:       %.6f\n", last_loss);
        printf("training_seconds: %.1f\n", train_sec);
        printf("total_seconds:    %.1f\n", wall_sec);
        printf("total_tokens_M:   %.1f\n", (double)total_steps_done * SEQ / 1e6);
        printf("num_steps:        %d\n", total_steps_done);
        printf("num_params_M:     %.1f\n", total_p / 1e6);
        printf("depth:            %d\n", NLAYERS);
        if (use_lora) printf("lora_rank:        %d\n", lora_rank);
        if (cpu_only) printf("mode:             cpu-only\n");
        if (act_clamp > 0) printf("act_clamp:        %.1f\n", act_clamp);
        if (grad_sanitize) printf("grad_sanitized:   %d\n", total_sanitized);
        if (adaptive_switch_step >= 0) printf("adaptive_switch:  %d\n", adaptive_switch_step);

        // Cleanup
        for (int L=0; L<NLAYERS; L++) {
            layer_weights_free(&lw[L]); layer_adam_free(&la[L]);
            layer_acts_free(&acts[L]); layer_grads_free(&grads[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
            if (use_lora) { lora_layer_free(&lora_layers[L]); lora_adam_free(&lora_adam[L]); lora_grads_free(&lora_grads_arr[L]); }
        }
        if (!cpu_only) {
            free_per_layer(pls, plr);
            free_kern(dk.sdpaFwd); free_kern(dk.woFwd); free_kern(dk.ffnFused);
            free_kern(dk.ffnBwdW2t); free_kern(dk.ffnBwdW13t); free_kern(dk.wotBwd);
            free_kern(dk.sdpaBwd1); free_kern(dk.sdpaBwd2);
            free_kern(dk.qBwd); free_kern(dk.kvBwd);
        }
        free(da_buf); free(k_tiled); free(v_tiled);
        free(dq_full); free(dk_full); free(dv_full);
        free(dq); free(dk_buf); free(dv);
        munmap(token_data, data_len); close(data_fd);
    }
    return 0;
}
