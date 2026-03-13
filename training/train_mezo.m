// train_mezo.m — Zeroth-Order (MeZO/SPSA) training on Apple Neural Engine
// Forward-pass only: no backward kernels, no gradients, no Adam state.
// Memory = inference memory (seed trick eliminates perturbation storage).
//
// Build: make mezo MODEL=autoresearch (or smollm2_135m, smollm2_360m)
// Usage: ./train_mezo --scratch --data data.bin --cpu-only --steps 1000
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include <math.h>

// Dynamic kernel set per layer (forward-only subset for MeZO)
typedef struct {
    // Fused forward kernels (legacy — fp16 RoPE+attention+SiLU+residual)
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    // Unfused forward kernels (matmul-only — RoPE+attention+SiLU+residual on CPU fp32)
    Kern *wqFwd;       // xnorm @ Wq → Q (DIM → Q_DIM)
    Kern *wkvFwd;      // xnorm @ Wk/Wv → K or V (DIM → KV_DIM) — shared kernel, separate surfaces
    Kern *w13Fwd;      // x2norm @ W1/W3 → h1 or h3 (DIM → HIDDEN) — shared kernel, separate surfaces
    Kern *w2Fwd;       // silu_out @ W2 → ffn_out (HIDDEN → DIM)
    // Backward kernels (unused in MeZO but kept for struct compatibility)
    Kern *ffnBwdW2t;
    Kern *ffnBwdW13t;
    Kern *wotBwd;
    Kern *sdpaBwd1;
    Kern *sdpaBwd2;
    Kern *qBwd;
    Kern *kvBwd;
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, (vDSP_Length)cols, (vDSP_Length)rows);
}

// Spatial (last dim) sizes for unfused matmul kernels: input is [1, IC, 1, SEQ+OC]
#define WQ_FWD_SP  (SEQ + Q_DIM)
#define WKV_FWD_SP (SEQ + KV_DIM)
#define W13_FWD_SP (SEQ + HIDDEN)
#define W2_FWD_SP  (SEQ + DIM)

// ===== Compile forward-only dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk, float res_alpha, bool unfused_fwd, bool compile_bwd) {
    (void)compile_bwd;  // MeZO never compiles backward kernels

    if (unfused_fwd) {
        // --- Unfused forward: individual matmul kernels ---
        printf("  Compiling wqFwd (DIM->Q_DIM)...\n");
        dk->wqFwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, Q_DIM, SEQ), @{},
            DIM*WQ_FWD_SP*2, Q_DIM*SEQ*2);
        if (!dk->wqFwd) return false;

        printf("  Compiling wkvFwd (DIM->KV_DIM)...\n");
        dk->wkvFwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, KV_DIM, SEQ), @{},
            DIM*WKV_FWD_SP*2, KV_DIM*SEQ*2);
        if (!dk->wkvFwd) return false;

        printf("  Compiling woFwd (Q_DIM->DIM)...\n");
        dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
            Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
        if (!dk->woFwd) return false;

        printf("  Compiling w13Fwd (DIM->HIDDEN)...\n");
        dk->w13Fwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, HIDDEN, SEQ), @{},
            DIM*W13_FWD_SP*2, HIDDEN*SEQ*2);
        if (!dk->w13Fwd) return false;

        printf("  Compiling w2Fwd (HIDDEN->DIM)...\n");
        dk->w2Fwd = compile_kern_mil_w(gen_dyn_matmul_mil(HIDDEN, DIM, SEQ), @{},
            HIDDEN*W2_FWD_SP*2, DIM*SEQ*2);
        if (!dk->w2Fwd) return false;
    }

    return true;
}

// ===== xoshiro256+ PRNG (fast, deterministic, high-quality) =====
// Used for Rademacher perturbation: z_i in {-1, +1}
// 33x faster than Box-Muller+drand48 (21ms vs 700ms per 36.4M params)
static uint64_t xo_s[4];

static inline uint64_t xo_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t xo_next(void) {
    uint64_t result = xo_s[0] + xo_s[3];
    uint64_t t = xo_s[1] << 17;
    xo_s[2] ^= xo_s[0]; xo_s[3] ^= xo_s[1]; xo_s[1] ^= xo_s[2]; xo_s[0] ^= xo_s[3];
    xo_s[2] ^= t; xo_s[3] = xo_rotl(xo_s[3], 45);
    return result;
}

// Initialize xoshiro from a single seed (splitmix64 expansion)
static void xo_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        xo_s[i] = z ^ (z >> 31);
    }
}

// ===== Rademacher perturbation: buf[i] += scale * z_i, z_i in {-1,+1} =====
// Extracts 4 bits per xoshiro call for maximum throughput
static void perturb_buffer(float *buf, size_t n, float scale) {
    float neg_scale = -scale;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        uint64_t r = xo_next();
        buf[i+0] += (r & 1) ? scale : neg_scale;
        buf[i+1] += (r & 2) ? scale : neg_scale;
        buf[i+2] += (r & 4) ? scale : neg_scale;
        buf[i+3] += (r & 8) ? scale : neg_scale;
    }
    for (; i < n; i++) {
        uint64_t r = xo_next();
        buf[i] += (r & 1) ? scale : neg_scale;
    }
}

// ===== Perturb ALL model weights using deterministic seed =====
// NOTE: perturbs `embed` (full VOCAB*DIM), not cembed. Caller must rebuild
// cembed = vocab_compact_embed(embed, &vm, DIM) before any forward pass.
static void perturb_all_weights(LayerWeights *lw, float *embed, float *rms_final,
                                uint64_t seed, float scale) {
    xo_seed(seed);
    perturb_buffer(embed, (size_t)VOCAB * DIM, scale);
    for (int L = 0; L < NLAYERS; L++) {
        perturb_buffer(lw[L].rms_att, DIM, scale);
        perturb_buffer(lw[L].Wq, WQ_SZ, scale);
        perturb_buffer(lw[L].Wk, WK_SZ, scale);
        perturb_buffer(lw[L].Wv, WV_SZ, scale);
        perturb_buffer(lw[L].Wo, WO_SZ, scale);
        perturb_buffer(lw[L].rms_ffn, DIM, scale);
        perturb_buffer(lw[L].W1, W1_SZ, scale);
        perturb_buffer(lw[L].W2, W2_SZ, scale);
        perturb_buffer(lw[L].W3, W3_SZ, scale);
    }
    perturb_buffer(rms_final, DIM, scale);
}

// ===== MeZO + LoRA: perturb ONLY adapter matrices =====
static void perturb_lora_weights(LoRALayer *ll, LayerWeights *lw,
                                 float *rms_final, int nlayers, uint64_t seed, float scale) {
    xo_seed(seed);
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        // Attention adapters
        perturb_buffer(ll[L].Aq, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bq, (size_t)Q_DIM * r, scale);
        perturb_buffer(ll[L].Ak, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bk, (size_t)KV_DIM * r, scale);
        perturb_buffer(ll[L].Av, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bv, (size_t)KV_DIM * r, scale);
        perturb_buffer(ll[L].Ao, (size_t)r * Q_DIM, scale);
        perturb_buffer(ll[L].Bo, (size_t)DIM * r, scale);
        // FFN adapters (if present)
        if (ll[L].has_ffn) {
            perturb_buffer(ll[L].A1, (size_t)r * DIM, scale);
            perturb_buffer(ll[L].B1, (size_t)HIDDEN * r, scale);
            perturb_buffer(ll[L].A2, (size_t)r * HIDDEN, scale);
            perturb_buffer(ll[L].B2, (size_t)DIM * r, scale);
            perturb_buffer(ll[L].A3, (size_t)r * DIM, scale);
            perturb_buffer(ll[L].B3, (size_t)HIDDEN * r, scale);
        }
        // RMS norms are still trainable (small, always perturbed)
        perturb_buffer(lw[L].rms_att, DIM, scale);
        perturb_buffer(lw[L].rms_ffn, DIM, scale);
    }
    perturb_buffer(rms_final, DIM, scale);
}

// Merge LoRA adapters into effective weights: W_eff = W_base + B @ A
static void lora_merge_all(LayerWeights *lw, LoRALayer *ll, int nlayers) {
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        lora_merge_weight(lw[L].Wq, ll[L].Wq_base, ll[L].Bq, ll[L].Aq, Q_DIM, r, DIM);
        lora_merge_weight(lw[L].Wk, ll[L].Wk_base, ll[L].Bk, ll[L].Ak, KV_DIM, r, DIM);
        lora_merge_weight(lw[L].Wv, ll[L].Wv_base, ll[L].Bv, ll[L].Av, KV_DIM, r, DIM);
        lora_merge_weight(lw[L].Wo, ll[L].Wo_base, ll[L].Bo, ll[L].Ao, DIM, r, Q_DIM);
        if (ll[L].has_ffn) {
            lora_merge_weight(lw[L].W1, ll[L].W1_base, ll[L].B1, ll[L].A1, HIDDEN, r, DIM);
            lora_merge_weight(lw[L].W2, ll[L].W2_base, ll[L].B2, ll[L].A2, DIM, r, HIDDEN);
            lora_merge_weight(lw[L].W3, ll[L].W3_base, ll[L].B3, ll[L].A3, HIDDEN, r, DIM);
        }
    }
}

// ===== Adapter-as-input: compute LoRA correction CPU-side, add to ANE output =====
// out += B @ (A @ x), where A[rank,in_dim], B[out_dim,rank], x[in_dim,SEQ]
static void lora_addmm(float *out, const float *A, const float *B,
                        const float *x, float *tmp_r,
                        int out_dim, int rank, int in_dim) {
    // tmp_r[rank, SEQ] = A[rank, in_dim] @ x[in_dim, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, SEQ, in_dim, 1.0f, A, in_dim, x, SEQ, 0.0f, tmp_r, SEQ);
    // out[out_dim, SEQ] += B[out_dim, rank] @ tmp_r[rank, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, SEQ, rank, 1.0f, B, rank, tmp_r, SEQ, 1.0f, out, SEQ);
}

// ===== MeZO checkpoint (BLZT v4, zeros for Adam state) =====
static void mezo_save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                                 double ct, double cw, int cs,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write checkpoint %s\n", path); return; }
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps; h.lr = lr; h.loss = loss;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM; h.hidden_dim = HIDDEN;
    h.n_heads = HEADS; h.seq_len = SEQ;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = 0;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    // Write weights + zeros for Adam state (layout matches train.m exactly)
    size_t max_sz = WQ_SZ > W1_SZ ? WQ_SZ : W1_SZ;
    if ((size_t)VOCAB * DIM > max_sz) max_sz = (size_t)VOCAB * DIM;
    float *zeros_big = (float*)safe_calloc(max_sz, 4);
    for (int L = 0; L < NLAYERS; L++) {
        // All weights first (same order as train.m)
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        // All Adam m/v states as zeros (same order as train.m)
        fwrite(zeros_big,4,WQ_SZ,f); fwrite(zeros_big,4,WQ_SZ,f);
        fwrite(zeros_big,4,WK_SZ,f); fwrite(zeros_big,4,WK_SZ,f);
        fwrite(zeros_big,4,WV_SZ,f); fwrite(zeros_big,4,WV_SZ,f);
        fwrite(zeros_big,4,WO_SZ,f); fwrite(zeros_big,4,WO_SZ,f);
        fwrite(zeros_big,4,W1_SZ,f); fwrite(zeros_big,4,W1_SZ,f);
        fwrite(zeros_big,4,W2_SZ,f); fwrite(zeros_big,4,W2_SZ,f);
        fwrite(zeros_big,4,W3_SZ,f); fwrite(zeros_big,4,W3_SZ,f);
        fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
        fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f); fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
    fwrite(embed,4,(size_t)VOCAB*DIM,f); fwrite(zeros_big,4,(size_t)VOCAB*DIM,f); fwrite(zeros_big,4,(size_t)VOCAB*DIM,f);
    free(zeros_big);
    fclose(f);
}

static bool mezo_load_checkpoint(const char *path, int *step, float *lr, float *loss,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return false; }
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    if (h.n_layers != NLAYERS || h.dim != DIM || h.vocab_size != VOCAB ||
        h.hidden_dim != HIDDEN || h.seq_len != SEQ || h.n_heads != HEADS ||
        h.kv_heads != KV_HEADS || h.head_dim != HD || h.q_dim != Q_DIM) {
        fprintf(stderr, "MeZO checkpoint mismatch\n"); fclose(f); return false;
    }
    *step = h.step; *lr = h.lr; *loss = h.loss;
    if (h.step < 0 || h.step > 10000000) {
        fprintf(stderr, "MeZO checkpoint has invalid step value\n"); fclose(f); return false;
    }
    // Read weights, skip Adam m/v (layout matches train.m exactly)
    size_t max_sz = WQ_SZ > W1_SZ ? WQ_SZ : W1_SZ;
    if ((size_t)VOCAB * DIM > max_sz) max_sz = (size_t)VOCAB * DIM;
    float *skip = (float*)safe_malloc(max_sz * 4);
    for (int L = 0; L < NLAYERS; L++) {
        // All weights first (same order as train.m)
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        // Skip all Adam m/v states
        fread(skip,4,WQ_SZ,f); fread(skip,4,WQ_SZ,f);
        fread(skip,4,WK_SZ,f); fread(skip,4,WK_SZ,f);
        fread(skip,4,WV_SZ,f); fread(skip,4,WV_SZ,f);
        fread(skip,4,WO_SZ,f); fread(skip,4,WO_SZ,f);
        fread(skip,4,W1_SZ,f); fread(skip,4,W1_SZ,f);
        fread(skip,4,W2_SZ,f); fread(skip,4,W2_SZ,f);
        fread(skip,4,W3_SZ,f); fread(skip,4,W3_SZ,f);
        fread(skip,4,DIM,f); fread(skip,4,DIM,f);
        fread(skip,4,DIM,f); fread(skip,4,DIM,f);
    }
    fread(rms_final,4,DIM,f); fread(skip,4,DIM,f); fread(skip,4,DIM,f);
    fread(embed,4,(size_t)VOCAB*DIM,f); fread(skip,4,(size_t)VOCAB*DIM,f); fread(skip,4,(size_t)VOCAB*DIM,f);
    free(skip);
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        // Defaults
        int total_steps = 999999;
        float lr = 1e-5f, base_lr = 1e-5f;
        float epsilon = 1e-3f;
        double time_budget_sec = 0;
        bool from_scratch = false, cpu_only = false, ane_matmul_only = false;
        bool lr_from_cli = false;
        bool use_lora = false;
        bool lora_split = false;  // adapter-as-input: no merge, no restage
        bool lora_ffn = false;    // also apply LoRA to W1, W2, W3
        int lora_rank = 8;
        long init_seed = 42;
        int val_every = 500;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *ckpt_load_path = NULL;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--cpu-only") == 0) cpu_only = true;
            else if (strcmp(argv[i], "--ane-matmul-only") == 0) ane_matmul_only = true;
            else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) { lr = atof(argv[++i]); base_lr = lr; lr_from_cli = true; }
            else if (strcmp(argv[i], "--epsilon") == 0 && i+1 < argc) epsilon = atof(argv[++i]);
            else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--time") == 0 && i+1 < argc) time_budget_sec = atof(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) init_seed = atol(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--val-every") == 0 && i+1 < argc) val_every = atoi(argv[++i]);
            else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) ckpt_load_path = argv[++i];
            else if (strcmp(argv[i], "--lora") == 0) use_lora = true;
            else if (strcmp(argv[i], "--lora-split") == 0) { use_lora = true; lora_split = true; }
            else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) lora_rank = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lora-ffn") == 0) lora_ffn = true;
        }

        if (!cpu_only && !ane_matmul_only) cpu_only = true;  // Default to CPU-only
        if (!cpu_only) {
            if (!ane_init()) {
                fprintf(stderr, "ANE init failed. Use --cpu-only.\n");
                return 1;
            }
        }

        // === Print config ===
        printf("=== MeZO (Zeroth-Order) Training: %s (%d layers, GQA %d/%d) ===\n",
               MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
        printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
               DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
        double total_p = (double)NLAYERS * LAYER_PARAMS + DIM + (double)VOCAB * DIM;
        printf("Params: %.1fM | Mode: %s\n", total_p / 1e6,
               cpu_only ? "CPU-only" : "ANE-matmul-only");
        if (use_lora) printf("MeZO+LoRA: lr=%g epsilon=%g seed=%ld val_every=%d rank=%d\n", lr, epsilon, init_seed, val_every, lora_rank);
        else printf("MeZO: lr=%g epsilon=%g seed=%ld val_every=%d\n", lr, epsilon, init_seed, val_every);
        printf("Memory: ~%.0fMB (inference only, no gradients/optimizer)\n",
               (total_p * 4 + SEQ * DIM * 4 * 10) / 1e6);

        // === Allocate weights ===
        LayerWeights lw[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) lw[L] = layer_weights_alloc();
        float *rms_final = (float*)safe_malloc(DIM * 4);
        float *embed = (float*)safe_malloc((size_t)VOCAB * DIM * 4);

        int start_step = 0;
        float resume_loss = 0;

        if (ckpt_load_path) {
            if (mezo_load_checkpoint(ckpt_load_path, &start_step, &lr, &resume_loss, lw, rms_final, embed)) {
                printf("[RESUMED from step %d, loss=%.4f]\n", start_step, resume_loss);
                if (lr_from_cli) {
                    lr = base_lr;  // CLI --lr overrides checkpoint lr
                    printf("  (using CLI lr=%g instead of checkpoint lr)\n", lr);
                } else {
                    base_lr = lr;
                }
            } else {
                fprintf(stderr, "Failed to load checkpoint %s\n", ckpt_load_path);
                return 1;
            }
        } else if (from_scratch) {
            printf("Initializing from scratch (seed=%ld)\n", init_seed);
            srand48(init_seed);
            float scale_d = 1.0f / sqrtf(DIM), scale_qd = 1.0f / sqrtf(Q_DIM);
            float scale_h = 1.0f / sqrtf(HIDDEN);
            float res_scale = 1.0f / sqrtf(2.0f * NLAYERS);
            for (int L = 0; L < NLAYERS; L++) {
                for (size_t i = 0; i < WQ_SZ; i++) lw[L].Wq[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WK_SZ; i++) lw[L].Wk[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WV_SZ; i++) lw[L].Wv[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WO_SZ; i++) lw[L].Wo[i] = scale_qd * res_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < W1_SZ; i++) lw[L].W1[i] = scale_h * (2 * drand48() - 1);
                for (size_t i = 0; i < W2_SZ; i++) lw[L].W2[i] = scale_d * res_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < W3_SZ; i++) lw[L].W3[i] = scale_h * (2 * drand48() - 1);
                for (int i = 0; i < DIM; i++) { lw[L].rms_att[i] = 1.0f; lw[L].rms_ffn[i] = 1.0f; }
            }
            for (int i = 0; i < DIM; i++) rms_final[i] = 1.0f;
            float escale = 0.02f;
            for (size_t i = 0; i < (size_t)VOCAB * DIM; i++) embed[i] = escale * (2 * drand48() - 1);
        } else {
            fprintf(stderr, "Must specify --scratch or --resume <path>\n");
            return 1;
        }

        // === LoRA initialization ===
        LoRALayer lora_layers[NLAYERS];
        (void)0;  // FFN LoRA uses same lora_tmp buffer (rank*SEQ)
        if (use_lora) {
            int r = lora_rank;
            float a_scale = 1.0f / sqrtf((float)r);
            srand48(init_seed + 12345);  // Separate seed for LoRA init
            size_t lora_params = 0;
            for (int L = 0; L < NLAYERS; L++) {
                lora_layers[L] = lora_layer_alloc(r, lora_ffn);
                // Copy base weights (frozen)
                memcpy(lora_layers[L].Wq_base, lw[L].Wq, WQ_SZ * 4);
                memcpy(lora_layers[L].Wk_base, lw[L].Wk, WK_SZ * 4);
                memcpy(lora_layers[L].Wv_base, lw[L].Wv, WV_SZ * 4);
                memcpy(lora_layers[L].Wo_base, lw[L].Wo, WO_SZ * 4);
                // Init A with small random, B with zero (LoRA starts as identity)
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Aq[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Ak[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Av[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * Q_DIM; i++) lora_layers[L].Ao[i] = a_scale * (2 * drand48() - 1);
                // B matrices stay zero (calloc)
                // Aq[r,DIM]+Bq[Q_DIM,r] + Ak[r,DIM]+Bk[KV_DIM,r] + Av[r,DIM]+Bv[KV_DIM,r] + Ao[r,Q_DIM]+Bo[DIM,r]
                lora_params += (size_t)r * DIM * 3 + (size_t)Q_DIM * r + (size_t)KV_DIM * r * 2 + (size_t)r * Q_DIM + (size_t)DIM * r;
                if (lora_ffn) {
                    memcpy(lora_layers[L].W1_base, lw[L].W1, W1_SZ * 4);
                    memcpy(lora_layers[L].W2_base, lw[L].W2, W2_SZ * 4);
                    memcpy(lora_layers[L].W3_base, lw[L].W3, W3_SZ * 4);
                    // A1[rank,DIM], B1[HIDDEN,rank], A3[rank,DIM], B3[HIDDEN,rank]
                    for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].A1[i] = a_scale * (2 * drand48() - 1);
                    for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].A3[i] = a_scale * (2 * drand48() - 1);
                    // A2[rank,HIDDEN]
                    float a2_scale = 1.0f / sqrtf((float)r);
                    for (size_t i = 0; i < (size_t)r * HIDDEN; i++) lora_layers[L].A2[i] = a2_scale * (2 * drand48() - 1);
                    // B matrices stay zero (calloc)
                    // A1[r,DIM]+B1[HIDDEN,r] + A2[r,HIDDEN]+B2[DIM,r] + A3[r,DIM]+B3[HIDDEN,r]
                    lora_params += (size_t)r * DIM * 2 + (size_t)HIDDEN * r * 2 + (size_t)r * HIDDEN + (size_t)DIM * r;
                }
            }
            size_t rms_params = (size_t)NLAYERS * 2 * DIM + DIM;
            printf("LoRA: rank=%d, adapter params=%.1fK, trainable RMS params=%.1fK\n",
                   r, (float)lora_params / 1e3, (float)rms_params / 1e3);
            if (lora_ffn) printf("  Adapters on: Wq, Wk, Wv, Wo, W1, W2, W3 | Frozen: embed\n");
            else printf("  Adapters on: Wq, Wk, Wv, Wo | Frozen: W1, W2, W3, embed\n");
            printf("  Perturbation: LoRA A/B + RMS only (~%.1fK params vs %.1fM full)\n",
                   (float)(lora_params + rms_params) / 1e3, total_p / 1e6);
            if (lora_split) printf("  Mode: adapter-as-input (zero restaging, CPU-side LoRA correction)\n");
            // FFN LoRA split uses same lora_tmp buffer (allocated later)
        }

        // === mmap token data ===
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { fprintf(stderr, "Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        if (data_len == 0 || data_len % 2 != 0) { fprintf(stderr, "FATAL: invalid data file\n"); close(data_fd); return 1; }
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); close(data_fd); return 1; }
        size_t n_tokens = data_len / 2;
        for (size_t i = 0; i < n_tokens; i++) {
            if (token_data[i] >= VOCAB) {
                fprintf(stderr, "FATAL: token[%zu]=%d >= VOCAB=%d\n", i, token_data[i], VOCAB);
                return 1;
            }
        }
        size_t val_start = (size_t)(n_tokens * 0.9);
        size_t train_tokens = val_start;
        size_t val_tokens = n_tokens - val_start;
        printf("Tokens: %zu (train: %zu, val: %zu)\n", n_tokens, train_tokens, val_tokens);
        if (train_tokens < (size_t)SEQ + 2) {
            fprintf(stderr, "FATAL: training split too small (%zu tokens < SEQ+2=%d)\n",
                    train_tokens, SEQ + 2);
            return 1;
        }

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d -> %d active\n", VOCAB, CV);
        float *cembed = vocab_compact_embed(embed, &vm, DIM);

        // Residual scaling: DeepNet for from-scratch, standard (1.0) for pretrained
        // SmolLM2/Llama models use alpha=1.0; DeepNet scaling only valid when
        // weights are initialized with matching 1/sqrt(2L) scale on Wo/W2
        float res_alpha = from_scratch ? 1.0f / sqrtf(2.0f * NLAYERS) : 1.0f;

        // === Forward buffers (reused across layers, no per-layer caching) ===
        float *x_cur = (float*)safe_malloc(SEQ * DIM * 4);
        float *xnorm_buf = (float*)safe_malloc(SEQ * DIM * 4);
        float *Q = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *K = (float*)safe_malloc(SEQ * KV_DIM * 4);
        float *V = (float*)safe_malloc(SEQ * KV_DIM * 4);
        float *attn_out = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *o_out = (float*)safe_malloc(SEQ * DIM * 4);
        float *h1 = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *h3 = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *silu_out = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *k_tiled = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *v_tiled = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *logits = (float*)safe_malloc(SEQ * CV * 4);
        float *dlogits = (float*)safe_malloc(SEQ * CV * 4);  // throwaway for cross_entropy_loss
        // Temp buffer for lora_split mode: A @ x intermediate result [rank, SEQ]
        float *lora_tmp = NULL;
        if (lora_split) {
            lora_tmp = (float*)safe_malloc((size_t)lora_rank * SEQ * 4);
        }

        // === Compile ANE kernels (forward only) ===
        DynLayerKernels dk;
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        memset(&dk, 0, sizeof(dk));
        memset(pls, 0, sizeof(pls));
        memset(plr, 0, sizeof(plr));

        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];

        if (!cpu_only) {
            printf("\nCompiling ANE forward kernels...\n");
            uint64_t t_compile = mach_absolute_time();
            if (!compile_dynamic_kernels(&dk, 1.0f / sqrtf(2.0f * NLAYERS), true, false)) {
                fprintf(stderr, "ANE kernel compilation failed\n"); return 1;
            }
            printf("Compiled %d kernels in %.0fms\n", g_compile_count,
                   tb_ms(mach_absolute_time() - t_compile));

            // Allocate transposed weight buffers + IOSurfaces
            for (int L = 0; L < NLAYERS; L++) {
                Wqt_buf[L] = (float*)safe_malloc(WQ_SZ * 4);
                Wkt_buf[L] = (float*)safe_malloc(WK_SZ * 4);
                Wvt_buf[L] = (float*)safe_malloc(WV_SZ * 4);
                Wot_buf[L] = (float*)safe_malloc(WO_SZ * 4);
                W1t_buf[L] = (float*)safe_malloc(W1_SZ * 4);
                W2t_buf[L] = (float*)safe_malloc(W2_SZ * 4);
                W3t_buf[L] = (float*)safe_malloc(W3_SZ * 4);

                // Per-layer IOSurfaces for unfused forward kernels
                pls[L].wqFwd_in = make_surface(DIM * WQ_FWD_SP * 2);
                pls[L].wkFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                pls[L].wvFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                pls[L].woFwd_in = make_surface(Q_DIM * WO_FWD_SP * 2);
                pls[L].w1Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
                pls[L].w3Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
                pls[L].w2Fwd_in = make_surface(HIDDEN * W2_FWD_SP * 2);

                // Per-layer requests
                plr[L].wqFwd = make_request(dk.wqFwd, pls[L].wqFwd_in);
                plr[L].wkFwd = make_request(dk.wkvFwd, pls[L].wkFwd_in);
                plr[L].wvFwd = make_request(dk.wkvFwd, pls[L].wvFwd_in);
                plr[L].woFwd = make_request(dk.woFwd, pls[L].woFwd_in);
                plr[L].w1Fwd = make_request(dk.w13Fwd, pls[L].w1Fwd_in);
                plr[L].w3Fwd = make_request(dk.w13Fwd, pls[L].w3Fwd_in);
                plr[L].w2Fwd = make_request(dk.w2Fwd, pls[L].w2Fwd_in);
            }
        } else {
            for (int L = 0; L < NLAYERS; L++) {
                Wqt_buf[L] = NULL; Wkt_buf[L] = NULL; Wvt_buf[L] = NULL; Wot_buf[L] = NULL;
                W1t_buf[L] = NULL; W2t_buf[L] = NULL; W3t_buf[L] = NULL;
            }
            printf("CPU-only mode: skipping ANE kernel compilation\n");
        }

        // ===== Transpose + stage weights into IOSurfaces (ANE only) =====
        // Extracted as a function for reuse after perturbation
        // (inline for now — called 2x per step in ANE mode)
        #define RETRANSPOSE_AND_STAGE() do { \
            for (int L = 0; L < NLAYERS; L++) { \
                transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM); \
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM); \
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM); \
                transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM); \
                transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM); \
                transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN); \
                transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM); \
                { IOSurfaceLock(pls[L].wqFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wqFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt_buf[L] + d*Q_DIM, Q_DIM); \
                  IOSurfaceUnlock(pls[L].wqFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wkFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wkFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wkFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wvFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wvFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wvFwd_in, 0, NULL); } \
                stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]); \
                { IOSurfaceLock(pls[L].w1Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w1Fwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W1t_buf[L] + d*HIDDEN, HIDDEN); \
                  IOSurfaceUnlock(pls[L].w1Fwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].w3Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w3Fwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W3t_buf[L] + d*HIDDEN, HIDDEN); \
                  IOSurfaceUnlock(pls[L].w3Fwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].w2Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w2Fwd_in); \
                  for (int h = 0; h < HIDDEN; h++) \
                      cvt_f32_f16(buf + h*W2_FWD_SP + SEQ, W2t_buf[L] + h*DIM, DIM); \
                  IOSurfaceUnlock(pls[L].w2Fwd_in, 0, NULL); } \
            } \
        } while(0)

        // Attention-only restage: for MeZO+LoRA, FFN weights (W1/W2/W3) never change
        #define RETRANSPOSE_ATTN_ONLY() do { \
            for (int L = 0; L < NLAYERS; L++) { \
                transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM); \
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM); \
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM); \
                transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM); \
                { IOSurfaceLock(pls[L].wqFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wqFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt_buf[L] + d*Q_DIM, Q_DIM); \
                  IOSurfaceUnlock(pls[L].wqFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wkFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wkFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wkFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wvFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wvFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wvFwd_in, 0, NULL); } \
                stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]); \
            } \
        } while(0)

        // Initial transpose + staging
        if (!cpu_only) {
            RETRANSPOSE_AND_STAGE();
            printf("Initial weight staging complete\n");
        }

        // ===== MeZO Training Loop =====
        float last_loss_plus = 999.0f, last_loss_minus = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();

        printf("\nStarting MeZO training...\n\n");

        for (int step = start_step; step < total_steps; step++) {
            // Time budget check
            if (time_budget_sec > 0 && step > start_step + 2) {
                double elapsed_sec = tb_ms(mach_absolute_time() - t_wall_start) / 1000.0;
                if (elapsed_sec >= time_budget_sec) {
                    printf("Time budget %.0fs reached at step %d (%.1fs elapsed)\n",
                           time_budget_sec, step, elapsed_sec);
                    total_steps = step;
                    break;
                }
            }

            uint64_t t_step = mach_absolute_time();
            double t_perturb = 0, t_fwd = 0, t_transpose = 0;

            // Sample data (from training split)
            size_t max_pos = train_tokens - SEQ - 1;
            srand48(init_seed + step * 7919LL);  // Deterministic data sampling per step
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_raw = token_data + pos + 1;
            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_raw[t]];

            // MeZO seed for this step
            uint64_t mezo_seed = (uint64_t)step * 1000003ULL + (uint64_t)init_seed;

            // ===== 1. Perturb +epsilon =====
            uint64_t t0 = mach_absolute_time();
            if (use_lora) {
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon);
                if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
            } else {
                perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
            }
            if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
            t_perturb += tb_ms(mach_absolute_time() - t0);

            if (!cpu_only && !lora_split) {
                t0 = mach_absolute_time();
                if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                else { RETRANSPOSE_AND_STAGE(); }
                t_transpose += tb_ms(mach_absolute_time() - t0);
            }

            // ===== 2. Forward pass -> loss_plus =====
            t0 = mach_absolute_time();
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ, VOCAB);

            for (int L = 0; L < NLAYERS; L++) {
                // RMSNorm pre-attention
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);

                if (cpu_only) {
                    // CPU matmuls (use effective weights: base or merged)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, Q, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, K, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, V, SEQ);
                    if (lora_split) {
                        // Add LoRA corrections: Q += Bq @ (Aq @ xnorm), etc.
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                    }
                } else {
                    // ANE matmuls (base weights baked in IOSurfaces)
                    io_write_dyn_acts(pls[L].wqFwd_in, xnorm_buf, DIM, SEQ, WQ_FWD_SP);
                    ane_eval_req(dk.wqFwd, plr[L].wqFwd);
                    io_read_dyn(dk.wqFwd->ioOut, Q, Q_DIM, SEQ);

                    io_write_dyn_acts(pls[L].wkFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                    ane_eval_req(dk.wkvFwd, plr[L].wkFwd);
                    io_read_dyn(dk.wkvFwd->ioOut, K, KV_DIM, SEQ);

                    io_write_dyn_acts(pls[L].wvFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                    ane_eval_req(dk.wkvFwd, plr[L].wvFwd);
                    io_read_dyn(dk.wkvFwd->ioOut, V, KV_DIM, SEQ);

                    if (lora_split) {
                        // Add LoRA corrections on CPU after ANE base matmul
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                    }
                }

                // RoPE + GQA tile + SDPA (always CPU fp32)
                rope_forward_inplace(Q, SEQ, Q_DIM, HD);
                rope_forward_inplace(K, SEQ, KV_DIM, HD);
                gqa_tile_kv(k_tiled, K, SEQ);
                gqa_tile_kv(v_tiled, V, SEQ);
                cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ);

                // Wo projection
                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, attn_out, SEQ, 0.0f, o_out, SEQ);
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                    }
                } else {
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                    ane_eval_req(dk.woFwd, plr[L].woFwd);
                    io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ);
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                    }
                }

                // Residual 1: x = x + res_alpha * o_out (DeepNet scaled residual)
                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));

                // RMSNorm pre-FFN
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);

                // FFN: h1 = xnorm @ W1, h3 = xnorm @ W3, silu_out = SiLU(h1) * h3, ffn_out = silu_out @ W2
                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, h1, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, h3, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                    }
                } else {
                    io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);

                    io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);

                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                    }
                }

                // SiLU(h1) * h3
                for (int i = 0; i < HIDDEN * SEQ; i++) {
                    float s = h1[i] / (1.0f + expf(-h1[i]));  // SiLU
                    silu_out[i] = s * h3[i];
                }

                // W2 projection
                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, silu_out, SEQ, 0.0f, o_out, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                    }
                } else {
                    io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                    ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                    io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                    }
                }

                // Residual 2: x = x + res_alpha * ffn_out (DeepNet scaled residual)
                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
            }

            // Final RMSNorm + classifier
            rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ);
            // logits[SEQ,CV] = x_final^T[SEQ,DIM] @ cembed^T[DIM,CV]  (row-major for cross_entropy_loss)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, xnorm_buf, SEQ, cembed, DIM, 0.0f, logits, CV);
            float loss_plus = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_fwd += tb_ms(mach_absolute_time() - t0);

            // ===== 3. Perturb -2*epsilon (to theta - epsilon*z) =====
            t0 = mach_absolute_time();
            if (use_lora) {
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, -2.0f * epsilon);
                if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
            } else {
                perturb_all_weights(lw, embed, rms_final, mezo_seed, -2.0f * epsilon);
            }
            if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
            t_perturb += tb_ms(mach_absolute_time() - t0);

            if (!cpu_only && !lora_split) {
                t0 = mach_absolute_time();
                if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                else { RETRANSPOSE_AND_STAGE(); }
                t_transpose += tb_ms(mach_absolute_time() - t0);
            }

            // ===== 4. Forward pass -> loss_minus =====
            t0 = mach_absolute_time();
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ, VOCAB);

            for (int L = 0; L < NLAYERS; L++) {
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, Q, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, K, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, V, SEQ);
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                    }
                } else {
                    io_write_dyn_acts(pls[L].wqFwd_in, xnorm_buf, DIM, SEQ, WQ_FWD_SP);
                    ane_eval_req(dk.wqFwd, plr[L].wqFwd);
                    io_read_dyn(dk.wqFwd->ioOut, Q, Q_DIM, SEQ);

                    io_write_dyn_acts(pls[L].wkFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                    ane_eval_req(dk.wkvFwd, plr[L].wkFwd);
                    io_read_dyn(dk.wkvFwd->ioOut, K, KV_DIM, SEQ);

                    io_write_dyn_acts(pls[L].wvFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                    ane_eval_req(dk.wkvFwd, plr[L].wvFwd);
                    io_read_dyn(dk.wkvFwd->ioOut, V, KV_DIM, SEQ);

                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                    }
                }

                rope_forward_inplace(Q, SEQ, Q_DIM, HD);
                rope_forward_inplace(K, SEQ, KV_DIM, HD);
                gqa_tile_kv(k_tiled, K, SEQ);
                gqa_tile_kv(v_tiled, V, SEQ);
                cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ);

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, attn_out, SEQ, 0.0f, o_out, SEQ);
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                    }
                } else {
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                    ane_eval_req(dk.woFwd, plr[L].woFwd);
                    io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ);
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                    }
                }

                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));

                rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, h1, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, h3, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                    }
                } else {
                    io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);

                    io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);

                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                    }
                }

                for (int i = 0; i < HIDDEN * SEQ; i++) {
                    float s = h1[i] / (1.0f + expf(-h1[i]));
                    silu_out[i] = s * h3[i];
                }

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, silu_out, SEQ, 0.0f, o_out, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                    }
                } else {
                    io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                    ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                    io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
                    if (lora_split && lora_ffn) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                    }
                }

                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
            }

            rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, xnorm_buf, SEQ, cembed, DIM, 0.0f, logits, CV);
            float loss_minus = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_fwd += tb_ms(mach_absolute_time() - t0);

            // ===== 5. Restore to original theta =====
            t0 = mach_absolute_time();
            if (use_lora) {
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon);
            } else {
                perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
            }
            t_perturb += tb_ms(mach_absolute_time() - t0);

            // ===== 6. Gradient estimate + update =====
            float proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            float update_scale = -lr * proj_grad;

            t0 = mach_absolute_time();
            if (use_lora) {
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, update_scale);
                if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
            } else {
                perturb_all_weights(lw, embed, rms_final, mezo_seed, update_scale);
            }
            t_perturb += tb_ms(mach_absolute_time() - t0);

            // Re-build compact embedding after weight update
            if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }

            // 7. Defer re-transpose: next step's +eps perturbation will restage anyway.
            //    Only restage here if validation is about to run (needs updated weights).
            if (!cpu_only && val_every > 0 && (step + 1) % val_every == 0 && val_tokens > SEQ + 1) {
                t0 = mach_absolute_time();
                if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                else { RETRANSPOSE_AND_STAGE(); }
                t_transpose += tb_ms(mach_absolute_time() - t0);
            }

            // 8. LR schedule (cosine decay, no warmup)
            float min_lr = base_lr * 0.1f;
            float decay = (float)(step - start_step) / (float)(total_steps - start_step);
            lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay)) * (base_lr - min_lr);

            double step_ms = tb_ms(mach_absolute_time() - t_step);
            total_train_ms += step_ms;
            total_steps_done++;
            last_loss_plus = loss_plus;
            last_loss_minus = loss_minus;

            // 9. Log
            if (step % 100 == 0 || step == start_step) {
                printf("step %d  loss_plus=%.4f  loss_minus=%.4f  proj_grad=%.6f  lr=%.2e  "
                       "step_ms=%.0f (fwd=%.0f perturb=%.0f transpose=%.0f)\n",
                       step, loss_plus, loss_minus, proj_grad, lr,
                       step_ms, t_fwd, t_perturb, t_transpose);
            }

            // 10. Validation
            if (val_every > 0 && (step + 1) % val_every == 0 && val_tokens > SEQ + 1) {
                float val_loss_sum = 0;
                int val_batches = 10;
                srand48(999);  // Fixed val seed
                for (int vb = 0; vb < val_batches; vb++) {
                    size_t vpos = val_start + (size_t)(drand48() * (val_tokens - SEQ - 1));
                    uint16_t *vinput = token_data + vpos;
                    uint16_t *vtarget_raw = token_data + vpos + 1;
                    uint16_t vctargets[SEQ];
                    for (int t = 0; t < SEQ; t++) vctargets[t] = (uint16_t)vm.full_to_compact[vtarget_raw[t]];

                    embed_lookup(x_cur, embed, vinput, DIM, SEQ, VOCAB);
                    for (int L = 0; L < NLAYERS; L++) {
                        rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, Q_DIM,SEQ,DIM, 1.0f, lw[L].Wq,DIM, xnorm_buf,SEQ, 0.0f, Q,SEQ);
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, KV_DIM,SEQ,DIM, 1.0f, lw[L].Wk,DIM, xnorm_buf,SEQ, 0.0f, K,SEQ);
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, KV_DIM,SEQ,DIM, 1.0f, lw[L].Wv,DIM, xnorm_buf,SEQ, 0.0f, V,SEQ);
                        } else {
                            io_write_dyn_acts(pls[L].wqFwd_in, xnorm_buf, DIM, SEQ, WQ_FWD_SP);
                            ane_eval_req(dk.wqFwd, plr[L].wqFwd);
                            io_read_dyn(dk.wqFwd->ioOut, Q, Q_DIM, SEQ);
                            io_write_dyn_acts(pls[L].wkFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                            ane_eval_req(dk.wkvFwd, plr[L].wkFwd);
                            io_read_dyn(dk.wkvFwd->ioOut, K, KV_DIM, SEQ);
                            io_write_dyn_acts(pls[L].wvFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP);
                            ane_eval_req(dk.wkvFwd, plr[L].wvFwd);
                            io_read_dyn(dk.wkvFwd->ioOut, V, KV_DIM, SEQ);
                        }
                        if (lora_split) {
                            LoRALayer *ll = &lora_layers[L];
                            lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                            lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                            lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        }
                        rope_forward_inplace(Q, SEQ, Q_DIM, HD);
                        rope_forward_inplace(K, SEQ, KV_DIM, HD);
                        gqa_tile_kv(k_tiled, K, SEQ);
                        gqa_tile_kv(v_tiled, V, SEQ);
                        cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ);
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, DIM,SEQ,Q_DIM, 1.0f, lw[L].Wo,Q_DIM, attn_out,SEQ, 0.0f, o_out,SEQ);
                        } else {
                            write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                            ane_eval_req(dk.woFwd, plr[L].woFwd);
                            io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ);
                        }
                        if (lora_split) {
                            LoRALayer *ll = &lora_layers[L];
                            lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                        }
                        vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                        rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, HIDDEN,SEQ,DIM, 1.0f, lw[L].W1,DIM, xnorm_buf,SEQ, 0.0f, h1,SEQ);
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, HIDDEN,SEQ,DIM, 1.0f, lw[L].W3,DIM, xnorm_buf,SEQ, 0.0f, h3,SEQ);
                            if (lora_split && lora_ffn) {
                                LoRALayer *ll = &lora_layers[L];
                                lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                                lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                            }
                        } else {
                            io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                            ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                            io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);
                            io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                            ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                            io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);
                            if (lora_split && lora_ffn) {
                                LoRALayer *ll = &lora_layers[L];
                                lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                                lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM);
                            }
                        }
                        for (int i = 0; i < HIDDEN*SEQ; i++) { float s = h1[i]/(1.0f+expf(-h1[i])); silu_out[i] = s*h3[i]; }
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, DIM,SEQ,HIDDEN, 1.0f, lw[L].W2,HIDDEN, silu_out,SEQ, 0.0f, o_out,SEQ);
                            if (lora_split && lora_ffn) {
                                LoRALayer *ll = &lora_layers[L];
                                lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                            }
                        } else {
                            io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                            ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                            io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
                            if (lora_split && lora_ffn) {
                                LoRALayer *ll = &lora_layers[L];
                                lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN);
                            }
                        }
                        vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                    }
                    rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ);
                    cblas_sgemm(CblasRowMajor,CblasTrans,CblasTrans, SEQ,CV,DIM, 1.0f, xnorm_buf,SEQ, cembed,DIM, 0.0f, logits,CV);
                    val_loss_sum += cross_entropy_loss(dlogits, logits, vctargets, CV, SEQ);
                }
                float val_loss = val_loss_sum / val_batches;
                printf("  [val_loss=%.4f at step %d]\n", val_loss, step + 1);

                // Checkpoint on best val
                if (val_loss < best_loss) {
                    best_loss = val_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    mezo_save_checkpoint(CKPT_PATH, step + 1, total_steps, lr, val_loss,
                                        total_train_ms, wall, total_steps_done,
                                        lw, rms_final, embed);
                    printf("  [ckpt saved, best_val=%.4f]\n", best_loss);
                }
            }
        }

        // ===== Final report =====
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== MeZO Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms / fmax(1, total_steps_done));
        printf("Wall time:    %.1fs\n", wall / 1000.0);
        printf("\n---\n");
        printf("final_loss_plus:  %.6f\n", last_loss_plus);
        printf("final_loss_minus: %.6f\n", last_loss_minus);
        printf("training_seconds: %.1f\n", total_train_ms / 1000.0);
        printf("total_seconds:    %.1f\n", wall / 1000.0);
        printf("num_steps:        %d\n", total_steps_done);
        printf("num_params_M:     %.1f\n", ((double)NLAYERS * LAYER_PARAMS + DIM + (double)VOCAB * DIM) / 1e6);
        printf("mode:             mezo-%s%s\n", cpu_only ? "cpu" : "ane",
               lora_split ? "-lora-split" : (use_lora ? "-lora" : ""));
        printf("epsilon:          %g\n", epsilon);
        printf("lr:               %g\n", lr);
        if (use_lora) printf("lora_rank:        %d\n", lora_rank);

        // Cleanup
        if (lora_tmp) free(lora_tmp);
        if (use_lora) {
            for (int L = 0; L < NLAYERS; L++) lora_layer_free(&lora_layers[L]);
        }
        for (int L = 0; L < NLAYERS; L++) {
            layer_weights_free(&lw[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free(rms_final); free(embed); free(cembed);
        free(x_cur); free(xnorm_buf);
        free(Q); free(K); free(V); free(attn_out); free(o_out);
        free(h1); free(h3); free(silu_out);
        free(k_tiled); free(v_tiled); free(logits); free(dlogits);
        munmap(token_data, data_len); close(data_fd);

        if (!cpu_only) {
            free_per_layer(pls, plr);
            free_kern(dk.wqFwd); free_kern(dk.wkvFwd); free_kern(dk.w13Fwd);
            free_kern(dk.w2Fwd); free_kern(dk.woFwd);
        }
    }
    return 0;
}
