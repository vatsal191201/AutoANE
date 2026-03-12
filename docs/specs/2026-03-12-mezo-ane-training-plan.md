# MeZO-on-ANE Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement zeroth-order (MeZO) training on Apple Neural Engine — first ZO training on any NPU hardware.

**Architecture:** Separate `train_mezo.m` binary (~600 lines) sharing headers with existing `train.m`. Forward-pass only — no backward kernels, no gradient buffers, no Adam state. SPSA gradient estimation via 2 forward passes per step with in-place seed trick for O(1) memory perturbation.

**Tech Stack:** C/Objective-C, Accelerate.framework, IOSurface.framework, AppleNeuralEngine.framework (private API), MIL (ML Intermediate Language)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `training/train_mezo.m` | Create | MeZO training loop: perturbation, 2x forward pass, ZO gradient estimate, SGD update |
| `training/Makefile` | Modify | Add `mezo` build target |
| `tests/test_training.sh` | Modify | Add Tests 9-12 for MeZO |

**Shared headers (read-only, no modifications needed):**
- `training/config.h` — Model structs (`LayerWeights`, `CkptHdr`), derived sizes (`WQ_SZ`, etc.), `ane_init()`, alloc helpers
- `training/cpu_ops.h` — `rmsnorm()` (forward only), `cross_entropy_loss()`, `embed_lookup()`, `VocabMap`
- `training/io.h` — IOSurface helpers, NEON fp16 conversion, kernel compile/eval, weight staging
- `training/mil_dynamic.h` — Forward MIL kernel generators (only forward kernels used)
- `training/models/*.h` — Model dimension constants

---

## Chunk 1: Core MeZO Binary (CPU-only path)

### Task 1: Add Makefile target

**Files:**
- Modify: `training/Makefile`

- [ ] **Step 1: Add `mezo` target to Makefile**

Add after the `train` target in `training/Makefile`:

```makefile
mezo: train_mezo.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@command -v xcrun >/dev/null 2>&1 || { echo "ERROR: Xcode Command Line Tools not found. Install with: xcode-select --install"; exit 1; }
	@echo "Building MeZO for model: $(MODEL)"
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -o train_mezo train_mezo.m
```

Update `clean` target:

```makefile
clean:
	rm -f train train_mezo benchmark
```

- [ ] **Step 2: Commit**

```bash
git add training/Makefile
git commit -m "build: add mezo target to Makefile"
```

---

### Task 2: Create train_mezo.m skeleton with arg parsing

**Files:**
- Create: `training/train_mezo.m`

- [ ] **Step 1: Create skeleton with includes, main, arg parsing**

Create `training/train_mezo.m` with:

```objc
// train_mezo.m — Zeroth-Order (MeZO/SPSA) training on Apple Neural Engine
// Forward-pass only: no backward kernels, no gradients, no Adam state.
// Memory = inference memory (seed trick eliminates perturbation storage).
//
// Build: make mezo MODEL=autoresearch (or smollm2_135m, smollm2_360m)
// Usage: ./train_mezo --scratch --data data.bin --cpu-only --steps 1000
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include <math.h>

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
    // Write weights + zeros for Adam state (compatibility with train.m)
    float *zeros_big = (float*)safe_calloc((size_t)fmax(fmax(WQ_SZ, W1_SZ), VOCAB*DIM), 4);
    for (int L = 0; L < NLAYERS; L++) {
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(zeros_big,4,WQ_SZ,f); fwrite(zeros_big,4,WQ_SZ,f);
        fwrite(lw[L].Wk,4,WK_SZ,f); fwrite(zeros_big,4,WK_SZ,f); fwrite(zeros_big,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(zeros_big,4,WV_SZ,f); fwrite(zeros_big,4,WV_SZ,f);
        fwrite(lw[L].Wo,4,WO_SZ,f); fwrite(zeros_big,4,WO_SZ,f); fwrite(zeros_big,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(zeros_big,4,W1_SZ,f); fwrite(zeros_big,4,W1_SZ,f);
        fwrite(lw[L].W2,4,W2_SZ,f); fwrite(zeros_big,4,W2_SZ,f); fwrite(zeros_big,4,W2_SZ,f);
        fwrite(lw[L].W3,4,W3_SZ,f); fwrite(zeros_big,4,W3_SZ,f); fwrite(zeros_big,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
        fwrite(lw[L].rms_ffn,4,DIM,f); fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
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
    // Read weights, skip Adam m/v (2 buffers per weight matrix)
    float *skip = (float*)safe_malloc((size_t)fmax(fmax(WQ_SZ, W1_SZ), VOCAB*DIM) * 4);
    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq,4,WQ_SZ,f); fread(skip,4,WQ_SZ,f); fread(skip,4,WQ_SZ,f);
        fread(lw[L].Wk,4,WK_SZ,f); fread(skip,4,WK_SZ,f); fread(skip,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(skip,4,WV_SZ,f); fread(skip,4,WV_SZ,f);
        fread(lw[L].Wo,4,WO_SZ,f); fread(skip,4,WO_SZ,f); fread(skip,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(skip,4,W1_SZ,f); fread(skip,4,W1_SZ,f);
        fread(lw[L].W2,4,W2_SZ,f); fread(skip,4,W2_SZ,f); fread(skip,4,W2_SZ,f);
        fread(lw[L].W3,4,W3_SZ,f); fread(skip,4,W3_SZ,f); fread(skip,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(skip,4,DIM,f); fread(skip,4,DIM,f);
        fread(lw[L].rms_ffn,4,DIM,f); fread(skip,4,DIM,f); fread(skip,4,DIM,f);
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
        long init_seed = 42;
        int val_every = 500;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *ckpt_load_path = NULL;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--cpu-only") == 0) cpu_only = true;
            else if (strcmp(argv[i], "--ane-matmul-only") == 0) ane_matmul_only = true;
            else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) { lr = atof(argv[++i]); base_lr = lr; }
            else if (strcmp(argv[i], "--epsilon") == 0 && i+1 < argc) epsilon = atof(argv[++i]);
            else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--time") == 0 && i+1 < argc) time_budget_sec = atof(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) init_seed = atol(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--val-every") == 0 && i+1 < argc) val_every = atoi(argv[++i]);
            else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) ckpt_load_path = argv[++i];
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
        printf("MeZO: lr=%g epsilon=%g seed=%ld val_every=%d\n", lr, epsilon, init_seed, val_every);
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
                base_lr = lr;
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

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d -> %d active\n", VOCAB, CV);
        float *cembed = vocab_compact_embed(embed, &vm, DIM);

        // Residual scaling (DeepNet: scaled residual connections)
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);

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
                      for (int d = 0; d < DIM; d++) \
                          buf[h*W2_FWD_SP + SEQ + d] = (_Float16)lw[L].W2[d*HIDDEN + h]; \
                  IOSurfaceUnlock(pls[L].w2Fwd_in, 0, NULL); } \
            } \
        } while(0)

        // Initial transpose + staging
        if (!cpu_only) {
            RETRANSPOSE_AND_STAGE();
            printf("Initial weight staging complete\n");
        }

        // ===== Forward pass (reusable for both loss_plus and loss_minus) =====
        // Returns cross-entropy loss over the sequence
        // This is a local lambda-like block — defined as a nested function via block/macro
        // For clarity, inline the forward pass in the training loop.

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
            perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
            free(cembed);
            cembed = vocab_compact_embed(embed, &vm, DIM);
            t_perturb += tb_ms(mach_absolute_time() - t0);

            if (!cpu_only) {
                t0 = mach_absolute_time();
                RETRANSPOSE_AND_STAGE();
                t_transpose += tb_ms(mach_absolute_time() - t0);
            }

            // ===== 2. Forward pass -> loss_plus =====
            t0 = mach_absolute_time();
            embed_lookup(x_cur, embed, input_tokens, DIM, SEQ, VOCAB);

            for (int L = 0; L < NLAYERS; L++) {
                // RMSNorm pre-attention
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);

                if (cpu_only) {
                    // CPU matmuls
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, Q, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, K, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, V, SEQ);
                } else {
                    // ANE matmuls
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
                } else {
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                    ane_eval_req(dk.woFwd, plr[L].woFwd);
                    io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ);
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
                } else {
                    io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);

                    io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);
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
                } else {
                    io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                    ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                    io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
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
            perturb_all_weights(lw, embed, rms_final, mezo_seed, -2.0f * epsilon);
            free(cembed);
            cembed = vocab_compact_embed(embed, &vm, DIM);
            t_perturb += tb_ms(mach_absolute_time() - t0);

            if (!cpu_only) {
                t0 = mach_absolute_time();
                RETRANSPOSE_AND_STAGE();
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

                rope_forward_inplace(Q, SEQ, Q_DIM, HD);
                rope_forward_inplace(K, SEQ, KV_DIM, HD);
                gqa_tile_kv(k_tiled, K, SEQ);
                gqa_tile_kv(v_tiled, V, SEQ);
                cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ);

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, attn_out, SEQ, 0.0f, o_out, SEQ);
                } else {
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                    ane_eval_req(dk.woFwd, plr[L].woFwd);
                    io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ);
                }

                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));

                rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, h1, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, h3, SEQ);
                } else {
                    io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);

                    io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                    ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                    io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);
                }

                for (int i = 0; i < HIDDEN * SEQ; i++) {
                    float s = h1[i] / (1.0f + expf(-h1[i]));
                    silu_out[i] = s * h3[i];
                }

                if (cpu_only) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, silu_out, SEQ, 0.0f, o_out, SEQ);
                } else {
                    io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                    ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                    io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
                }

                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
            }

            rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        SEQ, CV, DIM, 1.0f, xnorm_buf, SEQ, cembed, DIM, 0.0f, logits, CV);
            float loss_minus = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
            t_fwd += tb_ms(mach_absolute_time() - t0);

            // ===== 5. Restore to original theta (no cembed rebuild needed) =====
            t0 = mach_absolute_time();
            perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
            t_perturb += tb_ms(mach_absolute_time() - t0);

            // ===== 6. Gradient estimate + update =====
            float proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
            float update_scale = -lr * proj_grad;

            t0 = mach_absolute_time();
            // update_all_weights is just perturb_all_weights with update_scale
            perturb_all_weights(lw, embed, rms_final, mezo_seed, update_scale);
            t_perturb += tb_ms(mach_absolute_time() - t0);

            // Re-build compact embedding after weight update
            free(cembed);
            cembed = vocab_compact_embed(embed, &vm, DIM);

            // 7. Re-transpose for next step (ANE only)
            if (!cpu_only) {
                t0 = mach_absolute_time();
                RETRANSPOSE_AND_STAGE();
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
                        vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ*DIM));
                        rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, HIDDEN,SEQ,DIM, 1.0f, lw[L].W1,DIM, xnorm_buf,SEQ, 0.0f, h1,SEQ);
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, HIDDEN,SEQ,DIM, 1.0f, lw[L].W3,DIM, xnorm_buf,SEQ, 0.0f, h3,SEQ);
                        } else {
                            io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                            ane_eval_req(dk.w13Fwd, plr[L].w1Fwd);
                            io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ);
                            io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP);
                            ane_eval_req(dk.w13Fwd, plr[L].w3Fwd);
                            io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ);
                        }
                        for (int i = 0; i < HIDDEN*SEQ; i++) { float s = h1[i]/(1.0f+expf(-h1[i])); silu_out[i] = s*h3[i]; }
                        if (cpu_only) {
                            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, DIM,SEQ,HIDDEN, 1.0f, lw[L].W2,HIDDEN, silu_out,SEQ, 0.0f, o_out,SEQ);
                        } else {
                            io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP);
                            ane_eval_req(dk.w2Fwd, plr[L].w2Fwd);
                            io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ);
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
                                        total_train_ms, wall, total_steps_done);
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
        printf("mode:             mezo-%s\n", cpu_only ? "cpu" : "ane");
        printf("epsilon:          %g\n", epsilon);
        printf("lr:               %g\n", lr);

        // Cleanup
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
            free_kern(dk.wqFwd); free_kern(dk.wkvFwd); free_kern(dk.w13Fwd);
            free_kern(dk.w2Fwd); free_kern(dk.woFwd);
            for (int L = 0; L < NLAYERS; L++) {
                CFRelease(pls[L].wqFwd_in); CFRelease(pls[L].wkFwd_in);
                CFRelease(pls[L].wvFwd_in); CFRelease(pls[L].woFwd_in);
                CFRelease(pls[L].w1Fwd_in); CFRelease(pls[L].w3Fwd_in);
                CFRelease(pls[L].w2Fwd_in);
                CFRelease((id)plr[L].wqFwd); CFRelease((id)plr[L].wkFwd);
                CFRelease((id)plr[L].wvFwd); CFRelease((id)plr[L].woFwd);
                CFRelease((id)plr[L].w1Fwd); CFRelease((id)plr[L].w3Fwd);
                CFRelease((id)plr[L].w2Fwd);
            }
        }
    }
    return 0;
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd training && make mezo MODEL=autoresearch
```

Expected: `train_mezo` binary created with no errors.

- [ ] **Step 3: Commit**

```bash
git add training/train_mezo.m
git commit -m "feat: add MeZO zeroth-order training binary (CPU + ANE)"
```

---

## Chunk 2: Tests

### Task 3: Add MeZO tests to test_training.sh

**Files:**
- Modify: `tests/test_training.sh`

Add 4 tests after Test 8, before the Summary section:

- [ ] **Step 1: Add Test 9 (MeZO compilation)**

```bash
# Test 9: MeZO compilation
echo "Test 9: MeZO compilation"
make mezo MODEL=autoresearch >/dev/null 2>&1
if [ -f ./train_mezo ]; then
    pass "MeZO compiles cleanly"
else
    fail "MeZO compilation failed"
fi
```

- [ ] **Step 2: Add Test 10 (MeZO CPU-only forward, step 0 loss sanity)**

```bash
# Test 10: MeZO CPU-only forward (step 0 loss)
echo "Test 10: MeZO CPU-only forward"
MEZO_OUT=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --cpu-only --seed 42 2>&1)
MEZO_LOSS=$(echo "$MEZO_OUT" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
if [ -n "$MEZO_LOSS" ] && awk "BEGIN {exit !($MEZO_LOSS > 9.0 && $MEZO_LOSS < 10.5)}"; then
    pass "MeZO step 0 loss=$MEZO_LOSS (expected ~9.7)"
else
    fail "MeZO step 0 loss=$MEZO_LOSS outside range [9.0, 10.5]"
fi
```

- [ ] **Step 3: Add Test 11 (MeZO loss decreases over 200 steps)**

```bash
# Test 11: MeZO loss decreases (learning signal)
echo "Test 11: MeZO learning signal (200 steps)"
MEZO_OUT2=$(./train_mezo --scratch --data "$DATA" --lr 1e-4 --epsilon 1e-3 \
    --steps 200 --time 60 --cpu-only --seed 42 2>&1)
MEZO_FINAL=$(echo "$MEZO_OUT2" | grep "^final_loss_plus:" | awk '{print $2}')
MEZO_INIT=$(echo "$MEZO_OUT2" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
if [ -n "$MEZO_FINAL" ] && [ -n "$MEZO_INIT" ] && awk "BEGIN {exit !($MEZO_FINAL < $MEZO_INIT)}"; then
    pass "MeZO loss decreased ($MEZO_INIT -> $MEZO_FINAL)"
else
    fail "MeZO loss did not decrease ($MEZO_INIT -> $MEZO_FINAL)"
fi
```

- [ ] **Step 4: Add Test 12 (MeZO ANE mode)**

```bash
# Test 12: MeZO ANE matmul-only mode
echo "Test 12: MeZO ANE mode"
MEZO_ANE=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --ane-matmul-only --seed 42 2>&1)
if echo "$MEZO_ANE" | grep -q "Compiled"; then
    MEZO_ANE_LOSS=$(echo "$MEZO_ANE" | grep "^step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
    if [ -n "$MEZO_ANE_LOSS" ] && awk "BEGIN {exit !($MEZO_ANE_LOSS > 9.0 && $MEZO_ANE_LOSS < 10.5)}"; then
        pass "MeZO ANE: loss=$MEZO_ANE_LOSS"
    else
        fail "MeZO ANE: loss=$MEZO_ANE_LOSS outside range"
    fi
else
    fail "MeZO ANE kernel compilation failed"
fi
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd training && bash ../tests/test_training.sh
```

Expected: Tests 9-12 all PASS (Tests 1-8 should also still pass).

- [ ] **Step 6: Commit**

```bash
git add tests/test_training.sh
git commit -m "test: add MeZO training tests (compilation, loss sanity, learning signal, ANE mode)"
```

---

## Chunk 3: Run Experimental Matrix

### Task 4: Run the 8-condition experimental matrix

This is the research execution — running all conditions from the design spec section 7.

- [ ] **Step 1: Run Condition 1 (Backprop, CPU, from-scratch, 120s)**

```bash
cd training && make clean && make MODEL=autoresearch
./train --scratch --cpu-only --time 120 --seed 42 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition1_bp_cpu_scratch_120s.txt
```

- [ ] **Step 2: Run Condition 3 (MeZO, CPU, from-scratch, 120s)**

```bash
./train_mezo --scratch --cpu-only --time 120 --seed 42 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition3_mezo_cpu_scratch_120s.txt
```

- [ ] **Step 3: Run Condition 2 (Backprop, ANE, from-scratch, 120s)**

```bash
./train --scratch --ane-matmul-only --time 120 --seed 42 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition2_bp_ane_scratch_120s.txt
```

- [ ] **Step 4: Run Condition 4 (MeZO, ANE, from-scratch, 120s)**

```bash
./train_mezo --scratch --ane-matmul-only --time 120 --seed 42 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition4_mezo_ane_scratch_120s.txt
```

- [ ] **Step 5: Compare results — create results summary**

Parse all 4 output files: extract final_loss, num_steps, ms/step, mode. Compare:
- MeZO CPU vs Backprop CPU (steps/sec, loss trajectory)
- MeZO ANE vs MeZO CPU (ANE speedup for forward-only)
- From-scratch ZO learning signal (novel result — does loss decrease at all?)

- [ ] **Step 6: Commit results**

```bash
mkdir -p results
git add results/
git commit -m "data: MeZO vs backprop experimental results (conditions 1-4, from-scratch)"
```

---

### Task 5: Fine-tuning conditions (requires SmolLM2-135M checkpoint)

**Prerequisites:** `python3 tools/hf_to_ane.py HuggingFaceTB/SmolLM2-135M` must be run first to create the checkpoint.

- [ ] **Step 1: Convert SmolLM2-135M to BLZT format (if not already done)**

```bash
cd training && python3 ../tools/hf_to_ane.py HuggingFaceTB/SmolLM2-135M
```

- [ ] **Step 2: Build for SmolLM2-135M**

```bash
make clean && make MODEL=smollm2_135m && make mezo MODEL=smollm2_135m
```

- [ ] **Step 3: Run Conditions 5-8 (fine-tuning, 120s each)**

```bash
# Condition 5: Backprop CPU fine-tune
./train --cpu-only --time 120 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition5_bp_cpu_finetune_120s.txt

# Condition 6: Backprop ANE fine-tune
./train --ane-matmul-only --time 120 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition6_bp_ane_finetune_120s.txt

# Condition 7: MeZO CPU fine-tune
./train_mezo --resume ane_smollm2_135m_ckpt.bin --cpu-only --time 120 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition7_mezo_cpu_finetune_120s.txt

# Condition 8: MeZO ANE fine-tune
./train_mezo --resume ane_smollm2_135m_ckpt.bin --ane-matmul-only --time 120 --data ../tinystories_smollm2_data00.bin 2>&1 | tee ../results/condition8_mezo_ane_finetune_120s.txt
```

- [ ] **Step 4: Full analysis — all 8 conditions**

Create `results/analysis.md` comparing all conditions across:
- val_loss (start → end)
- Steps completed in time budget
- ms/step breakdown (forward, perturb, transpose)
- Peak memory
- Key finding: does MeZO-ANE beat MeZO-CPU for fine-tuning?

- [ ] **Step 5: Commit**

```bash
git add results/
git commit -m "data: complete 2x2x2 experimental matrix results"
```
