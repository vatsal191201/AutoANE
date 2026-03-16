# Phase 5: Sparse-HiZOO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sparse perturbation masking and diagonal Hessian preconditioning to the MeZO training loop to improve convergence rate, then validate with controlled experiments.

**Architecture:** All changes are in `train_mezo.m`. New `perturb_lora_hizoo()` function replaces `perturb_lora_weights()` when sparse/hessian flags are active. New `update_hessian()` and `compute_sparse_mask()` utility functions. Three new CLI flags. The forward pass macro (DO_FORWARD_PASS) and conv-fused kernels are untouched.

**Tech Stack:** C/Objective-C, Accelerate framework (vDSP), LAPACK (for sort), xoshiro256+ PRNG.

**Spec:** `docs/specs/2026-03-16-phase5-sparse-hizoo-design.md`

---

## Chunk 1: Core Infrastructure

### Task 1: Add CLI flags and state variables

**Files:**
- Modify: `training/train_mezo.m:631-681` (defaults and CLI parsing)

- [ ] **Step 1: Add new default variables after line 650**

Add these after the `probe_gradient` declaration (line 650):

```c
float sparse_ratio = 0.0f;   // Fraction of params to EXCLUDE (0=none, 0.8=keep 20%)
float hessian_alpha = 0.0f;  // Hessian EMA rate (0=disabled, >0=enable HiZOO)
int mask_refresh = 100;       // Recompute sparse mask every N steps
```

- [ ] **Step 2: Add CLI argument parsing after line 680**

Add these after the `--probe-gradient` parser:

```c
else if (strcmp(argv[i], "--sparse-ratio") == 0 && i+1 < argc) { sparse_ratio = atof(argv[++i]); }
else if (strcmp(argv[i], "--hessian-alpha") == 0 && i+1 < argc) { hessian_alpha = atof(argv[++i]); }
else if (strcmp(argv[i], "--mask-refresh") == 0 && i+1 < argc) { mask_refresh = atoi(argv[++i]); }
```

- [ ] **Step 3: Add validation after line 698**

```c
if (sparse_ratio < 0.0f || sparse_ratio >= 1.0f) {
    fprintf(stderr, "ERROR: --sparse-ratio must be in [0, 1)\n"); return 1;
}
if (hessian_alpha < 0.0f) {
    fprintf(stderr, "ERROR: --hessian-alpha must be >= 0\n"); return 1;
}
```

- [ ] **Step 4: Print config in the startup banner**

Find the existing config printout (search for `printf.*epsilon`) and add:

```c
if (sparse_ratio > 0) printf("  sparse_ratio=%.3f  mask_refresh=%d\n", sparse_ratio, mask_refresh);
if (hessian_alpha > 0) printf("  hessian_alpha=%.2e\n", hessian_alpha);
```

- [ ] **Step 5: Compile and verify no errors**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m`
Expected: Clean compilation with no warnings from new code.

- [ ] **Step 6: Verify backward compat — standard MeZO still works**

Run: `./train_mezo --resume ane_smollm2_360m_ckpt.bin --data ../tinystories_smollm2_data00.bin --lr 1e-4 --eps 1e-3 --steps 5 --cpu-only --lora --lora-rank 8 --lora-split --seed 42`
Expected: Runs 5 steps, prints loss values, no crash. Losses should match prior runs (same seed, same config).

- [ ] **Step 7: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: add CLI flags for sparse-ratio, hessian-alpha, mask-refresh"
```

---

### Task 2: Implement compute_sparse_mask()

**Files:**
- Modify: `training/train_mezo.m` (add function after `count_lora_params()`, around line 243)

- [ ] **Step 1: Add the comparison function for qsort**

Insert after `count_lora_params()`:

```c
// ===== Sparse MeZO: magnitude-based parameter mask =====
static int cmp_float_abs(const void *a, const void *b) {
    float fa = fabsf(*(const float *)a), fb = fabsf(*(const float *)b);
    return (fa > fb) - (fa < fb);
}
```

- [ ] **Step 2: Add compute_sparse_mask()**

```c
// Compute binary mask: mask[i]=1 for parameters to PERTURB (small magnitude),
// mask[i]=0 for parameters to SKIP (large magnitude).
// keep_ratio = 1.0 - sparse_ratio (fraction to keep).
// Uses global thresholding when sparse_ratio <= 0.05 (RMS-only exclusion),
// otherwise per-group thresholding is used.
static void compute_sparse_mask(LoRALayer *ll, LayerWeights *lw, float *rms_final,
                                int nlayers, uint8_t *mask, float sparse_ratio,
                                size_t total_params) {
    if (sparse_ratio <= 0.0f) {
        memset(mask, 1, total_params);  // All enabled
        return;
    }

    // Collect all parameter magnitudes into temp array
    float *mags = (float *)safe_malloc(total_params * sizeof(float));
    size_t idx = 0;
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        // LoRA attention matrices
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Aq[j]);
        for (size_t j = 0; j < (size_t)Q_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bq[j]);
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Ak[j]);
        for (size_t j = 0; j < (size_t)KV_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bk[j]);
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Av[j]);
        for (size_t j = 0; j < (size_t)KV_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bv[j]);
        for (size_t j = 0; j < (size_t)r * Q_DIM; j++) mags[idx++] = fabsf(ll[L].Ao[j]);
        for (size_t j = 0; j < (size_t)DIM * r; j++) mags[idx++] = fabsf(ll[L].Bo[j]);
        // RMS norms
        for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(lw[L].rms_att[j]);
        for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(lw[L].rms_ffn[j]);
    }
    for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(rms_final[j]);
    assert(idx == total_params);

    // Sort to find threshold at (1 - sparse_ratio) percentile
    float *sorted = (float *)safe_malloc(total_params * sizeof(float));
    memcpy(sorted, mags, total_params * sizeof(float));
    qsort(sorted, total_params, sizeof(float), cmp_float_abs);

    size_t keep_count = (size_t)((1.0f - sparse_ratio) * total_params);
    if (keep_count == 0) keep_count = 1;
    if (keep_count > total_params) keep_count = total_params;
    float threshold = sorted[keep_count - 1];  // Keep params with |val| <= threshold

    // Build mask
    for (size_t i = 0; i < total_params; i++) {
        mask[i] = (mags[i] <= threshold) ? 1 : 0;
    }

    // Report stats
    size_t n_active = 0;
    for (size_t i = 0; i < total_params; i++) n_active += mask[i];
    printf("  [Sparse mask] ratio=%.3f  threshold=%.6f  active=%zu/%zu (%.1f%%)\n",
           sparse_ratio, threshold, n_active, total_params,
           100.0f * n_active / total_params);

    free(mags);
    free(sorted);
}
```

- [ ] **Step 3: Compile and verify**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: add compute_sparse_mask() with global magnitude thresholding"
```

---

### Task 3: Implement update_hessian()

**Files:**
- Modify: `training/train_mezo.m` (add function after `compute_sparse_mask()`)

- [ ] **Step 1: Add update_hessian() function**

```c
// ===== HiZOO: Diagonal Hessian update via second-order finite difference =====
// Formula: h_est[i] = |ΔL/ε²| · z[i]²
//          H[i] = (1-α)·H[i] + α·h_est[i]
// z[i] is raw Rademacher (±1), so z[i]²=1 always.
// The per-element differentiation comes from EMA over many random draws.
// ΔL = L(θ+εz) + L(θ-εz) - 2·L(θ) is the second-order finite difference.
static void update_hessian(float *H, size_t n,
                           float loss_plus, float loss_minus, float loss_0,
                           float epsilon, float alpha) {
    float delta_L = loss_plus + loss_minus - 2.0f * loss_0;
    float curvature = fabsf(delta_L) / (epsilon * epsilon);

    // Since z[i]² = 1 for Rademacher, h_est = curvature for all i.
    // But the VALUE of curvature varies each step because different z's
    // probe different directions in parameter space.
    // EMA accumulates: steps where parameter i contributes more to loss
    // will see larger curvature on average.
    // NOTE: This is a simplification — exact HiZOO accounts for preconditioned
    // perturbation scaling. See design spec Section "Note on z_scaled".
    for (size_t i = 0; i < n; i++) {
        H[i] = (1.0f - alpha) * H[i] + alpha * curvature;
        if (H[i] < 1e-8f) H[i] = 1e-8f;    // Floor: 1/sqrt(1e-8) = 1e4 max amplification
        if (H[i] > 1e6f)  H[i] = 1e6f;      // Ceiling: 1/sqrt(1e6) ≈ 1e-3 min amplification
    }
}
```

- [ ] **Step 2: Add print_hessian_stats() diagnostic**

```c
static void print_hessian_stats(const float *H, size_t n, int step) {
    float h_min = H[0], h_max = H[0];
    double h_sum = 0, h_sum2 = 0;
    for (size_t i = 0; i < n; i++) {
        if (H[i] < h_min) h_min = H[i];
        if (H[i] > h_max) h_max = H[i];
        h_sum += H[i];
        h_sum2 += (double)H[i] * H[i];
    }
    double h_mean = h_sum / n;
    double h_var = h_sum2 / n - h_mean * h_mean;
    printf("  [Hessian@%d] min=%.4e max=%.4e mean=%.4e std=%.4e ratio=%.1f\n",
           step, h_min, h_max, (float)h_mean, (float)sqrt(fmax(h_var,0)),
           h_max / (h_min + 1e-30f));
}
```

- [ ] **Step 3: Compile and verify**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m`
Expected: Clean compilation.

- [ ] **Step 4: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: add update_hessian() and print_hessian_stats() diagnostic"
```

---

### Task 4: Implement perturb_lora_hizoo() — Hessian-scaled sparse perturbation

**Files:**
- Modify: `training/train_mezo.m` (add function after `perturb_lora_weights()`, around line 510)

- [ ] **Step 1: Add perturb_lora_hizoo()**

This function replaces `perturb_lora_weights()` when sparse/hessian is active. It applies: `param[i] += scale * z[i] * mask[i] / sqrt(H[i])`.

```c
// ===== Sparse-HiZOO: perturb with mask and Hessian preconditioning =====
// param[i] += scale * z[i] * mask[i] / sqrt(H[i])
// z[i] is Rademacher ±1 generated from seed (same sequence as perturb_lora_weights).
// When mask==NULL and H==NULL, reduces exactly to perturb_lora_weights().
static void perturb_lora_hizoo(LoRALayer *ll, LayerWeights *lw,
                                float *rms_final, int nlayers, uint64_t seed,
                                float scale, const uint8_t *mask, const float *H) {
    xo_seed(seed);
    size_t idx = 0;  // Tracks position in mask/H arrays

    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;

        // Macro: perturb a buffer with optional mask/Hessian
        #define PERTURB_HIZOO(buf, count) do { \
            size_t _n = (count); \
            for (size_t _i = 0; _i < _n; _i++) { \
                uint64_t _r = xo_next(); \
                float z_i = (_r & 1) ? 1.0f : -1.0f; \
                float s = scale; \
                if (mask && !mask[idx + _i]) s = 0.0f; \
                if (H) s /= sqrtf(H[idx + _i]); \
                (buf)[_i] += s * z_i; \
            } \
            idx += _n; \
        } while(0)

        // Attention adapters (same order as perturb_lora_weights)
        PERTURB_HIZOO(ll[L].Aq, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bq, (size_t)Q_DIM * r);
        PERTURB_HIZOO(ll[L].Ak, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bk, (size_t)KV_DIM * r);
        PERTURB_HIZOO(ll[L].Av, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bv, (size_t)KV_DIM * r);
        PERTURB_HIZOO(ll[L].Ao, (size_t)r * Q_DIM);
        PERTURB_HIZOO(ll[L].Bo, (size_t)DIM * r);
        // RMS norms
        PERTURB_HIZOO(lw[L].rms_att, DIM);
        PERTURB_HIZOO(lw[L].rms_ffn, DIM);

        #undef PERTURB_HIZOO
    }
    // rms_final
    {
        size_t _n = DIM;
        for (size_t _i = 0; _i < _n; _i++) {
            uint64_t _r = xo_next();
            float z_i = (_r & 1) ? 1.0f : -1.0f;
            float s = scale;
            if (mask && !mask[idx + _i]) s = 0.0f;
            if (H) s /= sqrtf(H[idx + _i]);
            rms_final[_i] += s * z_i;
        }
        idx += _n;
    }
}
```

- [ ] **Step 2: Verify parameter ordering matches perturb_lora_weights()**

Critical correctness check: the PRNG sequence in `perturb_lora_hizoo()` must consume the same number of random values in the same order as `perturb_lora_weights()`. Compare:
- `perturb_lora_weights`: uses `perturb_buffer()` which calls `xo_next()` once per 4 elements (4-bit extraction)
- `perturb_lora_hizoo`: calls `xo_next()` once per element (1-bit extraction)

**This is a MISMATCH.** `perturb_buffer()` extracts 4 bits per `xo_next()` call, while the HIZOO function extracts 1 bit per call. The PRNG sequences will diverge, meaning a +ε perturbation followed by a -ε perturbation using the old function won't cancel.

**Fix:** The HiZOO function must only be used for both +ε and -ε and restore. It cannot be mixed with `perturb_lora_weights()`. This is fine as long as we consistently use `perturb_lora_hizoo()` for all perturbation steps when sparse/hessian is enabled.

**Alternative fix:** Change the HiZOO function to also extract 4 bits per call, matching the existing PRNG consumption pattern. This is more complex but allows mixing.

We choose: **always use `perturb_lora_hizoo()` when sparse_ratio>0 or hessian_alpha>0**. Document this.

- [ ] **Step 3: Compile and verify**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m`
Expected: Clean compilation.

- [ ] **Step 4: Backward compat test — verify perturb_lora_hizoo with mask=NULL, H=NULL matches MeZO**

This requires a mini-test: run 5 steps with standard MeZO, record losses. Then run 5 steps using `perturb_lora_hizoo(mask=NULL, H=NULL)` with the same seed. Losses should be **different** (due to different PRNG consumption pattern — 1 bit vs 4 bits per call). This confirms the PRNG mismatch. Document this: when sparse/hessian is off, we still use the old `perturb_lora_weights()` for exact backward compat.

- [ ] **Step 5: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: add perturb_lora_hizoo() with mask and Hessian scaling"
```

---

## Chunk 2: Training Loop Integration

### Task 5: Allocate Hessian and mask buffers

**Files:**
- Modify: `training/train_mezo.m` (after LoRA layer allocation, around line 800-900)

- [ ] **Step 1: Add buffer allocation**

Find the section where `lora_layers` are allocated (search for `lora_layer_alloc`). After that block, add:

```c
// Phase 5: Sparse-HiZOO buffers
float *diag_hessian = NULL;
uint8_t *sparse_mask = NULL;
size_t hizoo_n_params = 0;
bool use_hizoo = (sparse_ratio > 0.0f || hessian_alpha > 0.0f);

if (use_lora && use_hizoo) {
    hizoo_n_params = count_lora_params(lora_layers, NLAYERS);
    printf("Phase 5: Sparse-HiZOO enabled (params=%zu, sparse=%.3f, alpha=%.2e)\n",
           hizoo_n_params, sparse_ratio, hessian_alpha);

    // Diagonal Hessian (initialized to 1.0 = identity preconditioner)
    if (hessian_alpha > 0.0f) {
        diag_hessian = (float *)safe_malloc(hizoo_n_params * sizeof(float));
        for (size_t i = 0; i < hizoo_n_params; i++) diag_hessian[i] = 1.0f;
        printf("  Hessian buffer: %.1f MB\n", hizoo_n_params * 4.0f / (1024*1024));
    }

    // Sparse mask (computed below)
    if (sparse_ratio > 0.0f) {
        sparse_mask = (uint8_t *)safe_calloc(hizoo_n_params, 1);
        // Print parameter magnitude stats before computing mask
        // (validates Assumption A4: RMS ~1.0 vs LoRA ~0.01)
        {
            float lora_sum = 0, rms_sum = 0;
            size_t lora_cnt = 0, rms_cnt = 0;
            for (int L = 0; L < NLAYERS; L++) {
                int r = lora_layers[L].rank;
                size_t attn_sz = (size_t)r*DIM*3 + (size_t)Q_DIM*r + (size_t)KV_DIM*r*2
                               + (size_t)r*Q_DIM + (size_t)DIM*r;
                // Rough sum of LoRA magnitudes
                for (size_t j = 0; j < (size_t)r*DIM; j++) lora_sum += fabsf(lora_layers[L].Aq[j]);
                lora_cnt += (size_t)r*DIM;
                // RMS magnitudes
                for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(lw[L].rms_att[j]);
                for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(lw[L].rms_ffn[j]);
                rms_cnt += 2 * DIM;
            }
            for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(rms_final[j]);
            rms_cnt += DIM;
            printf("  [Magnitude check] LoRA mean=%.6f (%zu params)  RMS mean=%.4f (%zu params)\n",
                   lora_sum/fmaxf(lora_cnt,1), lora_cnt, rms_sum/fmaxf(rms_cnt,1), rms_cnt);
        }
        compute_sparse_mask(lora_layers, lw, rms_final, NLAYERS,
                           sparse_mask, sparse_ratio, hizoo_n_params);
        printf("  Mask buffer: %.1f MB\n", hizoo_n_params / (1024.0*1024));
    }
}
```

- [ ] **Step 2: Add cleanup at program exit**

Find the cleanup section (search for `free(lora_layers)` or end of main). Add:

```c
if (diag_hessian) free(diag_hessian);
if (sparse_mask) free(sparse_mask);
```

- [ ] **Step 3: Compile and smoke test**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m && ./train_mezo --resume ane_smollm2_360m_ckpt.bin --data ../tinystories_smollm2_data00.bin --lr 1e-4 --eps 1e-3 --steps 3 --cpu-only --lora --lora-rank 8 --lora-split --sparse-ratio 0.037`
Expected: Prints magnitude check (LoRA mean ~0.01, RMS mean ~1.0), sparse mask stats, runs 3 steps.

- [ ] **Step 4: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: allocate Hessian and sparse mask buffers with magnitude diagnostics"
```

---

### Task 6: Integrate Sparse-HiZOO into the training loop

**Files:**
- Modify: `training/train_mezo.m:1777-1853` (standard MeZO step)

- [ ] **Step 1: Add L0 forward pass for Hessian estimation**

Before the existing `// 1. Perturb +epsilon` (line 1781), add:

```c
                // ===== Phase 5: L0 forward pass for Hessian estimation =====
                float loss_0 = 0.0f;
                if (hessian_alpha > 0.0f) {
                    t0 = mach_absolute_time();
                    DO_FORWARD_PASS(input_tokens, ctargets, loss_0);
                    t_fwd += tb_ms(mach_absolute_time() - t0);
                }
```

- [ ] **Step 2: Replace perturbation calls with HiZOO variants**

Replace the standard MeZO perturbation block (lines 1783-1849) with a conditional:

```c
                // 1. Perturb +epsilon
                t0 = mach_absolute_time();
                if (use_lora && use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed,
                                       +epsilon, sparse_mask, diag_hessian);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon);
                    if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
                }
                if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
                t_perturb += tb_ms(mach_absolute_time() - t0);
```

Do the same pattern for the -2ε perturbation (step 3), +ε restore (step 5), and the update (step 6). The key: replace `perturb_lora_weights(...)` with `perturb_lora_hizoo(..., sparse_mask, diag_hessian)` whenever `use_hizoo` is true.

For the update step specifically:
```c
                // 6. Gradient estimate + update
                proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
                float update_scale = -lr * proj_grad;

                t0 = mach_absolute_time();
                if (use_lora && use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed,
                                       update_scale, sparse_mask, diag_hessian);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, update_scale);
                    if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, update_scale);
                }
                t_perturb += tb_ms(mach_absolute_time() - t0);
```

- [ ] **Step 3: Add Hessian update after gradient estimate**

After the gradient estimate and before the update, add:

```c
                // Phase 5: Update diagonal Hessian estimate
                if (hessian_alpha > 0.0f && diag_hessian) {
                    update_hessian(diag_hessian, hizoo_n_params,
                                   loss_plus, loss_minus, loss_0,
                                   epsilon, hessian_alpha);

                    // Print diagnostics every 100 steps
                    if (step % 100 == 0 || step <= start_step + 10) {
                        print_hessian_stats(diag_hessian, hizoo_n_params, step);
                    }
                }
```

- [ ] **Step 4: Add mask refresh**

After the update step, add:

```c
                // Phase 5: Refresh sparse mask periodically
                if (sparse_ratio > 0.0f && sparse_mask && step > start_step &&
                    (step % mask_refresh) == 0) {
                    compute_sparse_mask(lora_layers, lw, rms_final, NLAYERS,
                                       sparse_mask, sparse_ratio, hizoo_n_params);
                }
```

- [ ] **Step 5: Update step logging to include HiZOO info**

Find the existing step logging `printf` (around line 1770 or the equivalent in the standard MeZO branch). Add L0 and Hessian info:

```c
                if (step % 100 == 0 || step == start_step) {
                    printf("step %d  loss+=%.4f  loss-=%.4f  proj_grad=%.6f  lr=%.2e  step_ms=%.0f",
                           step, loss_plus, loss_minus, proj_grad, lr, step_ms);
                    if (hessian_alpha > 0) printf("  L0=%.4f  dL=%.4e", loss_0, loss_plus+loss_minus-2*loss_0);
                    printf("\n");
                }
```

- [ ] **Step 6: Compile**

Run: `cd /Users/vatsalb/Desktop/AutoANE_repo/training && make MODEL=smollm2_360m`
Expected: Clean compilation.

- [ ] **Step 7: Commit**

```bash
git add training/train_mezo.m
git commit -m "Phase 5: integrate Sparse-HiZOO into MeZO training loop"
```

---

## Chunk 3: Validation & Experiments

### Task 7: Mini-validation — 10-step diagnostics

**Files:**
- No code changes, just running the binary with diagnostic output.

- [ ] **Step 1: Validate Assumption A4 — magnitude gap**

Run: `./train_mezo --resume ../ane_smollm2_360m_ckpt.bin --data ../tinystories_smollm2_data00.bin --lr 1e-4 --eps 1e-3 --steps 1 --cpu-only --lora --lora-rank 8 --lora-split --sparse-ratio 0.037`
Expected: Prints `[Magnitude check] LoRA mean=~0.01  RMS mean=~1.0`. The 100x gap confirms A4.

- [ ] **Step 2: Validate sparse mask at 0.037 excludes RMS norms**

Same run as Step 1. Check output for: `[Sparse mask] ratio=0.037  active=1638400/1700800 (96.3%)`
The active count should be close to 1,638,400 (LoRA params only, RMS excluded).

- [ ] **Step 3: Validate Hessian estimation — 10-step diagnostic**

Run: `./train_mezo --resume ../ane_smollm2_360m_ckpt.bin --data ../tinystories_smollm2_data00.bin --lr 1e-4 --eps 1e-3 --steps 10 --cpu-only --lora --lora-rank 8 --lora-split --hessian-alpha 1e-6`
Expected: Hessian stats printed every step for first 10 steps. Check:
- H min/max/mean stay in [1e-8, 1e6] range
- H mean stays near 1.0 (small alpha = slow update)
- L0 values are printed and are close to L+ and L- (all near 2.06)
- dL = L+ + L- - 2*L0 should be small (order 1e-4 to 1e-2)
- No NaN or Inf

- [ ] **Step 4: Validate Assumption A9 — L0 forward pass timing**

From the 10-step Hessian run, check the step_ms. It should be ~393ms (262ms baseline + 131ms for L0). If significantly different, note the actual value.

- [ ] **Step 5: Validate backward compat — standard MeZO unchanged**

Run: `./train_mezo --resume ../ane_smollm2_360m_ckpt.bin --data ../tinystories_smollm2_data00.bin --lr 1e-4 --eps 1e-3 --steps 10 --cpu-only --lora --lora-rank 8 --lora-split --seed 42`
Run AGAIN with same flags. Losses must match exactly (deterministic). This confirms no accidental changes to the standard MeZO path.

- [ ] **Step 6: Document diagnostic results**

Create a preliminary section in the results log noting all diagnostic outputs.

- [ ] **Step 7: Commit diagnostics documentation**

```bash
git commit -m "Phase 5: validated 10-step diagnostics (magnitude gap, Hessian stability, timing)"
```

---

### Task 8: Experiment 5a — Sparse MeZO

- [ ] **Step 1: Run baseline (500 steps, standard MeZO)**

```bash
cd /Users/vatsalb/Desktop/AutoANE_repo/training
./train_mezo --resume ../ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-4 --eps 1e-3 --steps 500 --cpu-only \
    --lora --lora-rank 8 --lora-split --seed 42 \
    --conv-fused --val-every 50 2>&1 | tee ../results/phase5_5a_baseline.txt
```
Record: val_loss at step 500, 50-step moving average.

- [ ] **Step 2: Run sparse-ratio=0.037 (RMS exclusion only)**

```bash
./train_mezo --resume ../ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-4 --eps 1e-3 --steps 500 --cpu-only \
    --lora --lora-rank 8 --lora-split --seed 42 \
    --conv-fused --val-every 50 --sparse-ratio 0.037 \
    2>&1 | tee ../results/phase5_5a_rms_only.txt
```

- [ ] **Step 3: Run sparse-ratio=0.5**

Same as Step 2 but `--sparse-ratio 0.5`.

- [ ] **Step 4: Run sparse-ratio=0.8**

Same as Step 2 but `--sparse-ratio 0.8`.

- [ ] **Step 5: Compare results and apply go/no-go gate**

Compare 500-step val_loss deltas. Go/no-go: improvement > 0.001 over baseline.

- [ ] **Step 6: Document results in results log**

- [ ] **Step 7: Commit results**

```bash
git add results/phase5_5a_*.txt
git commit -m "Phase 5 Experiment 5a: Sparse MeZO results (4 configs)"
```

---

### Task 9: Experiment 5b — HiZOO

- [ ] **Step 1: Run hessian-alpha=1e-8**

```bash
./train_mezo --resume ../ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-4 --eps 1e-3 --steps 500 --cpu-only \
    --lora --lora-rank 8 --lora-split --seed 42 \
    --conv-fused --val-every 50 --hessian-alpha 1e-8 \
    2>&1 | tee ../results/phase5_5b_alpha_1e8.txt
```

- [ ] **Step 2: Run hessian-alpha=1e-6**

Same but `--hessian-alpha 1e-6`.

- [ ] **Step 3: Run hessian-alpha=1e-4**

Same but `--hessian-alpha 1e-4`.

- [ ] **Step 4: Compare results and apply go/no-go gate**

Go/no-go: val_loss improvement > 1.5x baseline delta (accounting for extra compute).

- [ ] **Step 5: Document results**

- [ ] **Step 6: Commit results**

```bash
git add results/phase5_5b_*.txt
git commit -m "Phase 5 Experiment 5b: HiZOO diagonal Hessian results (3 alpha values)"
```

---

### Task 10: Experiment 5c — Combined (conditional)

Only execute if BOTH 5a and 5b show positive signal.

- [ ] **Step 1: Run combined best-of-5a + best-of-5b**

Use the best sparse_ratio from 5a and best alpha from 5b.

- [ ] **Step 2: Run sparse-only control (best sparse_ratio, alpha=0)**

- [ ] **Step 3: Run hessian-only control (alpha=best, sparse_ratio=0)**

- [ ] **Step 4: Compare and apply go/no-go gate**

Combined must beat the better individual method.

- [ ] **Step 5: Document results and commit**

---

## Chunk 4: Documentation & Cleanup

### Task 11: Write results log

**Files:**
- Create: `docs/specs/2026-03-16-phase5-sparse-hizoo-results.md`

- [ ] **Step 1: Write results document**

Include: all experimental configs, exact val_loss numbers, go/no-go gate outcomes, Hessian diagnostics, assumption validation results, comparison table, conclusions.

- [ ] **Step 2: Commit**

### Task 12: Update existing documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/FINDINGS.md`
- Modify: `docs/NEXT_STEPS.md`
- Modify: `docs/TECHNICAL_REPORT.md`

- [ ] **Step 1: Update all docs with Phase 5 results**

- [ ] **Step 2: Triple-check numerical consistency across all documents**

- [ ] **Step 3: Commit and push**

```bash
git push
```
