# Design: Zeroth-Order (MeZO) Training on Apple Neural Engine

**Date**: 2026-03-12
**Status**: Approved
**Goal**: First ZO training on NPU hardware + first empirical test of from-scratch ZO language model training

---

## 1. Problem Statement

Our 44 experiments show CPU-only beats ANE for backprop training because:
- IOSurface weight staging costs 4-8ms/step
- fp16 precision degrades quality ~16%
- Backward pass falls back to CPU for most ops (SDPA, dW)

MeZO (NeurIPS 2023) eliminates backward passes entirely — training uses only 2 forward passes per step. This plays to ANE's strength (2.5x matmul speedup) while removing its weakness (backward pass overhead).

**Novel contributions:**
1. First zeroth-order training on any NPU/Neural Engine hardware
2. First empirical test of from-scratch ZO language model training (literature is fine-tuning only)
3. Systematic 2×2×2 comparison: {ZO, BP} × {CPU, ANE} × {scratch, finetune}

## 2. MeZO Algorithm

SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimate:

```
∇̂L(θ; B) = [L(θ + εz; B) - L(θ - εz; B)] / (2ε) · z
```

where z ~ N(0, I_d) and ε is the perturbation scale.

**Per-step procedure (Algorithm 1 from the paper):**

1. Sample batch B and random seed s
2. Perturb: θ ← θ + ε·z (z regenerated deterministically from seed s)
3. Forward pass → loss_plus
4. Perturb: θ ← θ - 2ε·z (now at θ - ε·z)
5. Forward pass → loss_minus
6. Restore: θ ← θ + ε·z (back to original θ)
7. projected_grad = (loss_plus - loss_minus) / (2ε)
8. Update: θ_i ← θ_i - lr × projected_grad × z_i (regenerate z from seed s)

**In-place seed trick**: Instead of storing z ∈ R^d (36M+ floats = 140MB+), store a single uint64_t seed. Reset srand48(seed) to regenerate the identical z sequence each of the 4 times it's needed per step. Memory footprint = inference memory.

**Key properties:**
- 2 forward passes per step, 0 backward passes
- Memory = inference (no activations, gradients, or optimizer state stored)
- ~100x more steps needed than backprop for comparable fine-tuning quality
- Each step is faster (no backward), but convergence is slower
- Works for fine-tuning (proven). From-scratch is untested (theory predicts poor convergence due to high effective rank of random-init loss landscape).

## 3. Implementation: `training/train_mezo.m`

Separate binary (~600 lines) sharing headers with train.m. Forward-pass only — no backward kernels compiled.

### 3.1 Shared Code (via headers)

- `config.h`: Model structs, safe_malloc/calloc, checkpoint header, ane_init
- `cpu_ops.h`: rmsnorm (forward only), cross_entropy_loss, embed_lookup, VocabMap
- `io.h`: IOSurface helpers, kernel compile/eval, GQA tile (forward only)
- `mil_dynamic.h`: Forward MIL kernels only (sdpaFwd, woFwd, ffnFused, wqFwd, wkvFwd, w13Fwd, w2Fwd)

### 3.2 NOT Included (vs train.m)

- Backward MIL kernels (sdpaBwd1/2, qBwd, kvBwd, wotBwd, ffnBwdW2t, ffnBwdW13t)
- Backward IOSurfaces and requests
- LayerGrads allocation
- AdamState allocation (m, v buffers)
- Per-layer activation cache (LayerActs — Q, K, V, attn_out, layer_in, xnorm per layer)
- Gradient sanitization, accumulation, clipping, loss scaling
- LoRA infrastructure (deferred)

### 3.3 Memory Comparison (SmolLM2-135M, 576d/30L)

| Component | Backprop+Adam | MeZO |
|-----------|--------------|------|
| Weights (fp32) | 520MB | 520MB |
| Adam m state | 520MB | 0 |
| Adam v state | 520MB | 0 |
| Gradients | 520MB | 0 |
| Per-layer activation cache | ~200MB | 0 |
| Forward buffers (reusable) | ~50MB | ~50MB |
| Perturbation vector z | 0 | 0 (seed trick) |
| **Total** | **~2.3GB** | **~570MB** |

4.0x memory reduction. For SmolLM2-360M: backprop needs ~6.2GB (doesn't fit on 8GB devices), MeZO needs ~1.4GB.

### 3.4 Forward Pass (simplified)

Structurally identical to train.m's forward pass but buffers are reused across layers (no per-layer caching for backward):

```c
float *x_cur, *x_next, *xnorm;          // SEQ × DIM
float *Q;                                 // SEQ × Q_DIM
float *K, *V;                             // SEQ × KV_DIM
float *attn_out;                          // SEQ × Q_DIM
float *o_out;                             // SEQ × DIM
float *h1, *h3, *silu_out;              // SEQ × HIDDEN
float *k_tiled, *v_tiled;               // SEQ × Q_DIM (GQA)
float *logits;                            // SEQ × CV (compact vocab)
```

~50MB for 512d/4L. Does not grow with NLAYERS.

### 3.5 Perturbation Implementation

```c
// Box-Muller: 2 uniform → 1 standard normal
static inline float box_muller_next(void) {
    double u1 = drand48(), u2 = drand48();
    return (float)(sqrt(-2.0 * log(u1 + 1e-30)) * cos(6.283185307179586 * u2));
}

// Perturb a buffer in-place: buf[i] += scale * z_i
static void perturb_buffer(float *buf, size_t n, float scale) {
    for (size_t i = 0; i < n; i++)
        buf[i] += scale * box_muller_next();
}

// Perturb ALL model weights using deterministic seed
static void perturb_all_weights(LayerWeights *lw, float *embed, float *rms_final,
                                float *cls, int cv, uint64_t seed, float scale) {
    srand48((long)seed);
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
    perturb_buffer(cls, (size_t)cv * DIM, scale);
}
```

### 3.6 MeZO Training Loop

```c
for (int step = 0; step < total_steps; step++) {
    // Time budget check
    if (time_budget > 0 && elapsed >= time_budget) break;

    // Sample data
    size_t pos = (size_t)(drand48() * max_pos);
    uint16_t *input = token_data + pos;
    uint16_t *target = token_data + pos + 1;

    uint64_t seed = (uint64_t)step * 1000003ULL + init_seed;

    // 1. Perturb +ε
    perturb_all_weights(lw, embed, rms_final, cls, CV, seed, +epsilon);
    if (!cpu_only) retranspose_all_weights(...);

    // 2. Forward → loss_plus
    float loss_plus = forward_pass(lw, embed, rms_final, cls, input, target, ...);

    // 3. Perturb -2ε (to θ - εz)
    perturb_all_weights(lw, embed, rms_final, cls, CV, seed, -2.0f * epsilon);
    if (!cpu_only) retranspose_all_weights(...);

    // 4. Forward → loss_minus
    float loss_minus = forward_pass(lw, embed, rms_final, cls, input, target, ...);

    // 5. Restore to original θ
    perturb_all_weights(lw, embed, rms_final, cls, CV, seed, +epsilon);

    // 6. Gradient estimate + update
    float proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
    float update_scale = -lr * proj_grad;

    srand48((long)seed);
    // Update each weight: w_i += -lr * proj_grad * z_i
    update_buffer(embed, (size_t)VOCAB * DIM, update_scale);
    for (int L = 0; L < NLAYERS; L++) {
        update_buffer(lw[L].rms_att, DIM, update_scale);
        update_buffer(lw[L].Wq, WQ_SZ, update_scale);
        // ... all weight matrices
    }
    update_buffer(rms_final, DIM, update_scale);
    update_buffer(cls, (size_t)CV * DIM, update_scale);

    // 7. Re-transpose for next step (ANE only)
    if (!cpu_only) retranspose_all_weights(...);

    // 8. LR schedule (cosine decay, no warmup)
    lr = mezo_lr_schedule(step, total_steps, base_lr);

    // 9. Print/log
    if (step % 100 == 0) printf("step %d loss_plus=%.4f loss_minus=%.4f proj_grad=%.6f\n", ...);
}
```

### 3.7 ANE Weight Re-transposition

After each perturbation or update, ANE mode requires updating transposed weight buffers (the ANE forward kernels expect transposed weights packed into IOSurfaces). This is an overhead MeZO pays 4x per step:

```c
static void retranspose_all_weights(LayerWeights *lw, float **Wqt, ...) {
    for (int L = 0; L < NLAYERS; L++) {
        transpose_weight(Wqt[L], lw[L].Wq, Q_DIM, DIM);
        transpose_weight(Wkt[L], lw[L].Wk, KV_DIM, DIM);
        // ... all 7 matrices per layer
    }
}
```

This is O(total_params) per call — for 36M params at 4 bytes, ~140MB of memcpy-equivalent work. At ~10GB/s memory bandwidth, ~14ms per re-transpose. 4 per step = ~56ms overhead for ANE mode. This is significant and will reduce ANE's advantage.

Optimization: only re-transpose before forward passes (2x per step, not 4x). The restore and update steps don't need transposed weights.

## 4. CLI Interface

```
./train_mezo [options]

Options:
  --scratch           Random weight initialization (vs loading checkpoint)
  --data <path>       Training data file (required)
  --lr <float>        Learning rate (default: 1e-5)
  --epsilon <float>   SPSA perturbation scale (default: 1e-3)
  --steps <int>       Max training steps (default: 999999)
  --time <float>      Time budget in seconds (default: 0 = unlimited)
  --cpu-only          CPU fp32 only (recommended)
  --ane-matmul-only   ANE for matmuls, CPU for rest
  --seed <int>        Random seed (default: 42)
  --val-every <int>   Validate every N steps (default: 500)
```

## 5. Checkpoint Compatibility

- **Load**: Reads BLZT v4 checkpoints. Ignores Adam state if present.
- **Save**: Writes BLZT v4 with zeros for Adam m/v. Compatible with train.m and generate.py.
- Checkpoint saved at end of training and every val-every steps.

## 6. Build System

```makefile
# Add to training/Makefile
mezo: train_mezo.m mil_dynamic.h cpu_ops.h config.h io.h
	@command -v xcrun >/dev/null 2>&1 || { echo "Error: Xcode CLT required"; exit 1; }
	xcrun clang -O2 -fobjc-arc -fstack-protector-strong -D_FORTIFY_SOURCE=2 \
		-include models/$(MODEL).h \
		-framework Foundation -framework Accelerate -framework IOSurface \
		-o train_mezo train_mezo.m
```

## 7. Experimental Matrix

### 7.1 From-Scratch (512d/4L, 36.4M params)

| # | Optimizer | Compute | Time | Command |
|---|-----------|---------|------|---------|
| 1 | Backprop | CPU | 120s, 600s | `./train --scratch --cpu-only --time T` |
| 2 | Backprop | ANE | 120s, 600s | `./train --scratch --ane-matmul-only --time T` |
| 3 | MeZO | CPU | 120s, 600s | `./train_mezo --scratch --cpu-only --time T` |
| 4 | MeZO | ANE | 120s, 600s | `./train_mezo --scratch --ane-matmul-only --time T` |

Conditions 1-2 already have baselines from E39-E41.

### 7.2 Fine-Tuning (SmolLM2-135M, 576d/30L)

Requires: `python3 tools/hf_to_ane.py HuggingFaceTB/SmolLM2-135M`

| # | Optimizer | Compute | Time | Command |
|---|-----------|---------|------|---------|
| 5 | Backprop | CPU | 120s, 600s | `./train --cpu-only --time T` (MODEL=smollm2_135m) |
| 6 | Backprop | ANE | 120s, 600s | `./train --ane-matmul-only --time T` |
| 7 | MeZO | CPU | 120s, 600s | `./train_mezo --cpu-only --time T` |
| 8 | MeZO | ANE | 120s, 600s | `./train_mezo --ane-matmul-only --time T` |

### 7.3 Product Demo (SmolLM2-360M, 960d/32L)

MeZO at 1.4GB fits. Backprop at 6.2GB doesn't fit on 8GB devices.

| # | Optimizer | Compute | Time | Command |
|---|-----------|---------|------|---------|
| 9 | MeZO | CPU | 600s | `./train_mezo --cpu-only --time 600` (MODEL=smollm2_360m) |

### 7.4 Metrics Per Condition

- val_loss (start, end, every val-every steps)
- Total step count in time budget
- Average ms/step (with breakdown: forward_ms, perturb_ms, transpose_ms)
- Peak memory usage
- Step 0 loss (sanity: should match across optimizers for same init + seed)

## 8. Hyperparameter Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| ε | 1e-3 | MeZO paper default |
| lr | 1e-5 | MeZO needs much smaller LR than backprop (η_ZO ≈ r/d × η_SGD) |
| n (perturbations per step) | 1 | 2 forward passes, most efficient |
| warmup | 0 | MeZO paper uses no warmup |
| weight decay | 0 | Start without; add if needed |
| LR schedule | Cosine decay to 0.1×lr | Matches train.m convention |
| val-every | 500 | Enough resolution for loss curves |

## 9. Test Plan

Add to `tests/test_training.sh`:

```bash
# Test 9: MeZO compilation
make mezo MODEL=autoresearch
[ -f ./train_mezo ] && pass || fail

# Test 10: MeZO CPU-only forward (7 steps, seed=42)
OUTPUT=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --cpu-only --seed 42 2>&1)
# Step 0 loss should match train's step 0 (same init, same data)
MEZO_LOSS=$(echo "$OUTPUT" | grep "step 0" | grep -oE 'loss_plus=[0-9.]+' | cut -d= -f2)
# Should be close to ln(compact_vocab) ~ 9.73
[ "$MEZO_LOSS" > 9.0 ] && [ "$MEZO_LOSS" < 10.5 ] && pass || fail

# Test 11: MeZO loss decreases (fine-tuning signal exists)
# Run 200 steps, check final < initial
OUTPUT=$(./train_mezo --scratch --data "$DATA" --lr 1e-4 --epsilon 1e-3 \
    --steps 200 --time 60 --cpu-only --seed 42 2>&1)
# Check final_loss < step_0_loss

# Test 12: MeZO ANE mode
OUTPUT=$(./train_mezo --scratch --data "$DATA" --lr 1e-5 --epsilon 1e-3 \
    --steps 7 --time 30 --ane-matmul-only --seed 42 2>&1)
# Should compile forward kernels and produce loss
```

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| From-scratch ZO doesn't converge | HIGH (theory predicts this) | This IS the research result — negative results are publishable |
| ANE re-transposition overhead negates speedup | MEDIUM | Only re-transpose before forward passes (2x not 4x). Profile and optimize. |
| Perturbation at ε=1e-3 causes numerical issues in fp16 | LOW | Perturbation is in fp32 space; only the forward pass is fp16 |
| MeZO fine-tuning quality far below backprop on our data | MEDIUM | Expected to be within 5% on classification; LM may be harder. Document the gap. |
| srand48/drand48 not deterministic across platforms | LOW | We only target macOS/Apple Silicon. Verified deterministic. |
| Box-Muller performance bottleneck (called per-parameter, 4x/step) | MEDIUM | Profile. If slow, switch to vectorized Ziggurat method or vDSP_vgenp. |

## 11. Success Criteria

**Minimum viable result** (sufficient for a paper):
- All 8 conditions run to completion without crashes
- Step 0 loss matches across optimizers for same init (validates forward pass correctness)
- MeZO fine-tuning shows decreasing val_loss (confirms learning signal on ANE)
- Memory measurements confirm 4x reduction vs backprop
- Timing data shows MeZO step time vs backprop step time

**Best case result**:
- MeZO-ANE fine-tuning faster than MeZO-CPU (ANE forward advantage materializes)
- From-scratch ZO shows any learning signal (contradicts theoretical prediction)
- SmolLM2-360M fine-tunes successfully with MeZO (product demo)

## 12. Out of Scope (Deferred)

- MeZO + LoRA (combine ZO with parameter-efficient fine-tuning)
- ElasticZO-INT8 (integer-only ZO training leveraging ANE's INT8 throughput)
- MobiEdit-style knowledge editing
- Multi-perturbation (n > 1) variants
- iOS/iPad deployment
- MeZO-Adam or MeZO-momentum variants
