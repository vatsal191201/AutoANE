# P16 Implementation Plan: ANE Forward + CPU Backward Hybrid

**Date**: 2026-03-16
**Status**: READY TO IMPLEMENT
**Basis**: P16 design spec + code analysis of train.m and train_mezo.m

---

## Key Insight: We Already Have All the Pieces

After analyzing both source files:

| Component | Where | Lines | Status |
|-----------|-------|-------|--------|
| Conv-fused ANE forward pass | train_mezo.m:1335-1506 | DO_FORWARD_PASS macro | COMPLETE |
| CPU backward pass (full) | train.m:927-1100+ | use_cpu_bwd path | COMPLETE |
| LoRA forward corrections | train_mezo.m:1346-1367 | lora_addmm in conv_fused path | COMPLETE |
| Activation checkpointing | N/A | Not needed (31MB at DIM=960) | SKIP |
| Adam optimizer | cpu_ops.h | vDSP-vectorized | COMPLETE |
| Cross-entropy + dlogits | cpu_ops.h | Existing | COMPLETE |

**The implementation is primarily a MERGE of existing code paths**, not new algorithm development.

## What Needs to Be Built

### 1. LoRA Backward Pass (NEW — ~80 lines)

For each layer with LoRA adapters (Wq, Wk, Wv, Wo):
- **dA** = dx_proj^T @ (x @ B^T): gradient of LoRA A matrix
- **dB** = (A^T @ dx_proj)^T @ x: gradient of LoRA B matrix
- Both are small matmuls: rank-8 inner dimension

```c
// For projection P with LoRA: y = base(x) + B @ A @ x
// Backward: dx += A^T @ B^T @ dy (chain rule through LoRA)
//           dA = B^T @ dy @ x^T (gradient of A)
//           dB = dy @ x^T @ A^T (gradient of B)
// But simpler: y = base(x) + lora(x) where lora(x) = B @ (A @ x)
// dx_lora = A^T @ (B^T @ dy)  [rank-8 matmuls, negligible time]
// dA = (B^T @ dy) @ x^T [r x SEQ] @ [SEQ x DIM] = [r x DIM]
// dB = dy @ (A @ x)^T [OUT x SEQ] @ [SEQ x r] = [OUT x r]
```

### 2. Activation Storage During Forward (NEW — ~30 lines)

The existing backprop code saves activations to `LayerActs` structs.
The MeZO forward pass does NOT save activations (doesn't need them).
Need to add activation saves to the conv_fused forward path.

Per layer, need to save:
- `xnorm_buf` (pre-attention input): [DIM, SEQ] = 960*256*4 = 0.94MB
- `Q, K, V` (for SDPA backward): [Q_DIM+2*KV_DIM, SEQ] = 1.2MB
- `attn_out` (for Wo backward): [Q_DIM, SEQ] = 0.94MB
- `x2norm` (pre-FFN input): [DIM, SEQ] = 0.94MB
- `h1, h3` (for SiLU backward): [2*HIDDEN, SEQ] = 5.0MB
- `silu_out` (for W2 backward): [HIDDEN, SEQ] = 2.5MB

Total per layer: ~11.5MB. 32 layers: ~370MB.

**ASSUMPTION A_P16_1**: 370MB activation storage fits in M2 Pro 16GB alongside model weights (~1.4GB) and other buffers. Total: ~2GB. VERIFIED: well within 16GB.

### 3. Modified Training Loop (MERGE — ~150 lines)

```
for each step:
    1. Sample batch (existing tokenizer code)
    2. FORWARD (conv_fused ANE, from train_mezo.m):
       - Embed lookup
       - For L=0..31: ANE conv_fused forward + LoRA corrections + save activations
       - Final RMSNorm + classifier + cross-entropy loss
    3. BACKWARD (CPU, adapted from train.m):
       - dlogits from cross-entropy
       - Classifier backward (cblas_sgemm)
       - Final RMSNorm backward
       - For L=31..0: full layer backward using saved activations
         - FFN backward: dx through W2, SiLU derivative, dx through W1/W3
         - RMSNorm backward
         - Attention backward: dx through Wo, SDPA backward, dx through Wq/Wk/Wv
         - LoRA backward: compute dA, dB for each adapted projection
         - dW for base weights: SKIP (frozen in LoRA mode)
    4. OPTIMIZER:
       - Adam on LoRA A/B matrices only (1.7M params)
       - Adam on RMS norms (62.4K params)
```

### 4. CLI Flag (trivial — ~5 lines)

Add `--mode backprop-lora` to train_mezo.m's CLI parser.

## Files Modified

| File | Changes | New Lines |
|------|---------|-----------|
| train_mezo.m | Add backprop-lora mode: activation storage, backward pass, LoRA gradients | ~260 |
| cpu_ops.h | Add `lora_backward()` function | ~40 |
| config.h | Add LayerActs struct for activation storage | ~20 |

Total: ~320 new lines.

## What We Do NOT Need

1. **Gradient checkpointing**: 370MB activation storage fits in memory. No disk I/O needed.
2. **ANE backward kernels**: All backward runs on CPU (use_cpu_bwd path from train.m).
3. **Loss scaling**: fp32 backward doesn't underflow (no fp16 gradients).
4. **Base weight gradients**: Frozen in LoRA mode. Only LoRA A/B and RMS norms trained.

## Verification Plan

### V1: Numerical Correctness (before benchmarking)
- Run 1 forward + backward step on CPU-only (no ANE)
- Compare dA, dB gradients against numerical finite differences
- Tolerance: max_abs_diff < 1e-4 (fp32 accumulation)

### V2: ANE-CPU Consistency
- Run same input through conv_fused forward (ANE) vs cpu_only forward
- Compare activations at each layer
- Tolerance: cos_sim > 0.999 (fp16 rounding acceptable)

### V3: Training Convergence
- 100 steps of backprop-lora (conv_fused ANE forward + CPU backward)
- Compare val_loss trajectory against condition13 (CPU-only backprop)
- Should match within 0.01 nats at step 100

### V4: Timing Benchmark
- 50-step average ms/step
- Compare against predicted 400ms/step
- Profile breakdown: ANE forward / CPU backward / overhead

## Timeline

| Day | Task | Hours |
|-----|------|-------|
| 1 | Add activation storage to conv_fused forward in train_mezo.m | 3 |
| 1 | Port CPU backward from train.m (dx chain) | 4 |
| 2 | Implement LoRA backward (dA, dB gradients) | 3 |
| 2 | Verification V1 (numerical gradients) | 2 |
| 3 | Integration: full training loop with Adam | 3 |
| 3 | Verification V2 + V3 (ANE consistency + convergence) | 3 |
| 4 | V4 benchmark + optimization | 4 |
| 4 | Documentation + commit | 2 |

## Assumptions

| # | Assumption | Basis | Risk |
|---|-----------|-------|------|
| A_P16_1 | 370MB activation storage fits in 16GB | 370MB + 1.4GB model + 0.5GB misc = 2.3GB << 16GB | LOW |
| A_P16_2 | CPU backward at 7.8ms/layer (249ms total) | condition13 timing data, CPU-only no IOSurface pressure | MEDIUM |
| A_P16_3 | Conv_fused ANE forward is 131ms | Half of MeZO's 262ms (2 forward passes per MeZO step) | LOW |
| A_P16_4 | LoRA backward adds <5ms total | Rank-8 matmuls: 4 projections x 32 layers x 2 grads = 256 tiny matmuls | LOW |
| A_P16_5 | Val_loss trajectory matches CPU-only backprop within 0.01 | fp16 ANE forward may introduce small precision differences | MEDIUM |
| A_P16_6 | No new ANE compilation needed | Reuse existing conv_fused kernels from MeZO mode | LOW |
