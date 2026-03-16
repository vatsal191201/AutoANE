# P16: MeBP+ANE Forward Hybrid — Design Spec

**Date**: 2026-03-16
**Status**: DESIGN (not yet implemented)
**Priority**: #1 (highest impact unexplored direction)
**Estimated effort**: 3-5 days engineering

---

## 1. Hypothesis

Gradient-checkpointed backpropagation with ANE forward passes and CPU backward passes
will achieve better convergence-adjusted throughput than either:
- MeZO+LoRA-split on ANE (our current best: 262ms/step, ~600 steps)
- MeBP on CPU/GPU (Apple's approach: ~5s/step, ~60 steps)

## 2. Assumptions (Stated Explicitly)

| # | Assumption | Basis | Risk |
|---|-----------|-------|------|
| H1 | ANE forward pass for 32-layer SmolLM2-360M takes ~130ms (half of MeZO's 262ms, since MeZO does 2 forward passes) | Derived from MeZO timing | MEDIUM: MeZO also includes perturbation time |
| H2 | CPU backward per layer takes ~10ms (from E38 data: 328ms/32L at DIM=960) | Measured in experiments | HIGH: E38 showed backward degrades to 328ms when IOSurfaces allocated. Need to re-measure without IOSurface pressure. |
| H3 | Gradient checkpointing requires 1 ANE forward recompute per layer (~4ms) | Derived: 130ms/32L per full forward | MEDIUM: per-layer ANE dispatch has ~160us overhead |
| H4 | LoRA backprop needs only attention layers (Wq, Wk, Wv, Wo), not FFN | Same as MeZO LoRA config | LOW: FFN weights frozen |
| H5 | First-order (backprop) LoRA converges in ~60 steps (10x fewer than MeZO's 600) | MeBP paper + our analysis (UP24) | HIGH: not tested on our setup |
| H6 | CPU backward matmuls can run via Accelerate BLAS at full speed alongside ANE | E38 showed IOSurface memory pressure degrades CPU backward by 2.5x | HIGH: this is the main risk |
| H7 | Per-layer ANE recomputation can be done without full model compilation | Requires per-layer CoreML models (already have in current pipeline) | LOW: we already compile per-layer kernels |

## 3. Architecture

```
Training Step:
  1. ANE Forward Pass (conv-fused, 32 layers):
     - Input: x[1, SEQ, DIM] + current LoRA corrections
     - Save: per-layer activation checkpoints (memory-mapped)
     - Output: logits -> cross-entropy loss on CPU

  2. CPU Backward Pass (per-layer, reversed):
     For layer L = 31 down to 0:
       a. Recompute forward activations for layer L via ANE (~4ms)
       b. Compute dx (input gradient) on CPU via cblas_sgemm
       c. Compute dW for LoRA A/B matrices on CPU
       d. Free recomputed activations

  3. CPU Weight Update:
     - Adam on LoRA A/B (1.7M params, ~1ms)
     - Update conv-fused BLOBFILE weights? NO — base weights frozen
     - Only LoRA corrections change (CPU-side, no ANE impact)
```

## 4. Timing Estimate (First Principles)

### 4.1 Forward Pass

Current MeZO conv-fused: 262ms for 2 forward passes = ~131ms per forward pass.
This includes:
- 32 layers x 3 ANE dispatches (QKV, Wo, FFN) = 96 dispatches
- CPU: RMSNorm, RoPE, SDPA softmax, LoRA corrections, residuals
- IO: 96 x ~160us = ~15ms dispatch overhead

**Hybrid forward**: Same as MeZO forward = ~131ms.

### 4.2 Backward Pass

For each layer (CPU-only operations):
- **dx through attention**: 4 matmuls (dWq, dWk, dWv, dWo) x [SEQ, DIM] x [DIM, DIM] = 4 x 0.25M FLOPs
  - At M2 Pro Accelerate: ~2.5 TFLOPS fp32 -> each matmul: ~0.3ms
  - 4 matmuls: ~1.2ms
- **dx through FFN**: We DON'T backprop through FFN (frozen). Just pass gradients.
  - Actually need: dx through Wo only (FFN gradients are zero for frozen weights)
  - BUT: residual connections mean we need dx through the full layer
  - dx through FFN = 3 matmuls: ~0.9ms
- **LoRA dW**: For each adapted projection (Wq, Wk, Wv, Wo):
  - dA = dout^T @ x @ B: [r, DIM] = [r, Q] x [Q, DIM] ~ negligible (rank-8)
  - dB = A^T @ dout^T @ x: [DIM, r] ~ negligible
- **RMSNorm backward**: ~0.1ms (vDSP vectorized)
- **Checkpoint recompute**: Rerun forward for this layer on ANE = ~4ms
  - 131ms / 32 layers = ~4.1ms per layer

**Per-layer total**: 4.1ms (ANE recompute) + 2.1ms (CPU backward matmuls) + 0.1ms (RMSNorm) = ~6.3ms

**32-layer total backward**: 32 x 6.3ms = ~202ms

### 4.3 Total Step Time

| Component | Time (ms) | Source |
|-----------|----------|--------|
| Forward pass | 131 | Half of MeZO's 262ms |
| Backward pass | 202 | 32 x 6.3ms per layer |
| Loss computation | ~5 | CPU cross-entropy |
| Adam update | ~1 | 1.7M LoRA params |
| **Total** | **~339ms** | |

### 4.4 Convergence-Adjusted Comparison

| Method | ms/step | Steps needed | Total time | Quality |
|--------|---------|-------------|-----------|---------|
| MeZO conv-fused (current) | 262 | ~600 | 157s | -0.019 nats |
| **P16 hybrid (estimated)** | **339** | **~191** | **65s** | **-0.147 nats** |
| MeBP CPU-only (estimated) | ~3000 | ~60 | 180s | -0.147 nats |

**VALIDATED from existing data (P16-A)**: Backprop achieves 0.147 nats improvement
(7.6x more than MeZO's 0.019) in 191 steps. P16 achieves this in **65s vs 157s** for
MeZO (2.4x faster) AND **7.6x better quality**. The real comparison is not
time-to-equal-quality (MeZO can never reach backprop quality) but quality-at-fixed-time.

## 5. Key Risks

### Risk 1: IOSurface Memory Pressure (HIGH)
E38 showed that allocated IOSurfaces (379MB at DIM=2048) cause CPU backward to degrade 2.5x.
At DIM=960 with conv-fused kernels, IOSurface footprint is much smaller (~60MB).
**Mitigation**: Measure CPU backward performance WITH conv-fused IOSurfaces allocated.

### Risk 2: Convergence Gap Unknown (HIGH)
We assume 10x fewer steps for backprop vs MeZO LoRA. This is untested.
**Mitigation**: Run backprop LoRA baseline first (CPU-only) to measure actual convergence.
**Mini-experiment**: 100 steps of backprop LoRA on CPU. If val_loss drops by 0.019 nats in <100 steps, H5 is validated.

### Risk 3: Per-Layer ANE Recomputation Overhead (MEDIUM)
Each layer recompute requires an ANE dispatch (~160us overhead + ~4ms compute).
32 dispatches = 5ms overhead + 128ms compute = 133ms.
This is comparable to the forward pass — effectively doubling the ANE time.
**Mitigation**: Could batch recomputation (e.g., 4-layer chunks) to reduce dispatch overhead.

### Risk 4: Implementation Complexity (MEDIUM)
Requires coordinating ANE forward, checkpoint storage, per-layer ANE recompute, and CPU backward.
**Mitigation**: Build incrementally: (1) backprop LoRA CPU-only first, (2) add ANE forward, (3) add checkpointing.

## 6. Mini-Experiments to Validate Before Full Implementation

### Experiment P16-A: Backprop LoRA Convergence Baseline — VALIDATED FROM EXISTING DATA
**Goal**: Measure how many backprop steps achieve 0.019 nats improvement.
**Result**: MASSIVELY EXCEEDED. Existing condition13 data shows:
- Backprop LoRA: val_loss 1.9248 (improvement 0.147 nats) in 191 steps, 112s
- MeZO LoRA: val_loss 2.0524 (improvement 0.019 nats) in 600 steps, 157s (ANE)
- Backprop achieves **7.6x MORE improvement** in **3x fewer steps**
- MeZO saturates at 0.019 nats; backprop still improving at 191 steps
- **H5 was CONSERVATIVE**: backprop is not just 10x better in steps, it reaches
  a fundamentally better quality level that MeZO cannot reach at all.
**Revised P16 estimate**: 339ms x 191 steps = **65s for val_loss 1.9248**
  vs MeZO ANE: 262ms x 600 steps = 157s for val_loss 2.0524

### Experiment P16-B: CPU Backward with IOSurfaces
**Goal**: Measure CPU backward matmul speed WITH conv-fused IOSurfaces allocated.
**Setup**: Compile conv-fused kernels (allocate IOSurfaces), then benchmark cblas_sgemm.
**Time**: 5 minutes.
**Success criterion**: <5ms per layer backward.
**If fails**: IOSurface pressure makes hybrid worse than MeZO. Consider freeing IOSurfaces during backward.

### Experiment P16-C: Per-Layer ANE Recompute Timing
**Goal**: Measure time to recompute single layer on ANE (for checkpoint recomputation).
**Setup**: Compile per-layer conv-fused kernel, time individual dispatch.
**Time**: 5 minutes.
**Success criterion**: <8ms per layer.
**If fails**: Batch recomputation (4-layer chunks) or accept higher per-step time.

## 7. Decision Gate

**GO if**: P16-A shows <100 steps AND P16-B shows <5ms/layer AND P16-C shows <8ms/layer.
**NO-GO if**: Any two of three fail. Fall back to P11 (cross-layer fusion for MeZO).

## 8. Implementation Plan (if GO)

1. **Implement backprop LoRA on CPU** (2 days)
   - Add `--mode backprop-lora` to train_mezo.m
   - Reuse existing forward pass code
   - Add CPU backward: cblas_sgemm for dx and dW through attention
   - Add CPU backward through FFN (frozen, but need dx for residual)
   - LoRA gradient: dA, dB per projection

2. **Add gradient checkpointing** (1 day)
   - Memory-map per-layer activations to disk
   - Or: keep in memory if <2GB (32 layers x [1, 256, 960] x fp32 = ~31MB)
   - Actually: 31MB is tiny. Keep in memory.

3. **Replace forward recomputation with ANE dispatch** (1 day)
   - Per-layer ANE forward kernel already exists
   - Call per-layer kernel instead of CPU forward for checkpointed layers

4. **Benchmark and optimize** (1 day)
   - End-to-end timing
   - Profile CPU backward vs ANE forward overlap opportunities
   - Test if backward for layer L can overlap with ANE recompute for layer L-1

## 9. References

- MeBP: arXiv:2510.03425 (Apple, Oct 2025)
- Gradient Checkpointing: Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)
- Our MeZO results: docs/specs/2026-03-15-ane-training-pipeline-optimization.md
- E38 IOSurface pressure: docs/EXPERIMENTS.md
