# Recursive Transformers on ANE: Playing to the Hardware's Strengths

**Date**: 2026-03-17
**Status**: CORE RESEARCH DIRECTION
**Key insight**: Recursive (looped) transformers are a natural fit for ANE because
the hardware excels at running the SAME compiled kernel repeatedly in a tight loop.
Combined with per-iteration LoRA, this creates the ideal training setup for forward-only hardware.

---

## 1. Why Recursive Models + ANE Is the Right Combination

### ANE's actual strengths (from 44 experiments):
- Forward pass matmuls: 2.5x faster than CPU
- Conv1x1 with BLOBFILE: compile ONCE, run FOREVER — zero per-step overhead
- Dedicated silicon: runs independently, CPU/GPU free for other work
- Tight dispatch loop: ~160us per kernel invocation

### What recursive transformers do:
- ONE shared block of weights, iterated K times
- Quality comes from DEPTH OF ITERATION, not number of unique parameters
- More iterations = more reasoning = better output (like o1 "thinking")
- Weight sharing = compile once, loop K times on the SAME compiled kernel

### The match:
ANE compiles one conv-fused kernel (the shared block) and runs it in a tight loop.
Each iteration costs ~4ms (single layer forward on SmolLM2-360M scale).
32 iterations = 128ms. 64 iterations = 256ms. 128 iterations = 512ms.

The user controls quality by choosing iteration count. Easy inputs: 16 iterations.
Hard reasoning: 64+ iterations. ANE handles the compute while CPU/GPU are free.

## 2. Architecture: Relaxed Recursive Transformer

Based on Google DeepMind's Relaxed Recursive Transformer (ICLR 2025, arXiv:2410.20672):

```
Standard 32-layer model:
  Layer 0 (unique weights) → Layer 1 (unique weights) → ... → Layer 31 (unique weights)
  Total unique params: 361.8M
  LoRA params: 1,146,880 (35,840 per layer × 32)

Recursive model (1 shared block, 32 iterations):
  Block (shared weights) → same Block → ... → same Block (32 times)
  Total unique params: ~11.3M
  LoRA params: 35,840 (single block)

Relaxed Recursive (shared block + per-iteration LoRA):
  Block + LoRA_0 → Block + LoRA_1 → ... → Block + LoRA_31
  Total unique params: ~11.3M + 32 × tiny LoRA
  Per-iteration LoRA: rank-4 is enough (shared block does heavy lifting)
  LoRA params per iteration: ~8,000 (rank-4 on Q,K,V,O)
  Total LoRA: ~256,000
```

### Conversion from pretrained model:
The Relaxed Recursive Transformer paper (ICLR 2025) shows three initialization strategies:
1. **Stepwise**: Select every K-th layer from the pretrained model
2. **Average**: Average weight matrices across tied layers
3. **Lower**: Use the first K layers directly

For SmolLM2-360M: average the 32 layers into 1 shared block. Initialize per-iteration
LoRA from the SVD of the per-layer weight differences (captures what makes each layer unique).

## 3. Training on ANE

### Option A: MeZO on recursive LoRA (forward-only, ANE-native)

With only 256K LoRA params (vs 1.15M for standard):
- MeZO gradient quality: 1/sqrt(256,000) = 0.20% per step
- vs standard: 1/sqrt(1,146,880) = 0.093% per step
- **2.1x better gradient quality per step**
- **Predicted ceiling drop**: the MeZO quality ceiling should be significantly lower
  because the same model quality is achievable with fewer parameters

Step time: 2 × 32 iterations × 4ms = 256ms (comparable to current MeZO 262ms).
But with 2.1x better gradient quality → potentially 2.1x lower ceiling.

### Option B: P16 backprop-lora on recursive model

With only 256K LoRA params:
- Backward pass is 32x smaller (only 1 shared block's worth of params)
- Estimated backward: 353ms / 32 = ~11ms per block (single layer backward)
- But need to backprop through 32 iterations of the shared block → ~352ms
  (same as 32-layer, because the chain rule still traverses 32 iterations)
- However: the LoRA weight gradients ACCUMULATE across iterations
  (shared weights → gradients from all 32 iterations contribute to same params)

### Option C: MeZO + adaptive iteration count (THE NOVEL PART)

Standard MeZO: fixed 32 iterations per forward pass.

**Adaptive MeZO**: vary iteration count based on input difficulty.
- Easy inputs (high model confidence): 8 iterations → 32ms per forward pass
- Normal inputs: 32 iterations → 128ms
- Hard inputs (low confidence, reasoning needed): 64 iterations → 256ms

The MeZO gradient from harder inputs (more iterations) is MORE INFORMATIVE
because the model had more computation to form its prediction. The gradient
captures richer structure.

**This is test-time compute scaling for training, not just inference.**

## 4. Diffusion Language Model Variant

Discrete diffusion LMs (MDLM, Block Diffusion) generate text via iterative denoising.
Each denoising step = one forward pass through the SAME model.

A diffusion LM on ANE:
- Compile the denoiser ONCE as conv-fused BLOBFILE
- Run T denoising steps (each is an ANE forward pass)
- T=8: fast but lower quality. T=32: slower but better.
- Training: the denoising objective is LOCAL (predict clean from noisy)
  → can be trained with forward-only methods

Block Diffusion (ICLR 2025 Oral): interpolates between autoregressive and diffusion.
Achieves competitive perplexity with 8-16 denoising steps (128x speedup).

**ANE + Block Diffusion**: compile the shared denoiser once, run 8-16 ANE dispatches
per generated block. Each dispatch is ~4ms → total generation: 32-64ms per block.

## 5. Concrete Implementation Plan

### Phase 1: Convert SmolLM2-360M to Recursive (2-3 days)
1. Average all 32 layers into 1 shared block (weighted average)
2. Compute per-layer SVD residuals for LoRA initialization
3. Create rank-4 per-iteration LoRA adapters (32 × 8,000 = 256K params)
4. Validate: forward pass produces reasonable output (not garbage)

### Phase 2: Compile for ANE (1 day)
1. Compile shared block as single conv-fused BLOBFILE kernel
2. Run the compiled kernel 32 times in a loop
3. Apply per-iteration LoRA corrections on CPU between iterations
4. Benchmark: should be ~128ms for 32 iterations (vs 131ms for standard)

### Phase 3: MeZO training on recursive LoRA (2 days)
1. Perturb 256K LoRA params
2. Forward: 32 iterations of shared block + LoRA
3. ZO gradient from global loss
4. Update LoRA adapters
5. Compare val_loss ceiling vs standard MeZO (2.0524)

### Phase 4: Adaptive iteration count (2 days)
1. Add confidence-based early stopping
2. Vary iterations: 8, 16, 32, 64
3. Measure quality vs compute trade-off
4. Training: use curriculum (start with 8 iterations, increase over time)

## 6. Expected Results

| Setup | LoRA params | MeZO quality | Predicted ceiling | ANE step time |
|-------|------------|-------------|------------------|---------------|
| Standard 32L | 1,146,880 | 0.093%/step | val_loss 2.052 | 262ms |
| Recursive shared | 35,840 | 0.53%/step | < 2.03 (est.) | ~256ms |
| Relaxed recursive | 256,000 | 0.20%/step | < 2.04 (est.) | ~260ms |
| Adaptive (64 iter) | 256,000 | 0.20%/step | < 2.03 (est.) | ~520ms |

## 7. Why This Is Novel

1. **No one has combined recursive transformers with ZO training.**
   The Relaxed Recursive paper uses standard backprop for uptraining.

2. **No one has compiled a recursive transformer for ANE.**
   All ANE training uses standard (unique layer) architectures.

3. **Adaptive iteration count for ZO training is new.**
   MeZO always uses a fixed architecture. Varying depth per-input is unexplored.

4. **Diffusion LM on ANE is new.**
   No one has run discrete diffusion language models on NPU hardware.

## 8. References

- [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) — Google DeepMind, ICLR 2025
- [Scaling Test-Time Compute with Recurrent Depth](https://arxiv.org/abs/2502.05171) — 2025
- [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) — Adaptive per-token depth, 2025
- [SpiralFormer](https://arxiv.org/abs/2602.11698) — Looped multi-resolution recursion, 2026
- [Block Diffusion](https://arxiv.org/abs/2503.09573) — ICLR 2025 Oral
- [MDLM](https://arxiv.org/abs/2406.07524) — Masked Diffusion LM, NeurIPS 2024
- [Universal Transformer](https://arxiv.org/abs/1807.03819) — Original recurrent transformer, 2018
- [Improving Recursive Transformers with MoL](https://arxiv.org/abs/2512.12880) — Dec 2025
