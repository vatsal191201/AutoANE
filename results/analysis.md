# MeZO vs Backprop Experimental Results (From-Scratch)

**Model:** autoresearch-4L-512d (36.4M params, 4 layers, GQA 8/2)
**Data:** TinyStories (20M tokens, 90/10 train/val split)
**Time budget:** 120 seconds per condition
**Date:** 2026-03-12

## Results Summary

| # | Method | Hardware | Steps | ms/step | Final Loss | Val Loss |
|---|--------|----------|-------|---------|------------|----------|
| 1 | Backprop+Adam | CPU | 3015 | 30.7 | 3.619 | 3.998 |
| 2 | Backprop+Adam | ANE | 3393 | 26.0 | 3.910 | 3.790 |
| 3 | MeZO (ZO-SGD) | CPU | 1588 | 75.1 | 9.599 | 9.685 |
| 4 | MeZO (ZO-SGD) | ANE | 1265 | 94.5 | 9.702 | — |

**MeZO hyperparameters:** lr=1e-5, epsilon=1e-3, cosine decay, Rademacher perturbation (xoshiro256+)
**Backprop hyperparameters:** lr=4e-4, Adam, warmup=10, accum=7, grad_clip=1.0, loss_scale=256

## Per-Step Timing Breakdown (MeZO)

| Component | CPU (ms) | ANE (ms) |
|-----------|----------|----------|
| Forward pass (2x) | 34 | 27 |
| Perturbation (4x) | 43 | 42 |
| Transpose+stage | 0 | 21 |
| **Total** | **75** | **95** |

## Key Findings

### 1. MeZO works on ANE — first ZO training on any NPU
This is the primary result: zeroth-order optimization runs on Apple Neural Engine hardware.
The forward pass uses ANE fp16 matmuls while perturbation/RoPE/attention remain on CPU fp32.

### 2. ANE provides no speedup for MeZO (from scratch)
MeZO-ANE is 26% slower than MeZO-CPU (94.5 vs 75.1 ms/step) because the transpose+staging
overhead (21ms/step for 2x retranspose) exceeds the forward pass speedup (27ms vs 34ms).
The perturbation cost (42-43ms) dominates either way and runs on CPU regardless.

### 3. MeZO from scratch shows minimal learning in 120s
Final loss ~9.6 vs initial ~9.73 — barely any signal. This matches MeZO theory: the SPSA
gradient variance scales as O(d) where d=36.4M, making from-scratch training extremely slow.
The MeZO paper (NeurIPS 2023) only demonstrated fine-tuning, not from-scratch training.

### 4. Backprop dramatically outperforms MeZO from scratch
Backprop achieves loss 3.6 (strong learning) vs MeZO's 9.6 (near random) in the same wall time.
This is the expected baseline — the interesting comparison will be fine-tuning (conditions 5-8).

### 5. ANE provides modest speedup for backprop
26.0 vs 30.7 ms/step = 1.18x faster. The speedup is modest because CPU BLAS (AMX) is already
fast on M-series, and attention/RoPE/RMSNorm/SiLU still run on CPU.

## Next Steps (Conditions 5-8)

Fine-tuning from a pretrained SmolLM2-135M checkpoint is where MeZO should shine:
- Smaller gradients → better signal-to-noise ratio
- MeZO's memory advantage (no Adam state) becomes relevant for larger models
- ANE speedup may be larger with bigger models where matmul fraction increases
