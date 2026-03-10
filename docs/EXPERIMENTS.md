# AutoANE Experiment Log

## Experiment 1: CPU fp32 Attention Backward

**Branch**: `experiments/cpu-sdpa-backward`

**Hypothesis**: ANE fp16 SDPA backward produces zero attention gradients (dq/dk/dv = 0) due to fp16 underflow. Moving the attention backward path to CPU fp32 should fix this and improve training quality.

**Root Cause Analysis**:
1. `wotBwd` (ANE fp16): dx2_scaled (~0.0005) * Wo weights (~0.01) → product ~0.000005, below fp16 precision after matmul accumulation over DIM=768
2. `sdpaBwd` (ANE fp16): softmax probs (~1/256 = 0.004) * small da → underflows to zero in fp16
3. `qBwd`/`kvBwd` (ANE fp16): if dq/dk/dv are zero, these also produce zero, blocking gradient flow through the residual stream

**Change**: `--cpu-attn-bwd` flag moves 4 operations to CPU fp32:
- `wotBwd`: dx2 @ Wo^T → da (using cblas_sgemm)
- `sdpaBwd`: Full SDPA backward with recomputed softmax (custom implementation)
- `qBwd`: dq @ Wq^T → dx_q (using cblas_sgemm)
- `kvBwd`: dk @ Wk^T + dv @ Wv^T → dx_kv (using cblas_sgemm)

**Results (Stories110M, 12 layers, 109M params)**:

Quick test (100 steps / ~60s):
| Variant | Final Loss | L0 |dq| | L0 |dk| | L0 |dv| |
|---------|-----------|---------|---------|---------|
| ANE fp16 (baseline) | 7.258 | 0.000000 | 0.000000 | 0.000000 |
| CPU fp32 partial (wotBwd + SDPA only) | 7.247 | 0.000017 | 0.000025 | 0.000108 |
| CPU fp32 full (+ qBwd + kvBwd) | 7.264 | 0.000017 | 0.000025 | 0.000108 |

5-minute A/B test: [PENDING - running]

**Observations**:
- CPU fp32 correctly produces nonzero attention gradients
- dy magnitudes are 100x larger with full CPU backward (0.3 vs 0.003), indicating gradient flow through attention path now works
- Loss improvement is modest in 100 steps (~10 effective updates)
- Full CPU backward may need learning rate tuning since gradient magnitudes change significantly

---

## Planned Experiments

### Experiment 2: LoRA Fine-tuning
**Hypothesis**: Freeze base model weights, train only small adapter matrices (rank 4-16). LoRA gradients should be larger (fewer parameters concentrating gradient signal), avoiding the fp16 underflow that kills full fine-tuning.

### Experiment 3: Autoresearch Architecture Search
**Hypothesis**: With CPU attention backward enabled, the autoresearch agent should find better architectures that leverage attention learning. May favor deeper/narrower models.

### Experiment 4: Longer Training (overnight)
**Hypothesis**: The attention gradient effect compounds over longer training. Run 1-hour training to see if the gap widens.

### Experiment 5: SimpleStories Dataset
**Hypothesis**: SimpleStories (more diverse syntax) should produce better small models than TinyStories.
