# AutoANE Experiment Log

## Experiment 1: CPU fp32 Attention Backward

**Branch**: `experiments/cpu-sdpa-backward` (merged to main)
**Status**: COMPLETE

**Hypothesis**: ANE fp16 SDPA backward produces zero attention gradients due to fp16 underflow. Moving attention backward to CPU fp32 should fix this.

**Root Cause**: Three cascading fp16 underflows:
1. `wotBwd`: dx2_scaled (~0.0005) * Wo (~0.01) → ~0.000005, below fp16 precision
2. `sdpaBwd`: softmax probs (~1/256) * tiny da → underflows to zero
3. `qBwd`/`kvBwd`: zero dq/dk/dv → zero dx, blocking gradient flow through residual stream

**Fix**: `--cpu-attn-bwd` flag moves wotBwd, SDPA backward, qBwd, kvBwd to CPU fp32.

**Results**:
- Before: `L0 sdpa_bwd: |dq|=0.000000 |dk|=0.000000 |dv|=0.000000`
- After: `L0 sdpa_bwd: |dq|=0.000017 |dk|=0.000025 |dv|=0.000108`
- dy magnitudes 100x larger with full CPU backward (0.3 vs 0.003)
- SmolLM2-360M ANE baseline: 200 steps/5min, loss = 5.682

---

## Experiment 2: Architecture Search (Manual)

**Branch**: `experiments/autoresearch-run-1`
**Status**: COMPLETE

**Hypothesis**: In a fixed time budget, shallower/wider models get more optimizer steps and may achieve better loss than deep/narrow models.

**Setup**: All runs use CPU_ATTN_BWD=True, SmolLM2 tokenizer (VOCAB=49152), 2-minute time budget.

| Config | Params | Steps | Loss | ms/step | Tokens/s |
|--------|--------|-------|------|---------|----------|
| 24L, dim=384, 6Q/3KV | 57.8M | 430 | 6.125 | 204 | 1,255 |
| 16L, dim=512, 8Q/4KV | 72.4M | 443 | 6.048 | 188 | 1,362 |
| 8L, dim=768, 12Q/4KV | 88.1M | 770 | 5.617 | 130 | 1,969 |
| **4L, dim=1024, 16Q/4KV** | **95.4M** | **900** | **5.357** | **111** | **2,306** |

**Comparison to SmolLM2-360M baseline** (5-minute run, ANE fp16 only):
| Config | Params | Steps | Loss | Time |
|--------|--------|-------|------|------|
| SmolLM2-360M (32L, dim=960) | 362M | 200 | 5.682 | 5 min |
| **4L, dim=1024 (autoresearch)** | **95.4M** | **900** | **5.357** | **2 min** |

**Key Finding**: The 4-layer model beats SmolLM2-360M by 0.32 loss in 60% less time with 3.8x fewer parameters. This validates the autoresearch insight: **more optimizer steps beats more parameters in a fixed time window**.

**Why shallow wins on ANE**:
- Fewer layers = fewer sequential ANE kernel launches = less overhead per step
- Wider layers utilize ANE matmul parallelism better (larger matrices)
- More steps = more data seen = better generalization in early training
- DeepNet scaling (1/sqrt(2*N)) attenuates residual connections more with depth

---

## Planned Experiments

### Experiment 3: LoRA Fine-tuning
**Hypothesis**: Freeze base model weights, train only small adapter matrices (rank 4-16). Small adapters keep gradients in fp16 range, potentially enabling fine-tuning of pretrained models.

### Experiment 4: Optimal Architecture at 5-minute Budget
**Hypothesis**: The optimal depth/width tradeoff may differ at longer training budgets. Search at 5 minutes to find the best config for the autoresearch default.

### Experiment 5: Higher Learning Rate for Shallow Models
**Hypothesis**: With fewer layers, we can use higher LR (5e-4 or 1e-3) since DeepNet scaling is less aggressive and gradients are larger.

### Experiment 6: Longer Training (1 hour)
**Hypothesis**: At longer budgets, depth matters more (deeper models keep learning while shallow ones plateau). The optimal depth should increase with budget.
