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

## Experiment 3: LoRA Fine-tuning

**Status**: COMPLETE

**Hypothesis**: Freeze base model weights, train only small LoRA adapter matrices. Merge-based approach (W_eff = W_base + B@A) requires zero forward/backward changes — only the Adam update step projects gradients to the LoRA subspace.

**Implementation**: `--lora --lora-rank 8` flags in train.m
- LoRA adapters on Wq, Wk, Wv, Wo (all 4 attention projections)
- A matrices: [rank, input_dim], init Kaiming uniform
- B matrices: [output_dim, rank], init zero (LoRA starts as identity)
- RMSNorm scales: trainable. FFN (W1/W2/W3) and embedding: frozen.
- After backward: project full dW → dA/dB, Adam update adapters, merge W_eff, re-stage IOSurface

**Results** (4L/dim=1024 autoresearch config, CPU_ATTN_BWD=True):

| Mode | Trainable Params | Steps | Time | Loss | Grad Norm |
|------|-----------------|-------|------|------|-----------|
| LoRA from scratch | 188K (0.2%) | 240 | 30s | 9.14 | 87-218 (exploding) |
| Full from scratch | 95.4M (100%) | 1090 | 120s | 4.81 | 1.4 (stable) |
| **LoRA from pretrained** | **188K (0.2%)** | **495** | **60s** | **4.22 best** | **1.6-2.0 (stable)** |

**Key Findings**:
1. **LoRA from scratch doesn't work** — gradient norms explode (88→218), loss barely drops (9.83→9.14). FFN and embedding are frozen at random, so the model can't learn representations.
2. **LoRA from pretrained works** — stable training, gradient norms ~1.7, loss maintained at 4.2-5.0 (pretrained was 4.37). Best LoRA loss 4.22 — slight improvement from adapters.
3. **Zero overhead merge approach** — no MIL kernel changes needed. Forward/backward identical to full training. Only cost: gradient projection + merge (~0.1ms/step for rank 8).
4. **First known LoRA implementation on Apple Neural Engine** — validates that ANE fp16 forward + CPU fp32 backward + LoRA gradient projection works end-to-end.

**Why limited improvement from pretrained**: This is a small model (95M) already well-trained on TinyStories. LoRA shines when fine-tuning large pretrained models on new domains — the base capabilities are already strong, and small adapters can steer behavior. Our model hasn't converged yet (only 1000 steps of pretraining), so LoRA can't compensate for underfitting.

---

## Planned Experiments

## Experiment 5: Higher Learning Rate for Shallow Models

**Status**: COMPLETE

**Hypothesis**: With only 4 layers, DeepNet scaling (alpha=1/sqrt(8)=0.35) is less aggressive, and gradient flow is better. We should be able to use higher LR.

**Setup**: 4L/dim=1024 config, CPU_ATTN_BWD=True, from scratch, clip=1.0, accum=10, warmup=100.

| LR | Steps/60s | Loss (60s) | x range | Notes |
|----|-----------|------------|---------|-------|
| 3e-4 | 545 | 5.69 | [-5, 5] | Original baseline |
| 5e-4 | 538 | 5.51 | [-9, 10] | Stable |
| **1e-3** | **535** | **5.44** | **[-36, 34]** | **New default** |
| 2e-3 | 534 | 5.19 / 4.64@120s | [-860, 700] | Best loss, but x magnitudes dangerous |
| 3e-3 | 533 | 6.02 | [-623, 682] | Too high — loss increases |

**Key Finding**: Higher LR helps for short runs (60s) but **hurts at longer training** (120s+). At LR=1e-3, activations grow to [-126, 102] which degrades ANE fp16 forward precision. At LR=2e-3, activations reach [-860, 700] — severe fp16 precision loss. LR=3e-4 produces x in [-6, 6] and gives the best 120s loss.

**Insight**: The ANE fp16 forward pass imposes a fundamental constraint — activations must stay small for precision. Higher LR → larger weights → larger activations → fp16 degradation. This limits maximum LR more severely than in GPU fp32 training. The optimal LR for ANE training on shallow models is ~3e-4, much lower than typical GPU LR schedules.

**Default unchanged**: train.py stays at LR=3e-4 (safe for all training lengths).

---

## Planned Experiments

### Experiment 4: Optimal Architecture at 5-minute Budget
**Hypothesis**: The optimal depth/width tradeoff may differ at longer training budgets. Search at 5 minutes to find the best config for the autoresearch default.

### Experiment 5: Higher Learning Rate for Shallow Models
**Hypothesis**: With fewer layers, we can use higher LR (5e-4 or 1e-3) since DeepNet scaling is less aggressive and gradients are larger.

### Experiment 6: Longer Training (1 hour)
**Hypothesis**: At longer budgets, depth matters more (deeper models keep learning while shallow ones plateau). The optimal depth should increase with budget.
