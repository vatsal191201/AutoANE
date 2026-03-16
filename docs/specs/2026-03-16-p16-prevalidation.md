# P16 Pre-Validation: Before Implementing, Verify Key Assumptions

**Date**: 2026-03-16
**Purpose**: Before writing ~300 lines of Obj-C, validate the 3 highest-risk assumptions
using data we already have or quick experiments.

---

## Pre-validation 1: Can we reuse condition13's backward timing for P16?

**Concern**: condition13 was a full backprop run with ALL weight gradients (Wq/Wk/Wv/Wo/W1/W2/W3).
P16 only needs LoRA gradients (dA, dB for Wq/Wk/Wv/Wo). The FFN backward (W1/W2/W3) accounts
for ~60% of backward compute.

**Answer**: YES, we still need full dx backward through ALL layers for the gradient chain.
Even though we only UPDATE LoRA weights, the dx must flow through the entire network.
The FFN backward dx is needed for the residual connection to propagate gradients to earlier layers.

However: we can SKIP the dW computation for frozen weights (W1/W2/W3/Wq/Wk/Wv/Wo base).
The dW matmuls in train.m are dispatched to `dw_q` (background GCD queue):
- Lines 1021-1035: dW for FFN (3 cblas_sgemm calls, async)
- Lines 1076-1092: dW for attention (4 cblas_sgemm calls, async)

For P16, we replace these with LoRA gradient projection:
- `lora_grad_project()` for each of Wq, Wk, Wv, Wo (4 calls, rank-8, negligible time)
- Skip dW for W1, W2, W3 entirely (frozen, no LoRA on FFN)

**Estimated time savings**: The async dW matmuls run in parallel with ANE dispatches.
In CPU-only mode, they contribute to total step time. In P16, we skip 7 large dW matmuls
and replace with 8 tiny LoRA grad projections. Net: ~0ms change (dW was async anyway).

**STATUS**: VALIDATED. No additional experiment needed.

## Pre-validation 2: Forward pass activation memory

**Concern**: Need to save per-layer activations during forward for backward use.

**Calculation** (DIM=960, SEQ=256, HIDDEN=2560, 32 layers):

| Buffer | Shape | Size per layer | Purpose |
|--------|-------|---------------|---------|
| xnorm (pre-attention) | [DIM, SEQ] | 0.94 MB | Input to attention backward |
| Q | [Q_DIM, SEQ] | 0.94 MB | SDPA backward |
| K | [KV_DIM, SEQ] | 0.31 MB | SDPA backward |
| V | [KV_DIM, SEQ] | 0.31 MB | SDPA backward |
| attn_out | [Q_DIM, SEQ] | 0.94 MB | Wo backward |
| x2 (post-attention) | [DIM, SEQ] | 0.94 MB | FFN backward input |
| x2norm (pre-FFN) | [DIM, SEQ] | 0.94 MB | FFN backward |
| h1 | [HIDDEN, SEQ] | 2.50 MB | SiLU backward |
| h3 | [HIDDEN, SEQ] | 2.50 MB | SiLU backward |
| silu_out | [HIDDEN, SEQ] | 2.50 MB | W2 backward |
| **Total per layer** | | **12.8 MB** | |
| **Total 32 layers** | | **410 MB** | |

**System memory budget**:
- Model weights: ~1.4 GB (361.8M params x 4 bytes)
- LoRA adapters: ~6.5 MB
- LoRA Adam state: ~13 MB
- LoRA gradients: ~6.5 MB
- Activations: 410 MB
- Working buffers: ~100 MB
- **Total: ~1.94 GB** << 16 GB

**STATUS**: VALIDATED. 410 MB fits easily.

**CAVEAT**: condition13 does NOT save activations (they're computed and used immediately
in each backward layer). The existing train.m LayerActs struct saves only a subset:
- x (residual stream): saved
- xnorm: saved
- Q, K, V: saved
- attn_out: saved
- x2: saved
- x2norm: saved
- h1, h3: saved
- silu_out: saved

All needed activations ARE already saved in LayerActs (train.m:156-173).
We just need to allocate and populate them during the forward pass.

## Pre-validation 3: Conv-fused forward produces correct activations for backward

**Concern**: Conv-fused forward uses ANE (fp16). Backward expects fp32 activations.
The conv_fused path in train_mezo.m reads ANE output with cvt_f16_f32, so activations
are already in fp32 by the time they reach CPU. But are they accurate enough?

**From Finding 5 (E36)**: ANE matmul-only matches CPU quality to 4 decimal places.
Conv-fused uses the same matmul kernels. The fp16->fp32 conversion introduces
at most 1 ULP of error per element.

**From condition13 vs condition14**: condition14 (ANE forward + CPU backward) achieves
val_loss 1.97 vs condition13 (CPU-only) 1.92. The 0.05 difference is from fp16
non-linear ops (RoPE, softmax, SiLU on ANE). In our P16 setup, these run on CPU
(same as train_mezo.m's conv_fused path: conv1x1 for matmuls, CPU for everything else).

**STATUS**: VALIDATED. Conv-fused forward produces fp32-quality activations.
The only fp16 operations are the 7 projection matmuls (conv1x1), and Finding 5
confirmed these match CPU to 4 decimals.

---

## Summary: All Pre-validations Pass

| # | Question | Answer | Status |
|---|---------|--------|--------|
| 1 | Can we skip dW for frozen weights? | Yes, LoRA grad projection replaces dW | PASS |
| 2 | Does 410MB activation storage fit? | Yes, total ~2GB << 16GB | PASS |
| 3 | Are conv-fused activations accurate enough? | Yes, fp16 matmuls match CPU to 4 decimals | PASS |

**Proceed to implementation.**
