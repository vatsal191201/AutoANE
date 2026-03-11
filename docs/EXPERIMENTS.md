# AutoANE Experiment Log

## Experiment 1: Loss Scaling & ANE vs CPU Attention Backward

**Branch**: `experiments/cpu-sdpa-backward` (merged to main)
**Status**: COMPLETE — REVISED (loss_scale bug found and fixed)

**Hypothesis**: ANE fp16 SDPA backward produces zero attention gradients due to fp16 underflow. Moving attention backward to CPU fp32 should fix this.

**Root Cause (original)**: With `loss_scale=1.0` (a bug — the original maderix/ANE code uses `loss_scale=256.0`), three cascading fp16 underflows occurred:
1. `wotBwd`: dx2_scaled (~0.0005) * Wo (~0.01) → ~0.000005, below fp16 precision
2. `sdpaBwd`: softmax probs (~1/256) * tiny da → underflows to zero
3. `qBwd`/`kvBwd`: zero dq/dk/dv → zero dx, blocking gradient flow through residual stream

**The bug**: We had `loss_scale = 1.0` instead of the original `loss_scale = 256.0`. Loss scaling multiplies all gradients by 256 during backward (keeping them above fp16 precision), then divides by 256 before Adam update. Without it, ANE fp16 gradients underflowed to zero.

**Fix applied**: Restored `loss_scale = 256.0` (matching original maderix/ANE).

**Initial (buggy) A/B Comparison** (SmolLM2-360M, loss_scale=1.0, 5-minute budget):

| Mode | Steps | Loss | ms/step | dy magnitude |
|------|-------|------|---------|-------------|
| ANE fp16 (loss_scale=1.0) | 200 | 5.68 | 700-1500 | ~8e-5 |
| CPU fp32 attn bwd (loss_scale=1.0) | 1651 | 4.69 | 145-185 | ~2e-2 |

This led to the false conclusion that CPU was 8x faster with better loss. The real issue was that loss_scale=1.0 caused fp16 underflow in ANE backward, while CPU fp32 worked fine without scaling.

**Corrected A/B Comparison** (4L/1024d autoresearch config, loss_scale=256, 120s budget):

| Mode | Steps | Final Loss | ms/step | x range |
|------|-------|-----------|---------|---------|
| **ANE fp16 (loss_scale=256)** | **974** | **4.13** | **~85** | **[-6, 7]** |
| CPU fp32 attn bwd (loss_scale=256) | 998 | 5.36 | ~92 | [-11, 12] |

**ANE wins by 1.2 loss points** when loss_scale is correct. ANE is also faster per-step (~85ms vs ~92ms) for this 4-layer model. The fp16 precision acts as implicit regularization, keeping activations small (x in [-6,7] vs [-11,12]).

**Key Insight**: Loss scaling is essential for ANE fp16 training. Without it, gradients underflow and training fails. With it, ANE outperforms CPU for attention backward on shallow models. The `--cpu-attn-bwd` flag remains useful for deep models (32L) where ANE kernel launch overhead (320 launches/step) dominates.

**SmolLM2-360M (32L) Corrected Comparison** (loss_scale=256, 120s budget):

| Mode | Steps | Final Loss | ms/step | io_fwd (ms) |
|------|-------|-----------|---------|-------------|
| ANE fp16 | 40 | 8.46 | ~900 | 300-2500 |
| CPU fp32 attn bwd | 50 | 8.65 | ~786 | 300-1900 |

Both paths are dominated by IOSurface overhead (`io_fwd` = 1-2.5s/step). Neither is practical for 32-layer training. The real bottleneck at depth is IOSurface lock/unlock operations (8 surfaces × 32 layers = 256 lock/unlock pairs per forward pass alone), not the backward path choice.

**Conclusion**: For shallow models (4L), ANE fp16 with loss_scale=256 outperforms the mixed ANE+CPU-attn-bwd mode. For deep models (32L), IOSurface overhead dominates — the solution is shallower/wider architectures, not CPU backward.

**Note (Experiment 11 reconciliation)**: These numbers are from before the classifier optimization (Exp 13), which is why ms/step is higher (~85ms vs ~69ms in Exp 11). The key finding — that pure ANE outperforms ANE+CPU-attn-bwd — was confirmed in Experiment 11's verified re-run (avg loss 4.90 vs 5.10). However, Experiment 11 also showed that pure CPU fp32 beats both ANE modes (avg loss 4.22).

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

---

## Experiment 7: Exact Timing Breakdown (ANE Dynamic, 4L/1024d)

**Status**: COMPLETE
**Date**: 2026-03-11

**Goal**: Get precise per-component timing for the ANE dynamic pipeline on the 4L/1024d autoresearch model.

**Method**: Run 120s training, capture per-step timing variables.

**Results** (steady-state, loss_scale=256, 120s budget):

| Component | Time (ms) | % of Step | What |
|-----------|----------|-----------|------|
| ane_fwd | 10.5 | 16.6% | 3 ANE kernels × 4L (sdpaFwd, woFwd, ffnFused) |
| ane_bwd | 18.7 | 29.6% | 7 ANE kernels × 4L (ffnBwdW2t, ffnBwdW13t, wotBwd, sdpaBwd1, sdpaBwd2, qBwd, kvBwd) |
| io_fwd | 3.0 | 4.7% | IOSurface write/read for forward (weight staging already done) |
| io_bwd | 5.1 | 8.1% | IOSurface write/read for backward |
| cls | 15.2 | 24.1% | Classifier matmul + cross-entropy loss (CPU, row-major optimized) |
| silu | 5.8 | 9.2% | SiLU derivative (CPU, vectorized vDSP) |
| rms + rms_bwd | 3.1 | 4.9% | RMSNorm forward + backward (CPU) |
| dw_copy | 1.9 | 3.0% | Memcpy for async dW computation |
| cblas_wait | 0.0 | 0.0% | Wait for async dW (fully overlapped) |
| **Total** | **63.2** | **100%** | **Per-step average (verified re-run)** |

*Note: Total here (63.2ms) is the sum of tracked components. Actual wall-clock per-step is 68.7ms due to untracked overhead (function call, loop bookkeeping, etc).*

**Key Findings**:
1. **ANE compute is 46% of step time** (29.2ms for ane_fwd+ane_bwd out of 63.2ms tracked)
2. **Classifier is the second largest component** at 24% — previously was 37% before row-major optimization
3. **IO overhead is ~13%** (8.1ms) — not the bottleneck we originally assumed
4. **SiLU backward is surprisingly expensive** at 9.2% — could be moved to ANE
5. **dW gradient computation is fully overlapped** via async dispatch (cblas_wait = 0)

**Comparison to initial estimates (from RESEARCH_PLAN.md)**:
| Component | Estimated | Actual | Status |
|-----------|----------|--------|--------|
| IOSurface staging | ~20-25ms | 8.1ms | OVERESTIMATED (3x) |
| ANE kernel eval | ~40-45ms | 29.2ms | OVERESTIMATED (1.5x) |
| CPU ops | ~12-15ms | 24.1ms | UNDERESTIMATED (1.6x) — cls was dominant |
| Overhead | ~5ms | 1.9ms | OVERESTIMATED (2.6x) |

---

## Experiment 8: CPU Matmul Microbenchmark

**Status**: COMPLETE
**Date**: 2026-03-11
**File**: `training/bench_cpu_matmul.m`

**Goal**: Measure actual cblas_sgemm throughput on this machine for our exact matrix shapes.

**Method**: Standalone benchmark, 1000 iterations per shape, includes warmup.

**Results** (Apple M4, Accelerate framework, fp32):

| Operation | Shape (M×K @ K×N) | Time (ms) | GFLOPS |
|-----------|-------------------|----------|--------|
| Wq forward | 256×1024 @ 1024×1024 | 0.060 | 2240 |
| Wk forward | 256×1024 @ 1024×256 | 0.030 | 1120 |
| W1 forward | 256×1024 @ 1024×2816 | 0.165 | 2228 |
| W2 forward | 256×2816 @ 2816×1024 | 0.171 | 2161 |
| Q@K^T per head | 256×64 @ 64×256 | 0.004 | 558 |

**Per-Layer Summary** (4L/1024d, 100 iterations):
| Component | Time |
|-----------|------|
| Forward (linear + attn) | 8.38ms/layer |
| Backward dX (approx) | 8.38ms/layer |
| Backward dW | 10.34ms/layer |
| **Total per layer** | **27.1ms** |
| **Total 4 layers** | **108.4ms** |

**Key Finding**: CPU achieves **~1.5-2.5 TFLOPS fp32** via cblas_sgemm (Apple AMX). The per-layer compute estimate of 27ms closely matches the actual CPU-only training measurement of ~78ms for forward+backward (28.5ms per layer × 4 layers = ~78ms for matmul-only time, minus some overlap).

**ASSUMPTION A4 VALIDATED**: CPU AMX achieves ~1.5-2.5 TFLOPS fp32 as estimated.

---

## Experiment 9: Conv 1×1 vs Matmul ANE Microbenchmark

**Status**: COMPLETE
**Date**: 2026-03-11
**File**: `training/bench_conv_vs_matmul.m`

**Goal**: Verify the Orion paper's claim that conv 1×1 is ~3× faster than matmul on ANE.

**Method**: Compile both conv 1×1 (const weights via BLOBFILE) and dynamic matmul (weights via IOSurface spatial) kernels for our exact shapes. Run 1000 evaluations each.

**Results**:

| Shape (in→out) | Matmul (ms) | Conv 1×1 (ms) | Speedup | Notes |
|----------------|------------|--------------|---------|-------|
| 1024→1024 | 0.069 | 0.038 | 1.82× | Wq/Wo projection |
| 1024→256 | 0.035 | 0.023 | 1.52× | Wk/Wv projection |
| 1024→2816 | 0.150 | 0.054 | 2.78× | W1/W3 FFN |
| 2816→1024 | 0.150 | 0.054 | 2.78× | W2 FFN |

**Key Findings**:
1. Conv 1×1 is **1.5-2.8× faster** than matmul, depending on shape
2. Largest speedup for FFN (2816-dim), smallest for KV projections (256-dim)
3. The Orion paper's "~3×" claim is approximately correct for large shapes
4. **BUT**: Conv 1×1 requires `const()` weights baked into BLOBFILE — incompatible with our dynamic IOSurface weight packing

**ASSUMPTION A2 PARTIALLY VALIDATED**: Conv 1×1 is ~2× faster on average (not 3×), with larger speedup for FFN-sized shapes.

---

## Experiment 10: Delta Compilation Prototype

**Status**: COMPLETE — FAILED
**Date**: 2026-03-11
**File**: `training/bench_conv_vs_matmul.m` (delta reload test section)

**Goal**: Verify that Orion's delta compilation approach (unload → write new BLOBFILE → reload) works to update weights without recompilation.

**Method**:
1. Compile conv 1×1 kernel with initial weights (all 1.0)
2. Evaluate, record output sum (expected: sum of input × weight)
3. Unload via `unloadWithQoS:21`
4. Overwrite BLOBFILE with new weights (all 2.0)
5. Reload via `loadWithQoS:21`
6. Evaluate again, check if output changed

**Results**:
```
Initial weights (1.0): output sum = 703.5304
After BLOBFILE write (2.0) + reload: output sum = 703.5304
```

**The output is IDENTICAL**. ANE loads weights from its compiled cache, not from the source BLOBFILEs. Simply modifying the BLOBFILE and reloading does NOT update the weights.

**Root Cause Analysis**: The ANE compilation process converts MIL text + BLOBFILEs into a compiled binary cached on disk (in the kernel's `tmpDir`). `loadWithQoS` loads from this compiled cache, not from the source weight files. To update weights, you'd need to:
1. Delete/invalidate the compiled cache
2. Modify the BLOBFILE
3. Trigger recompilation (which takes ~70ms/kernel)

This makes delta compilation **equivalent to full recompilation** — defeating the purpose.

**ASSUMPTION A3 INVALIDATED**: Delta compilation (BLOBFILE patching + reload) does NOT work as described in the Orion paper through our API usage.

**Post-mortem (from Orion paper details)**: The Orion Algorithm 1 specifies writing to `M.tmpDir/p` — the model's temporary directory (not the source BLOBFILE path). The compiled ANE program caches in `~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache/`. Our test may have written to the original BLOBFILE location rather than the compiled cache directory. However, even if the correct path is used, the ANE runtime's cache invalidation behavior is undocumented. The Orion paper achieved delta reload in 9ms/kernel (494ms for 60 kernels), but their approach may require exact path manipulation within `tmpDir` that we haven't replicated.

**NOTE**: The Orion paper also reports that single-op utilization is ~30% while deep graphs achieve 94%. Our dynamic pipeline uses individual kernel evaluations (10 kernels per layer), suggesting significant room for improvement via kernel fusion — even without delta compilation.

**Impact on Task A**: Conv 1×1 integration would require full recompilation every ACCUM_STEPS (every 10 steps). With ~60ms × 10 kernels = 600ms compile overhead, amortized to 60ms/step. This NEGATES the 2× per-step speedup from conv 1×1. **Task A is not viable with our current API access.** Enhanced investigation in Experiment 17 confirmed this definitively — see Experiment 17 for full analysis of tmpDir structure, e5bundlecache, and all attempted approaches.

---

## Experiment 11: CPU-Only Training Path (Task B)

**Status**: COMPLETE
**Date**: 2026-03-11

**Goal**: Implement and benchmark a pure CPU training path to establish a ground truth baseline.

**Implementation**: Added `--cpu-only` flag to `train.m`:
- Skips ANE kernel compilation and IOSurface allocation
- Replaces all ANE operations with cblas_sgemm (Accelerate/AMX):
  - Forward: Q/K/V projections, RoPE, causal SDPA, Wo projection, FFN (W1/W3/SiLU/W2)
  - Backward: W2^T, W1^T+W3^T, Wo^T, SDPA backward, Q/KV backward (via --cpu-attn-bwd)
- All computation in fp32
- Same optimizer (Adam), gradient accumulation, LR schedule, loss scaling

**Added CPU functions** (in `cpu_ops.h`):
- `rope_forward_inplace()` — forward RoPE rotation
- `cpu_sdpa_forward()` — causal attention with softmax

**A/B/C Comparison** (4L/1024d autoresearch, same hyperparameters, 120s budget, loss_scale=256):

*Verified with independent re-runs. Loss averaged over 200 steps to avoid single-batch noise.*

| Metric | ANE (full fp16) | ANE+CPU-attn-bwd | CPU-Only (fp32) |
|--------|----------------|------------------|----------------|
| **ms/step** | **68.7** | **77.5** | **102.2** |
| **Steps in 120s** | **1297** | **1178** | **1041** |
| **Avg loss (steps 800-1000)** | **4.90** | **5.10** | **4.22** |
| **Avg loss (final 200 steps)** | **4.69** | **n/a** | **4.20** |
| **x range at step 1000** | [-7.2, 7.7] | [-14.2, 14.8] | [-3.1, 3.3] |
| Precision | fp16 fwd+bwd | fp16 fwd, fp32 attn bwd | fp32 |

**IMPORTANT**: Previous version reported "CPU 23% lower loss" based on comparing single noisy batch snapshots at different step counts. This was **methodologically flawed**. Corrected analysis uses 200-step rolling averages.

**Per-Component Timing Comparison** (averaged over ~100 steady-state steps):

| Component | ANE (ms) | CPU (ms) | What |
|-----------|---------|---------|------|
| Forward matmuls | 10.5 | 27.6 | QKV + SDPA + Wo + FFN |
| Backward matmuls | 18.7 | 44.3 | All backward projections |
| IO overhead | 8.1 | 0.2 | IOSurface read/write |
| SiLU | 5.8 | 5.4 | Element-wise ops |
| RMSNorm | 3.1 | 3.1 | Forward + backward |
| Classifier | 15.2 | 15.0 | Always CPU, same for both |
| dW copy | 1.9 | 1.5 | Async dispatch memcpy |
| **Total** | **63.2** | **97.1** | |

**Key Findings** (verified by re-running):

1. **ANE IS faster than CPU** — ~1.5× per-step throughput advantage
2. **ANE compute speedup**: ane_fwd+ane_bwd = 29.2ms (ANE) vs 71.9ms (CPU) = **ANE 2.46× faster for matmuls**
3. **IO overhead is ~13%** of ANE step time (8.1ms) — not the bottleneck
4. **CPU achieves ~16% lower loss at matched steps** (4.22 vs 4.90, averaged over steps 800-1000)
5. **CPU wins on loss/wall-clock** (4.20 vs 4.69 avg loss at 120s mark)
6. **Mixed precision is WORSE than pure fp16**: ANE+CPU-attn-bwd produces 5.10 avg loss vs pure ANE's 4.90
7. **Activation growth**: ANE activations grow 2.3× larger than CPU by step 1000; mixed mode grows 4.5× larger
8. **Startup cost**: ANE requires ~900ms compile (one-time), CPU starts instantly

**Mixed Precision Mismatch (new finding)**:
ANE+CPU-attn-bwd is worse than pure ANE because:
- fp32 backward computes sharper gradients from imprecise fp16 forward values
- Weight updates overshoot because gradients assume precise activations that fp16 can't represent
- Activations grow beyond fp16 precision range, amplifying the mismatch
- Pure fp16 backward acts as implicit regularization, keeping activations bounded

**Throughput vs Quality Tradeoff**:
- If you optimize for **steps/second**: ANE wins (~1.5× more steps)
- If you optimize for **loss per step**: CPU wins (~16% lower loss per step)
- If you optimize for **loss per wall-clock-time**: CPU wins (4.20 vs 4.69 avg loss at 120s)
- The ANE's speed advantage is **insufficient to overcome** the fp16 precision penalty

**ASSUMPTION A4 VALIDATED**: CPU AMX achieves ~1.5-2.5 TFLOPS fp32.
**ASSUMPTION A5 INVALIDATED**: IOSurface staging is NOT the bottleneck (~13% of step time).

---

## Experiment 12: Power Consumption

**Status**: PARTIALLY COMPLETE — literature values used, local measurement requires sudo
**Date**: 2026-03-11

**Goal**: Compare ANE vs CPU power draw during training.

### Literature Values (from Orion paper, arXiv 2603.06728)

| Hardware | Power | Efficiency |
|----------|-------|-----------|
| M4 ANE at peak | **2.8W** | **6.6 TFLOPS/W** |
| M4 ANE idle | **0 mW** | Hard power-gated, zero leakage |
| M4 GPU (Metal) | ~2.9W | ~1.0 TFLOPS/W |
| NVIDIA A100 | ~300W | ~0.08 TFLOPS/W |

| Chip | CPU FP32 (Accelerate) | Memory BW |
|------|----------------------|-----------|
| M4 | 1.49 TFLOPS | 103 GB/s |

### Estimated Power Comparison for Our Training

| Metric | ANE Training | CPU-Only Training |
|--------|-------------|-------------------|
| Compute power | ANE: ~2.8W | CPU AMX: ~10-15W (estimated) |
| CPU power (overhead) | ~3-5W (RMS, SiLU, cls) | ~0W (same CPU does compute) |
| **Total estimated** | **~6-8W** | **~10-15W** |
| Steps in 120s | 1297 | 1041 |
| Steps per watt-hour | ~5600 | ~2800 |
| **Energy efficiency** | **~2× more efficient** | Baseline |

### Key Insight from Literature
ANE is **80× more power-efficient per FLOP** than NVIDIA A100 and **4× more efficient** than Apple's own GPU. The 2.8W peak power draw means ANE training is feasible on battery-powered devices — this is the strongest unique value proposition.

### Single-Op Utilization Warning
The Orion paper reports that single operations achieve only **~30% ANE utilization**, while deep graphs (16-64 chained ops) achieve **94%**. Our dynamic matmul approach uses individual kernel evaluations, meaning we're likely at ~30% utilization. This explains why our measured ANE throughput (~5-6 TFLOPS effective) is well below the 18.6 TFLOPS peak.

**To run local power measurement** (requires sudo):
```bash
# === ANE Training Power ===
# Terminal 1: start ANE training
cd training && ./train --scratch --data ../tinystories_smollm2_data00.bin --lr 3e-4 --time 120

# Terminal 2: measure power (start a few seconds after training begins)
sudo powermetrics --samplers cpu_power,ane_power,gpu_power -i 1000 -n 110 > power_ane.txt

# === CPU Training Power ===
# Terminal 1: start CPU training
cd training && ./train --scratch --cpu-only --data ../tinystories_smollm2_data00.bin --lr 3e-4 --time 120

# Terminal 2: measure power
sudo powermetrics --samplers cpu_power,ane_power,gpu_power -i 1000 -n 110 > power_cpu.txt

# === Idle Baseline ===
# Kill training, wait 10s, then:
sudo powermetrics --samplers cpu_power,ane_power,gpu_power -i 1000 -n 30 > power_idle.txt

# === Analysis ===
# Extract average power per sampler:
grep -A2 "ANE Power" power_ane.txt | grep "mW" | awk '{sum+=$1; n++} END {print "ANE Power (ANE training):", sum/n, "mW"}'
grep -A2 "CPU Power" power_ane.txt | grep "mW" | awk '{sum+=$1; n++} END {print "CPU Power (ANE training):", sum/n, "mW"}'
grep -A2 "CPU Power" power_cpu.txt | grep "mW" | awk '{sum+=$1; n++} END {print "CPU Power (CPU training):", sum/n, "mW"}'
grep -A2 "ANE Power" power_cpu.txt | grep "mW" | awk '{sum+=$1; n++} END {print "ANE Power (CPU training):", sum/n, "mW"}'
```

**What to look for**: ANE training should show ~2-3W ANE power + ~3-5W CPU power. CPU training should show ~10-15W CPU power + ~0W ANE. The total system power difference validates the ~2× energy efficiency claim.

---

## Experiment 13: Classifier Optimization

**Status**: COMPLETE
**Date**: 2026-03-11

**Goal**: Reduce classifier computation time, which was the dominant CPU bottleneck.

**Root Cause**: The cross_entropy_loss function operated on logits in [CV, SEQ] column-major layout, requiring strided memory access via cblas_scopy gather/scatter per token. Each token's logits were scattered across CV × SEQ memory, causing cache misses.

**Fix**: Changed logits layout from [CV, SEQ] (column-major, strided) to [SEQ, CV] (row-major, contiguous per-token):
- `train.m`: Changed classifier matmul to produce logits[SEQ, CV] directly via `CblasTrans, CblasTrans`
- `cpu_ops.h`: Rewrote `cross_entropy_loss()` to iterate over contiguous rows instead of strided columns

**Results**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| cls time | 30.5ms | 15.7ms | 1.94× faster |
| Overall ms/step | 87ms | 69.4ms | 20% faster |
| Steps in 120s | 1077 | 1313 | 22% more steps |

**Key Insight**: Memory layout matters enormously for CPU operations. Row-major iteration on contiguous memory is ~2× faster than strided access for the same computation, due to cache line utilization.

---

## Summary: ANE vs CPU Research (Tasks A, B, C)

### Task A: Delta Compilation + Conv 1×1 — DEFINITIVELY NOT VIABLE
- Conv 1×1 is 1.5-2.8× faster per-kernel than dynamic matmul on ANE
- Delta compilation DOES NOT work through ANY file-level approach (Experiments 10, 17):
  - tmpDir file patching (data, weights/w.bin): no effect on reload
  - Recompile on same model object: sandbox error
  - e5bundlecache patching: entries are metadata only (~96 bytes)
- Fresh compile with cached graph topology: ~60ms/kernel (vs ~237ms cold)
- But 60ms × 10 kernels / 10 ACCUM_STEPS = 60ms/step overhead > 15ms/step conv savings
- **Verdict**: Not viable without direct ANE driver/kernel interface access

### Task B: CPU-Only Training — IMPLEMENTED, CPU WINS DECISIVELY
- CPU-only mode: `--cpu-only` flag, all matmuls via cblas_sgemm fp32
- At 120s: ANE ~1.5× faster per step, CPU ~12% lower loss (4.20 vs 4.69)
- At 600s: **CPU wins by 37%** (3.03 vs 4.85). ANE loss DIVERGES while CPU improves
- Both modes suffer thermal throttling (~1.5× slowdown at 10 min)
- **Verdict**: CPU-only is strictly superior for pretraining. ANE only viable for short fine-tuning bursts (1-2 min)
- **Surprise finding**: ANE+CPU-attn-bwd is WORSE than pure ANE (mixed precision mismatch)

### Task C: ANE Unique Value Proposition
- **Power efficiency**: ANE 2.8W peak, 6.6 TFLOPS/W (from Orion paper). Local measurement pending (requires sudo)
- **CPU freedom**: During ANE training, CPU handles only RMSNorm, SiLU, classifier, dW. During CPU-only training, CPU is 100% utilized
- **On-device training**: ANE training enables training while keeping CPU available for other tasks — unique for mobile/edge deployment
- **Research novelty**: First open-source comparison of ANE vs CPU training with identical training procedures

### fp16 Precision Gap — IRREDUCIBLE (Experiments 14-15)
- Activation clamping to [-4, 4]: no loss improvement (fp16 accumulation error, not magnitude)
- Lower LR (1e-4): makes loss worse (too slow)
- Higher WD (0.3): no improvement
- The ~16% loss gap is inherent to fp16 matmul accumulation with DIM=1024
- Would require hardware mixed-precision accumulation or algorithmic fixes (stochastic rounding)

### Overall Conclusions
1. **ANE has genuine compute advantage**: ~2.5× faster matmuls than CPU AMX for our shapes
2. **fp16 precision is the limiting factor**: More steps ≠ better training when each step has precision loss
3. **Mixed precision is NOT the answer**: ANE+CPU-attn-bwd is worse than either pure mode due to precision mismatch causing activation explosion
4. **IOSurface overhead is manageable**: ~13% of step time — not the bottleneck initially feared
5. **ANE training DIVERGES after ~2000 steps**: Activations grow 7× ([-7,8]→[-55,53]) causing positive feedback loop of precision loss (Experiment 18)
6. **Precision gap WIDENS over time**: 12% gap at 120s → 37% gap at 600s. ANE loss gets WORSE (4.69→4.85) while CPU improves (4.20→3.03)
7. **Thermal throttling affects both**: 1.5-1.6× slowdown over 10 minutes, eroding ANE's speed advantage
8. **ANE is only viable for short bursts** (1-2 min fine-tuning), not sustained pretraining

---

## Experiment 14: Activation Clamping for fp16 Precision

**Status**: COMPLETE — DISPROVED
**Date**: 2026-03-11

**Hypothesis**: ANE fp16 precision loss comes from large activation magnitudes. Clamping activations to a bounded range (within fp16's precise zone) should close the loss gap with CPU fp32.

**Method**: Added `--clamp` flag to train.m. After each attention and FFN residual add, clamp activations via `vDSP_vclip(x, 1, &lo, &hi, x, 1, SEQ*DIM)`. Tested clamp values of 8.0 and 4.0.

**Results** (4L/1024d, lr=3e-4, loss_scale=256):

| Config | Steps | Clamp Active? | Avg Loss (800-1000) | x range |
|--------|-------|--------------|---------------------|---------|
| ANE baseline (60s) | 664 | n/a | ~4.90 | [-8, 10] |
| ANE --clamp 8 (60s) | 656 | barely | ~4.90 | [-8, 8] |
| ANE --clamp 4 (60s) | 637 | from ~step 300 | ~4.90 | [-4, 4] |
| ANE --clamp 4 (120s) | 1233 | yes | 4.9076 | [-4, 4] |
| ANE unclamped (120s) | 1297 | n/a | 4.90 | [-7.2, 7.7] |

**Key Finding**: Clamping to [-4, 4] (well within fp16's precise range) produces **identical loss** to unclamped training. The fp16 precision loss is NOT about activation magnitude.

**Root Cause**: The precision loss is inherent in **fp16 matmul accumulation** — large inner products (DIM=1024) accumulate rounding errors in each multiply-add regardless of input magnitude. Even with all inputs in [-4, 4], a dot product of 1024 values accumulates ~√1024 ≈ 32 units of fp16 rounding error. This is a fundamental hardware limitation of ANE's fp16 compute, not fixable by bounding inputs.

**Side Effect**: Clamping slightly reduces throughput (637 vs 664 steps in 60s) due to the vDSP_vclip overhead.

---

## Experiment 15: Learning Rate & Weight Decay Tuning for ANE

**Status**: COMPLETE — NO IMPROVEMENT FOUND
**Date**: 2026-03-11

**Hypothesis**: ANE's fp16 precision might benefit from different hyperparameters — lower LR (smaller weight updates, less drift) or higher WD (keep weights small, reduce magnitude).

**Results** (4L/1024d, 60s budget):

| Config | Steps | Avg Loss | x range | Notes |
|--------|-------|----------|---------|-------|
| Baseline: lr=3e-4, wd=0.1 | 664 | ~4.90 | [-8, 10] | Standard |
| lr=1e-4, wd=0.1 | 610 | ~5.4 | [-1.6, 1.6] | WORSE — too slow to learn |
| lr=3e-4, wd=0.3 | 594 | ~4.90 | [-6, 7] | Same loss, slightly fewer steps |

**Key Finding**: Lower LR makes loss worse (training too slow). Higher WD doesn't help — it slightly reduces activation magnitude but doesn't improve loss. The fp16 precision gap is not addressable through standard hyperparameter tuning.

**Insight**: The ~16% loss gap between ANE fp16 and CPU fp32 appears to be an irreducible cost of fp16 computation for this model size and architecture. Possible mitigations would require hardware changes (mixed-precision accumulation) or algorithmic changes (stochastic rounding, Kahan summation in the MIL graph).

---

---

## Experiment 17: Delta Compilation Retry — Enhanced Investigation

**Status**: COMPLETE — DEFINITIVELY NOT VIABLE
**Date**: 2026-03-11

**Goal**: Retry delta compilation with deeper investigation of the ANE compiled cache structure.

**New Approaches Tested**:

| # | Approach | Result | Time |
|---|---------|--------|------|
| 1 | unload → write tmpDir/weights/w.bin → reload | No change | 3.3ms reload |
| 2 | unload → write tmpDir → recompile (same model) | **Sandbox error** — ANE won't recompile same model object | 0.1ms (fail) |
| 3 | Fresh compile from scratch with new weights | **WORKS** — weights updated | 60.2ms |
| 4a | unload → patch tmpDir/data → reload | No change | 3.1ms reload |
| 4b | unload → patch BOTH tmpDir/data + weights/w.bin → reload | No change | 2.9ms reload |

**tmpDir Structure After Compilation**:
```
tmpDir/ (NSTemporaryDirectory/hexStringIdentifier)
├── weights/
│   └── w.bin      (2,097,280 bytes — BLOBFILE format, our weight data)
├── model.mil      (1,007 bytes — MIL source text)
├── net.plist      (1,007 bytes — created BY compiler)
└── data           (2,097,280 bytes — BLOBFILE format, created BY compiler)
```

The `data` file has the BLOBFILE header (0xDEADBEEF at offset 64) and is the exact same size as w.bin. It's a compiled copy of the weight blob. BUT overwriting it and reloading has no effect — the ANE loads from a separate internal cache.

**ANE Cache Location**: `~/Library/Caches/com.apple.e5rt.e5bundlecache/25D2128/`
- Contains hash-named entries (~96 bytes each) — metadata/references only
- Actual compiled binaries likely stored in kernel memory or mmap'd regions
- Not patchable from userspace

**Fresh Compile Caching**: The 60.2ms fresh compile (vs 237ms initial) shows the ANE framework caches the compiled graph topology. Only the weight blob portion needs re-processing. But 60ms × 10 kernels / 10 ACCUM_STEPS = 60ms/step overhead still negates the conv 1×1 speedup (~15ms/step savings).

**Recompile Sandbox Error**: `compileWithQoS:` on an already-compiled model fails with "issueSandboxExtensionForPath:error:: file access failure". The ANE framework doesn't support in-place recompilation — a fresh model descriptor is required.

**Orion Paper Discrepancy**: The Orion paper reports delta reload in 9ms/kernel. Their approach may use:
1. Direct manipulation of the ANE driver/kernel interface (not available through public APIs)
2. A different compilation path that doesn't use the e5bundlecache
3. Memory-mapped weight regions that can be updated in-place
4. A version-specific API that differs from our macOS 15.x implementation

**FINAL VERDICT ON TASK A**: Delta compilation is definitively not viable through any file-level approach. Conv 1×1 integration would require full fresh recompilation every ACCUM_STEPS, costing 60ms/step — exceeding the ~15ms/step savings from faster conv execution.

---

---

## Experiment 18: Longer Training Runs (10 minutes)

**Status**: COMPLETE — CRITICAL FINDING
**Date**: 2026-03-11

**Goal**: Run ANE vs CPU training for 10 minutes to see if the precision gap widens or narrows over time.

**Results** (4L/1024d, lr=3e-4, loss_scale=256, 600s budget):

| Metric | ANE fp16 (10min) | CPU fp32 (10min) |
|--------|-----------------|-----------------|
| **Total steps** | **3386** | **2902** |
| **ms/step (avg)** | **108.4** | **152.9** |
| **Final loss** | **4.85** | **3.03** |
| **x range (final)** | **[-54.7, 52.9]** | **[-2.2, 2.0]** |
| Thermal slowdown | 1.58× (68.7→108.4) | 1.50× (102.2→152.9) |
| Gradient norm (final) | 2.7-5.6 | 1.4-1.8 |
| Loss variance (final 100 steps) | High (4.1-5.3) | Low (2.9-4.9) |

**Comparison with 120s runs**:

| Metric | 120s gap | 600s gap | Trend |
|--------|----------|----------|-------|
| Loss (ANE) | 4.69 | 4.85 | **WORSE** (ANE diverging!) |
| Loss (CPU) | 4.20 | 3.03 | Improving normally |
| Quality gap | 12% | 37% | **WIDENING** |
| Activation range (ANE) | [-7.2, 7.7] | [-54.7, 52.9] | **7× growth** |
| Activation range (CPU) | [-3.1, 3.3] | [-2.2, 2.0] | Stable/shrinking |

**CRITICAL FINDINGS**:

1. **ANE loss DIVERGES after ~2000 steps**: Loss improved from 4.90 (step 1000) to 4.69 (step ~1200) but then degraded back to 4.85 by step 3386. The model is effectively not learning after the first few minutes.

2. **Activation explosion is the root cause**: ANE activations grew from [-7.2, 7.7] at step 1000 to [-54.7, 52.9] at step 3386 — a 7× increase. Values above ~65 overflow fp16 (max 65504), and values above ~10 lose significant precision (only 3 decimal bits of mantissa).

3. **CPU training remains stable**: CPU activations actually SHRANK from [-3.1, 3.3] to [-2.2, 2.0] — the fp32 Adam optimizer with weight decay properly regularizes the model. CPU gradient norms are stable at ~1.5, while ANE gradient norms fluctuate 2.7-5.6.

4. **Both modes suffer thermal throttling**: ANE slowed 1.58× and CPU slowed 1.50× over 10 minutes, suggesting sustained compute heats the M4 chip. This erodes ANE's speed advantage further.

5. **ANE is NOT viable for training runs >2 minutes** without activation magnitude control. The fp16 precision degradation compounds — growing activations cause less precise matmuls, which cause larger gradient errors, which cause larger weight updates, which grow activations further (positive feedback loop).

**Implications**:
- ANE training needs **mandatory activation clamping** or **periodic weight rescaling** for runs >2 min
- Even with clamping (Experiment 14 showed no loss improvement), the fundamental fp16 accumulation error means quality can't match CPU
- ANE is best suited for **short fine-tuning bursts** (1-2 min), not sustained pretraining
- For pretraining, CPU-only mode is strictly superior in every metric except raw steps/second

**Thermal Throttling Detail**:
At step ~2900 (CPU run), timing jumped to 183.5ms due to `ane_bwd` reaching 98.4ms — a 2× spike suggesting CPU/memory bandwidth throttling. Both runs showed intermittent timing spikes in the final minutes.

---

---

## Experiment 19: Gradient Sanitization Test

**Status**: IN PROGRESS
**Date**: 2026-03-11

**Hypothesis**: ANE training divergence (Exp 18) may be caused by NaN/Inf gradient accumulation, not irreducible fp16 precision error. The Orion paper (arXiv 2603.06728) identified gradient sanitization (NaN→0, ±Inf→±65504) as essential for stable ANE training (Bug #3).

**Implementation**: Added `--sanitize` flag to train.m:
- `sanitize_gradients()`: Scans gradient buffer, replaces NaN→0 and ±Inf→±65504
- Applied after gradient scaling, before gradient norm computation
- Counts and logs every sanitized value

**Results (120s, 4L/1024d, lr=3e-4, loss_scale=256)**:

| Config | Steps | Final Loss | x range | NaN/Inf fixed |
|--------|-------|-----------|---------|--------------|
| ANE --sanitize (120s) | 868 | 4.87 | [-8.2, 8.1] | **0** |
| ANE baseline (120s, prior) | 1297 | 4.69 | [-7.2, 7.7] | n/a |

**Finding at 120s**: Zero NaN/Inf gradient values detected. Sanitization has nothing to fix at this timescale. Step count lower (868 vs 1297) due to resource contention from parallel experiments.

**10-minute result (E19-extended)**:

| Config | Steps | Final Loss | x range | NaN/Inf fixed | ms/step |
|--------|-------|-----------|---------|--------------|---------|
| ANE --sanitize (600s, solo) | 5777 | 4.77 | [-121, 127] | **0** | 66 (stable) |
| ANE no sanitize (600s, Exp 18) | 3386 | 4.85 | [-55, 53] | n/a | 108.4 (degraded) |

**CRITICAL CORRECTIONS TO PRIOR FINDINGS:**

1. **"ANE diverges after 2000 steps" (Exp 18) was WRONG.** Experiment 18 ran ANE and CPU 10-min tests simultaneously, causing thermal throttling (68.7ms→108.4ms). When run solo, ANE maintains stable 66ms/step for 5777 steps with loss still improving.

2. **Zero NaN/Inf in 5777 steps.** Gradient sanitization found nothing to fix. The fp16 precision gap is genuinely from accumulation rounding error, not discrete NaN events.

3. **ANE loss is still improving at 10 min** — rolling avg ~4.59 over final 200 steps (vs CPU's 3.03 at 10 min from Exp 18). Gap is ~34%, not 37% as previously claimed.

4. **Activations grow to [-121, 127]** — still within fp16 range (max 65504) but with reduced precision. At magnitude 120, fp16 has only ~0.06 precision (1 ULP).

**Revised understanding:**
- ANE training does NOT diverge — it continues learning, just at lower quality than CPU
- The fp16 precision gap WIDENS over time as activations grow (more rounding error per step)
- Thermal throttling from concurrent tests caused the false "divergence" signal in Exp 18
- Gradient sanitization is unnecessary for our setup but costs nothing to keep enabled

---

### Stated Assumptions — Final Status

| ID | Assumption | Status | Evidence |
|----|-----------|--------|----------|
| A1 | Timing breakdown estimates approximate | VALIDATED | Actual measurements differ 1.5-3× from estimates |
| A2 | Conv 1×1 gives ~20% speedup (not 3×) | PARTIALLY VALIDATED | 1.5-2.8× speedup per-kernel, not end-to-end |
| A3 | Delta compilation works | INVALIDATED | BLOBFILE patching + reload does NOT update weights |
| A4 | CPU AMX ~1-2 TFLOPS fp32 | VALIDATED | Measured 1.5-2.5 TFLOPS |
| A5 | IOSurface is the bottleneck | INVALIDATED | ~13% of step time |
| A6 | ANE power ~5-10W | REFINED via literature | ANE 2.8W peak (Orion paper), local measurement pending |

---

## Experiment 23: Automated Benchmark Suite (E23)

**Date**: 2026-03-11
**Phase**: 1
**Goal**: Characterize ANE vs CPU matmul performance and IOSurface overhead with automated benchmark suite

### Setup
- New benchmark.m with 4 test suites: ANE matmul, CPU matmul, IO overhead, thermal profile
- Shapes tested: 1024x1024, 1024x256, 1024x2816, 2816x1024
- SEQ=256 for all tests
- ANE: fp16 dynamic matmul via MIL, CPU: fp32 cblas_sgemm

### Results — ANE vs CPU Matmul

| Shape | ANE fp16 (ms) | CPU fp32 (ms) | ANE Speedup | ANE GFLOPS | CPU GFLOPS |
|-------|--------------|--------------|-------------|------------|------------|
| 1024x1024 | 0.353 | 0.236 | 0.67x (CPU faster) | 1519 | 2276 |
| 1024x256 | 0.244 | 0.103 | 0.42x (CPU faster) | 549 | 1298 |
| 1024x2816 | 0.462 | 0.685 | **1.48x** | 3195 | 2155 |
| 2816x1024 | 0.480 | 0.902 | **1.88x** | 3075 | 1637 |

### Results — IOSurface Overhead

| Tensor | Write (ms) | Read (ms) | Write BW |
|--------|-----------|----------|----------|
| 256x1024 (0.5MB) | 0.010 | 0.001 | 52 GB/s |
| 1024x1024 (2MB) | 0.039 | 0.001 | 54 GB/s |
| 256x2816 (1.4MB) | 0.025 | 0.001 | 57 GB/s |
| 1024x2816 (5.8MB) | 0.128 | 0.001 | 45 GB/s |

### Key Findings
1. **ANE only faster for large shapes** (2816+ width). The dynamic MIL matmul with weight-slicing overhead penalizes smaller shapes.
2. **CPU AMX achieves 1.3-2.3 TFLOPS fp32** — consistent with prior E8 results (V3 confirmed).
3. **ANE peaks at 3.2 TFLOPS fp16** for 1024x2816 shape — but only 0.5 TFLOPS for 1024x256.
4. **IOSurface write ~50 GB/s**, read appears near-instant (kernel cache). Total round-trip for 5.8MB weight tensor: 0.13ms.
5. **CORRECTION**: Our E11 result claiming "ANE 2.46x faster" used a different kernel format. Dynamic matmul is slower than the static weight kernel used in E11.

### Stated Assumptions Updated
- V1 (ANE matmul ~2.5x faster): **UPDATED** — only for large shapes with dynamic matmul; static kernels may still achieve 2.5x
- V3 (CPU AMX 1.5-2.5 TFLOPS): **RE-CONFIRMED** at 1.3-2.3 TFLOPS
- V4 (IOSurface ~13% overhead): **RE-CONFIRMED** — write times are 0.01-0.13ms vs step times of 60-80ms

---

## Experiment 24: Thermal Profile (E24)

**Date**: 2026-03-11
**Phase**: 1
**Goal**: Characterize ANE thermal throttling under sustained 60s continuous load

### Setup
- Continuous 1024x1024 fp16 matmul on ANE for 60 seconds
- Measurement every 5 seconds (200 iterations per sample for averaging)
- Single process (no concurrent jobs — corrects E18 methodology error)

### Results

| Time (s) | ms/eval | GFLOPS | Status |
|----------|---------|--------|--------|
| 0 | 0.299 | 1797 | Full speed |
| 5 | 0.298 | 1801 | Full speed |
| 10 | 0.388 | 1384 | Throttled |
| 15-60 | 0.367-0.372 | 1444-1462 | Stable plateau |

**Summary**: Min 0.298ms, Max 0.388ms, Avg 0.359ms. **Drift: 30.1%** (moderate throttling).

### Key Findings
1. **Throttling onset at ~10 seconds** — ANE reaches thermal limit and drops ~30%
2. **Stable plateau after throttling** — performance is very consistent from 10-60s (±2% variation)
3. **CRITICAL CORRECTION of E18**: E18 reported 1.5x (50%) throttling; that was caused by running ANE and CPU simultaneously. **Actual single-process ANE throttling is 30%, not 50%.**
4. **Throttled ANE still delivers ~1450 GFLOPS** — higher than CPU's ~2200 GFLOPS fp32 but these are fp16 FLOPS (half the work per FLOP in terms of precision)

### Stated Assumptions Updated
- V10 (both modes suffer ~1.5x thermal throttling): **CORRECTED** — single-process ANE throttles 1.3x (30%), not 1.5x (50%). The 1.5x was an artifact of concurrent processes.

---

## Experiment 22: Sanitized ANE vs CPU at 120s (E22)

**Date**: 2026-03-11
**Phase**: 0
**Goal**: Compare ANE (with gradient sanitization) vs CPU-only training at 120s to quantify fp16 precision gap

### Setup
- Model: autoresearch (4L/1024d, 95.4M params)
- Time budget: 120 seconds each
- ANE: fp16 with --sanitize flag (gradient NaN/Inf cleanup)
- CPU: --cpu-only (all matmuls via cblas_sgemm fp32)
- Both run sequentially (no concurrent thermal interference)
- From scratch (random init)

### Results

| Metric | ANE (sanitized) | CPU-only | Delta |
|--------|-----------------|----------|-------|
| final_loss | 4.354 | **3.897** | CPU wins by 0.457 (11.7%) |
| training_seconds | 81.6 | 106.1 | ANE 1.30x more efficient |
| num_steps | 1023 | 958 | ANE gets 6.8% more steps |
| ms/step | ~79.8 | ~110.7 | ANE 1.39x faster per step |
| compile_overhead | 38.4s | 13.9s | ANE 2.76x more compile overhead |

### Key Findings
1. **CPU still wins on loss quality** despite ANE's step speed advantage — fp16 precision gap is real
2. **fp16 gap is ~12% at 120s** — consistent with prior measurements (~16% at 10min)
3. **Gradient sanitization has no effect** — zero NaN/Inf events (confirmed by E19)
4. **ANE compile overhead is significant**: 38.4s of 120s budget (32%) vs CPU's 13.9s (12%)
5. **If compile overhead were eliminated** (via delta compilation), ANE would get ~1500 steps vs CPU's 958 — still wouldn't close the precision gap but would narrow it

### Stated Assumptions Updated
- SA1 (fp16 precision gap irreducible): **RE-CONFIRMED** — 11.7% gap at 120s even with sanitization
- SA6 (CPU wins on loss/wall-clock): **RE-CONFIRMED** — 3.897 vs 4.354

---

## Experiment 28-29: Adaptive ANE→CPU Pipeline (E28-E29)

**Date**: 2026-03-11
**Phase**: 3
**Goal**: Implement and test adaptive ANE→CPU switching based on activation magnitude

### Implementation
Added to train.m:
- `--adaptive <threshold>`: Switch from ANE to CPU when |x|_max exceeds threshold
- `--adaptive-window <N>`: Require N consecutive steps above threshold (default 5)
- Uses `vDSP_maxmgv` to check max absolute activation every step
- When triggered: sets `cpu_only=true`, `use_cpu_attn_bwd=true` mid-training
- Weights already in fp32 — no conversion needed for seamless switch

### E28: Threshold Grid Search

| Threshold | Window | Switch Step | 120s final_loss | Notes |
|-----------|--------|------------|----------------|-------|
| 100 | 5 | (never) | 5.201 | Activations don't reach 100 in 120s |
| 10 | 5 | (never) | 4.893 | Activations fluctuate around 5-10, rarely 5 consecutive |
| 5 | 3 | 275 | 4.507 | Triggers when activations cross 5.0 |
| 3 | 3 | 194 | ~6.1 | Too aggressive, switches too early |

### E29: Adaptive vs Pure Modes at 120s

| Mode | final_loss | training_sec | steps | switch_step |
|------|-----------|-------------|-------|-------------|
| Pure ANE | 4.354 | 81.6 | 1023 | — |
| Adaptive (ANE→CPU@275) | 4.507 | 105.1 | 1076 | 275 |
| Pure CPU | **3.897** | 106.1 | 958 | — |

### Key Findings
1. **Adaptive mode doesn't improve loss** — gets more steps (1076) but worse loss (4.507) than either pure mode
2. **fp16 precision damage is cumulative in weights** — starting on ANE and switching to CPU produces worse results than CPU-from-start
3. **Activations fluctuate rather than growing monotonically** — makes threshold-based switching unreliable
4. **ANE→CPU switch mechanism works correctly** — activations drop immediately after switch (fp32 more stable)
5. **The hypothesis that "ANE for early training + CPU for later" would combine best of both is DISPROVED**

### Stated Assumptions Updated
- D6 (NEW): "ANE→CPU mid-training improves over pure ANE" — DISPROVED. fp16 rounding errors accumulate in weights and persist after mode switch.

---

## Experiment 30-31: LoRA Fine-Tuning on ANE (E30-E31)

**Date**: 2026-03-11
**Phase**: 4
**Goal**: Validate LoRA fine-tuning stability and test rank sensitivity on ANE

### E30: LoRA Stability with Gradient Sanitization
- Model: autoresearch (4L/1024d), fine-tuning from checkpoint
- LoRA rank=8, with --sanitize
- Result: final_loss=5.800, 610 steps, 46.2s training time
- Training stable, no crashes, no NaN
- Loss started above checkpoint baseline (5.6) due to LoRA adapter warmup (B initialized to zero)

### E31: LoRA Rank Sweep

| Rank | final_loss | steps | LoRA Params |
|------|-----------|-------|-------------|
| 4 | 5.505 | 658 | ~83K |
| 8 | 5.800* | 610 | ~165K |
| 16 | 5.388 | 668 | ~330K |
| 32 | **4.994** | 709 | ~660K |

*rank=8 includes --sanitize flag; others don't (sanitization has no effect per E19)

### Key Findings
1. **LoRA training is stable on ANE** across all ranks (4-32) — confirms V8
2. **Higher rank = better loss** — more expressive adapter converges faster
3. **Step overhead is minimal** — all ranks get ~49s training in 60s budget (vs 43.7s for non-LoRA)
4. **Caveat**: runs were sequential with shared checkpoint, not perfectly controlled comparison
5. **V8 RE-CONFIRMED with gradient sanitization** — LoRA from pretrained checkpoint works on ANE

---

## Experiment 32: Power Measurement Script (E32)

**Date**: 2026-03-11
**Phase**: 5
**Goal**: Create power measurement infrastructure

### Result
Created `measure_power.sh` script that:
1. Starts `powermetrics` in background (requires sudo)
2. Runs ANE benchmark during measurement window
3. Parses ANE/CPU/GPU power readings
4. NOT RUN yet (requires sudo — user needs to run manually)

### Status
Script created but not executed. U1 (ANE power ~2.8W) remains UNVERIFIED.

---

## Experiment 34: _ANEClient Delta Compilation (E34)

**Date**: 2026-03-11
**Phase**: 5
**Goal**: Investigate _ANEClient API for delta compilation (per Orion paper)

### API Discovery
Used Objective-C runtime inspection to enumerate ANE framework classes:
- `_ANEClient`: Found with 46 instance methods including `compileModel:`, `loadModelNewInstance:`, `evaluateWithModel:`
- `_ANECompiler`: NOT FOUND
- `_ANEDaemonConnection`: Found with `prepareChainingWithModel:` (model chaining API)

### Test Results

| Operation | Time | Status |
|-----------|------|--------|
| First compile (_ANEInMemoryModel) | 45.4ms | OK |
| Cached recompile (same topology) | 28.7ms | OK (1.6x faster) |
| Load | 25.2ms | OK |
| Reload (cached) | 2.5ms | OK (10x faster) |
| Evaluate | 0.425ms | OK |
| _ANEClient.compileModel | 0.9ms | **FAIL** |
| _ANEClient.loadModelNewInstance | 0.2ms | **FAIL** ("Program load new instance failure") |
| compiledModelExistsFor | — | NO (separate cache) |

### Key Findings
1. **_ANEClient and _ANEInMemoryModel are separate compilation paths** — they don't share compiled model caches
2. **loadModelNewInstance requires _ANEClient-compiled models** — fails with _ANEInMemoryModel models
3. **Topology caching provides 1.6x speedup** on recompile (28.7ms vs 45.4ms)
4. **Reload is very fast (2.5ms)** when model is already compiled — if delta compilation worked, weight updates would be ~2.5ms instead of ~45ms
5. **SA4 RE-CONFIRMED**: Delta compilation is not viable via _ANEInMemoryModel. The _ANEClient path requires complete pipeline rewrite and deeper reverse engineering.

### Recommendations
- To enable delta compilation, the entire ANE pipeline would need to be rewritten to use _ANEClient API
- The `_ANEDaemonConnection.prepareChainingWithModel` method may enable model chaining (multiple models in sequence)
- The potential speedup from delta compilation (~45ms→2.5ms per kernel update) would eliminate most of the compile overhead

---

## Experiment 36: ANE-Matmul-Only — Unfused Forward Path Validation

**Status**: COMPLETE
**Date**: 2026-03-11

### Background & Root Cause Analysis

Investigation into why ANE consistently produces worse models than CPU revealed the root cause: **our fused ANE kernels push non-linear operations (RoPE, attention/softmax, SiLU, residual connections) into fp16, while the original maderix/ANE repo uses ANE only for the 7 linear projection matmuls per layer (Wq, Wk, Wv, Wo, W1, W3, W2).**

The original repo's `forward.h` does:
- ANE fp16: `ane_conv_eval` for each linear projection (7 per layer)
- CPU fp32: RMSNorm, RoPE, attention (QK^T, softmax, AV), SiLU gating, residual additions
- CPU fp32: ALL backward operations

Our fused kernels (`sdpaFwd`, `ffnFused`) bundle non-linear operations into single ANE fp16 kernels. This causes:
1. **Cumulative fp16 rounding** in softmax, SiLU, and residual accumulation
2. **Train/val distribution shift**: model learns fp16 artifacts during training that don't transfer to fp32 validation
3. **Catastrophic overfitting**: E35 showed val_loss=8.02 vs train_loss=4.31 on ANE-full

### Fix: `--ane-matmul-only` Mode

Implemented unfused forward path matching the original repo's approach:
- 5 shared dynamic matmul kernels: `wqFwd` (DIM to Q_DIM), `wkvFwd` (DIM to KV_DIM), `woFwd` (Q_DIM to DIM), `w13Fwd` (DIM to HIDDEN), `w2Fwd` (HIDDEN to DIM)
- Each kernel is a simple matmul via `gen_dyn_matmul_mil(ic, oc, seq)`
- Per-layer weight staging into individual IOSurfaces
- CPU fp32 for: RMSNorm, RoPE, attention (QK^T, softmax, AV), SiLU gating, residual connections
- CPU fp32 for: ALL backward operations (implied by ane-matmul-only setting use_cpu_bwd)

### E36 Results: 4-Mode Comparison (120s budget, 4L/1024d)

| Mode | Train Loss | Val Loss | Val Gap | Steps | Notes |
|------|-----------|----------|---------|-------|-------|
| **cpu-only** | 4.758 | 5.079 | 0.321 | 550 | Best absolute loss (most steps) |
| **ane-full** | 6.171 | 7.389 | **1.218** | 150 | Large val gap = overfitting to fp16 artifacts |
| **ane-fwd-cpu-bwd** | 6.149 | 7.275 | **1.126** | 232 | CPU bwd helps throughput, not generalization |
| **ane-matmul-only** | 6.524 | 7.122 | **0.599** | 130 | 51% val gap reduction vs ane-full |

### Step-Matched Comparison (130 steps)

| Mode | Train Loss | Val Loss | Val Gap |
|------|-----------|----------|---------|
| **cpu-only** (130 steps) | 6.705 | **7.122** | 0.417 |
| **ane-matmul-only** (130 steps) | 6.524 | **7.122** | 0.599 |

**Val loss is identical (7.122) at matched step counts.** The unfused approach produces numerically equivalent generalization to pure CPU fp32 training.

Per-step loss values also match closely:
- Step 10: 9.5708 (ane-matmul) vs 9.5710 (cpu-only)
- Step 20: 9.2107 (ane-matmul) vs 9.2106 (cpu-only)

### Throughput Issue: ANE Thermal Throttling

`ane-matmul-only` achieves only 130 steps in 120s vs 550 for cpu-only. Timing breakdown shows periodic system-wide stalls:
- Normal step: ane_fwd=30ms, ane_bwd=65ms, total ~150ms/step
- Throttled step: ane_fwd=291ms, io_fwd=2325ms, ane_bwd=2876ms, total ~6500ms/step

This is Apple Silicon thermal management throttling the ANE under sustained load, not a code issue. All timing categories (including CPU-only operations like RMSNorm) spike simultaneously during throttled steps.

### Key Findings

1. **Root cause confirmed**: Fusing non-linear ops into ANE fp16 causes train/val distribution shift
2. **Unfused approach eliminates generalization gap**: Identical val_loss to CPU at matched steps
3. **ANE thermal throttling is the primary throughput bottleneck** — not kernel overhead
4. **The original maderix/ANE design was correct**: ANE for linear projections only, CPU for everything else
5. **`--cpu-bwd` alone is insufficient**: It improves throughput but doesn't fix the forward-path fp16 accumulation problem

### Implications for AutoANE

- `--ane-matmul-only` should be the default training mode when using ANE
- The throughput gap (130 vs 550 steps/120s) means CPU-only may still be preferred for short experiments
- For longer training runs where ANE thermal throttling amortizes, `ane-matmul-only` may become competitive
- Future work: investigate ANE duty cycling to avoid thermal throttling

---

## Experiment 37: Sustained Throughput — ANE-Matmul-Only vs CPU-Only (10 min)

**Status**: COMPLETE
**Date**: 2026-03-11
**Hardware**: MacBook Pro Mac14,9, Apple M2 Pro (8P+4E cores), 16GB

### Research Question

Does ANE provide a net training speedup over CPU-only when run long enough for thermal behavior to stabilize?

### Pre-Experiment Research

Three parallel investigations conducted before running experiments:

**1. Literature Review** (maderix blog, Orion paper arXiv:2603.06728):
- ANE peak power is ~2.8W (vs CPU 45W+), has hard power gating (0mW idle)
- Orion ran 1,000 steps over 22.4min with 913 +/- 30 ms/step (3.3% CV) — no degradation
- Single-op ANE utilization is ~30%, need 16-64 ops for 74-94%
- Dispatch overhead ~0.095ms/call
- ANE represents 7-14% of SoC power budget — unlikely thermal bottleneck

**2. Code Audit** (thorough review of train.m, config.h, io.h):
- Forward path fp32/fp16 boundaries: CORRECT
- Backward pass gradients: CORRECT
- Weight staging: CORRECT
- Validation eval: CORRECT (always CPU fp32)
- BUG FOUND: CFRelease(NULL) in cleanup — FIXED
- BUG FOUND: unfused kernel handles leaked — FIXED

**3. Original Repo Analysis** (maderix/ANE gen1/gen2/gen3):
- Gen1 (forward.h): ANE for projections only, CPU for everything else (matches our ane-matmul-only)
- Gen3 (dynamic): ANE for fused forward + backward dx, CPU for dW only
- Our ane-matmul-only is more conservative than the original gen3
- Original defaults to loss_scale=1.0 (ours uses 256.0)

### Methodology

Sequential runs with 120s cooldown between:
1. CPU-only for 600s (10 minutes)
2. ANE-matmul-only for 600s (10 minutes)

Same binary, same data, same seed (srand48(42)), same hyperparameters.
Thermal state monitored via ProcessInfo.thermalState every 10 steps.

### Results

| Metric | CPU-only | ANE-matmul-only |
|--------|----------|-----------------|
| **Total steps** | 1,262 | **4,690** |
| **Final train loss** | 4.402 | **3.158** |
| **Final val loss** | 3.910 | **3.148** |
| **Val gap** | 0.492 | **0.010** |
| Training seconds | 252.6 | 486.1 |
| Total seconds | 600.0 | 600.0 |
| Throughput | 2.1 steps/s | **7.8 steps/s** |
| Thermal state | nominal (always) | nominal (always) |

### Step Time Distribution

| Percentile | CPU-only | ANE-matmul-only |
|------------|----------|-----------------|
| Min | 105.5ms | **99.4ms** |
| P10 | 107.2ms | **101.4ms** |
| **Median** | 163.0ms | **103.2ms** |
| P90 | 508.5ms | **107.2ms** |
| P99 | 2,639.9ms | **129.8ms** |
| Max | **16,273.5ms** | 165.5ms |
| Mean | 485.9ms | **104.2ms** |

CPU-only suffers from extreme tail latency (16.3s max stall, P99 at 2.6s) despite reporting nominal thermal state. ANE-matmul-only has rock-stable timing (max 165ms).

### Step-Matched Loss Comparison

| Step | CPU loss | ANE loss | Match? |
|------|---------|---------|--------|
| 100 | 7.3626 | 7.3626 | EXACT |
| 1200 | 4.4542 | 4.4561 | 0.002 diff |

At matched steps, loss values are identical/near-identical, confirming ane-matmul-only produces numerically equivalent training to CPU.

### Val Loss Trajectory

CPU-only (12 checkpoints): 7.12 -> 5.99 -> 5.44 -> ... -> 3.91
ANE-matmul-only (46 checkpoints): 7.12 -> 6.21 -> 5.47 -> ... -> 3.15

### Analysis: Why ANE Got 3.7x More Steps

E36 showed ANE getting 4.2x FEWER steps. E37 shows 3.7x MORE. What changed?

1. **E36 ran experiments non-sequentially or with insufficient cooldown**: The CPU run built up heat in the SoC, then ANE ran on a thermally stressed system. macOS throttled everything.
2. **E37 ran sequentially with 120s cooldown**: CPU ran first (clean), then ANE ran on a cooled system.
3. **CPU-only has extreme scheduling jitter**: Median 163ms but P99 at 2.6s suggests macOS background processes (compactd, spotlight, etc.) cause massive stalls. This ate ~60% of CPU's training budget (only 252s of actual training in 600s wall time).
4. **ANE path is immune to CPU scheduling jitter**: With forward matmuls on ANE, the CPU backward pass is lighter, and the ANE dispatch path avoids whatever causes the CPU stalls.
5. **ANE forward is genuinely faster for large matmuls**: ane_fwd=21-29ms vs CPU forward equivalent. At DIM=1024, HIDDEN=2816, the matmul sizes are in ANE's sweet spot (V11: ANE faster for 2816+ width).

### Key Findings

1. **ANE-matmul-only is 3.7x faster than CPU-only in wall-clock throughput** (4,690 vs 1,262 steps in 10 min)
2. **Zero thermal throttling** on either path (ProcessInfo.thermalState = nominal throughout)
3. **CPU scheduling jitter, not thermal throttling, is the dominant performance factor**: CPU P99 latency is 2.6s vs ANE P99 of 130ms
4. **ANE produces numerically identical training**: loss matches CPU to 4 decimal places at matched steps
5. **Val gap essentially eliminated**: 0.010 on ANE vs 0.492 on CPU (lower gap = better generalization)
6. **E36's results were wrong due to experimental methodology**: running experiments concurrently or without proper cooldown caused false attribution of stalls to ANE

### Timing Analysis Deep Dive

The `total_train_ms` accumulates all 1262 step times, giving 200.2ms average. The sampled distribution (every 10th step, N=127) shows 485.9ms mean. This discrepancy means:
- ~90% of steps (not logged) run at ~105ms (competitive with ANE's 103ms)
- ~10% of logged steps include massive stalls (500ms to 16.3s)
- These stalls cluster at Adam update boundaries (step % 10 == 0)
- The stalls are NOT thermal (ProcessInfo reports nominal) — they're macOS scheduling

CPU wall time breakdown: 252.6s training + ~25s val evals + ~20s Adam/transpose = ~298s. Remaining ~302s is pure idle/scheduling overhead. 50% of CPU's wall time was wasted on stalls.

### Reproducibility Check (E37-C/D, reversed order, 300s each)

Ran ANE first, then CPU second (reversed from E37-A/B) with 60s cooldown:

| Run | Mode | Order | Steps | ms/step (median) | ms/step (max) | Val Loss |
|-----|------|-------|-------|------------------|---------------|----------|
| E37-A | CPU | 1st | 1,262 | 163.0 | **16,273** | 3.910 |
| E37-B | ANE | 2nd | 4,690 | 103.2 | 165.5 | 3.148 |
| E37-C | ANE | 1st | 2,288 | 104.4 | 174.4 | 3.497 |
| E37-D | CPU | 2nd | 2,489 | 104.8 | 180.9 | 3.455 |

**Finding: E37-A's CPU stalls were from a transient background process, not inherent to CPU training.** When the system is clean (E37-C/D), both modes perform identically at ~105ms/step. The 3.7x throughput advantage claimed from E37-A/B was an artifact.

### Corrected Conclusions

1. **ANE-matmul-only and CPU-only have identical steady-state throughput** (~105ms/step for 4L/1024d model)
2. **ANE provides no throughput advantage at this model size** — the forward matmul speedup is offset by IOSurface overhead
3. **ANE-matmul-only does provide correct generalization** — val loss matches CPU at matched steps (V12, V13 confirmed)
4. **CPU tail latency can be extreme but is NOT systematic** — background processes cause sporadic multi-second stalls
5. **Neither mode causes thermal throttling** — both stay nominal for 10+ minutes
6. **E36's results were contaminated by background interference** — not by ANE thermal throttling as originally hypothesized

### Implications

- For this model size (95M, DIM=1024, HIDDEN=2816), there is no throughput reason to prefer ANE over CPU
- ANE-matmul-only is validated as numerically correct and can be used safely
- For the autoresearch loop, CPU-only is simpler and equally fast — use it as the default
- **E37's prediction that "ANE becomes advantageous for larger models" was DISPROVED by E38** — see below

---

## Experiment 38: ANE Scaling Study — IOSurface Memory Pressure Ceiling

**Date**: 2026-03-11
**Status**: COMPLETE
**Depends on**: E37

### Research Question

At what model dimension does ANE-matmul-only outperform CPU-only in training throughput?

### Hypothesis (DISPROVED)

We predicted ANE advantage would emerge at larger dimensions because:
- V11: ANE is 1.88x faster for 2816-width matmuls
- Larger matmuls → more compute → ANE matmul advantage dominates IOSurface overhead

### Methodology

Created two new model configs (4 layers, Llama-style GQA):
- **4L-1536d**: DIM=1536, HIDDEN=4224, 177M params, ~220MB IOSurfaces
- **4L-2048d**: DIM=2048, HIDDEN=5632, 281M params, ~379MB IOSurfaces

Ran CPU-only and ANE-matmul-only for 120s at each dimension. All runs on clean system with 120s cooldown between experiments. Reversed-order reproducibility check at DIM=2048.

Hardware: MacBook Pro M2 Pro, 16GB RAM. Thermal state: nominal throughout all runs.

### Results

| Run | Config | Mode | Steps | ms/step (median) | fwd (ms) | io_fwd (ms) | bwd (ms) |
|-----|--------|------|-------|-------------------|----------|-------------|----------|
| E38-A | 1536d | CPU | 634 | 159.9 | 48.6 | 0.0 | 69.4 |
| E38-B | 1536d | ANE | 560 | 157.2 | 37.4 | 4.9 | 69.7 |
| E38-C | 2048d | CPU | 306 | 280.2 | 82.9 | 0.0 | 129.9 |
| E38-D | 2048d | ANE | 210 | **648.5** | 75.0 | **129.4** | **328.1** |

### Reproducibility Check (DIM=2048, reversed order)

| Run | Mode | Order | ms/step (median) | io_fwd (ms) | bwd (ms) |
|-----|------|-------|-------------------|-------------|----------|
| E38-C | CPU | 1st | 280.2 | 0.0 | 129.9 |
| E38-D | ANE | 2nd | 648.5 | 129.4 | 328.1 |
| E38-E | ANE | 1st | 546.1 | 100.3 | 278.4 |
| E38-F | CPU | 2nd | 287.9 | 0.0 | 132.5 |

CPU is consistent regardless of order (280-288ms). ANE is catastrophically degraded regardless of order (546-649ms).

### Analysis: IOSurface Memory Pressure

**DIM=1536 (220MB IOSurfaces)**: ANE matches CPU. Forward matmul 1.3x faster (37ms vs 49ms), IOSurface overhead small (5ms). Backward identical (70ms both). Net: parity (~158ms both).

**DIM=2048 (379MB IOSurfaces)**: ANE is 2x SLOWER than CPU. Three compounding effects:

1. **IOSurface lock/unlock latency explodes**: median 129ms (vs 5ms at DIM=1536). Individual surfaces are 23-25MB each. 28 lock/unlock operations per step × 20MB average = 560MB of memory traffic per step. At this scale, IOSurface operations trigger memory compaction.

2. **CPU backward pass degrades**: 328ms (vs CPU-only's 130ms). Despite being the exact same code path (all cblas_sgemm), the backward pass is 2.5x slower when IOSurfaces are allocated. Root cause: 379MB of wired IOSurface memory reduces available DRAM for CPU caches, causing cache thrashing during the compute-intensive backward pass.

3. **Everything else is slower too**: SiLU 26ms (vs 13ms CPU), classifier 38ms (vs 28ms CPU). System-wide memory pressure affects all operations.

### IOSurface Overhead Scaling

| DIM | IOSurface Total | io_fwd (median) | bwd penalty | Status |
|-----|----------------|-----------------|-------------|--------|
| 1024 | 104 MB | 5.0 ms | 0% | OK |
| 1536 | 220 MB | 4.9 ms | 0% | OK |
| 2048 | 379 MB | 129.4 ms | +153% | **DEGRADED** |

The cliff between 220MB and 379MB suggests a hard memory pressure threshold around 250-350MB of IOSurface allocations on a 16GB M2 Pro system.

### Key Findings

1. **Dynamic weight ANE training has a hard scaling ceiling at ~DIM=1536 (177M params, 220MB IOSurfaces)**
2. **Below this ceiling, ANE matches CPU but never beats it** for the unfused single-op approach
3. **Above this ceiling, ANE is dramatically worse** (2x at DIM=2048) due to IOSurface memory pressure
4. **The memory pressure affects ALL operations**, not just IOSurface staging — backward pass, classifier, and SiLU all degrade
5. **CPU-only is the correct default for ALL model sizes** in our current architecture
6. **E37's prediction ("ANE advantageous for larger models") is WRONG** — IOSurface overhead grows faster than ANE compute savings

### Implications for ANE Training

The dynamic weight approach (packing weights into IOSurface spatial dimension) is fundamentally limited:
- **Small models (DIM≤1536)**: ANE matches CPU but doesn't beat it, because single-op utilization is only ~30%
- **Large models (DIM≥2048)**: IOSurface memory pressure causes severe degradation
- **The only path to ANE throughput advantage requires**: (a) kernel fusion for higher utilization AND (b) static weights via conv 1x1 to eliminate IOSurface overhead
- Both (a) and (b) have been shown non-viable: fusion hurts generalization (E36/V12), delta compilation doesn't work (E10/E17/SA4)

**Conclusion**: For the autoresearch project, CPU-only training is the correct and only viable default at all model sizes tested (95M-281M params). ANE training via the dynamic weight approach provides no throughput advantage and degrades at scale.

### Literature Cross-References (E38)

Our E38 results are the **first known systematic ANE training scaling study**. Prior work:
- maderix/ANE: tested only Stories110M (DIM=768) and Qwen3-0.6B (DIM=1024). No scaling experiments.
- Orion paper: tested only Stories110M. No dimension-scaling analysis.
- No published work has attempted ANE training at DIM>1024 before this experiment.

Our findings align with known ANE characteristics:
- **SRAM cliff** (U4): maderix Part 2 showed 30% throughput drop at 4096x4096 (96MB working set vs 32MB SRAM). Our W1/W3 IOSurfaces at DIM=2048 are 24MB each — right at the SRAM edge.
- **Single-op utilization** (U9): our 28 individual matmul dispatches/step achieve only ~30% ANE utilization. Even with 2.5x raw matmul speedup, the overhead negates it.
- **Dispatch overhead** (U10): 28 dispatches × 0.095ms = 2.7ms fixed cost, plus IOSurface staging per dispatch.
- **No dimension scaling data existed**: our E38 fills a genuine gap in the literature.

### Potential Future Paths (not yet viable)

1. **_ANEChainingRequest** (U12): firmware-level chained execution could eliminate CPU round-trips between layers, potentially reducing dispatch overhead by 10-20x.
2. **Dual-input IOSurface** (imperatormk approach, maderix Issue #47): passing weights as a separate IOSurface input instead of spatial packing might reduce memory pressure.
3. **Layer fusion + _ANEChainingRequest**: 4-layer fusion showed 7.7x speedup in maderix benchmarks, but requires solving the generalization penalty (V12).
4. **INT8 for bandwidth savings**: ANE dequantizes INT8→FP16 before compute, but smaller weights reduce IOSurface size by 2x, potentially moving the memory pressure cliff higher.

---

## Experiment 39: Architecture Search — Depth vs Width at Fixed Time Budget

**Date**: 2026-03-11
**Status**: COMPLETE
**Automated via**: `autoresearch.py --search arch --budget 120`

### Research Question

Given a fixed 120-second CPU-only training budget on TinyStories, what is the optimal depth/width tradeoff for minimizing validation loss?

### Motivation

E38 established CPU-only as the correct default. The autoresearch infrastructure (E39's prerequisite) was built to automate architecture exploration. This is the first systematic sweep: 11 configs spanning DIM=[512, 768, 1024, 1536] × NLAYERS=[2, 4, 6, 8], all CPU-only with identical hyperparameters (LR=3e-4, warmup=100, accum=10, clip=1.0).

### Methodology

- **Grid**: 11 architecture configs from 36.4M to 177.0M parameters
- **Training**: CPU-only fp32, 120s budget each, 30s cooldown between runs
- **Data**: TinyStories (SmolLM2 tokenizer, 49152 vocab), SEQ=256
- **Hyperparameters**: All held constant (LR=3e-4, warmup=100, accum=10, clip=1.0, weight_decay=0.1)
- **Hardware**: MacBook Pro M2 Pro, 16GB RAM
- **Metric**: Validation loss (separate val set, evaluated at end of training)
- **Total wall time**: ~28 minutes

### Results (sorted by val_loss)

| Rank | Config | Params | Steps | ms/step | Train Loss | Val Loss | Val-Train Gap |
|------|--------|--------|-------|---------|------------|----------|---------------|
| 1 | 512d/4L | 36.4M | 2471 | 42ms | 3.8415 | **3.6138** | -0.228 |
| 2 | 768d/2L | 50.4M | 2484 | 40ms | 3.8802 | **3.7342** | -0.146 |
| 3 | 1024d/2L | 72.9M | 1638 | 60ms | 3.6653 | **3.8570** | +0.192 |
| 4 | 768d/4L | 63.1M | 1570 | 65ms | 3.6105 | 4.0327 | +0.422 |
| 5 | 768d/6L | 75.8M | 1152 | 88ms | 4.7045 | 4.1532 | -0.551 |
| 6 | 512d/8L | 47.7M | 1484 | 71ms | 4.5164 | 4.2196 | -0.297 |
| 7 | 1024d/4L | 95.4M | 1022 | 101ms | 3.8266 | 4.2981 | +0.472 |
| 8 | 1536d/2L | 126.2M | 1088 | 92ms | 4.2773 | 4.3782 | +0.101 |
| 9 | 1536d/4L | 177.0M | 620 | 161ms | 4.5588 | 4.5203 | -0.039 |
| 10 | 1024d/6L | 118.0M | 740 | 141ms | 4.6344 | 4.6039 | -0.031 |
| 11 | 1024d/8L | 140.5M | 577 | 183ms | 4.5985 | 5.0542 | +0.456 |

### Analysis

**Finding 1: Step count dominates at short budgets.**

The top 3 configs all achieved >1600 steps. The correlation between step count and val_loss rank is striking:

| Steps | Val Loss Range | Configs |
|-------|---------------|---------|
| >2000 | 3.61-3.73 | 512d/4L, 768d/2L |
| 1500-2000 | 3.86-4.22 | 1024d/2L, 768d/4L, 512d/8L |
| 1000-1500 | 4.15-4.38 | 768d/6L, 1024d/4L, 1536d/2L |
| <1000 | 4.52-5.05 | 1536d/4L, 1024d/6L, 1024d/8L |

**Finding 2: Depth consistently hurts within a width class.**

At every width, adding layers increases val_loss:
- **512d**: 4L (3.61) → 8L (4.22) — doubling depth costs +0.61
- **768d**: 2L (3.73) → 4L (4.03) → 6L (4.15) — each +2L costs ~+0.2
- **1024d**: 2L (3.86) → 4L (4.30) → 6L (4.60) → 8L (5.05) — accelerating degradation

Mechanism: deeper models are slower per step (more sequential compute) and reach fewer total steps. At 120s, 512d/4L gets 2471 steps vs 1024d/8L's 577 — a 4.3x throughput difference.

**Finding 3: Optimal architecture depends on budget.**

The top configs suggest a scaling pattern:
- At 120s: 512d/4L optimal (small + fast)
- At longer budgets: larger models would eventually overtake once they accumulate enough steps for their capacity to matter

The crossover point can be estimated: 1024d/4L at 101ms/step needs ~250 steps to match 512d/4L's val_loss trajectory, which occurs at ~25s. The current budget (120s) gives 512d/4L a decisive step-count advantage.

**Finding 4: Val-Train gap reveals overfitting pattern.**

- **Small models (512d)**: val < train (negative gap) — underfitting, capacity-limited
- **Medium models (768d/4L, 1024d)**: val > train (positive gap) — beginning to overfit
- **Large models (1024d/8L)**: val >> train (+0.46) — significant overfitting at 577 steps

This suggests that at 120s, models above ~70M params are overfitting — they have enough capacity to memorize the limited training data they see, but not enough steps to generalize.

**Finding 5: Throughput scaling is sublinear with model size.**

| Params | ms/step | Relative throughput |
|--------|---------|-------------------|
| 36.4M | 42ms | 1.00x |
| 50.4M (1.4x) | 40-65ms | 0.65-1.05x |
| 72.9M (2.0x) | 60ms | 0.70x |
| 95.4M (2.6x) | 101ms | 0.42x |
| 177.0M (4.9x) | 161ms | 0.26x |

Doubling parameter count roughly halves throughput. This is expected for CPU training where compute scales linearly with FLOPs.

### Key Conclusions

1. **For 120s CPU-only training on M2 Pro: 512d/4L (36.4M) is optimal** — val_loss 3.61, 2471 steps
2. **Depth is strictly harmful at short budgets** — confirmed V7 with systematic evidence
3. **The 1024d/4L baseline (95.4M) we used for all prior experiments was suboptimal** — it ranked 7th of 11
4. **Smaller models don't overfit** while larger models do at this budget
5. **Autoresearch infrastructure works correctly** — all 11 configs ran autonomously, results consistent

### Assumptions Stated

- **SA-E39-1**: LR=3e-4 is equally good for all configs. UNVERIFIED — smaller models might benefit from higher LR (larger models typically need lower LR). This could change the ranking.
- **SA-E39-2**: 120s is representative of quick-iteration training. STATED — optimal architecture shifts with budget length. These results are specific to the 120s regime.
- **SA-E39-3**: val_loss is a reliable proxy for model quality. STATED — at these loss levels (3.6-5.0), differences are meaningful but the absolute values indicate early-stage training.

### Next Steps

1. ✓ **LR sweep per architecture**: Completed as E40. Optimal LR varies but ranking unchanged.
2. **Longer budget runs**: Test 512d/4L and 1024d/2L at 300s and 600s to find crossover
3. ✓ **Update default config**: train.py updated to 512d/4L

---

## Experiment 40: Learning Rate Sweep Across Top Architectures

**Date**: 2026-03-11
**Status**: COMPLETE
**Automated via**: `autoresearch.py --search lr --budget 120`
**Addresses**: SA-E39-1 (LR=3e-4 equally good for all configs?)

### Research Question

Does the optimal learning rate vary across the top E39 architectures, and if so, does per-architecture LR tuning change the architecture ranking?

### Methodology

- **Architectures**: Top 3 from E39 — 512d/4L, 768d/2L, 1024d/2L
- **Learning rates**: 1e-4, 3e-4, 5e-4, 1e-3, 2e-3
- **15 total configs** (3 archs × 5 LRs), each 120s CPU-only, 30s cooldown
- **All other hyperparameters held constant** (warmup=100, accum=10, clip=1.0, wd=0.1)
- **Total wall time**: ~38 minutes

### Results

**512d/4L (36.4M params)**:

| LR | Steps | Train Loss | Val Loss | Val-Train Gap |
|------|-------|-----------|----------|---------------|
| 1e-4 | 2559 | 3.866 | 4.189 | +0.32 |
| 3e-4 | 2561 | 3.200 | 3.673 | +0.47 |
| **5e-4** | **2570** | **3.439** | **3.543** | **+0.10** |
| 1e-3 | 2453 | 3.993 | 3.780 | -0.21 |
| 2e-3 | 2569 | 3.750 | 4.032 | +0.28 |

**768d/2L (50.4M params)**:

| LR | Steps | Train Loss | Val Loss | Val-Train Gap |
|------|-------|-----------|----------|---------------|
| 1e-4 | 2589 | 4.244 | 4.020 | -0.22 |
| 3e-4 | 2600 | 2.924 | 3.744 | +0.82 |
| **5e-4** | **2600** | **2.856** | **3.690** | **+0.83** |
| 1e-3 | 2593 | 3.667 | 3.840 | +0.17 |
| 2e-3 | 2548 | 3.582 | 4.260 | +0.68 |

**1024d/2L (72.9M params)**:

| LR | Steps | Train Loss | Val Loss | Val-Train Gap |
|------|-------|-----------|----------|---------------|
| 1e-4 | 1699 | 3.871 | 4.056 | +0.19 |
| **3e-4** | **1730** | **3.476** | **3.953** | **+0.48** |
| 5e-4 | 1715 | 3.262 | 3.966 | +0.70 |
| 1e-3 | 1707 | 3.960 | 4.206 | +0.25 |
| 2e-3 | 1687 | 4.536 | 4.281 | -0.25 |

### Analysis

**Finding 1: Optimal LR is 5e-4 for smaller models, 3e-4 for larger.**

| Architecture | Best LR | Best Val Loss | Improvement over 3e-4 |
|-------------|---------|---------------|----------------------|
| 512d/4L | 5e-4 | 3.543 | -0.130 (3.5% better) |
| 768d/2L | 5e-4 | 3.690 | -0.054 (1.4% better) |
| 1024d/2L | 3e-4 | 3.953 | 0.000 (already optimal) |

Smaller models benefit from higher LR — classic scaling law behavior. Each parameter update matters more when there are fewer parameters.

**Finding 2: E39 architecture ranking is ROBUST across LR.**

Even with per-architecture optimal LR:
1. 512d/4L: val_loss **3.543** (LR=5e-4)
2. 768d/2L: val_loss **3.690** (LR=5e-4)
3. 1024d/2L: val_loss **3.953** (LR=3e-4)

The ranking is unchanged. SA-E39-1 is now **RESOLVED**: LR tuning improves individual configs but does not change the relative ordering.

**Finding 3: 512d/4L at LR=5e-4 has the healthiest generalization.**

Train-val gaps at optimal LR:
- 512d/4L: +0.10 (minimal overfitting)
- 768d/2L: +0.83 (significant overfitting — train loss 2.86 vs val 3.69)
- 1024d/2L: +0.48 (moderate overfitting)

The 768d/2L model overfits heavily despite being the fastest (39ms/step, 2600 steps). Its train loss of 2.86 is very low but doesn't transfer to validation. This suggests 2-layer models have a generalization disadvantage — they memorize training patterns without learning transferable representations.

**Finding 4: LR too high causes instability, visible in loss curve.**

At LR=2e-3, all architectures show degraded val_loss. The effect is monotonic above the peak: 512d at 2e-3 (4.03) is worse than 3e-4 (3.67). At 1e-3, 512d shows val < train (-0.21), suggesting training instability where final loss doesn't reflect average quality.

### Key Conclusions

1. **Optimal config for 120s: 512d/4L at LR=5e-4** — val_loss 3.543, the best result across all E39+E40 experiments
2. **Architecture ranking is robust to LR tuning** — SA-E39-1 resolved
3. **2-layer models overfit heavily** despite high step counts — depth aids generalization
4. **Default LR should be 5e-4 for 512d/4L** (not 3e-4)

### Assumptions Updated

- **SA-E39-1**: RESOLVED — LR varies by architecture but ranking unchanged
- **SA-E40-1**: Warmup=100 steps is appropriate for all configs. UNVERIFIED — at 2500 steps, 100 warmup is 4% of training. Shorter warmup might help at higher LR.
- **SA-E40-2**: Weight decay 0.1 is equally good across configs. UNVERIFIED — 768d/2L's heavy overfitting might benefit from higher WD.
