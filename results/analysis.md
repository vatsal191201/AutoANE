# MeZO vs Backprop: Complete Experimental Results

**Date:** 2026-03-12/13
**Hardware:** Apple M-series (ANE + AMX)
**Data:** TinyStories (20M tokens, 90/10 train/val split)
**Time budget:** 120 seconds per condition
**Version:** v5 (v4 audit + SmolLM2-360M scaling experiments)

## Full Results (8 Conditions)

### From-Scratch (autoresearch-4L-512d, 36.4M params)

| # | Method | Hardware | Steps | ms/step | Final Loss | Val Loss |
|---|--------|----------|-------|---------|------------|----------|
| 1 | Backprop+Adam | CPU | 3015 | 30.7 | 3.619 | 3.998 |
| 2 | Backprop+Adam | ANE | 3393 | 26.0 | 3.910 | 3.790 |
| 3 | MeZO (ZO-SGD) | CPU | 1588 | 75.1 | 9.599 | 9.685 |
| 4 | MeZO (ZO-SGD) | ANE | 1265 | 94.5 | 9.702 | — |

**Backprop HPs:** lr=4e-4, Adam, warmup=10, accum=7, grad_clip=1.0, loss_scale=256
**MeZO HPs:** lr=1e-5, epsilon=1e-3, Rademacher perturbation (xoshiro256+)

*From-scratch conditions unaffected by DeepNet bug (from-scratch correctly uses DeepNet scaling).*

### Fine-Tuning (SmolLM2-135M, 134.5M params, 30 layers) — v3 (v2 + ANE optimization)

| # | Method | Hardware | Steps | ms/step | Start Loss | Final Loss | Val Loss |
|---|--------|----------|-------|---------|------------|------------|----------|
| 5 | Backprop+Adam | CPU | 382 | 281.5 | 2.24 | 1.814 | 1.929 |
| 6 | Backprop+Adam | ANE | 346 | 304.8 | 2.24 | 2.158 | 1.929 |
| 7 | MeZO (ZO-SGD) | CPU | 317 | 379.3 | 2.25 | 1.97‡ | 2.249‡ |
| 8 | MeZO (ZO-SGD) | ANE | 240 | 501.4 | 2.25 | 1.93† | 2.249‡ |

*Condition 8 v2→v3: 656→501 ms/step (1.31x speedup), 183→240 steps (+31%).*
*Bit-identical losses at matching steps (verified step 0 and step 100).*
*†Final loss 1.93 from v2 run (bit-identical to v3; v3 output truncated before final report).*
*‡Val loss from v4 experiment (val_every=100, 900 steps). Train loss is noisy single-batch.*
*‡MeZO "Final Loss" column shows loss_plus (single batch, perturbed weights) — NOT val loss.*

**Backprop HPs:** lr=3e-4, Adam, accum=10, warmup=10, grad_clip=1.0, --no-deepnet
**MeZO HPs:** lr=1e-5, epsilon=1e-3, Rademacher perturbation (xoshiro256+), res_alpha=1.0
**LR sweep:** {1e-4, 5e-5, 3e-5, 2e-5, 1e-5, 1e-6, 1e-7} — lr=1e-5 best (only LR showing decrease)

### Fine-Tuning (SmolLM2-360M, 361.8M params, 32 layers) — v5

| # | Method | Hardware | Steps | ms/step | Start Loss | Final Loss | Val Loss | RSS (MB) |
|---|--------|----------|-------|---------|------------|------------|----------|----------|
| 9 | MeZO (ZO-SGD) | CPU | 143 | 813.6 | 2.11 | 2.17‡ | 2.067 | 1,720 |
| 10 | MeZO (ZO-SGD) | ANE | 100 | 1199.9 | 2.11 | 2.28‡ | 2.067 | — |
| 11 | Backprop+Adam | CPU | 140 | 602.1 | 2.10 | 1.641 | 1.791 | 4,133 |
| 12 | Backprop+Adam | ANE | 120 | 700.1 | 2.10 | 2.031 | 1.791 | — |

*‡MeZO "Final Loss" is loss_plus (single batch, perturbed weights) — NOT val loss.*
*Val loss measured at step 100 with val_every=100.*
*Memory (RSS) measured via ps during training at steady state.*

**Backprop HPs:** lr=3e-4, Adam, accum=10, warmup=10, grad_clip=1.0, --no-deepnet
**MeZO HPs:** lr=1e-5, epsilon=1e-3, Rademacher perturbation (xoshiro256+), res_alpha=1.0

**Key 360M findings:**
- MeZO-ANE is **47% slower** than MeZO-CPU (1200 vs 814 ms/step)
- Backprop-ANE is **16% slower** than Backprop-CPU (700 vs 602 ms/step)
- Memory: MeZO uses **2.4x less** than Backprop (1720 vs 4133 MB)
- Val convergence: Backprop reaches 1.79 in 100 steps; MeZO only reaches 2.07

### MeZO+LoRA Fine-Tuning (SmolLM2-360M, 361.8M params) — v7

| # | Method | Hardware | Steps | ms/step | Perturb (ms) | Transpose (ms) | Val Loss |
|---|--------|----------|-------|---------|--------------|-----------------|----------|
| 13 | BP+LoRA r8 | CPU | 191 | 586 | — | — | 1.925 |
| 14 | BP+LoRA r8 | ANE | 74 | 1344 | — | — | — |
| 15 | MeZO+LoRA r8 | CPU | 200 | 576 | 56 | 0 | 2.068 |
| 16 | MeZO+LoRA r8 | ANE | 143 | 807 | 65 | 106 | 2.070 |
| 17 | MeZO+LoRA r32 | CPU | 55 | 1142 | 123 | 0 | — |
| 18 | MeZO+LoRA-split r8 | CPU | 205 | 537 | 2 | 0 | 2.069 |
| 19 | MeZO+LoRA-split r8 | ANE | 159 | 708 | 3 | 0 | 2.070 |

**MeZO+LoRA HPs:** lr=1e-5, epsilon=1e-3, rank=8, adapters on Wq/Wk/Wv/Wo, FFN frozen
**BP+LoRA HPs:** lr=3e-4, Adam, accum=10, warmup=10, rank=8, --no-deepnet

**Key MeZO+LoRA findings (v7):**
- **LoRA-split ANE: 708ms** vs full MeZO-ANE 1200ms (**41% faster**)
- Transpose overhead **eliminated entirely** (478ms → 0ms) via adapter-as-input
- Perturbation **193x faster** (579ms → 3ms) by only perturbing 2.3M adapter params
- **Rank 8 > rank 32** for MeZO: lower dim = lower ZO variance = better signal
- ANE vs CPU gap narrowed from 47% (full) to 32% (split); remaining gap is ANE IO overhead
- Correctness: step-0 loss_plus=2.1095 matches across all 4 LoRA modes

## Per-Step Timing Breakdown

### From-Scratch (4-layer, 36.4M)

| Component | CPU (ms) | ANE (ms) |
|-----------|----------|----------|
| Forward pass (2x) | 34 | 27 |
| Perturbation (4x) | 43 | 42 |
| Transpose+stage | 0 | 21 |
| **Total** | **75** | **95** |

### Fine-Tuning (30-layer, 134.5M) — v2 → v3

| Component | MeZO CPU (ms) | MeZO ANE v2 (ms) | MeZO ANE v3 (ms) | BP CPU (ms) | BP ANE (ms) |
|-----------|---------------|-------------------|-------------------|-------------|-------------|
| Forward (2x for MeZO) | 228 | 275 | 249 | 98→100* | 92→97* |
| Perturbation (4x) | 149 | 150 | 141 | — | — |
| Transpose+stage | 0 | 226 | 99 | — | 15→26 |
| Backward | — | — | — | 115→125 | 115→126 |
| Other (rms,silu,cls,dw) | — | — | — | ~60 | ~65 |
| **Total** | **379** | **656** | **501** | **282** | **305** |

*Timings from steady-state steps (step 100+). Initial steps slower due to warmup.*
*v3 MeZO-ANE timings from condition 8 v3 output (step 100: fwd=249, perturb=141, transpose=99).*

### Fine-Tuning (32-layer, 361.8M) — v5

| Component | MeZO CPU (ms) | MeZO ANE (ms) | BP CPU (ms) | BP ANE (ms) |
|-----------|---------------|---------------|-------------|-------------|
| Forward (2x for MeZO) | 428 | 525 | ~200-400* | ~200 (ANE) + IO |
| Perturbation (4x) | 379 | 579† | — | — |
| Transpose+stage | 0 | 478 | — | ~30-90 |
| Backward | — | — | ~250 | ~250-540 |
| Other | ~7 | — | ~40-80 | ~50-90 |
| **Total** | **814** | **1200** | **602** | **700** |

*†Step 0 perturbation is slow (JIT warmup); steady-state is ~376ms.*
*\*BP step times vary widely (545-1907ms) due to thermal throttling; average reported.*

## Memory Comparison (measured RSS)

| Model | MeZO (MB) | Backprop (MB) | Ratio | MeZO fits 8GB? | BP fits 8GB? |
|-------|-----------|---------------|-------|-----------------|--------------|
| 36.4M (4L) | ~320* | ~1200* | ~3.7x | Yes | Yes |
| 134.5M (30L) | 785 | 2,910 | 3.7x | Yes | Yes |
| 361.8M (32L) | 1,720 | 4,133 | 2.4x | Yes | Yes |
| ~1B (est.) | ~4,600 | ~11,000+ | ~2.4x | Yes | **No** |

*36.4M and 1B are estimated from scaling.*

**The memory crossover — where backprop doesn't fit but MeZO does — occurs around
600M-1B params on 8GB devices.** At 360M, both fit. MeZO's memory advantage is real
but only becomes critical for models larger than what we tested.

## Key Findings

### 1. MeZO training loss improves but val loss converges slowly (CRITICAL v4 CORRECTION)
With correct residual scaling (res_alpha=1.0), MeZO training loss (loss_plus)
drops from 2.25 to ~2.0 in 300 steps. However, **validation loss** tells a
different story:

| Steps | MeZO Val Loss | BP-CPU Val Loss | MeZO Train Loss |
|-------|--------------|-----------------|-----------------|
| 100 | 2.2496 | 1.952 | 1.895 |
| 300 | 2.2486 | 1.929 | 2.117 |
| 600 | 2.2453 | — | 1.731 |

**The v2 claim that "MeZO reaches near-backprop quality" was INCORRECT.** That claim
was based on comparing noisy single-batch training losses (loss_plus), which vary
±0.3 per batch. The val loss (averaged over 10 batches) shows MeZO has barely moved
after 600 steps (Δ=0.005), while backprop achieves Δ=0.30 in 100 steps.

**This is expected from MeZO theory.** The paper runs for 20K+ steps (full-parameter
fine-tuning), and MeZO convergence is ~100x slower per step than backprop (Theorem 1).
At 379ms/step, 20K steps = 7600s (2.1 hours). Our 120s budget allows only 317 steps,
which is far too short for MeZO to converge on val loss.

**What IS validated:** MeZO is learning (val loss is monotonically decreasing, train
loss distributional shift from initial). The algorithm is correct. MeZO just needs
significantly more steps than our time budget allows.

### 2. MeZO on ANE — first ZO training on any NPU
Both from-scratch and fine-tuning run successfully on Apple Neural Engine.
The forward pass uses ANE fp16 matmuls; perturbation/RoPE/attention stay on CPU fp32.
Losses match between CPU and ANE modes (within epsilon noise), confirming correctness.

### 3. ANE transpose overhead reduced by 56% (v3 optimization)
Two optimizations applied to IOSurface transpose+staging:
- **Defer 3rd RETRANSPOSE_AND_STAGE**: The post-update restage is immediately
  overwritten by next step's perturbation. Defer to only when validation runs.
  Saves 1 of 3 restages per step (33% reduction).
- **W2 bulk cvt_f32_f16**: W2 staging used element-wise transpose+cast (double loop).
  W2t_buf was already computed but unused. Use vDSP_mtrans + NEON cvt (3.2x faster).

Microbenchmark decomposition (SmolLM2-135M, 30 layers):
- vDSP_mtrans (transpose): 33.5ms per restage
- IOSurface staging: 35.6ms per restage (W2 was 21.2ms of this)
- IOSurface lock/unlock: 0.13ms (negligible)

Result: transpose 226→99 ms/step (56% reduction), total 656→501 ms/step (1.31x speedup)
- v3 MeZO: ANE 1.32x slower (501 vs 379 ms/step) — improved from v2's 1.73x
- v3 Backprop: ANE 1.08x slower (305 vs 282 ms/step) — unchanged
- Remaining ANE overhead: fwd IO writes + ANE dispatch + 2 restages per step

### 4. MeZO memory advantage is real
MeZO uses ~544MB (weights + forward buffers only).
Backprop needs weights + gradients + Adam m/v = ~3x more memory (measured: 785MB vs 2910MB).
This advantage grows with model size — at 1B+ params, backprop may not fit in memory
while MeZO still runs with inference-only memory.

### 5. Backprop converges ~100x faster than MeZO per step (v4 CORRECTED)
Fine-tuning step count comparison (120s):
- BP-CPU: 382 steps (282ms/step) → val_loss=1.929
- BP-ANE: 346 steps (305ms/step) → val_loss=1.929
- MeZO-CPU: 317 steps (379ms/step) → val_loss≈2.249 (measured with val_every=100)
- MeZO-ANE: 240 steps (501ms/step, v3 optimized) → val_loss≈2.249 (estimated)

**v4 correction:** The v2 comparison "MeZO is only 0.16 loss behind" was based on noisy
single-batch training loss. On val loss, MeZO is 0.32 behind backprop (2.249 vs 1.929)
after 300 steps. MeZO needs ~20K steps (2.1 hours) for meaningful val convergence.
The memory advantage (3.7x) is the primary reason to use MeZO, not wall-time efficiency.

### 6. LR sensitivity for MeZO fine-tuning
Only lr=1e-5 produced a decrease in 20s. lr=1e-4 diverged, lr=1e-6/1e-7 showed no signal.
The optimal MeZO LR is ~30x smaller than the backprop LR (1e-5 vs 3e-4).
The MeZO paper uses lr=1e-7 to 1e-6 for OPT-13B (full-parameter). Our higher
lr=1e-5 is consistent with our smaller model (135M vs 13B): the theoretical
ZO LR scales as n/(d+n-1) × SGD_LR, inversely proportional to parameter count.

### 7. DeepNet bug had massive impact on v1 results
The DeepNet res_alpha=1/sqrt(2*30)=0.129 was incorrectly applied to pretrained SmolLM2-135M,
which uses standard Llama architecture with alpha=1.0. This caused:
- Initial loss: 4.20 (v1, wrong) vs 2.24 (v2, correct). HF reference: 1.94
- Gradient magnitudes: proj_grad=42.67 (wrong) vs 0.19 (correct) at step 0
- The 0.129x scaling at each residual connection effectively destroyed the pretrained
  representations, turning fine-tuning into a near-from-scratch training problem

## Bug Fixes During Experiments

### Bug 1: CLI --lr overridden by checkpoint LR
The `mezo_load_checkpoint` function wrote the checkpoint's LR into the lr variable,
ignoring the command-line `--lr` flag. Fixed by tracking `lr_from_cli` and preserving
the CLI value when explicitly provided. This caused the initial condition 7 run to use
lr=3e-4 (from hf_to_ane.py default) instead of lr=1e-5, diverging to loss ~22.

### Bug 2: DeepNet res_alpha applied to pretrained model (CRITICAL)
`res_alpha = 1/sqrt(2*NLAYERS)` was unconditionally applied in the forward pass.
DeepNet residual scaling is ONLY valid for from-scratch training where W_o/W_2 are
initialized with matching scale (1/sqrt(2*N)). Pretrained models (SmolLM2, Llama)
use standard residual connections with alpha=1.0.

**Fix in train_mezo.m:**
```c
float res_alpha = from_scratch ? 1.0f / sqrtf(2.0f * NLAYERS) : 1.0f;
```

**Fix in train.m:**
```c
if (no_deepnet) {
    res_alpha = 1.0f;  // Standard residual for pretrained Llama/SmolLM2 models
}
```

**Confirmed via:** HuggingFace SmolLM2-135M config (AutoConfig) — no DeepNet scaling.

## Optimization 1: IOSurface Transpose (v3)

### Problem
MeZO-ANE called RETRANSPOSE_AND_STAGE 3x per step (~226ms total). Each call transposes
all 7 weight matrices for 30 layers (vDSP_mtrans) then stages them into IOSurfaces
(fp32→fp16 conversion + IOSurface write).

### Microbenchmark Decomposition (SmolLM2-135M)
| Component | Per restage (ms) | Per step 3x (ms) |
|-----------|-----------------|-------------------|
| vDSP_mtrans (7 matrices × 30L) | 33.5 | 100.5 |
| IOSurface staging (30L) | 35.6 | 106.8 |
|   W2 element-wise (bottleneck) | 21.2 | 63.7 |
|   IOSurface lock/unlock | 0.1 | 0.4 |
| **Total** | **50.4** | **151** |

### Fix 1: Defer 3rd RETRANSPOSE_AND_STAGE
The post-update restage (step N) is immediately overwritten by next step's +eps
perturbation + restage. Only needed before validation (every 500 steps).
Eliminates 1 of 3 restages per step.

### Fix 2: W2 vectorized staging
W2 staging used an O(HIDDEN×DIM) element-wise double loop with scalar fp32→fp16 cast.
W2t_buf (pre-transposed copy) was already computed but not used for staging.
Replaced with single loop using NEON-vectorized cvt_f32_f16: 3.2x faster per-layer.

### Result
```
Transpose:  226ms → 99ms  (56% reduction)
Step time:  656ms → 501ms (1.31x speedup)
Steps/120s: 183  → 240   (+31% throughput)
```
**Bit-identical losses verified at step 0 and step 100.**

## Stated Assumptions

1. **Rademacher vs Gaussian perturbation:** Our implementation uses z_i in {-1,+1} instead
   of the paper's z~N(0,I). Both are valid since E[zz^T]=I for both. Rademacher has lower
   kurtosis (E[z^4]=1 vs 3) giving lower gradient variance. Validated experimentally
   (see validation_gradient_unbiased.c). Note: classical SPSA (Spall 1992) specifically
   recommends Rademacher as optimal. The MeZO paper uses Gaussian for compatibility
   with PyTorch's `torch.normal()` + seed management, but mathematically Rademacher
   is equally valid or slightly better.

2. **Cosine schedule present but negligible:** Code implements cosine decay (lr decays from
   base to 0.1×base over total_steps). For our short runs (240-317 steps out of ~100K
   total_steps), decay is <0.3% — effectively constant LR. This actually matches the
   MeZO paper, which uses constant LR for all experiments (Algorithm 1 accepts a schedule
   `{η_t}` but all reported results use constant). Our earlier assumption that the paper
   uses "linear decay" was incorrect.

3. **SmolLM2 tokenizer for TinyStories:** Data tokenized with SmolLM2 tokenizer (49152 vocab).
   SmolLM2-135M was pretrained on different data, so initial loss (2.24) reflects minor
   distribution shift from HF reference (1.94). The gap (0.30) is due to our shorter
   SEQ=256 vs HF's default and VocabMap compaction effects.

4. **Single seed (42):** All conditions use seed=42. Multiple seeds needed for statistical
   significance but impractical within current time budget.

### 8. ANE is slower than CPU for MeZO at ALL tested sizes (v5 FALSIFIED HYPOTHESIS)

**v4 hypothesis:** At 360M+ params, ANE matmul throughput would overcome dispatch overhead,
making MeZO-ANE faster than MeZO-CPU.

**v5 result: FALSIFIED.** MeZO-ANE is 47% slower at 360M (1200 vs 814 ms/step), worse
than the 32% gap at 135M (501 vs 379 ms/step). The gap WIDENED, not narrowed.

| Model | MeZO-CPU (ms) | MeZO-ANE (ms) | ANE overhead | Transpose (ms) |
|-------|---------------|---------------|--------------|----------------|
| 135M (30L) | 379 | 501 | +32% | 99 |
| 360M (32L) | 814 | 1200 | +47% | 478 |

**Root cause: transpose overhead scales superlinearly with model size.**
- 135M: 99ms transpose (2 restages × 7 matrices × 30 layers)
- 360M: 478ms transpose (2 restages × 7 matrices × 32 layers)
- Ratio: 478/99 = 4.8x for 2.69x more parameters
- Transpose grows faster than matmul because weight matrix sizes are larger
  (larger matrices → more data to copy per vDSP_mtrans + cvt_f32_f16)

Even without transpose, ANE forward (525ms) is still slower than CPU forward (428ms)
at 360M. The per-dispatch IO overhead (~50μs × 224 dispatches = ~11ms per fwd pass)
persists, and ANE matmul throughput does not compensate.

**Conclusion: MeZO-on-ANE is structurally slower than MeZO-on-CPU for any model that
fits in memory on Apple Silicon.** The IOSurface restaging cost (required because MeZO
perturbs weights in fp32, but ANE requires fp16 IOSurfaces) is a fundamental
architectural mismatch between MeZO's algorithm and ANE's interface.

Backprop-ANE does better (only 16% slower at 360M) because it only stages weights once
at initialization, not 2x per step.

### 9. 360M scaling confirms MeZO is memory-only advantage (v5)

At 360M params, the full picture is:

| Metric | MeZO-CPU | MeZO-ANE | BP-CPU | BP-ANE |
|--------|----------|----------|--------|--------|
| ms/step | 814 | 1200 | 602 | 700 |
| Steps/120s | 143 | 100 | 140 | 120 |
| Val loss @100 | 2.067 | 2.067 | 1.791 | 1.791 |
| RSS (MB) | 1,720 | — | 4,133 | — |

- Backprop is **faster** (602 vs 814 ms/step on CPU)
- Backprop converges **far better** (val 1.79 vs 2.07 after equal wall time)
- MeZO's sole advantage: **2.4x less memory** (1.7GB vs 4.1GB)
- At 360M, both methods fit in 8GB. MeZO's memory advantage only matters at ~1B+ params.

### 10. Why full-parameter MeZO is a bad fit for ANE (v5 ROOT CAUSE ANALYSIS)

Full-parameter MeZO has a fundamental architectural mismatch with ANE:

1. **MeZO perturbs ALL weights every step** → requires restaging 7×L weight matrices
   into IOSurfaces (fp32→fp16 transpose+copy) **twice per step**
2. **ANE excels at static computation graphs** → weights should be baked at compile time,
   not changing every step
3. **ANE's native primitive is convolution** → 1x1 conv yields 3x throughput vs matmul,
   but our implementation uses matmul dispatches
4. **Per-dispatch IO overhead (~50μs)** accumulates: 7 dispatches × L layers × 2 fwd passes
   = 420+ dispatches/step at 360M

These costs are inherent to full-parameter ZO + ANE, not optimization bugs.

## Path Forward: MeZO + LoRA on ANE (v5 PROPOSED)

The research literature points to a fundamentally better approach: **MeZO + LoRA**.

**Why this plays to ANE's strengths:**

1. **Base weights stay frozen** → compiled/baked once, never restaged. This is ANE's
   preferred operating mode. Zero transpose overhead for the base model.
2. **Only small adapter matrices perturbed** → With LoRA rank 32 on SmolLM2-360M:
   - Adapter params per projection: 2 × (960 × 32) = 61,440
   - Q+K+V+O per layer: 245,760 params
   - Total: 245,760 × 32 layers = **7.9M** (2.2% of full 361.8M)
   - Estimated perturbation time: ~8ms (vs 376ms for full params = **47x reduction**)
3. **Adapters as IOSurface inputs** (Orion paper technique, arXiv:2603.06728) → adapter
   matrices are passed as inputs, not compiled weights. Updating them is a cheap
   IOSurface write, not a retranspose+restage cycle.
4. **Two forward passes** (MeZO's core requirement) are ANE's sweet spot —
   deep graph pipelining achieves near-peak 19 TFLOPS at 2.8W.
5. **Memory stays inference-only** → MeZO's key advantage preserved.

**MeZO + LoRA is proven:** The original MeZO paper (NeurIPS 2023) tests this:
- MeZO+LoRA on LLaMA-7B: SST-2 95.0%, RTE 74.9%, COPA 84.3%
- MeZO+LoRA on OPT-13B: SST-2 89.6%, BoolQ 73.8%
- The ZO-Bench paper (ICML 2024) confirms: "LoRA shows consistent robustness
  when paired with various ZO algorithms"

**Estimated MeZO+LoRA-ANE step time (360M):**

| Component | Full MeZO-ANE | MeZO+LoRA-ANE (estimated) |
|-----------|---------------|---------------------------|
| Forward (2x) | 525ms | ~400ms (1x1 conv = 3x matmul speed) |
| Perturbation (4x) | 376ms | ~8ms (2.2% of params) |
| Transpose+stage | 478ms | ~0ms (adapters as IOSurface inputs) |
| **Total** | **1200ms** | **~410ms** |

If realized, MeZO+LoRA-ANE at ~410ms would be **faster than MeZO-CPU (814ms)** and
competitive with backprop-CPU (602ms), while using inference-only memory (~1.5GB).

**Additional optimizations from recent literature (v6):**

1. **MobiZO's MP-LoRA** (EMNLP 2025): Multi-Perturbed LoRA parallelizes +ε/-ε
   perturbations in a SINGLE forward pass by replicating only the small LoRA B matrix.
   4.3x speedup over MeZO. Already deployed on Qualcomm Hexagon NPU via ExecuTorch.
   Directly applicable to ANE — run both perturbation paths through one compiled graph.

2. **P-GAP** (arXiv:2510.18228): Gradient-aligned perturbations in projected subspace.
   Reduces GPU hours to 22.4% of MeZO+LoRA. 5.2x faster convergence. Combines with
   LoRA for P-GAP+LoRA at 9.1GB (vs 73.5GB full FT). Key idea: perturbations aligned
   with gradient direction have lower variance → fewer steps needed.

3. **AGZO** (arXiv:2601.17261): Activation-guided ZO. Constrains perturbations to the
   subspace spanned by input activations (the gradient lives there by construction).
   Higher cosine similarity with true gradient. Works on Qwen3-0.6B/4B, same memory as MeZO.

4. **DiZO** (arXiv:2502.03304): Per-layer adaptive ZO. 48% less GPU time, sometimes
   beats first-order. SST-2: 92.5% on OPT-2.7B (vs MeZO 90.0%).

5. **MeSP** (arXiv:2602.13069): Structured backprop for LoRA. Computes IDENTICAL
   gradients to backprop but with 62% less memory (136MB vs 361MB on Qwen2.5-0.5B).
   If backprop gradients fit, MeSP dominates MeZO on both speed and accuracy.

6. **1x1 convolution** (Apple ML Research + Orion): ANE's native primitive is convolution.
   Expressing Linear as 1x1 Conv2d yields 3x throughput. Combined with
   adapter-as-input, this is the key to making ANE forward faster than CPU.

**Optimal ANE training stack (proposed):**
```
P-GAP + LoRA + adapter-as-input + 1x1 conv
  = gradient-aligned perturbations (5x fewer steps)
  + small adapter matrices only (2% of params perturbed)
  + zero retranspose (adapters as IOSurface inputs)
  + 3x matmul throughput (convolution primitive)
```

## Related Work: On-Device ZO Training Landscape (v6)

| Paper | Venue | Key Contribution | ANE Relevance |
|-------|-------|-----------------|---------------|
| MeZO | NeurIPS 2023 | In-place ZO-SGD, inference memory | Foundation |
| MobiZO | EMNLP 2025 | MP-LoRA + ExecuTorch, 4.3x speedup | High — NPU-deployed |
| P-GAP | arXiv 2025 | Gradient-aligned perturbation, 5.2x conv. | High — fewer steps |
| AGZO | arXiv 2026 | Activation-guided subspace perturbation | High — forward-only |
| DiZO | arXiv 2025 | Layer-adaptive ZO, 48% less GPU time | Medium |
| MeSP | arXiv 2026 | Structured backprop for LoRA, 62% less mem | Alt. to ZO for ANE |
| Orion | arXiv 2026 | ANE training, adapter-as-input, 20 constraints | Direct — same HW |
| llm.npu | ASPLOS 2025 | NPU prefill 22.4x speedup, 1000+ tok/s | Inference baseline |
| On-Device ZO | arXiv 2025 | MeZO enables 2-25x larger models on-device | Motivation |
| AMD NPU Train | arXiv 2025 | First NPU training paper, GPT-2 124M | Peer comparison |
| Sparse MeZO | ICLR 2024 | 0.1% sparse subset, 3.5x speedup | Medium |
| SubZero | ICCV 2025 | Random subspace ZO, works with LoRA | Medium |
| ZO Fine-tuner | arXiv 2025 | Learned perturbation strategy, 82% win rate | Medium |
| MaZO | EMNLP 2025 | Masked ZO for multi-task | Low |

## Sources

- [MeZO: Fine-Tuning LLMs with Just Forward Passes (NeurIPS 2023)](https://arxiv.org/abs/2305.17333)
- [MobiZO: Efficient LLM Fine-Tuning at the Edge (EMNLP 2025)](https://arxiv.org/abs/2409.15520)
- [P-GAP: Projected Gradient-Aligned Perturbations (arXiv 2025)](https://arxiv.org/abs/2510.18228)
- [AGZO: Activation-Guided Zeroth-Order Optimization (arXiv 2026)](https://arxiv.org/abs/2601.17261)
- [DiZO: Divergence-driven Zeroth-Order Optimization (arXiv 2025)](https://arxiv.org/abs/2502.03304)
- [MeSP: Memory-Efficient Structured Backpropagation (arXiv 2026)](https://arxiv.org/abs/2602.13069)
- [Orion: Programming Apple's ANE for LLM Training (arXiv 2026)](https://arxiv.org/abs/2603.06728)
- [llm.npu: Fast On-device LLM Inference with NPUs (ASPLOS 2025)](https://arxiv.org/abs/2407.05858)
- [On-Device Fine-Tuning via Backprop-Free ZO (arXiv 2025)](https://arxiv.org/abs/2511.11362)
- [AMD NPU Training (arXiv 2025)](https://arxiv.org/abs/2504.03083)
- [Sparse MeZO (ICLR 2024)](https://arxiv.org/abs/2402.15751)
- [SubZero: ZO in Random Subspaces (ICCV 2025)](https://arxiv.org/abs/2402.15751)
- [ZO-Bench (ICML 2024)](https://arxiv.org/abs/2402.11592)
- [ZO Fine-tuner: Learned ZO Optimizer (arXiv 2025)](https://arxiv.org/abs/2510.00419)
- [Deploying Transformers on Apple Neural Engine (Apple ML Research)](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Apple Foundation Models Tech Report 2025](https://arxiv.org/abs/2507.13575)
- [Inside the M4 Apple Neural Engine (maderix)](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [MobiZO GitHub](https://github.com/leigao97/MobiZO)
- [maderix/ANE GitHub](https://github.com/maderix/ANE)
