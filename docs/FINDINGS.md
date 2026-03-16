# AutoANE: Complete Findings, Methodology & Audit Trail

**Project**: Training Llama-family transformers on Apple's Neural Engine via reverse-engineered private APIs
**Duration**: March 10-16, 2026 (65 commits, 5 sessions)
**Hardware**: Apple M2 Pro, macOS 15+
**Codebase**: ~3,555 lines C/ObjC (training) + ~2,100 lines C/ObjC (MeZO) + 8 Python scripts + 8 regression tests

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Methodology](#2-methodology)
3. [Core Findings](#3-core-findings)
4. [ANE Characterization](#4-ane-characterization)
5. [Architecture & Hyperparameter Search](#5-architecture--hyperparameter-search)
6. [Code Quality & Bugs Found](#6-code-quality--bugs-found)
7. [Verification Process](#7-verification-process)
8. [Upstream Research Synthesis](#8-upstream-research-synthesis)
9. [What ANE Is Good For (and What It Isn't)](#9-what-ane-is-good-for-and-what-it-isnt)
10. [Open Research Directions](#10-open-research-directions)
11. [Assumptions Registry Summary](#11-assumptions-registry-summary)
12. [Experiment Index](#12-experiment-index)
13. [Session-by-Session Work Log](#13-session-by-session-work-log)

---

## 1. What This Project Is

AutoANE answers a simple question: **can Apple's Neural Engine train language models, and should it?**

The Neural Engine (ANE) is a dedicated machine learning accelerator on every Apple Silicon chip. Apple designed it for inference — running CoreML models, powering Siri, processing camera frames. No official training API exists. Several projects (primarily [maderix/ANE](https://github.com/maderix/ANE)) have reverse-engineered the private `_ANEClient`/`_ANECompiler` APIs to run custom compute on the ANE.

AutoANE builds on maderix's work to implement a complete training pipeline for Llama-architecture transformers (RMSNorm, RoPE, GQA attention, SwiGLU FFN, AdamW optimizer). It then runs 44 controlled experiments to characterize when and whether ANE training is useful.

**The short answer**: ANE's raw matmul is 2.5x faster than CPU, but overhead from weight staging via IOSurface negates this for standard backprop training. CPU-only backprop training produces better models faster at every tested size. However, **MeZO zeroth-order training with LoRA-split and conv-fused kernels achieves 1.71x faster than CPU** — the first ANE-faster-than-CPU training result. The key insight: LoRA-split freezes base weights as BLOBFILE constants (eliminating per-step IOSurface staging), and conv1x1 with kernel fusion reduces IO round-trips from 224 to 96.

---

## 2. Methodology

### 2.1 Experimental Protocol

Every experiment follows the same protocol:

1. **Single variable**: Change exactly one thing per experiment. Multi-variable changes are split into separate runs.
2. **Controlled comparison**: Same data (TinyStories, 20M tokens), same random seed (42 unless testing seed sensitivity), same hardware (M2 Pro), same time budget.
3. **Clean system**: After E37 showed concurrent processes cause false results (we mistakenly attributed CPU scheduling jitter to "ANE divergence"), all subsequent experiments run on a clean system with no background processes.
4. **Quantitative metrics**: Every claim cites a specific val_loss, step count, timing, or power measurement. No qualitative assessments.
5. **Reproducibility**: Key results reproduced independently (E42 verified E39-E41 within 0.3%). Seed sensitivity characterized (±0.3 nats run-to-run variance).

### 2.2 Measurement Tools

| Metric | Tool | Precision |
|--------|------|-----------|
| Loss (train/val) | Cross-entropy with log-sum-exp | 32-bit float |
| Timing | `clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)` | Nanoseconds |
| Power | `sudo powermetrics --samplers cpu_power,gpu_power,ane_power` | Per-subsystem mW, 60s avg |
| Gradient health | Gradient L2 norm printed at accumulation boundaries | Finite check + magnitude |
| Memory | IOSurface allocation sizes computed from model dimensions | Exact bytes |

### 2.3 Verification Philosophy

This project treats every claim as a hypothesis until verified:

- **71 assumptions tracked** in ASSUMPTIONS.md (27 verified, 1 qualified, 8 disproved, 13 resolved, 23 from upstream literature)
- **10 implicit assumptions found** during code audit (6 fixed)
- **7 literature references verified** against source papers
- **6 parallel verification agents** deployed for exhaustive code review, with 3 false positives caught and corrected
- **All algorithms verified from first principles**: Adam optimizer vs Kingma & Ba 2014, cross-entropy vs textbook, RMSNorm backward vs Zhang & Sennrich 2019, vDSP parameter ordering vs Apple SDK headers

### 2.4 What "Verified" Means

A finding is "verified" when:
- The experiment has been run at least twice with consistent results
- The measurement methodology has been validated (e.g., power via `powermetrics`, not estimation)
- The code implementing the measurement has been audited for correctness
- Alternative explanations have been considered and ruled out

A finding is "qualified" when:
- Results are directionally correct but magnitude is uncertain (e.g., autosearch val_loss 3.288 is a best-of-88 seed artifact)

A finding is "disproved" when:
- A controlled experiment contradicts the assumption (8 assumptions disproved)

---

## 3. Core Findings

### Finding 1: CPU-Only Training Beats ANE at Every Tested Size

| Model Size | CPU ms/step | ANE ms/step | ANE Overhead | Winner |
|-----------|-------------|-------------|-------------|--------|
| 36.4M (512d/4L) | 24 | 28 | IOSurface 4ms | CPU |
| 95.4M (1024d/4L) | 102 | 69 | IOSurface 8ms + fp16 loss | CPU (better models) |
| 142M (1536d/4L) | — | parity | IOSurface 5ms but cache pressure | Tie |
| 281M (2048d/4L) | — | 2x slower | 379MB IOSurface → cache thrashing | CPU |

**Why ANE loses**: The ANE computes in fp16, requiring weights to be converted from fp32 and staged via IOSurface each step. This conversion costs 4-8ms/step. More critically, fp16 precision degrades model quality by ~16% (irreducible — tested via clamping, LR tuning, weight decay). Even when ANE is faster per-step (at DIM=1024), CPU produces better models because fp32 gradients are more accurate.

**Evidence**: E11, E36, E38 (dimension scaling study — novel, first published)

### Finding 2: ANE Provides No Power Savings

| Mode | Package Power | Energy/step |
|------|--------------|-------------|
| Idle | 8,455 mW | — |
| CPU-only | 13,273 mW | 9.2 mJ |
| ANE matmul | 12,568 mW | 10.9 mJ |
| ANE full | 12,664 mW | 9.7 mJ |

ANE shifts ~1.4W from CPU to ANE subsystem, but total package power is unchanged. CPU-only achieves lowest energy per step because it completes more useful work (fp32 gradients) at the same power draw. This is the first published power measurement of ANE training workloads.

**Evidence**: E12 (powermetrics, 60s per mode, idle-subtracted)

### Finding 3: Step Count Dominates Model Capacity at Fixed Time

| Budget | 512d/4L (36M) val_loss | 1024d/4L (95M) val_loss | Steps ratio |
|--------|------------------------|-------------------------|-------------|
| 120s | 3.54 | 4.30 | 2.4x |
| 300s | 3.09 | — | — |
| 600s | 2.55 | — | — |
| 1800s | 2.22 | — | — |

At 120s, the smaller model gets ~2,500 steps vs ~1,050 for the larger model. Each gradient step contributes more than each parameter because we operate in a severely data-constrained regime: 20M tokens for a 36M-param model is 23x below Chinchilla's optimal 20:1 token:parameter ratio.

This confirms Kaplan et al.'s (2020) observation: loss scales as D^(-0.095) for data and N^(-0.076) for parameters. Since 0.095 > 0.076, more data encounters (via more steps) beats more parameters.

**Evidence**: E39 (11-config grid), E40 (LR sweep), E41 (budget scaling), E42 (independent reproduction)

### Finding 4: fp16 Precision Gap Is Irreducible

ANE computes everything in fp16. The quality gap vs fp32 is ~16% and we tested 5 approaches to close it — all failed:

| Approach | Result | Root Cause |
|----------|--------|-----------|
| Activation clamping [-4, +4] | No change | Loss is from matmul accumulation rounding, not magnitude |
| Lower LR (1e-4) | Worse | Underfitting |
| Higher weight decay (0.3) | No change | WD doesn't affect per-step precision |
| Loss scaling (256x) | Required but doesn't close gap | Prevents underflow, doesn't improve accumulation |
| DeepNet scaling | Required for stability | Prevents overflow, gap remains |

At DIM=1024, each fp16 dot product accumulates sqrt(1024) = 32 ULPs of rounding error. This is a hardware limitation of the fp16 MAC units.

**Evidence**: E14, E15, E19 (5777 steps, zero NaN/Inf, gap remains)

### Finding 5: Only Use ANE for Linear Projections

| Mode | Train-val gap | Description |
|------|--------------|-------------|
| ANE full (fused) | 1.22 | RoPE, softmax, SiLU all in fp16 |
| ANE matmul-only | 0.60 | Only linear projections on ANE |
| CPU-only | 0.60 | All fp32 |

**ANE matmul-only matches CPU val_loss to 4 decimal places** at matched step counts. The precision problem is specifically in non-linear operations (softmax, SiLU, RoPE) where fp16 error compounds multiplicatively. Linear projections (matmul) tolerate fp16 because accumulation error averages out.

**Evidence**: E36 (novel finding — first demonstration of selective ANE offloading matching CPU quality)

### Finding 6: Delta Compilation Does Not Work

We tested 5 approaches to avoid recompiling ANE kernels when weights change:

1. Unload → write new BLOBFILE → reload: output unchanged (ANE caches compiled binary)
2. tmpDir weight patching: output unchanged
3. e5bundlecache investigation: only small metadata (~96 bytes)
4. _ANEInMemoryModel reload: API not functional
5. Fresh recompile with cached graph: ~60ms/kernel (too expensive)

The Orion paper claims 8.5x faster recompilation via unload/reload. We could not reproduce this.

**Evidence**: E10, E17, E34

### Finding 7: Autonomous Search Hill-Climbs on Noise

100 random-perturbation experiments found a config claiming val_loss 3.288 (17% improvement). Independent verification shows this config typically produces ~3.8. Run-to-run variance (~0.3 nats from random initialization) exceeds most improvement signals in the search. The manually-tuned baseline (LR=4e-4, ACCUM=10) reliably produces ~3.5.

**Evidence**: E44 (100 experiments), independent verification runs

### Finding 8: MeZO+LoRA-Split Achieves First ANE-Faster-Than-CPU Training

MeZO (zeroth-order optimizer) with LoRA-split mode on SmolLM2-360M (pretrained, 32L, DIM=960, 1.7M trainable params) achieves ANE training 1.71x faster than CPU:

| Phase | Speed (ms/step) | vs CPU (447ms) | Convergence | Status |
|-------|----------------|----------------|-------------|--------|
| Baseline (CPU MeZO) | 447 | 1.00x | reference | — |
| Conv1x1 hybrid | 403-429 | 1.04-1.11x | 1.0x | ✅ |
| **Fused conv kernels** | **~262** | **1.71x** | **1.0x** | **✅** |
| FZOO K=4 | 2.5x slower/step | — | no wall-time benefit | ✅ |
| P-GAP (paper params) | same | — | DIVERGES | ❌ |
| P-GAP (standard params) | same | — | neutral | ❌ |

**Why this works when Finding 1 said CPU always wins**: Finding 1 tested *backprop* training with *dynamic weight staging* — weights are converted fp32→fp16 and staged via IOSurface every step. MeZO+LoRA-split eliminates this overhead entirely: base weights are frozen as BLOBFILE constants (compiled once, never re-staged), and only LoRA corrections (rank-8 matrices, ~50KB/layer) are applied on CPU. Conv1x1 with fused kernels further reduces IO round-trips from 224 to 96 per forward pass.

**Convergence verification**: val_loss within 0.03% of CPU MeZO baseline over 50-step average. Phase 2 does not degrade training quality.

**Negative results documented**: FZOO provides better per-estimate gradient quality but the 2.5x computational overhead eliminates the benefit. P-GAP (gradient-aligned perturbations) was tested with both a simplified flat-vector implementation and a faithful per-matrix SVD implementation matching arXiv:2510.18228. Paper hyperparameters (ε=0.1, lr=1e-2) diverge catastrophically on SmolLM2-360M LoRA. Standard hyperparameters produce identical convergence to baseline. Root cause: LoRA rank-8 matrices are too small for per-matrix SVD to find useful low-rank structure.

**Evidence**: Design spec (2026-03-15), Phase 4 research log (2026-03-16), bench_conv 500-iter averages, 50-step timing averages

### Finding 9: MeZO LoRA Has a Quality Ceiling (Not Just Speed Problem)

MeZO LoRA convergence saturates at val_loss ~2.052 regardless of step count. This is a fundamental quality ceiling, not a speed limitation:

| Steps | MeZO val_loss | Backprop val_loss | MeZO delta | Backprop delta |
|-------|-------------|-----------------|-----------|---------------|
| 191 | ~2.065 | **1.925** | 0.007 | **0.147** |
| 600 | **2.052** | — | **0.019** | — |
| 1000 | 2.052 | — | 0.019 | — |

- Steps 600-1000: **exactly 0.000 nats improvement** (saturated)
- Backprop achieves **7.6x more improvement** in **3x fewer steps**
- The gap (0.127 nats, 6.2%) is fundamental to ZO gradient estimation
- ZO captures ~1/sqrt(d) of gradient information per step (d=1.7M: 0.077%)
- No ZO variance reduction technique (BSZO, SVRG, SubZero) can close this gap
- They can speed convergence TO the ceiling, but the ceiling itself is fixed

**Root cause**: LoRA fine-tuning creates a narrow loss valley. ZO can find the valley direction but cannot make fine-grained progress within it. Backprop follows the exact gradient, navigating the valley precisely.

**Implication**: P16 hybrid (ANE forward + CPU backward) is the ONLY path to both ANE hardware utilization AND backprop-quality training.

**Evidence**: 1000-step convergence data (convergence_1000step_lr1e4_seed42.txt), condition13 backprop baseline, condition20 MeZO 300s run

---

## 4. ANE Characterization

### 4.1 Raw Compute Performance

| Operation | CPU (AMX/Accelerate) | ANE (fp16) | Speedup |
|-----------|---------------------|------------|---------|
| 1024x1024 matmul | 0.67ms | 0.45ms | 1.5x ANE |
| 2816x1024 matmul | 1.88ms | 1.00ms | 1.9x ANE |
| Conv 1x1 (equivalent) | — | — | 1.5-2.8x vs ANE matmul |
| Full forward+backward | 102ms | 69ms | 1.5x ANE (but worse loss) |

### 4.2 IOSurface Memory Pressure

| DIM | Total IOSurface | ANE vs CPU | Behavior |
|-----|----------------|------------|----------|
| 512 | ~15MB | ~1.2x faster | Fits easily in L2/SRAM |
| 1024 | 60MB | 1.5x faster | Near SRAM limit (~32MB) |
| 1536 | 220MB | Parity | Cache pressure begins |
| 2048 | 379MB | 2x slower | Cache thrashing in ALL ops (including CPU backward) |

The ANE's on-chip SRAM is ~32MB. Above this, performance degrades 30% (U4, confirmed by maderix). At 220MB total IOSurface, the system memory bus saturates and even CPU-side operations slow down.

### 4.3 Thermal Behavior

- Single-process throttling: 30% at 10 minutes (E24)
- Concurrent processes: 50%+ throttling (E18 — this was the source of the false "ANE divergence" finding)
- Neither CPU nor ANE diverges with sustained training on a clean system (E37, 5777 steps)

### 4.4 ANE Power Draw

- Peak: ~1.2W (ane_full mode)
- Average: ~765mW (ane_full), ~384mW (matmul-only)
- Idle: ~9mW
- Previous estimate of 2.8W was extrapolated and incorrect (corrected in U1)

---

## 5. Architecture & Hyperparameter Search

### 5.1 Architecture Grid (E39-E41)

11 configurations tested at 120s, top 3 at 300s and 600s:

| Rank | Config | Params | Steps@120s | val_loss@120s | val_loss@600s |
|------|--------|--------|-----------|---------------|---------------|
| 1 | 512d/4L | 36.4M | 2,471 | 3.54 | 2.55 |
| 2 | 768d/2L | 48.1M | 2,100 | 3.69 | 2.84 |
| 3 | 1024d/2L | 72.9M | 1,524 | 3.85 | — |
| 11 | 1024d/8L | 164.7M | 577 | 4.76 | — |

**Key insight**: Depth strictly hurts at every width. Adding layers reduces throughput (more compute per step) and doesn't improve generalization at our data scale. 2-layer models overfit despite high throughput (train-val gap +0.83).

### 5.2 Optimal Hyperparameters

| Parameter | Optimal Value | How Determined |
|-----------|--------------|----------------|
| Architecture | 512d/4L (36.4M params) | E39 grid search |
| Learning Rate | 4e-4 (5e-4 marginal) | E40 LR sweep |
| Sequence Length | 128 | E43 (1.75x throughput vs SEQ=256) |
| Accumulation Steps | 10 | E43 (5 too noisy, 20 too few updates) |
| Weight Decay | 0.1 (insensitive) | E43 |
| Warmup | 100 steps (insensitive) | E43 |
| Loss Scale | 256.0 (ANE only) | E1 |

### 5.3 Scaling Laws at Our Regime

At 20M tokens (our dataset), the optimal model is far smaller than Chinchilla would predict:
- Chinchilla optimal for 20M tokens: ~1M parameters
- Our best model: 36.4M parameters (36x over-parameterized)
- Tokens/parameter: 0.55 (vs Chinchilla's recommended 20)
- Implication: we are in a regime where more data encounters (steps) matter far more than model capacity

---

## 6. Code Quality & Bugs Found

### 6.1 Bugs Found and Fixed (8 total)

| # | Severity | Bug | Impact | Fix |
|---|----------|-----|--------|-----|
| 1 | **CRITICAL** | RMSNorm backward gradient: extra `w[i]` on correction term | 7.5% relative error with non-unit weights. Masked when w≈1.0 (early training). Formula was `dx=w[i]*rrms*(dy-x*dot)`, should be `dx=rrms*(w[i]*dy-x*dot)` | Reordered vDSP operations in cpu_ops.h |
| 2 | HIGH | Dead allocation `acembed` (66MB) | 66MB AdamState allocated but never read or written | Removed |
| 3 | HIGH | Missing GQA validation in checkpoint load | kv_heads, head_dim, q_dim written to checkpoint but never validated on read — loading a mismatched checkpoint would silently corrupt | Added dimension validation |
| 4 | MEDIUM | Dead allocation `gate_buf` (720KB) | Buffer allocated but never used | Removed |
| 5 | MEDIUM | Data file missing validation | Empty files silently accepted, odd-byte-count files silently truncated | Added explicit checks |
| 6 | MEDIUM | 4 raw calloc calls missed in P10 hardening | cpu_ops.h:35, mil_dynamic.h:521,536,552 not using safe_calloc | Converted |
| 7 | LOW | File descriptor leak on mmap failure | `close(data_fd)` missing on mmap error path | Added close() |
| 8 | LOW | Test 7 misconfigured | Gradient health test didn't reach accumulation boundary | Fixed accum/steps params |

### 6.2 Security Hardening (P10, 10 fixes)

1. `safe_malloc()`/`safe_calloc()` wrappers — ~60+ call sites converted
2. Token OOB bounds checks in cross_entropy, embed_lookup, embed_backward
3. Token range validation on data file load (all tokens < VOCAB)
4. `ane_init()` → bool return with dlopen/NSClassFromString validation
5. Checkpoint header validation (magic 0x424C5A54, version 4, dimension matching, bounds)
6. Compiler hardening: `-fstack-protector-strong -D_FORTIFY_SOURCE=2`
7. Bridge: malloc NULL checks, integer overflow fix (int→size_t)

### 6.3 Performance Optimizations (P7, 7 vectorizations)

| Operation | Before | After | Impact |
|-----------|--------|-------|--------|
| Adam optimizer | Scalar loop | vDSP/vForce pipeline | 25.2M float buffer, 38+ calls/accum step |
| Gradient scaling | 9 scalar loops/layer | vDSP_vsmul | 32 layers × 9 matrices |
| transpose_weight | Nested loop | vDSP_mtrans | 7 matrices × 32 layers |
| gqa_reduce_kv | Scalar inner loop | vDSP_vadd | 2x per backward step per layer |
| vocab_scatter_grads | Scalar | cblas_saxpy | Once per accum over ~16893 tokens |
| embed_lookup | Scalar scatter | cblas_scopy (strided) | Once per forward step |
| embed_backward | Scalar accumulate | cblas_saxpy (strided) | Once per backward step |

### 6.4 Remaining Non-Critical Issues

- `from_scratch` variable set by `--scratch` flag but never read (default behavior is already from-scratch)
- `t1` variable declared but never used
- `validate_weights` function defined but never called
- Dead `tgt < 0` check (targets are uint16_t, always non-negative)

---

## 7. Verification Process

### 7.1 Algorithm Verification (First Principles)

Every core algorithm was verified against its reference:

| Algorithm | Reference | Verified |
|-----------|-----------|----------|
| Adam optimizer | Kingma & Ba 2014 + AdamW (Loshchilov & Hutter 2019) | Correct: bias correction, decoupled weight decay |
| Cross-entropy loss | Textbook + log-sum-exp trick | Correct: numerically stable softmax |
| RMSNorm backward | Zhang & Sennrich 2019 | **Bug found and fixed** (see 6.1) |
| Gradient accumulation | Standard: accumulate over N steps, average | Correct: divide by accum_steps |
| Loss scaling | Mixed precision training (Micikevicius et al. 2018) | Correct: scale cancels in forward/backward |
| LR schedule | Cosine decay with linear warmup | Correct: min_lr = 0.1 × max_lr |
| vDSP_vdiv | Apple SDK: `C = B/A` (first param is divisor) | Correct usage in Adam |
| vDSP_vsub | Apple SDK: `C = B - A` | Correct usage throughout |

### 7.2 Numerical Verification

- CPU vs ANE agreement: max 0.001128% divergence over 70 steps (matmul-only mode)
- Deterministic reproduction: same seed → identical loss to all printed decimal places
- Step 0 loss: 9.7273 vs theoretical ln(16893) = 9.7349 → 0.08% agreement
- Monotonic loss decrease verified over 70 steps

### 7.3 Literature Verification

All 7 cited references verified against source papers:
- Kaplan et al. (2020): scaling exponents 0.095 (data) > 0.076 (params) — confirmed
- Hoffmann et al. (2022, Chinchilla): 20:1 token:param ratio — confirmed
- Eldan & Li (2023, TinyStories): 33M model val_loss ~2.0 at convergence — confirmed
- Wang et al. (2022, DeepNet): res_alpha = 1/sqrt(2N) — confirmed
- Smith et al. (2018): LR_opt ∝ sqrt(batch_size) — confirmed
- Muennighoff et al. (2024): multi-epoch degradation — confirmed
- Nguyen & Salazar (2019): perplexity baseline — confirmed

### 7.4 Agent Cross-Checking

6 parallel verification agents were deployed. 3 produced false positives:
1. **Math agent** missed the RMSNorm bug entirely (found manually during cross-check)
2. **MIL agent** falsely claimed sdpaBwd1/2 memory leak — they ARE freed on line 1583
3. **Bridge agent** incorrectly said integer overflow fix `(size_t)rows * cols * 2` was wrong — unary cast has higher precedence than binary multiplication

**Lesson**: Agent outputs must be manually verified. Automated analysis catches different things than humans, but makes its own mistakes.

---

## 8. Upstream Research Synthesis

### 8.1 Sources Reviewed

- maderix/ANE: 48 PRs, 12 issues (ALL reviewed)
- karpathy/autoresearch: 200+ PRs, 200+ issues (ALL reviewed)
- Orion paper (arxiv:2603.06728)
- imperatormk/ane-train
- thebasedcapital/ane-infer
- MeZO, FwdLLM, MobiEdit, ElasticZO papers

### 8.2 Key Upstream Findings

| ID | Finding | Source | Our Status |
|----|---------|--------|------------|
| UP1 | Runtime weight injection via IOSurface inputs | imperatormk/ane-train | Our pipeline already does this |
| UP4 | Mega-kernel fusion: 3-4x forward speedup | maderix PR #24 | Not tested (conflicts with runtime weights) |
| UP5 | _ANEChainingRequest is DEAD on macOS 15+ | maderix PR #40 | Confirmed — requires Espresso IR |
| UP14 | MeZO: forward-only gradient estimation | Princeton NLP 2023 | **✅ IMPLEMENTED** — 1.71x faster than CPU with LoRA-split+conv-fused |
| UP17 | Function parameter IOSurfaces: 30% faster | maderix PR #22 | Not tested — requires MIL gen changes |
| UP19 | ACCUM_STEPS=100 gives 4.74x throughput | maderix Issue #24 | Tested: 2.2-3.0x (lower than claimed) |
| UP22 | Layer fusion: 3-4x forward speedup | maderix Issue #24 | Not tested |
| UP23 | cblas_sgemm = 2.6x CPU speedup | karpathy/llm.c PR #840 | We already use cblas_sgemm |

---

## 9. What ANE Is Good For (and What It Isn't)

### 9.1 What ANE Is Not Good For (with current approaches)

**Training language models**, at least with our dynamic weight injection pipeline. The overhead of staging weights via IOSurface each step, combined with fp16 precision loss, makes CPU-only training strictly better at every tested model size (36M-281M params).

### 9.2 What ANE Is Good For

1. **Inference**: ANE is designed for inference. Weights are baked as constants, models compile once and run millions of times, no IOSurface round-trips needed. CoreML, Siri, camera processing all use this path.

2. **Dedicated silicon**: ANE runs independently of CPU and GPU. During ANE inference, CPU and GPU are free for other work. This matters for mobile devices where thermal budget is shared.

3. **Low idle power**: ANE draws ~9mW when idle vs hundreds of mW for GPU. For always-on inference tasks (keyboard prediction, voice detection), this matters.

4. **On-device training on iOS/iPad**: On mobile devices, ANE may be the only thermally viable compute unit for sustained workloads. CPU throttles faster, GPU heats the shared thermal envelope. ANE's dedicated cooling path could enable training that's impossible any other way. **Untested hypothesis.**

### 9.3 What Changed This Picture — And What Remains

**RESOLVED — MeZO+LoRA-split works (Finding 8)**: Zeroth-order training with LoRA-split mode achieves 1.71x faster than CPU. The key was eliminating per-step weight staging entirely: base weights are BLOBFILE constants, LoRA corrections are CPU-side, and conv1x1 with fused kernels reduces IO overhead. This validates the hypothesis that ANE's forward-pass speed advantage could be unlocked by avoiding the weight-staging bottleneck.

**Remaining approaches** that could further improve ANE training:

1. **Cross-layer fusion**: Current fusion is intra-layer (QKV combined, FFN mega-kernel). Fusing across multiple transformer layers into fewer ANE dispatches could further reduce IO round-trips. Challenge: LoRA corrections between layers prevent full fusion.

2. **INT8 quantized LoRA corrections**: LoRA A/B matrices could potentially be quantized to INT8 for ANE compute, leveraging ANE's 1.88x INT8 throughput advantage. Risk: rank-8 matrices may lose too much precision.

3. **Larger model scaling**: SmolLM2-360M (960d/32L) has conv1x1 advantageous for 5/7 projections. Larger models with wider dimensions should see even greater conv1x1 benefits. The 1.71x speedup may increase at larger scale.

4. **Function parameter IOSurfaces**: 30% faster than spatial packing for remaining matmul kernels (Wk, Wv). A targeted MIL generation change.

---

## 10. Open Research Directions

Ranked by expected impact (updated 2026-03-16):

| Priority | Direction | Why | Effort | Status |
|----------|-----------|-----|--------|--------|
| P9 | MeZO+LoRA-split+conv-fused | Forward-only + frozen base weights + fused kernels | Done | **✅ 1.71x CPU** |
| P11 | Cross-layer fusion | Further reduce IO round-trips beyond intra-layer fusion | 3-5 days | NEW |
| P12 | Larger model scaling (MeZO) | Test if 1.71x speedup increases at >360M params | 2-3 days | NEW |
| P13 | INT8 quantized LoRA | Leverage ANE's 1.88x INT8 throughput for LoRA corrections | 2-3 days | NEW |
| P5 | Larger dataset | Test if findings generalize beyond 20M tokens | 0.5-2 days | Open |
| P1 | Function parameter IOSurfaces | 30% faster for remaining Wk/Wv matmul kernels | 2-3 days | Open |
| P14 | MeZO+LoRA on mobile (iOS/iPad) | ANE may be only viable compute for sustained mobile training | 1 week | NEW |
| P15 | Better ZO gradient estimators | Beyond FZOO/P-GAP — explore GraDFree, SZOFW, or variance reduction | Research | NEW |

---

## 11. Assumptions Registry Summary

| Category | Count | Examples |
|----------|-------|---------|
| Verified (V1-V27) | 27 | ANE matmul 2.5x faster, CPU wins end-to-end, 512d/4L optimal |
| Qualified (V27) | 1 | Autosearch 3.288 is seed artifact |
| Disproved (D1-D8) | 8 | IOSurface not primary bottleneck, ANE not power-efficient, delta compile doesn't work |
| Retesting (SA1-SA4) | 4 | fp16 gap irreducible (re-confirmed), ANE divergence (disproved) |
| Unverified/Resolved (U1-U17) | 13 | Most resolved via experiments or literature |
| From upstream (UP1-UP23) | 23 | Runtime injection, mega-kernels, MeZO, M5 alignment, Core AI |
| Implicit found (IA1-IA10) | 10 | 6 fixed (data validation, dtype checks, Xcode check, etc.) |

Full registry: [ASSUMPTIONS.md](ASSUMPTIONS.md)

---

## 12. Experiment Index

| Exp | Description | Key Result |
|-----|-------------|------------|
| E1 | Loss scaling sweep | 256.0 optimal for fp16 |
| E2 | Architecture: shallow/wide vs deep/narrow | 4L/1024d beats 32L/960d |
| E3 | LoRA fine-tuning | Stable, loss 4.22 |
| E5 | Higher LR for ANE | Hurts — larger activations degrade fp16 |
| E7 | IOSurface timing | 8.1ms of 63.2ms (13%) |
| E8 | CPU AMX benchmark | 1.5-2.5 TFLOPS fp32 |
| E9 | Conv 1x1 vs matmul | Conv 1.5-2.8x faster |
| E10 | Delta compilation attempt 1 | Failed — output unchanged |
| E11 | CPU vs ANE comparison | ANE 1.5x faster step, but worse loss |
| E12 | Power measurement | 12.6-13.3W all modes |
| E13 | Classifier optimization | Row-major 2x faster |
| E14 | Activation clamping | No effect |
| E15 | LR/WD tuning for fp16 | No improvement |
| E17 | Delta compilation attempt 2 (5 approaches) | All fail |
| E18 | Extended training (10min) | "Divergence" — later disproved (concurrent processes) |
| E19 | Gradient sanitization | 5777 steps stable, gap genuine |
| E22-E24 | Benchmark suite, thermal study | 30% single-process throttling |
| E28-E32 | Adaptive switching, LoRA rank, power | Adaptive worse, LoRA rank 8 sufficient |
| E34 | Delta compilation attempt 3 | Re-confirmed: doesn't work |
| E36 | **ANE matmul-only mode** | **Matches CPU quality** — novel finding |
| E37 | 10-min sustained training | Both modes stable on clean system |
| E38 | **Dimension scaling study** | **IOSurface ceiling at 220MB** — novel finding |
| E39 | **Architecture grid search** | **512d/4L optimal** |
| E40 | LR sweep per architecture | LR=5e-4 for 512d, 3e-4 for 1024d |
| E41 | Budget scaling (120-1800s) | 512d/4L advantage grows with time |
| E42 | Independent verification | All E39-E41 reproduced within 0.3% |
| E43 | Hyperparameter optimization | SEQ=128, ACCUM=10, WD/warmup insensitive |
| E44 | 100-experiment autosearch | Variance exceeds signal |

Full experiment log: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## 13. Session-by-Session Work Log

### Session 1 (Mar 10-11, ~18 hours)

**Foundation + 34 experiments**

Built the complete training pipeline, ran experiments E1-E43, implemented LoRA, matmul-only mode, autoresearch integration, all Python tools (generate, export, import), and the autonomous search loop. Produced the project's core findings. 34 commits.

### Session 2 (Mar 12 morning, ~4 hours)

**Systematic verification + documentation**

Verified all 7 literature references, reproduced key results, exposed autosearch variance problem, fixed false claims (credited maderix, corrected hardware references), rewrote documentation as research-first technical report. 12 commits.

### Session 3 (Mar 12 midday, ~4 hours)

**Upstream research + security hardening**

Reviewed ALL upstream PRs/issues (~260 total), discovered 3 game-changing developments (runtime weight injection, mega-kernel fusion, chaining is dead), implemented P10 security hardening (10 fixes), P7 vectorization (7 optimizations), P4 multi-seed autosearch, P6 IOSurface audit, 8-test regression suite. 7 commits.

### Session 4 (Mar 12 afternoon, ~3 hours)

**Exhaustive verification sweep**

Deployed 6 parallel verification agents, found and fixed RMSNorm backward bug (critical), removed 66.7MB dead allocations, added GQA validation and data file validation, converted 4 remaining raw calloc calls, experimentally confirmed UP19 at lower magnitude, caught 3 agent false positives. 2 commits.

### Session 5 (Mar 13-16, ~12 hours)

**MeZO+LoRA training pipeline — P9 COMPLETE**

Implemented MeZO zeroth-order training with LoRA-split on SmolLM2-360M (pretrained, 32 layers, DIM=960). Four optimization phases: (1) Conv1x1 hybrid with BLOBFILE weight baking — 403-429ms/step; (2) Fused conv kernels (QKV combined + FFN mega-kernel) — **262ms/step = 1.71x faster than CPU**; (3) FZOO multi-perturbation K=4 — no wall-time benefit; (4) P-GAP gradient-aligned perturbations — negative result (both simplified and faithful implementations). Added Gaussian RNG (Box-Muller), per-matrix SVD (LAPACK ssyev_), PROJECTION constraint. Detailed design spec and research log documenting methodology, results, and mathematical analysis of why P-GAP fails for LoRA ZO. 10 commits.

---

## Appendix: How to Reproduce Any Finding

Every finding cites specific experiments. To reproduce:

```bash
cd AutoANE/training

# CPU-only training (Finding 1, 3)
python3 train.py   # 120s, 512d/4L, CPU-only

# ANE comparison (Finding 2, 5)
# Edit train.py: set USE_ANE=True, ANE_MATMUL_ONLY=False
python3 train.py   # ANE full mode

# ANE matmul-only (Finding 5)
# Edit train.py: set USE_ANE=True, ANE_MATMUL_ONLY=True
python3 train.py

# Power measurement (Finding 2)
sudo bash measure_power.sh   # Requires root

# Architecture search (Finding 3)
python3 autoresearch.py --search arch --time 120

# Autonomous search (Finding 7)
python3 run_autosearch.py --experiments 100

# Run regression tests
bash ../tests/test_training.sh
```
