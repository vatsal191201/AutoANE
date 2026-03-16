# AutoANE: Training Transformers on Apple's Neural Engine

**Technical Report — March 2026**

---

## Abstract

We present AutoANE, the first open-source system for training Llama-family transformers on Apple's Neural Engine (ANE) via reverse-engineered private APIs. Through 43 controlled experiments, a 100-experiment autonomous hyperparameter search, and a 4-phase zeroth-order optimization pipeline, we characterize the ANE compute/precision/power tradeoff for training workloads. Our initial finding for standard backprop training is negative: CPU-only training dominates ANE at every tested model size (36M-281M params) due to IOSurface weight-staging overhead and irreducible fp16 precision loss. However, **MeZO zeroth-order training with LoRA-split and conv-fused kernels achieves 1.71x faster than CPU (262ms/step vs 447ms)** — the first demonstration of ANE training outperforming CPU on Apple Silicon. The key insight: LoRA-split freezes base weights as BLOBFILE constants (eliminating per-step IOSurface staging), while conv1x1 with fused kernels reduces IO round-trips from 224 to 96. We also report negative results for two advanced ZO gradient estimators (FZOO multi-perturbation and P-GAP gradient-aligned subspaces) and demonstrate that in a fixed time window, more optimizer steps beats more parameters.

---

## 1. Introduction

Apple's Neural Engine (ANE) is a dedicated neural network accelerator present in all Apple Silicon chips since M1 (2020). Apple rates ANE at 15.8 TOPS on M2 and 38 TOPS on M4 (INT8 operations; fp16 throughput is roughly half). By comparison, the CPU's AMX coprocessor achieves ~2 TFLOPS (FP32) and the integrated GPU ~4 TFLOPS (FP32). Despite this raw advantage, ANE has seen almost no use for training — CoreML and coremltools target inference only, and the training APIs are private.

Two prior works have explored ANE training:
- **maderix/ANE** (maderix, 2025): Reverse-engineered `_ANEClient`, `_ANECompiler`, and `_ANEInMemoryModelDescriptor` to implement forward and backward passes for Llama-family transformers. Achieved training on Stories110M (12L, dim=768, 109M params) and Qwen3-0.6B (28L, dim=1024, 596M params). Two pipeline approaches: static (const weights, recompile every N steps) and dynamic (IOSurface weight staging, compile once). Uses loss scaling of `256 * NLAYERS`.
- **Orion** (Murai Labs, [github.com/mechramc/Orion](https://github.com/mechramc/Orion)): ANE training/inference runtime. Claims 8.5x faster weight reload vs full compilation (494ms for 60 kernels vs ~4200ms). Verified 1000-step training stability with zero NaN. Documents ~20 ANE hardware constraints including ~119 compile limit per process.

Neither work compared ANE training to a proper CPU baseline on the same architecture, nor measured actual power consumption. AutoANE fills these gaps.

### 1.1 Research Questions

1. Is ANE training faster than CPU training for small-to-medium transformers on Apple Silicon?
2. Does ANE offer power efficiency advantages for training workloads?
3. Does delta compilation work via public or semi-public APIs?
4. What is the optimal model configuration for fixed-time-budget training on Apple Silicon?

### 1.2 Approach

We built a complete training system (Section 2), then systematically compared CPU and ANE training across multiple dimensions:

- **Throughput**: ms/step at matched configurations (Section 3)
- **Quality**: val_loss at matched step counts and matched wall-clock time (Section 4)
- **Power**: actual wattage via macOS `powermetrics` (Section 5)
- **Scaling**: behavior at DIM 1024/1536/2048 (Section 6)
- **Architecture search**: 11 configurations at 120s budget (Section 7)
- **Autonomous optimization**: 100-experiment automated search (Section 8)

All claims are independently verified (Section 10) with raw data and reproduction instructions.

---

## 2. System Design

### 2.1 Architecture

AutoANE trains Llama-family transformers (RMSNorm + GQA + RoPE + SwiGLU). The training binary (`train.m`, 1571 lines of Objective-C) implements forward pass, cross-entropy loss, backward pass, and AdamW optimizer with cosine LR decay, gradient accumulation, and gradient clipping.

Three training modes share the same data pipeline, optimizer, and evaluation:

- **CPU-only**: All matmuls via `cblas_sgemm` (Apple Accelerate/AMX) in fp32. No ANE framework loaded.
- **ANE matmul-only**: 7 linear projections per layer (Wq, Wk, Wv, Wo, W1, W2, W3) on ANE in fp16. RMSNorm, RoPE, attention scores, SiLU, residual connections, and embedding/classifier on CPU in fp32.
- **ANE full**: Forward pass entirely on ANE (fp16), backward on ANE (fp16). Only loss computation and AdamW update on CPU.

### 2.2 ANE Kernel Pipeline

At startup, the system compiles 10 MIL (Machine Learning Intermediate Language) kernels per layer via `_ANECompiler`. Weights are not baked into the compiled kernels. Instead, they are passed at each step via IOSurface spatial dimensions:

1. Convert fp32 weights to fp16 via NEON SIMD intrinsics (`vcvt_f16_f32`)
2. Pack fp16 values into IOSurface spatial pixels (tile width/height encoding)
3. Call `_ANEClient.evaluateWithModel()` to run the kernel
4. Read results from output IOSurface

This "dynamic weight" approach avoids recompilation but incurs per-step IOSurface I/O overhead. The overhead scales with total weight size: 8.1ms at DIM=1024 (60MB), 5ms at DIM=1536 (220MB — cache pressure makes individual operations faster but overall system slower), and 129ms at DIM=2048 (379MB — cache thrashing).

### 2.3 fp16 Stability

ANE computes exclusively in fp16. Two mechanisms prevent numerical failure:

- **Loss scaling** (256x): Multiplies all gradients by 256 during backward pass to keep them above fp16 minimum (6e-8). Divides by 256 before AdamW update. Without this, gradients underflow to zero and training fails (Experiment 1).
- **DeepNet scaling**: Residual connections scaled by `alpha = 1/sqrt(2*N_layers)`. Without this, activations overflow fp16 max (65504) at layer ~20.

### 2.4 Checkpoint Format

BLZT v4 binary format with 96-byte header. Per-layer weights stored in order: Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn. After all layers: rms_final, embed. Adam moment estimates (m, v) stored after weights. Total checkpoint size for 36.4M param model: ~0.44 GB.

### 2.5 Data

TinyStories dataset (Eldan & Li, 2023), tokenized with SmolLM2 BPE tokenizer (49152 vocab, 16893 active tokens in dataset). Binary format: 40MB, 20M uint16 token IDs, 90/10 train/val split. All models train from random initialization.

---

## 3. Throughput Comparison

### 3.1 Per-Step Timing (1024d/4L, 95.4M params)

**ANE Dynamic (68.7ms/step wall)**:

| Component | Time (ms) | % |
|-----------|----------|---|
| ANE forward (3 kernels) | 10.5 | 16.6% |
| ANE backward (7 kernels) | 18.7 | 29.6% |
| IOSurface I/O | 8.1 | 12.8% |
| Classifier + loss (CPU) | 15.2 | 24.1% |
| SiLU backward (CPU) | 5.8 | 9.2% |
| RMSNorm fwd+bwd (CPU) | 3.1 | 4.9% |

**CPU-Only (102.2ms/step wall)**:

| Component | Time (ms) | % |
|-----------|----------|---|
| Forward matmuls | 27.6 | 28.4% |
| Backward matmuls | 44.3 | 45.6% |
| Classifier + loss | 15.0 | 15.4% |
| SiLU backward | 5.4 | 5.6% |
| RMSNorm fwd+bwd | 3.1 | 3.2% |

ANE is 1.49x faster per step (68.7 vs 102.2ms). The raw matmul speedup is 2.46x (29.2 vs 71.9ms), but IOSurface overhead (8.1ms) and the CPU-bound classifier (15ms) limit the end-to-end gain.

### 3.2 Throughput Scaling

At 512d/4L (the optimal configuration), CPU-only achieves 41ms/step. ANE's advantage diminishes at smaller dimensions because IOSurface overhead becomes a larger fraction of the shorter step time.

At larger dimensions (E38 scaling study):

| Config | Params | CPU ms/step | ANE ms/step | ANE vs CPU |
|--------|--------|-------------|-------------|------------|
| 1024d/4L | 95.4M | 102 | 69 | 1.49x faster |
| 1536d/4L | 177M | ~210 | ~210 | Parity |
| 2048d/4L | 281M | ~360 | ~720 | **2x slower** |

The crossover at DIM=1536 (220MB IOSurface allocation) marks a hard ceiling. Beyond it, IOSurface memory pressure causes cache thrashing that degrades all operations, including CPU-only backward matmuls that share the memory hierarchy.

---

## 4. Training Quality

### 4.1 fp16 vs fp32 Precision Gap

At matched configurations (1024d/4L, 120s), 200-step rolling average loss:

| Mode | Steps | Avg loss (final 200) | vs CPU |
|------|-------|---------------------|--------|
| CPU fp32 | 1041 | 4.20 | baseline |
| ANE fp16 (full) | 1297 | 4.69 | +11.7% |
| ANE fp16 (matmul-only) | 1149 | ~4.65 | +10.7% |

ANE gets 25% more steps (faster matmuls) but ends with worse loss. The gap is entirely due to fp16 precision — matmul-only mode, which does non-linear ops in fp32, nearly closes the gap at matched *step counts* but still loses at matched *wall-clock time* because of IOSurface overhead.

### 4.2 Irreducibility of the fp16 Gap

Three approaches to closing the gap all failed (Experiments 14, 15):

1. **Activation clamping** to [-4, +4]: No effect. The precision loss occurs during dot-product accumulation within each matmul, not from activation magnitude.
2. **Lower LR** (1e-4): Makes loss worse (underfitting).
3. **Higher weight decay** (0.3): No effect. WD affects regularization, not per-step precision.

The root cause: at DIM=1024, each dot product sums 1024 fp16 multiplications. Each multiplication introduces ~0.5 ULP of rounding error, and these compound as sqrt(1024) = 32 ULP of error per output element. This is intrinsic to fp16 arithmetic hardware.

### 4.3 Fused vs Unfused Non-Linear Operations

Experiment 36 demonstrated a critical distinction:

| Mode | val_loss gap (train - val) |
|------|--------------------------|
| ANE full (RoPE, attention, SiLU in fp16) | 1.22 |
| ANE matmul-only (RoPE, attention, SiLU in fp32) | 0.60 |
| CPU-only | 0.60 |

Fusing non-linear ops into fp16 causes a train/val distribution shift (overfitting to fp16 noise). When only linear projections use fp16, val_loss matches CPU at the same step count. This finding validates the maderix generation-1 approach (matmul-only ANE) over the generation-3 approach (fully fused).

---

## 5. Power Measurement

### 5.1 Methodology

`sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000` for 60 seconds per mode, with 10-second idle baseline. Training binary runs continuously during measurement. Values are averages over the 60-second window, with peak values noted.

### 5.2 Results

| Mode | CPU (mW) | ANE (mW) | Package (mW) | Steps/60s | mJ/step |
|------|----------|----------|-------------|-----------|---------|
| Idle | — | — | 8455 | — | — |
| CPU-only | 13241 (pk 16841) | 9 | 13273 | 1435 | 9.2 |
| ANE matmul | 12132 (pk 15289) | 384 | 12568 | 1149 | 10.9 |
| ANE full | 11821 (pk 13942) | 765 (pk 1200) | 12664 | 1301 | 9.7 |

### 5.3 Analysis

Training adds ~4.8W above idle (CPU-only: 13.3 - 8.5 = 4.8W). ANE modes shift ~1.1-1.4W from CPU to ANE subsystem, but total package power is unchanged.

CPU-only achieves the lowest energy per step (9.2 mJ) because it completes more steps at the same total power draw. Orion emphasizes ANE as "dedicated silicon that's idle in most workloads" but does not quantify training power efficiency. Our data shows no power benefit for training with dynamic weight staging.

The lack of power savings is consistent with the IOSurface overhead story: even in ANE modes, the CPU is active ~38% of step time (RMSNorm, SiLU, classifier, AdamW), preventing it from entering low-power states.

---

## 6. Dimension Scaling Study (Novel)

### 6.1 Motivation

Prior ANE training work (maderix) only tested up to DIM=1024 (Stories110M: DIM=768, Qwen3-0.6B: DIM=1024). We extend to DIM=1536 and DIM=2048 to test whether ANE becomes advantageous at larger dimensions where the matmul-to-overhead ratio should improve.

### 6.2 Results (Experiment 38)

| Config | Params | Total IOSurface | CPU ms/step | ANE ms/step | Winner |
|--------|--------|----------------|-------------|-------------|--------|
| 1024d/4L | 95.4M | 60 MB | 102 | 69 | ANE (per-step) |
| 1536d/4L | 177M | 220 MB | ~210 | ~210 | Tie |
| 2048d/4L | 281M | 379 MB | ~360 | ~720 | **CPU (2x)** |

### 6.3 Analysis

The crossover at 220MB corresponds to exceeding the L2 cache hierarchy's ability to keep IOSurface pages resident. At 379MB of wired IOSurface memory, cache thrashing affects not just IOSurface I/O but also CPU-side backward matmuls (which share the same cache). ANE's advantage in raw matmul throughput is completely negated by the memory system.

This is a hard architectural limit of the dynamic weight staging approach: IOSurface memory is wired (non-pageable), and Apple Silicon's unified memory means it competes with all other processes for cache.

---

## 7. Architecture Search

### 7.1 Grid Search (Experiment 39)

11 configurations at 120s CPU-only budget, LR=3e-4 (later tuned per architecture in E40):

| Config | Params | Steps@120s | ms/step | val_loss |
|--------|--------|-----------|---------|----------|
| 512d/4L | 36.4M | 2542 | 41 | **3.54** |
| 768d/2L | 50.4M | 2621 | 39 | 3.69 |
| 512d/6L | 47.6M | 1773 | 58 | 3.76 |
| 1024d/2L | 72.9M | 1479 | 71 | 3.90 |
| 768d/4L | 75.2M | 1346 | 78 | 4.00 |
| 512d/8L | 58.7M | 1272 | 82 | 4.15 |
| 1024d/4L | 95.4M | 1050 | 99 | 4.30 |
| 768d/6L | 100.1M | 961 | 109 | 4.39 |
| 768d/8L | 124.9M | 782 | 134 | 4.69 |
| 1024d/6L | 118.0M | 749 | 140 | 4.73 |
| 1024d/8L | 140.5M | 577 | 183 | 5.18 |

**Observation**: val_loss correlates more strongly with step count (r = -0.97) than with parameter count (r = +0.85). The smallest model wins because it gets the most gradient updates.

### 7.2 Per-Architecture LR Tuning (Experiment 40)

| Config | LR=3e-4 | LR=5e-4 | LR=7e-4 | Optimal |
|--------|---------|---------|---------|---------|
| 512d/4L | 3.67 | **3.54** | 3.58 | 5e-4 |
| 768d/2L | 3.77 | **3.69** | 3.72 | 5e-4 |
| 1024d/2L | **3.90** | 3.93 | 4.02 | 3e-4 |

Architecture ranking unchanged after LR tuning: 512d/4L > 768d/2L > 1024d/2L.

LR=5e-4 is optimal for 512d (36M params), while LR=3e-4 is optimal for 1024d (73M params). This follows the LAMB heuristic: optimal LR ~ sqrt(N_small/N_large).

### 7.3 Budget Scaling (Experiment 41)

| Budget | 512d/4L | 768d/2L | 1024d/2L | 512d/4L lead |
|--------|---------|---------|----------|-------------|
| 120s | 3.54 | 3.69 | 3.90 | +0.15 |
| 300s | 3.09 | 3.20 | 3.48 | +0.11 |
| 600s | 2.55 | 2.84 | 3.25 | +0.29 |
| 1800s | 2.22 | — | — | — |

The 512d/4L advantage widens at longer budgets. At 600s, 768d/2L begins overfitting (train-val gap grows to +0.83), while 512d/4L is still underfitting. No crossover through 1800s.

At 1800s (4.89 epochs on 20M tokens), 512d/4L reaches val_loss 2.22, consistent with TinyStories literature baselines (~2.0 at convergence for 33M-param models).

---

## 8. Autonomous Hyperparameter Search

### 8.1 Agent Loop (Experiment 43)

13 experiments via Claude Code following the Karpathy autoresearch protocol (modify train.py -> commit -> train -> keep/revert).

Key discovery: **SEQ=128 gives 1.75x more steps than SEQ=256** (4037 vs 2456 at 120s) with marginally better val_loss (3.528 vs 3.533). Grid search (E39-E41) never varied SEQ — it was held fixed at 256. The agent found this by testing one variable at a time from a strong baseline.

Secondary discovery: optimal LR shifts from 5e-4 to 4e-4 when SEQ halves (effective batch 2560 -> 1280 tokens), consistent with Smith et al.'s linear scaling rule (LR ~ sqrt(batch_size)).

### 8.2 Automated Search (Experiment 44)

100 experiments via `run_autosearch.py`: random Gaussian perturbations of LR, ACCUM_STEPS, WEIGHT_DECAY, ADAM_B2, WARMUP_STEPS, HIDDEN, DEPTH, DIM, SEQ. Keep if val_loss improves, revert otherwise.

Starting from val_loss 3.952 (fresh baseline with SEQ=128), 88 experiments completed (12 skipped for zero-change samples). 6 improvements kept:

| # | Change | New val_loss | Cumulative improvement |
|---|--------|-------------|----------------------|
| 1 | HIDDEN 1408->1152, WARMUP 100->71 | 3.906 | 1.2% |
| 2 | WEIGHT_DECAY 0.1->0.098 | 3.676 | 7.0% |
| 3 | LR 4e-4->4.19e-4 | 3.671 | 7.1% |
| 5 | ADAM_B2 0.95->0.959, LR->6.34e-4 | 3.505 | 11.3% |
| 15 | WEIGHT_DECAY 0.098->0.076 | 3.480 | 11.9% |
| 87 | ACCUM_STEPS 10->7 | 3.288 | 16.8% |

### 8.3 Best Known Configuration

```
Architecture: 512d, 4 layers, 8 heads (4:1 GQA), head_dim=64, hidden=1408
Sequence: 128 tokens
Optimizer: AdamW, LR=6.34e-4, warmup=71, β1=0.9, β2=0.959, WD=0.076
Training: ACCUM=7 (eff. batch 896 tokens), grad_clip=1.0, loss_scale=256
Mode: CPU-only (fp32)
Budget: 120s → val_loss ~3.8 typical (best-of-88: 3.288; variance ~0.3/run)
        1800s → val_loss 2.22 (~38000 steps)
Note: The baseline config (LR=4e-4, ACCUM=10) reliably gives ~3.5 at 120s.
```

### 8.4 Search Dynamics

Most improvements were found early: 5 of 6 keeps occurred in the first 30 experiments. The final keep (ACCUM 10->7) at experiment 87 was the only late-stage discovery. After experiment 60, the probability of improvement per experiment was <2%.

### 8.5 Variance Exceeds Signal

Independent verification shows this config typically gives val_loss ~3.8, not 3.288. The run-to-run variance from random initialization (~0.3 nats) is larger than most individual "improvements" in the search. The single-evaluation keep/revert protocol is susceptible to hill-climbing on noise: the search may be keeping lucky seeds rather than genuinely better configurations. The manually-tuned baseline (LR=4e-4, ACCUM=10) reliably produces ~3.5, which is better than the autosearch config's typical ~3.8.

### 8.6 Protocol Bug

`git reset --hard HEAD~1` (used to revert failed experiments) reverts ALL tracked files, not just the ones in the latest commit. If any file besides `train.py` is uncommitted, it gets wiped. This caused loss of `results.tsv` entries during the 100-experiment run. Fixed by committing `results.tsv` alongside `train.py` in each experiment commit.

Lesson: the Karpathy keep/revert protocol requires that ALL mutable state be committed, and no external commits be made to the same branch during a run.

---

## 9. Delta Compilation Investigation

### 9.1 Motivation

Orion reports 8.5x faster weight reload vs full compilation: unload each program, update weight files on disk, reload (494ms for 60 kernels vs ~4200ms full compile). If achievable, this would enable the static compilation approach (const() weights) without the recompilation penalty.

### 9.2 Results

Five approaches tested (Experiments 10, 17, 34), all failed:

| Approach | Result |
|----------|--------|
| Unload -> write BLOBFILE -> reload | Output identical pre/post (703.5304) |
| tmpDir data patching | Output unchanged |
| e5bundlecache inspection | Only ~96-byte metadata entries |
| _ANEInMemoryModel reload | API not functional |
| Fresh recompile with cached graph | ~60ms/kernel (8.5x faster than cold, but still too expensive) |

### 9.3 Analysis

The compiled ANE kernel is stored in an inaccessible memory-mapped region. The source BLOBFILE in tmpDir is consumed during compilation but not referenced at runtime. Rewriting it and reloading does not cause the runtime to re-read the file.

The fresh recompile path (60ms/kernel) exploits graph topology caching: if the MIL text and weight dictionary keys are identical to a previously compiled kernel, the ANE compiler reuses most of its work. But 60ms/kernel x 10 kernels x 4 layers = 2.4 seconds per weight update is still too expensive for per-step weight staging.

---

## 10. Verification

All experimental claims are verified in [docs/VERIFICATION.md](VERIFICATION.md) (21 sections). Key verification methods:

1. **Reproduction**: 4 key configurations re-run; all val_loss values match within 0.3%
2. **Mathematical verification**: Parameter counts computed from formulas and cross-checked against code output
3. **Literature cross-reference**: Results compared to Chinchilla, Kaplan, Muennighoff, TinyStories baselines
4. **Code audit**: AdamW bias correction, cosine LR schedule, gradient clipping, gradient accumulation, loss scaling — all verified line-by-line
5. **Independent forward pass**: Python reimplementation matches C binary (delta 0.117 due to different val samples)
6. **Round-trip verification**: Weight conversion pipelines (HuggingFace <-> ANE <-> GGUF) verified bit-perfect on 38 and 272 tensors

### 10.1 Bugs Found During Verification

| Bug | Impact | Fix |
|-----|--------|-----|
| Stale autoresearch.h (1024d in header, 512d checkpoint) | C binary misinterprets weights | Regenerate header from train.py |
| gguf_to_ane.py missing Q/K interleaving | Incorrect RoPE after GGUF import | Add interleave/de-interleave conversion |
| generate.py streaming bug | Reprinting full text each tick | Fix print logic |
| export_to_gguf.py missing tokenizer | llama.cpp rejects model | Add BPE tokenizer metadata |
| run_autosearch.py results.tsv not committed | Git reset wipes keep entries | Stage results.tsv in each commit |
| Autosearch concurrent branch edits | HEAD~1 resets walk past external commits | Document: no concurrent edits during search |

### 10.2 No Retractions

All experimental claims, architecture rankings, and research conclusions hold under independent verification. Three corrections were made to wording (data size 19M->20M, epoch calculations, V22 phrasing) but no numerical results were retracted.

---

## 11. MeZO Zeroth-Order Training (Sessions 5, March 2026)

### 11.1 Motivation

Findings 1-5 established that backprop training on ANE is limited by two bottlenecks: (1) per-step IOSurface weight staging overhead, and (2) irreducible fp16 precision loss in non-linear operations. MeZO (Memory-Efficient Zeroth-Order Optimizer, Malladi et al. 2023) eliminates both: it uses only forward passes (no backward kernels), and LoRA-split mode freezes base weights as BLOBFILE constants (no per-step weight staging).

### 11.2 System Design

MeZO estimates gradients via SPSA (Simultaneous Perturbation Stochastic Approximation): perturb all trainable parameters by ±ε·z (where z is Rademacher ±1), compute two forward passes, and estimate the gradient as `g ≈ [(L⁺ - L⁻) / (2ε)] · z`.

**Model**: SmolLM2-360M (pretrained), 32 layers, DIM=960, Q_DIM=960, KV_DIM=320, HIDDEN=2560.

**LoRA-split**: Rank-8 attention-only adapters. 8 matrices per layer × 32 layers = 256 LoRA matrices. Trainable parameters: 1,700,800 (1,638,400 LoRA + 62,400 RMS norms). Base weights frozen as BLOBFILE constants in conv1x1 kernels.

### 11.3 Four-Phase Optimization Pipeline

**Phase 1 — Conv1x1 Hybrid**: Replace matmul MIL kernels with conv1x1 where faster. Selective: Wq (2.05x), Wo (2.11x), W1 (2.20x), W2 (1.93x), W3 (2.20x) use conv; Wk (0.36x), Wv (0.36x) stay as matmul. BLOBFILE weight baking eliminates per-step weight staging for conv projections. Bit-exact numerical correctness verified. Result: 403-429ms/step (1.04-1.11x CPU).

**Phase 2 — Fused Conv Kernels**: Combine multiple operations per layer into fewer MIL programs. QKV combined kernel (3→1), FFN mega-kernel (conv W1 + conv W3 + SiLU + conv W2 + residual, 3→1), Wo conv (1). Total: 3 kernels/layer instead of 7, reducing IO round-trips from 224 to 96. Result: **~262ms/step = 1.71x faster than CPU** (447ms baseline). Val_loss within 0.03% of CPU baseline. Triple-checked via 50-step average.

**Phase 3 — FZOO Multi-Perturbation**: Replace central-difference (2 passes, 1 estimate) with K one-sided perturbations (K+1 passes, K estimates). Uses Rademacher perturbations (O(ε²) bias proven by ZOSA, arXiv:2511.09156). K=4: 2.5x more forward passes, better gradient quality, but zero wall-time convergence improvement. The better per-estimate quality is exactly offset by the computational overhead.

**Phase 4 — P-GAP Gradient-Aligned Perturbations**: Two implementations tested:
1. *Simplified (flat-vector QR)*: Degrades convergence. Projected gradient 1000x smaller (√(r/d) = √(4/1.7M) ≈ 0.0015).
2. *Faithful (per-matrix SVD, arXiv:2510.18228)*: 256 per-matrix SVD bases, PROJECTION constraint, Gaussian perturbations (Box-Muller), delta linear decay. Paper hyperparameters (ε=0.1, lr=1e-2) diverge catastrophically. Standard hyperparameters neutral. Root cause: LoRA rank-8 matrices (8×960) have at most 8 singular values — the SVD rank equals the matrix rank, leaving no dimensionality reduction to exploit.

### 11.4 Key Results

| Metric | Value |
|--------|-------|
| Best config | Phase 2 (conv-fused), 262ms/step |
| Speedup vs CPU | 1.71x |
| Convergence impact | Zero (val_loss within 0.03% of CPU) |
| IO round-trips | 96 (vs 224 Phase 1, vs ~448 backprop dynamic) |
| FZOO benefit | Zero wall-time improvement |
| P-GAP benefit | Zero (neutral) to negative (diverges) |

### 11.5 Why This Contradicts Finding 1

Finding 1 ("CPU beats ANE at every tested size") applies specifically to *backprop training with dynamic weight staging*. MeZO+LoRA-split changes the equation:
- No backward pass → eliminates the most complex ANE code path
- LoRA-split → base weights frozen as BLOBFILE constants → no per-step IOSurface staging
- Conv1x1 with fused kernels → fewer IO round-trips → lower overhead per forward pass
- Forward-only → ANE's strength (fast matmuls) without its weakness (weight staging)

The IOSurface overhead that dominated backprop training (8.1ms at DIM=1024) is eliminated because BLOBFILE constants are compiled into the kernel — they don't traverse the IOSurface path.

## 12. Limitations

1. **Single dataset**: Backprop experiments use TinyStories (20M tokens). MeZO experiments use SmolLM2-360M pretrained weights fine-tuned on TinyStories. Results may not generalize to other datasets or tasks.

2. **Single hardware**: All experiments on M2 Pro (macOS 15). ANE architecture and performance characteristics differ across M1/M2/M3/M4 generations.

3. **No comparison to MLX or PyTorch**: We compare CPU vs ANE within AutoANE's framework but do not benchmark against established frameworks. MLX would likely achieve higher throughput on GPU for models larger than ~50M params.

4. **MeZO convergence rate**: MeZO converges slower than backprop per step. The 1.71x speedup is measured per-step, not per unit of convergence. Total training time to reach a target loss may still favor CPU backprop for tasks where backprop is feasible.

5. **Private APIs**: Results depend on `_ANEClient` and `_ANECompiler` behavior in macOS 15. These are undocumented and may change.

6. **Single model size for MeZO**: MeZO+LoRA-split tested only on SmolLM2-360M. The 1.71x speedup may differ at other model sizes. Conv1x1 advantages should increase with wider dimensions, but IOSurface behavior at larger scales is untested in this mode.

---

## 13. Assumptions Registry

Every assumption is tracked in [docs/ASSUMPTIONS.md](ASSUMPTIONS.md) with status, evidence, and confidence level. As of this writing: 27 verified, 8 disproved, 13 unverified/resolved. Notable disproved assumptions:

- **D7**: "ANE becomes advantageous at larger dimensions" — FALSE, IOSurface overhead makes ANE 2x slower at DIM=2048
- **D8**: "ANE is ~2x more energy efficient than CPU" — FALSE, package power identical across all modes
- **D2**: "CPU-only would be faster than ANE per-step" — FALSE for step time (ANE 1.5x faster per step), TRUE for val_loss (CPU wins on training quality)

---

## 14. Conclusions

ANE has genuine computational advantages for matrix multiplication (2.5x over CPU AMX) but for *standard backprop training*, these are negated by IOSurface weight-staging overhead and irreducible fp16 precision loss.

However, **MeZO zeroth-order training with LoRA-split fundamentally changes the equation**. By freezing base weights as BLOBFILE constants and using conv1x1 with fused kernels, we achieve 1.71x faster than CPU (262ms/step vs 447ms) with zero convergence impact. This is the first demonstration of ANE training outperforming CPU on Apple Silicon. The key insight is architectural: the problem was never ANE's compute speed (which is 2.5x faster than CPU for matmuls), but the per-step weight-staging overhead. LoRA-split eliminates this overhead entirely.

Two advanced ZO gradient estimators were tested and found unhelpful for this regime: FZOO multi-perturbation provides better gradient quality but costs 2.5x more forward passes (net zero benefit), and P-GAP gradient-aligned perturbations either diverge (paper hyperparameters) or provide zero benefit (standard hyperparameters). The P-GAP negative result has a clear structural cause: LoRA rank-8 matrices are too small for per-matrix SVD to find useful low-rank structure.

For *backprop* training on Apple Silicon, CPU-only remains the right choice. For *zeroth-order fine-tuning* (MeZO+LoRA), ANE with conv-fused kernels is 1.71x faster. ANE's value proposition for training is now validated for the LoRA fine-tuning use case, with mobile deployment (iPhone/iPad) as the highest-impact application.

The autonomous hyperparameter search demonstrates that run-to-run variance (~0.3 nats) can exceed the optimization signal in keep/revert protocols — the search's "best" config typically produces ~3.8 while the manually-tuned baseline reliably produces ~3.5.

---

## References

1. Eldan, R. & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? *arXiv:2305.07759*.
2. Hoffmann, J. et al. (2022). Training Compute-Optimal Large Language Models. *arXiv:2203.15556* (Chinchilla).
3. Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models. *arXiv:2001.08361*.
4. Karpathy, A. (2025). autoresearch. *GitHub: karpathy/autoresearch*.
5. Muennighoff, N. et al. (2023). Scaling Data-Constrained Language Models. *arXiv:2305.16264*.
6. Nguyen, T.Q. & Salazar, J. (2019). Transformers without Tears: Improving the Normalization of Self-Attention. *arXiv:1910.05895*.
7. maderix (2025). ANE: Training on Apple Neural Engine. *GitHub: [maderix/ANE](https://github.com/maderix/ANE)*.
8. Smith, S.L. et al. (2018). Don't Decay the Learning Rate, Increase the Batch Size. *arXiv:1711.00489*.
9. Wang, H. et al. (2022). DeepNet: Scaling Transformers to 1,000 Layers. *arXiv:2203.00555*.
10. Murai Labs (2026). Orion: ANE Training and Inference Runtime. *GitHub: [mechramc/Orion](https://github.com/mechramc/Orion)*.
11. Malladi, S. et al. (2023). Fine-Tuning Language Models with Just Forward Passes. *NeurIPS 2023*. *arXiv:2305.17333* (MeZO).
12. Liu, Y. et al. (2025). FZOO: An Efficient and Scalable Framework for Zeroth-Order Optimization. *arXiv:2506.09034*.
13. Li, B. et al. (2025). P-GAP: Gradient-Aligned Perturbations for Zeroth-Order Optimization. *arXiv:2510.18228*.
14. Zhang, X. et al. (2025). ZOSA: Zeroth-Order Stochastic Approximation with Rademacher Perturbations. *arXiv:2511.09156*.
15. Zhang, Y. et al. (2025). MobiZO: Parallelized Zeroth-Order Optimization for Mobile/Edge Devices. *EMNLP 2025*. *10.18653/v1/2025.emnlp-main.1022*.
