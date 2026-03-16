# AutoANE Literature Review: NPU Training Landscape (March 2026)

**Date**: 2026-03-16
**Scope**: All published work on NPU/ANE training, on-device LLM fine-tuning, and related techniques
**Purpose**: Ground our research in existing literature, identify gaps, cross-reference claims

---

## 1. ANE Training Systems

### 1.1 Orion (arXiv:2603.06728, March 2026)
**First end-to-end ANE training system.** Bypasses CoreML via _ANEClient/_ANECompiler.
- GPT-2 124M inference: 170+ tok/s on M4 Max
- Stories110M training: 1000 steps in 22 min, 107ms/step, zero NaN
- Delta compilation: 4200ms -> 494ms (8.5x) via unload/patch/reload
- 20 ANE constraints cataloged (14 previously undocumented)
- Power: 6.6 TFLOPS/W at 2.8W (NOTE: our measurement shows 1.2W — discrepancy)
- **Our status**: Referenced extensively. We share the maderix codebase ancestry. Our MeZO approach avoids their compilation bottleneck entirely.

### 1.2 maderix/ANE (GitHub, 2025-2026)
**Our direct upstream.** Reverse-engineered ANE private APIs.
- 5.3k stars, 798 forks (March 2026)
- Stories110M: 91ms/step, Qwen3-0.6B: 412ms/step (dynamic pipeline)
- 48 PRs, 12 issues reviewed in Session 3
- Key PRs: #22 (function params 30% faster), #24 (mega-kernel 3-4x), #40 (chaining dead)
- **Our status**: Fully integrated. Our MeZO builds on their training_dynamic pipeline.

### 1.3 imperatormk/ane-train
**Runtime weight injection.** Weights as IOSurface inputs, not constants.
- Compiles once, never recompiles
- Our dynamic pipeline already uses this approach
- **Our status**: Architecture adopted (UP1).

### 1.4 thebasedcapital/ane-infer
**_ANEChainingRequest works** (error was wrong factory method).
- But: requires Espresso IR from disk-compiled models (maderix PR #40)
- Dead for in-memory MIL path
- **Our status**: Confirmed dead (P3 invalidated).

---

## 2. On-Device LLM Fine-Tuning

### 2.1 Apple MeBP (arXiv:2510.03425, October 2025)
**Memory-efficient backpropagation on iPhone.** Apple Research.
- Gradient checkpointing + lazy weight decompression + memory-mapped activations
- Qwen2.5/Gemma3 0.5B-4B on iPhone 15 Pro Max in <1GB
- MeBP vs MeZO per step: 1.5-2x slower per step, but 10-100x fewer steps
- Runs on CPU/GPU only (NOT ANE)
- **Our analysis**: See docs/2026-03-16-mebp-cross-reference.md. Does NOT invalidate our approach. MeBP+ANE hybrid (P16) is the most promising new direction.

### 2.2 Apple LCSB (arXiv:2602.13073, February 2026)
**Layer-Cyclic Selective Backpropagation.** Apple follow-up to MeBP.
- Selective backprop through cycling layer subsets
- Further memory reduction
- **Our status**: Not yet analyzed in detail. Likely complementary to P16.

### 2.3 Apple Memory-Efficient Structured Backprop (arXiv:2602.13069, February 2026)
**Structured backpropagation.** Apple parallel work.
- Different approach to same problem (memory-efficient on-device backprop)
- **Our status**: Not yet analyzed.

### 2.4 MobileFineTuner (arXiv:2512.08211, December 2025)
**C++ on-device LLM fine-tuning framework.**
- GPT-2, Gemma3, Qwen2.5 on real phones
- ~16GB RAM per 1B params FP16
- Parameter sharding, gradient accumulation, energy-aware scheduling
- Runs on CPU (no NPU)
- **Our status**: Informational. Could provide reference CPU backward pass implementation.

### 2.5 MobiLLM (arXiv:2502.20421, February 2025)
**Server-assisted side-tuning.**
- Mobile device runs frozen backbone (forward only)
- Server handles backprop for trainable side-network
- Privacy: one-way activation transfer with quantization
- **Our status**: Different paradigm (requires server). ANE approach is fully on-device.

### 2.6 FwdLLM (USENIX ATC 2024)
**BP-free federated LLM fine-tuning.** 1.5GB peak for LLaMA-7B.
- Forward-only via perturbed inferences (ZO-like)
- 14.6x memory reduction
- **Our status**: Validates ZO approach for on-device training (UP20).

### 2.7 MobiEdit (ICLR 2026)
**Quantized forward-only gradient estimation for NPUs.**
- W8A16, 80% edit success
- 7.1x memory, 3.4x latency, 15.8x energy reduction
- **Our status**: Complementary to MeZO (UP21). INT8 quantization aligns with P13.

---

## 3. Zeroth-Order Optimization

### 3.1 MeZO (arXiv:2305.17333, NeurIPS 2023)
**Our primary algorithm.** Memory-efficient ZO for LLM fine-tuning.
- SPSA gradient estimation via perturbation
- Memory: same as inference (no gradients/optimizer state beyond perturbation seed)
- **Our status**: Fully implemented. 1.71x faster than CPU on ANE (Phase 2).

### 3.2 FZOO (arXiv:2506.09034, 2025)
**Multi-perturbation ZO with adaptive step size.**
- K one-sided Rademacher perturbations
- Sigma-normalized updates
- **Our status**: Tested (Phase 3). No wall-time benefit for LoRA ZO.

### 3.3 P-GAP (arXiv:2510.18228, 2025)
**Gradient-aligned perturbations via per-matrix SVD.**
- **Our status**: Tested (Phase 4). Negative for LoRA ZO. Root cause: LoRA rank-8 too small for SVD.

### 3.4 Sparse MeZO (NeurIPS 2025)
**Magnitude-based parameter selection.**
- 3.5x speedup, +9% accuracy (full-parameter)
- **Our status**: Tested (Phase 5a). NEGATIVE for LoRA ZO. -31% to -87% worse.

### 3.5 HiZOO (ICLR 2025)
**Hessian-informed diagonal preconditioning.**
- 8x speedup (full-parameter), ~1% for LoRA
- **Our status**: Tested (Phase 5b). NEGATIVE for LoRA ZO. -34% to -82% worse.

### 3.6 ZOSA (arXiv:2511.09156)
**Proved one-sided Rademacher has O(e^2) bias.**
- Key theoretical result we use for FZOO justification
- **Our status**: Cited, theory verified in our FZOO implementation.

### 3.7 ElasticZO (January 2025)
**INT8 ZO for edge devices.**
- Could leverage ANE's 1.88x INT8 throughput
- **Our status**: Not yet tested. Aligns with P13.

---

## 4. NPU Training Landscape

### 4.1 Apple ANE
- Only unofficial training via reverse-engineered APIs
- Two independent systems: maderix/ANE + Orion
- Our AutoANE: first MeZO on ANE, first ANE-faster-than-CPU training
- CoreML official path: too limited (updatable layers only, no arbitrary backprop)

### 4.2 Huawei Ascend
- **Only vendor with official NPU training** (CANN + PyTorch/MindSpore)
- Ascend 910B+ supports distributed training
- torchtune integration (2025)
- Fundamentally different: designed for training from ground up

### 4.3 Qualcomm Hexagon
- Inference only (QNN framework)
- 45 TOPS (Snapdragon 8 Elite)
- No training APIs

### 4.4 Google Edge TPU
- Last-layer transfer learning only
- 4 TOPS at 2W
- Cannot train full networks

### 4.5 Samsung Exynos NPU
- Inference only
- ExecuTorch support for PyTorch models

### 4.6 MediaTek APU
- Inference only (LiteRT NeuroPilot)
- 50+ TOPS in newer SoCs

### Summary Table

| Platform | Training | Max Demonstrated | Our Relation |
|----------|---------|-----------------|-------------|
| **Apple ANE (us)** | Unofficial (MeZO) | 360M, 1.71x CPU | Primary |
| Apple ANE (Orion) | Unofficial (backprop) | 110M, 22 min/1K steps | Referenced |
| Huawei Ascend | Official | Distributed 910B+ | Analogous |
| Qualcomm | None | N/A | — |
| Google Edge TPU | Last-layer only | — | — |
| Samsung | None | N/A | — |
| MediaTek | None | N/A | — |

---

## 5. Upcoming: Core AI Framework

9to5Mac and AppleInsider (March 2026) report Apple will announce "Core AI" at WWDC 2026 (June) as a replacement/evolution of CoreML for iOS 27.

**If Core AI exposes ANE training APIs**: Our private-API approach becomes unnecessary. The entire AutoANE project would shift to using official APIs.

**If Core AI is inference-only**: AutoANE remains the only path for ANE training. Our findings become more valuable.

**Action**: Monitor WWDC 2026 closely (UP29).

---

## 6. Key Takeaways for AutoANE

1. **We are the only MeZO-on-NPU implementation.** No other project combines ZO optimization with NPU hardware.

2. **MeBP is complementary, not competing.** It runs on CPU/GPU; we run on ANE. Hybrid (P16) could combine both strengths.

3. **LoRA ZO has different dynamics than full-param ZO.** Phases 3-5 confirm that techniques designed for full-param ZO do not transfer. This is a structural finding the community needs to know.

4. **The 1.71x ANE speedup is the strongest result.** No other project achieves ANE-faster-than-CPU for training.

5. **Core AI framework is the strategic risk.** If Apple provides official ANE training, our reverse-engineering work becomes a stepping stone rather than the final solution.
