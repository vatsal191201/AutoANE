# AutoANE: Stated Assumptions Registry

**Purpose**: Every assumption in this project must be explicitly stated, tracked, and verified. No implicit assumptions allowed. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) Section 12 for summary.

**Last updated**: 2026-03-12 | **Totals**: 26 verified, 1 qualified, 8 disproved, 13 unverified/resolved, 23 new from upstream

---

## Category: RETESTING (previously classified, needs re-verification)

| ID | Assumption | Prior Status | Why Retesting |
|----|-----------|-------------|---------------|
| SA1 | fp16 precision gap (~16%) is irreducible via software | **RE-CONFIRMED (E19)** | Zero NaN/Inf in 5777 steps. Gap is genuine fp16 accumulation rounding, not fixable by sanitization. |
| SA2 | ANE training diverges after ~2000 steps | **DISPROVED (E19-ext)** | 5777 steps, loss still improving (avg ~4.59). Exp 18 "divergence" was thermal throttling from running concurrent tests on same machine. |
| SA3 | Activation clamping to [-4,4] doesn't improve loss | VERIFIED (Exp 14) | Still needs testing: activations reach [-121, 127] at 10min — approaching fp16 max (65504). Overflow clamping [-65504, +65504] untested. |
| SA4 | Delta compilation not viable | **RE-CONFIRMED (E34)** but **NUANCED by UP1**: _ANEClient.loadModelNewInstance fails for baked-weight models. However, runtime weight injection (weights as IOSurface inputs, not const()) eliminates the need for delta compilation entirely. Our dynamic pipeline already does this. The bottleneck is IOSurface overhead + fp16 precision, not the compilation model. |

## Category: VERIFIED (confirmed by experiment)

| ID | Assumption | Evidence | Confidence |
|----|-----------|----------|------------|
| V1 | ANE matmul ~2.5x faster than CPU AMX | Exp 11: 29.2ms vs 71.9ms | HIGH |
| V2 | Conv 1x1 is 1.5-2.8x faster than matmul on ANE | Exp 9: microbenchmark | HIGH |
| V3 | CPU AMX achieves 1.5-2.5 TFLOPS fp32 | Exp 8: cblas_sgemm benchmark | HIGH |
| V4 | IOSurface overhead is ~13% of step time at DIM=1024 | Exp 7: 8.1ms of 63.2ms. **CLARIFIED (E38)**: scales dramatically — 5ms at DIM=1536, 129ms at DIM=2048. Not a fixed percentage. | CLARIFIED |
| V5 | Mixed precision (ANE fwd + CPU bwd) worse than pure | Exp 11: 5.10 vs 4.90 avg loss. **CLARIFIED (E36)**: the issue is fused non-linear ops in forward, not mixed precision itself. ane-matmul-only (ANE fwd matmul + CPU everything else) matches CPU. | CLARIFIED |
| V6 | Loss scaling (256.0) essential for ANE fp16 | Exp 1: gradients underflow without it | HIGH |
| V7 | Shallow/wide beats deep/narrow in fixed time | Exp 2: 4L/1024d beats 32L/960d | HIGH |
| V8 | LoRA from pretrained works on ANE | Exp 3: stable training, loss 4.22 | HIGH |
| V9 | Classifier row-major optimization ~2x faster | Exp 13: 30.5ms → 15.7ms | HIGH |
| V10 | Both modes suffer ~1.5x thermal throttling at 10min | CORRECTED — single-process throttling is 30% (1.3x), not 50% (1.5x). E18's 1.5x was from concurrent processes. | CORRECTED (E24) |
| V11 | ANE only faster than CPU for large matmul shapes (2816+ width) | E23: 1024x1024 CPU 0.67x faster, 2816x1024 ANE 1.88x faster. **CAVEAT (E38)**: raw matmul speedup exists but IOSurface overhead at larger dimensions negates it. Net effect: no throughput advantage at any tested dimension. | CLARIFIED |
| V14 | Dynamic weight IOSurface approach has hard scaling ceiling at ~220MB total surfaces | E38: DIM=1536 (220MB) parity, DIM=2048 (379MB) ANE 2x slower. IOSurface memory pressure causes cache thrashing in ALL operations including CPU-only backward pass. | HIGH |
| V15 | CPU-only is correct default for ALL model sizes with dynamic weight approach | E38: tested 95M-281M params. ANE never faster, and dramatically slower at DIM=2048. | HIGH |
| V16 | At 120s CPU-only budget, smaller/shallower models achieve lower val_loss | E39: 512d/4L (36.4M) val_loss 3.61 beats all 11 configs tested. Step count (2471 vs 577 for 1024d/8L) is the dominant factor. Depth strictly hurts at every width. | HIGH |
| V17 | Depth is strictly harmful at fixed short time budgets (120s) | E39: at every width tested, adding layers increases val_loss. 512d: 4L→8L costs +0.61. 768d: 2L→4L→6L costs +0.30/+0.12. 1024d: 2L→4L→6L→8L costs +0.44/+0.31/+0.45. | HIGH |
| V18 | Optimal LR is 5e-4 for small models (512d), 3e-4 for larger (1024d) | E40: 512d/4L improves from val 3.67→3.54 at LR 5e-4. 1024d/2L unchanged at 3e-4. Classic scaling law behavior. | HIGH |
| V19 | Architecture ranking (E39) is robust to per-architecture LR tuning | E40: 512d/4L > 768d/2L > 1024d/2L with optimal LRs. Order unchanged. Resolves SA-E39-1. | HIGH |
| V20 | 2-layer models overfit heavily despite high throughput | E40: 768d/2L train-val gap +0.83 at optimal LR. E41: confirmed at 300s and 600s. Depth aids generalization even when it costs throughput. | HIGH |
| V21 | 512d/4L advantage increases with longer training budgets (120-600s) | E41: lead grows from 0.15 (120s) to 0.29 (600s). Larger models overfit while 512d/4L is still underfitting at 600s. | HIGH |
| V22 | Data volume, not model capacity, is the bottleneck at our training scale | E41: at ~31M tokens (600s), 36.4M param model sees 0.86 tokens/param — 23× below Chinchilla's 20:1 optimum. 72.9M param model needs ~73M tokens but only sees ~21M. | HIGH |
| V23 | SEQ=128 is optimal for 120s budget (throughput dominates context) | E43: SEQ=128 gives 1.75× throughput (24ms vs 42ms/step), 4037 vs 2456 steps. Val improves 3.533→3.528 despite halved context. SEQ=64 degrades (too short for coherent gradients). | HIGH |
| V24 | Optimal LR co-varies with effective batch size (linear scaling rule) | E43: halving SEQ (256→128) halves effective batch (2560→1280 tokens). LR must decrease: 5e-4→4e-4 optimal. Consistent with Smith et al. (2018): LR_opt ∝ sqrt(batch_size). | HIGH |
| V25 | ACCUM=10 is optimal; deviations in either direction hurt | E43: ACCUM=5 (noisier gradients, val +0.17) and ACCUM=20 (fewer updates, val +0.30) both worse than ACCUM=10. The 10-step accumulation balances gradient quality against update frequency. | HIGH |
| V12 | Fusing non-linear ops into ANE fp16 causes train/val distribution shift | E36: val gap 1.218 (ane-full) vs 0.599 (ane-matmul-only). Identical val_loss to CPU at matched steps. | HIGH |
| V13 | ANE should only be used for linear projections (matmul), not attention/SiLU/residual | E36: matches original maderix/ANE gen1 design. Step-10 loss matches CPU to 4 decimal places. **NOTE**: original gen3 (dynamic pipeline) fuses more into ANE — our approach is more conservative. | HIGH |
| V26 | ANE provides no power savings over CPU-only for training | Powermetrics data (2026-03-12): package power 12.6-13.3W for all modes. CPU-only is most energy-efficient per step (9.2 mW/step vs 10.9 for ANE matmul). | HIGH |
| V27 | 100-experiment autosearch finds config with best-of-88 val_loss 3.288 | E44: val_loss 3.952→3.288 (best seed). **CAVEAT**: Independent verification shows typical val_loss ~3.8 for this config. Run-to-run variance (~0.3 nats) exceeds signal. The baseline config (LR=4e-4, ACCUM=10) reliably gives ~3.5. The "17% improvement" reflects seed selection, not genuine hyperparameter improvement. | QUALIFIED |

## Category: UNVERIFIED (stated but not tested)

| ID | Assumption | Source | Risk |
|----|-----------|--------|------|
| U1 | ~~ANE power draw is ~2.8W at peak~~ | **CORRECTED (2026-03-12)**: Actual ANE peak is ~1.2W (from ane_full mode), not 2.8W. Average 765 mW, peak 1200 mW. The 2.8W estimate was extrapolated, not from Orion. Moved to CORRECTED status. | CORRECTED |
| U2 | Deep graph compilation achieves 94% ANE utilization | Orion + maderix (not our test) | MEDIUM |
| U3 | INT8 provides no compute speedup over fp16 | maderix blog + Orion paper: ANE dequantizes INT8→FP16 before compute. INT8 saves only memory bandwidth (1.88x throughput from smaller weight loads), not compute cycles. True peak is 19 TFLOPS FP16 on M4 (lower on earlier chips: ~15.8 TOPS INT8 on M2). | CONFIRMED (literature) |
| U4 | SRAM is ~32MB with 30% cliff above | maderix Part 2: 2048x2048 (24MB) gets 5.7 TFLOPS, 4096x4096 (96MB) drops to 4.0 TFLOPS (30% drop). **E38 corroboration**: our W1 surface at DIM=2048 is 24MB — right at the cliff edge. | CONFIRMED (literature + E38) |
| U5 | _ANEClient API enables delta compilation | Orion paper (not our test) | HIGH |
| U6 | ~~Gradient sanitization will fix 10-min divergence~~ | Orion paper analogy (hypothesis). **PREMISE INVALID (E37)**: the 10-min divergence was caused by concurrent processes, not ANE. With clean system, neither mode diverges. Gradient sanitization is moot. | INVALID |
| U7 | Kernel fusion (16-64 ops) will improve our throughput | **DISPROVED for training (E36)**: fusing non-linear ops into ANE fp16 causes overfitting. May still hold for inference-only. | DISPROVED (training) |
| U8 | ANE thermal throttling causes our observed step-time stalls | **DISPROVED (E37)**: ANE max step time 165ms, CPU max 16,273ms. ANE had zero stalls. CPU scheduling jitter was the cause all along. | DISPROVED |
| U9 | Single-op ANE kernels get only ~30% utilization (vs 74-94% for deep graphs) | maderix Part 2 benchmarks. Our unfused approach uses 28 single-matmul dispatches/step. 4-layer fusion achieves 7.7x speedup but hurts generalization (V12). | HIGH |
| U10 | ANE dispatch overhead is ~0.095ms per call | maderix Part 2. 28 dispatches/step = ~2.7ms fixed cost. | MEDIUM |
| U11 | M2 Pro only supports ch=512 for conv 1x1 operations | maderix Issue #3: M1/M2/M3 Pro only compile ch=512. M4+ supports flexible channels. Does not affect our matmul-based approach. | MEDIUM |
| U12 | ~~_ANEChainingRequest could eliminate CPU round-trips between layers~~ | M5 benchmark report: supports loopback, firmware-level enqueue, shared memory pools. **INVALIDATED (2026-03-12)**: maderix/ANE PR #40 definitively shows chaining requires Espresso IR from disk-compiled models. Our in-memory MIL path cannot produce the required format. Dead on macOS 15+. See UP5 update. | DEAD |
| U13 | No one has trained models larger than DIM=1024 before our E38 | maderix tested Stories110M (DIM=768) and Qwen3-0.6B (DIM=1024). Our DIM=1536 and DIM=2048 experiments are novel. | CONFIRMED (literature) |
| U14 | LR=3e-4 is equally good for all architectures in E39 grid | **RESOLVED (E40)**: LR=5e-4 optimal for 512d/4L, 3e-4 for 1024d/2L. Ranking unchanged. SA-E39-1 resolved. | RESOLVED |
| U15 | 120s budget is representative of quick-iteration training regime | **RESOLVED (E41)**: 512d/4L wins at 120s, 300s, AND 600s. Gap actually widens at longer budgets. No crossover observed. | CONFIRMED |
| U16 | ~~Warmup=100 steps is appropriate for all configs at all LRs~~ | **RESOLVED (E43)**: Warmup 100→50 produced val 3.540 vs baseline 3.533 — within noise. 100 steps is fine. | RESOLVED |
| U17 | ~~Weight decay 0.1 is equally good across all architectures~~ | **RESOLVED (E43)**: WD 0.1→0.05 produced val 3.533 — identical to baseline. WD is not a sensitive parameter in this regime. | RESOLVED |

## Category: NEW FROM UPSTREAM (2026-03-12, not yet tested by us)

| ID | Assumption | Source | Implications | Risk |
|----|-----------|--------|-------------|------|
| UP1 | Runtime weight injection via IOSurface inputs eliminates recompilation | [imperatormk/ane-train](https://github.com/imperatormk/ane-train), maderix/ANE Issue #47. Demonstrated on ConvNeXt UNet (96→384ch, 256×256) at ~3 it/s on M1. Weights passed as runtime IOSurface inputs to matmul, not baked as const(). | **Invalidates the premise of SA4/D5**: delta compilation is not needed if you compile with runtime inputs from the start. Our dynamic pipeline (mil_dynamic.h) already does this — the issue was never that runtime weights don't work, but that IOSurface overhead + fp16 precision made it uncompetitive vs CPU. | HIGH |
| UP2 | IOSurface slot sizes must be strictly ascending (inputs) / descending (outputs) — violations produce silent zeros | imperatormk/ane-train ANE_TRAINING.md. No error, no warning. | **AUDITED (2026-03-12)**: Our single-surface spatial packing architecture provides structural immunity — all 14 kernels use exactly 1 input and 1 output IOSurface. The Kern struct (`config.h:63`) has scalar ioIn/ioOut fields, making multi-slot impossible. Only `bridge/ane_bridge.m` supports multi-slot and performs no ordering check. See P6 audit report. | VERIFIED SAFE (training pipeline) |
| UP3 | Matmul inner dim (Ci) must be a multiple of 32 — non-multiples silently produce zeros | imperatormk/ane-train. Tested: Ci=16,48,80,112 give eval=0. | Our DIM=512 (multiple of 32) is fine. But configs with non-standard dimensions would silently fail. | MEDIUM |
| UP4 | Mega-kernel fusion (N layers in single MIL program) achieves 3-4x forward speedup | maderix/ANE PR #24 (filipexyz). Full transformer fusion: stories15M 4.17x, stories110M 3.00x. XPC overhead ~160μs/eval is the bottleneck. | Could improve our ANE mode significantly, but weights must still be const() in fused kernels (conflicts with UP1). Trade-off: runtime weights (no recompile) vs fused kernels (3x faster but recompile on weight update). | HIGH |
| UP5 | ~~`_ANEChainingRequest` actually works — error was wrong factory method~~ | [thebasedcapital/ane-infer](https://github.com/thebasedcapital/ane-infer), maderix/ANE Issue #44. | **INVALIDATED (2026-03-12)**: maderix/ANE PR #40 definitively shows `_ANEChainingRequest` is **dead on macOS 15+**. It requires Espresso IR from disk-compiled `_ANEModel`. Our in-memory MIL path (`_ANEInMemoryModelDescriptor`) cannot produce the required format. Do not pursue for training. | DEAD |
| UP6 | M3 Ultra ANE only supports ch=512 (exactly, not minimum). Peak 8.77 TFLOPS at 128x conv | maderix/ANE Issue #42 (pudepiedj). All other channel configs fail with -4. Only one ANE die active on UltraFusion. | Extends U11 (M1/M2/M3 Pro 512ch constraint) to M3 Ultra. Our matmul-based approach avoids this constraint. | LOW |
| UP7 | Delta compilation via unload/reload achieves 8.5x speedup (4200ms→494ms/step) | Orion paper (arxiv:2603.06728). `unloadWithQoS(21)` → patch weight files on disk → `loadWithQoS(21)`. Bypasses `ANECCompile()` entirely. | Different from our runtime weight injection approach. For our 36.4M model at 24ms/step, the 494ms reload would be prohibitive. May be useful for larger models (DIM≥1024) where compute amortizes overhead. Also eliminates the ~119 compilation limit per process. | MEDIUM |
| UP8 | IOSurface multi-input constraint: uniform alloc size (not ascending order) | Orion paper constraint #18: "All IOSurface inputs must have the same byte allocation size" (0x1d error if violated). **Contradicts UP2** which says ascending order. | May depend on MIL target version (ios16 vs ios18) or chip generation. Both may be partially correct for different contexts. Our single-input architecture is immune. | MEDIUM |
| UP9 | IOSurface parameters bound alphabetically by MIL parameter name | Orion paper constraint #19. Input IOSurfaces bound to MIL parameters in alphabetical order by name, not declaration order. Same for outputs. Violation produces silent wrong data. | Critical for any multi-input MIL programs (P1 Phase 2+). Must name weight parameter alphabetically before activation parameter. | HIGH |
| UP10 | ~119 compilations per process limit — subsequent compilations silently fail | Orion paper + maderix characterization. Built into ANE hardware state management. **CONTRADICTED by maderix/ANE Issue #24 (macOS 26.2)**: 312 compiles completed without restart. Limit may be OS-version-specific or only on certain hardware. | Not relevant for our single-compilation-at-startup approach. May not be a hard limit on macOS 26+. | LOW (possibly outdated) |
| UP11 | `concat` MIL op causes compile failure on some targets | Orion paper constraint. | **CONTRADICTED BY OUR TESTING (2026-03-12)**: Our ios18 target compiles and executes 10 concat ops successfully (sdpaFwd, ffnFused, sdpaBwd1, sdpaBwd2). Likely target-version-specific — Orion may use ios16. | DISPROVED (ios18) |
| UP12 | 1×1 conv delivers 3× throughput vs equivalent matmul on ANE | Orion paper + maderix Part 2 benchmarks. Our V2 measured 1.5-2.8×. | Consistent with V2 range. We use matmul-based approach for simplicity and MIL compatibility. Conv-based approach would need channel constraints (UP6, U11). | CONFIRMED |
| UP13 | M5 requires 128-byte IOSurface alignment | maderix/ANE PR #35. Without this, M5 ANE silently rejects surfaces or produces wrong results. | Must implement before targeting M5 hardware. Our current IOSurface creation has no alignment enforcement. | MEDIUM |
| UP14 | Backpropagation-free training (MeZO/ZO methods) can fine-tune LLMs with forward passes only | MeZO (Princeton NLP, 2023), MobiZO (EMNLP 2025), ElasticZO (Jan 2025). MeZO: 2-25× memory reduction. ElasticZO-INT8: integer-only arithmetic. | Could eliminate backward kernels entirely for fine-tuning. Particularly interesting with ANE's 1.88× INT8 throughput advantage. Not yet tested for from-scratch training. | HIGH (research) |
| UP15 | M5 GPU Neural Accelerators achieve ~70 TFLOPS FP16 (3.5× ANE's 19 TFLOPS) | Apple ML Research blog, macOS 26.2 MLX support. Dedicated matmul units in GPU pipeline. | Apple is investing in GPU ML acceleration, not ANE training. Makes ANE training a more niche but unique contribution. ANE advantage: zero idle power, dedicated silicon leaves GPU/CPU free. | INFORMATIONAL |
| UP16 | Core AI framework replacing CoreML at WWDC 2026 (June) | AppleInsider, 9to5Mac reports (March 2026). Focus on third-party AI model integration. | Risk: if Core AI exposes official ANE training APIs, it could supersede our private-API approach. Opportunity: if inference-only, AutoANE remains the only training path. Monitor closely. | HIGH (strategic) |
| UP17 | Function parameter IOSurfaces are 30% faster than spatial packing | maderix/ANE PR #22 (fspecii). Weights as native MIL function parameters backed by persistent IOSurfaces. Eliminates 12 slice/reshape/transpose ops per attention kernel. 110ms→76.9ms/step. | Our architecture uses spatial packing. Function params could eliminate `slice_by_size+reshape+transpose` overhead. Would require MIL gen changes but no architectural changes. | HIGH |
| UP18 | M3 Ultra ANE hard-limits to exactly 512 channels | maderix/ANE Issue #42. Only accepts exactly 512 channels. Peak sustained 8.77 TFLOPS (may be single die of UltraFusion). | Hardware-specific constraint. Our DIM values (512, 768, 1024) should work. Larger dims may need chunking. | MEDIUM |
| UP19 | ACCUM_STEPS=100 gives 4.74x throughput improvement | maderix/ANE Issue #24. At stories15M: 0.66 steps/s (accum=1) → 3.15 steps/s (accum=100). Amortizes compile overhead across more forward/backward passes. **EXPERIMENTALLY VERIFIED (2026-03-12)**: Our testing shows CPU 3.0x at accum=50, ANE 2.2x at accum=100 — significant but lower than upstream's 4.74x. Difference likely due to model size (36.4M vs 15M), architecture, or measurement methodology. Direction confirmed, magnitude overstated. | CONFIRMED (lower magnitude) |
| UP20 | FwdLLM: BP-free federated LLM fine-tuning with 1.5GB peak memory | USENIX ATC '24. LLaMA-7B fine-tuning via perturbed inferences. 14.6x memory reduction. | Alternative to backprop for ANE. Only forward passes needed. Could leverage ANE's forward speed advantage. | HIGH (research) |
| UP21 | MobiEdit: Quantized forward-only gradient estimation for NPUs | ICLR 2026. 7.1x memory, 3.4x latency, 15.8x energy reduction vs standard fine-tuning. W8A16 with 80% edit success. | Complementary to UP14 (MeZO). More practical for NPU hardware like ANE. | HIGH (research) |
| UP22 | Mega-kernel layer fusion gives 3-4x forward speedup | maderix/ANE Issue #24. Fusing 6 transformer layers into single ANE eval: 4.17x forward speedup at stories15M. | Aligns with P2 priority. Requires const() weights. Trade-off: recompilation per weight update. | HIGH |
| UP23 | karpathy/llm.c PR #840: Apple Accelerate BLAS = 2.6x CPU e2e speedup | Replaces hand-rolled matmul with `cblas_sgemm`. Links Apple Accelerate framework. Karpathy Discussion #253 greenlit BLAS/SIMD. | We already use cblas_sgemm extensively. Validates our vectorization approach. | CONFIRMED |

## Category: DISPROVED (tested and found wrong)

| ID | Assumption | Evidence |
|----|-----------|----------|
| D1 | IOSurface is the primary bottleneck | Exp 7: only 13%, not 40%+ |
| D2 | CPU-only training would be faster than ANE per-step | Exp 11: ANE 1.5x faster |
| D3 | ANE+CPU-attn-bwd would be best of both worlds | Exp 11: worst of all 3 modes |
| D4 | Higher LR helps ANE training | Exp 5: larger activations degrade fp16 |
| D5 | Delta compilation works via _ANEInMemoryModel reload | Exp 10, 17: output unchanged |
| D6 | ANE→CPU mid-training would improve over pure modes | E29: adaptive 4.507 vs pure ANE 4.354 vs pure CPU 3.897. fp16 damage is cumulative in weights. |
| D7 | ANE becomes advantageous at larger model dimensions | E38: tested DIM=1024/1536/2048. ANE never faster. At DIM=2048, ANE 2x SLOWER due to IOSurface memory pressure (379MB wired memory causes cache thrashing). |
| D8 | ANE is ~2x more energy efficient than CPU for training | Powermetrics measurement (2026-03-12): CPU-only 13273 mW, ANE matmul 12568 mW, ANE full 12664 mW. Package power nearly identical. CPU-only is actually most energy-efficient per step. Orion does not quantify training power efficiency; this assumption was extrapolated from general ANE claims. |

---

## Rules for This Document

1. Every new assumption gets an ID and category
2. When an experiment changes an assumption's status, update here with evidence
3. RETESTING items must be resolved before building features that depend on them
4. UNVERIFIED items must be noted in any code/doc that relies on them
