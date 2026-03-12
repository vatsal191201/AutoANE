# AutoANE: Stated Assumptions Registry

**Purpose**: Every assumption in this project must be explicitly stated, tracked, and verified. No implicit assumptions allowed.

**Last updated**: 2026-03-12

---

## Category: RETESTING (previously classified, needs re-verification)

| ID | Assumption | Prior Status | Why Retesting |
|----|-----------|-------------|---------------|
| SA1 | fp16 precision gap (~16%) is irreducible via software | **RE-CONFIRMED (E19)** | Zero NaN/Inf in 5777 steps. Gap is genuine fp16 accumulation rounding, not fixable by sanitization. |
| SA2 | ANE training diverges after ~2000 steps | **DISPROVED (E19-ext)** | 5777 steps, loss still improving (avg ~4.59). Exp 18 "divergence" was thermal throttling from running concurrent tests on same machine. |
| SA3 | Activation clamping to [-4,4] doesn't improve loss | VERIFIED (Exp 14) | Still needs testing: activations reach [-121, 127] at 10min — approaching fp16 max (65504). Overflow clamping [-65504, +65504] untested. |
| SA4 | Delta compilation not viable | **RE-CONFIRMED (E34)** | _ANEClient.loadModelNewInstance fails; _ANEClient and _ANEInMemoryModel are separate compilation paths. Full pipeline rewrite needed. |

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
| V27 | 100-experiment autosearch improves val_loss by 17% from baseline | E44: val_loss 3.952→3.288. Best config: LR=6.34e-4, ACCUM=7, WD=0.076, ADAM_B2=0.959. 6 keeps from 88 completed experiments. | HIGH |

## Category: UNVERIFIED (stated but not tested)

| ID | Assumption | Source | Risk |
|----|-----------|--------|------|
| U1 | ~~ANE power draw is ~2.8W at peak~~ | **CORRECTED (2026-03-12)**: Actual ANE peak is ~1.2W (from ane_full mode), not 2.8W. Average 765 mW, peak 1200 mW. The Orion paper's estimate was for a different workload. Moved to CORRECTED status. | CORRECTED |
| U2 | Deep graph compilation achieves 94% ANE utilization | Orion + maderix (not our test) | MEDIUM |
| U3 | INT8 provides no compute speedup over fp16 | maderix blog + Orion paper: ANE dequantizes INT8→FP16 before compute. INT8 saves only memory bandwidth (1.88x throughput from smaller weight loads), not compute cycles. True peak is 19 TFLOPS FP16 regardless. | CONFIRMED (literature) |
| U4 | SRAM is ~32MB with 30% cliff above | maderix Part 2: 2048x2048 (24MB) gets 5.7 TFLOPS, 4096x4096 (96MB) drops to 4.0 TFLOPS (30% drop). **E38 corroboration**: our W1 surface at DIM=2048 is 24MB — right at the cliff edge. | CONFIRMED (literature + E38) |
| U5 | _ANEClient API enables delta compilation | Orion paper (not our test) | HIGH |
| U6 | ~~Gradient sanitization will fix 10-min divergence~~ | Orion paper analogy (hypothesis). **PREMISE INVALID (E37)**: the 10-min divergence was caused by concurrent processes, not ANE. With clean system, neither mode diverges. Gradient sanitization is moot. | INVALID |
| U7 | Kernel fusion (16-64 ops) will improve our throughput | **DISPROVED for training (E36)**: fusing non-linear ops into ANE fp16 causes overfitting. May still hold for inference-only. | DISPROVED (training) |
| U8 | ANE thermal throttling causes our observed step-time stalls | **DISPROVED (E37)**: ANE max step time 165ms, CPU max 16,273ms. ANE had zero stalls. CPU scheduling jitter was the cause all along. | DISPROVED |
| U9 | Single-op ANE kernels get only ~30% utilization (vs 74-94% for deep graphs) | maderix Part 2 benchmarks. Our unfused approach uses 28 single-matmul dispatches/step. 4-layer fusion achieves 7.7x speedup but hurts generalization (V12). | HIGH |
| U10 | ANE dispatch overhead is ~0.095ms per call | maderix Part 2. 28 dispatches/step = ~2.7ms fixed cost. | MEDIUM |
| U11 | M2 Pro only supports ch=512 for conv 1x1 operations | maderix Issue #3: M1/M2/M3 Pro only compile ch=512. M4+ supports flexible channels. Does not affect our matmul-based approach. | MEDIUM |
| U12 | _ANEChainingRequest could eliminate CPU round-trips between layers | M5 benchmark report: supports loopback, firmware-level enqueue, shared memory pools. Untested for training. | HIGH |
| U13 | No one has trained models larger than DIM=1024 before our E38 | maderix tested Stories110M (DIM=768) and Qwen3-0.6B (DIM=1024). Our DIM=1536 and DIM=2048 experiments are novel. | CONFIRMED (literature) |
| U14 | LR=3e-4 is equally good for all architectures in E39 grid | **RESOLVED (E40)**: LR=5e-4 optimal for 512d/4L, 3e-4 for 1024d/2L. Ranking unchanged. SA-E39-1 resolved. | RESOLVED |
| U15 | 120s budget is representative of quick-iteration training regime | **RESOLVED (E41)**: 512d/4L wins at 120s, 300s, AND 600s. Gap actually widens at longer budgets. No crossover observed. | CONFIRMED |
| U16 | ~~Warmup=100 steps is appropriate for all configs at all LRs~~ | **RESOLVED (E43)**: Warmup 100→50 produced val 3.540 vs baseline 3.533 — within noise. 100 steps is fine. | RESOLVED |
| U17 | ~~Weight decay 0.1 is equally good across all architectures~~ | **RESOLVED (E43)**: WD 0.1→0.05 produced val 3.533 — identical to baseline. WD is not a sensitive parameter in this regime. | RESOLVED |

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
| D8 | ANE is ~2x more energy efficient than CPU for training | Powermetrics measurement (2026-03-12): CPU-only 13273 mW, ANE matmul 12568 mW, ANE full 12664 mW. Package power nearly identical. CPU-only is actually most energy-efficient per step. Orion paper's estimate was for inference, not training. |

---

## Rules for This Document

1. Every new assumption gets an ID and category
2. When an experiment changes an assumption's status, update here with evidence
3. RETESTING items must be resolved before building features that depend on them
4. UNVERIFIED items must be noted in any code/doc that relies on them
