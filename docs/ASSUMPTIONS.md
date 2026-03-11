# AutoANE: Stated Assumptions Registry

**Purpose**: Every assumption in this project must be explicitly stated, tracked, and verified. No implicit assumptions allowed.

**Last updated**: 2026-03-11

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
| V4 | IOSurface overhead is ~13% of step time | Exp 7: 8.1ms of 63.2ms | HIGH |
| V5 | Mixed precision (ANE fwd + CPU bwd) worse than pure | Exp 11: 5.10 vs 4.90 avg loss | HIGH |
| V6 | Loss scaling (256.0) essential for ANE fp16 | Exp 1: gradients underflow without it | HIGH |
| V7 | Shallow/wide beats deep/narrow in fixed time | Exp 2: 4L/1024d beats 32L/960d | HIGH |
| V8 | LoRA from pretrained works on ANE | Exp 3: stable training, loss 4.22 | HIGH |
| V9 | Classifier row-major optimization ~2x faster | Exp 13: 30.5ms → 15.7ms | HIGH |
| V10 | Both modes suffer ~1.5x thermal throttling at 10min | CORRECTED — single-process throttling is 30% (1.3x), not 50% (1.5x). E18's 1.5x was from concurrent processes. | CORRECTED (E24) |
| V11 | ANE only faster than CPU for large matmul shapes (2816+ width) | E23: 1024x1024 CPU 0.67x faster, 2816x1024 ANE 1.88x faster | HIGH |

## Category: UNVERIFIED (stated but not tested)

| ID | Assumption | Source | Risk |
|----|-----------|--------|------|
| U1 | ANE power draw is ~2.8W at peak | maderix blog (not our measurement) | MEDIUM |
| U2 | Deep graph compilation achieves 94% ANE utilization | Orion + maderix (not our test) | MEDIUM |
| U3 | INT8 provides no compute speedup over fp16 | maderix blog (not our test) | LOW |
| U4 | SRAM is ~32MB with 30% cliff above | maderix blog (not our test) | LOW |
| U5 | _ANEClient API enables delta compilation | Orion paper (not our test) | HIGH |
| U6 | Gradient sanitization will fix 10-min divergence | Orion paper analogy (hypothesis) | HIGH |
| U7 | Kernel fusion (16-64 ops) will improve our throughput | Orion + maderix (not our test) | HIGH |

## Category: DISPROVED (tested and found wrong)

| ID | Assumption | Evidence |
|----|-----------|----------|
| D1 | IOSurface is the primary bottleneck | Exp 7: only 13%, not 40%+ |
| D2 | CPU-only training would be faster than ANE per-step | Exp 11: ANE 1.5x faster |
| D3 | ANE+CPU-attn-bwd would be best of both worlds | Exp 11: worst of all 3 modes |
| D4 | Higher LR helps ANE training | Exp 5: larger activations degrade fp16 |
| D5 | Delta compilation works via _ANEInMemoryModel reload | Exp 10, 17: output unchanged |
| D6 | ANE→CPU mid-training would improve over pure modes | E29: adaptive 4.507 vs pure ANE 4.354 vs pure CPU 3.897. fp16 damage is cumulative in weights. |

---

## Rules for This Document

1. Every new assumption gets an ID and category
2. When an experiment changes an assumption's status, update here with evidence
3. RETESTING items must be resolved before building features that depend on them
4. UNVERIFIED items must be noted in any code/doc that relies on them
