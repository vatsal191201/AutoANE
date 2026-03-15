# ANE Training Pipeline Optimization — Design Spec

## Goal

Increase effective training throughput of MeZO+LoRA-split on SmolLM2-360M by 8-15x through 4 additive optimizations: conv1x1 hybrid kernels, intra-layer fusion, multi-perturbation gradient estimation (FZOO), and gradient-aligned perturbations (P-GAP).

## Measured Baselines

| Metric | Value | Source |
|--------|-------|--------|
| CPU-only step | 452.5 ms | Measured (LoRA-split, lr=1e-4, seed 42) |
| CPU forward pass | 222 ms | Derived (step/2 - perturbation) |
| CPU matmul portion | 90 ms (40%) | NumPy/Accelerate benchmark |
| CPU ops (RMS/RoPE/SDPA/SiLU/LoRA) | 132 ms (60%) | Derived |
| ANE-matmul step | 528.0 ms | Measured (1.17x SLOWER than CPU) |
| ANE matmul compute | 68 ms (26% of ANE fwd) | bench_conv 500-iter avg |
| IO staging overhead | 60 ms (23% of ANE fwd) | Derived: 224 trips x 0.27ms |
| Conv1x1 ANE compute | 36.5 ms (hybrid) | bench_conv 500-iter avg |
| Baseline val_loss | 2.0718 | Independently verified |
| 500-step improvement (lr=1e-4) | 0.0133 (0.64%) | Multi-seed sweep |

## Key Finding

ANE mode is currently **1.17x slower** than CPU mode because IOSurface staging overhead (60ms) + CPU ops (132ms) dominate the forward pass. Pure ANE matmul (68ms) is faster than CPU matmul (90ms), but the overhead negates the advantage. Conv1x1 alone does NOT make ANE faster than CPU. **Fusion is required.**

## Phase 1: Conv1x1 Hybrid with BLOBFILE Constants

**What**: Replace matmul MIL kernels with conv1x1 for projections where conv is faster. Bake base weights as BLOBFILE constants (no per-step staging).

**Projection-level speedups** (bench_conv, 500 iterations):

| Projection | Shape | Matmul ms | Conv ms | Speedup | Use Conv? |
|-----------|-------|-----------|---------|---------|-----------|
| Wq | 960→960 | 0.265 | 0.129 | 2.05x | YES |
| Wk | 960→320 | 0.123 | 0.346 | 0.36x | NO |
| Wv | 960→320 | 0.123 | 0.346 | 0.36x | NO |
| Wo | 960→960 | 0.269 | 0.127 | 2.11x | YES |
| W1 | 960→2560 | 0.449 | 0.204 | 2.20x | YES |
| W2 | 2560→960 | 0.447 | 0.232 | 1.93x | YES |
| W3 | 960→2560 | 0.449 | 0.204 | 2.20x | YES |

**Numerical correctness**: Bit-exact (max_abs_diff=0.000000 for all projections, test_conv_num).

**IO surface reduction**: Conv input is activation-only [1,IC,1,SEQ] vs matmul [1,IC,1,SEQ+OC]. Wq surface shrinks 4.8x (2.34MB→0.49MB).

**Changes**:
- `mil_dynamic.h`: Port `gen_conv1x1_mil()` from bench_conv_vs_matmul.m
- `io.h`: Add `compile_conv1x1_kern()` with BLOBFILE weight baking
- `train_mezo.m`: `--conv-hybrid` flag, conv kernel compilation, simplified forward (no RETRANSPOSE)
- `config.h`: Conv kernel pointers in DynLayerKernels

**Constraint**: Conv1x1 BLOBFILE only works with LoRA-split (base weights frozen). This is a hard requirement.

**Go/no-go gate**: Must achieve ≤452ms/step (CPU parity).

## Phase 2: Fused Conv Kernels

**What**: Combine multiple conv operations per layer into fewer MIL programs.

**LoRA constraint**: LoRA corrections happen between base projection and RoPE/SDPA. Full attention fusion (Q/K/V+RoPE+SDPA) is NOT possible — corrections must be applied per-projection.

**Fusion targets**:
1. **QKV combined kernel**: 3 conv1x1 ops (Wq, Wk, Wv baked as BLOBFILE), shared input, separate outputs concatenated. 1 kernel instead of 3.
2. **FFN mega-kernel**: conv(W1) + conv(W3) + SiLU + conv(W2) + residual. All weights baked. 1 kernel instead of 3. No LoRA needed (attention-only LoRA mode).

**Result**: 3 kernels/layer (QKV + Wo + FFN) instead of 7 → 96 IO round-trips instead of 224.

**Projected**: ~325ms/step (1.39x over CPU).

**Go/no-go gate**: Must achieve ≤350ms/step (1.3x over CPU).

## Phase 3: FZOO-Style Multi-Perturbation

**What**: Replace single central-difference with K one-sided gradient estimates averaged per step. Add FZOO's adaptive step size (sigma-normalized).

**Theory** (verified via literature):
- One-sided Rademacher has O(ε²) bias (same as central-difference)
- "The second-order Hessian cross-term E[(u^T H u) * u] = 0 because (u^T H u) is even in u while the outer u is odd" (ZOSA, arXiv:2511.09156)
- This is specific to Rademacher (NOT Gaussian)

**Algorithm**:
```
For each step:
  1. Sample data batch
  2. Forward pass (unperturbed) → loss_0                    [1 pass]
  3. For k in 1..K:
     a. Generate Rademacher z_k
     b. Perturb +ε*z_k → forward → loss_k → restore        [K passes]
  4. grad = (1/K) * sum_k [(loss_k - loss_0) / ε] * z_k
  5. sigma = std(loss_1, ..., loss_K)
  6. Update: θ -= lr * (grad / sigma)                       [normalized-SGD]
```

**Forward passes**: K+1 per step (vs 2 for current central-difference).

| K | Passes/step | Gradient estimates | vs Central (2 passes, 1 estimate) |
|---|-------------|-------------------|-----------------------------------|
| 1 | 2 | 1 | Same cost, biased |
| 2 | 3 | 2 | 1.5x cost, 2x estimates |
| 4 | 5 | 4 | 2.5x cost, 4x estimates |

**Expected benefit**: Better gradient quality → fewer total steps. FZOO reports 3x fewer total forward passes on average across tasks. MobiZO reports q=4 gives ~4% accuracy improvement.

**FZOO adaptive step size**: Division by sigma_t is equivalent to normalized-SGD (Proposition 3.2 in FZOO paper). Larger steps in flat regions, smaller in steep regions.

**Go/no-go gate**: K=4 must show better convergence than central-difference K=1 at 500 steps with same total compute budget.

## Phase 4: P-GAP (Gradient-Aligned Perturbations)

**What**: Project perturbations into a gradient-aligned subspace via SVD.

**Algorithm** (from arXiv:2510.18228):
1. Every k steps (lazy update, k=16 or k=512):
   a. Estimate gradient direction using h probe perturbations
   b. SVD: G_hat ≈ U_r * S_r * V_r^T (retain r leading directions)
2. Between updates:
   a. Generate random Z_init in r-dimensional space
   b. Project onto alignment hyperplane
   c. Map back: Z_f = U_r * Z * V_r^T
3. Use Z_f as perturbation direction

**Memory**: ~0.7-1.9 GB overhead for basis matrices (OPT-2.7B scale). For our 1.7M trainable params: negligible.

**Reported speedups**: 5.25x on SST-2, 3.33x on SNLI, 5.88x on DROP (iterations to same loss).

**Complexity**: Higher than other phases — requires SVD computation, subspace management. Consider deferring if Phases 1-3 deliver sufficient improvement.

**Go/no-go gate**: Must show measurable convergence improvement at 500 steps.

## Combined Projection

| Phase | Speed Factor | Convergence Factor | Evidence |
|-------|-------------|-------------------|----------|
| 1: Conv1x1 | ~1.0-1.2x | 1.0x | bench_conv measured |
| 2: Fusion | ~1.2-1.4x cumulative | 1.0x | IO analysis derived |
| 3: FZOO K=4 | ~0.4x (2.5x more passes) | ~3x fewer steps | FZOO paper |
| 4: P-GAP | 1.0x | ~3-5x fewer steps | P-GAP paper |
| **Combined** | ~0.5-0.6x speed per step | ~9-15x convergence | **~5-8x effective** |

Note: Phase 3 trades speed-per-step for convergence. Combined with Phase 1+2 speed improvements, net throughput improves.

## Assumptions

| # | Assumption | Basis | Risk |
|---|-----------|-------|------|
| A1 | Conv BLOBFILE reduces IO overhead via smaller surfaces | Smaller surfaces = less write time | Medium |
| A2 | FFN fusion kernel compiles within ANE limits | Existing gen_ffn_fused works | Low |
| A3 | QKV combined conv kernel compiles | 3 conv ops + concat | Medium |
| A4 | One-sided Rademacher has O(ε²) bias | ZOSA paper, verified | Low |
| A5 | FZOO's sigma-normalized update helps convergence | FZOO paper, normalized-SGD theory | Low |
| A6 | P-GAP transfers to LoRA fine-tuning | Tested on full-param; LoRA subspace may differ | High |
| A7 | Conv1x1 requires LoRA-split (frozen base) | BLOBFILE constants can't change | None (hard constraint) |

## Literature References

| Paper | Venue | Key Contribution | DOI/ArXiv |
|-------|-------|-----------------|-----------|
| MeZO | NeurIPS 2023 | SPSA for LLM fine-tuning | arXiv:2305.17333 |
| MobiZO | EMNLP 2025 | MP-LoRA, batch perturbation | 10.18653/v1/2025.emnlp-main.1022 |
| FZOO | 2025 | One-sided + adaptive step | arXiv:2506.09034 |
| P-GAP | 2025 | Gradient-aligned subspace | arXiv:2510.18228 |
| ZOSA | 2025 | O(ε²) bias proof for Rademacher | arXiv:2511.09156 |
| Orion | 2026 | ANE conv1x1 + constraints | arXiv:2603.06728 |
