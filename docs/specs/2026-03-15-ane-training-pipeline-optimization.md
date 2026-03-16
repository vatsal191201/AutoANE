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

**What**: Project perturbations into a gradient-aligned subspace via per-matrix SVD.

**Algorithm** (from arXiv:2510.18228, corrected after deep literature review):

**Probe phase** (every k=100 steps, costs 2h forward passes):
1. For j=1..h: sample Gaussian Q_j for all params, SPSA → scalar ρ_j
2. For each LoRA matrix W_l: accumulate G_l += (ρ_j/h) × Q_l^j
3. Per-matrix SVD: G_l = U_r × S_r × V_r^T (truncated to rank r)

**Training step** (between probes):
1. For each LoRA matrix W_l:
   a. Sample Z_init ~ N(0, I_{r×r})
   b. PROJECTION: Z = Z_init - α×S_r where α = (⟨S_r,Z_init⟩_F - ξ√δ‖S_r‖_F)/‖S_r‖²_F
   c. Map back: Z_f = U_r × Z × V_r^T ∈ R^{m×n}
2. For RMS norms (1D): z ~ N(0, I)
3. Concatenate into full perturbation z, SPSA with ε → scalar G_t
4. Update: W_l -= η × G_t × Z_f^l

**Paper's hyperparameters for LoRA**: ε=0.1, lr={1e-2..5e-2}, r=8, k=100, h=10, δ: 2→0

**Memory**: ~22 MB (per-matrix SVD bases + perturbation buffers). Negligible.

**Reported speedups**: P-GAP+LoRA on OPT-2.7B: 76.6% SQuAD (vs 63.4% MeZO LoRA), 8.7% of training GPU hours on RTE.

**Go/no-go gate**: Must show measurable convergence improvement at 500 steps.

## Measured Results

| Phase | Speed (ms/step) | vs CPU | Convergence | Status |
|-------|----------------|--------|-------------|--------|
| Baseline (CPU) | 447 | 1.00x | reference | - |
| 1: Conv1x1 hybrid | 403-429 | 1.04-1.11x | 1.0x | ✅ Done |
| 2: Fused conv kernels | 262 | 1.71x | 1.0x | ✅ Done |
| 3: FZOO K=4 | 2.5x slower/step | no wall-time benefit | ✅ Done |
| 4: Flat-vector subspace (NOT P-GAP) | same speed | degrades | ❌ Negative |
| 4: Faithful P-GAP | — | — | 🔄 In progress |

**Best configuration: Phase 2 (conv-fused) at ~262ms/step = 1.71x faster than CPU.**
Triple-checked: 50-step average. val_loss within 0.03% of CPU baseline.

Phase 3 (FZOO) provides better gradient quality but costs 2.5x more forward passes, netting zero wall-time convergence improvement. Previous Phase 4 test used a simplified flat-vector approach that differs fundamentally from P-GAP (per-matrix SVD, gradient alignment constraint, 100x larger ε). Faithful P-GAP implementation in progress.

## Assumptions — Final Status

| # | Assumption | Risk | Outcome |
|---|-----------|------|---------|
| A1 | Conv BLOBFILE reduces IO overhead | Medium | ✅ Confirmed (4.8x surface reduction) |
| A2 | FFN fusion kernel compiles | Low | ✅ Confirmed |
| A3 | QKV combined conv kernel compiles | Medium | ✅ Confirmed |
| A4 | One-sided Rademacher has O(ε²) bias | Low | ✅ Confirmed (literature) |
| A5 | FZOO sigma-normalized update helps | Low | ⚠️ Helps gradient quality, not wall-time |
| A6 | P-GAP transfers to LoRA fine-tuning | High | ⚠️ Simplified version failed; faithful impl. in progress |
| A7 | Conv1x1 requires LoRA-split | None | ✅ Hard constraint, satisfied |

## Literature References

| Paper | Venue | Key Contribution | DOI/ArXiv |
|-------|-------|-----------------|-----------|
| MeZO | NeurIPS 2023 | SPSA for LLM fine-tuning | arXiv:2305.17333 |
| MobiZO | EMNLP 2025 | MP-LoRA, batch perturbation | 10.18653/v1/2025.emnlp-main.1022 |
| FZOO | 2025 | One-sided + adaptive step | arXiv:2506.09034 |
| P-GAP | 2025 | Gradient-aligned subspace | arXiv:2510.18228 |
| ZOSA | 2025 | O(ε²) bias proof for Rademacher | arXiv:2511.09156 |
| Orion | 2026 | ANE conv1x1 + constraints | arXiv:2603.06728 |
