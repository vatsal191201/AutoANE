# Phase 4: P-GAP Research Log — Validated Negative Result

## Summary

P-GAP (Gradient-Aligned Perturbations) does NOT work for LoRA-based zeroth-order training. The projected perturbation captures ~r/d ≈ 2.4×10⁻⁶ of gradient energy, rendering exploitation steps effectively useless. Convergence improvement is proportional to the fraction of standard (unprojected) MeZO steps, confirming that P-GAP-projected steps contribute zero useful gradient signal.

## Background

P-GAP (arXiv:2510.18228) proposes projecting SPSA perturbations into a gradient-aligned subspace estimated via SVD. The paper reports 3-5x fewer iterations on SST-2, SNLI, DROP for full-parameter ZO optimization.

Our setup: SmolLM2-360M with LoRA rank-8, attention-only adapters. Trainable params d = 1,700,800 (1,638,400 LoRA + 62,400 RMS norms).

## Experiment V1: Gradient Subspace Probe (FLAWED)

**Setup**: h=32 central-difference perturbations on fixed data batch, build G[d×32], SVD via G^T G eigendecomposition.

**Results**: Singular values decay from 5186 to 64. Top 14/32 directions capture 90% variance.

**CRITICAL FLAW**: This SVD spectrum is meaningless for gradient rank estimation. Mathematical analysis proves:

G^T G has entries:
- Diagonal: (G^T G)_{ii} = scalar_i² × d ≈ O(d)
- Off-diagonal: (G^T G)_{ij} = scalar_i × scalar_j × (z_i · z_j) ≈ O(√d)

Since d = 1.7M >> √d ≈ 1304, the diagonal dominates. The eigenvalues of G^T G are approximately sorted(scalar_i² × d), which is just the distribution of scalar gradient projections — NOT the gradient rank.

**Verification**: Predicted singular values (√(scalar_i² × d)) match measured values to 3+ decimal places. Ratio: 1.000 for all 10 checked.

**Lesson**: When d >> h, random Rademacher vectors z_i are nearly orthogonal (||z_i|| = √d, z_i · z_j = O(√d)), so G is approximately rank-h regardless of the true gradient's rank. The G^T G approach CANNOT detect gradient low-rankness.

## Experiment V2: P-GAP Convergence Test

**Implementation**: Standard MeZO with periodic P-GAP subspace estimation.
- Collection steps (step % k < r): standard MeZO + extract z, store gradient g = scalar × z
- After r collections: QR decomposition → orthonormal basis Q ∈ R^{d×r}
- Exploitation steps (step % k ≥ r): z_proj = Q @ z_small (z_small ∈ {±1}^r), SPSA with z_proj

**Results (500 steps, lr=1e-4, seed 42)**:

| Config | val_loss@100 | val_loss@300 | val_loss@500 | Δloss | Effective |
|--------|-------------|-------------|-------------|-------|-----------|
| Baseline (100% standard) | 2.0663 | 2.0599 | 2.0576 | 0.0142 | 1.00x |
| P-GAP r=8, k=16 (50% proj) | 2.0673 | 2.0642 | 2.0623 | 0.0095 | 0.67x |
| P-GAP r=4, k=16 (75% proj) | 2.0699 | 2.0678 | 2.0668 | 0.0050 | 0.35x |

**Key observation**: proj_grad on projected steps is ~1000x smaller than standard:
- Standard MeZO: |proj_grad| ≈ 2.0-3.8
- P-GAP projected: |proj_grad| ≈ 0.001-0.004

This matches the theoretical prediction: |z_proj^T g| / |z^T g| ≈ √(r/d) = √(4/1.7M) ≈ 0.0015.

## Mathematical Analysis: Why P-GAP Fails for LoRA ZO

### Signal-to-Noise Ratio of Gradient Estimates

Each SPSA gradient estimate: g_hat = scalar × z, where scalar = (L⁺ - L⁻)/(2ε).

For element j: g_hat_j = scalar × z_j (where z_j ∈ {±1}).

True gradient element: g_j ≈ ||g|| / √d ≈ 0.00092 (assuming uniform magnitude).

Std of noise per element: ≈ ||g|| ≈ 1.2.

**SNR per element = g_j / std ≈ 0.00077.**

To achieve SNR=1 from averaging h probes: need h ≈ (std/g_j)² ≈ 1.7M probes. **Completely impractical.**

### Why the Basis Q is Random

Each gradient estimate g_hat = scalar × z is a random direction (z dominates, scalar only scales magnitude). The QR decomposition of r such estimates gives r approximately random orthonormal vectors in R^{1.7M}.

A random r-dimensional subspace in R^d captures fraction r/d of any fixed direction's energy:
- r=4: captures 4/1.7M = 2.4×10⁻⁶ of gradient energy
- r=8: captures 8/1.7M = 4.7×10⁻⁶ of gradient energy

### Structural Mismatch with LoRA

P-GAP was designed for full-parameter ZO where weight matrices are large (e.g., W ∈ R^{960×960}) and the gradient matrix may be low-rank (rank 5-20 in a 960-dimensional space).

For LoRA:
- A matrices: R^{8×960} (rank ≤ 8 by construction)
- B matrices: R^{960×8} (rank ≤ 8 by construction)

The gradient of these tiny matrices has at most rank 8. There is minimal room for P-GAP to exploit additional low-rank structure. The LoRA constraint already performs the dimensionality reduction that P-GAP aims to achieve.

### Fundamental Circularity

P-GAP needs a good gradient estimate to build the subspace. But getting a good gradient estimate IS the hard problem in ZO optimization. If we could estimate the gradient well enough to build a useful subspace, we wouldn't need P-GAP — we'd just use the gradient directly.

This circularity is masked in the original paper because:
1. They use full-parameter training where d is the full model size (2.7B) but individual weight matrices have meaningful internal structure
2. Their experimental setup may differ in ways that make the subspace more informative

## Assumption Validation

**A6 (from design spec): "P-GAP transfers to LoRA fine-tuning — HIGH RISK"**

**VALIDATED AS HIGH-RISK, OUTCOME: NEGATIVE.** P-GAP does not transfer to LoRA ZO training for three independently confirmed reasons:

1. **Theoretical**: SNR of gradient estimates is √(h/d) ≈ 0.004 with h=32, d=1.7M. Basis Q is random, not gradient-aligned.
2. **Empirical**: Projected grad is 1000x smaller. Convergence degrades proportionally to fraction of projected steps.
3. **Structural**: LoRA already constrains to a low-rank subspace. P-GAP's matrix-level SVD has ≤8 singular values to work with.

## Cost of Validation

- V1 probe: ~30 seconds (32 probes × 2 forward passes)
- V2a baseline: 249 seconds (500 steps × 450ms)
- V2b P-GAP r=4: 248 seconds
- V2c P-GAP r=8: 248 seconds
- Total: ~13 minutes machine time
- Code written: ~200 lines (helper functions + training loop branch)
- Analysis time: Several hours (mathematical derivation, literature review)

## Recommendations

1. **Do not pursue P-GAP for LoRA ZO training.** The fundamental SNR limitation makes it impossible to build a useful gradient subspace from practical numbers of probes.

2. **P-GAP might work for full-parameter ZO** where per-layer gradient matrices have meaningful low-rank structure. This was not tested (we only use LoRA).

3. **Focus on proven optimizations**: Phase 2 (conv-fused) delivers 1.75x speedup (254ms/step vs 452ms CPU) with zero convergence impact. This is the best result from the pipeline.

## Phase Summary

| Phase | Status | Result |
|-------|--------|--------|
| 1: Conv1x1 hybrid | ✅ Done | 403ms/step (0.89x CPU, IO-limited) |
| 2: Fused conv kernels | ✅ Done | 254ms/step (1.78x CPU, 1.75x gate) |
| 3: FZOO multi-perturbation | ✅ Done | No wall-time convergence benefit |
| **4: P-GAP** | **❌ Negative** | **Degrades convergence vs baseline** |
