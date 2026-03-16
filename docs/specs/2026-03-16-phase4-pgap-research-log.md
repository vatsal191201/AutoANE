# Phase 4: P-GAP Research Log — Partial Negative Result (Simplified Implementation)

## Summary

A simplified flat-vector subspace projection approach was tested and found to degrade convergence. However, post-hoc literature review reveals that the implementation differs fundamentally from the actual P-GAP algorithm in several critical ways. The negative result validates that "flat-vector QR-based random subspace projection with standard MeZO hyperparameters" does not work, but does NOT invalidate P-GAP as described in the paper.

## Background

P-GAP (arXiv:2510.18228) proposes projecting SPSA perturbations into a gradient-aligned subspace estimated via SVD. The paper reports 3-5x fewer iterations on SST-2, SNLI, DROP, and specifically tests P-GAP+LoRA with strong results (76.6% SQuAD vs 63.4% MeZO LoRA on OPT-2.7B).

## CRITICAL: Implementation vs Paper Differences

Post-experiment literature review revealed that our implementation differs from P-GAP in fundamental ways:

| Aspect | Our Implementation | Paper's Algorithm |
|--------|-------------------|-------------------|
| Subspace structure | Flat d=1.7M vector, QR | Per-matrix SVD (each LoRA matrix separately) |
| Perturbation type | Rademacher {±1}^d | Gaussian N(0, I_{r×r}) |
| Epsilon | 0.001 (MeZO default) | **0.1 for LoRA** (100x larger) |
| Learning rate | 1e-4 | **1e-2 to 5e-2** (100-500x larger) |
| Gradient alignment | None (random QR basis) | Delta-projection constraint |
| Probes per estimate | h=1 (single SPSA scalar × z) | h=10 (averaged gradient) |
| SVD granularity | Global (all params as vector) | Per-layer per-matrix |

These differences explain the negative result: our approach is not P-GAP, it's a much simpler "random subspace projection" that lacks P-GAP's core innovations (per-matrix structure, gradient alignment constraint, aggressive hyperparameters).

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

## Experiment V3: Faithful P-GAP Implementation

After identifying the critical implementation differences (V2 above), a faithful P-GAP implementation was built matching the paper algorithm:

**Implementation details:**
- Per-matrix SVD bases: 256 bases (32 layers × 8 LoRA matrices)
- Gaussian perturbations (Box-Muller from xoshiro256+)
- PROJECTION constraint: Z = Z_init - α×S_r for gradient alignment
- Probe phase: h=10 Gaussian SPSA probes, accumulate G per-matrix, SVD each
- Training step: Z_f = U × Z × V^T per matrix, concatenate, SPSA update
- Delta linear decay (2→0)

**Results (pretrained SmolLM2-360M, 500 steps from step 100, seed 42):**

| Config | val@200 | val@300 | val@400 | val@500 | Δloss | Result |
|--------|---------|---------|---------|---------|-------|--------|
| Baseline (lr=1e-4, ε=1e-3) | 2.0648 | 2.0596 | 2.0576 | 2.0571 | -0.0077 | Reference |
| P-GAP paper (lr=1e-2, ε=0.1) | 15.7452 | 15.3284 | 15.2529 | 15.0869 | DIVERGED | ❌ |
| P-GAP intermediate (lr=1e-3, ε=0.01) | 2.1983 | 6.5786 | 7.6616 | 7.6157 | DIVERGED | ❌ |
| P-GAP standard (lr=1e-4, ε=1e-3) | 2.0648 | 2.0587 | 2.0579 | 2.0571 | -0.0077 | Neutral |

**Key findings:**
1. Paper's hyperparameters (ε=0.1, lr=1e-2) diverge catastrophically on SmolLM2-360M LoRA
2. Intermediate hyperparameters (ε=0.01, lr=1e-3) also diverge, more slowly
3. Standard MeZO hyperparameters: P-GAP matches baseline exactly — no benefit, no harm
4. The per-matrix SVD structure and PROJECTION constraint are mathematically correct (verified) but do not improve convergence

**Analysis of divergence with paper hyperparams:**
The paper tests P-GAP+LoRA on OPT-2.7B (2.7B params). SmolLM2-360M is 7.5x smaller. The ε=0.1 perturbation with LoRA rank-8 matrices (8×960) applies perturbation magnitude ~0.1 × ||z|| where ||z|| ≈ √(8×960) ≈ 87.6, giving ||ε*z|| ≈ 8.76. For matrices with ||W|| ≈ O(1), this is a massive perturbation. The paper's aggressive hyperparameters assume much larger weight matrices where the relative perturbation is proportionally smaller.

**Analysis of neutral result with standard hyperparams:**
With ε=1e-3, the per-matrix SVD projection does not change convergence because:
1. For LoRA rank-8 matrices, SVD has at most 8 singular values — the subspace IS the full matrix
2. The PROJECTION constraint aligns perturbation with gradient direction, but with r=svd_r=8 (same as matrix rank), the projection is nearly identity
3. The probe cost (10 probes × 2 forward passes = ~8s every 100 steps) adds overhead with zero convergence benefit

## Recommendations

1. **Do not pursue P-GAP for LoRA ZO training.** Tested both simplified (V2) and faithful (V3) implementations. Neither provides convergence improvement. Paper hyperparams diverge; standard hyperparams are neutral.

2. **Root cause**: LoRA rank-8 matrices are too small for P-GAP's per-matrix SVD to find meaningful low-rank structure. The SVD rank equals the matrix rank — there is no dimensionality reduction to exploit.

3. **P-GAP might work for full-parameter ZO** where per-layer gradient matrices are large (960×960) and may have genuine low-rank structure. Not tested (we only use LoRA).

4. **Focus on proven optimizations**: Phase 2 (conv-fused) delivers 1.71x speedup (~262ms/step vs 447ms CPU) with zero convergence impact. This is the best result from the pipeline.

## Phase Summary

| Phase | Status | Result |
|-------|--------|--------|
| 1: Conv1x1 hybrid | ✅ Done | 403ms/step (1.04-1.11x CPU) |
| 2: Fused conv kernels | ✅ Done | ~262ms/step (1.71x CPU) |
| 3: FZOO multi-perturbation | ✅ Done | No wall-time convergence benefit |
| 4: Simplified flat-vector subspace | ❌ Negative | Degrades convergence |
| **4: Faithful P-GAP** | **❌ Negative** | **Diverges (paper params) or neutral (standard params)** |
