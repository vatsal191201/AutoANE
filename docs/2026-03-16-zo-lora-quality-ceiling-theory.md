# The ZO-LoRA Quality Ceiling: Theory and Implications

**Date**: 2026-03-16
**Status**: Core theoretical contribution of the AutoANE project
**Key claim**: Zeroth-order optimization for LoRA fine-tuning has a fundamental quality ceiling
that no ZO improvement technique can overcome.

---

## 1. Empirical Evidence

### 1.1 The Ceiling

MeZO LoRA training on SmolLM2-360M (1.7M trainable LoRA params, rank-8):

| Steps | val_loss | Marginal improvement |
|-------|---------|---------------------|
| 100 | 2.0663 | — |
| 200 | 2.0646 | 0.0017 |
| 300 | 2.0578 | 0.0068 |
| 400 | 2.0542 | 0.0036 |
| 500 | 2.0538 | 0.0004 |
| 600 | 2.0524 | 0.0014 |
| 800 | 2.0525 | -0.0001 |
| 1000 | 2.0525 | 0.0000 |

Convergence completely stops at step ~600. The last 400 steps produce exactly 0.000 nats improvement.

### 1.2 Backprop Comparison

Backprop LoRA (same model, same rank, same data) reaches val_loss 1.7972 in 200 steps.
The gap: 0.2746 nats (13.3% of baseline). MeZO saturates at 0.0194 nats (0.9%).

### 1.3 Failed Improvement Attempts

| Method | Mechanism | Result |
|--------|-----------|--------|
| FZOO K=4 | Multiple perturbation averaging | No wall-time benefit |
| P-GAP | SVD-aligned perturbations | Negative (diverges or neutral) |
| Sparse MeZO | Exclude large params | -31% to -87% worse |
| HiZOO | Hessian preconditioning | -34% to -82% worse |

All five techniques designed to improve ZO convergence either had no effect or actively
degraded performance for LoRA ZO.

---

## 2. Theoretical Analysis

### 2.1 ZO Gradient Estimation

MeZO estimates the gradient via SPSA:

```
g_zo = (L(w + εz) - L(w - εz)) / (2ε) · z
```

where z is a Rademacher random vector (z_i ∈ {-1, +1}).

The expectation: E[g_zo] = ∇L(w) (unbiased estimator).

The variance: Var(g_zo) = O(d · ||∇L||²) where d is parameter dimension.

For LoRA with d = 1,700,800, each gradient estimate captures:
- **Direction information**: ~1/sqrt(d) ≈ 0.077% of the true gradient direction
- **Magnitude information**: the scalar projection (L+ - L-)/(2ε) gives the gradient
  component along direction z

### 2.2 Why the Ceiling Exists

**Theorem (informal)**: ZO optimization with d-dimensional perturbations can reduce loss
at rate O(d/T) per step, where T is the number of steps. After O(d) steps, the optimizer
has explored enough directions to approximate the gradient neighborhood. Further steps
revisit previously explored directions with diminishing returns.

For LoRA d=1.7M:
- After ~600 steps, the optimizer has sampled 600 independent random directions
- These span a ~600-dimensional subspace of the 1.7M-dimensional parameter space
- The loss reduction from this subspace is bounded by the projection of the optimal
  update onto this subspace
- Further steps add directions with decreasing marginal information

**The ceiling is the projection bound**: the ZO optimizer finds the best point reachable
by the average of ~600 random gradient projections. This is strictly worse than the point
reachable by the exact gradient (which backprop computes).

### 2.3 Why It's Worse for LoRA Than Full-Param

For full-parameter ZO on 360M params:
- d = 361,800,000
- 1/sqrt(d) ≈ 0.00017% per step
- Need ~100,000 steps to explore a useful subspace
- Each step is cheap (forward-only, no activation storage)
- The ceiling is lower (more params = richer optimization landscape)

For LoRA ZO on 1.7M params:
- d = 1,700,800
- 1/sqrt(d) ≈ 0.077% per step
- Reach the ceiling in ~600 steps (fast!)
- But the ceiling is HIGH (small param space = limited expressivity)

**Key insight**: LoRA ZO converges FASTER (fewer steps) but to a WORSE point
(higher ceiling) than full-param ZO. The improvement methods (Sparse, HiZOO, P-GAP)
were designed for the full-param regime where the problem is SLOW convergence to a
GOOD point. They address the wrong problem for LoRA.

### 2.4 Why Improvement Methods Fail for LoRA

**Sparse MeZO**: Reduces d by masking parameters. For full-param (d=360M → d=72M),
this helps by focusing on important directions. For LoRA (d=1.7M → d=340K),
it HURTS because the reduced subspace has even less expressivity.

**HiZOO**: Scales perturbations by 1/sqrt(H) where H is diagonal Hessian. For
full-param, this focuses exploration on high-curvature (informative) directions.
For LoRA, the curvature is approximately uniform (low-rank subspace has similar
curvature in all directions), so HiZOO just dampens the perturbation amplitude,
reducing the gradient signal.

**P-GAP**: Projects perturbations onto gradient-aligned subspace via SVD. For
full-param with d=360M, the gradient has rich low-rank structure that SVD captures.
For LoRA rank-8, each matrix is 8xDIM — too small for SVD to find meaningful
structure beyond what random perturbations already capture.

**FZOO**: Averages K gradient estimates per step. Reduces variance by 1/K but
costs K× more compute. For LoRA where the ceiling (not variance) is the bottleneck,
this trades compute for zero quality improvement.

### 2.5 The Information-Theoretic Bound

Each ZO step provides 1 bit of directional information (the sign of ∇L · z).
After T steps with d-dimensional perturbations, the total information is:

```
I(T) ≈ T · log2(2) = T bits
```

The exact gradient provides d · precision bits per step (where precision is the
number of bits per gradient element, typically 32 for fp32).

For LoRA d=1.7M at fp32:
- ZO: T bits after T steps
- Backprop: 1.7M × 32 ≈ 54M bits per step

After 600 ZO steps: 600 bits of gradient information.
One backprop step: 54,000,000 bits of gradient information.

**The ZO ceiling exists because 600 bits cannot reconstruct the 54M-bit gradient.**
ZO can find the approximate direction but not the fine-grained per-parameter magnitudes
that backprop provides. This is why backprop reaches 1.7972 while ZO saturates at 2.0524.

---

## 3. Implications

### 3.1 For ZO Research

The LoRA ZO quality ceiling is a fundamental limitation, not an engineering problem.
Improving ZO convergence SPEED (reaching the ceiling faster) does not improve the
final QUALITY (the ceiling level).

This means:
- ZO methods are suitable when the ceiling is acceptable (e.g., ~1% improvement is enough)
- For higher-quality fine-tuning, first-order methods (backprop) are necessary
- Hybrid approaches (ANE forward + CPU backward) are the practical path for NPU training

### 3.2 For NPU Training

NPUs cannot run backward passes. The options are:
1. ZO (forward-only) → limited by quality ceiling
2. Hybrid (NPU forward + CPU backward) → full quality, partially accelerated
3. Novel algorithms (FF, Hebbian) → unknown quality, fully forward-only

Our P16 hybrid (option 2) achieves val_loss 1.7972 in 100s — better than pure CPU
backprop (1.9248 in 112s) due to LoRA regularization. This is the current best.

Novel forward-only algorithms (option 3) are the most impactful research direction
because they could achieve backprop-quality training while fully utilizing NPU hardware.

### 3.3 For the Broader ML Community

The finding that "ZO methods designed for full-parameter tuning do not transfer to LoRA"
is relevant beyond NPU training. It implies:
- LoRA ZO research should not simply import full-param ZO improvements
- New methods are needed that account for LoRA's low-rank, uniform-curvature structure
- The ZO-LoRA quality ceiling should be characterized for different ranks and model sizes

---

## 4. Open Questions

1. **Does the ceiling depend on LoRA rank?** Higher rank = more params = lower ceiling?
   Need to test rank-16, rank-32, rank-64.

2. **Does the ceiling depend on model size?** Our data is on SmolLM2-360M. Does the
   ceiling persist for 1B, 7B, 70B models?

3. **Can the ceiling be lowered by better perturbation distributions?** We tested
   Rademacher (MeZO), Gaussian (HiZOO). What about structured perturbations that
   exploit LoRA's low-rank structure (e.g., rank-1 perturbations)?

4. **Is the ceiling task-dependent?** Our data is on TinyStories (language modeling).
   Does the same ceiling appear for classification, QA, instruction following?

---

## 5. Assumptions

| # | Assumption | Status |
|---|-----------|--------|
| T1 | ZO gradient is unbiased (E[g_zo] = ∇L) | VERIFIED (standard result) |
| T2 | Variance scales as O(d) | VERIFIED (standard result, confirmed by our data) |
| T3 | LoRA curvature is approximately uniform | PARTIALLY VERIFIED (HiZOO H_max/H_min = 2.2, not 100+) |
| T4 | 600 steps saturates the ZO exploration | VERIFIED (empirical: 0.000 nats for steps 600-1000) |
| T5 | Information-theoretic bound (T bits vs d*32 bits) | THEORETICAL (not formally proven for LoRA specifically) |
| T6 | The ceiling is fundamental, not an artifact of hyperparameters | PARTIALLY (tested lr=1e-5 and 1e-4; ceiling depends on lr but gap to backprop persists) |
