# Phase 5: Sparse-HiZOO for LoRA Zeroth-Order Training — Design Spec

## Goal

Improve MeZO+LoRA-split convergence rate through two orthogonal techniques: (1) sparse perturbation that excludes large-magnitude parameters (Sparse MeZO), and (2) diagonal Hessian preconditioning that adapts perturbation scale per-parameter (HiZOO). Three sequential experiments with go/no-go gates determine whether each technique, and their combination, benefits our specific setup.

## Context

Phase 2 achieved 262ms/step = 1.71x faster than CPU via conv-fused kernels. The remaining bottleneck is MeZO's convergence rate: ~100x slower per step than backprop. Two recent papers address ZO convergence:

- **Sparse MeZO** (NeurIPS 2025, arXiv:2402.15751): Perturb only small-magnitude parameters. 9% accuracy improvement and 3.5x speedup on full-parameter ZO.
- **HiZOO** (ICLR 2025, arXiv:2402.15173): Diagonal Hessian-informed perturbation scaling. 8x convergence speedup on full-parameter ZO.

### Critical Literature Caveats (verified from source code + paper)

1. **HiZOO's "8x speedup" is for full-parameter fine-tuning.** With LoRA, improvement is only ~1% accuracy (62.1% vs 61.0% MeZO). The big wins do not transfer to LoRA in the paper's experiments.
2. **HiZOO's default config (`hessian_smooth_type='constant0'`) is literally MeZO** — alpha=0 means no Hessian update. You must set alpha>0 explicitly.
3. **Alpha requires per-task grid search** with no principled default. Values range 1e-12 to 1e-2. Too large causes gradient explosion.
4. **Sparse MeZO was never tested with LoRA.** The small-parameter insight is unvalidated in LoRA subspace.
5. **LOREN outperforms HiZOO by 6%** on RoBERTa-large (AAAI 2026), suggesting diagonal Hessian may be insufficient.
6. **bf16 Hessian estimation causes catastrophic failures** on OPT-13B (accuracy drops from 69.5% to 55.6%). We use fp32, so this risk is mitigated.

### Why we test anyway

- Our setup differs from the papers: SmolLM2-360M (not OPT/RoBERTa), attention-only LoRA rank-8, conv-fused ANE, TinyStories data
- Sparse MeZO's insight about large-magnitude parameters may apply to our RMS norms (~1.0) vs LoRA weights (~0.01-0.1) — a 100x magnitude difference
- Even marginal improvements compound: 1% accuracy + 10% fewer steps = meaningful at scale
- Both experiments are low-cost (<30 minutes total compute)
- Negative results are documented with equal rigor

## Model & Baseline

| Parameter | Value |
|-----------|-------|
| Model | SmolLM2-360M (pretrained) |
| Architecture | 32 layers, DIM=960, Q_DIM=960, KV_DIM=320, HIDDEN=2560 |
| LoRA | Rank-8, attention-only (Aq/Bq/Ak/Bk/Av/Bv/Ao/Bo × 32 layers) |
| Trainable params | 1,700,800 (1,638,400 LoRA + 62,400 RMS norms) |
| Baseline speed | 262 ms/step (conv-fused), 447 ms/step (CPU) |
| Baseline val_loss | ~2.06 (pretrained checkpoint) |
| Optimizer | MeZO SPSA, lr=1e-4, ε=1e-3 |
| Data | TinyStories, SEQ=256 |

## Algorithm: Three-Pass Sparse-HiZOO Step

### Standard MeZO (current, 2 forward passes):

```
1. Generate Rademacher z from seed
2. Perturb θ + ε·z         → forward → L+
3. Perturb θ - 2ε·z        → forward → L-
4. Restore θ + ε·z
5. grad = (L+ - L-) / (2ε)
6. Update: θ -= lr · grad · z
```

### Sparse-HiZOO (new, 3 forward passes):

```
1. Forward at θ (unperturbed) → L0                    [NEW]
2. Generate Rademacher z from seed
3. Apply sparse mask: z_masked[i] = z[i] · mask[i]    [NEW]
4. Scale by INVERSE Hessian: z_scaled[i] = z_masked[i] / √H[i]  [NEW]
   (High curvature → smaller perturbation; low curvature → larger)
5. Perturb θ + ε·z_scaled  → forward → L+
6. Perturb θ - 2ε·z_scaled → forward → L-
7. Restore θ + ε·z_scaled
8. grad = (L+ - L-) / (2ε)
9. Hessian update (per-element, NO extra H[i] factor):  [NEW]
   ΔL = L+ + L- - 2·L0
   H[i] = (1-α)·H[i] + α·|ΔL/ε²|·z[i]²
   H[i] = clamp(H[i], 1e-8, 1e6)
10. Update: θ[i] -= lr · grad · z_scaled[i]
    (= θ[i] -= lr · grad · z[i] · mask[i] / √H[i])
```

**Why inverse scaling (step 4)**: HiZOO's preconditioner uses H^{-1/2} so that
high-curvature parameters get smaller perturbations (safer exploration) and
high-curvature parameters also get smaller updates (step 10). This is the
standard Newton-method preconditioning: step ∝ H^{-1} · gradient.

**Why no H[i] factor in Hessian update (step 9)**: The second-order finite
difference estimator for the i-th diagonal Hessian element is:
`h_ii ≈ z_i² · (L+ + L- - 2·L0) / ε²`. For Rademacher z, z_i²=1, so each
step gives a noisy global curvature estimate. Over many EMA steps, different
random draws of z create per-element differentiation because each draw weights
parameters differently through the loss landscape. Including an extra H[i]
factor would make the update multiplicative (H[i] ← H[i]·scalar), preserving
all ratios forever and preventing any differentiation — defeating the purpose.

**Note on z_scaled perturbation vs raw z in Hessian**: We perturb by
ε·z/√H but estimate the Hessian using raw z[i]². This is a simplification.
The exact HiZOO formulation accounts for the preconditioned perturbation in
the estimator, but since z[i]²=1 for Rademacher, the correction is a global
scalar that the EMA absorbs. The per-element differentiation comes from the
loss response ΔL varying with different random z vectors across steps.

### Backward compatibility

| alpha | sparse_ratio | Behavior |
|-------|-------------|----------|
| 0 | 0 | Standard MeZO (2 passes, no mask, no Hessian) |
| 0 | >0 | Sparse MeZO only (2 passes, mask, no Hessian) |
| >0 | 0 | HiZOO only (3 passes, no mask, Hessian scaling) |
| >0 | >0 | Sparse-HiZOO (3 passes, mask + Hessian) |

When alpha=0, the L0 forward pass is skipped entirely (2-pass mode preserved).
When alpha=0, H[i] stays at its initial value of 1.0 forever, so the sqrt(H)
scaling in step 4 becomes a no-op (1/√1 = 1). Both the perturbation and
update reduce exactly to standard MeZO.

## New State

| Buffer | Size | Bytes | Purpose |
|--------|------|-------|---------|
| `diag_hessian` | 1,700,800 floats | 6.8 MB | Diagonal Hessian EMA estimate |
| `sparse_mask` | 1,700,800 uint8 | 1.7 MB | Binary perturbation mask |
| **Total** | | **8.5 MB** | Negligible vs model (1.4 GB) |

Initialization:
- `diag_hessian[i] = 1.0` for all i (identity preconditioner = MeZO)
- `sparse_mask[i]` computed from parameter magnitudes at step 0, refreshed every `mask_refresh` steps

## Sparse Mask Computation

```c
void compute_sparse_mask(LoRALayer *ll, LayerWeights *lw, float *rms_final,
                         int nlayers, uint8_t *mask, float keep_ratio) {
    // 1. Collect all |param| values into a temporary array
    // 2. Sort to find the keep_ratio percentile threshold
    // 3. mask[i] = (|param[i]| <= threshold) ? 1 : 0
    //    (keep SMALL parameters, exclude LARGE ones)
}
```

**Special case**: `sparse_ratio = 0.037` → uses GLOBAL thresholding (not per-group) so that all RMS norms (~1.0 magnitude) are excluded while all LoRA matrices (~0.01-0.1) are kept. The 100x magnitude gap between RMS and LoRA makes this a clean separation.

**Per-group thresholding** (for sparse_ratio > 0.037): When the ratio is large enough to cut into LoRA parameters, switch to per-group thresholds (A matrices, B matrices, RMS norms separately) to avoid skewing the mask toward one parameter type. The cutoff between global and per-group mode is at sparse_ratio = 0.05 (above which global thresholding would start cutting LoRA params).

**Risk of frozen RMS norms**: When RMS norms are excluded from perturbation and update, they become effectively frozen at their pretrained values. RMS norms control activation scale — freezing them means the model cannot adapt its internal normalization to compensate for LoRA weight changes. This might be fine (pretrained norms are well-calibrated for the base model) or harmful (LoRA changes may need compensating norm adjustments). Experiment 5a directly tests this.

## Hessian Update Details

```c
void update_hessian(float *H, const float *z, size_t n,
                    float loss_plus, float loss_minus, float loss_0,
                    float epsilon, float alpha) {
    float delta_L = loss_plus + loss_minus - 2.0f * loss_0;
    // Per-element Hessian estimate: h_ii ≈ |ΔL/ε²| · z_i²
    // For Rademacher z, z_i²=1, so this is a global curvature estimate.
    // Over many EMA steps, different z draws differentiate per-element.
    float curvature = fabsf(delta_L) / (epsilon * epsilon);
    for (size_t i = 0; i < n; i++) {
        float h_est = curvature * z[i] * z[i];  // NO H[i] factor
        H[i] = (1.0f - alpha) * H[i] + alpha * h_est;
        H[i] = fmaxf(H[i], 1e-8f);   // Floor: prevents 1/sqrt(H) explosion
        H[i] = fminf(H[i], 1e6f);    // Ceiling: prevents 1/sqrt(H) collapse to 0
    }
}
```

**Why no H[i] factor in h_est**: Including H[i] makes the update multiplicative
(H[i] ← H[i] · (1 + α·(curvature-1))), which scales all elements uniformly
and never differentiates between parameters. The correct estimator uses only
the raw curvature and z[i]², allowing the EMA to build per-element estimates
from the stochastic variation across steps.

**Floor 1e-8 (not 1e-30)**: Since H is used as 1/√H in perturbation scaling,
H=1e-30 would give 1/√H ≈ 3.16e+15, causing overflow. H=1e-8 gives
1/√H ≈ 1e+4, which is the maximum practical perturbation amplification.

**Note on z[i]**: We need the raw z values (before masking/scaling) for the
Hessian update. The mask is applied to the perturbation but NOT to the Hessian
estimation — we want curvature info for all parameters.

**Caveat on masked parameters**: When mask[i]=0, the perturbation does not
probe parameter i, so ΔL contains no curvature information specific to that
parameter. The Hessian estimate for masked parameters tracks the global
curvature scalar (from the unmasked parameters' contribution to ΔL), which is
a biased but non-zero estimate. This is acceptable because: (1) the mask
refreshes every 100 steps, (2) stale Hessian estimates are smoothed by the
EMA, and (3) the alternative (not updating masked params' Hessian) risks
stale H values persisting indefinitely after a mask change.

**Assumption A3 (explicit)**: Sparse mask should NOT affect Hessian update.
This is our design choice — the papers don't address this interaction since
they don't combine the two methods.

## CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sparse-ratio` | float | 0.0 | Fraction of params to EXCLUDE (0=none, 0.8=keep 20%) |
| `--hessian-alpha` | float | 0.0 | Hessian EMA rate (0=disabled, >0=enable HiZOO) |
| `--mask-refresh` | int | 100 | Recompute sparse mask every N steps |

## Experimental Protocol

### Experiment 5a: Sparse MeZO (RMS Exclusion)

**Hypothesis**: RMS norm weights (~1.0 magnitude, 62,400 params, 3.7% of total) inject disproportionate ZO noise. Excluding them improves convergence.

**Configs**:

| Run | sparse_ratio | alpha | Description |
|-----|-------------|-------|-------------|
| 5a-baseline | 0.0 | 0.0 | Standard MeZO (control) |
| 5a-rms-only | 0.037 | 0.0 | Exclude RMS norms only |
| 5a-50pct | 0.5 | 0.0 | Exclude largest 50% by magnitude |
| 5a-80pct | 0.8 | 0.0 | Exclude largest 80% (Sparse MeZO's best) |

**Protocol**: 500 steps each, same checkpoint, same data seed. Report 50-step moving average val_loss at step 500.

**Go/no-go**: Val_loss improvement must be > 0.001 (>7.5% relative to baseline 500-step delta of ~0.013). A single run with marginal improvement could be noise — we require a meaningful signal. If borderline, re-run with 2 additional seeds (43, 44) and require mean improvement > 0.

**Duration**: 4 × 500 × 262ms ≈ 9 minutes.

**Mini validation before experiment**: Print parameter magnitude histogram (LoRA A mean/std, LoRA B mean/std, RMS mean/std) to verify the magnitude gap assumption.

### Experiment 5b: HiZOO (Diagonal Hessian)

**Hypothesis**: Curvature-adaptive perturbation scaling improves convergence enough to justify the 50% compute overhead (3 passes vs 2).

**Configs**:

| Run | sparse_ratio | alpha | Description |
|-----|-------------|-------|-------------|
| 5b-baseline | 0.0 | 0.0 | Standard MeZO (same as 5a-baseline) |
| 5b-alpha-8 | 0.0 | 1e-8 | Conservative Hessian |
| 5b-alpha-6 | 0.0 | 1e-6 | Moderate Hessian |
| 5b-alpha-4 | 0.0 | 1e-4 | Aggressive Hessian |

**Protocol**: 500 steps each. 3 forward passes per step (extra ~131ms for L0).

**Go/no-go**: val_loss improvement at 500 steps must be >1.5× baseline delta (to justify 1.5× compute cost per step).

**Duration**: 4 × 500 × 393ms ≈ 13 minutes.

**Mini validation before experiment** (10-step diagnostic):
1. Print L0, L+, L- at each step to verify ΔL = L+ + L- - 2L0 is well-behaved
2. Print H[i] stats (min, max, mean, std) to verify no explosion/collapse
3. Verify H stays in [1e-30, 1e+6] range

### Experiment 5c: Sparse-HiZOO (Combined)

**Prerequisite**: Both 5a and 5b show positive signal.

**Configs**:

| Run | sparse_ratio | alpha | Description |
|-----|-------------|-------|-------------|
| 5c-combined | best from 5a | best from 5b | Combined |
| 5c-vs-sparse | best from 5a | 0.0 | Sparse-only (for comparison) |
| 5c-vs-hessian | 0.0 | best from 5b | Hessian-only (for comparison) |

**Go/no-go**: Combined must beat the better individual method.

**Duration**: (2 × 500 × 393ms) + (1 × 500 × 262ms) ≈ 8.7 minutes.
(Run 5c-vs-sparse uses 2 passes since alpha=0; other two use 3 passes.)

## Files Modified

| File | Changes |
|------|---------|
| `train_mezo.m` | Sparse mask computation, Hessian buffer, modified perturbation, 3-pass step, CLI flags, diagnostics |

**Files NOT modified**: config.h, cpu_ops.h, mil_dynamic.h, io.h, DO_FORWARD_PASS macro, conv-fused kernels.

**Estimated new code**: ~150-200 lines in train_mezo.m.

## Assumptions Registry

| # | Assumption | Risk | Validation |
|---|-----------|------|------------|
| A1 | Same batch for L0/L+/L- | None | Code inspection (batch selected once per step) |
| A2 | sqrt(H) safe with floor 1e-30 | Low | 10-step Hessian diagnostic |
| A3 | Sparse mask should not affect Hessian update | Medium | Design choice; ablation if time permits |
| A4 | RMS norms (~1.0) are "large" vs LoRA (~0.01) | Low | Print magnitude stats before experiment |
| A5 | HiZOO's marginal LoRA results may differ for our setup | Medium | Experiment 5b directly tests this |
| A6 | fp32 Hessian estimation is stable | Low | We use fp32; paper warns bf16 fails |
| A7 | Sparse MeZO's small-param insight transfers to LoRA | High | Experiment 5a directly tests this |
| A8 | Alpha=1e-6 is reasonable starting point | Medium | Sweep 1e-8, 1e-6, 1e-4 |
| A9 | L0 forward pass costs ~131ms on conv-fused ANE | Low | Measure in 10-step diagnostic |
| A10a | Hessian feedback loop does not cause oscillation | Medium | Monitor H variance across steps (not just min/max/mean) |
| A11 | Frozen RMS norms do not impair LoRA convergence | Medium | Experiment 5a-rms-only directly tests this |
| A10 | Mask refresh every 100 steps is sufficient | Low | LoRA weights change slowly under ZO |

## Explicitly Not Assumed

- HiZOO will work for LoRA (paper shows marginal results)
- Sparse MeZO works with LoRA (never tested in literature)
- Combined approach is better than either individual method
- Any specific alpha value is correct without testing
- The magnitude gap between RMS and LoRA params persists during training

All three are hypotheses to be tested. Negative results documented with equal rigor.

## References

| Paper | Venue | Key Contribution | arXiv |
|-------|-------|-----------------|-------|
| HiZOO | ICLR 2025 | Diagonal Hessian ZO preconditioning | 2402.15173 |
| Sparse MeZO | NeurIPS 2025 | Magnitude-based parameter selection for ZO | 2402.15751 |
| LOREN | AAAI 2026 | Low-rank curvature preconditioner | 2511.07971 |
| MeZO | NeurIPS 2023 | SPSA for LLM fine-tuning | 2305.17333 |
| ZOSA | 2025 | O(ε²) bias proof for Rademacher | 2511.09156 |
| SubZero | ICCV 2025 | Random subspace ZO | 2410.08989 |
