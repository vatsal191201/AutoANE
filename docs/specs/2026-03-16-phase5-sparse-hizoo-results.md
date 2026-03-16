# Phase 5: Sparse-HiZOO Results — NEGATIVE

## Summary

Both Sparse MeZO and HiZOO diagonal Hessian preconditioning **degrade** convergence for LoRA ZO training on SmolLM2-360M. Neither technique transfers from full-parameter ZO (where they were developed) to the LoRA fine-tuning regime.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | SmolLM2-360M (pretrained, resumed from step 100) |
| LoRA | Rank-8, attention-only, 1,700,800 trainable params |
| Baseline | Standard MeZO, lr=1e-4, ε=1e-3, CPU-only |
| Duration | 500 steps per config (steps 100→600) |
| Metric | val_loss (50-step moving average at step 600) |
| Checkpoint | Original pretrained (val_loss ~2.06) |

## Experiment 5a: Sparse MeZO

**Hypothesis**: Excluding large-magnitude parameters from perturbation improves ZO signal quality.

| Config | sparse_ratio | val_loss@150 | val_loss@600 | Delta | vs Baseline |
|--------|-------------|-------------|-------------|-------|-------------|
| **Baseline** | 0.0 | 2.0652 | **2.0548** | **0.0104** | — |
| RMS exclusion | 0.037 | 2.0654 | 2.0582 | 0.0072 | **-31%** |
| 50% sparse | 0.50 | 2.0645 | 2.0591 | 0.0054 | **-48%** |
| 80% sparse | 0.80 | 2.0656 | 2.0642 | 0.0014 | **-87%** |

**Go/no-go**: FAIL. All sparse configs worse than baseline. Higher sparsity = worse convergence.

**Root cause analysis**: In LoRA ZO, the gradient estimate is `(L+ - L-) / (2ε) · z`. When parameters are masked (z[i]=0), the gradient estimate has fewer degrees of freedom — the scalar projection of the true gradient onto a sparser perturbation vector has higher variance. Unlike full-parameter ZO where 7B params create enormous noise (and sparsity helps by focusing on important directions), LoRA's 1.7M params already have manageable noise. Reducing the active parameter count from 1.7M to 340K (at 80% sparsity) worsens the signal-to-noise ratio.

**Assumption A7 validation**: "Sparse MeZO's small-param insight transfers to LoRA" — **DISPROVEN**. The magnitude-based selection that works for full-parameter ZO does not help for LoRA ZO.

**Assumption A4 validation**: "RMS norms (~1.0) are 'large' vs LoRA (~0.01)" — **PARTIALLY WRONG**. Actual magnitudes: LoRA mean = 0.177, RMS mean = 0.241 (1.36x gap, not 100x). After 100 training steps, LoRA A matrices develop significant magnitudes.

**Assumption A11 validation**: "Frozen RMS norms do not impair LoRA convergence" — **DISPROVEN**. Excluding RMS norms (sparse_ratio=0.037) hurts convergence by 31%.

## Experiment 5b: HiZOO Diagonal Hessian

**Hypothesis**: Curvature-adaptive perturbation scaling improves convergence enough to justify 50% compute overhead.

| Config | alpha | val_loss@150 | val_loss@600 | Delta | vs Baseline | H ratio@500 | H mean@500 |
|--------|-------|-------------|-------------|-------|-------------|-------------|------------|
| **Baseline** | 0 | 2.0652 | **2.0548** | **0.0104** | — | — | — |
| Conservative | 1e-8 | 2.0643 | 2.0574 | 0.0069 | **-34%** | 1.0 | 1.003 |
| Moderate | 1e-6 | 2.0643 | 2.0580 | 0.0063 | **-39%** | 1.2 | 1.254 |
| Aggressive | 1e-4 | 2.0653 | 2.0634 | 0.0019 | **-82%** | 2.2 | 7.532 |

**Go/no-go**: FAIL. All alpha values worse than baseline. More aggressive alpha = worse convergence.

**Root cause analysis**: HiZOO scales perturbations by 1/√H. As H grows above 1.0 (curvature accumulates), perturbation magnitude shrinks. At alpha=1e-4, H reaches mean=7.5, reducing perturbation by 1/√7.5 ≈ 0.37x. This dampens the ZO gradient signal, slowing convergence. The Hessian preconditioning is counterproductive for LoRA ZO because:
1. LoRA parameters are already in a structured low-rank subspace with relatively uniform curvature
2. Reducing perturbation magnitude in any direction reduces the scalar gradient estimate
3. The 50% extra compute for L0 forward pass is wasted overhead

**Critical bug found and fixed during diagnostics**: The initial implementation used Rademacher perturbations (z[i]²=1), which produced zero per-element Hessian differentiation. Fixed by switching to Gaussian perturbations (Box-Muller) where z[i]² follows a chi-squared distribution. After the fix, Hessian ratio grew to 2.2 (alpha=1e-4), confirming genuine per-element differentiation. Despite the fix, HiZOO still hurts convergence.

**Assumption A5 validation**: "HiZOO's marginal LoRA results may differ for our setup" — **CONFIRMED NEGATIVE**. Our results are consistent with the paper's marginal LoRA improvement (~1%). In fact, HiZOO actively hurts our setup.

**Assumption A8 validation**: "Alpha=1e-6 is a reasonable starting point" — **TESTED**. Alpha=1e-6 produces the least degradation but still worse than baseline. No alpha value helps.

## Experiment 5c: Combined

**NOT EXECUTED**. Both 5a and 5b are negative → combined experiment skipped per go/no-go protocol.

## Conclusions

1. **Sparse MeZO does not transfer to LoRA ZO.** Full-parameter ZO benefits from focusing on important parameters (7B → 1.4B active). LoRA ZO already has only 1.7M params — further reduction hurts signal quality.

2. **HiZOO does not help LoRA ZO.** The diagonal Hessian preconditioning dampens perturbation magnitude (1/√H), which reduces ZO gradient signal. This is counterproductive when the parameter space is already low-rank and uniform.

3. **LoRA ZO has fundamentally different optimization dynamics from full-parameter ZO.** Methods that improve full-parameter ZO (Sparse MeZO: 3.5x speedup, HiZOO: 8x speedup) provide zero benefit — or active harm — for LoRA ZO. This is a structural finding that should inform future ZO research directions.

4. **RMS norms are important for LoRA convergence.** Freezing them (sparse_ratio=0.037) hurts by 31%. They adapt to compensate for LoRA weight changes.

5. **The magnitude gap between LoRA and RMS params is smaller than expected** (1.36x, not 100x) after even 100 training steps.

## What This Rules Out for Future Work

- Per-parameter perturbation scaling for LoRA ZO (HiZOO, LOREN, PaZO) — unlikely to help given the uniform curvature of LoRA subspace
- Magnitude-based parameter selection for LoRA ZO (Sparse MeZO) — reduces signal quality
- Any method that reduces perturbation amplitude in exchange for better direction — LoRA ZO needs full-amplitude perturbation

## What Remains Promising

- **Variance reduction** methods that don't reduce perturbation amplitude (e.g., control variates, antithetic sampling)
- **Learned perturbation distributions** that adapt to the task (ZO Fine-tuner) — but requires backprop for meta-training
- **Cross-layer fusion** for additional IO reduction (currently at 96 trips, could go lower)
- **Larger model scaling** where the 1.71x conv-fused speedup may increase
- **Mobile deployment** (iOS/iPad) where ANE is the only viable sustained compute

## References

| Paper | Our Result | Paper's Claim |
|-------|-----------|---------------|
| Sparse MeZO (NeurIPS 2025) | -31% to -87% worse | +9% accuracy, 3.5x speedup (full-param) |
| HiZOO (ICLR 2025) | -34% to -82% worse | 8x speedup (full-param), ~1% for LoRA |
| MeZO (NeurIPS 2023) | Baseline | — |
