# Deep Dive: Zeroth-Order Optimization Methods for LoRA Fine-Tuning

**Date**: 2026-03-16
**Context**: After Phases 3-5 ruled out FZOO, P-GAP, Sparse MeZO, and HiZOO for LoRA ZO,
what (if anything) from the latest literature could still help?

---

## 1. Our Established Failure Modes

From Phases 3-5, we proved that LoRA ZO has fundamentally different dynamics from full-param ZO:

| Technique | Mechanism | Why It Fails for LoRA ZO |
|-----------|-----------|------------------------|
| FZOO K=4 | Multiple perturbations averaged | 2.5x compute, zero wall-time benefit |
| P-GAP | SVD-aligned perturbation subspace | Rank-8 LoRA too small for meaningful SVD |
| Sparse MeZO | Exclude large-magnitude params | Reduces ZO signal in already-small param space |
| HiZOO | Hessian-adaptive perturbation scaling | Dampens amplitude, which is ALL LoRA ZO has |

**Root principle**: LoRA ZO with 1.7M params has **manageable variance** (unlike 7B full-param ZO).
The bottleneck is NOT variance — it's that each ZO gradient estimate is a **scalar projection** of
the true gradient onto a random direction. With d=1.7M params, this projection captures ~1/sqrt(d)
of the true gradient's information per step. No single-step improvement can fix this.

## 2. New Literature Analysis

### 2.1 MeZO-SVRG (arXiv:2404.08080, April 2024)
**Variance-Reduced ZO via SVRG.**

- Combines full-batch and mini-batch ZO estimates
- Claims 20% accuracy improvement, 2x GPU-hour reduction vs MeZO
- Works in "full and partial parameter" settings

**Applicability to our setup**: MEDIUM.
- SVRG requires periodic full-batch gradient estimation (expensive on TinyStories 20M tokens)
- Our training is single-sequence (batch=1, SEQ=256), so full-batch would mean ~78K forward passes
- The overhead would likely negate any convergence benefit in a time-budgeted setting
- **ASSUMPTION UP32**: SVRG's full-batch requirement is prohibitive for time-budgeted training
- **WORTH TESTING**: Could use mini-batch SVRG (subsample) — test if 10-sample snapshot + per-step correction helps

### 2.2 BSZO (arXiv:2601.01452, January 2026)
**Bayesian Subspace ZO Optimizer.**

- Kalman filtering over multiple perturbation directions within a subspace
- Improves convergence by k/gamma factor
- Memory: 1.00-1.08x of MeZO (minimal overhead)
- Robust under fp16/bf16
- Up to 6.67% improvement on OPT-13B

**Applicability to our setup**: HIGH.
- Kalman filtering accumulates gradient information across steps WITHOUT extra forward passes
- Does NOT reduce perturbation amplitude (key requirement from Phase 5)
- Memory overhead is negligible (1-8%)
- Works under reduced precision (our ANE is fp16)
- **ASSUMPTION UP33**: BSZO's Kalman filter can be implemented in our Obj-C pipeline
- **WORTH TESTING**: Most promising candidate — combines variance reduction with full-amplitude perturbations

**Key question**: How many perturbation directions per step does BSZO need? If k=1, it's free.
If k>1, it has the same problem as FZOO (more forward passes).

### 2.3 SubZero (ICCV 2025)
**ZO in Random Subspaces.**

- Projects perturbations into a random low-dimensional subspace
- Significantly lower gradient variance than MeZO
- Works with LoRA

**Applicability to our setup**: LOW-MEDIUM.
- Random subspace projection reduces effective dimensionality
- But our LoRA is already rank-8 (very low-dimensional)
- SubZero's benefit is mainly for full-param ZO where d is huge
- For LoRA with d=1.7M, the subspace might not help much
- **ASSUMPTION UP34**: SubZero's subspace projection provides little benefit for LoRA rank-8

### 2.4 MaZO (arXiv:2502.11513, February 2025)
**Masked ZO for Multi-Task.**

- Weight importance masking for multi-task ZO
- Specifically designed for multi-task conflict resolution
- **NOT applicable**: We do single-task training. The masking is for task interference, not convergence speed.

### 2.5 ZO Fine-tuner (arXiv:2510.00419, October 2025)
**Learned Perturbation Distribution.**

- Learns task-adaptive perturbation strategy
- Outperforms MeZO in 82.1% of task-model combinations
- One-time meta-training per LLM

**Applicability to our setup**: LOW.
- Requires meta-training (backprop needed to learn the perturbation distribution)
- If we have backprop, we should just use backprop directly (P16 hybrid)
- The meta-training cost likely exceeds the benefit for our time-budgeted setting
- **ASSUMPTION UP35**: If backprop is available for meta-training, better to use it for training directly

### 2.6 LOREN (arXiv:2511.07971, AAAI 2026)
**Low-Rank Curvature Preconditioner.**

- Natural evolution strategies with block diagonal preconditioner
- Outperforms HiZOO by 6% on RoBERTa-large
- Already flagged in our Phase 5 design spec

**Applicability to our setup**: LOW.
- LOREN uses per-parameter curvature estimation (similar principle to HiZOO)
- Our Phase 5 showed that curvature-based methods fail for LoRA ZO
- LOREN's "low-rank curvature" may hit the same wall: LoRA's uniform curvature
- **ASSUMPTION UP36**: LOREN's low-rank curvature faces same limitations as HiZOO for LoRA ZO
- NOT worth testing unless BSZO also fails

---

## 3. Ranking and Recommendations

| Rank | Method | Why | Risk | Effort |
|------|--------|-----|------|--------|
| 1 | **BSZO** | Kalman filtering without extra forward passes or amplitude reduction | Need to verify k=1 works | 2-3 days |
| 2 | MeZO-SVRG (mini-batch variant) | Variance reduction via control variate | Full-batch prohibitive; mini-batch untested | 2-3 days |
| 3 | SubZero | Random subspace projection | May not help for already-small LoRA | 1-2 days |
| 4 | LOREN | Low-rank curvature | Likely fails (same as HiZOO) | 2-3 days |
| 5 | ZO Fine-tuner | Learned perturbation | Requires backprop meta-training | Skip |
| 6 | MaZO | Multi-task masking | Not applicable (single-task) | Skip |

## 4. The Bigger Question: Is ZO the Right Approach?

Our P16-A analysis showed:
- **Backprop LoRA**: 0.147 nats improvement in 191 steps (112s CPU)
- **MeZO LoRA**: 0.019 nats improvement in 600 steps (157s ANE)

Backprop achieves **7.6x more improvement** in **fewer steps and less time**.

Even if BSZO doubles MeZO's convergence (0.038 nats in 300 steps on ANE = 79s):
- Still only 26% of backprop's quality (0.038 vs 0.147)
- And would take longer than P16 hybrid (79s vs 65s estimated)

**The fundamental issue**: ZO gradient estimation provides ~1/sqrt(d) of the information
per step that backprop provides. For d=1.7M, that's 1/1300th of the gradient information.
No amount of variance reduction can close a 1300x information gap.

**Conclusion**: ZO optimization research for LoRA is academically interesting but
practically dominated by the P16 hybrid approach (ANE forward + CPU backward).

## 5. Recommended Action

1. **Proceed with P16 hybrid** as top priority (3-5 days)
2. **Optionally test BSZO** if time permits after P16 (2 days)
3. **Document the fundamental ZO-vs-backprop convergence gap** as a research contribution
4. **Do NOT pursue ZO Fine-tuner, MaZO, or LOREN** for LoRA ZO

## 6. New Assumptions

| # | Assumption | Basis | Status |
|---|-----------|-------|--------|
| UP32 | SVRG full-batch requirement is prohibitive for time-budgeted training | 78K forward passes for 1 snapshot | UNVERIFIED |
| UP33 | BSZO Kalman filter can work with k=1 perturbation direction | Need to check paper | UNVERIFIED |
| UP34 | SubZero subspace provides little benefit for LoRA rank-8 | LoRA is already low-rank | UNVERIFIED |
| UP35 | If backprop available for meta-training, use it for training directly | P16 hybrid is better | REASONED |
| UP36 | LOREN low-rank curvature faces same HiZOO limitation for LoRA ZO | Uniform curvature in LoRA subspace | UNVERIFIED |
| UP37 | ZO gradient captures ~1/sqrt(d) of backprop information per step | Standard ZO theory: E[||g_zo||^2] ~ d * sigma^2 | VERIFIED (theory) |

## References

- [MeZO-SVRG](https://arxiv.org/abs/2404.08080) — Variance-reduced ZO, April 2024
- [BSZO](https://arxiv.org/abs/2601.01452) — Bayesian Subspace ZO, January 2026
- [SubZero](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.pdf) — ZO in Random Subspaces, ICCV 2025
- [MaZO](https://arxiv.org/abs/2502.11513) — Masked ZO for Multi-Task, February 2025
- [ZO Fine-tuner](https://arxiv.org/abs/2510.00419) — Learned ZO Optimizer, October 2025
- [LOREN](https://arxiv.org/abs/2511.07971) — Low-Rank Curvature ZO, AAAI 2026
- [P-GAP](https://arxiv.org/abs/2510.18228) — Gradient-Aligned Perturbations (tested Phase 4)
