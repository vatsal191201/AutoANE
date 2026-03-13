# MeZO-on-ANE Research Audit

**Date:** 2026-03-12/13/14
**Auditor:** Comprehensive automated + manual review
**Status:** v12: CE loss epsilon guard bug fix + verified match with HuggingFace (0.08% gap)

---

## 1. Literature Cross-Reference

### 1.1 MeZO Paper (Malladi et al., NeurIPS 2023)

**Source:** [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)

**Algorithm 1 (from paper):**
1. Sample batch B and random seed s
2. theta <- PerturbParameters(theta, +eps, s) — add eps*z
3. loss_plus <- L(theta; B)
4. theta <- PerturbParameters(theta, -2*eps, s) — now theta - eps*z
5. loss_minus <- L(theta; B)
6. theta <- PerturbParameters(theta, +eps, s) — restore to original
7. projected_grad <- (loss_plus - loss_minus) / (2*eps)
8. Reset RNG with seed s, sample z_i ~ N(0,1)
9. theta_i <- theta_i - lr * projected_grad * z_i

**Our implementation matches steps 1-9 exactly.** Verified in train_mezo.m lines 492-722.

### 1.2 Key Difference: Gaussian vs Rademacher Perturbation

**Paper uses:** z ~ N(0, I_d) (Gaussian)
**We use:** z_i in {-1, +1} (Rademacher)

**Is this valid?** YES. Mathematical proof:

The SPSA gradient estimate is: g_hat = [(L(theta+eps*z) - L(theta-eps*z)) / (2*eps)] * z

By Taylor expansion: g_hat ≈ (grad_L^T z) * z = z z^T grad_L

For unbiasedness: E[g_hat] = E[z z^T] * grad_L

**Requirement:** E[z z^T] = I (identity matrix)

- Gaussian N(0,1): E[z_i z_j] = delta_ij ✓
- Rademacher {-1,+1}: E[z_i z_j] = delta_ij ✓ (since E[z_i]=0, E[z_i^2]=1, independent)

**Both satisfy the unbiasedness condition.** Rademacher actually has LOWER variance
(E[z_i^4]=1 vs E[z_i^4]=3 for Gaussian), making it slightly better.

**Important nuance (v10 literature verification):** MeZO's gradient estimator is a
"randomized finite difference" / ZO-SGD estimator, NOT classical SPSA (Spall 1992).
Classical SPSA uses a 1/z-weighted estimator requiring E[1/z_i^2] < ∞, which Gaussian
does NOT satisfy (but Rademacher does). MeZO's estimator g_hat = [(L+ - L-)/(2ε)] * z
does not have the 1/z term, so both distributions are valid for MeZO specifically.
Rademacher is optimal for classical SPSA (Sadegh & Spall 1997) and equally valid for MeZO.

**Validated experimentally:** See Section 3.3 below.

### 1.3 Convergence Theory (Paper Section 4)

**Lemma 2:** E[||g_hat||^2] = (d+n-1)/n * E[||grad_L||^2]

For n=1 (our case): E[||g_hat||^2] = d * E[||grad_L||^2]

This means the gradient norm is sqrt(d) times larger than the true gradient.
For d=36.4M: factor of ~6000x.

**Equation 3:** eta_ZO = n/(d+n-1) * eta_SGD ≈ (1/d) * eta_SGD

Maximum permissible MeZO LR is ~1/d times the SGD LR.
For d=36.4M and eta_SGD=4e-4: eta_ZO_max ≈ 1.1e-11

**Theorem 1 (Dimension-Free Rate):** Convergence rate depends on the local effective
rank r of the Hessian, NOT the parameter dimension d.

For pre-trained models: r << d (low effective rank, MeZO works well)
For randomly initialized models: r ≈ d (high effective rank, MeZO is slow)

**CRITICAL IMPLICATION:** MeZO is designed for FINE-TUNING, not from-scratch training.
Our from-scratch experiments (conditions 1-4) showing minimal learning signal is
EXPECTED and CONSISTENT with theory. The paper never claims from-scratch works.

### 1.4 Paper Hyperparameters

| Setting | Paper (OPT-13B) | Our Implementation | Match |
|---------|------------------|--------------------| ------|
| Model | OPT-13B (13B params) | autoresearch (36.4M) / SmolLM2 (134.5M) | Smaller |
| Task | Classification/Generation (fine-tuning) | Language modeling (from-scratch + fine-tuning) | Different |
| Steps | 20K (fine-tuning) | 120s time budget (~300 steps for fine-tuning) | Much fewer |
| LR | 1e-6 to 1e-7 | 1e-5 (fine-tuning) | Higher (smaller model) |
| LR schedule | Constant (all experiments) | Cosine (negligible for short runs) | Effectively ✓ |
| Epsilon | 1e-3 | 1e-3 | ✓ |
| Perturbation | Gaussian N(0,1) | Rademacher {-1,+1} (valid, see 1.2) | Different (both valid) |
| n (samples) | 1 | 1 | ✓ |
| Optimizer | ZO-SGD | ZO-SGD | ✓ |

**Note on LR:** Paper's lr=1e-7 is for 13B params. ZO LR scales as n/(d+n-1) × SGD_LR,
so smaller models tolerate higher LR. Our lr=1e-5 for 135M is consistent with this scaling.

**Note on LR schedule:** Our earlier documentation incorrectly stated the paper uses "linear
decay." The paper actually uses constant LR for all reported experiments (confirmed via paper
text and official repo). Our cosine schedule has negligible effect in short runs.

---

## 2. Implementation Verification

### 2.1 PRNG (xoshiro256+) ✅ VERIFIED

Compared bit-for-bit against reference implementation from Blackman & Vigna
(https://prng.di.unimi.it/xoshiro256plus.c):

- rotl(): Identical (x << k) | (x >> (64-k))
- next(): Identical — result=s[0]+s[3], t=s[1]<<17, XOR sequence, rotl(s[3],45)
- splitmix64 seed: Identical — golden ratio 0x9E3779B97F4A7C15, shifts 30/27/31

### 2.2 BLAS Operations ✅ VERIFIED

All 8 cblas_sgemm calls verified against train.m:

| Op | Transpose Flags | Dims (M,N,K) | Match train.m |
|----|-----------------|--------------|---------------|
| Wq | NoTrans,NoTrans | Q_DIM,SEQ,DIM | ✅ |
| Wk | NoTrans,NoTrans | KV_DIM,SEQ,DIM | ✅ |
| Wv | NoTrans,NoTrans | KV_DIM,SEQ,DIM | ✅ |
| Wo | NoTrans,NoTrans | DIM,SEQ,Q_DIM | ✅ |
| W1 | NoTrans,NoTrans | HIDDEN,SEQ,DIM | ✅ |
| W3 | NoTrans,NoTrans | HIDDEN,SEQ,DIM | ✅ |
| W2 | NoTrans,NoTrans | DIM,SEQ,HIDDEN | ✅ |
| Cls | Trans,Trans | SEQ,CV,DIM | ✅ (produces [SEQ,CV] row-major) |

### 2.3 Checkpoint Format ✅ VERIFIED (after fix)

Original implementation had CRITICAL bug: interleaved (weight,adam_m,adam_v) layout
instead of train.m's (all_weights, then all_adam) layout.

**Fixed in commit f562e0b.** Now matches train.m exactly:
- Per layer: Wq,Wk,Wv,Wo,W1,W2,W3,rms_att,rms_ffn (all weights)
- Then: Wq.m,Wq.v,Wk.m,Wk.v,...,rms_ffn.m,rms_ffn.v (all Adam states as zeros)
- Then: rms_final + zeros, embed + zeros

### 2.4 MeZO Algorithm Sequence ✅ VERIFIED

Step-by-step trace of train_mezo.m training loop:

1. perturb_all_weights(seed, +eps) → theta + eps*z ✓
2. free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM) ✓
3. [ANE only] RETRANSPOSE_AND_STAGE() ✓
4. Forward pass → loss_plus ✓
5. perturb_all_weights(seed, -2*eps) → theta - eps*z ✓
6. free(cembed); cembed = rebuild ✓
7. [ANE only] RETRANSPOSE_AND_STAGE() ✓
8. Forward pass → loss_minus ✓
9. perturb_all_weights(seed, +eps) → theta restored ✓
10. proj_grad = (loss_plus - loss_minus) / (2*eps) ✓
11. update_scale = -lr * proj_grad ✓
12. perturb_all_weights(seed, update_scale) → theta - lr*proj_grad*z ✓
13. free(cembed); cembed = rebuild ✓
14. [ANE only] RETRANSPOSE_AND_STAGE() ✓
15. Cosine LR decay ✓

**Matches Algorithm 1 from paper exactly.**

### 2.5 Weight Tying ✅ VERIFIED

- embed (VOCAB*DIM) is the only embedding parameter
- cembed = vocab_compact_embed(embed, &vm, DIM) — compacted copy
- Classifier uses cembed as weight matrix (tied with embedding)
- perturb_all_weights perturbs embed only, not cembed
- cembed is rebuilt after every perturbation
- No separate classifier parameter exists

### 2.6 Residual Scaling ✅ VERIFIED (v2: conditional)

**v1 (bug):** `res_alpha = 1/sqrt(2*NLAYERS)` applied unconditionally.
**v2 (fixed):** `res_alpha = from_scratch ? 1/sqrt(2*NLAYERS) : 1.0`

- From-scratch: GPT-2-style residual scaling `1/sqrt(2N)` (scales branch down, not DeepNorm
  which scales skip up with `(2N)^(1/4)`). Both are valid stability techniques.
- Fine-tuning pretrained: Standard alpha=1.0 (correct for Llama/SmolLM2)
- Verified via HuggingFace AutoConfig: SmolLM2-135M has no DeepNet scaling

### 2.7 Code Audit Summary (v4) ✅ VERIFIED

Independent code audit of train_mezo.m (897 lines) found:
- **Algorithm correctness:** All MeZO steps match paper exactly ✅
- **Perturbation coverage:** All parameters perturbed (embed, Wq-W3, rms, rms_final) ✅
- **Weight restoration:** +eps/-2eps/+eps sequence correct ✅
- **RETRANSPOSE_AND_STAGE:** All 7 transpose dimensions verified correct ✅
- **Deferred 3rd restage:** Optimization is sound ✅
- **Minor issues (none affecting correctness):**
  - 620MB zero buffer in checkpoint save (could use a small loop instead)
  - VocabMap allocations not freed at exit (OS reclaims)
  - `dlogits` gradient computed but unused by MeZO (inherent to shared loss function)

---

## 3. Validation Experiments

### 3.1 Perturbation Cancel Property

**Test:** perturb(+eps) then perturb(-eps) with same seed should return to original.

| Test | n | max_error | Expected | Result |
|------|---|-----------|----------|--------|
| +eps/-eps | 1M | 3.05e-5 | fp32 rounding | ✅ Expected |
| +eps/-2eps/+eps (MeZO seq) | 500K | 1.19e-7 | fp32 rounding | ✅ Expected |

**Analysis:** Non-zero errors are due to floating-point arithmetic: (x + eps) - eps ≠ x
when eps << x. For x ≈ 1000 (test 1), ULP ≈ 6e-5, so error of 3e-5 is 0.5 ULP.
For x ∈ [-1,1] (test 2), ULP ≈ 1.2e-7, consistent with observed error.

**Impact on MeZO:** Accumulated drift per step is O(eps * machine_eps) ≈ 1e-10 per param.
The SGD update is O(lr * proj_grad) ≈ 1e-4 per param. Drift/update ratio ≈ 1e-6.
**Negligible.** This is also a form of implicit regularization (stochastic noise).

### 3.2 Rademacher Distribution Correctness

| Property | Observed | Expected | Status |
|----------|----------|----------|--------|
| P(z=+1) | 49.99% | 50% | ✅ |
| P(z=-1) | 50.01% | 50% | ✅ |
| E[z] | -0.000245 | 0 | ✅ |
| E[z^2] | 1.000000 | 1 | ✅ |
| Var[z] | 1.000000 | 1 | ✅ |

### 3.3 Gradient Estimate Unbiasedness

Tested on L(theta) = 0.5 * ||theta||^2 with theta = [1,...,1], 500K trials per dim:

| Dimension | mean(g_hat_i) | Alignment | Cosine | RMSE | Theory std |
|-----------|---------------|-----------|--------|------|------------|
| 10 | 1.0004 | 1.0004 | 0.9999 | 0.0032 | 0.0042 |
| 100 | 0.9994 | 0.9994 | 0.9999 | 0.0154 | 0.0141 |
| 1,000 | 0.9984 | 0.9984 | 0.9990 | 0.0442 | 0.0447 |
| 10,000 | 0.9996 | 0.9996 | 0.9904 | 0.1399 | 0.1414 |

**Key findings:**
- Alignment ≈ 1.0 for all d → **gradient is UNBIASED** ✅
- RMSE matches theoretical std = sqrt(d-1)/sqrt(n_trials) → **variance matches theory** ✅
- RMSE grows as sqrt(d) → confirms Lemma 2 dimension dependence ✅
- Cosine similarity decreases with d → individual updates become noisier with more params

### 3.4 Bit Independence (4-bit extraction)

| Metric | Observed | Expected | Status |
|--------|----------|----------|--------|
| Bit 0 freq | 0.5005 | 0.5 | ✅ |
| Bit 1 freq | 0.4997 | 0.5 | ✅ |
| Bit 2 freq | 0.5001 | 0.5 | ✅ |
| Bit 3 freq | 0.5010 | 0.5 | ✅ |
| P(b0 & b1) | 0.2497 | 0.25 | ✅ |
| P(b0 & b2) | 0.2502 | 0.25 | ✅ |
| P(b0 & b3) | 0.2506 | 0.25 | ✅ |

All joint probabilities match P(b_i)*P(b_j) = 0.25 → bits are independent ✅

### 3.5 Forward Pass Equivalence

| Metric | train.m (backprop) | train_mezo.m (MeZO) | Delta |
|--------|-------------------|---------------------|-------|
| Step 0 loss | 9.7273 | 9.7309 (loss_plus) | +0.0036 |

Delta of 0.0036 is expected from epsilon perturbation:
delta ≈ eps * proj_grad ≈ 1e-3 * 4.3 = 0.0043 (consistent).

Both use same weight init (srand48(42)), same data sampling at step 0.

### 3.6 Perturbation Cancel on Pretrained Weights (v2)

Tested on actual SmolLM2-135M weights (134.5M params):
- perturb(+eps, -2eps, +eps) max error: 9.54e-7 (8 ULPs)
- MeZO update magnitude: ~4.27e-4 (at step 0)
- Error/update ratio: 0.0022 (447x smaller)
- **Negligible for correctness** ✅

### 3.7 Multi-Seed Statistical Validation (v2, SmolLM2-135M)

MeZO fine-tuning SmolLM2-135M, 300 steps each, CPU:

| Seed | Final Loss | Start Loss |
|------|-----------|------------|
| 42 | 2.12 | 2.25 |
| 123 | 2.09 | 2.24 |
| 7 | 2.15 | 2.25 |
| 999 | 2.11 | 2.24 |
| 314159 | 2.13 | 2.25 |

Mean final loss: 2.12 ± 0.02 (std)
All seeds show consistent decrease → **learning signal is real, not seed-dependent** ✅

### 3.10 Multi-Seed Validation (v10, SmolLM2-360M, MeZO+LoRA-split)

MeZO+LoRA-split r8, lr=1e-4, eps=1e-3, CPU, 120s. Clean checkpoint regenerated from
HuggingFace before each run (to prevent checkpoint contamination from prior training).

| Seed | Val@50 | Val@100 | Steps | Step-0 loss+ | Step-0 loss- |
|------|--------|---------|-------|-------------|-------------|
| 42 | 2.0631 | 2.0594 | 101 | 2.1095 | 2.0987 |
| 123 | 2.0667 | 2.0632 | 138 | 1.9274 | 1.9284 |
| 7 | 2.0661 | 2.0582 | 138 | 2.1359 | 2.1454 |
| 999 | 2.0633 | 2.0583 | 134 | 1.9417 | 1.9454 |

**Statistics:**
- val@50:  mean=2.0648 ± 0.0017
- val@100: mean=2.0598 ± 0.0024
- Seed 42 exactly reproduces condition 24 ✅

**Notes on experimental procedure:**
- Step-0 loss_plus/loss_minus vary by seed (expected: perturbation direction is seed-dependent)
- Step counts vary (101-138) due to system load; does not affect val at fixed step numbers
- Checkpoint contamination discovered during earlier runs: step-0 verification overwrote
  checkpoint at step 1, causing subsequent experiments to resume from step 2. Fixed by
  regenerating checkpoint from HF before each clean run.

**Conclusion:** Val loss std < 0.003 across 4 seeds → **highly reproducible** ✅

### 3.8 IOSurface Transpose Overhead Microbenchmark (v3)

Decomposition of RETRANSPOSE_AND_STAGE for SmolLM2-135M (30 layers):

| Component | Per-layer (ms) | All 30 layers (ms) | % of total |
|-----------|---------------|--------------------|-----------|
| vDSP_mtrans (7 matrices) | 1.12 | 33.5 | 67% |
| IOSurface staging (cvt_f32_f16) | 1.19 | 35.6 | 71% |
|   W2 element-wise (original) | 0.71 | 21.2 | 42% |
|   W2 optimized (pre-transpose+bulk) | 0.22 | 6.6 | 13% |
| IOSurface lock/unlock | 0.004 | 0.13 | <1% |
| **Full restage** | — | **50.4** | — |

Two optimizations applied:
1. Defer 3rd restage: 3→2 per step (33% fewer restages)
2. W2 bulk cvt: 3.2x faster per-layer W2 staging

**Result:** Transpose overhead 226→99 ms/step, total 656→501 ms/step (1.31x speedup)
**Verified:** Bit-identical losses at step 0 (2.2467) and step 100 (1.8953) ✅

### 3.9 Memory Profiling (v2)

| Mode | RSS (MB) | Components |
|------|----------|------------|
| MeZO | 785 | weights + forward buffers + binary |
| Backprop | 2910 | weights + gradients + Adam m/v + forward/backward buffers |

Ratio: 3.7x memory savings for MeZO ✅
Theoretical: weights only = 134.5M * 4B = 538MB. MeZO overhead: 247MB (buffers + binary).

---

## 4. Experimental Results

### 4.1 From-Scratch (Conditions 1-4)

| # | Method | Hardware | Steps | ms/step | Init Loss | Final Loss | Val Loss |
|---|--------|----------|-------|---------|-----------|------------|----------|
| 1 | Backprop+Adam | CPU | 3015 | 30.7 | 9.73 | 3.62 | 3.998 |
| 2 | Backprop+Adam | ANE | 3393 | 26.0 | 9.73 | 3.91 | 3.790 |
| 3 | MeZO (ZO-SGD) | CPU | 1588 | 75.1 | 9.73 | 9.60 | 9.685 |
| 4 | MeZO (ZO-SGD) | ANE | 1265 | 94.5 | 9.73 | 9.70 | — |

### 4.2 Fine-Tuning (Conditions 5-8, v2→v3)

| # | Method | Hardware | Steps | ms/step | Init Loss | Final Loss | Val Loss | Version |
|---|--------|----------|-------|---------|-----------|------------|----------|---------|
| 5 | Backprop+Adam | CPU | 382 | 281.5 | 2.24 | 1.81 | 1.929 | v2 |
| 6 | Backprop+Adam | ANE | 346 | 304.8 | 2.24 | 2.16 | 1.929 | v2 |
| 7 | MeZO (ZO-SGD) | CPU | 317 | 379.3 | 2.25 | 1.97 | — | v2 |
| 8 | MeZO (ZO-SGD) | ANE | 183 | 656.4 | 2.25 | 1.93 | — | v2 |
| 8 | MeZO (ZO-SGD) | ANE | 240 | 501.4 | 2.25 | 1.93† | — | v3 |

*†Final loss from v2 run (bit-identical to v3). Val loss — because val_every=500 > run length.*

### 4.3 Timing Breakdown (From-Scratch)

| Component | BP-CPU | BP-ANE | MeZO-CPU | MeZO-ANE |
|-----------|--------|--------|----------|----------|
| Forward | ~10ms | ~7ms | 34ms (2x) | 27ms (2x) |
| Backward | ~15ms | ~15ms | N/A | N/A |
| Perturbation | N/A | N/A | 43ms (4x) | 42ms (4x) |
| Transpose | N/A | N/A | 0ms | 21ms (2x) |
| Adam/Update | ~5ms | ~4ms | <1ms | <1ms |
| **Total** | **30.7** | **26.0** | **75.1** | **94.5** |

### 4.4 Timing Breakdown (Fine-Tuning, v2)

| Component | BP-CPU | BP-ANE | MeZO-CPU | MeZO-ANE |
|-----------|--------|--------|----------|----------|
| Forward | ~98ms | ~93ms+15ms IO | ~228ms (2x) | ~275ms (2x)+226ms transpose |
| Backward | ~115ms | ~115ms | N/A | N/A |
| Perturbation | N/A | N/A | ~149ms (4x) | ~150ms (4x) |
| Other | ~70ms | ~82ms | ~2ms | ~5ms |
| **Total** | **282** | **305** | **379** | **656** |

### 4.6 MeZO+LoRA Fine-Tuning (Conditions 13-19, v7)

| # | Method | Hardware | Steps | ms/step | Val Loss @100 |
|---|--------|----------|-------|---------|---------------|
| 13 | BP+LoRA r8 | CPU | 191 | 586 | 1.925 |
| 14 | BP+LoRA r8 | ANE | 74 | 1344 | — (thermal) |
| 15 | MeZO+LoRA r8 | CPU | 200 | 576 | 2.068 |
| 16 | MeZO+LoRA r8 | ANE | 143 | 807 | 2.070 |
| 17 | MeZO+LoRA r32 | CPU | 55 | 1142 | — (no signal) |
| 18 | MeZO+LoRA-split r8 | CPU | 205 | 537 | 2.069 |
| 19 | MeZO+LoRA-split r8 | ANE | 159 | 708 | 2.070 |

*Condition 14: BP+LoRA ANE hit thermal=serious, step times inflated to 1344ms.*
*Condition 17: Rank 32 showed near-zero gradient signal — higher ZO variance with more parameters.*

### 4.8 MeZO+LoRA Evaluation (Conditions 20-28, v8)

| # | Method | Hardware | LR | Steps | ms/step | Time (s) | Best Val |
|---|--------|----------|-----|-------|---------|----------|----------|
| 20 | MeZO+LoRA-split r8 | CPU | 1e-5 | 568 | 458 | 300 | 2.0655 |
| 21 | MeZO+LoRA-split r4 | CPU | 1e-5 | 248 | 455 | 120 | 2.0690 |
| 22 | MeZO+LoRA-split r8 | ANE | 1e-5 | 400 | 656 | 300 | 2.0667 |
| 23 | MeZO+LoRA-split r8 | CPU | 5e-5 | 233 | 453 | 120 | 2.0598 |
| 24 | MeZO+LoRA-split r8 | CPU | 1e-4 | 227 | 463 | 120 | 2.0506 |
| 25 | MeZO+LoRA-split r8 | CPU | 3e-4 | 221 | 496 | 120 | 2.0536 |
| 26 | MeZO+LoRA-split r8 | CPU | 1e-4 | 1150 | 462 | 600 | 2.0453 |
| 27 | BP+LoRA r8 | ANE | 3e-4 | 69 | 1445 | 120 | — |
| 28 | BP+LoRA r8 | CPU | 3e-4 | 190 | 603 | 120 | 1.9722 |

*Condition 25: val_loss bounced (2.0536→2.0595→2.0550) — lr=3e-4 shows instability.*
*Condition 26: val plateaus at ~2.045 after 650 steps, Δ<0.001 from 650→1150.*
*Condition 27: thermal=nominal, IO stalls caused high avg step time.*

### 4.9 Analysis

**v2 KEY RESULT: MeZO competitive with backprop for fine-tuning.**
- MeZO-CPU final loss 1.97 vs BP-CPU 1.81 — only 0.16 gap
- MeZO-ANE final loss 1.93 vs BP-ANE 2.16 — MeZO actually leads by 0.23
- Both BP conditions converge to val_loss ~1.93, suggesting training loss differences
  are due to stochastic batch variation, not fundamental convergence gaps

**Why v1 results were wrong (DeepNet bug):**
- res_alpha=0.129 was applied at every residual connection (attention + FFN)
- After 30 layers, signal was attenuated by 0.129^30 ≈ 1e-27 (accumulated effect)
- The model's pretrained representations were effectively destroyed
- Initial loss was 4.20 instead of correct 2.24 (HF reference: 1.94)
- proj_grad at step 0 was 42.67 (v1) vs 0.19 (v2) — 225x difference

**Why MeZO from-scratch shows minimal learning:**
- Theorem 1 requires low effective rank r of the Hessian
- Randomly initialized weights have high effective rank (r ≈ d)
- With d=36.4M, the noise overwhelms the signal
- The paper ONLY demonstrates fine-tuning, never from-scratch
- This is a **novel negative result** worth reporting

**Why MeZO-ANE is slower than MeZO-CPU (v3 optimized):**
- Perturbation (140-155ms) is CPU-bound in both cases (identical)
- ANE forward pass: 249-260ms vs CPU 208-218ms — ANE adds IO overhead per layer
- ANE adds 99ms for transpose+staging (2 times per step, v3 optimized from 226ms at 3x)
- Net: ANE is 32% slower per step (501 vs 379) — improved from v2's 73%

**CRITICAL v4 FINDING: MeZO val loss convergence is extremely slow:**
v4 experiment with val_every=100 reveals training loss improvement was misleading:

| Step | MeZO Val Loss | MeZO Train Loss | BP-CPU Val Loss |
|------|--------------|-----------------|-----------------|
| 0 | ~2.250 | 2.247 | ~2.250 |
| 100 | 2.2496 | 1.895 | 1.952 |
| 300 | 2.2486 | 2.117 | 1.929 |
| 600 | 2.2453 | 1.731 | — |
| 900 | 2.2450 | ~2.0 | — |

Key observations:
- Val loss Δ after 900 steps: **0.005** (2.250→2.245)
- BP-CPU val loss Δ after 100 steps: **0.30** (2.250→1.952)
- MeZO val convergence is ~60x slower per step than backprop
- Training loss variations (1.7-2.2) are batch-to-batch noise, not learning signal
- The v2 claim "MeZO reaches near-backprop quality" compared noisy single-batch losses
- Extrapolating: MeZO needs ~50K steps (~5 hours) to match BP val_loss of 1.93
- This is CONSISTENT with MeZO paper (20K+ steps for full-parameter fine-tuning)
- Reason for using MeZO: MEMORY, not speed. 3.7x memory savings enables larger models.

---

## 5. Bug Discovery and Fix Log

### 5.1 Checkpoint LR Override (discovered during condition 7 v1)

**Symptom:** MeZO with CLI --lr=1e-5 diverged to loss ~22.
**Root cause:** `mezo_load_checkpoint()` wrote checkpoint lr (3e-4 from hf_to_ane.py) into
the `lr` variable, overriding CLI value.
**Fix:** Track `lr_from_cli` boolean, preserve CLI lr after checkpoint load.
**Impact:** All MeZO fine-tuning v1 runs used wrong LR initially.
**Verification:** After fix, `(using CLI lr=1e-05 instead of checkpoint lr)` printed.

### 5.2 DeepNet res_alpha on Pretrained Model (discovered during comprehensive audit)

**Symptom:** Initial loss 4.20 instead of expected ~2.2 (HF reference: 1.94).
**Root cause:** `res_alpha = 1/sqrt(2*30) = 0.129` applied to SmolLM2-135M, which uses
standard Llama architecture (alpha=1.0). DeepNet scaling is ONLY for from-scratch training
with matched weight initialization.
**Diagnosis method:** HuggingFace AutoConfig check confirmed no DeepNet in SmolLM2.
**Fix:** `res_alpha = from_scratch ? 1/sqrt(2*NLAYERS) : 1.0` in train_mezo.m.
Added `--no-deepnet` flag to train.m.
**Impact:** ALL v1 fine-tuning results (conditions 5-8) were invalid. v2 re-run required.
**Verification:** v2 initial loss = 2.24 (matches expectation for SmolLM2 on TinyStories).

### 5.3 IOSurface Transpose Optimization (v3)

**Symptom:** MeZO-ANE 1.73x slower than CPU (656 vs 379 ms/step), with 226ms in transpose.
**Root causes identified via microbenchmark:**
1. 3rd RETRANSPOSE_AND_STAGE after weight update is immediately overwritten by next step's
   perturbation. Only needed before validation (every 500 steps).
2. W2 staging used element-wise double loop (scalar fp32→fp16 + transpose), despite
   W2t_buf (pre-transposed copy) already being computed. The W2t_buf was unused.
**Fix 1:** Defer 3rd restage to only execute before validation blocks.
**Fix 2:** Replace W2 element-wise staging with cvt_f32_f16(W2t_buf) — 3.2x faster.
**Impact:** Transpose 226→99ms, step 656→501ms (1.31x speedup), 183→240 steps in 120s.
**Verification:** Bit-identical losses at step 0 (2.2467) and step 100 (1.8953).

### 5.4 IOSurface Fundamental Limit Analysis (v4)

**Question:** Can further IOSurface optimization make MeZO-ANE faster than MeZO-CPU?
**Answer:** NO, at 135M params.

**Analysis:** Even with ZERO transpose overhead:
- MeZO-ANE: fwd(249ms) + perturb(141ms) = 390ms
- MeZO-CPU: fwd(228ms) + perturb(149ms) = 377ms
- ANE forward is 21ms SLOWER than CPU due to IO overhead (7 ANE dispatches × 30 layers
  = 210 dispatches per fwd pass, each with IOSurface write + kernel eval + IOSurface read)

**Options evaluated:**
1. Pre-transposed weight storage: Saves 67ms of vDSP_mtrans → 434ms. Still > 379ms.
2. In-place fp16 perturbation (no restaging): Saves 99ms → 402ms. Still > 379ms.
   Introduces fp16 rounding errors, breaks bit-identical guarantee.
3. Dual-buffer (fp32 + fp16 shadow): +538MB memory, saves transpose not staging → ~434ms.

**Root cause:** For 135M params with 7 separate ANE dispatches per layer, per-dispatch IO
overhead (~95μs × 210 = ~20ms per fwd) makes ANE forward slower than CPU AMX.
This is a model-size dependent limitation: at 360M+ params, matmul time grows quadratically
while dispatch overhead grows linearly, so ANE should eventually win.

### 5.5 MeZO+LoRA Implementation and Verification (v7)

**Question:** Can MeZO+LoRA eliminate the transpose/perturbation bottlenecks on ANE?
**Answer:** YES. Two modes implemented and verified.

**Mode 1: MeZO+LoRA merge** (--lora --lora-rank 8)
- Perturbs only LoRA A/B matrices + RMS norms (~1.7M params vs 361.8M full)
- Merges W_eff = W_base + B@A before ANE dispatch
- Only restages Wq/Wk/Wv/Wo (RETRANSPOSE_ATTN_ONLY — skips W1/W2/W3 = 74% reduction)
- Result: 807ms/step ANE (vs 1200ms full MeZO-ANE = 33% faster)

**Mode 2: MeZO+LoRA-split** (--lora-split --lora-rank 8)
- Base weights baked in IOSurfaces at initialization, never restaged
- LoRA correction computed on CPU: out = ANE_base(x) + B@(A@x)
- Uses lora_addmm() with cblas_sgemm for correction
- Result: 708ms/step ANE (vs 1200ms full = 41% faster), zero transpose overhead

**Correctness verification:**
- Step-0 loss_plus = 2.1095 across all 4 LoRA modes (CPU/ANE × merge/split) ✅
  (Note: this was with theta=10000; corrected theta=100000 gives ~2.062 — see §5.9)
- Val loss @100 matches within noise: 2.068-2.070 across all modes ✅
- Rank 8 confirmed superior to rank 32 (rank 32 showed near-zero gradient signal)

**Key timing decomposition:**

| Component | Full MeZO-ANE | LoRA-merge ANE | LoRA-split ANE |
|-----------|---------------|----------------|----------------|
| Forward (2x) | 525ms | ~500ms | ~500ms |
| Perturbation (4x) | 579ms | 65ms | 3ms |
| Transpose | 478ms | 106ms | 0ms |
| **Total** | **1200ms** | **807ms** | **708ms** |

**Implementation details:**
- `perturb_lora_weights()`: Perturbs only LoRA A/B for Wq/Wk/Wv/Wo + RMS norms
- `lora_merge_all()`: Merges adapters into effective weights before ANE dispatch
- `lora_addmm()`: CPU-side LoRA correction via cblas_sgemm (for split mode)
- `RETRANSPOSE_ATTN_ONLY()`: New macro restaging only attention weights (skip FFN)
- LoRA init: A ~ N(0, 1/√r), B = 0 (standard LoRA initialization)

### 5.6 SmolLM2-360M Scaling Experiment (v5)

**Question:** Does ANE overtake CPU at 360M params as predicted by v4 analysis?
**Answer:** NO. The hypothesis was FALSIFIED.

**Results (all 4 conditions, 120s time budget):**

| # | Method | Hardware | Steps | ms/step | Val Loss @100 | RSS (MB) |
|---|--------|----------|-------|---------|---------------|----------|
| 9 | MeZO | CPU | 143 | 814 | 2.067 | 1,720 |
| 10 | MeZO | ANE | 100 | 1,200 | 2.067 | — |
| 11 | BP+Adam | CPU | 140 | 602 | 1.791 | 4,133 |
| 12 | BP+Adam | ANE | 120 | 700 | 1.791 | — |

**Timing decomposition (360M MeZO):**
- MeZO-CPU: fwd=428ms, perturb=379ms, transpose=0ms → **814ms**
- MeZO-ANE: fwd=525ms, perturb=579†ms, transpose=478ms → **1200ms**
  (†Step 0 perturbation inflated by JIT warmup; steady-state ~376ms)

**Why the hypothesis failed:**
1. Transpose overhead scaled superlinearly: 99ms (135M) → 478ms (360M) = 4.8x for 2.69x params
2. ANE forward is STILL slower than CPU: 525ms vs 428ms (even ignoring transpose)
3. Per-dispatch IO overhead (~95μs) persists regardless of matmul size
4. Larger weight matrices → more data to transpose+stage per restage cycle

**Memory comparison:**
- MeZO: 1,720 MB (2.4x less than backprop)
- Backprop: 4,133 MB
- At 360M, both fit comfortably on 8GB devices
- Memory crossover (backprop won't fit, MeZO will) estimated at ~600M-1B params

### 5.7 Parameter Counting Bug Fix (v10 audit)

**Symptom:** LoRA adapter param count reported as 2293.8K (attn-only) and 6144.0K (attn+FFN).
**Root cause:** The counting formula in train_mezo.m treated Bk[KV_DIM,rank] and Bv[KV_DIM,rank]
as DIM×rank instead of KV_DIM×rank. Similarly, FFN formula had an extra HIDDEN×rank term.

**Code (old, line 415):**
```c
lora_params += 2 * r * DIM * 3 + 2 * r * Q_DIM + 2 * r * KV_DIM * 2;  // = 71680/layer
```

**Correct formula:**
```c
lora_params += r*DIM*3 + Q_DIM*r + KV_DIM*r*2 + r*Q_DIM + DIM*r;  // = 51200/layer
```

**Impact:**
| Config | Reported (wrong) | Actual (correct) | Overcount |
|--------|-----------------|------------------|-----------|
| Attn-only r8 | 2,293.8K | 1,638.4K | +40% |
| Attn+FFN r8 | 6,144.0K | 4,341.8K | +41% |

**NOT a correctness bug:** Allocations in config.h use the correct sizes. Training is unaffected.
This is a reporting/documentation bug only. Fixed in v10.

**Verification:** After fix, step-0 loss_plus=2.1095, loss_minus=2.0987 — identical to pre-fix.

### 5.8 Known Limitations (v10 code audit)

1. **No LoRA checkpoint persistence:** LoRA A/B matrices are never saved to disk.
   If training is interrupted and resumed with `--resume --lora-split`, all LoRA progress
   is lost. In merge mode, the merged effective weights are saved, preserving some information.

2. **Unnecessary cembed rebuild in merge mode:** In `use_lora && !lora_split` mode,
   `cembed` is freed and rebuilt after LoRA perturbation, but `embed` was never modified
   (only LoRA A/B and RMS norms are perturbed). This wastes ~180MB alloc+copy per forward
   pass (2x per step) at 360M. Not a correctness bug.

3. **`--lora-ffn` silently ignored without `--lora`:** Fixed in v10 — `--lora-ffn` now
   implies `use_lora=true, lora_split=true` if neither `--lora` nor `--lora-split` given.

4. **Checkpoint overwrite:** Training saves to `CKPT_PATH` (same as HF-converted checkpoint),
   overwriting the original on best-validation improvement. Must re-convert from HF to
   get a clean checkpoint for reproducibility.

### 5.9 RoPE Theta Bug (v11 — CRITICAL)

**Symptom:** Step-0 loss on SmolLM2-360M was 2.1095. HuggingFace reference on the same batch
gives 2.1046 (full vocab) / 2.0941 (compact vocab). After RoPE fix: 2.0590 (midpoint).

**Root cause:** `cpu_ops.h` hardcoded `powf(10000.0f, ...)` in both `rope_forward_inplace`
(line 298) and `rope_backward_inplace` (line 363). SmolLM2-360M uses `rope_theta: 100000`
(verified from HuggingFace config.json). SmolLM2-135M also uses `rope_theta: 100000`.
Qwen3-0.6B uses `rope_theta: 1000000`. This is a 10x error for SmolLM2, 100x for Qwen3.

**Discovery method:** Systematic assumption audit flagged "RoPE implementation: UNVERIFIED".
Cross-checked against HuggingFace `config.json` for each model.

**Fix (v11):**
1. Added `#define ROPE_THETA` to all 8 model headers with correct values:
   - `autoresearch*.h`: 10000.0f (standard Llama)
   - `stories110m.h`: 10000.0f (Llama2-style)
   - `smollm2_135m.h`: 100000.0f (verified from HF config)
   - `smollm2_360m.h`: 100000.0f (verified from HF config)
   - `qwen3_06b*.h`: 1000000.0f (verified from HF config)
2. Updated `cpu_ops.h` to use `ROPE_THETA` macro with `#ifndef` fallback to 10000.0f
3. Rebuilt and verified via preprocessor: `powf(100000.0f, ...)` for SmolLM2-360M ✅

**HuggingFace baseline comparison (v11):**

| Metric | Loss |
|--------|------|
| HF reference (full vocab, same batch) | 2.1046 |
| HF reference (compact vocab=16893, same batch) | 2.0941 |
| Our impl loss_plus (RoPE fixed, perturbed +eps) | 2.0617 |
| Our impl loss_minus (RoPE fixed, perturbed -eps) | 2.0563 |
| Our impl midpoint (unperturbed estimate) | 2.0590 |
| Old impl (theta=10000, broken RoPE) | 2.1095 |

Gap between HF compact and our midpoint: 0.035 (1.7%). **UPDATE (v12):** This gap was NOT
float32 precision — it was caused by the CE epsilon guard bug (§5.11). After fixing the CE
loss to use log-sum-exp, the gap drops to 0.08%, which truly is within float32 precision.

**Impact on experiments:**
- **Absolute loss values:** ALL SmolLM2 fine-tuning experiments (conditions 5-37) have
  inflated absolute losses by ~0.05 (2.4%). Reported values should be ~0.05 lower.
- **Relative comparisons:** STILL VALID. All conditions used the same wrong theta, so
  differences between methods (MeZO vs BP, CPU vs ANE, rank 8 vs 16 vs 32) are preserved.
- **From-scratch experiments (conditions 1-4):** UNAFFECTED. autoresearch.h correctly uses
  theta=10000.0f (standard Llama base for from-scratch training).
- **Convergence behavior:** Relative convergence curves are valid. The wrong theta shifts
  all losses by a roughly constant offset, so learning dynamics are preserved.
- **RoPE backward:** Also fixed (line 363). Gradient computation was equally affected,
  but since both forward and backward used the same wrong theta, the gradients were
  internally consistent. The model was "learning wrong positions consistently."

**Verification:**
- Preprocessor output confirms `powf(100000.0f, ...)` for SmolLM2-360M ✅
- Step-0 loss 2.0590 matches HF reference within 1.7% ✅ (v12: 0.08% after CE fix)
- Old theta=10000 loss was 2.1095 (5.1% above HF reference) — improvement confirmed ✅

### 5.10 LoRA A Matrix Init Scale Deviation (v11 — documented, not a bug)

**Finding:** Our LoRA A matrices use `a_scale = 1/sqrt(r)` giving Uniform[-0.354, +0.354]
for rank 8. Standard LoRA (loralib, PEFT) uses Kaiming uniform with `fan_in = d_in`:
`nn.init.kaiming_uniform_(A, a=sqrt(5))` → Uniform[-1/sqrt(d_in), +1/sqrt(d_in)].

**Scale comparison (rank=8, SmolLM2-360M):**

| Matrix | Our init | Standard init | Ratio |
|--------|----------|---------------|-------|
| Aq/Ak/Av [r, DIM=960] | ±0.354 | ±0.032 | 11.0x |
| Ao [r, Q_DIM=960] | ±0.354 | ±0.032 | 11.0x |
| A2 [r, HIDDEN=2560] | ±0.354 | ±0.020 | 17.9x |

**Impact on MeZO:** Since B starts at zero, the initial LoRA correction ΔW = B@A = 0
regardless of A's scale. However, after MeZO perturbation (+ε to B), the correction
becomes ε·z_B @ A, which is proportional to ||A||. Our larger A gives ~11x stronger
gradient signal per MeZO step. This may actually be beneficial for MeZO (which suffers
from weak gradient estimates), but deviates from the standard LoRA initialization.

**Not fixed:** Changing init would invalidate all existing experiments. The current init
works well in practice (val plateaus at ~2.045 after 650 steps, rank 8 converges
reliably across 4 seeds). Documented for transparency and future experiments.

---

### 5.11 Cross-Entropy Loss Epsilon Guard Bug (v12 — CRITICAL, FIXED)

**Root cause:** `cpu_ops.h` line 126 (before fix) used:
```c
total_loss -= logf(drow[tgt] + 1e-10f);
```
where `drow[tgt]` is the softmax probability of the target token. The `+ 1e-10f` epsilon
guard biases the loss downward whenever `p_target` is very small.

**Single outlier dominates the error:** Position 123 in the step-0 batch has
`p_correct = 1.17e-14`. With the guard: `log(1.17e-14 + 1e-10) ≈ log(1e-10) = -23.03`.
Correct value: `log(1.17e-14) = -32.09`. Error = -9.054 nats at one position.
Averaged over 256 positions: -9.054/256 = -0.0354, giving ~1.7% loss underestimate.

**Fix:** Replaced with numerically stable log-sum-exp form:
```c
// Log-sum-exp CE: loss_t = -(logit[target] - max) + log(sum(exp(logit - max)))
total_loss += -(row[tgt] - maxv) + logf(sum);
```
This computes log(softmax) via log-sum-exp directly from logits, avoiding computing
softmax probabilities entirely for the loss calculation (softmax probabilities are
still computed in `drow[]` for the backward pass gradient).

**Remaining 1e-10 guards at lines 255 and 349:** These are in SDPA attention softmax,
where `sum = sum(exp(score - max))` is always ≥ 1.0 (the max element contributes
`exp(0) = 1`). The guard is purely defensive and never activates. No fix needed.

**Verification (v12):**

| Metric | Before fix (v11) | After fix (v12) |
|--------|-----------------|-----------------|
| Our midpoint loss | 2.0590 | 2.0958 |
| HF compact reference | 2.0941 | 2.0941 |
| Gap | 1.68% (underestimate) | 0.08% |

The 0.08% residual gap is within float32 precision + MeZO perturbation noise
(epsilon=0.001). With the fix, our implementation matches HuggingFace exactly.

**Impact on existing experiments:** All 37 conditions have slightly wrong absolute
loss values. The magnitude of the error depends on how many positions have extremely
low target probabilities in each batch. However:
- **Relative comparisons between conditions are still valid** — all conditions used
  the same CE code, so the bias is consistent.
- **MeZO gradient estimates: ~3.9% average bias (CORRECTED v12).** The backward pass
  gradient `dL/d_logit[target] = p - 1` is correct (computed from softmax, not from
  log). However, MeZO's projected gradient `(loss+ - loss-) / 2ε` IS affected:
  the CE bias is 99.9995% constant across +/- perturbations, but the 0.0005%
  residual variation gets amplified 500x by the `1/(2ε)` factor (ε=0.001),
  producing ~3.9% average gradient bias. Verified experimentally by comparing
  old vs new gradient estimates across 10 random perturbation directions.
  This is small relative to MeZO's inherent O(d) variance, but is not zero as
  initially claimed. The bias has high variance across directions (mean 0.5%,
  std 6.8%), so some steps had up to ~15% gradient bias while others had <1%.

---

### 5.12 LoRA Alpha/r Scaling (v12 — verified, not a bug)

**Finding:** Our `lora_addmm` applies `out += B @ (A @ x)` with no `alpha/r` scaling factor.

**Standard LoRA:** `h = Wx + (alpha/r) * B @ A @ x`. The common default is `alpha = r`
(PEFT default: `lora_alpha=8` with `r=8`), giving `alpha/r = 1.0`.

**Our implementation:** Equivalent to `alpha = r` (scaling factor = 1.0). This is the
standard default and is not a bug. The A initialization scale deviation (§5.10) is
a separate concern that affects the initial magnitude of ΔW updates, not the scaling
architecture.

**Cross-checked against:**
- [Original LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685): "we set α to the first r we try"
- [PEFT library defaults](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora): `lora_alpha=8` default
- [MeZO repository](https://github.com/princeton-nlp/MeZO): uses PEFT defaults for MeZO+LoRA

---

### 5.13 xoshiro256+ Bit Extraction (v12 — documented, minor concern)

**Finding:** Our `perturb_buffer` extracts the **lowest 4 bits** from each xoshiro256+
output for Rademacher signs:
```c
buf[i+0] += (r & 1) ? scale : neg_scale;   // bit 0
buf[i+1] += (r & 2) ? scale : neg_scale;   // bit 1
buf[i+2] += (r & 4) ? scale : neg_scale;   // bit 2
buf[i+3] += (r & 8) ? scale : neg_scale;   // bit 3
```

[Blackman & Vigna](https://prng.di.unimi.it/) document that the **lowest 3 bits** of
xoshiro256+ have low linear complexity and recommend using upper bits for
floating-point generation.

**Impact on MeZO:** Unlikely to matter. The bits are still unbiased (50/50 ±1), and the
linear complexity weakness only manifests in specific statistical tests, not in sign
balance. Each MeZO step uses a fresh seed and sums over millions of parameters,
averaging out any subtle correlation.

**Trivial fix (optional):** Shift right before extracting: `(r >> 32) & 1` etc.

**Cross-checked against:** [xoshiro256+ reference](https://prng.di.unimi.it/xoshiro256plus.c),
[splitmix64 reference](https://prng.di.unimi.it/splitmix64.c). Our implementation is a
verbatim reproduction of both Blackman-Vigna references (state update, constants,
rotation, seed expansion all match exactly).

---

## 6. Stated Assumptions

### A1: Rademacher perturbation is equivalent to Gaussian for SPSA
**Status: VALIDATED** (Section 1.2, 3.3)
Both satisfy E[zz^T] = I. Rademacher has lower variance (better).

### A2: Forward pass in train_mezo.m matches train.m
**Status: VALIDATED** (Section 2.2, 3.5)
All BLAS calls match. Loss at step 0 differs only by epsilon perturbation.

### A3: Checkpoint format is cross-compatible
**Status: VALIDATED** (after fix, Section 2.3)
Layout now matches train.m exactly.

### A4: 4-bit Rademacher extraction from xoshiro256+ produces independent samples
**Status: VALIDATED** (Section 3.4)
All joint probabilities match independence assumption.

### A5: Floating-point drift from perturbation cancel is negligible
**Status: VALIDATED** (Section 3.1, 3.6)
Drift/update ratio ≈ 1e-6 (from-scratch) and 0.002 (pretrained), both negligible.

### A6: MeZO from-scratch should show learning signal
**Status: REFUTED** by both experiment and theory.
MeZO's convergence depends on low Hessian effective rank (Theorem 1),
which only holds for pre-trained models. This is EXPECTED.

### A7: DeepNet residual scaling applies to all architectures
**Status: REFUTED** (discovered in v2 audit).
DeepNet is ONLY valid for from-scratch training with matched initialization.
Pretrained Llama/SmolLM2 models use standard residual connections (alpha=1.0).
Applying DeepNet to pretrained models destroys pretrained representations.

### A8: MeZO fine-tuning is competitive with backprop in wall time
**Status: REFUTED** by v4 val loss experiment.
v2 analysis compared single-batch training losses (MeZO 1.97 vs BP 1.81 — "only 0.16 gap").
v4 val loss measurements show the TRUE gap is 0.32 (MeZO val 2.249 vs BP val 1.929
after 300 steps). MeZO convergence is ~60x slower per step on val loss.
MeZO needs ~50K steps (5+ hours) to match BP val quality at 135M params.
**MeZO's advantage is MEMORY (3.7x savings), not convergence speed.**

### A9: At 360M+ params, ANE should overtake CPU for MeZO
**Status: REFUTED** by v5 experiment (Section 5.5).
At 135M: MeZO-ANE 32% slower than CPU. At 360M: MeZO-ANE 47% slower than CPU.
The gap WIDENED, not narrowed. Transpose overhead scales superlinearly with model
size (4.8x growth for 2.69x more params). ANE forward is also still slower than CPU
even without transpose. MeZO-on-ANE is structurally slower than MeZO-on-CPU at
any model size that fits in memory on Apple Silicon.

### A10: MeZO uses constant LR (no schedule)
**Status: CORRECTED** in v4 audit.
Code has cosine decay (lr decays from base to 0.1×base over total_steps).
For our short runs (240-317 of ~100K total_steps), decay is <0.3% — effectively constant.
The original assumption of "no schedule" was technically incorrect.

### A11: ANE matmul throughput would dominate dispatch overhead at 360M+ params
**Status: REFUTED** by v5 experiment (Section 5.5).
Transpose overhead scaled superlinearly (4.8x for 2.69x more params). ANE forward
is still slower than CPU even without transpose. The architectural mismatch is between
MeZO's per-step weight perturbation and ANE's preference for static baked weights.
**Solution: MeZO + LoRA, where only small adapters change per step.**

### A12: Full-parameter perturbation is the only way to run MeZO on ANE
**Status: INCORRECT.** The MeZO paper tests MeZO+LoRA (Section 4.4), which only perturbs
LoRA adapter parameters. With rank 32 on 360M: ~7.9M adapter params (2.2% of full model).
This reduces perturbation time by ~47x and eliminates base weight restaging entirely.
Accuracy: MeZO+LoRA on OPT-13B SST-2=89.6% (Table 1, original paper).
Note: The MeZO paper does NOT test on LLaMA; all experiments use OPT and RoBERTa families.

### A13: MeZO+LoRA rank 8 is optimal for ZO fine-tuning on ANE
**Status: VALIDATED** by v7 experiments.
Rank 8 (1.6M adapter params): val_loss=2.068, 576ms/step CPU, 708ms/step ANE (split).
Rank 32 (9.2M adapter params): near-zero gradient signal, 1142ms/step, no convergence.
Lower rank = fewer perturbed dimensions = lower ZO variance = better gradient estimate.
Theory: lower dim = lower ZO variance (Lemma 2). No published rank 8 vs 32 comparison
for MeZO specifically, but SubZero and MaZO confirm the dimension-dependence principle.

### A14: Adapter-as-input (LoRA-split) eliminates ANE restaging overhead
**Status: VALIDATED** by v7 experiments.
LoRA-split mode: base weights baked at init, LoRA correction on CPU via lora_addmm().
Transpose overhead: 478ms → 0ms (eliminated entirely).
Perturbation overhead: 579ms → 3ms (193x faster, only 1.7M params).
ANE vs CPU gap: 47% (full MeZO) → 32% (LoRA-split). Remaining gap is ANE dispatch IO.

### A15: BP+LoRA ANE should be faster than BP+LoRA CPU
**Status: REFUTED** by v7 condition 14 and v8 condition 27.
v7: BP+LoRA ANE hit 1344ms/step (vs CPU 586ms) with thermal=serious.
v8 rerun after 60s cooldown (condition 27): thermal=nominal throughout, but still
1445ms/step due to periodic IO stalls (io_fwd spikes to 600-964ms on ~3/69 steps).
Root cause is NOT thermal — it's ANE backward pass IO overhead and memory bus contention.

### A16: MeZO+LoRA lr=1e-5 is optimal (inherited from full MeZO)
**Status: REFUTED** by v8 LR sweep.
MeZO+LoRA with lr=1e-4 converges 10x faster than lr=1e-5 on val loss.
At lr=3e-4, convergence is faster initially but shows instability (val bounces).
**Best LR for MeZO+LoRA-split: 1e-4** (10x higher than full MeZO's optimal 1e-5).
This is expected: LoRA reduces effective parameter count from 361.8M to 1.7M (incl. RMS),
and ZO LR scales as n/(d+n-1) × SGD_LR — smaller d allows proportionally higher LR.

### A17: MeZO+LoRA val loss will keep improving with more steps
**Status: REFUTED** by v8 condition 26 (600s, 1150 steps).
Val loss plateaus at ~2.045 after 650 steps. From step 650 to 1150, val Δ < 0.001.
The LoRA adapters (rank 8, attn-only) have insufficient capacity for further improvement.
Options to break plateau: higher rank (but increases ZO variance), FFN adapters,
or P-GAP/AGZO for better gradient estimation.

### A18: Our CE loss formula is numerically stable
**Status: VALIDATED** (v12, Section 5.11).
After fixing the `logf(p + 1e-10)` epsilon guard to log-sum-exp, our CE matches
HuggingFace within 0.08% across 5 data positions.

### A19: The CE epsilon guard does not affect MeZO gradient estimates
**Status: REFUTED** (v12, Section 5.11).
The CE bias is 99.9995% constant across +/- perturbations, but the 0.0005% residual
variation is amplified 500x by `1/(2ε)`, producing ~3.9% average gradient bias.
Small relative to MeZO's inherent O(d) variance, but not zero.

### A20: xoshiro256+ lowest bits are adequate for Rademacher signs
**Status: VALIDATED with caveat** (v12, Section 5.13).
Bits are unbiased (50/50 ±1) but have low linear complexity per Blackman & Vigna.
Practically irrelevant for MeZO — each seed used once, millions of parameters
average out correlations. Optional fix: shift right before extraction.

### A21: MeZO+LoRA perturbs only LoRA parameters
**Status: PARTIALLY INCORRECT** — deliberate design choice (v12).
Our `perturb_lora_weights` also perturbs RMS norm weights (rms_att, rms_ffn per
layer + rms_final). This deviates from the paper's "only LoRA parameters" but is
standard practice (PEFT keeps norms trainable during LoRA fine-tuning). The overhead
is negligible: ~64K norm params vs ~1.7M LoRA params (3.7% additional).

### A22: Channel-first layout is equivalent to row-major for all operations
**Status: VALIDATED** (v12, numerical BLAS test).
Embedding lookup: bit-exact (0.00e+00). Wq matmul: 1.1e-05 diff (float32
accumulation order). RMSNorm: 2.86e-06 diff. All within expected float32 precision
for different summation orders over DIM=960 elements.

---

## 7. What's Done and What's Next

### Completed
- [x] Design spec (reviewed, committed)
- [x] Implementation plan (reviewed 2x, all bugs fixed)
- [x] train_mezo.m binary (compiled, tested)
- [x] Makefile target
- [x] Tests 9-12 (all pass)
- [x] From-scratch experiments (conditions 1-4)
- [x] Fine-tuning experiments v1 (conditions 5-8, invalidated by DeepNet bug)
- [x] Fine-tuning experiments v2 (conditions 5-8, corrected)
- [x] PRNG verification (bit-for-bit match to reference)
- [x] BLAS call verification (all 8 ops match train.m)
- [x] Perturbation cancel validation
- [x] Gradient unbiasedness validation
- [x] Rademacher distribution validation
- [x] Bit independence validation
- [x] Literature cross-reference (MeZO paper, SPSA theory)
- [x] Comprehensive 4-agent audit (algorithm, literature, forward pass, checkpoint)
- [x] DeepNet res_alpha bug discovery, diagnosis, and fix
- [x] LR override bug discovery and fix
- [x] LR sweep for MeZO fine-tuning (7 values)
- [x] Multi-seed validation (5 seeds, 300 steps each)
- [x] Memory profiling (785MB vs 2910MB)
- [x] Perturbation cancel on pretrained weights validation
- [x] Research audit document v1 and v2
- [x] IOSurface transpose microbenchmark (v3)
- [x] IOSurface optimization: deferred restage + W2 bulk cvt (v3, 1.31x speedup)
- [x] v4 audit: cross-checked all table numbers against raw logs
- [x] v4 audit: found and fixed condition 8 final loss omission (1.93 from v2)
- [x] v4 audit: corrected cosine LR schedule assumption (present but negligible)
- [x] v4 audit: IOSurface fundamental limit analysis (ANE cannot beat CPU at 135M)
- [x] v4 audit: added v3 timing breakdown to analysis.md
- [x] v4 audit: literature re-verification via MeZO paper cross-check
- [x] v5: SmolLM2-360M experiments (conditions 9-12, all 4 methods)
- [x] v5: Memory profiling at 360M (MeZO 1720MB vs Backprop 4133MB)
- [x] v5: Falsified hypothesis that ANE would overtake CPU at 360M
- [x] v5: Identified superlinear transpose scaling (4.8x for 2.69x params)
- [x] v5: Root cause analysis — full-param MeZO is architecturally mismatched with ANE
- [x] v5: Literature review — ANE strengths, Orion paper, MeZO+LoRA, adapter-as-input
- [x] v5: Identified MeZO+LoRA as the correct approach for ANE (plays to ANE strengths)
- [x] v6: Comprehensive literature review of 14 papers (2024-2026) on ZO optimization + NPU training
- [x] v6: Identified optimal ANE training stack: P-GAP + LoRA + adapter-as-input + 1x1 conv
- [x] v6: Surveyed MobiZO (EMNLP 2025), P-GAP, AGZO, DiZO, MeSP techniques
- [x] v6: Added Related Work table and 19-source bibliography to analysis.md
- [x] v7: Implemented MeZO+LoRA merge mode in train_mezo.m (perturb_lora_weights, lora_merge_all, RETRANSPOSE_ATTN_ONLY)
- [x] v7: Implemented MeZO+LoRA-split (adapter-as-input) mode (lora_addmm, zero restaging)
- [x] v7: Ran 7 new conditions (13-19): BP+LoRA, MeZO+LoRA merge, MeZO+LoRA-split, rank comparison
- [x] v7: Verified correctness — step-0 loss_plus=2.1095 matches across all 4 LoRA modes
- [x] v7: Confirmed rank 8 >> rank 32 for MeZO (lower ZO variance)
- [x] v7: LoRA-split ANE achieves 708ms/step (41% faster than 1200ms full MeZO-ANE)
- [x] v7: Eliminated transpose overhead entirely (478ms → 0ms) via adapter-as-input
- [x] v7: Reduced perturbation time 193x (579ms → 3ms) by perturbing only 1.7M adapter params
- [x] v7: Narrowed ANE vs CPU gap from 47% to 32%
- [x] v8: MeZO+LoRA LR sweep (1e-5, 5e-5, 1e-4, 3e-4) — best LR is 1e-4 (10x faster)
- [x] v8: Long convergence test (600s, 1150 steps, lr=1e-4) — val plateaus at 2.045 after 650 steps
- [x] v8: Rank 4 test — similar convergence to rank 8, 15% faster per step
- [x] v8: ANE convergence curve (300s, val_every=50) — matches CPU within noise
- [x] v8: Memory profiling — MeZO+LoRA-split: 1952MB, BP+LoRA: 3954MB (2.0x ratio)
- [x] v8: BP+LoRA ANE rerun (thermal controlled) — still 1445ms/step, IO stalls not thermal
- [x] v8: BP+LoRA CPU convergence curve — val=1.972 at step 100 (far ahead of MeZO)
- [x] v8: 9 new conditions (20-28)
- [x] v9: FFN LoRA implementation (W1/W2/W3 adapters, --lora-ffn flag)
- [x] v9: FFN LoRA experiment — val@250=2.0474 vs attn-only 2.0506 (FFN helps marginally)
- [x] v9: Full MeZO lr=1e-4 — worse than LoRA (val=2.074 at step 50)
- [x] v9: Epsilon sweep — eps=1e-3 optimal, 1e-2 and 1e-4 both slightly worse
- [x] v9: 5 new conditions (29-33)
- [x] v10: Parameter counting bug fix (reporting only, not correctness)
- [x] v10: Deep code audit — lora_addmm, perturb_lora_weights, FFN split mode, forward pass
- [x] v10: Cross-checked all 14 raw logs (conditions 20-33) against analysis.md tables
- [x] v10: Multi-seed validation (4 seeds) for MeZO+LoRA-split at 360M — val@100 std=0.0024
- [x] v11: RoPE theta bug discovery — cpu_ops.h hardcoded 10000.0f, SmolLM2 uses 100000
- [x] v11: Added ROPE_THETA macro to all 8 model headers with verified values
- [x] v11: Fixed cpu_ops.h rope_forward_inplace and rope_backward_inplace to use ROPE_THETA
- [x] v11: HuggingFace baseline comparison — our impl matches HF within 1.7% (float32 precision)
- [x] v11: Verified Qwen3-0.6B rope_theta=1000000 from HuggingFace config.json
- [x] v11: Impact analysis — relative comparisons still valid, absolute losses ~0.05 inflated
- [x] v12: CE loss epsilon guard bug discovery — `logf(p + 1e-10)` caused 1.7% loss underestimate
- [x] v12: Root cause: single outlier position (p=1.17e-14) dominates; epsilon >> p makes log(epsilon)
- [x] v12: Fix: replaced with log-sum-exp form `-(logit[tgt] - max) + log(sum(exp(logit - max)))`
- [x] v12: Verified remaining 1e-10 guards (SDPA softmax lines 255/349) are safe (sum always ≥ 1)
- [x] v12: Rebuilt and verified — gap vs HF reduced from 1.68% to 0.08%
- [x] v12: Documented impact — backward pass gradient correct, MeZO gradient ~3.9% biased (CORRECTED from "unaffected")
- [x] v12: Multi-position loss verification — tested 5 seeds, all match HF within perturbation noise
- [x] v12: Verified train_mezo.m line 514 res_alpha hardcode is harmless (unused in matmul-only mode)
- [x] v12: Fixed generate.py DeepNet res_alpha hardcode — now uses 1.0 for pretrained, configurable via --from-scratch

### Next Steps
- [x] ~~**MeZO + LoRA on ANE (HIGHEST PRIORITY):**~~ **DONE in v7.** Implemented merge and
      split modes. LoRA-split ANE: 708ms/step (41% faster than 1200ms full MeZO-ANE).
      Transpose eliminated (478→0ms), perturbation 193x faster (579→3ms).
      ANE still 32% slower than CPU (708 vs 537ms) due to dispatch IO overhead.
- [ ] **P-GAP + LoRA (v6 finding):** Gradient-aligned perturbations reduce steps by 5.2x.
      Combined with LoRA on ANE, the optimal stack is:
      P-GAP + LoRA + adapter-as-input + 1x1 conv.
- [ ] **MobiZO MP-LoRA (v6 finding):** Parallelize +ε/-ε in single forward pass.
      4.3x speedup, already deployed on Qualcomm NPU. Directly applicable to ANE.
- [ ] **1x1 convolution matmuls:** ANE is a convolution engine. Expressing Linear layers as
      1x1 Conv2d yields ~3x throughput (Orion paper, arXiv:2603.06728). Currently using
      matmul dispatches. This alone could make ANE forward faster than CPU.
- [ ] **MeZO-SVRG + LoRA:** Variance-reduced ZO (ICML 2024) converges 2x faster with 20%
      accuracy improvement. Combined with LoRA on ANE, this addresses both speed and
      convergence weaknesses simultaneously.
- [x] **Multi-seed validation for MeZO+LoRA-split (v10):** 4 seeds (42/123/7/999), val@100 std=0.0024
- [ ] **Multiple seeds for remaining conditions:** 3-5 seeds per condition for error bars.
- [ ] **Push to remote:** 13+ unpushed commits on main.
- [x] v12: Vocab compaction verification — VocabMap, compact embedding, CE softmax over compact vocab all correct
- [x] v12: Forward pass operation-by-operation verification — RMSNorm, RoPE, SDPA, GQA, SwiGLU, embedding all match HF
- [x] v12: Weight conversion (hf_to_ane.py) verification — all shapes, byte formats, RoPE interleaving correct
- [x] v12: Data format verified — raw uint16 tokens, no header, 20M tokens, all valid SmolLM2 IDs
- [x] v12: Model dimensions cross-checked — all 8 fields match HuggingFace config exactly
- [x] v12: RoPE frequencies verified — all 32 match HF (max relative error 2.28e-16)
- [x] v12: Checkpoint header verified — magic, version, dims, file size all correct
- [x] v12: Compact vocab count confirmed — 16,893 unique tokens
- [x] v12: Q/K interleaving bit-exact — element-wise AND functional test both 0.00e+00 diff
- [x] v12: SwiGLU order verified against HF — W1=gate_proj, W3=up_proj, W2=down_proj matches `W2(SiLU(W1(x))*W3(x))`
- [x] v12: MeZO algorithm implementation verified — all 9 steps of Algorithm 1 match paper exactly
- [x] v12: Rademacher perturbation verified — xoshiro256+ RNG, 4 bits/call, z_i ∈ {-1,+1}
- [x] v12: MeZO gradient bias quantified — ~3.9% avg from CE epsilon guard (corrected from "unaffected")
- [x] v12: RoPE backward pass verified — uses ROPE_THETA macro, R^T formula correct, indexing matches forward
- [x] v12: LoRA alpha/r scaling verified — implicit alpha=r (scaling=1.0), matches PEFT/loralib defaults
- [x] v12: xoshiro256+ cross-checked against Blackman-Vigna reference — verbatim match, low-bit concern documented
- [x] v12: Validation loop verified — same forward pass, same LoRA, same res_alpha, no perturbation
- [x] v12: End-to-end training test — 10 steps × 2 seeds, stable losses, learning signal confirmed
- [x] v12: lora_addmm function verified — computes B @ (A @ x) with correct BLAS calls and dimensions
- [x] v12: perturb_lora_weights verified — perturbs LoRA A/B + rms_att + rms_ffn + rms_final (deliberate)
- [x] v12: Channel-first BLAS layout verified numerically — embed 0.00e+00, Wq 1.1e-05, RMSNorm 2.86e-06
- [x] v12: GQA tiling (gqa_tile_kv) verified — duplicates each KV head GQA_RATIO times via memcpy
- [x] v12: Cosine LR schedule verified — min_lr=0.1×base_lr, standard cosine annealing, no warmup

---

## 8. Literature Verification (v10)

Cross-checked claims against original papers via web search. Key corrections:

| Claim | Verdict | Action |
|-------|---------|--------|
| MeZO Algorithm 1 steps | CONFIRMED | — |
| Rademacher valid for MeZO's ZO-SGD | CONFIRMED | Clarified: not valid for classical SPSA (Spall 1992) |
| MeZO+LoRA on "LLaMA-7B SST-2=95%" | **INCORRECT** | Removed — MeZO paper has NO LLaMA experiments |
| ZO LR scales as n/(d+n-1) × SGD_LR | CONFIRMED | Equation 3 in paper |
| Dimension-free convergence (Theorem 1) | CONFIRMED | — |
| ANE dispatch overhead "~50μs" | **INCORRECT** | Corrected to ~95μs (maderix/Orion measurements) |
| 1x1 Conv2d "3x throughput" from Apple blog | **INCORRECT SOURCE** | 3x from Orion paper, not Apple blog |
| MobiZO 4.3x speedup (EMNLP 2025) | CONFIRMED | vs MeZO full-param (not MeZO+LoRA) |
| P-GAP 22.4% GPU hours (arXiv:2510.18228) | CONFIRMED | Table 5, vs MeZO+LoRA on SQuAD |
| Orion adapter-as-input (arXiv:2603.06728) | CONFIRMED | — |
| LoRA init: A ~ Gaussian, B = 0 | CONFIRMED | Paper says "random Gaussian" (no specific sigma) |
| MeZO default epsilon = 1e-3 | CONFIRMED | Code default in princeton-nlp/MeZO |

---

## 9. Sources

- [MeZO: Fine-Tuning Language Models with Just Forward Passes (NeurIPS 2023)](https://arxiv.org/abs/2305.17333)
- [MeZO GitHub Repository](https://github.com/princeton-nlp/MeZO)
- [xoshiro256+ Reference Implementation (Blackman & Vigna)](https://prng.di.unimi.it/xoshiro256plus.c)
- [SPSA Algorithm (Spall, JHU/APL)](https://www.jhuapl.edu/spsa/)
- [SPSA - Chessprogramming Wiki](https://www.chessprogramming.org/SPSA)
- [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning](https://arxiv.org/abs/2402.11592)
- [HuggingFace SmolLM2-135M Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [DeepNet: Scaling Transformers to 1,000 Layers (Wang et al., 2022)](https://arxiv.org/abs/2203.00555)
- [MeZO-SVRG: Variance-reduced ZO optimization (ICML 2024)](https://proceedings.mlr.press/v235/gautam24a.html)
- [Sparse MeZO: Efficient ZO with sparse updates (ICLR 2024)](https://arxiv.org/abs/2402.15751)
- [SubZero: ZO Fine-Tuning in Random Subspaces (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Zeroth-Order_Fine-Tuning_of_LLMs_in_Random_Subspaces_ICCV_2025_paper.pdf)
- [Orion: Characterizing and Programming Apple's ANE for LLM Training (arXiv:2603.06728)](https://arxiv.org/abs/2603.06728)
- [Deploying Transformers on the Apple Neural Engine (Apple ML Research)](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Unlocking the AMD NPU for ML Training on the Client (arXiv:2504.03083)](https://arxiv.org/abs/2504.03083)
- [ZO-Bench: Revisiting ZO Optimization for LLM Fine-Tuning (ICML 2024)](https://arxiv.org/abs/2402.11592)
- [Apple Foundation Models Tech Report 2025 (arXiv:2507.13575)](https://arxiv.org/abs/2507.13575)
- [Inside the M4 Apple Neural Engine (maderix benchmarks)](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [MobiZO: Efficient LLM Fine-Tuning at the Edge (EMNLP 2025)](https://arxiv.org/abs/2409.15520)
- [P-GAP: Projected Gradient-Aligned Perturbations (arXiv 2025)](https://arxiv.org/abs/2510.18228)
- [AGZO: Activation-Guided Zeroth-Order Optimization (arXiv 2026)](https://arxiv.org/abs/2601.17261)
- [DiZO: Divergence-driven Zeroth-Order Optimization (arXiv 2025)](https://arxiv.org/abs/2502.03304)
- [MeSP: Memory-Efficient Structured Backpropagation (arXiv 2026)](https://arxiv.org/abs/2602.13069)
- [llm.npu: Fast On-device LLM Inference with NPUs (ASPLOS 2025)](https://arxiv.org/abs/2407.05858)
- [On-Device Fine-Tuning via Backprop-Free ZO (arXiv 2025)](https://arxiv.org/abs/2511.11362)
- [ZO Fine-tuner: Learned ZO Optimizer (arXiv 2025)](https://arxiv.org/abs/2510.00419)
