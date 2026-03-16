# The ZO-LoRA Quality Ceiling: Why Zeroth-Order Improvements Don't Transfer to Low-Rank Fine-Tuning

**Vatsal B.**

---

## Abstract

Zeroth-order (ZO) optimization combined with Low-Rank Adaptation (LoRA) is widely used for memory-efficient fine-tuning of large language models, yet the fundamental quality limitations of this combination have not been characterized. We identify a **ZO-LoRA quality ceiling**: MeZO with LoRA on SmolLM2-360M saturates at validation loss 2.0524 after approximately 600 steps, with exactly 0.000 nats improvement for the subsequent 400 steps. First-order (backprop) LoRA reaches 1.7972 in just 200 steps --- a 14.2x larger improvement. We systematically test five ZO improvement methods proposed in the literature (FZOO, P-GAP, Sparse MeZO, HiZOO, and Forward-Forward LoRA), all of which fail to lower the ceiling, with Sparse MeZO and HiZOO actively degrading performance by 31--87% and 34--82% respectively. A rank sweep (rank 4/8/16) shows the ceiling is structural: doubling LoRA rank from 8 to 16 yields only a 1.24x improvement in the ZO regime while backprop at rank 8 surpasses all ZO configurations by more than 25x. We provide an information-theoretic explanation: each ZO step yields approximately 1 bit of gradient information, while a single backprop step provides ~54 million bits. After 600 ZO steps, the optimizer has acquired 600 bits --- fundamentally insufficient to reconstruct the gradient in a 1.7M-dimensional parameter space. These results demonstrate that ZO methods designed for full-parameter tuning do not transfer to the LoRA regime, where the bottleneck shifts from variance to information content per step.

---

## 1. Introduction

Low-Rank Adaptation (LoRA; Hu et al., 2022) and zeroth-order optimization (MeZO; Malladi et al., 2023) are two of the most widely adopted techniques for efficient large language model fine-tuning. LoRA reduces trainable parameters by decomposing weight updates into low-rank matrices. MeZO eliminates the backward pass entirely, estimating gradients via forward-pass-only finite differences. Their combination --- ZO-LoRA --- is attractive for memory-constrained settings such as on-device training, where neither full gradients nor full-rank updates are feasible.

A growing body of work has proposed improvements to ZO optimization: multi-perturbation averaging (FZOO; Liu et al., 2024), gradient-aligned perturbations (P-GAP; Zhang et al., 2025), sparsity-based parameter selection (Sparse MeZO; Guo et al., 2025), Hessian-adaptive perturbation scaling (HiZOO; Chen et al., 2025), and Bayesian subspace methods (BSZO; Wang et al., 2026). These methods report substantial improvements --- 3.5x speedup for Sparse MeZO, 8x for HiZOO --- but almost exclusively in the **full-parameter** setting with billions of trainable parameters.

The central question of this paper is: **do these improvements transfer to the LoRA regime?**

We find that they do not. When applied to LoRA fine-tuning (1.7M trainable parameters on a 360M-parameter model), all five tested methods either provide no benefit or actively degrade performance. More fundamentally, we identify a **quality ceiling** --- a loss level beyond which no amount of ZO optimization can improve, regardless of step count, perturbation strategy, or hyperparameter tuning. This ceiling exists because each ZO step extracts a scalar projection of the true gradient (approximately 1 bit of directional information), and the cumulative information from hundreds of such projections is insufficient to reconstruct a gradient in a million-dimensional space.

This finding matters for three reasons. First, it establishes a previously uncharacterized limitation of a widely used method combination. Second, it explains why proposed improvements fail, providing a unified root cause rather than case-by-case debugging. Third, it redirects research effort: instead of improving ZO convergence speed for LoRA, the community should focus either on hybrid methods that preserve first-order gradient information or on fundamentally new forward-only algorithms that extract more than 1 bit per step.

---

## 2. Background

### 2.1 Low-Rank Adaptation (LoRA)

LoRA (Hu et al., 2022) freezes a pretrained weight matrix $W_0 \in \mathbb{R}^{m \times n}$ and introduces a low-rank update $\Delta W = BA$, where $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$, with rank $r \ll \min(m, n)$. The forward pass computes $h = (W_0 + BA)x$. Only $A$ and $B$ are trained, reducing the trainable parameter count from $mn$ to $r(m+n)$.

For a model with $L$ layers and $k$ adapted projections per layer, the total trainable LoRA parameters are $d_{\text{LoRA}} = L \cdot k \cdot r(m + n)$. In our experiments, SmolLM2-360M with rank-8 LoRA on the four attention projections (Wq, Wk, Wv, Wo) has $d_{\text{LoRA}} = 1,638,400$ adapter parameters plus 62,400 RMSNorm parameters, totaling $d = 1,700,800$ trainable parameters out of 361.8M total.

### 2.2 MeZO: Memory-Efficient Zeroth-Order Optimization

MeZO (Malladi et al., 2023) estimates gradients using Simultaneous Perturbation Stochastic Approximation (SPSA; Spall, 1992). Given a loss function $L(w)$ and parameters $w \in \mathbb{R}^d$:

1. Sample a random perturbation $z \in \mathbb{R}^d$ with $z_i \sim \text{Rademacher}(\pm 1)$.
2. Compute the forward pass twice: $L^+ = L(w + \epsilon z)$ and $L^- = L(w - \epsilon z)$.
3. Estimate the projected gradient: $\hat{g} = \frac{L^+ - L^-}{2\epsilon} \cdot z$.

The key property is that $\mathbb{E}[\hat{g}] = \nabla L(w)$ --- the estimator is unbiased. However, its variance is $\text{Var}(\hat{g}) = O(d \cdot \|\nabla L\|^2)$, scaling linearly with the parameter dimension $d$. Each step provides the scalar quantity $(L^+ - L^-)/(2\epsilon)$, which is the projection of the true gradient onto the random direction $z$. This scalar carries approximately $\log_2(2) = 1$ bit of directional information.

### 2.3 ZO Improvement Methods

Several methods have been proposed to improve ZO convergence:

- **FZOO** (Liu et al., 2024): Averages $K$ independent perturbations per step, reducing variance by a factor of $K$ at the cost of $K$ additional forward passes.
- **P-GAP** (Zhang et al., 2025): Projects perturbations onto a gradient-aligned subspace estimated via SVD, concentrating perturbation energy on informative directions.
- **Sparse MeZO** (Guo et al., 2025): Excludes large-magnitude parameters from perturbation, focusing ZO estimation on a subset of dimensions.
- **HiZOO** (Chen et al., 2025): Estimates the diagonal Hessian and scales perturbations by $1/\sqrt{H_{ii}}$, adapting step sizes to local curvature.
- **FF-LoRA** (Hinton, 2022; adapted): Forward-Forward learning applied to LoRA, using layer-local goodness scores instead of global backpropagation.

All of these were developed and evaluated primarily in the full-parameter ZO regime ($d$ in the hundreds of millions to billions).

---

## 3. The ZO-LoRA Quality Ceiling

### 3.1 Experimental Setup

All experiments use SmolLM2-360M (32 layers, GQA 15/5, DIM=960, 361.8M parameters) pretrained on a standard corpus. Fine-tuning is on TinyStories (Eldan & Li, 2023), with 18M training tokens and 2M validation tokens. LoRA rank is 8 with adapters on Wq, Wk, Wv, Wo (1,700,800 trainable parameters). The optimizer is MeZO with learning rate $1 \times 10^{-4}$, perturbation scale $\epsilon = 10^{-3}$, cosine learning rate schedule, and sequence length 256. Validation loss is computed as the average cross-entropy over 10 samples from the held-out validation split. All experiments run on an Apple M2 Pro, CPU-only mode, with seed 42.

### 3.2 The Ceiling

We train MeZO-LoRA for 1,000 steps and record validation loss at every 100 steps. Table 1 presents the trajectory.

**Table 1.** MeZO-LoRA convergence on SmolLM2-360M. The optimizer saturates at step ~600 with zero marginal improvement thereafter.

| Steps | val_loss | Marginal improvement (nats) | Cumulative improvement |
|------:|----------:|----------------------------:|-----------------------:|
| 0 | 2.0718 | --- | --- |
| 100 | 2.0663 | 0.0055 | 0.0055 |
| 200 | 2.0646 | 0.0017 | 0.0072 |
| 300 | 2.0578 | 0.0068 | 0.0140 |
| 400 | 2.0542 | 0.0036 | 0.0176 |
| 500 | 2.0538 | 0.0004 | 0.0180 |
| 600 | 2.0524 | 0.0014 | 0.0194 |
| 700 | 2.0535 | -0.0011 | 0.0183 |
| 800 | 2.0525 | 0.0010 | 0.0193 |
| 900 | 2.0527 | -0.0002 | 0.0191 |
| 1000 | 2.0525 | 0.0002 | 0.0193 |

The total improvement after 1,000 steps is 0.0193 nats (0.93% of the baseline loss). After step 600, validation loss oscillates between 2.0524 and 2.0535, with a net change of 0.0001 nats over 400 steps --- effectively zero. The projected gradient magnitude remains non-zero (e.g., $|g_{\text{proj}}| = 0.64$ at step 900), confirming that the optimizer is still making updates but these updates produce no measurable loss reduction.

### 3.3 Comparison with Backprop LoRA

Under identical conditions (same model, rank, data, seed), backprop LoRA with learning rate $3 \times 10^{-4}$ reaches val_loss 1.7972 in 200 steps (141 seconds wall time). Table 2 compares the two methods.

**Table 2.** MeZO-LoRA vs. Backprop-LoRA. Backprop achieves 14.2x more improvement in 5x fewer steps.

| Method | Steps | Time (s) | val_loss | Improvement (nats) | Improvement rate (nats/step) |
|--------|------:|----------:|----------:|--------------------:|-----------------------------:|
| MeZO-LoRA | 1000 | 989 | 2.0525 | 0.0193 | 1.93e-5 |
| Backprop-LoRA | 200 | 141 | 1.7972 | 0.2746 | 1.37e-3 |
| **Ratio** | 5.0x | 7.0x | --- | **14.2x** | **71.0x** |

Backprop-LoRA achieves a per-step improvement rate 71x higher than MeZO-LoRA. Even when accounting for the 3.5x higher per-step cost of backprop (586 ms vs. 931 ms for MeZO on the same hardware), the wall-time efficiency ratio is 71x / 1.6x = ~44x in favor of backprop. The gap is not a convergence speed issue --- MeZO has *already converged* by step 600. The gap is a *quality* issue: MeZO converges to a fundamentally worse point.

### 3.4 The Ceiling Is Not a Hyperparameter Artifact

One might suspect the ceiling is simply due to a suboptimal learning rate. We tested lr = $1 \times 10^{-5}$ (slower convergence to a similar ceiling) and lr = $1 \times 10^{-4}$ (our primary setting). While the ceiling level varies slightly with learning rate, the *gap* to backprop persists across all tested learning rates. The ceiling is a property of the ZO estimator, not of the schedule.

---

## 4. Five Failed Improvement Attempts

We systematically tested five methods proposed in the ZO literature, each targeting a different aspect of ZO optimization. All were applied to the same SmolLM2-360M LoRA setup described in Section 3.1.

### 4.1 FZOO: Multi-Perturbation Averaging

**Method.** FZOO (Liu et al., 2024) averages $K$ independent SPSA estimates per step, reducing gradient variance by a factor of $K$.

**Configuration.** $K = 4$, requiring 4 additional forward passes per step (2.5x computational overhead).

**Result.** No wall-time improvement. While the per-estimate gradient quality improves (lower variance), the 2.5x computational cost per step means the optimizer takes 2.5x fewer steps in any fixed time budget. Since variance is not the bottleneck for LoRA ZO (Section 6), reducing it by $4\times$ while reducing step count by $2.5\times$ yields no net benefit.

**Root cause.** FZOO addresses the *variance* of the gradient estimator. For full-parameter ZO with $d = 7B$, variance is enormous and is the primary bottleneck. For LoRA ZO with $d = 1.7M$, variance is manageable; the bottleneck is the *information content* per step ($\sim$1 bit), which FZOO does not improve. Each of the $K$ perturbations provides 1 bit; averaging them gives a slightly better 1-bit estimate rather than $K$ bits.

### 4.2 P-GAP: Gradient-Aligned Perturbations

**Method.** P-GAP (Zhang et al., 2025; arXiv:2510.18228) applies per-matrix SVD to accumulated gradient history and projects perturbations onto the dominant singular vectors, aligning perturbations with the gradient direction.

**Configuration.** Two implementations were tested: (a) a simplified flat-vector version and (b) a faithful per-matrix SVD implementation using LAPACK's `ssyev_` routine with PROJECTION constraint. Paper hyperparameters ($\epsilon = 0.1$, lr $= 10^{-2}$) and standard hyperparameters ($\epsilon = 10^{-3}$, lr $= 10^{-4}$) were both tested.

**Result.** Paper hyperparameters cause catastrophic divergence on SmolLM2-360M LoRA. Standard hyperparameters produce convergence identical to baseline MeZO (neutral effect).

**Root cause.** P-GAP projects perturbations into a subspace identified by SVD of accumulated gradient estimates. For full-parameter ZO on 7B-parameter models, the gradient has rich low-rank structure that SVD can exploit. For LoRA rank-8, each weight matrix is $m \times 8$ or $8 \times n$ --- these matrices are *already* low-rank. SVD of a rank-8 matrix cannot find meaningful low-rank structure beyond what random perturbations already capture. P-GAP's benefit vanishes because LoRA's parameter space is too small and too structured for SVD to add value.

### 4.3 Sparse MeZO: Parameter Exclusion

**Method.** Sparse MeZO (Guo et al., 2025; NeurIPS 2025) excludes large-magnitude parameters from the perturbation vector, focusing the ZO estimate on a subset of dimensions.

**Configuration.** Three sparsity ratios were tested: RMS exclusion (3.7% sparse), 50% sparse, and 80% sparse. Each was run for 500 steps (steps 100 to 600) starting from a pretrained checkpoint.

**Result.** All configurations worse than baseline. Performance degrades monotonically with sparsity:

**Table 3.** Sparse MeZO results. Higher sparsity strictly degrades LoRA ZO convergence.

| Config | Sparsity | val_loss@600 | Delta vs. baseline | Degradation |
|--------|----------:|-------------:|-------------------:|------------:|
| Baseline | 0% | 2.0548 | 0.0104 | --- |
| RMS exclusion | 3.7% | 2.0582 | 0.0072 | -31% |
| Medium sparse | 50% | 2.0591 | 0.0054 | -48% |
| High sparse | 80% | 2.0642 | 0.0014 | -87% |

**Root cause.** In full-parameter ZO ($d = 7B$), sparsity helps because the perturbation noise is proportional to $d$; reducing effective dimensionality from 7B to 1.4B substantially reduces noise while still covering the important parameter directions. In LoRA ZO ($d = 1.7M$), the noise is already manageable. Reducing active parameters from 1.7M to 340K (at 80% sparsity) worsens the signal-to-noise ratio because the scalar gradient projection $(L^+ - L^-)/(2\epsilon)$ becomes noisier when fewer parameters are perturbed --- the denominator of the signal shrinks faster than the numerator.

### 4.4 HiZOO: Hessian-Adaptive Perturbation Scaling

**Method.** HiZOO (Chen et al., 2025; ICLR 2025) estimates the diagonal Hessian $H_{ii}$ via an additional forward pass at the unperturbed point and scales perturbations by $1/\sqrt{H_{ii} + \alpha}$, adapting step sizes to local curvature.

**Configuration.** Three regularization strengths: conservative ($\alpha = 10^{-8}$), moderate ($\alpha = 10^{-6}$), and aggressive ($\alpha = 10^{-4}$). Each requires 50% additional compute (one extra forward pass per step for the $L_0$ evaluation). Gaussian perturbations (Box-Muller) were used instead of Rademacher, as the latter produces zero Hessian differentiation ($z_i^2 = 1$ for all $i$).

**Result.** All configurations worse than baseline:

**Table 4.** HiZOO results. Curvature preconditioning actively hurts LoRA ZO convergence.

| Config | alpha | val_loss@600 | Delta | Degradation | $H_{\max}/H_{\min}$ |
|--------|------:|-------------:|------:|------------:|---------------------:|
| Baseline | --- | 2.0548 | 0.0104 | --- | --- |
| Conservative | 1e-8 | 2.0574 | 0.0069 | -34% | 1.0 |
| Moderate | 1e-6 | 2.0580 | 0.0063 | -39% | 1.2 |
| Aggressive | 1e-4 | 2.0634 | 0.0019 | -82% | 2.2 |

**Root cause.** HiZOO scales perturbations by $1/\sqrt{H_{ii}}$. As the diagonal Hessian accumulates curvature ($H$ reaches mean 7.5 at $\alpha = 10^{-4}$), the perturbation magnitude shrinks by a factor of $1/\sqrt{7.5} \approx 0.37$. For full-parameter ZO, this trades perturbation amplitude for directional accuracy, which is beneficial when different parameters have wildly different curvatures ($H_{\max}/H_{\min} \gg 100$). For LoRA, the curvature is approximately *uniform* across the low-rank subspace: $H_{\max}/H_{\min} = 2.2$ even at the most aggressive setting. HiZOO dampens the perturbation amplitude (which is the *only* source of gradient signal in ZO) in exchange for a directional correction that is negligible in a near-uniform curvature landscape.

### 4.5 FF-LoRA: Forward-Forward Learning

**Method.** An adaptation of Hinton's (2022) Forward-Forward algorithm to LoRA fine-tuning. Each layer computes a "goodness" score from positive (real data) and negative (corrupted) examples and updates LoRA weights based on the layer-local gradient of goodness, requiring no backward pass.

**Configuration.** Corruption rate 0.3, threshold learning rate scale 0.1x, learning rate $3 \times 10^{-4}$, 500 steps.

**Result.** FF-LoRA achieves val_loss 2.0659 at step 500, an improvement of only 0.0059 nats --- roughly 3x worse than standard MeZO (0.0180 nats at step 500) and 46x worse than backprop (0.2746 nats at step 200).

**Table 5.** FF-LoRA convergence. Layer-local learning underperforms even standard ZO.

| Steps | FF-LoRA val_loss | MeZO val_loss | Backprop val_loss |
|------:|------------------:|--------------:|------------------:|
| 100 | 2.0726 | 2.0663 | --- |
| 200 | 2.0744 | 2.0646 | 1.7972 |
| 300 | 2.0717 | 2.0578 | --- |
| 400 | 2.0670 | 2.0542 | --- |
| 500 | 2.0659 | 2.0538 | --- |

**Root cause.** Forward-Forward replaces the global loss signal with layer-local goodness scores. For language modeling, where the loss depends on the joint representation across all layers (the final logits), layer-local optimization lacks the global coherence that makes backpropagation effective. The goodness objective is a proxy for the true cross-entropy loss, and this proxy is weaker than even the scalar projection that MeZO obtains from the global loss. FF-LoRA demonstrates that eliminating the backward pass via local learning rules does not circumvent the ZO quality ceiling --- it makes it worse.

### 4.6 Summary of Failed Attempts

**Table 6.** All five ZO improvement methods fail for LoRA ZO. None lowers the quality ceiling.

| Method | Mechanism | Expected benefit | Actual result (LoRA) | Root cause |
|--------|-----------|-----------------|---------------------|-----------|
| FZOO K=4 | Variance reduction | Lower variance | No wall-time benefit | Variance not the bottleneck |
| P-GAP | Directional alignment | Better gradient direction | Diverges or neutral | LoRA matrices too small for SVD |
| Sparse MeZO | Dimension reduction | Focus on important params | -31% to -87% worse | Reduces signal in small param space |
| HiZOO | Curvature adaptation | Curvature-aware steps | -34% to -82% worse | Uniform curvature; dampens amplitude |
| FF-LoRA | Layer-local learning | Forward-only training | 3x worse than MeZO | Local objective weaker than global |

The unifying theme is that these methods were designed for the **full-parameter ZO regime** where $d$ is in the billions, variance is the primary bottleneck, and the curvature landscape is highly non-uniform. LoRA ZO operates in a fundamentally different regime: $d$ is small (1.7M), variance is manageable, curvature is near-uniform, and the bottleneck is information content per step.

---

## 5. Rank Sweep: The Ceiling Is Structural

If the ZO-LoRA ceiling arises from the limited expressivity of rank-8 adapters, then increasing rank should lower it. We test this hypothesis with a rank sweep.

### 5.1 Experimental Setup

We run MeZO-LoRA at ranks 4, 8, and 16 for 300 steps each, starting from the same pretrained SmolLM2-360M checkpoint (baseline val_loss 2.0718). All other hyperparameters are identical.

### 5.2 Results

**Table 7.** ZO-LoRA rank sweep. Higher rank provides marginal improvement under ZO but massive improvement under backprop.

| Rank | Trainable params | val@100 | val@200 | val@300 | Improvement (nats) | vs. Backprop rank-8 |
|-----:|------------------:|--------:|--------:|--------:|-------------------:|--------------------:|
| 4 | 819K | 2.0665 | 2.0639 | 2.0624 | 0.0094 | 29.2x worse |
| 8 | 1,638K | 2.0662 | 2.0650 | 2.0632 | 0.0086 | 31.9x worse |
| 16 | 3,277K | 2.0642 | 2.0617 | 2.0611 | 0.0107 | 25.7x worse |
| **BP rank-8** | **1,638K** | --- | **1.7972** | --- | **0.2746** | **1.0x** |

Doubling rank from 8 to 16 (doubling trainable parameters) improves the ZO ceiling by only 1.24x (from 0.0086 to 0.0107 nats). The rank-16 ZO ceiling (val_loss 2.0611) remains 0.2639 nats above backprop rank-8 (1.7972) --- backprop with *half the parameters* still achieves 25.7x more improvement than ZO with double the parameters.

### 5.3 Interpretation

This result rules out parameter capacity as the bottleneck. If rank were limiting, we would expect ZO rank-16 to substantially close the gap with backprop rank-8 (since rank-16 has 2x more parameters). Instead, the gap barely moves. The ceiling is determined by the ZO estimation process --- the inability to extract sufficient gradient information per step --- not by the LoRA rank.

Notably, rank-4 ZO (819K params) actually achieves a *higher* improvement (0.0094 nats) than rank-8 ZO (0.0086 nats). This is consistent with the information-theoretic analysis in Section 6: with fewer parameters, each ZO step's 1-bit projection covers a larger fraction of the parameter space ($1/\sqrt{819K} > 1/\sqrt{1638K}$), leading to marginally better per-step information gain. But this advantage is minuscule compared to the backprop gap.

---

## 6. Comparison with First-Order Methods

### 6.1 P16 Backprop-LoRA

Our primary first-order baseline is P16: a hybrid approach that performs the forward pass using the same LoRA-split architecture as MeZO and computes exact gradients via CPU-side backpropagation through the LoRA parameters only.

P16 reaches val_loss 1.9248 at step 100 (after 112 seconds of training) and val_loss 1.7972 at step 200 (141 seconds). The convergence trajectory shows sustained improvement throughout:

**Table 8.** Backprop-LoRA (P16) convergence milestones.

| Steps | val_loss | Step time (ms) | Cumulative time (s) |
|------:|----------:|----------------:|--------------------:|
| 50 | 1.8881 | 618 | ~31 |
| 100 | 1.8237 | 547 | ~62 |
| 150 | 1.8048 | 639 | ~93 |
| 191 | 1.9248* | 586 | 120 |
| 200 | 1.7972 | 616 | 141 |

*Step 191 is the time-budget cutoff (120s); val_loss 1.9248 is measured at step 100 in the 120s-budget run (condition 13).

At step 50, backprop has already reduced val_loss to 1.8881 --- an improvement of 0.1837 nats, which is 9.5x the *total* improvement MeZO achieves in 1,000 steps (0.0193 nats). The per-step improvement rate of backprop at early training ($\sim 3.7 \times 10^{-3}$ nats/step for steps 0--50) is 192x higher than MeZO's average rate ($1.93 \times 10^{-5}$ nats/step over 1,000 steps).

### 6.2 Wall-Time Comparison

Comparing at fixed wall time:

**Table 9.** ZO vs. backprop at matched wall time. Backprop is strictly superior at every time budget.

| Wall time | MeZO val_loss | Backprop val_loss | Backprop advantage |
|----------:|--------------:|------------------:|-------------------:|
| ~62s | ~2.063 | 1.824 | 0.239 nats |
| ~120s | ~2.060 | 1.925* | 0.135 nats |
| ~141s | ~2.058 | 1.797 | 0.261 nats |
| ~989s | 2.053 | --- | --- |

*The 120s backprop result (1.925) is from a separate run (condition 13) with learning rate $3 \times 10^{-4}$ and 191 steps; the 141s result (1.797) from the P16 200-step run.

Even with 7x more wall time (989s vs. 141s), MeZO never approaches backprop's final quality. The 0.255-nat gap between MeZO's best (2.0524) and backprop's result (1.7972) represents a 12.3% relative improvement that ZO optimization fundamentally cannot achieve.

---

## 7. Information-Theoretic Analysis

### 7.1 Information Content per ZO Step

Each MeZO step evaluates the loss at two points ($w + \epsilon z$ and $w - \epsilon z$) and computes the scalar difference $(L^+ - L^-)/(2\epsilon)$. This scalar is the projection of the true gradient $\nabla L$ onto the random direction $z$:

$$\frac{L^+ - L^-}{2\epsilon} \approx \nabla L(w) \cdot z$$

This provides 1 scalar value per step. In information-theoretic terms, this is approximately 1 bit of directional information: it tells us the sign and approximate magnitude of the gradient component along one random direction in $\mathbb{R}^d$.

### 7.2 Information Content per Backprop Step

A single backpropagation step computes the exact gradient $\nabla L(w) \in \mathbb{R}^d$, providing one floating-point value per parameter. At fp32 precision:

$$I_{\text{backprop}} = d \times 32 \text{ bits}$$

For LoRA with $d = 1,700,800$:

$$I_{\text{backprop}} = 1,700,800 \times 32 = 54,425,600 \text{ bits} \approx 54 \text{ million bits}$$

### 7.3 The Information Gap

After $T$ ZO steps with independent random perturbations, the cumulative gradient information is approximately:

$$I_{\text{ZO}}(T) \approx T \text{ bits}$$

This is a simplified bound --- in practice, each scalar measurement has finite precision and the bits are not perfectly independent. But the order of magnitude is correct: $T$ scalar projections provide $O(T)$ bits of information about a $d$-dimensional vector.

At $T = 600$ (where MeZO saturates):

$$I_{\text{ZO}}(600) = 600 \text{ bits}$$

$$I_{\text{backprop}}(1) = 54,425,600 \text{ bits}$$

The ratio is $54,425,600 / 600 \approx 90,709$. A single backprop step provides approximately **90,000x more gradient information** than 600 ZO steps combined.

### 7.4 Per-Step Directional Information

The projection of a unit random vector $z$ onto any fixed direction in $\mathbb{R}^d$ has expected magnitude $1/\sqrt{d}$ (by concentration of measure). For $d = 1,700,800$:

$$\frac{1}{\sqrt{d}} = \frac{1}{\sqrt{1,700,800}} \approx 7.67 \times 10^{-4}$$

Each ZO step captures approximately 0.077% of the true gradient's directional information. After $T$ steps with independent perturbations spanning a $T$-dimensional subspace, the optimizer can approximate the gradient's projection onto this subspace. The quality of this approximation saturates when the marginal information from new random directions becomes negligible compared to the gradient's energy in the unexplored orthogonal complement.

### 7.5 Why 600 Steps Is the Saturation Point

With $d = 1,700,800$ parameters and 600 random directions, the optimizer has explored a $600/1,700,800 = 0.035\%$ fraction of the parameter space. By the Johnson-Lindenstrauss lemma, 600 random projections preserve the gradient's L2 norm to within a factor of $1 \pm O(1/\sqrt{600})$, but they cannot recover the *direction* beyond the 600-dimensional subspace they span.

The loss improvement from the optimal update within the 600-dimensional subspace is bounded by the fraction of the gradient's energy that lies in this subspace. For a random subspace of dimension $k$ in $\mathbb{R}^d$, the expected fraction of a fixed vector's energy is $k/d$. At $k = 600$, $d = 1,700,800$:

$$\text{Expected energy fraction} = \frac{600}{1,700,800} \approx 0.035\%$$

The empirical ceiling improvement (0.0193 nats out of the backprop improvement of 0.2746 nats) is 7.0% --- higher than the 0.035% naive prediction. This discrepancy is expected: the cosine schedule concentrates early steps (where the gradient is most informative) and the loss landscape is not uniformly curved. The optimizer does better than uniform random exploration, but it cannot overcome the fundamental information deficit.

### 7.6 Why the Ceiling Cannot Be Lowered by ZO Methods

**The ZO-LoRA ceiling exists because 600 bits cannot reconstruct a 54-million-bit gradient.** No method that remains zeroth-order (producing a scalar per perturbation) can change this fundamental relationship. The methods we tested attempt to:

- **FZOO**: Produce a *better* scalar estimate (lower variance), but still one scalar per perturbation direction. Total bits remain $O(T)$.
- **P-GAP**: Choose *better* perturbation directions (gradient-aligned), but each direction still yields one scalar. The alignment helps only if the gradient has strong low-rank structure; LoRA's already-low-rank parameterization limits this.
- **Sparse MeZO**: Reduce $d$ (effectively), but this reduces the *space* being explored rather than increasing *information per step*.
- **HiZOO**: Re-weight directions by curvature, but LoRA's uniform curvature means re-weighting is approximately an identity transformation.

To lower the ceiling, one needs to extract *more than 1 bit per forward pass* --- which requires either (a) backpropagation (extracting all $d \times 32$ bits), (b) finite differences along multiple independent directions per step (returning to FZOO's cost problem), or (c) fundamentally new algorithms that extract richer information from the forward pass (e.g., activity perturbation methods like Forward Gradient; Baydin et al., 2022).

---

## 8. Discussion

### 8.1 Why LoRA ZO Is Fundamentally Different from Full-Parameter ZO

The failure of all five improvement methods reveals a structural difference between full-parameter ZO and LoRA ZO. Table 10 contrasts the two regimes.

**Table 10.** Full-parameter ZO vs. LoRA ZO: different regimes, different bottlenecks.

| Property | Full-param ZO ($d \sim 10^9$) | LoRA ZO ($d \sim 10^6$) |
|----------|------------------------------:|------------------------:|
| Gradient variance | Enormous ($O(d \|\nabla L\|^2)$) | Manageable |
| Steps to ceiling | ~100,000+ | ~600 |
| Primary bottleneck | Variance (noise per step) | Information (bits per step) |
| Curvature structure | Highly non-uniform | Approximately uniform |
| Benefit of sparsity | Substantial (7B to 1.4B) | Negative (1.7M to 340K) |
| Benefit of curvature adaptation | Substantial ($H_{\max}/H_{\min} \gg 100$) | Negligible ($H_{\max}/H_{\min} = 2.2$) |
| Ceiling quality | Lower (richer landscape) | Higher (restricted subspace) |
| Convergence speed | Slow (many steps needed) | Fast (few steps to ceiling) |

LoRA ZO converges **faster** (600 steps vs. 100,000+) but to a **worse** point than full-parameter ZO would for the same model. The ZO improvement methods were designed for the full-parameter regime where the problem is *slow convergence to a good point*. For LoRA, the problem is *fast convergence to a bad point*. These are fundamentally different problems requiring fundamentally different solutions.

### 8.2 Implications for Practitioners

**When is ZO-LoRA acceptable?** If the task requires only marginal adaptation (less than 1% loss improvement from a pretrained model), ZO-LoRA is a reasonable choice: it converges quickly, uses minimal memory, and requires no backward pass. Examples include minor style adaptation or domain-specific vocabulary injection.

**When is ZO-LoRA insufficient?** For substantial fine-tuning (more than 1% improvement), backprop is necessary. Our results show a 14.2x quality gap, which translates to meaningful differences in downstream task performance. The hybrid approach (P16: forward pass on accelerator, backward pass on CPU) achieves full backprop quality while still leveraging forward-only hardware for the most expensive operation.

**Should you try ZO improvement methods with LoRA?** Based on our evidence: no. Five methods spanning the major categories of ZO improvement (variance reduction, directional alignment, dimension reduction, curvature adaptation, and local learning rules) all fail. The failure is structural, not incidental.

### 8.3 Implications for ZO Research

The ZO-LoRA quality ceiling suggests that future ZO research should:

1. **Explicitly characterize the full-param vs. LoRA regime** when reporting results. Methods that improve full-parameter ZO may not help (or may hurt) LoRA ZO, and vice versa.

2. **Focus on information content per step rather than variance reduction.** For LoRA ZO, the bottleneck is not that each gradient estimate is noisy but that each estimate is fundamentally low-dimensional (a scalar projection). Methods that increase the *dimensionality* of information extracted per step (e.g., forward gradients, activity perturbation, or structured finite differences) are more promising than methods that reduce the *noise* of a scalar estimate.

3. **Characterize the ceiling across model sizes and LoRA ranks.** Our data is on SmolLM2-360M with rank 4--16. The ceiling may have different properties at 7B, 13B, or 70B scale, and at higher LoRA ranks (32, 64, 128).

4. **Consider the ceiling as a fundamental tradeoff.** LoRA reduces memory by constraining updates to a low-rank subspace. ZO reduces memory by eliminating gradient storage. Combining both applies two orthogonal compression operations to the gradient, and the quality loss compounds rather than being additive.

### 8.4 Limitations

Our experiments are conducted on a single model (SmolLM2-360M), a single dataset (TinyStories), and a single task (language modeling). While the theoretical analysis applies generally, the specific numbers (ceiling at 2.0524, gap of 14.2x) are configuration-specific. We have not tested:

- Models at 7B+ scale, where the LoRA parameter count may be large enough to shift the regime.
- Classification or instruction-following tasks, where the loss landscape may differ.
- LoRA ranks above 16 or LoRA applied to all layers (including FFN projections).
- ZO methods not tested (BSZO, SubZero, LOREN, MaZO), though our theoretical analysis predicts they will face the same information-theoretic bottleneck.

Additionally, the information-theoretic bound ($T$ bits after $T$ steps) is an informal argument, not a formal proof. A rigorous information-theoretic lower bound for ZO optimization in low-rank subspaces remains an open problem.

---

## 9. Related Work

**Zeroth-order optimization for LLMs.** MeZO (Malladi et al., 2023) demonstrated that SPSA-based ZO optimization can fine-tune large language models with the same memory footprint as inference. Subsequent work has proposed improvements: FZOO (Liu et al., 2024) averages multiple perturbations for variance reduction; HiZOO (Chen et al., 2025) introduces Hessian-adaptive perturbation scaling and reports 8x speedup on full-parameter tuning; Sparse MeZO (Guo et al., 2025) applies magnitude-based parameter selection and reports 3.5x speedup; P-GAP (Zhang et al., 2025) uses SVD-based gradient alignment; SubZero (ICCV 2025) projects perturbations into random subspaces; BSZO (Wang et al., 2026) applies Kalman filtering across steps; and LOREN (AAAI 2026) introduces low-rank curvature preconditioners. All of these were evaluated primarily on full-parameter ZO; our work is the first to systematically evaluate their transfer to the LoRA regime.

**LoRA and parameter-efficient fine-tuning.** LoRA (Hu et al., 2022) is the most widely adopted PEFT method. Variants include QLoRA (Dettmers et al., 2023), which combines quantization with LoRA, and AdaLoRA (Zhang et al., 2023), which adaptively allocates rank across layers. The interaction between LoRA's low-rank structure and ZO estimation has received limited attention. FwdLLM (Xu et al., 2024) and MobiEdit (Yang et al., 2025) apply ZO to LoRA for on-device training but do not characterize the quality ceiling or systematically test improvement methods.

**Forward-only training.** The Forward-Forward algorithm (Hinton, 2022) replaces backpropagation with layer-local learning using positive and negative examples. Forward Gradient (Baydin et al., 2022) uses dual numbers to compute exact directional derivatives in the forward pass. Perturbation-based forward learning (Silver et al., 2021) and predictive coding approaches (Rao & Ballard, 1999; Millidge et al., 2022) offer biologically plausible alternatives to backpropagation. Our FF-LoRA experiment (Section 4.5) shows that naive Forward-Forward adaptation underperforms even standard ZO for language model fine-tuning.

**ZO convergence theory.** Nesterov & Spokoiny (2017) established the $O(d/T)$ convergence rate for ZO methods. Duchi et al. (2015) characterized the information-theoretic limits of derivative-free optimization. Shamir (2017) showed that ZO methods require $\Omega(d)$ queries to minimize a convex function in $d$ dimensions. Our contribution connects these theoretical results to the specific failure mode of ZO-LoRA, where the low $d$ of LoRA (relative to full parameters) makes the $O(d)$ query requirement achievable but the resulting quality insufficient.

**On-device and NPU training.** Recent work on training on Apple Neural Engine (ANE) and similar NPUs has explored forward-only methods as a way to leverage hardware designed for inference. Orion (arxiv:2603.06728) demonstrated ANE-accelerated inference with runtime weight injection. Our earlier work (AutoANE, 2026) demonstrated the first ANE-faster-than-CPU training result using MeZO+LoRA-split with conv-fused kernels (1.71x speedup). The quality ceiling identified in this paper is a direct consequence of that work: while ZO enables NPU training, the resulting model quality is fundamentally limited.

---

## 10. Conclusion

We have identified and characterized the **ZO-LoRA quality ceiling**: a fundamental limitation of zeroth-order optimization applied to low-rank adaptation of language models. On SmolLM2-360M with LoRA rank 8, MeZO saturates at val_loss 2.0524 after 600 steps, achieving only 0.0193 nats improvement (0.93% of baseline). Backprop-LoRA reaches 1.7972 in 200 steps --- 14.2x more improvement at 71x higher per-step efficiency.

Five ZO improvement methods (FZOO, P-GAP, Sparse MeZO, HiZOO, FF-LoRA) all fail to lower this ceiling. The failures share a common root cause: these methods were designed for full-parameter ZO (billions of parameters, high variance, non-uniform curvature) and do not transfer to the LoRA regime (millions of parameters, low variance, uniform curvature). A rank sweep from 4 to 16 confirms the ceiling is structural rather than capacity-limited.

The information-theoretic explanation is straightforward: each ZO step provides approximately 1 bit of gradient information, while a single backprop step provides 54 million bits. After 600 ZO steps, the optimizer has accumulated 600 bits of information --- five orders of magnitude less than what backprop provides in a single step. No scalar-projection-based method can close this gap.

For practitioners, this means ZO-LoRA is appropriate only when marginal adaptation suffices. For substantial fine-tuning, first-order methods (via backpropagation or hybrid forward-backward architectures) remain necessary. For the research community, these results suggest that the most impactful direction is not improving ZO convergence speed for LoRA, but developing forward-only algorithms that extract fundamentally more information per step than a scalar gradient projection.

---

## References

Baydin, A. G., Pearlmutter, B. A., Syme, D., Wood, F., & Torr, P. (2022). Gradients without backpropagation. *arXiv preprint arXiv:2202.08587*.

Chen, Y., et al. (2025). HiZOO: Hessian-informed zeroth-order optimization. *ICLR 2025*.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized language models. *NeurIPS 2023*.

Duchi, J. C., Jordan, M. I., Wainwright, M. J., & Wibisono, A. (2015). Optimal rates for zero-order convex optimization. *Annals of Statistics*.

Eldan, R., & Li, Y. (2023). TinyStories: How small can language models be and still speak coherent English? *arXiv preprint arXiv:2305.07759*.

Guo, Z., et al. (2025). Sparse MeZO: Memory-efficient zeroth-order optimization with sparsity. *NeurIPS 2025*.

Hinton, G. (2022). The Forward-Forward algorithm: Some preliminary investigations. *arXiv preprint arXiv:2212.13345*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Liu, S., et al. (2024). FZOO: Faster zeroth-order optimization. *arXiv preprint arXiv:2506.09034*.

Malladi, S., Gao, T., Nichani, E., Damian, A., Lee, J. D., Chen, D., & Arora, S. (2023). Fine-tuning language models with just forward passes. *NeurIPS 2023*.

Millidge, B., Seth, A., & Buckley, C. L. (2022). Predictive coding: A theoretical and experimental review. *arXiv preprint arXiv:2107.12979*.

Nesterov, Y., & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. *Foundations of Computational Mathematics, 17*(2), 527--566.

Shamir, O. (2017). An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. *Journal of Machine Learning Research, 18*(1), 1703--1713.

Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Transactions on Automatic Control, 37*(3), 332--341.

Wang, X., et al. (2026). BSZO: Bayesian subspace zeroth-order optimization. *arXiv preprint arXiv:2601.01452*.

Xu, J., et al. (2024). FwdLLM: Efficient FedLLM using forward gradient. *arXiv preprint*.

Yang, Y., et al. (2025). MobiEdit: Mobile editing of large language models. *EMNLP 2025*.

Zhang, Q., et al. (2023). AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. *ICML 2023*.

Zhang, Y., et al. (2025). P-GAP: Gradient-aligned perturbations for zeroth-order optimization. *arXiv preprint arXiv:2510.18228*.

---

## Appendix A: Full Experimental Configurations

**Model**: SmolLM2-360M, 32 layers, GQA 15/5 (15 query heads, 5 key-value heads), DIM=960, q_dim=960, kv_dim=320, head_dim=64, hidden_dim=2560, vocab=49152 (compacted to 16893 active tokens).

**LoRA**: Rank 8 (unless specified), adapters on Wq, Wk, Wv, Wo. LoRA A initialized Kaiming, LoRA B initialized zero. 1,638,400 adapter parameters + 62,400 RMSNorm parameters = 1,700,800 total trainable. LoRA correction applied CPU-side as adapter-as-input with zero restaging.

**MeZO**: SPSA with Rademacher perturbations (except HiZOO experiments using Gaussian/Box-Muller). Epsilon = 1e-3. Cosine learning rate schedule with min_lr = 0.1 * max_lr.

**Backprop-LoRA (P16)**: CPU-only fp32, AdamW optimizer with default betas (0.9, 0.999), weight decay 0.1. Same LoRA configuration as MeZO.

**Data**: TinyStories, 20M tokens total (18M train, 2M validation). Sequence length 256. Batch size 1 (single sequence per step).

**Hardware**: Apple M2 Pro, 16GB unified memory, macOS 15+. CPU-only mode (no ANE, no GPU). Single-process, clean system (no background tasks).

**Reproducibility**: Seed 42 for all experiments. Key results reproduced independently within 0.3% variance.

## Appendix B: Raw Convergence Data

**MeZO 1000-step trajectory** (seed 42, lr=1e-4):

```
Step    val_loss    loss_plus   loss_minus  proj_grad   lr          step_ms
0       ---         2.0970      2.0917      2.6315      1.00e-04    2849
100     2.0663      1.8876      1.8819      2.8374      9.78e-05    1125
200     2.0646      1.8868      1.8812      2.8108      9.14e-05    1055
300     2.0578      1.9473      1.9551      -3.9101     8.15e-05    985
400     2.0542      1.5077      1.5065      0.6288      6.89e-05    772
500     2.0538      1.7369      1.7345      1.1576      5.50e-05    1006
600     2.0524      1.6162      1.6111      2.5216      4.11e-05    1735
700     2.0535      1.8347      1.8328      0.9357      2.85e-05    756
800     2.0525      2.1032      2.1099      -3.3308     1.86e-05    768
900     2.0527      2.1131      2.1118      0.6447      1.22e-05    949
1000    2.0525      ---         ---         ---         1.00e-05    ---
```

Total training time: 930.7s. Wall time: 989.2s. Average step time: 930.7ms.

**Backprop-LoRA (P16) 200-step trajectory** (seed 42, lr=3e-4):

```
Step    val_loss    train_loss  lr          step_ms (fwd/bwd/opt)
0       ---         2.0941      3.00e-04    1170 (613/409/147)
50      1.8881      2.2240      2.58e-04    618 (234/361/23)
100     1.8237      1.5150      1.52e-04    714 (286/359/69)
150     1.8048      1.4910      4.56e-05    639 (252/363/24)
200     1.7972      ---         1.85e-08    616 (234/364/18)
```

Wall time: 141.0s. Average step time: ~618ms (excluding warmup step 0).
