# Hebbian LoRA: Activity-Based Learning for LLM Fine-tuning on NPU

**Date**: 2026-03-16
**Status**: DESIGN (speculative research direction)
**Priority**: Exploratory (high novelty, uncertain payoff)
**Estimated effort**: 2-3 days implementation, 1-2 days evaluation
**Depends on**: Existing LoRA infrastructure (config.h LoRALayer), ANE forward pass pipeline

---

## 0. Executive Summary

We propose **Hebbian LoRA**, a biologically-plausible fine-tuning method that derives LoRA weight updates from forward-pass activations alone, requiring no backward pass. This directly exploits the Apple Neural Engine's architecture: ANE executes forward passes at 2.5x CPU speed but cannot run backward passes at all. If Hebbian LoRA produces any learning signal, it would be the first local-learning-rule-based LoRA method for LLM fine-tuning, and the first training algorithm that runs *entirely* on an NPU without CPU backward fallback.

**Key question**: Can a local Hebbian rule applied to LoRA adapters produce a learning signal sufficient for fine-tuning, even if worse than backprop? Specifically, can it beat MeZO's quality ceiling of 0.026 nats improvement (val_loss 2.0455 from 2.0718 baseline)?

**Honest assessment**: Probably not. But the theoretical analysis and negative result would be valuable.

---

## 1. Background and Motivation

### 1.1 The ANE Training Problem

Apple's Neural Engine runs forward passes only. Our current training approaches:

| Method | Forward | Backward | Quality (val_loss delta) | ms/step |
|--------|---------|----------|--------------------------|---------|
| Backprop LoRA (CPU) | CPU | CPU | -0.147 nats (1.9248) | 586 |
| Backprop LoRA (P16 hybrid) | ANE | CPU | -0.275 nats (1.7972) | 618 |
| MeZO LoRA (CPU) | 2x CPU | None | -0.026 nats (2.0455) | 462 |
| MeZO LoRA (ANE) | 2x ANE | None | -0.019 nats (2.0524) | 262 |
| **Hebbian LoRA (proposed)** | **1x ANE** | **None** | **??? nats** | **~131** |

Hebbian LoRA would use a single forward pass (no perturbation pair needed, unlike MeZO's two), with weight updates derived from activation statistics. If it works at all, it would be the fastest method per step (~131ms, half of MeZO-ANE).

### 1.2 LoRA Architecture Recap

For SmolLM2-360M with rank-8 LoRA on attention projections:

```
W_eff = W_base + B @ A

Wq: A[8, 960],   B[960, 8]     ->  W_eff[960, 960]
Wk: A[8, 960],   B[320, 8]     ->  W_eff[320, 960]
Wv: A[8, 960],   B[320, 8]     ->  W_eff[320, 960]
Wo: A[8, 960],   B[960, 8]     ->  W_eff[960, 960]
```

Per layer: 4 x (8 x 960 + d_out x 8) = 4 x (7680 + d_out x 8) parameters.
Total across 32 layers: 1,638,400 LoRA params + 62,400 RMS norm params.

The forward pass computes, for each projection:
```
y = W_eff @ x = (W_base + B @ A) @ x = W_base @ x + B @ (A @ x)
```

Where:
- `x` = pre-activation (layer input after RMSNorm), shape [DIM, SEQ] = [960, 256]
- `y` = post-activation (Q, K, V, or attention output), shape [d_out, SEQ]

### 1.3 Hebbian Learning: First Principles

Hebb's postulate (1949): "When an axon of cell A is near enough to excite cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased."

Mathematically: **dW ~ y * x^T** (outer product of post-synaptic and pre-synaptic activity).

This is purely local: each synapse (weight) updates based only on the activities of the two neurons it connects. No global loss signal, no error propagation.

---

## 2. Hebbian Update Rules for LoRA

### 2.1 Naive Hebbian Rule (Unsupervised, Unstable)

For the projection `y = (W_base + B @ A) @ x`, define the LoRA correction as `delta_y = B @ A @ x = B @ z` where `z = A @ x` is the rank-8 intermediate representation.

**Naive Hebbian updates:**
```
dA = eta * z_post @ x^T        where z_post = B^T @ y     [8, SEQ] @ [SEQ, 960] -> [8, 960]
dB = eta * y @ z^T              where z = A @ x            [d_out, SEQ] @ [SEQ, 8] -> [d_out, 8]
```

This is the standard Hebbian outer product applied to the LoRA factorization. Matrix A learns to project inputs into the rank-8 subspace that correlates with the overall output y. Matrix B learns to project the rank-8 representation z back into output space in a way that correlates with y.

**Problem 1: No loss signal.** This rule strengthens whatever the network already does. It will amplify the dominant mode of the pre-trained network, not adapt it to new data. For fine-tuning, we need the update to move the network toward better predictions on the target task.

**Problem 2: Weight explosion.** Since dW ~ y * x^T, and larger weights produce larger y, which produces larger dW, there is positive feedback. Weights diverge exponentially: ||W(t)|| ~ exp(eta * lambda_max * t).

### 2.2 Incorporating the Loss Signal: Modulated Hebbian Learning

To make Hebbian learning supervised, we need a modulatory signal that encodes whether the current prediction is good or bad. Options:

**Option A: Surprise modulation.**
```
M = L(y_pred, y_true) - L_baseline
dA = eta * M * (B^T @ y) @ x^T
dB = eta * M * y @ (A @ x)^T
```

Where M is the "surprise" (excess loss above a running baseline). This is equivalent to REINFORCE with a baseline: the Hebbian update is modulated by the scalar reward signal. When loss is high (bad prediction), M > 0 and updates are positive (strengthening current behavior is wrong -- we need the sign to flip). Actually this requires careful sign analysis.

**Corrected surprise modulation:**
```
M = L_baseline - L(y_pred, y_true)    (positive = good, negative = bad)
dA = eta * M * (B^T @ y) @ x^T
dB = eta * M * y @ (A @ x)^T
```

When the model does well (M > 0), strengthen the current input-output mapping. When it does poorly (M < 0), weaken it. This is a form of reward-modulated Hebbian learning (R-STDP in neuroscience).

**Theoretical concern:** This has the same variance problem as REINFORCE. The scalar M modulates a rank-1 direction in parameter space, giving O(d) variance in the gradient estimate -- identical to MeZO's single-perturbation estimate. We would expect convergence no better than MeZO.

**Option B: Error-driven Hebbian (delta rule analog).**
```
e = y_target - y_pred              (requires knowing target activations)
dA = eta * (B^T @ e) @ x^T
dB = eta * e @ (A @ x)^T
```

This is just the gradient of ||y - y_target||^2 with respect to A and B. It IS backprop for a single linear layer -- but requires knowing y_target, which requires propagating the loss backward through subsequent layers. This defeats the purpose.

**Option C: Contrastive Hebbian (our proposed approach -- see Section 3).**

### 2.3 Deriving Updates for A and B Separately

For `y = W_base @ x + B @ A @ x`, define:
- `pre_A = x` (input to A), shape [DIM, SEQ]
- `post_A = z = A @ x` (output of A), shape [r, SEQ]
- `pre_B = z` (input to B), shape [r, SEQ]
- `post_B = B @ z` (the LoRA correction to y), shape [d_out, SEQ]

The Hebbian rule for each matrix individually:
```
dA = eta * post_A @ pre_A^T = eta * (A @ x) @ x^T           [r, DIM]
dB = eta * post_B @ pre_B^T = eta * (B @ A @ x) @ (A @ x)^T [d_out, r]
```

Note: dA depends on A (through post_A = A @ x) and dB depends on both A and B. This creates a coupled nonlinear dynamical system. The fixed points of this system correspond to the leading singular vectors of the input covariance matrix -- not to the loss-minimizing LoRA adapters.

**Key insight:** Unmodulated Hebbian learning on LoRA performs unsupervised dimensionality reduction (PCA) on the input activations, not task-specific adaptation. This is potentially useful for initialization but not for fine-tuning.

---

## 3. Contrastive Hebbian Learning for LoRA

### 3.1 The Contrastive Hebbian Principle

Contrastive Hebbian Learning (CHL) uses two phases:
1. **Positive phase**: Clamp the network to real data (input, correct target). Record activations.
2. **Negative phase**: Let the network free-run (or use corrupted/model-generated data). Record activations.
3. **Update**: dW = eta * (y_pos @ x_pos^T - y_neg @ x_neg^T)

The key theorem (Movellan 1991, Xie & Seung 2003): In a network at thermal equilibrium, CHL computes the exact gradient of the log-likelihood. For feedforward networks away from equilibrium, CHL approximates the gradient with a bias that depends on how far the positive and negative phases diverge.

### 3.2 Contrastive Hebbian LoRA Update Rule

For each LoRA-adapted projection in each layer:

**Positive phase** (real data): Run forward pass with real input tokens.
```
x_pos = layer_input_after_RMSNorm    [DIM, SEQ]
z_pos = A @ x_pos                     [r, SEQ]
y_pos = W_base @ x_pos + B @ z_pos    [d_out, SEQ]
```

**Negative phase** (corrupted data): Run forward pass with corrupted/model-generated tokens.
```
x_neg = layer_input_after_RMSNorm    [DIM, SEQ]
z_neg = A @ x_neg                     [r, SEQ]
y_neg = W_base @ x_neg + B @ z_neg    [d_out, SEQ]
```

**Update rules:**
```
dA = eta * (B^T @ y_pos @ x_pos^T - B^T @ y_neg @ x_neg^T) / SEQ      [r, DIM]
dB = eta * (y_pos @ z_pos^T - y_neg @ z_neg^T) / SEQ                    [d_out, r]
```

Or equivalently, using only the LoRA corrections (subtracting the base model contribution):
```
delta_y_pos = B @ z_pos    (LoRA correction, positive phase)
delta_y_neg = B @ z_neg    (LoRA correction, negative phase)

dA = eta * (B^T @ delta_y_pos @ x_pos^T - B^T @ delta_y_neg @ x_neg^T) / SEQ
dB = eta * (delta_y_pos @ z_pos^T - delta_y_neg @ z_neg^T) / SEQ
```

### 3.3 Derivation and Relationship to Gradient

Consider the contrastive loss:
```
L_contrastive = -log p(x_pos) + log p(x_neg)
```

For an energy-based model E(x) = -sum_l ||y_l||^2 (goodness function, as in Hinton's Forward-Forward), the gradient with respect to parameters in layer l is:

```
dL/dW_l = -(y_pos @ x_pos^T) + (y_neg @ x_neg^T)
```

This is exactly the negative of our CHL update. So **CHL with goodness-based energy minimizes the contrastive loss**, which is related to (but not identical to) the cross-entropy loss we use for language modeling.

**Critical gap:** The contrastive loss optimizes for discrimination between real and corrupted data, not for next-token prediction. The quality of learning depends entirely on how well the negative examples probe the model's weaknesses. This is analogous to how GAN training depends on generator quality.

### 3.4 Generating Negative Examples

For language modeling, negative examples must be "close enough" to real data that distinguishing them requires genuine language understanding. Options:

**Option A: Token corruption.** Replace a random fraction (e.g., 15%) of input tokens with random tokens from the vocabulary. Cheap, but may be too easy to distinguish (model barely changes activations).

**Option B: Model-generated continuations.** Use the model's own greedy/sampled predictions as negative examples. Run forward pass on real prefix, sample next tokens autoregressively, then use the sampled sequence as the negative phase. Expensive (requires autoregressive generation).

**Option C: Shuffled sequences.** Shuffle token order within the sequence. Destroys positional coherence while preserving token statistics.

**Option D: Adjacent-batch negatives.** Use a different training example as the negative. Simplest implementation but the "negative" is still valid text, just unrelated to the positive.

**Recommended: Option A (token corruption)** for initial experiments. It requires one additional forward pass with corrupted input (total: 2 forward passes, same as MeZO), is trivially parallelizable, and the corruption rate is a tunable hyperparameter.

### 3.5 Full Contrastive Hebbian LoRA Algorithm

```
For each training step:
  1. Sample batch x_real (256 tokens)
  2. Corrupt: x_corrupt = corrupt(x_real, corruption_rate=0.15)
  3. Forward pass on x_real:
     - For each layer l = 0..31:
       - Save x_pos[l] = post-RMSNorm input
       - Compute z_pos[l] = A[l] @ x_pos[l]
       - Compute y_pos[l] = full projection output (Q/K/V/Wo)
  4. Forward pass on x_corrupt:
     - For each layer l = 0..31:
       - Save x_neg[l] = post-RMSNorm input
       - Compute z_neg[l] = A[l] @ x_neg[l]
       - Compute y_neg[l] = full projection output
  5. For each layer l, for each projection p in {q,k,v,o}:
     - dA[l,p] = eta * (B[l,p]^T @ y_pos[l,p] @ x_pos[l]^T
                       - B[l,p]^T @ y_neg[l,p] @ x_neg[l]^T) / SEQ
     - dB[l,p] = eta * (y_pos[l,p] @ z_pos[l,p]^T
                       - y_neg[l,p] @ z_neg[l,p]^T) / SEQ
     - Apply Oja stabilization (Section 4)
     - A[l,p] += dA[l,p]
     - B[l,p] += dB[l,p]
  6. Re-merge: W_eff[l,p] = W_base[l,p] + B[l,p] @ A[l,p]
```

### 3.6 Why Contrastive Hebbian Might (or Might Not) Work for LLMs

**Arguments for:**
- CHL is proven equivalent to gradient descent at equilibrium in Boltzmann machines (Ackley et al. 1985).
- The contrastive signal encodes "real text should have higher activation norms than corrupted text," which is a meaningful linguistic signal.
- Each layer can update independently, enabling full pipeline parallelism.
- Two forward passes (same as MeZO), but the update uses richer per-layer information rather than a single scalar gradient estimate.

**Arguments against:**
- Transformers are feedforward, not recurrent/equilibrium networks. The CHL-gradient equivalence breaks down.
- The contrastive objective (discriminate real vs. corrupted) is not the same as the LM objective (predict next token). The LoRA adapters will learn to distinguish real from corrupted text, which is a different (easier) task.
- In feedforward networks, CHL accumulates approximation error across layers. With 32 layers, the top-layer signal is very different from the bottom-layer signal, but all layers apply the same rule.
- Token corruption may be too easy: a pre-trained LM can already trivially distinguish real from 15%-corrupted text, so the contrastive signal saturates quickly.

---

## 4. Anti-Hebbian Stabilization: Oja's Rule for LoRA

### 4.1 The Weight Explosion Problem

Standard Hebbian learning is unstable because it has a positive feedback loop:
```
||W(t+1)||^2 = ||W(t) + eta * y @ x^T||^2
             = ||W(t)||^2 + 2*eta * tr(W(t)^T @ y @ x^T) + O(eta^2)
             = ||W(t)||^2 + 2*eta * y^T @ W(t) @ x * ||x||^2 / SEQ + ...
```

Since y = W @ x, the trace term is proportional to ||W||^2 * ||x||^4, which is positive definite. Hence ||W|| grows exponentially.

### 4.2 Oja's Rule

Oja (1982) proposed a self-normalizing modification:
```
dW = eta * (y @ x^T - diag(y @ y^T) @ W)
```

For a single neuron (w a vector), this becomes:
```
dw = eta * (y*x - y^2*w)    where y = w^T @ x
```

**Stability proof (sketch):**
Define the Lyapunov function V(w) = (1 - ||w||^2)^2.
```
dV/dt = -4*(1 - ||w||^2) * w^T * dw
      = -4*(1 - ||w||^2) * w^T * eta*(y*x - y^2*w)
      = -4*eta*(1 - ||w||^2) * (y * w^T*x - y^2 * ||w||^2)
      = -4*eta*(1 - ||w||^2) * y^2 * (1 - ||w||^2)
      = -4*eta*y^2*(1 - ||w||^2)^2
      = -4*eta*y^2*V(w) <= 0
```

So V is a Lyapunov function: it decreases monotonically, and V = 0 iff ||w|| = 1. The weight vector converges to unit norm. Moreover, it converges to the principal eigenvector of the input covariance matrix E[x @ x^T].

### 4.3 Oja's Rule Applied to LoRA Matrices

For LoRA matrix A [r, DIM]:
```
z = A @ x                                          [r, SEQ]
dA_hebb = z @ x^T / SEQ                            [r, DIM]
dA_oja  = eta * (dA_hebb - diag(z @ z^T / SEQ) @ A)
        = eta * (z @ x^T / SEQ - (1/SEQ * sum_t z_t*z_t^T) @ A)
```

The second term `(z @ z^T / SEQ) @ A` is the anti-Hebbian decay. It acts as a soft weight normalization: rows of A that produce large outputs (large ||z_i||) are penalized more.

For LoRA matrix B [d_out, r]:
```
delta_y = B @ z                                     [d_out, SEQ]
dB_hebb = delta_y @ z^T / SEQ                       [d_out, r]
dB_oja  = eta * (dB_hebb - diag(delta_y @ delta_y^T / SEQ) @ B)
```

**Stability guarantee:** Each row of A converges to a unit vector aligned with a principal component of the input covariance. Each row of B converges to a unit vector aligned with a principal component of the intermediate representation covariance. The norms are bounded: ||A_row_i|| -> 1, ||B_row_j|| -> 1.

### 4.4 Contrastive Oja's Rule for LoRA (Combined)

Combining Sections 3 and 4, the full stabilized contrastive Hebbian update is:

```
// Positive phase activations
z_pos = A @ x_pos
y_pos = full_projection(x_pos)    // includes base + LoRA

// Negative phase activations
z_neg = A @ x_neg
y_neg = full_projection(x_neg)

// Contrastive Hebbian term
H_A = (B^T @ y_pos @ x_pos^T - B^T @ y_neg @ x_neg^T) / SEQ
H_B = (y_pos @ z_pos^T - y_neg @ z_neg^T) / SEQ

// Oja decay (computed from positive phase only)
z_cov = z_pos @ z_pos^T / SEQ           [r, r]
dy_cov = (B @ z_pos) @ (B @ z_pos)^T / SEQ   [d_out, d_out]

// Combined update
dA = eta * (H_A - alpha_oja * diag(z_cov) @ A)
dB = eta * (H_B - alpha_oja * diag(dy_cov) @ B)
```

Where `alpha_oja` controls the strength of the anti-Hebbian term (default: 1.0).

### 4.5 Additional Stabilization: Weight Clipping and EMA

Even with Oja's rule, transient instabilities may occur in the coupled A-B system. Additional safeguards:

1. **Hard weight clipping:** `A = clip(A, -clip_val, clip_val)`, with clip_val = 1.0.
2. **EMA smoothing of updates:** `dA_smooth = beta * dA_smooth_prev + (1-beta) * dA`, with beta = 0.9.
3. **Learning rate warmup:** Start with eta_0 = 0 and linearly increase to eta_max over 50 steps.
4. **Per-row norm monitoring:** If any row of A or B exceeds norm 5.0, scale it back to 1.0.

---

## 5. Per-Layer Local Learning

### 5.1 Locality Property

The key property that makes Hebbian LoRA compatible with forward-only hardware: **each layer's update depends only on that layer's own activations.**

For layer l, the update to (A_l, B_l) requires:
- `x_pos[l]`, `x_neg[l]`: pre-activation inputs (available from forward pass)
- `y_pos[l]`, `y_neg[l]`: post-activation outputs (available from forward pass)
- `A_l`, `B_l`: the layer's own LoRA weights
- No information from any other layer

This means:
1. No gradient chain through layers (no backward pass)
2. No cross-layer communication needed during the update
3. Each layer can update immediately after its forward pass completes
4. In principle, updates could be pipelined with subsequent layer computations

### 5.2 Comparison to Backpropagation

| Property | Backpropagation | Contrastive Hebbian LoRA |
|----------|----------------|--------------------------|
| Passes required | 1 forward + 1 backward | 2 forward (pos + neg) |
| Cross-layer communication | Full gradient chain | None |
| Loss signal | Exact gradient | Contrastive discrimination |
| Update timing | After full backward sweep | Immediately per layer |
| Memory per layer | Activations + gradients | 2x activations (pos + neg) |
| Parallelism | Sequential (backward chain) | Fully parallel per layer |
| Convergence rate | O(1/sqrt(T)) SGD | Unknown (likely much slower) |
| Quality ceiling | Limited by model capacity | Limited by contrastive signal quality |

### 5.3 The Credit Assignment Problem

In backpropagation, the loss signal is propagated exactly through the chain rule. Layer l receives gradient information about how its output affects the final loss, through all subsequent layers.

In local Hebbian learning, layer l only knows: "did I see different patterns for real vs. corrupted data?" This local signal is a noisy, biased estimate of the global gradient. The bias comes from:

1. **Missing higher-order effects:** Layer l's change affects layers l+1..31, which is not accounted for.
2. **Objective mismatch:** The contrastive discrimination objective differs from next-token prediction.
3. **Phase interference:** In the negative phase, all layers process corrupted input, so the "negative" activations at layer l are affected by corruption propagated through layers 0..l-1.

This is the fundamental limitation of local learning rules. Theoretical results (Bartunov et al. 2018) show that local learning rules achieve ~60-80% of backprop accuracy on image classification. For language modeling, where long-range dependencies are critical, the gap is likely larger.

---

## 6. Implementation

### 6.1 Available Activations from Forward Pass

Our existing forward pass (in train_mezo.m and the ANE conv-fused pipeline) already computes these per-layer activations:

| Activation | Shape | Use in Hebbian update |
|------------|-------|----------------------|
| `xnorm` (post-RMSNorm) | [DIM, SEQ] = [960, 256] | Input to all projections (x in update rule) |
| `Q` (query) | [Q_DIM, SEQ] = [960, 256] | Post-activation for Wq (y for Wq update) |
| `K` (key) | [KV_DIM, SEQ] = [320, 256] | Post-activation for Wk (y for Wk update) |
| `V` (value) | [KV_DIM, SEQ] = [320, 256] | Post-activation for Wv (y for Wv update) |
| `attn_out` (pre-Wo) | [Q_DIM, SEQ] = [960, 256] | Input to Wo (x for Wo update) |
| `o_out` (post-Wo) | [DIM, SEQ] = [960, 256] | Post-activation for Wo (y for Wo update) |

All activations needed for Hebbian updates are already available. No new tensor computations are required beyond the update rule itself.

### 6.2 Per-Step Computation (Positive + Negative Phase)

**Forward pass (positive phase):** ~131ms on ANE (existing pipeline).

Activations to save per layer: xnorm + Q + K + V + attn_out + o_out = (960 + 960 + 320 + 320 + 960 + 960) x 256 x 4 bytes = 4,480 x 256 x 4 = ~4.4MB per layer.
Total across 32 layers: **~141MB** for positive phase.

**Forward pass (negative phase):** ~131ms on ANE.
Same activation storage: **~141MB** for negative phase.

**Hebbian update computation (CPU):** For each layer, for each projection:

For Wq (A[8,960], B[960,8]):
```
z_pos = A @ x_pos                    [8, 256]     = 8*960*256 = ~2M FLOPs
B_T_y_pos = B^T @ y_pos             [8, 256]     = 8*960*256 = ~2M FLOPs
H_A = B_T_y_pos @ x_pos^T           [8, 960]     = 8*256*960 = ~2M FLOPs
(same for negative phase)
Oja_decay = diag(z_cov) @ A          [8, 960]     = 8*960     = ~8K FLOPs
```

Total per projection: ~12M FLOPs (pos + neg + Oja).
4 projections per layer: ~48M FLOPs.
32 layers: **~1.54G FLOPs**.

At M2 Pro CPU Accelerate (~2.5 TFLOPS fp32): **~0.6ms**.

The update computation is negligible compared to the forward passes.

### 6.3 Total Step Time Estimate

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| Forward pass (positive, ANE) | 131 | Existing conv-fused pipeline |
| Forward pass (negative, ANE) | 131 | Same pipeline, corrupted input |
| Hebbian update computation (CPU) | 1 | 1.54 GFLOP at 2.5 TFLOPS |
| LoRA merge (CPU) | 2 | Re-merge B@A into W_eff |
| ANE weight re-staging | 14 | Re-transpose for conv-fused kernels |
| **Total** | **~279ms** | |

Compare: MeZO-ANE = 262ms/step. Hebbian LoRA would be similar speed per step, but uses a richer per-layer signal rather than a single scalar gradient estimate.

### 6.4 Memory Overhead

| Component | Size | Notes |
|-----------|------|-------|
| Base model weights | 1,380 MB | Frozen, shared |
| LoRA A/B matrices | 12.5 MB | 1,638,400 params x 4 bytes x 2 (A+B) |
| Positive activations (all layers) | 141 MB | Could be computed on-the-fly |
| Negative activations (all layers) | 141 MB | Could be computed on-the-fly |
| Update accumulators (dA, dB) | 12.5 MB | Same size as LoRA params |
| EMA smoothing buffers | 12.5 MB | Optional |
| **Total additional** | **~320 MB** | Beyond inference baseline of 1,457 MB |

Memory-optimized variant: Compute updates layer-by-layer during the forward pass, discarding activations after the update. This reduces positive/negative activation storage from 282MB to ~9MB (single layer pair). **Total additional: ~47 MB.**

### 6.5 Implementation in C (Sketch)

```c
// hebbian_lora_update() — called once per step, after both forward passes complete
//
// For each layer and each projection, computes:
//   dA = eta * (B^T @ y_pos @ x_pos^T - B^T @ y_neg @ x_neg^T) / SEQ
//         - alpha_oja * diag(z_cov) @ A
//   dB = eta * (y_pos @ z_pos^T - y_neg @ z_neg^T) / SEQ
//         - alpha_oja * diag(dy_cov) @ B

static void hebbian_lora_update_projection(
    float *A, float *B,                  // LoRA matrices [r, d_in] and [d_out, r]
    const float *x_pos, const float *y_pos,  // positive phase [d_in, SEQ], [d_out, SEQ]
    const float *x_neg, const float *y_neg,  // negative phase
    int d_in, int d_out, int r,
    float eta, float alpha_oja)
{
    // Intermediate: z_pos = A @ x_pos [r, SEQ]
    float *z_pos = (float*)safe_malloc(r * SEQ * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                r, SEQ, d_in, 1.0f, A, d_in, x_pos, SEQ, 0.0f, z_pos, SEQ);

    // B^T @ y_pos [r, SEQ]
    float *Bt_y_pos = (float*)safe_malloc(r * SEQ * 4);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                r, SEQ, d_out, 1.0f, B, r, y_pos, SEQ, 0.0f, Bt_y_pos, SEQ);

    // H_A_pos = Bt_y_pos @ x_pos^T [r, d_in]
    float *H_A = (float*)safe_calloc(r * d_in, 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                r, d_in, SEQ, 1.0f/SEQ, Bt_y_pos, SEQ, x_pos, SEQ, 0.0f, H_A, d_in);

    // H_B_pos = y_pos @ z_pos^T [d_out, r]
    float *H_B = (float*)safe_calloc(d_out * r, 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                d_out, r, SEQ, 1.0f/SEQ, y_pos, SEQ, z_pos, SEQ, 0.0f, H_B, r);

    // Subtract negative phase (same computation with x_neg, y_neg)
    float *z_neg = (float*)safe_malloc(r * SEQ * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                r, SEQ, d_in, 1.0f, A, d_in, x_neg, SEQ, 0.0f, z_neg, SEQ);

    float *Bt_y_neg = (float*)safe_malloc(r * SEQ * 4);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                r, SEQ, d_out, 1.0f, B, r, y_neg, SEQ, 0.0f, Bt_y_neg, SEQ);

    // H_A -= Bt_y_neg @ x_neg^T (subtract negative phase)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                r, d_in, SEQ, -1.0f/SEQ, Bt_y_neg, SEQ, x_neg, SEQ, 1.0f, H_A, d_in);

    // H_B -= y_neg @ z_neg^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                d_out, r, SEQ, -1.0f/SEQ, y_neg, SEQ, z_neg, SEQ, 1.0f, H_B, r);

    // Oja decay for A: diag(z_pos @ z_pos^T / SEQ) @ A
    // z_cov_diag[i] = sum_t z_pos[i,t]^2 / SEQ
    for (int i = 0; i < r; i++) {
        float z_sq_sum = 0;
        for (int t = 0; t < SEQ; t++) z_sq_sum += z_pos[i*SEQ+t] * z_pos[i*SEQ+t];
        z_sq_sum /= SEQ;
        for (int j = 0; j < d_in; j++) {
            H_A[i*d_in+j] -= alpha_oja * z_sq_sum * A[i*d_in+j];
        }
    }

    // Oja decay for B: diag(delta_y @ delta_y^T / SEQ) @ B
    // delta_y_pos = B @ z_pos [d_out, SEQ]
    float *dy_pos = (float*)safe_malloc(d_out * SEQ * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                d_out, SEQ, r, 1.0f, B, r, z_pos, SEQ, 0.0f, dy_pos, SEQ);
    for (int i = 0; i < d_out; i++) {
        float dy_sq_sum = 0;
        for (int t = 0; t < SEQ; t++) dy_sq_sum += dy_pos[i*SEQ+t] * dy_pos[i*SEQ+t];
        dy_sq_sum /= SEQ;
        for (int j = 0; j < r; j++) {
            H_B[i*r+j] -= alpha_oja * dy_sq_sum * B[i*r+j];
        }
    }

    // Apply updates
    for (int i = 0; i < r * d_in; i++) A[i] += eta * H_A[i];
    for (int i = 0; i < d_out * r; i++) B[i] += eta * H_B[i];

    // Cleanup
    free(z_pos); free(z_neg); free(Bt_y_pos); free(Bt_y_neg);
    free(H_A); free(H_B); free(dy_pos);
}
```

### 6.6 Integration with Existing Pipeline

The implementation plugs into the existing `train_mezo.m` architecture:

1. **Positive forward pass**: Identical to current MeZO forward pass, but save per-layer activations (xnorm, Q, K, V, attn_out, o_out) -- same tensors already saved in backprop_lora.h's BP_LayerActs.
2. **Corruption**: On CPU, corrupt the input token sequence (replace 15% with random tokens).
3. **Negative forward pass**: Identical forward pass with corrupted input.
4. **Hebbian update**: Call `hebbian_lora_update_projection()` for each layer/projection.
5. **LoRA re-merge**: Call existing `lora_merge_weight()` for each adapted projection.
6. **Weight re-staging**: Existing `retranspose_all_weights()` for ANE.

No new ANE kernels needed. No backward pass kernels needed.

---

## 7. Testable Hypotheses

### Hypothesis H1: Contrastive Hebbian LoRA Produces a Non-Zero Learning Signal

**Prediction:** After 200 steps of Contrastive Hebbian LoRA, val_loss will decrease by at least 0.001 nats from baseline (2.0718). This is 20x less than MeZO's 0.019 nats improvement, but would confirm that the contrastive Hebbian signal carries *some* task-relevant information.

**Test:** Run CHL LoRA for 200 steps (2x forward pass each, ~56s on ANE), measure val_loss. Compare to baseline val_loss (no training).

**Null hypothesis:** CHL LoRA produces zero learning signal (val_loss change < 0.001, i.e., within noise). This would indicate the contrastive discrimination objective does not transfer to next-token prediction, and local learning rules cannot fine-tune LLMs.

**Success criterion:** val_loss < 2.070 (> 0.001 nats improvement).

### Hypothesis H2: Oja Stabilization Prevents Weight Explosion

**Prediction:** Without Oja's rule, LoRA weight norms will grow exponentially, reaching ||A|| > 100 within 50 steps. With Oja's rule (alpha_oja=1.0), weight norms will remain bounded (||A_row|| converges to ~1.0).

**Test:** Run CHL LoRA for 100 steps with and without Oja's rule. Monitor ||A||_F and ||B||_F per layer every 10 steps.

**Success criterion:** With Oja, max(||A_row||) < 5.0 at step 100. Without Oja, max(||A_row||) > 50.0.

### Hypothesis H3: Hebbian LoRA Quality is Bounded Below MeZO but Above Random

**Prediction:** Contrastive Hebbian LoRA will achieve at most 0.01 nats improvement (less than half of MeZO's 0.026 nats from condition 26, 600s). However, it will be strictly better than zero, placing it between "no training" and "MeZO" on the quality spectrum.

**Rationale:** The contrastive signal is richer per-step than MeZO's scalar gradient estimate (it uses per-layer, per-neuron correlation information). But the contrastive objective is misaligned with next-token prediction, introducing systematic bias that limits asymptotic quality.

**Test:** Run CHL LoRA for 600s (same budget as condition 26). Compare val_loss to:
- Baseline (no training): 2.0718
- MeZO LoRA (600s): 2.0455 (improvement: 0.0263 nats)
- Backprop LoRA (200 steps): 1.7972 (improvement: 0.2746 nats)

---

## 8. Expected Quality Analysis

### 8.1 Theoretical Quality Ordering

From best to worst:
```
Backprop LoRA >> MeZO LoRA >> Hebbian LoRA >> No training
   -0.275         -0.026        -0.005(?)        0.000
```

**Backprop LoRA** computes the exact gradient of the cross-entropy loss. It has no bias, only variance from mini-batch sampling. Converges at O(1/sqrt(T)).

**MeZO LoRA** estimates the gradient as a scalar projection onto a random direction. Unbiased but O(d) variance. Converges at O(d/sqrt(T)), where d = 1.7M for our LoRA setup. Effectively needs d/r times more steps than backprop, where r is the "effective rank" of the gradient (empirically ~10x for our setup).

**Hebbian LoRA** uses a different objective (contrastive discrimination vs. next-token prediction). The gradient it computes is exact for the contrastive objective but biased for the LM objective. The bias is:

```
E[dW_CHL] = nabla_W L_contrastive =/= nabla_W L_CE
```

The relationship between L_contrastive and L_CE depends on the quality of negative examples. In the limit where negative examples are drawn from the model's own distribution, L_contrastive approaches the log-likelihood gradient (Hinton 2002). But with simple corruption, the contrastive signal saturates once the model can easily distinguish corrupted from real text.

### 8.2 Why Hebbian LoRA is Probably Worse Than MeZO

MeZO, despite its high variance, optimizes the correct objective (cross-entropy). Its gradient estimate is unbiased: E[g_MeZO] = nabla L_CE.

Hebbian LoRA optimizes the wrong objective. Even with perfect optimization of L_contrastive, the resulting LoRA adapters are tuned for discrimination, not generation. This is analogous to training a classifier when you want a generator.

However, discrimination and generation are related: a model that is better at predicting next tokens is also better at distinguishing real from shuffled text. The question is how tight this coupling is.

### 8.3 Comparison to MeZO Quality Ceiling

Our best MeZO result (condition 26, 600s, lr=1e-4):
- val_loss: 2.0455 (improvement: 0.0263 nats from 2.0718)
- Steps: 1150
- MeZO appears to saturate around 0.026 nats; additional steps do not improve val_loss

**Hebbian LoRA prediction**: 0.002-0.010 nats improvement (conservatively 10-40% of MeZO).

**Can it beat MeZO?** Almost certainly not in val_loss. MeZO optimizes the right objective; Hebbian optimizes the wrong one. The only scenario where Hebbian could win: if the per-step information gain from CHL's richer per-layer signal outweighs MeZO's correct-objective advantage by >10x. This seems unlikely given that CHL's signal saturates.

### 8.4 Is it Better Than MeZO's Quality Ceiling of 0.019 nats?

Referring to condition 22 (MeZO LoRA split, ANE, 300s): best val_loss = 2.0524, improvement = 0.019 nats.

**Hebbian LoRA is unlikely to beat 0.019 nats.** The contrastive objective mismatch introduces systematic bias that MeZO does not have. Even with infinite steps, CHL converges to a solution that is optimal for discrimination, which is a different point in parameter space than the LM-optimal point.

**The honest answer:** Hebbian LoRA will almost certainly be worse than MeZO in absolute quality. Its value proposition is purely scientific: testing whether local learning rules carry *any* signal for LLM fine-tuning.

---

## 9. Risks and Mitigations

### Risk 1: Weight Explosion (HIGH)

**Mechanism:** Positive feedback in Hebbian updates causes exponential weight growth.
**Likelihood:** Near-certain without stabilization. Transient instabilities likely even with Oja.
**Detection:** Monitor ||A||_F, ||B||_F per layer every step. If any exceeds 10.0, halt.
**Mitigation:** Oja's rule (Section 4), hard clipping, per-row norm capping.
**Residual risk after mitigation:** MEDIUM. Oja stabilizes single-matrix updates but the coupled A-B system may still have unstable modes.

### Risk 2: Mode Collapse (HIGH)

**Mechanism:** All LoRA adapters converge to amplify the same dominant activation pattern. A matrices across all layers learn the same principal component of their input.
**Likelihood:** High, because Hebbian learning is an unsupervised dimensionality reduction that finds dominant modes.
**Detection:** Compute cosine similarity between A matrices across layers. If cos(A_l, A_m) > 0.9 for l != m, mode collapse is occurring.
**Mitigation:** Add decorrelation penalty across layers, or use different random seeds per layer.

### Risk 3: Contrastive Signal Saturation (HIGH)

**Mechanism:** The pre-trained LM can already easily distinguish real from corrupted text, so the contrastive signal has near-zero magnitude from the start.
**Likelihood:** Very high for simple corruption (random token replacement). The 360M-parameter model has already learned robust language representations.
**Detection:** Compute ||y_pos - y_neg|| per layer. If < 0.01 * ||y_pos|| at step 0, signal is saturated.
**Mitigation:** Use harder negatives (model-generated text, adversarial corruption, graduated difficulty). Increase corruption rate to 30-50%. Use token deletion/insertion rather than replacement.

### Risk 4: Divergence from Coupled A-B Updates (MEDIUM)

**Mechanism:** Updating A and B simultaneously with Hebbian rules creates a nonlinear coupled system. A's update depends on B (through B^T @ y) and B's update depends on A (through A @ x). The joint system may oscillate or diverge even when each individual update is stable.
**Likelihood:** Medium. Theory (Oja 1989) only guarantees single-matrix stability.
**Mitigation:** Alternating updates: update A on even steps, B on odd steps. Or use a very small learning rate (eta < 1e-6).

### Risk 5: Zero Learning Signal (HIGH)

**Mechanism:** The contrastive Hebbian signal, after Oja stabilization, may contain zero information about the language modeling task. The updates drive LoRA adapters toward principal components of the input distribution, which the pre-trained model already captures.
**Likelihood:** This is the most likely outcome. Local learning rules have never been shown to fine-tune a pre-trained LLM.
**Detection:** val_loss does not decrease below 2.070 after 500 steps.
**Mitigation:** This IS the research question. A negative result is informative: it would establish that local contrastive Hebbian rules are insufficient for LM fine-tuning, providing a lower bound on what forward-only learning can achieve.

### Risk 6: Implementation Bugs Masquerading as Negative Results (MEDIUM)

**Mechanism:** Subtle sign errors, transposition bugs, or scaling issues in the Hebbian update cause zero or destructive learning.
**Mitigation:** Validate on a tiny synthetic task first (2-layer MLP, learn XOR with contrastive Hebbian). If CHL cannot learn XOR, the implementation is buggy. Additionally, verify that the contrastive signal has non-zero norm and the correct sign pattern.

---

## 10. Literature Review

### 10.1 Directly Related Work

**LoRA-HeRO (ECAI 2024):** "Going Beyond LoRA Fine-Tuning with Hebb Learning" (IOS Press, DOI 10.3233/FAIA251089). The closest prior work. Combines Hebbian updates with LoRA for vision-language models (InternVL-1B, StableDiffusion). Reports 48-50% fine-tuning speedup with lossless quality. Key difference: LoRA-HeRO uses Hebbian learning to *accelerate* LoRA (replacing some backprop steps), not to *replace* backprop entirely. They still use backprop for a subset of layers identified by weight importance analysis. Our proposal eliminates backprop completely, which is necessary for NPU-only training but likely worse in quality.

**Plastic Transformers with Hebbian Rules (arXiv:2510.21908, Oct 2025):** Augments transformers with online plasticity using either neuromodulated Hebbian rules or gradient-based plasticity. Finds Hebbian updates "dominate in structured, sparse-supervision settings" while gradient-based plasticity is better for long-range credit assignment. Language modeling requires long-range credit assignment, which is a negative signal for our Hebbian approach.

### 10.2 Contrastive Hebbian Learning Theory

**Movellan (1991), "Contrastive Hebbian Learning in the Continuous Hopfield Model":** Proved that CHL computes the gradient of the log-likelihood in Boltzmann machines at thermal equilibrium. This is the theoretical foundation for our approach, but the equivalence breaks down in feedforward networks.

**Xie & Seung (2003), "Equivalence of backpropagation and contrastive Hebbian learning in a layered network" (Neural Computation, DOI 10.1162/089976603762552988):** Showed that in a specific class of layered networks with symmetric feedback, CHL converges to the same solution as backprop. Requires symmetric weights (W_forward = W_backward^T), which does not hold in transformers.

**Hoier et al. (ICML 2024), "Two Tales of Single-Phase Contrastive Hebbian Learning" (PMLR v235):** Demonstrates single-phase CHL (no separate positive/negative phases) that approaches backprop quality. Uses "dual propagation" with oppositely-nudged compartments. Achieves near-backprop accuracy on CIFAR-10 with MLPs. Transformers not tested.

**Detorakis & Bartley (2018), "Contrastive Hebbian Learning with Random Feedback Weights" (arXiv:1806.07406):** Shows CHL works even with random (not learned) feedback pathways, increasing biological plausibility. This is encouraging for our approach since we do not have learned feedback weights.

### 10.3 Forward-Forward and Forward-Only Methods

**Hinton (2022), "The Forward-Forward Algorithm" (arXiv:2212.13345):** Uses goodness (sum of squared activations) as a per-layer objective. Positive data should produce high goodness, negative data low goodness. Tested on MNIST/CIFAR-10, not on language models. Accuracy gap vs. backprop: ~5-15% on image tasks.

**Self-Contrastive Forward-Forward (Nature Communications, 2025):** Improves FF with contrastive learning, achieving 80.75% on CIFAR-10 (vs. 94% backprop). Still a significant gap, and language modeling is harder than classification.

**NoProp (arXiv:2503.24322, March 2025):** Training without full forward or backward propagation. Each block independently learns via local denoising objectives. Tested on MLPs; transformer/LLM results not reported.

### 10.4 Zeroth-Order Methods (Our Baseline)

**MeZO (NeurIPS 2023):** Our current best forward-only method. Uses SPSA gradient estimate. Quality ceiling on our setup: -0.026 nats improvement (1150 steps, 600s).

**MeBP (arXiv:2510.03425, Apple, Oct 2025):** Memory-efficient backprop for on-device training. Our P16 hybrid design is based on this. Achieves backprop quality with reduced memory.

### 10.5 Oja's Rule and Stability

**Oja (1982), "Simplified neuron model as a principal component analyzer":** Original paper. Weight converges to first principal component with bounded norm.

**Oja (2008), "Oja learning rule" (Scholarpedia):** Comprehensive review. Convergence proof via Lyapunov analysis. Global convergence guaranteed for single-neuron case; multi-neuron case requires additional assumptions (non-degenerate eigenvalues).

**arXiv:2408.08408 (Aug 2024), "Oja's plasticity rule overcomes challenges of training neural networks under biological constraints":** Recent paper showing Oja's rule can train networks that satisfy biological constraints (Dale's law, non-negative weights). Supports viability of Oja-stabilized updates in neural network training.

### 10.6 Quality Gap: Local Learning vs. Backprop

**SoftHebb (NeurIPS 2022):** Best Hebbian method for image classification. CIFAR-10: 80.3% (vs. 94% backprop). ImageNet top-1: 27.3% (vs. 76%+ backprop). Gap: 14-49 percentage points depending on task complexity.

**Bartunov et al. (NeurIPS 2018), "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures":** Local learning rules achieve 60-80% of backprop quality on image tasks, degrading on harder tasks with longer-range dependencies.

**Implication for LLM fine-tuning:** Language modeling has the longest-range dependencies of any standard ML task (context windows of 256+ tokens). If local rules lose 20-40% quality on image tasks, they likely lose >50% on language modeling. This predicts Hebbian LoRA at <50% of MeZO quality, i.e., <0.013 nats improvement.

---

## 11. Variants and Extensions

### 11.1 Reward-Modulated Hebbian (R-Hebb)

Instead of contrastive phases, modulate the Hebbian update by a scalar reward signal:
```
R = L_baseline - L(current_batch)     (positive = model did well)
dA = eta * R * (B^T @ y) @ x^T / SEQ
dB = eta * R * y @ (A @ x)^T / SEQ
```

This requires computing the loss (one forward pass + cross-entropy on CPU), but the weight update is still purely local and forward-only. It is equivalent to REINFORCE applied to each layer's Hebbian update.

**Advantage:** Directly optimizes the LM loss (correct objective).
**Disadvantage:** Same O(d) variance as MeZO (single scalar modulating d-dimensional update). Expected quality: similar to MeZO.

### 11.2 Predictive Coding / Target Propagation Hybrid

Each layer predicts its own target from the output of the next layer (top-down connections). The Hebbian update is modulated by the prediction error:
```
e_l = y_l - target_l     (prediction error at layer l)
dA = eta * (B^T @ e_l) @ x^T / SEQ
```

This requires additional top-down connections (not present in standard transformers) and is closer to predictive coding than pure Hebbian learning. More complex to implement but theoretically stronger.

### 11.3 Layer-Wise Goodness Maximization (Forward-Forward)

Maximize "goodness" (sum of squared activations) on positive data and minimize on negative data, per layer:
```
G_l = sum_t ||y_l[:,t]||^2 / SEQ     (goodness of layer l)
dA = eta * sign(pos_or_neg) * 2 * B^T @ (y @ x^T) / SEQ
```

This is Hinton's Forward-Forward algorithm applied to LoRA. Simpler than full CHL but weaker signal.

---

## 12. Experimental Plan

### Phase 1: Synthetic Validation (1 day)

1. Implement CHL LoRA update function in C.
2. Test on synthetic task: 2-layer MLP with LoRA, learn to classify positive/negative patterns.
3. Verify: (a) weights stay bounded with Oja, (b) weight explosion without Oja, (c) loss decreases.

### Phase 2: SmolLM2-360M Evaluation (1 day)

1. Integrate into train_mezo.m as `--mode hebbian-lora`.
2. Run Contrastive Hebbian LoRA for 200 steps on TinyStories data.
3. Measure: val_loss, per-layer weight norms, contrastive signal magnitude.
4. Compare to MeZO baseline (condition 20, 300s).

### Phase 3: Ablation Study (1 day)

| Variant | Expected Result |
|---------|----------------|
| CHL + Oja (full) | Baseline Hebbian result |
| CHL without Oja | Weight explosion within ~50 steps |
| Naive Hebbian (no contrastive) | PCA behavior, no task learning |
| R-Hebb (reward modulated) | Similar to MeZO quality |
| CHL + higher corruption (50%) | Possibly stronger signal |
| CHL on ANE (full pipeline) | Same quality as CPU, ~2x faster |

### Phase 4: Documentation (0.5 day)

Document results as `2026-03-XX-hebbian-lora-results.md`. Record whether each hypothesis was confirmed or rejected.

---

## 13. Decision Framework

**GO for implementation if:**
- The theoretical analysis in this document does not reveal a fundamental impossibility (it does not -- CHL is a valid learning rule, just likely weak)
- We want to establish a complete picture of forward-only learning methods on ANE
- Implementation effort is bounded (2-3 days)

**STOP after Phase 2 if:**
- val_loss does not decrease at all (H1 rejected): Hebbian LoRA carries zero signal for LLM fine-tuning
- Weight explosion despite all stabilization (H2 rejected): the coupled A-B system is fundamentally unstable

**CONTINUE to Phase 3 if:**
- Any measurable val_loss improvement (even 0.001 nats): worth characterizing the quality frontier

**The most likely outcome is a well-documented negative result:** Contrastive Hebbian LoRA produces a weak learning signal (0.001-0.005 nats) that is substantially worse than MeZO (0.026 nats) and incomparably worse than backprop (0.275 nats). The value is in establishing the lower bound of what local learning rules can achieve for LLM fine-tuning on NPU hardware.

---

## 14. References

1. Hebb, D.O. (1949). *The Organization of Behavior*. Wiley.
2. Oja, E. (1982). "Simplified neuron model as a principal component analyzer." J. Math. Biology, 15(3), 267-273.
3. Ackley, D.H., Hinton, G.E., & Sejnowski, T.J. (1985). "A learning algorithm for Boltzmann machines." Cognitive Science, 9(1), 147-169.
4. Movellan, J.R. (1991). "Contrastive Hebbian learning in the continuous Hopfield model." Connectionist Models, 10-17.
5. Xie, X. & Seung, H.S. (2003). "Equivalence of backpropagation and contrastive Hebbian learning in a layered network." Neural Computation, 15(2), 441-454.
6. Hu, E.J. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.
7. Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345.
8. Malladi, S. et al. (2023). "Fine-Tuning Language Models with Just Forward Passes." NeurIPS 2023.
9. Bartunov, S. et al. (2018). "Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures." NeurIPS 2018.
10. Detorakis, G. & Bartley, T. (2018). "Contrastive Hebbian Learning with Random Feedback Weights." arXiv:1806.07406.
11. Hoier, A. et al. (2024). "Two Tales of Single-Phase Contrastive Hebbian Learning." ICML 2024, PMLR v235.
12. IOS Press (2024). "Going Beyond LoRA Fine-Tuning with Hebb Learning: Blazingly Fast and Accurate." DOI 10.3233/FAIA251089 (LoRA-HeRO, ECAI 2024).
13. Nature Communications (2025). "Self-Contrastive Forward-Forward Algorithm." DOI 10.1038/s41467-025-61037-0.
14. arXiv:2503.24322 (2025). "NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation."
15. arXiv:2510.21908 (2025). "Plastic Transformers with Hebbian and Gradient-Based Plasticity Rules."
16. arXiv:2408.08408 (2024). "Oja's plasticity rule overcomes challenges of training neural networks under biological constraints."
17. Apple (2025). "MeBP: Memory-Efficient Backpropagation for On-Device Training." arXiv:2510.03425.
