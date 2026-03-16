# Design: Forward-Forward + LoRA for LLM Fine-Tuning on Apple Neural Engine

**Date**: 2026-03-16
**Status**: Proposal
**Author**: Research team
**Goal**: Design a novel training algorithm combining Hinton's Forward-Forward (FF) algorithm with Low-Rank Adaptation (LoRA) for fine-tuning LLMs on forward-only hardware (Apple Neural Engine).

---

## 0. Executive Summary

We propose **FF-LoRA**: a layer-local, forward-only training algorithm for LLM fine-tuning that combines the Forward-Forward algorithm's backprop-free learning with LoRA's parameter efficiency. Each transformer layer independently learns to distinguish "positive" (real) data from "negative" (corrupted) data by maximizing/minimizing a local goodness function over its hidden activations. Only the LoRA adapter matrices A and B are updated, using gradients derived entirely from local forward-pass quantities -- no backward pass, no cross-layer gradient chain.

**Why this matters for ANE**: The Apple Neural Engine is forward-only hardware (18.6 TFLOPS fp16 via conv1x1). Our existing MeZO approach requires 2 full forward passes per step. FF-LoRA replaces full-model forward passes with per-layer forward passes, and the goodness gradient is computable from layer-local activations alone. This could unlock true layer-parallel training on ANE.

**Novelty claim**: As of March 2026, no published work combines Forward-Forward with LoRA for LLM fine-tuning. The closest works are: (a) Contrastive Forward-Forward for Vision Transformers (Aghagolzadeh & Ezoji, 2025), which applies FF to ViTs but uses full-parameter updates; (b) PEPITA/MEMPEPITA for LLMs (Pau et al., 2024), which uses a different forward-learning rule (input perturbation, not goodness-based); (c) MeZO (Malladi et al., 2023), which is forward-only but uses zeroth-order global gradient estimation, not layer-local learning. FF-LoRA is novel in all three axes: FF for LLMs, FF with LoRA, and FF on NPU hardware.

---

## 1. Algorithm: How FF+LoRA Works for Transformers

### 1.1 Background: The Forward-Forward Algorithm

Hinton (arXiv:2212.13345) proposes replacing backpropagation with two forward passes:

1. **Positive pass**: Real data flows through the network. Each layer's activations should have high "goodness."
2. **Negative pass**: Corrupted/fake data flows through the network. Each layer's activations should have low "goodness."

Each layer updates its weights independently to satisfy a local objective, with no gradient flowing backward through the network.

**Goodness function** (Hinton's original, for layer l):

```
G_l = sum_j (h_l,j)^2   (sum of squared activations)
```

The probability that an input is positive, at layer l:

```
p_l = sigma(G_l - theta_l)   where sigma is the logistic sigmoid
```

The per-layer loss:

```
L_l = -log p_l(x_pos) - log(1 - p_l(x_neg))
```

This is a binary cross-entropy loss: layer l must learn to assign G > theta to positive data and G < theta to negative data.

### 1.2 Goodness Function for a Transformer Layer

A transformer layer is not a simple MLP -- it contains RMSNorm, multi-head attention (Q/K/V projections, RoPE, SDPA, output projection), and a SwiGLU FFN. We must define "goodness" carefully.

**Definition**: For transformer layer l with input x_l in R^{DIM x SEQ} and output x_{l+1} in R^{DIM x SEQ}, define the goodness over the layer's output activations:

```
G_l = (1 / (DIM * SEQ)) * sum_{d,t} (x_{l+1}[d,t])^2
```

This is the mean squared activation (MSA) of the layer output, averaged over both dimension and sequence position. We use the mean rather than the sum to make the threshold theta_l comparable across layers with different dimensions.

**Why this works for transformers**: The residual connection means x_{l+1} = x_l + alpha * FFN(Attn(RMSNorm(x_l))). The goodness captures how much "energy" the layer adds to the residual stream. For positive data, the layer should coherently add energy; for negative data, activations should be suppressed or incoherent.

**Alternative considered -- attention-weighted goodness**: We could define goodness over attention outputs or FFN outputs separately. We reject this because: (a) it requires choosing which sub-component to measure, adding a hyperparameter; (b) the residual output integrates all sub-components naturally; (c) the residual stream is what flows to the next layer, making it the most semantically meaningful signal.

**Threshold theta_l**: Initialized to the mean goodness of the pretrained model on a calibration batch. Specifically, run one forward pass on 10 batches of real data, compute G_l for each layer, and set theta_l = mean(G_l). This ensures the threshold starts at a reasonable operating point. theta_l is a learnable scalar per layer.

### 1.3 Per-Layer Learning Objective

For layer l, given positive input x_pos and negative input x_neg:

```
L_l^FF = -log sigma(G_l(x_pos) - theta_l) - log sigma(theta_l - G_l(x_neg))
```

The gradient of L_l^FF with respect to any parameter w in layer l is:

```
dL_l/dw = -(1 - sigma(G_l(x_pos) - theta_l)) * dG_l(x_pos)/dw
           +(1 - sigma(theta_l - G_l(x_neg))) * dG_l(x_neg)/dw
```

Since G_l = (1/(DIM*SEQ)) * ||x_{l+1}||_F^2 and x_{l+1} depends on the layer's weights, we need dG_l/dw. This is where the key insight lies: **dG_l/dw can be computed from a single forward pass through layer l**, because it only depends on x_{l+1} and the local Jacobian dx_{l+1}/dw.

### 1.4 How Layers Connect Without Backward Pass

In standard FF, each layer receives its input from the previous layer's *detached* output. That is:

```
h_0 = input
h_1 = Layer_1(h_0.detach())       # no gradient flows from layer 1 to layer 0
h_2 = Layer_2(h_1.detach())       # etc.
```

For our transformer, during the **positive pass**:
1. Embed tokens -> x_0
2. For l = 0, 1, ..., L-1:
   - x_{l+1} = TransformerBlock_l(x_l)   [full forward, storing activations locally]
   - Compute G_l^+ = MSA(x_{l+1})

During the **negative pass** (with corrupted input):
1. Embed corrupted tokens -> x_0'
2. For l = 0, 1, ..., L-1:
   - x'_{l+1} = TransformerBlock_l(x'_l)
   - Compute G_l^- = MSA(x'_{l+1})

Each layer l then updates its parameters using only (G_l^+, G_l^-, theta_l) and local activations.

**Critical assumption**: The layer inputs are treated as fixed (detached). This means layer l does not receive credit assignment from downstream layers. This is the fundamental tradeoff of FF: no global gradient signal, but no backward pass either.

---

## 2. Negative Data Generation for Language Models

### 2.1 The Problem

Hinton's original FF work uses images, where negative data is generated by overlaying incorrect labels onto images (label-on-input). For autoregressive language models, we need a different strategy. The negative data must be "hard enough" that the model learns useful representations, but "clearly wrong" enough that the goodness signal is informative.

### 2.2 Option A: Token Corruption (Recommended)

**Method**: Given a sequence of tokens [t_1, t_2, ..., t_S], generate a negative sequence by corrupting a fraction p_corrupt of positions:

```
t'_i = { random token from vocab    with probability p_corrupt
        { t_i                        with probability 1 - p_corrupt
```

**Corruption rate**: p_corrupt = 0.15 (following BERT's masking convention, well-studied).

**Advantages**:
- Simple, fast (O(SEQ) random samples per batch)
- Preserves partial sequence structure (the model must detect subtle corruption)
- No additional forward passes needed
- Well-understood from denoising autoencoder literature (Vincent et al., 2008)

**Disadvantages**:
- At low corruption rates, negatives may be too similar to positives (weak signal)
- At high corruption rates, negatives are trivially distinguishable (no learning pressure)
- Does not capture the model's own failure modes

**Tuning**: We propose an adaptive corruption schedule: start at p_corrupt = 0.5 (easy negatives) and anneal to p_corrupt = 0.15 over the first 1000 steps. This gives strong initial signal then sharpens discrimination.

### 2.3 Option B: Sequence Shuffling

**Method**: Randomly permute token positions within the sequence:

```
[t_1, t_2, t_3, t_4] -> [t_3, t_1, t_4, t_2]
```

**Advantages**:
- Preserves token frequency distribution exactly
- Destroys sequential dependencies (which is what the model should learn)

**Disadvantages**:
- May be too easy: attention patterns and positional encodings immediately detect shuffling
- Does not test the model's understanding of token co-occurrence at the local level
- RoPE-equipped transformers are explicitly designed to distinguish positions

**Verdict**: Likely too easy for any model with positional encoding. Not recommended as primary method.

### 2.4 Option C: Model-Generated Negatives (Contrastive)

**Method**: Run the model itself to generate tokens autoregressively, then use these as negative data:

```
For each position t: sample t'_i ~ p_model(. | t_1, ..., t_{i-1})
```

**Advantages**:
- Negatives are at the model's current decision boundary (hardest negatives)
- Self-improving: as the model gets better, negatives get harder
- Theoretically elegant: resembles GAN-style training

**Disadvantages**:
- **Requires an additional full forward pass** for generation (SEQ autoregressive steps)
- Extremely expensive: O(SEQ * L * DIM^2) compute just for negative generation
- May cause mode collapse: if the model assigns high goodness to its own outputs, it learns nothing
- Autoregressive generation is sequential (not parallelizable on ANE)

**Verdict**: Too expensive for ANE. Each negative generation would cost as much as an entire training step. Not recommended for initial implementation.

### 2.5 Option D: Next-Token Shifted Negatives

**Method**: Use a shifted version of the target sequence as negative data:

```
Positive: [t_1, t_2, t_3, ..., t_S]    (input tokens)
Negative: [t_2, t_3, t_4, ..., t_{S+1}] (target tokens shifted left by 1)
```

**Advantages**:
- Zero additional compute (we already have both sequences from the training data)
- Captures the autoregressive structure: the model should represent "current context" differently from "next-token predictions"

**Disadvantages**:
- Very subtle difference -- may not provide enough signal
- For well-trained models, the representations of input and target may already be very similar

**Verdict**: Worth trying as a secondary method. Zero cost.

### 2.6 Recommended Strategy: Hybrid Token Corruption + Shifted Negatives

**Primary negatives**: Token corruption with adaptive rate (Section 2.2).
**Secondary negatives**: Shifted sequence (Section 2.5), used as an additional negative sample at zero extra cost.

Per step, each layer sees:
- 1 positive forward pass (real data)
- 1 negative forward pass (corrupted data)
- Optionally: 1 negative forward pass (shifted data, free)

Total: 2-3 forward passes per step.

---

## 3. LoRA Update Rule from FF Goodness Objective

### 3.1 Setup

For a single linear projection in layer l (e.g., the query projection):

```
W_eff = W_base + B @ A
```

where W_base in R^{out x in} is frozen, A in R^{r x in}, B in R^{out x r} are trainable LoRA matrices with rank r.

The projection output is:

```
y = W_eff @ x = (W_base + B @ A) @ x = W_base @ x + B @ (A @ x)
```

where x in R^{in x SEQ} is the layer input (from RMSNorm).

### 3.2 Goodness Gradient for the Full Layer

The layer output x_{l+1} is a complex function of all projections (Q, K, V, O, W1, W2, W3). However, for the FF update rule, we need dG_l/dA and dG_l/dB for each projection's LoRA adapters.

**Key simplification**: Rather than propagating through the full transformer block (which would be a local backward pass within the layer), we define a **projection-local goodness** for each linear projection. This is a further localization beyond Hinton's layer-local approach:

For projection p (e.g., Wq) with output y_p in R^{out_p x SEQ}:

```
G_l,p = (1 / (out_p * SEQ)) * ||y_p||_F^2
```

The total layer goodness is the sum over all projections, but each projection's LoRA adapters are updated using only their own goodness. This avoids any within-layer backward pass.

**Assumption A1**: Projection-local goodness is a sufficient learning signal. This is a strong assumption -- it means the query projection learns independently from the value projection within the same layer. We discuss risks in Section 7.

### 3.3 Deriving the LoRA Gradient

For the query projection in layer l, with LoRA output:

```
y_q = (W_q^base + B_q @ A_q) @ x_norm
```

where x_norm = RMSNorm(x_l) in R^{DIM x SEQ}.

The projection-local goodness:

```
G_q = (1 / (Q_DIM * SEQ)) * ||y_q||_F^2
      = (1 / (Q_DIM * SEQ)) * tr(y_q^T @ y_q)
```

The gradient with respect to B_q:

```
dG_q/dB_q = (2 / (Q_DIM * SEQ)) * y_q @ x_norm^T @ A_q^T
```

**Derivation**: Since y_q = (W_q^base + B_q @ A_q) @ x_norm, we have dy_q/dB_q = (dy_q)_{ij} / d(B_q)_{kl} = delta_{ik} * (A_q @ x_norm)_{lj}. Then:

```
dG_q / d(B_q)_{kl} = (2 / (Q_DIM * SEQ)) * sum_j y_q[k,j] * (A_q @ x_norm)[l,j]
                    = (2 / (Q_DIM * SEQ)) * [y_q @ (A_q @ x_norm)^T]_{kl}
```

Therefore:

```
dG_q/dB_q = (2 / (Q_DIM * SEQ)) * y_q @ x_norm^T @ A_q^T     ... (Eq. 1)
```

Dimensions: [out x SEQ] @ [SEQ x in] @ [in x r] = [out x r]. Matches B_q shape.

Similarly for A_q:

```
dG_q/dA_q = (2 / (Q_DIM * SEQ)) * B_q^T @ y_q @ x_norm^T     ... (Eq. 2)
```

Dimensions: [r x out] @ [out x SEQ] @ [SEQ x in] = [r x in]. Matches A_q shape.

### 3.4 The FF-LoRA Update Rule

For each projection p in layer l, at each training step:

```
1. Positive forward: y_p^+ = (W_p^base + B_p @ A_p) @ x_norm^+
   Compute G_p^+ = MSA(y_p^+)

2. Negative forward: y_p^- = (W_p^base + B_p @ A_p) @ x_norm^-
   Compute G_p^- = MSA(y_p^-)

3. Compute FF scaling factors:
   s^+ = 1 - sigma(G_p^+ - theta_p)      (want to increase G^+)
   s^- = 1 - sigma(theta_p - G_p^-)      (want to decrease G^-)

4. Update B_p:
   B_p <- B_p - lr * [ -s^+ * (2/(out*SEQ)) * y_p^+ @ (x_norm^+)^T @ A_p^T
                        +s^- * (2/(out*SEQ)) * y_p^- @ (x_norm^-)^T @ A_p^T ]

5. Update A_p:
   A_p <- A_p - lr * [ -s^+ * (2/(out*SEQ)) * B_p^T @ y_p^+ @ (x_norm^+)^T
                        +s^- * (2/(out*SEQ)) * B_p^T @ y_p^- @ (x_norm^-)^T ]

6. Update theta_p:
   theta_p <- theta_p + lr_theta * (s^- - s^+)
```

### 3.5 What Each Step Requires (Compute Analysis)

For a single projection Wq with A_q[r, DIM] and B_q[Q_DIM, r]:

| Operation | FLOPs | Needed for |
|-----------|-------|-----------|
| A_q @ x_norm (positive) | 2 * r * DIM * SEQ | Forward pass (already computed in lora_addmm) |
| B_q @ (A_q @ x_norm) (positive) | 2 * Q_DIM * r * SEQ | Forward pass (already computed) |
| y_q^+ @ (x_norm^+)^T | 2 * Q_DIM * SEQ * DIM | dG/dB numerator |
| ... @ A_q^T | 2 * Q_DIM * DIM * r | dG/dB final |
| B_q^T @ y_q^+ | 2 * r * Q_DIM * SEQ | dG/dA numerator |
| ... @ (x_norm^+)^T | 2 * r * SEQ * DIM | dG/dA final |

The dominant cost is the outer product y_q^+ @ (x_norm^+)^T: this is [Q_DIM x SEQ] @ [SEQ x DIM] = [Q_DIM x DIM], costing 2 * Q_DIM * SEQ * DIM FLOPs. For SmolLM2-360M: 2 * 960 * 256 * 960 = 471M FLOPs per projection, per polarity.

**Critical optimization**: We do NOT need to form the full [Q_DIM x DIM] outer product. Instead, we can fuse it with the A_q^T multiplication:

```
dG/dB_q = (2/(Q_DIM*SEQ)) * y_q @ (x_norm^T @ A_q^T)
        = (2/(Q_DIM*SEQ)) * y_q @ (A_q @ x_norm)^T
```

Note that (A_q @ x_norm) in R^{r x SEQ} is already computed during the forward pass! So:

```
dG/dB_q = (2/(Q_DIM*SEQ)) * y_q @ z^T      where z = A_q @ x_norm, already available
```

This is [Q_DIM x SEQ] @ [SEQ x r] = [Q_DIM x r], costing only 2 * Q_DIM * SEQ * r FLOPs.

For SmolLM2-360M with r=8: 2 * 960 * 256 * 8 = 3.9M FLOPs. **120x cheaper** than the naive approach.

Similarly:

```
dG/dA_q = (2/(Q_DIM*SEQ)) * B_q^T @ y_q @ x_norm^T
```

This requires [r x Q_DIM] @ [Q_DIM x SEQ] = [r x SEQ] (cost: 2 * r * Q_DIM * SEQ = 3.9M), then [r x SEQ] @ [SEQ x DIM] = [r x DIM] (cost: 2 * r * SEQ * DIM = 3.9M). Total: 7.8M FLOPs.

**Summary per projection per polarity**: ~12M FLOPs for gradient computation, on top of ~4M for the forward LoRA correction. The gradient computation is approximately 3x the forward cost -- far cheaper than a full backward pass.

### 3.6 Why This Needs Only Local Activations

The update rule for layer l's projections requires only:
- x_norm^+ and x_norm^- (inputs to the projection, from this layer's RMSNorm)
- y_p^+ and y_p^- (outputs of the projection, from this layer's forward)
- z_p^+ = A_p @ x_norm^+ (intermediate LoRA activation, from this layer's forward)
- A_p, B_p (this layer's LoRA weights)
- theta_p (this layer's threshold scalar)

No quantity from any other layer is needed. No backward pass through attention, RoPE, or softmax is required. The gradient flows only through the linear LoRA projection itself.

---

## 4. Implementation Plan

### 4.1 New File: `train_ff.m` (or mode in `train_mezo.m`)

We recommend adding FF-LoRA as a new mode in `train_mezo.m` (via `--ff-lora` flag), sharing the existing infrastructure for LoRA-split, conv-fused kernels, and the forward pass macro. This avoids code duplication.

### 4.2 New Data Structures

```c
// Per-projection FF state (threshold + intermediate activations)
typedef struct {
    float theta;            // Goodness threshold (learnable)
    float *z_pos;           // [rank, SEQ] = A @ x_norm (positive), stored from fwd pass
    float *z_neg;           // [rank, SEQ] = A @ x_norm (negative), stored from fwd pass
    float *y_pos;           // [out_dim, SEQ] projection output (positive)
    float *y_neg;           // [out_dim, SEQ] projection output (negative)
} FFProjState;

// Per-layer FF state (4 attention projections + optionally 3 FFN)
typedef struct {
    FFProjState q, k, v, o;
    FFProjState w1, w2, w3;  // NULL if no FFN LoRA
    float G_pos, G_neg;      // Layer-level goodness (for logging)
} FFLayerState;
```

### 4.3 Changes to Forward Pass Macro

We need a modified `DO_FORWARD_PASS_FF` that, for each projection, stores the intermediate LoRA activation z = A @ x_norm (which `lora_addmm` already computes internally but does not expose). The change is minimal:

```c
// Modified lora_addmm that also stores the intermediate z = A @ x
static void lora_addmm_ff(float *out, const float *A, const float *B,
                           const float *x, float *tmp_r, float *z_out,
                           int out_dim, int rank, int in_dim) {
    // z[rank, SEQ] = A[rank, in_dim] @ x[in_dim, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, SEQ, in_dim, 1.0f, A, in_dim, x, SEQ, 0.0f, tmp_r, SEQ);
    // Store z for gradient computation
    memcpy(z_out, tmp_r, (size_t)rank * SEQ * 4);
    // out[out_dim, SEQ] += B[out_dim, rank] @ z[rank, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, SEQ, rank, 1.0f, B, rank, tmp_r, SEQ, 1.0f, out, SEQ);
}
```

### 4.4 Negative Data Generation

```c
// Generate corrupted token sequence (token corruption)
static void generate_negative_tokens(uint16_t *neg_tokens, const uint16_t *pos_tokens,
                                      int seq_len, int vocab_size, float corrupt_rate) {
    for (int i = 0; i < seq_len; i++) {
        float r = (float)drand48();
        if (r < corrupt_rate) {
            neg_tokens[i] = (uint16_t)(drand48() * vocab_size);
        } else {
            neg_tokens[i] = pos_tokens[i];
        }
    }
}
```

### 4.5 Per-Layer FF-LoRA Gradient Update

```c
// Compute FF-LoRA gradient for one projection and apply update
// y: [out_dim, SEQ], z: [rank, SEQ], x_norm: [in_dim, SEQ]
// A: [rank, in_dim], B: [out_dim, rank]
static void ff_lora_update_projection(
    float *A, float *B, float *theta,
    const float *y_pos, const float *z_pos, const float *x_norm_pos,
    const float *y_neg, const float *z_neg, const float *x_norm_neg,
    int out_dim, int rank, int in_dim, float lr)
{
    float norm = 1.0f / (float)(out_dim * SEQ);

    // Compute goodness: G = (1/(out*SEQ)) * ||y||_F^2
    float G_pos = 0, G_neg = 0;
    for (int i = 0; i < out_dim * SEQ; i++) G_pos += y_pos[i] * y_pos[i];
    for (int i = 0; i < out_dim * SEQ; i++) G_neg += y_neg[i] * y_neg[i];
    G_pos *= norm;
    G_neg *= norm;

    // FF scaling factors
    float s_pos = 1.0f - 1.0f / (1.0f + expf(-(G_pos - *theta)));  // 1 - sigma(G+ - theta)
    float s_neg = 1.0f - 1.0f / (1.0f + expf(-(*theta - G_neg)));  // 1 - sigma(theta - G-)

    float scale_pos = -s_pos * 2.0f * norm;   // want to INCREASE G_pos
    float scale_neg = +s_neg * 2.0f * norm;    // want to DECREASE G_neg

    // dB += scale * y @ z^T   (for both polarities)
    // [out_dim x rank] += [out_dim x SEQ] @ [SEQ x rank]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, SEQ, lr * scale_pos, y_pos, SEQ, z_pos, SEQ, 1.0f, B, rank);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                out_dim, rank, SEQ, lr * scale_neg, y_neg, SEQ, z_neg, SEQ, 1.0f, B, rank);

    // dA: need B^T @ y @ x_norm^T
    // tmp[rank, SEQ] = B^T @ y   then   dA += scale * tmp @ x_norm^T
    float *tmp = (float*)safe_malloc((size_t)rank * SEQ * 4);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, SEQ, out_dim, 1.0f, B, rank, y_pos, SEQ, 0.0f, tmp, SEQ);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, in_dim, SEQ, lr * scale_pos, tmp, SEQ, x_norm_pos, SEQ, 1.0f, A, in_dim);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, SEQ, out_dim, 1.0f, B, rank, y_neg, SEQ, 0.0f, tmp, SEQ);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, in_dim, SEQ, lr * scale_neg, tmp, SEQ, x_norm_neg, SEQ, 1.0f, A, in_dim);

    free(tmp);

    // Update threshold
    float lr_theta = lr * 0.1f;  // slower learning rate for threshold
    *theta += lr_theta * (s_neg - s_pos);
}
```

### 4.6 Training Loop Structure

```
Per training step:
  1. Sample batch of real tokens [SEQ+1]
     - input_tokens = tokens[0:SEQ]
     - targets = tokens[1:SEQ+1]

  2. Generate negative tokens (corruption)
     - neg_tokens = corrupt(input_tokens, p_corrupt)

  3. Positive forward pass (real data):
     - embed_lookup(x_cur, embed, input_tokens)
     - For each layer L:
       - xnorm = RMSNorm(x_cur, rms_att[L])
       - Q = Wq_base @ xnorm;  lora_addmm_ff(Q, Aq, Bq, xnorm, ..., z_q_pos)
       - K = Wk_base @ xnorm;  lora_addmm_ff(K, Ak, Bk, xnorm, ..., z_k_pos)
       - V = Wv_base @ xnorm;  lora_addmm_ff(V, Av, Bv, xnorm, ..., z_v_pos)
       - Store y_q_pos = Q, y_k_pos = K, y_v_pos = V, x_norm_pos = xnorm
       - RoPE, SDPA, Wo projection (similarly storing z_o_pos, y_o_pos)
       - FFN (similarly, if --lora-ffn)
       - x_cur = x_cur + alpha * ffn_out
     - Compute cross-entropy loss on logits (for monitoring only)

  4. Negative forward pass (corrupted data):
     - embed_lookup(x_cur, embed, neg_tokens)
     - For each layer L:
       - Same as above, storing z_*_neg, y_*_neg, x_norm_neg

  5. Per-layer FF-LoRA update:
     - For each layer L:
       - For each projection (q, k, v, o, optionally w1, w2, w3):
         - ff_lora_update_projection(A_p, B_p, &theta_p,
             y_p_pos, z_p_pos, x_norm_pos,
             y_p_neg, z_p_neg, x_norm_neg, ...)

  6. (Optional) Compute cross-entropy loss on validation set (for tracking)
```

### 4.7 Forward Passes Per Step

| Method | Forward passes | Backward passes | Total model-traversals |
|--------|---------------|-----------------|----------------------|
| Backprop | 1 | 1 | 2 (but backward is ~2x forward) |
| MeZO | 2 | 0 | 2 |
| FF-LoRA | 2 | 0 | 2 |
| FF-LoRA + shifted neg | 3 | 0 | 3 |

FF-LoRA has the same number of model traversals as MeZO, but the gradient signal is fundamentally different: MeZO gets a noisy scalar gradient projected along a random direction; FF-LoRA gets a per-projection gradient computed from a structured (positive/negative) signal.

### 4.8 Memory Overhead

| Component | MeZO+LoRA-split | FF-LoRA (additional) |
|-----------|----------------|---------------------|
| Base weights (frozen) | 690 MB | 0 (shared) |
| LoRA A, B matrices | 4.7 MB (rank 8) | 0 (shared) |
| Forward activations (x_cur, Q, K, V, etc.) | ~10 MB | ~10 MB (for negative pass -- can reuse) |
| FF-specific: z_pos, z_neg per projection | 0 | 7 projections * 2 * r * SEQ * 4 = 7*2*8*256*4 = 112 KB |
| FF-specific: y_pos, y_neg per projection | 0 | Need y storage: ~2 * 10MB = 20 MB |
| FF-specific: x_norm_pos, x_norm_neg | 0 | 2 * DIM * SEQ * 4 = 2 * 960 * 256 * 4 = 1.9 MB |
| FF-specific: theta per projection per layer | 0 | 32 * 7 * 4 = 896 bytes |
| **Total additional** | -- | **~22 MB** |

The 22 MB overhead is for storing positive-pass activations while running the negative pass. This can be reduced to ~11 MB by processing one polarity at a time per layer (interleaving positive and negative forward passes within each layer), at the cost of breaking the clean two-pass structure.

**Optimization**: Process layers sequentially. For layer l, run positive forward, store z_pos and y_pos, then run negative forward using the same layer weights, compute gradient, update. This requires storing only one layer's worth of FF state at a time (~3 MB), reducing peak additional memory to ~13 MB.

However, this interleaved approach changes the semantics: layer l+1's negative input would come from a different set of weights (post-update) than its positive input. This is actually consistent with FF's "detached" semantics, where each layer treats its input as fixed.

### 4.9 ANE Integration

FF-LoRA is naturally compatible with the existing conv-fused + LoRA-split architecture:

1. **Base projections on ANE**: W_base @ x_norm runs as conv1x1 on ANE (existing qkvConv, ffnConv kernels).
2. **LoRA corrections on CPU**: A @ x_norm and B @ z on CPU (existing lora_addmm).
3. **FF gradient on CPU**: The gradient computation (Eq. 1, Eq. 2) uses only CPU BLAS operations on small matrices (rank 8).
4. **No kernel recompilation**: Base weights are baked into conv1x1 kernels and never change. Only LoRA A, B are updated (on CPU). This is identical to MeZO+LoRA-split.

The ANE pipeline requires **zero changes** from the existing conv-fused mode. The only new code is the FF gradient computation and negative data generation, both CPU-side.

---

## 5. Hypotheses

### Hypothesis H1: FF-LoRA converges faster per step than MeZO+LoRA

**Rationale**: MeZO estimates a single scalar gradient (the directional derivative along a random direction) from 2 forward passes. FF-LoRA computes a structured, per-projection gradient from 2 forward passes. The FF gradient has O(r * (in + out)) dimensions of useful information per projection, vs. 1 dimension for MeZO. This should translate to faster convergence in terms of loss decrease per step.

**Success criterion**: After 500 steps of fine-tuning SmolLM2-360M on TinyStories with identical lr:
- FF-LoRA achieves at least 2x greater val_loss reduction compared to MeZO+LoRA-split
- Specifically: if MeZO achieves delta_val = 0.005 in 500 steps (observed in existing experiments), FF-LoRA should achieve delta_val >= 0.01

**Measurement**: Run both methods from the same checkpoint, same data, same lr, same rank-8 LoRA, 500 steps. Report val_loss at steps 0, 100, 200, 300, 400, 500.

### Hypothesis H2: FF-LoRA achieves comparable wall-clock throughput to MeZO on ANE

**Rationale**: Both methods use 2 forward passes per step. The FF gradient computation adds O(7 * 2 * 2 * r * out * SEQ) FLOPs per layer on CPU, which for r=8, out=960, SEQ=256 is approximately 7 * 2 * 2 * 8 * 960 * 256 = 55M FLOPs per layer, or 1.76G FLOPs total across 32 layers. The forward pass itself is approximately 2 * 362M * 256 = 185G FLOPs. So the FF gradient overhead is approximately 1% of the forward pass compute.

**Success criterion**: FF-LoRA step time on ANE (conv-fused mode) is within 10% of MeZO+LoRA-split step time (baseline: 262ms/step).

**Measurement**: Time 100 steps of each method on the same hardware. Report mean and std of step times.

### Hypothesis H3: FF-LoRA's layer-local objective does NOT achieve the same final quality as backprop LoRA

**Rationale**: Backprop provides an exact gradient of the global cross-entropy loss with respect to every LoRA parameter. FF-LoRA provides a local goodness gradient that may not align with the global loss surface. The layer-local signal does not provide end-to-end credit assignment: early layers cannot receive signal about how their representations affect the final prediction. This is a fundamental limitation of all local learning methods.

**Success criterion**: After 2000 steps from the same SmolLM2-360M checkpoint:
- Backprop LoRA achieves val_loss at least 0.1 nats better than FF-LoRA
- If FF-LoRA matches or beats backprop, this hypothesis is disproved (which would be a very exciting result)

**Measurement**: Run backprop-lora (existing P16 mode) and FF-LoRA from the same checkpoint, same data, 2000 steps. Report val_loss curves.

---

## 6. Comparison: FF-LoRA vs. MeZO vs. Backprop

| Property | Backprop LoRA | MeZO+LoRA | FF-LoRA |
|----------|---------------|-----------|---------|
| Forward passes / step | 1 | 2 | 2 |
| Backward passes / step | 1 | 0 | 0 |
| Gradient quality | Exact (global loss) | Noisy scalar (random projection) | Structured local (per-projection goodness) |
| Gradient dimensions per step | All LoRA params (~150K) | 1 (scalar, projected onto all) | All LoRA params (~150K), locally |
| Memory overhead | Activations for all layers + grad buffers (~50 MB) | Seed only (8 bytes) | Per-layer FF state (~22 MB) |
| Convergence rate (per step) | Best | Worst (~100x slower) | Unknown (predicted: between MeZO and backprop) |
| ANE compatibility | Requires backward kernels | Forward-only | Forward-only |
| End-to-end credit assignment | Yes | Yes (global loss) | No (local objectives) |
| Layer parallelism potential | No (sequential backward) | No (global forward) | Yes (layers are independent) |
| Step time (SmolLM2-360M, ANE) | ~800ms (fwd+bwd) | 262ms (conv-fused) | ~265ms (predicted, +1% overhead) |
| Expected convergence quality | Best | Worst | Middle |

### Key Tradeoffs

**FF-LoRA vs. MeZO**: Both are forward-only with 2 passes per step. FF-LoRA provides a much richer gradient signal (structured per-projection gradient vs. noisy scalar). However, FF-LoRA's gradient optimizes a proxy objective (local goodness) rather than the true loss. If the goodness proxy aligns well with the global loss, FF-LoRA wins decisively. If not, MeZO's global (albeit noisy) signal may be more useful.

**FF-LoRA vs. Backprop**: Backprop is strictly better in gradient quality. FF-LoRA's advantage is hardware: on ANE, backprop requires backward kernels (SDPA backward, dW computation), which either fall back to CPU or require complex MIL programs. FF-LoRA uses only forward kernels (existing conv1x1) plus small CPU-side BLAS. The question is whether the gradient quality gap is small enough that FF-LoRA's hardware advantages matter.

**The scaling argument**: MeZO's convergence rate degrades as O(d) where d is the number of trainable parameters (variance of the ZO gradient scales with d). FF-LoRA's convergence depends on the goodness proxy quality, which is independent of d. This suggests FF-LoRA may scale better to larger models where MeZO struggles.

---

## 7. Risks and Failure Modes

### Risk R1: Layer-local goodness is a poor proxy for language modeling loss (HIGH RISK)

The goodness function (mean squared activation) measures activation magnitude, not semantic quality. A layer could achieve high goodness by simply scaling up all activations without improving language modeling. The RMSNorm after each layer partially mitigates this (it normalizes activation scale), but the goodness signal may still be too weak to drive useful learning.

**Mitigation**: Use layer normalization *before* computing goodness (i.e., measure goodness on post-RMSNorm activations of the *next* layer's input, which removes scale information). Alternatively, use cosine similarity between positive and negative representations as an additional signal.

**Detection**: If goodness increases for both positive and negative data simultaneously (both converge to the same value), the objective is degenerate.

### Risk R2: Negative data too easy or too hard (MEDIUM RISK)

If token corruption is too aggressive (p=0.5), negatives are trivially different -- the model can distinguish them by surface statistics without learning deep structure. If too subtle (p=0.05), the goodness difference is negligible, providing no learning signal.

**Mitigation**: Adaptive corruption rate. Monitor the goodness gap (G_pos - G_neg) per layer. If gap > 2*theta, decrease corruption (make it harder). If gap < 0.1*theta, increase corruption (make it easier). This creates a curriculum.

**Detection**: Plot G_pos and G_neg per layer over training. Healthy: G_pos > theta > G_neg with moderate gap. Unhealthy: both above or both below theta, or gap is maximal from step 0.

### Risk R3: Projection-local goodness destroys inter-projection coordination (HIGH RISK)

In a standard transformer, the Q, K, V projections are coordinated through end-to-end training: Q and K learn complementary representations because the attention mechanism (Q @ K^T) provides a gradient signal that links them. In FF-LoRA, Q and K are trained independently against their own goodness objectives. There is no mechanism for Q to "know" what K is doing.

**Mitigation**: Use the full layer output goodness (post-residual x_{l+1}) instead of per-projection goodness. This requires a within-layer backward pass from x_{l+1} to each projection's LoRA parameters. The backward pass through attention (SDPA) is the expensive part. We can approximate:
- Use the Jacobian-vector product trick: compute d(||x_{l+1}||^2)/dB_q by treating x_{l+1} as a composition and using the chain rule within the layer only.
- This is a "shallow backward" -- backward through one transformer block, not the entire network.
- This is more expensive but still avoids cross-layer gradients.

**Alternative mitigation**: Joint goodness over concatenated [Q; K; V; O] outputs. This links the projections through a shared objective without requiring backward through attention.

**Detection**: Monitor attention entropy. If attention becomes uniform (all positions attend equally), Q-K coordination has failed.

### Risk R4: Pretrained representations already have high goodness -- no room to learn (MEDIUM RISK)

SmolLM2-360M is a pretrained model with well-structured activations. The goodness function may already be near-optimal for real data. The LoRA adapters start at zero, so initially W_eff = W_base and the goodness is entirely determined by the pretrained weights. The FF signal may be too weak to drive adaptation.

**Mitigation**: Initialize theta_l below the pretrained goodness (e.g., theta_l = 0.8 * G_l_pretrained). This creates an initial "surplus" of goodness on positive data, providing a clear signal direction.

**Detection**: If FF-LoRA's loss curve is flat from step 0, the FF signal is too weak.

### Risk R5: FF-LoRA may not converge at all for autoregressive LMs (MEDIUM RISK)

All published FF results are on classification tasks (MNIST, CIFAR-10, sentiment analysis, ViT classification). Language modeling is fundamentally different: the loss is token-level, sequential, and the output space is enormous (49K tokens). The goodness function may not carry enough information for the model to learn which tokens to predict.

**Mitigation**: Add a global cross-entropy loss term as a regularizer. Use FF-LoRA for the per-layer update, but also compute the global loss and use it to scale the FF learning rate. This is a hybrid: FF provides the gradient direction, global loss provides the magnitude.

**Detection**: If val_loss increases or stays flat after 500 steps while MeZO decreases, FF-LoRA has failed for autoregressive LMs.

### Risk R6: Stability issues from simultaneous per-layer updates (LOW RISK)

All layers update simultaneously based on their local objectives. If multiple layers make large updates, the composed effect could be destabilizing (since each layer assumed its neighbors' weights were fixed).

**Mitigation**: Small learning rate and gradient clipping on the FF-LoRA updates. Use separate lr for FF-LoRA that is 0.1x the MeZO lr.

---

## 8. Literature Review

### 8.1 Forward-Forward Algorithm (Core)

**Hinton (2022)**. "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345.
The foundational paper. Defines goodness as sum of squared activations. Tests on MNIST, achieves 98.7% (vs. 99.3% for backprop). Uses label-on-input for positive/negative generation. Notes FF is "considerably slower than backpropagation" but "may prove better for hardware that can only do forward computation."

### 8.2 FF for Vision Transformers

**Aghagolzadeh & Ezoji (2025)**. "Contrastive Forward-Forward: A Training Algorithm of Vision Transformer." arXiv:2502.00571. Accepted in Neural Networks.
First application of FF to ViTs. Replaces goodness/badness with supervised contrastive loss at each layer. Achieves up to 10% improvement over baseline FF and 5-20x convergence speedup. Uses full-parameter updates (no LoRA). **Our work differs in: (a) LLMs not ViTs; (b) LoRA not full params; (c) NPU hardware target.**

### 8.3 Self-Contrastive FF

**Chen, Liu, Laydevant & Grollier (2025)**. "Self-Contrastive Forward-Forward algorithm." Nature Communications 16, 5978.
Extends FF to unsupervised learning using contrastive principles. Contrasts each sample against augmented versions of itself. First application of FF to sequential data (time series). **Relevant insight**: contrastive structure improves FF significantly; we adopt this principle in our negative data design.

### 8.4 FF for CNNs

**Scientific Reports (2025)**. "Training convolutional neural networks with the Forward-Forward Algorithm." Nature/Scientific Reports.
Extends FF to convolutional architectures. Shows FF can train CNNs competitively but requires careful architectural choices.

### 8.5 Adaptive Spatial Goodness Encoding

**arXiv:2509.12394 (2025)**. "Adaptive Spatial Goodness Encoding: Advancing and Scaling Forward-Forward Learning Without Backpropagation."
Proposes spatial goodness encoding to scale FF to larger problems. Addresses the fundamental challenge of FF's layer-local signal being too weak for complex tasks.

### 8.6 NoProp

**Bhatt et al. (2025)**. "NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation." CoLLAs 2025. arXiv:2503.24322.
Block-local learning inspired by diffusion models. Each block independently denoises a noisy label. Evaluated on CIFAR-10/100, comparable to backprop. **Relevant**: demonstrates block-local learning can work, though uses a different mechanism (denoising, not goodness).

### 8.7 PEPITA/MEMPEPITA for LLMs

**Pau et al. (2024)**. "Forward Learning of Large Language Models by Consumer Devices." Electronics 13(2), 402.
Analyzes PEPITA and MEMPEPITA computational costs for GPT-3 Small, DistilBERT, AlexaTM. Shows 30-50% compute reduction and 50-94% memory reduction vs. backprop. **Closest to our work**: forward-only learning for LLMs on edge devices. Key difference: PEPITA uses input perturbation (not goodness), and does not use LoRA.

### 8.8 MeZO

**Malladi et al. (2023)**. "Fine-Tuning Language Models with Just Forward Passes." NeurIPS 2023. arXiv:2305.17333.
Forward-only via zeroth-order optimization. Trains 30B parameter models on a single GPU. ~100x more steps than backprop for comparable fine-tuning quality. **Our baseline comparison method.**

### 8.9 FwdLLM

**Xu et al. (2024)**. "FwdLLM: Efficient FedLLM using Forward Gradient." USENIX ATC 2024.
Forward-only federated LLM fine-tuning. 14.6x memory reduction. Uses forward gradients (not FF goodness) for mobile deployment.

### 8.10 LoRA

**Hu et al. (2021)**. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.
Freezes base weights, trains low-rank A[r,d] and B[out,r] adapters. Reduces trainable parameters by 10,000x. Foundation for our adapter architecture.

### 8.11 Orion

**Murai et al. (2026)**. "Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference." arXiv:2603.06728.
First end-to-end ANE training system. Characterizes ANE constraints. **Relevant**: validates ANE as a training target. Our work builds on the same hardware understanding.

### 8.12 Novelty Assessment

No published work (as of March 2026) combines:
1. Forward-Forward goodness-based learning
2. LoRA low-rank adapters
3. Large language models (autoregressive transformers)

The closest intersection is PEPITA+LLM (Pau et al., 2024), but PEPITA uses a fundamentally different learning rule (input perturbation) and does not use LoRA. Contrastive FF for ViT (Aghagolzadeh & Ezoji, 2025) uses FF+transformer but not LoRA and not LLMs. **FF-LoRA for LLM fine-tuning is novel.**

---

## 9. Explicit Assumptions

| ID | Assumption | Category | Status |
|----|-----------|----------|--------|
| A1 | Projection-local goodness is a sufficient learning signal for LoRA adaptation | Algorithm | Untested; high risk |
| A2 | Token corruption at rate 0.15-0.50 generates informative negatives for LLMs | Data | Supported by BERT literature; untested for FF |
| A3 | The goodness function (MSA) is meaningful after RMSNorm | Algorithm | Partially supported (Hinton notes layer norm removes goodness) |
| A4 | Per-projection goodness does not require Q-K coordination signal | Algorithm | Untested; high risk |
| A5 | Pretrained model activations have learnable goodness gap between real and corrupted text | Algorithm | Untested; medium risk |
| A6 | SmolLM2-360M's 32-layer depth does not cause FF signal degradation in later layers | Architecture | Untested; Hinton tested up to 4 layers |
| A7 | FF-LoRA gradient computation overhead is <10% of forward pass time | Performance | Supported by FLOP analysis (Section 3.5) |
| A8 | Adaptive corruption rate prevents trivial or degenerate goodness separation | Training | Standard curriculum learning; untested for FF |
| A9 | LoRA rank 8 provides enough capacity for goodness-based learning | Architecture | Supported by MeZO rank-8 results |
| A10 | Layer-local learning can improve language modeling loss despite no global credit assignment | Algorithm | Not supported by existing FF literature (all classification) |

---

## 10. Implementation Timeline

| Phase | Task | Estimated effort |
|-------|------|-----------------|
| P1 | Add `--ff-lora` mode to `train_mezo.m`: negative data generation, modified forward pass with z storage | 1 day |
| P2 | Implement `ff_lora_update_projection()` and per-layer update loop | 0.5 day |
| P3 | Add goodness logging, threshold initialization from pretrained model | 0.5 day |
| P4 | Baseline comparison: 500-step convergence test (FF-LoRA vs. MeZO vs. backprop-lora) | 0.5 day |
| P5 | Tune corruption rate, learning rate, threshold lr | 1 day |
| P6 | Test layer-level goodness (Risk R3 mitigation) vs. projection-level goodness | 0.5 day |
| P7 | Write results, update MEZO_AUDIT_REPORT.md, add to FINDINGS.md | 0.5 day |
| **Total** | | **4.5 days** |

---

## 11. Summary

FF-LoRA is a novel algorithm that fills an unexplored point in the design space:

```
                     Global loss          Local loss
                   +-----------------+------------------+
  Exact gradient   | Backprop (BP)   | FF (Hinton)      |
                   +-----------------+------------------+
  Noisy gradient   | MeZO (ZO)       | FF-LoRA (ours)   |
                   +-----------------+------------------+
                     ^                  ^
                     Requires backward  Forward-only
```

Wait -- this framing is not quite right. FF-LoRA actually computes an *exact* gradient of a *local* objective. Let us correct:

```
                     Global objective      Local objective
                   +--------------------+--------------------+
  Exact gradient   | Backprop LoRA      | FF-LoRA (ours)     |
  of objective     | (gold standard)    | (exact local grad) |
                   +--------------------+--------------------+
  Noisy gradient   | MeZO+LoRA         |                    |
  of objective     | (ZO, 1-dim)       |                    |
                   +--------------------+--------------------+
```

FF-LoRA trades global optimality for hardware compatibility. It computes exact gradients of a local objective using only forward passes and small CPU-side BLAS operations. If the local objective (goodness) aligns with the global loss (language modeling), FF-LoRA could significantly outperform MeZO while matching its hardware requirements.

The key open question is assumption A10: can layer-local learning improve language modeling at all? This is genuinely unknown -- all FF results to date are on classification. A positive result would be a meaningful contribution to both the FF and on-device learning literatures. A negative result would sharpen our understanding of why global credit assignment matters for language models.

---

## References

- Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345. https://arxiv.org/abs/2212.13345
- Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Malladi, S. et al. (2023). "Fine-Tuning Language Models with Just Forward Passes." NeurIPS 2023. arXiv:2305.17333. https://arxiv.org/abs/2305.17333
- Aghagolzadeh, H. & Ezoji, M. (2025). "Contrastive Forward-Forward: A Training Algorithm of Vision Transformer." arXiv:2502.00571. https://arxiv.org/abs/2502.00571
- Chen, X. et al. (2025). "Self-Contrastive Forward-Forward algorithm." Nature Communications 16, 5978. https://www.nature.com/articles/s41467-025-61037-0
- Bhatt, M. et al. (2025). "NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation." CoLLAs 2025. arXiv:2503.24322. https://arxiv.org/abs/2503.24322
- Pau, D. et al. (2024). "Forward Learning of Large Language Models by Consumer Devices." Electronics 13(2), 402. https://www.mdpi.com/2079-9292/13/2/402
- Xu, M. et al. (2024). "FwdLLM: Efficient FedLLM using Forward Gradient." USENIX ATC 2024. arXiv:2308.13894. https://arxiv.org/abs/2308.13894
- Murai, K. et al. (2026). "Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference." arXiv:2603.06728. https://arxiv.org/html/2603.06728v1
- Wang, H. et al. (2022). "DeepNet: Scaling Transformers to 1,000 Layers." arXiv:2203.00555. https://arxiv.org/abs/2203.00555
- Vincent, P. et al. (2008). "Extracting and composing robust features with denoising autoencoders." ICML 2008.
- arXiv:2509.12394 (2025). "Adaptive Spatial Goodness Encoding: Advancing and Scaling Forward-Forward Learning Without Backpropagation." https://arxiv.org/abs/2509.12394
- Scientific Reports (2025). "Training convolutional neural networks with the Forward-Forward Algorithm." https://www.nature.com/articles/s41598-025-26235-2
