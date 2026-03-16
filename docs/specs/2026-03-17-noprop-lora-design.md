# Design: NoProp + LoRA for LLM Fine-Tuning on Apple Neural Engine

**Date**: 2026-03-17
**Status**: Proposal
**Author**: Research team
**Goal**: Design and implement NoProp (arXiv:2503.24322) -- a diffusion-inspired, block-local training algorithm -- combined with LoRA for fine-tuning SmolLM2-360M on Apple Neural Engine. No forward chain. No backward chain. Each transformer layer trains independently.

---

## 0. Executive Summary

We propose **NoProp-LoRA**: a training algorithm where each transformer layer independently learns to denoise a noisy version of the target token embedding, using only local backpropagation within the layer's LoRA adapters. There is no forward propagation chain (each layer receives the original input embeddings via broadcast, not the output of the preceding layer) and no backward propagation chain (gradients never flow between layers). Only LoRA A/B matrices are updated; all base weights are frozen.

**Why this matters for ANE**: Our existing methods face fundamental tradeoffs:
- **MeZO** (zeroth-order): Forward-only, but 2 full sequential forward passes per step. Convergence ceiling at ~4.6 loss on TinyStories.
- **Backprop-LoRA** (P16): First-order gradients, but requires storing all 32 layers' activations and a full backward pass.
- **Forward-Forward**: Layer-local, but goodness function is a proxy that may not align with language modeling.

NoProp-LoRA breaks the sequential dependency entirely. Each of the 32 layers can be trained with a single forward + local backward through just that layer. The denoising objective is well-founded (connection to score matching / diffusion models). ANE runs the frozen base-weight forward pass; CPU computes the tiny LoRA gradient within each layer.

**Novelty claim**: As of March 2026, NoProp has been demonstrated only on MNIST, CIFAR-10, and CIFAR-100 image classification with MLP/CNN blocks. No published work applies NoProp to: (a) transformers, (b) language models, (c) LoRA fine-tuning, or (d) NPU hardware. This would be the first in all four axes.

**Key numbers for SmolLM2-360M (32 layers, DIM=960, SEQ=256)**:
- Trainable params per layer: ~122K (rank-8 LoRA on Wq/Wk/Wv/Wo)
- Memory per layer training: ~2.4 MB activations + 0.5 MB LoRA state = ~2.9 MB
- Total memory (all 32 layers simultaneously): ~93 MB (vs ~1.8 GB for full backprop)
- Training: 32 independent layer-local steps per sample (parallelizable)

---

## 1. NoProp Adaptation for Transformers

### 1.1 Original NoProp Algorithm (Recap)

From arXiv:2503.24322 (Kopitkov & Bhambri, CoLLAs 2025):

**Training**: Given input x and target label y:
1. Compute label embedding u_y (one-hot, learned embedding, or prototype).
2. For each block t in {1, ..., T}:
   - Sample noise level sigma_t from the schedule.
   - Create noisy target: z_t = alpha_t * u_y + sigma_t * epsilon, where epsilon ~ N(0, I).
   - Block t receives (x, z_t) and predicts u_hat_t = f_theta_t(x, z_t).
   - Loss: L_t = ||u_hat_t - u_y||^2 (MSE denoising loss).
   - Backprop within block t only. No cross-block gradients.

**Inference**: Starting from z_0 ~ N(0, I):
- For each block t: z_t = z_{t-1} + eta * (f_theta_t(x, z_{t-1}) - z_{t-1})
- After T blocks, z_T should approximate u_y. Decode to label.

**Noise schedule**: The original paper uses:
- Discrete-time (NoProp-DT): Fixed cosine schedule, alpha = linspace(1.0, 0.1, T).
- Continuous-time (NoProp-CT): Learnable SNR schedule via a small neural network.

**Performance**: CIFAR-10: 80.54% (NoProp-DT with prototypes), CIFAR-100: 46.06%. Competitive with local-learning baselines but below full backprop (~95% CIFAR-10).

### 1.2 Adaptation to Transformer Layers

A transformer layer is not a simple MLP -- it contains RMSNorm, multi-head self-attention (QKV + RoPE + SDPA + Wo), residual connections, and a SwiGLU FFN. We must carefully define what "input" and "denoising" mean in this context.

**Architecture of NoProp-adapted layer l**:

```
Inputs:
  x_embed[DIM, SEQ]   -- token embeddings from the embedding table (BROADCAST to all layers)
  y_noisy[DIM, SEQ]   -- noisy target embedding at noise level sigma_l

Processing (layer l uses its OWN frozen weights + LoRA adapters):
  1. combined[2*DIM, SEQ] = concat(x_embed, y_noisy)        -- concatenate along channel dim
  2. h[DIM, SEQ] = NoPropProjection_l(combined)               -- project 2*DIM -> DIM
  3. h = TransformerBlock_l(h)                                 -- standard: RMSNorm + Attn + Res + RMSNorm + FFN + Res
  4. y_pred[DIM, SEQ] = NoPropHead_l(h)                       -- project DIM -> DIM (denoised prediction)

Loss:
  L_l = (1 / (DIM * SEQ)) * ||y_pred - y_target||^2          -- MSE denoising loss
  Backprop ONLY through LoRA parameters of layer l.
```

**Critical design choice -- how to inject the noisy target**:

We considered three approaches:

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Concatenation** (chosen) | concat(x, y_noisy) along channel dim, project to DIM | Clean separation, standard in diffusion | Requires extra projection layer (2*DIM -> DIM) |
| B. Addition | h = x + y_noisy (element-wise) | Zero extra params | Conflates input signal with noise; attention computes on mixed signal |
| C. Cross-attention | y_noisy as queries, x as keys/values | Elegant separation | Requires architectural change to self-attention; breaks frozen weight reuse |

**We choose Approach A (concatenation)** because:
1. It is the standard approach in diffusion models (U-Net concatenates condition with noisy input).
2. The projection from 2*DIM to DIM can be implemented as a LoRA-style low-rank matrix, keeping it tiny.
3. The transformer block operates on DIM-dimensional activations, preserving compatibility with frozen base weights.

**Simplification for implementation**: Instead of a full 2*DIM -> DIM projection, we implement the injection as:

```
h[DIM, SEQ] = x_embed + W_inject_B @ (W_inject_A @ y_noisy)
```

where W_inject_A is [rank, DIM] and W_inject_B is [DIM, rank] -- a rank-8 LoRA-style injection. This adds y_noisy's information to x_embed via a low-rank pathway, avoiding a full 2*DIM -> DIM matrix (which would be 1.8M params). The injection matrices are trainable per layer.

Then the transformer layer processes h normally through its frozen weights + LoRA adapters.

### 1.3 The Denoising Head

After the transformer block produces its output, we need to extract the denoised target prediction. The output is already in R^{DIM, SEQ}, same dimension as the target embedding. We use:

```
y_pred = RMSNorm(layer_output) @ W_denoise + layer_output_residual
```

where W_denoise is either:
- The identity (simplest: the layer output IS the prediction), or
- A learned linear projection (LoRA-style: B_denoise @ A_denoise).

**We choose the identity + residual approach**: y_pred = layer_output. The transformer layer's existing residual stream and RMSNorm already produce well-scaled outputs. Adding a denoising head would increase per-layer parameters. We let the LoRA adapters within the transformer block (Wq, Wk, Wv, Wo) handle the denoising task.

This means: **the denoising objective directly trains the same LoRA adapters that will be used for inference**. No auxiliary heads, no extra parameters that get discarded.

---

## 2. LoRA Integration

### 2.1 Which Parameters Are Trained

Per layer l, the trainable parameters are:

| Parameter | Shape | Size | Purpose |
|-----------|-------|------|---------|
| W_inject_A_l | [8, 960] | 7,680 | Noisy target injection (A matrix) |
| W_inject_B_l | [960, 8] | 7,680 | Noisy target injection (B matrix) |
| Aq_l | [8, 960] | 7,680 | LoRA for Wq |
| Bq_l | [960, 8] | 7,680 | LoRA for Wq |
| Ak_l | [8, 960] | 7,680 | LoRA for Wk |
| Bk_l | [320, 8] | 2,560 | LoRA for Wk |
| Av_l | [8, 960] | 7,680 | LoRA for Wv |
| Bv_l | [320, 8] | 2,560 | LoRA for Wv |
| Ao_l | [8, 960] | 7,680 | LoRA for Wo |
| Bo_l | [960, 8] | 7,680 | LoRA for Wo |
| rms_att_l | [960] | 960 | RMSNorm (attention) |
| rms_ffn_l | [960] | 960 | RMSNorm (FFN) |
| **Total per layer** | | **~76K** | |
| **Total (32 layers)** | | **~2.4M** | |

Note: The injection matrices (W_inject_A, W_inject_B) are additional NoProp-specific parameters beyond standard LoRA. They are needed only during NoProp training and can be discarded at inference time when switching to standard autoregressive generation.

### 2.2 Forward Pass for NoProp-LoRA Layer l

```
// INPUTS: x_embed[DIM, SEQ], y_noisy[DIM, SEQ]
// FROZEN: lw[l].Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn
// TRAINABLE: lora[l].Aq, Bq, Ak, Bk, Av, Bv, Ao, Bo, inject_A, inject_B

// Step 1: Inject noisy target into input embedding
//   h = x_embed + inject_B @ (inject_A @ y_noisy)   [DIM, SEQ]
tmp_r[rank, SEQ] = inject_A[rank, DIM] @ y_noisy[DIM, SEQ]
h[DIM, SEQ] = x_embed + inject_B[DIM, rank] @ tmp_r[rank, SEQ]

// Step 2: Standard transformer layer forward (frozen base + LoRA addmm)
xnorm = rmsnorm(h, rms_att)
Q = Wq_base @ xnorm + Bq @ (Aq @ xnorm)
K = Wk_base @ xnorm + Bk @ (Ak @ xnorm)
V = Wv_base @ xnorm + Bv @ (Av @ xnorm)
Q, K = rope(Q), rope(K)
attn_out = sdpa(Q, K, V)
o_out = Wo_base @ attn_out + Bo @ (Ao @ attn_out)
x2 = h + alpha * o_out              // residual
x2norm = rmsnorm(x2, rms_ffn)
h1 = W1 @ x2norm                    // FFN frozen (no LoRA)
h3 = W3 @ x2norm
ffn_out = W2 @ (silu(h1) * h3)
y_pred = x2 + alpha * ffn_out       // final output = denoised prediction

// Step 3: Denoising loss
loss_l = mean((y_pred - y_target)^2)

// Step 4: Local backward through LoRA params only
// Gradients: d(loss_l)/d(Aq), d(loss_l)/d(Bq), ..., d(loss_l)/d(inject_A), d(loss_l)/d(inject_B)
// NO gradient flows to other layers.
```

### 2.3 Local Backward Pass

The backward pass is confined to layer l. We compute:

1. **d_y_pred** = (2 / (DIM * SEQ)) * (y_pred - y_target)  [DIM, SEQ]
2. Backprop through FFN residual: d_x2 = d_y_pred, d_ffn = alpha * d_y_pred
3. Backprop through frozen W2, SiLU, W1/W3 to get d_x2norm (no weight gradients for frozen FFN)
4. Backprop through RMSNorm(ffn) to get d_x2_input
5. d_h_attn = d_x2_input, d_o_out = alpha * d_x2_input
6. Backprop through Wo (frozen) to get d_attn_out; project to LoRA grads: dAo, dBo
7. Backprop through SDPA to get dQ, dK, dV
8. Backprop through RoPE (inverse rotation)
9. Backprop through Wq/Wk/Wv (frozen) to get d_xnorm; project to LoRA grads: dAq, dBq, dAk, dBk, dAv, dBv
10. Backprop through RMSNorm(att) to get d_h
11. Backprop through injection: d_inject_B, d_inject_A from d_h

This is exactly the same backward pass as our existing backprop_lora.h (P16 hybrid), except:
- We stop at step 11 (no gradient flows to previous layers).
- We additionally compute gradients for inject_A and inject_B.
- The input to the layer is (x_embed + injection) rather than the previous layer's output.

**Memory**: The backward requires storing one layer's activations at a time: ~2.4 MB. Compare to full backprop which stores all 32 layers simultaneously: ~77 MB.

---

## 3. Noise Schedule

### 3.1 Design

NoProp maps each layer index l in {0, 1, ..., 31} to a noise level sigma_l:

```
Layer 0:  sigma = 1.0   (pure Gaussian noise, target is fully corrupted)
Layer 31: sigma = 0.0   (clean target, no noise)
Layer l:  sigma_l = 1.0 - l / (NLAYERS - 1)    [linear interpolation]
```

The noisy target at layer l is:

```
y_noisy_l = (1 - sigma_l) * y_target + sigma_l * epsilon
          = (l / 31) * y_target + (1 - l/31) * epsilon

where epsilon ~ N(0, I) is sampled ONCE per training step and shared across layers.
```

This is a **flow matching** (linear interpolation) schedule rather than a variance-preserving diffusion schedule. We choose flow matching because:
1. The original NoProp paper reports both variants; flow matching is simpler.
2. Linear interpolation has a cleaner gradient landscape (no SNR clipping issues).
3. The target representation is in embedding space (bounded), so variance preservation is less critical.

### 3.2 Alternative Schedules Considered

| Schedule | Formula | Rationale | Issue |
|----------|---------|-----------|-------|
| **Linear** (chosen) | sigma_l = 1 - l/31 | Simple, symmetric | May concentrate signal in middle layers |
| Cosine | sigma_l = cos(pi*l/(2*31)) | More time at low noise | Biases toward clean; early layers wasted |
| Learned | sigma_l = MLP(l/31) | Adaptive | Extra parameters, harder to debug |
| Variance-preserving | z = sqrt(alpha)*y + sqrt(1-alpha)*eps | Classic DDPM | Requires careful scaling for embeddings |

### 3.3 Noise Sharing vs. Independent Noise

**Shared noise** (chosen): Sample epsilon ~ N(0, I) once per training step. All 32 layers see the same random noise, but at different mixing ratios with y_target. This ensures coherent denoising across the layer chain during inference.

**Independent noise** (alternative): Each layer samples its own epsilon_l. Simpler mathematically but breaks the sequential denoising interpretation at inference time.

---

## 4. Target Representation

### 4.1 What IS the Target for Language Modeling?

For classification (NoProp's original domain), the target is a class label -- easily represented as a one-hot vector or learned embedding. For language modeling, the "target" at each sequence position t is the next token y_t = tokens[t+1].

We analyze four options:

### Option A: Token Embedding of the Correct Next Token

```
y_target[d, t] = embed[next_token[t]][d]    for d in {0..DIM-1}, t in {0..SEQ-1}
```

Dimension: [DIM, SEQ] = [960, 256]. Same dimension as the transformer's hidden state.

**Pros**:
- Natural: the embedding table already maps tokens to DIM-dimensional vectors.
- Same dimensionality as hidden states -- no projection needed.
- The pretrained embedding table encodes semantic similarity (similar tokens have similar embeddings).
- At inference, we decode by finding the nearest embedding: argmin_v ||y_pred[:, t] - embed[v]||^2.

**Cons**:
- The embedding space may not be ideal for denoising (embeddings are normalized, clustered).
- Multiple tokens may have similar embeddings, making the MSE loss ambiguous.
- The embedding table is 49152 x 960 = 47M params -- large for nearest-neighbor search.

### Option B: One-Hot Logit Vector

```
y_target[v, t] = 1 if v == next_token[t], else 0    for v in {0..VOCAB-1}, t in {0..SEQ-1}
```

Dimension: [VOCAB, SEQ] = [49152, 256]. Enormous.

**Pros**: Unambiguous target. Standard classification target.
**Cons**: Dimension 49152 is absurd for denoising. Would require per-layer projection heads. Memory: 49152 * 256 * 4 = 50 MB per layer. Rejected.

### Option C: Hidden State from a Teacher Model

```
y_target = teacher_model.layer[l].output(input_tokens)
```

**Pros**: Layer-specific targets. Proven in knowledge distillation.
**Cons**: Requires running a teacher model (doubles inference cost). The teacher IS the pretrained model -- we would be distilling it into itself. Circular.

### Option D: Compact Learned Embedding (Recommended Variant of A)

```
y_target[d, t] = compact_target_embed[next_token[t]][d]   where d in {0..TARGET_DIM-1}
```

Use a separate, smaller target embedding table of dimension TARGET_DIM < DIM (e.g., TARGET_DIM = 64 or 128). This is what the original NoProp paper calls "learned embeddings" (they use dim=20 for CIFAR-10).

**Pros**:
- Smaller target space = easier denoising problem.
- Learned embeddings can be optimized jointly with the denoiser.
- Reduces memory: 49152 * 64 * 4 = 12.6 MB for the target embedding table.

**Cons**:
- Adds a trainable embedding table (12.6 MB at TARGET_DIM=64).
- At inference, must project from DIM to TARGET_DIM for decoding.
- Introduces a new hyperparameter (TARGET_DIM).

### 4.2 Recommendation: Option A (Token Embedding) with Fallback to Option D

**Primary**: Use the pretrained token embedding table directly. y_target[DIM, SEQ] is the embedding lookup of the next-token sequence. This is the simplest approach, requires zero extra parameters, and leverages the pretrained embedding geometry.

**Decoding at inference**: After the final layer (layer 31) produces y_pred[DIM, SEQ], decode each position by computing:

```
logits[t] = y_pred[:, t] @ embed^T      // [VOCAB] = [DIM]^T @ [DIM, VOCAB]
token[t] = argmax(logits[t])
```

This is exactly the standard LM head (logits = hidden @ embed^T), applied to the NoProp-denoised output.

**Fallback (Option D)**: If Option A fails to converge (denoising in 960-dim space is too hard), fall back to a compact target embedding of dimension 64 or 128. This adds ~12 MB but makes the denoising problem much easier.

### 4.3 Compact Vocab for Efficiency

Following the existing codebase pattern (VocabMap in cpu_ops.h), we use compact vocab during training. The target embedding lookup only touches tokens present in the training data, reducing the effective vocabulary from 49152 to typically ~5000-10000 tokens. Nearest-neighbor decoding at inference also uses compact vocab.

---

## 5. Inference: Sequential Denoising

### 5.1 Inference Algorithm

At test time, NoProp-LoRA performs sequential denoising across the 32 layers:

```
// Input: x_embed[DIM, SEQ] = embedding lookup of input tokens
// Initialize: z_0[DIM, SEQ] ~ N(0, I)   (pure noise)

for l = 0 to 31:
    // Layer l denoises z_l → z_{l+1}
    sigma_l = 1.0 - l / 31.0

    // Inject z_l into layer l's input
    h = x_embed + inject_B_l @ (inject_A_l @ z_l)

    // Run transformer layer l (frozen base + LoRA)
    y_pred_l = transformer_layer_l(h)

    // Denoising step: move z toward the prediction
    eta = 1.0 / NLAYERS    // step size
    z_{l+1} = z_l + eta * (y_pred_l - z_l)

// Decode final output
logits = z_32 @ embed^T
tokens = argmax(logits, dim=vocab)
```

### 5.2 Inference Cost Analysis

Each inference step requires running one transformer layer, identical to the standard forward pass cost of that layer. The total cost is:

```
NoProp inference = 32 layers * (1 layer forward) = 1 full forward pass
```

This is the **same cost as standard inference**. The only overhead is:
- Generating z_0 from N(0, I): negligible.
- The inject_A/inject_B matmul per layer: 2 * rank * DIM * SEQ = 2 * 8 * 960 * 256 = 3.9M FLOPs per layer. Compare to Wq matmul: DIM * Q_DIM * SEQ = 960 * 960 * 256 = 236M FLOPs. Overhead: 1.6%.

### 5.3 Hybrid Inference Mode (Practical)

In practice, we may not want pure NoProp inference. An alternative "hybrid" mode:

1. Run the standard autoregressive forward pass (all 32 layers sequentially, as in normal inference).
2. Take the final hidden state h_final[DIM, SEQ].
3. Compute logits = h_final @ embed^T.

The LoRA adapters trained via NoProp are already merged into the weight matrices. The standard forward pass will use them. **The NoProp denoising structure is only for training -- at inference, the model runs normally.**

This is a critical insight: **NoProp is a training algorithm, not an inference algorithm**. The LoRA adapters it trains are standard LoRA adapters that can be used with normal autoregressive inference. The denoising interpretation is what enables block-local training, but the learned adapters function identically to adapters trained with backprop.

**However**, there is a gap: the adapters were trained with a denoising objective (predict clean target from noisy input), not the standard next-token prediction objective. Whether these adapters generalize to standard inference is the central open question (see Section 8).

---

## 6. Implementation Plan

### 6.1 New Files

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `training/train_noprop.m` | Main training loop for NoProp-LoRA | ~400 |
| `training/noprop_types.h` | NoProp-specific types (injection matrices, noise schedule) | ~80 |

### 6.2 Modified Files

| File | Changes |
|------|---------|
| `training/config.h` | Add NoPropLayer struct (injection A/B matrices + Adam state) |
| `training/Makefile` | Add `noprop` target |

### 6.3 Core Data Structures (noprop_types.h)

```c
// NoProp injection matrices per layer (trainable, for noisy target injection)
typedef struct {
    float *inject_A;    // [rank, DIM] project y_noisy down
    float *inject_B;    // [DIM, rank] project back up
} NoPropInject;

// NoProp per-layer training state
typedef struct {
    NoPropInject inject;
    AdamState adam_inject_A;
    AdamState adam_inject_B;
    // Activation storage for local backward (only 1 layer at a time)
    float *h_input;         // [DIM, SEQ] layer input (after injection)
    float *y_pred;          // [DIM, SEQ] layer output (denoised prediction)
} NoPropLayerState;

// Noise schedule
typedef struct {
    int nlayers;
    float *sigma;           // [NLAYERS] noise level per layer: sigma[l] = 1 - l/(L-1)
    float *alpha;           // [NLAYERS] signal level: alpha[l] = l/(L-1)
} NoPropSchedule;

static NoPropSchedule noprop_schedule_init(int nlayers) {
    NoPropSchedule s;
    s.nlayers = nlayers;
    s.sigma = (float*)safe_malloc(nlayers * sizeof(float));
    s.alpha = (float*)safe_malloc(nlayers * sizeof(float));
    for (int l = 0; l < nlayers; l++) {
        s.alpha[l] = (float)l / (float)(nlayers - 1);
        s.sigma[l] = 1.0f - s.alpha[l];
    }
    return s;
}
```

### 6.4 Training Loop Pseudocode (train_noprop.m, ~200 key lines)

```c
// === NoProp Training Loop ===
for (int step = 0; step < total_steps; step++) {
    // 1. Sample batch: input_tokens[SEQ], target_tokens[SEQ]
    size_t pos = (((uint64_t)step * 7 + init_seed) % (train_tokens - SEQ - 1));
    uint16_t *input_tokens = token_data + pos;
    uint16_t *target_tokens = token_data + pos + 1;

    // 2. Compute x_embed[DIM, SEQ] = embed_lookup(input_tokens)
    embed_lookup(x_embed, embed, input_tokens, DIM, SEQ, VOCAB);

    // 3. Compute y_target[DIM, SEQ] = embed_lookup(target_tokens)
    embed_lookup(y_target, embed, target_tokens, DIM, SEQ, VOCAB);

    // 4. Sample noise: epsilon[DIM, SEQ] ~ N(0, I)
    //    (one noise sample shared across all layers)
    xo_seed(step * 999983ULL + init_seed);
    gaussian_fill(noise_buf, (size_t)DIM * SEQ);

    // 5. For each layer l (can be parallelized, but sequential for simplicity):
    float total_loss = 0;
    for (int l = 0; l < NLAYERS; l++) {
        float alpha_l = schedule.alpha[l];   // l / 31
        float sigma_l = schedule.sigma[l];   // 1 - l/31

        // 5a. Create noisy target for this layer
        //     y_noisy_l = alpha_l * y_target + sigma_l * noise_buf
        for (size_t i = 0; i < (size_t)DIM * SEQ; i++)
            y_noisy[i] = alpha_l * y_target[i] + sigma_l * noise_buf[i];

        // 5b. Inject noisy target into input
        //     h = x_embed + inject_B_l @ (inject_A_l @ y_noisy)
        memcpy(h_buf, x_embed, DIM * SEQ * sizeof(float));
        lora_addmm(h_buf, noprop[l].inject.inject_A, noprop[l].inject.inject_B,
                    y_noisy, lora_tmp, DIM, rank, DIM);

        // 5c. Save layer input for backward
        memcpy(bp_acts[0].x_pre, h_buf, DIM * SEQ * sizeof(float));

        // 5d. Forward through transformer layer l (frozen + LoRA)
        //     Uses existing forward code: RMSNorm -> QKV+LoRA -> RoPE -> SDPA -> Wo+LoRA -> Res -> FFN -> Res
        noprop_layer_forward(l, h_buf, lw, lora_layers, &bp_acts[0],
                             Q, K, V, k_tiled, v_tiled, attn_out, o_out,
                             xnorm_buf, h1, h3, silu_out, lora_tmp);
        // h_buf now contains y_pred_l[DIM, SEQ]

        // 5e. Compute MSE denoising loss
        float layer_loss = 0;
        for (size_t i = 0; i < (size_t)DIM * SEQ; i++) {
            float diff = h_buf[i] - y_target[i];
            layer_loss += diff * diff;
        }
        layer_loss /= (float)(DIM * SEQ);
        total_loss += layer_loss;

        // 5f. Compute d_y_pred = (2 / (DIM*SEQ)) * (y_pred - y_target)
        float grad_scale = 2.0f / (float)(DIM * SEQ);
        for (size_t i = 0; i < (size_t)DIM * SEQ; i++)
            dy_buf[i] = grad_scale * (h_buf[i] - y_target[i]);

        // 5g. Local backward through layer l
        //     Reuse backprop_lora.h backward but stop at layer boundary.
        //     Compute: d(inject_A), d(inject_B), d(Aq), d(Bq), ... d(rms_att), d(rms_ffn)
        noprop_layer_backward(l, dy_buf, lw, lora_layers, &bp_acts[0],
                              &lora_grads[l], &bp_work,
                              Q_rope, K_rope, k_tiled, v_tiled);

        // 5g'. Backward through injection: d_h -> d(inject_A), d(inject_B)
        //      d(inject_B) += d_h @ (inject_A @ y_noisy)^T
        //      d(inject_A) += inject_B^T @ d_h @ y_noisy^T
        //      (Reuse lora_grad_project with appropriate reshaping)
        noprop_inject_backward(l, dy_buf /* actually d_h from rmsnorm bwd */,
                               y_noisy, noprop, lora_tmp);

        // 5h. Adam update for layer l's LoRA params + injection params
        adam_t_l[l]++;
        // Update Aq, Bq, Ak, Bk, Av, Bv, Ao, Bo
        adam_update(lora_layers[l].Aq, lora_grads[l].Aq, &lora_adam[l].Aq,
                    adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.01f);
        adam_update(lora_layers[l].Bq, lora_grads[l].Bq, &lora_adam[l].Bq,
                    adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.01f);
        // ... repeat for all LoRA matrices ...
        // Update inject_A, inject_B
        adam_update(noprop[l].inject.inject_A, noprop_grads[l].inject_A,
                    &noprop[l].adam_inject_A, adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.0f);
        adam_update(noprop[l].inject.inject_B, noprop_grads[l].inject_B,
                    &noprop[l].adam_inject_B, adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.0f);
        // Update rms_att, rms_ffn
        adam_update(lw[l].rms_att, grms_att[l], &la_rms_att[l],
                    adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.0f);
        adam_update(lw[l].rms_ffn, grms_ffn[l], &la_rms_ffn[l],
                    adam_t_l[l], lr, 0.9f, 0.999f, 1e-8f, 0.0f);

        // 5i. Zero gradients for next layer
        lora_grads_zero(&lora_grads[l], rank);
        memset(grms_att[l], 0, DIM * sizeof(float));
        memset(grms_ffn[l], 0, DIM * sizeof(float));
        noprop_grads_zero(&noprop_grads[l], rank);

        // 5j. Re-merge LoRA for next forward pass (if using merged mode)
        lora_merge_weight(lw[l].Wq, lora_layers[l].Wq_base, lora_layers[l].Bq,
                          lora_layers[l].Aq, Q_DIM, rank, DIM);
        // ... repeat for Wk, Wv, Wo ...
    }

    // 6. Logging
    float avg_loss = total_loss / NLAYERS;
    if (step % 100 == 0)
        printf("step %d  avg_denoise_loss=%.4f  lr=%.2e\n", step, avg_loss, lr);
}
```

### 6.5 Key Implementation Functions

**noprop_layer_forward()**: Identical to the body of DO_FORWARD_PASS for a single layer, but:
- Input is h_buf (post-injection) rather than x_cur (from previous layer).
- Saves activations into bp_acts for local backward.
- Uses lora_split mode (frozen base on ANE/CPU, LoRA addmm on CPU).

**noprop_layer_backward()**: Reuses the backward pass from backprop_lora.h, but:
- Input gradient is d_y_pred (from MSE loss), not d_logits (from cross-entropy).
- Stops at the layer boundary (does not propagate d_h to previous layer).
- Computes LoRA gradients via lora_grad_project() for Wq/Wk/Wv/Wo.

**noprop_inject_backward()**: New function, ~20 lines:
```c
static void noprop_inject_backward(int l, const float *d_h, const float *y_noisy,
                                    NoPropLayerState *noprop, float *tmp_r) {
    int rank = NOPROP_RANK;
    // d(inject_B) += d_h @ (A @ y_noisy)^T  =>  d_h[DIM,SEQ] @ tmp_r^T[SEQ,rank]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, SEQ, DIM, 1.0f,
                noprop[l].inject.inject_A, DIM, y_noisy, SEQ, 0.0f, tmp_r, SEQ);
    // dB[DIM, rank] += d_h[DIM, SEQ] @ tmp_r^T[SEQ, rank]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                DIM, rank, SEQ, 1.0f,
                d_h, SEQ, tmp_r, SEQ, 1.0f,
                noprop_grads[l].inject_B, rank);
    // dA[rank, DIM] += inject_B^T[rank, DIM] @ d_h[DIM, SEQ] ... @ y_noisy^T
    // Actually: d(inject_A) comes from chain rule through B @ (A @ y_noisy)
    // d_tmp_r[rank, SEQ] = inject_B^T[rank, DIM] @ d_h[DIM, SEQ]
    float *d_tmp = tmp_r;  // reuse buffer
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                rank, SEQ, DIM, 1.0f,
                noprop[l].inject.inject_B, rank, d_h, SEQ, 0.0f, d_tmp, SEQ);
    // dA[rank, DIM] += d_tmp[rank, SEQ] @ y_noisy^T[SEQ, DIM]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rank, DIM, SEQ, 1.0f,
                d_tmp, SEQ, y_noisy, SEQ, 1.0f,
                noprop_grads[l].inject_A, DIM);
}
```

### 6.6 ANE Integration

For the ANE-accelerated mode:
- **Forward**: Use conv-fused kernels (qkvConv, ffnConv) for the frozen base-weight matmuls, same as existing backprop_lora mode.
- **LoRA addmm**: CPU-side, same as existing lora_split mode.
- **Injection matmul**: CPU-side (tiny: rank-8, ~4K FLOPs).
- **Backward**: CPU-side, reusing backprop_lora.h functions.

The ANE evaluation pattern per layer is identical to P16:
1. Write xnorm to QKV conv input IOSurface.
2. ANE eval qkvConv[l].
3. Read Q, K, V from output IOSurface.
4. CPU: LoRA addmm for Q, K, V corrections.
5. CPU: RoPE, SDPA, Wo (ANE conv + LoRA addmm), residual.
6. ANE eval ffnConv[l] (or CPU if --cpu-only).
7. CPU: residual to get y_pred.

### 6.7 Build and CLI

```makefile
# Makefile addition
noprop: training/train_noprop.m training/noprop_types.h training/config.h training/cpu_ops.h training/backprop_lora.h
	$(CC) $(CFLAGS) -include models/$(MODEL).h -o train_noprop training/train_noprop.m $(LDFLAGS)
```

```bash
# Usage
./train_noprop --resume ckpt.bin --data data.bin --lora --lora-rank 8 --lr 1e-4 \
    --steps 5000 --cpu-only --noprop-schedule linear --target-embed input
```

CLI flags:
- `--noprop-schedule {linear|cosine|learned}`: Noise schedule type.
- `--target-embed {input|compact-64|compact-128}`: Target representation.
- `--noprop-eta <float>`: Inference denoising step size (default: 1/NLAYERS).
- `--denoise-eval`: At validation, use NoProp denoising inference instead of standard forward.

---

## 7. Hypotheses

### Hypothesis 1: NoProp-LoRA achieves lower denoising loss than random baseline

**Claim**: After 1000 training steps, the average denoising MSE loss across all 32 layers decreases from the initial (untrained) value.

**Success criterion**: Average MSE loss at step 1000 < 0.5 * Average MSE loss at step 0.

**Rationale**: This tests the most basic question: can the LoRA adapters learn to denoise at all? The untrained model maps (x_embed + noisy_target) through frozen weights, producing output uncorrelated with y_target. After training, the LoRA adapters should steer the output toward y_target. If this fails, the injection mechanism or loss function is broken.

**Measurement**: Log per-layer denoising MSE at steps 0, 100, 500, 1000. Plot learning curves per layer.

### Hypothesis 2: Later layers (lower noise) achieve lower denoising loss than early layers (higher noise)

**Claim**: Layer 31 (sigma=0, clean input) achieves lower denoising MSE than layer 0 (sigma=1, pure noise).

**Success criterion**: MSE(layer 31) < 0.3 * MSE(layer 0) after 2000 steps.

**Rationale**: Layer 31 receives the clean target as input (sigma=0), so its denoising task is trivial -- just pass through the target. Layer 0 receives pure noise, so it must reconstruct the target entirely from x_embed. The loss gradient should be monotonically decreasing with layer index. If it is not, the noise schedule or injection mechanism is misconfigured.

**Measurement**: Plot MSE vs layer index at convergence. Expect monotone decreasing curve.

### Hypothesis 3: NoProp-LoRA adapters produce coherent text when used in standard inference mode

**Claim**: After 5000 NoProp training steps, running the model in standard autoregressive mode (ignoring the denoising structure) produces text with lower perplexity than the base model without LoRA adapters.

**Success criterion**: Validation cross-entropy loss with NoProp-trained LoRA adapters < validation cross-entropy loss with zero-initialized LoRA adapters (base model).

**Rationale**: This is the crucial test. NoProp trains each layer's LoRA adapters to denoise, but we want those adapters to also improve standard next-token prediction. The hypothesis is that learning to predict clean target embeddings from noisy versions also teaches the layer useful representations for language modeling. This is plausible because:
- The denoising task forces each layer to extract information about the target from the input context x_embed.
- Layers that extract useful features for denoising should also extract useful features for next-token prediction.
- The LoRA adapters modify Q, K, V, O projections -- the same projections used in standard inference.

**Failure modes**:
- The adapters could learn denoising-specific features that are orthogonal to language modeling.
- The injection pathway (inject_A, inject_B) provides a "shortcut" that does not transfer to standard inference (where there is no y_noisy input).
- The MSE loss in embedding space may not align with cross-entropy loss in logit space.

**Measurement**: After NoProp training, run standard forward pass (no denoising) and compute cross-entropy loss on validation set. Compare to base model. Also compare to MeZO-LoRA and backprop-LoRA baselines at same number of training steps.

---

## 8. The Key Question: Does NoProp's Proxy Objective Produce Useful Fine-Tuning?

### 8.1 The Fundamental Gap

MeZO optimizes the **actual loss** (cross-entropy on next-token prediction) using zeroth-order gradients of the true objective. Its estimate is unbiased: E[g_MeZO] = true gradient (in expectation over perturbation directions).

NoProp optimizes a **proxy loss** (MSE denoising in embedding space) using first-order gradients of the proxy. The proxy is well-defined and locally backproppable, but there is no guarantee that minimizing the denoising loss minimizes the language modeling loss.

### 8.2 When the Proxy IS Sufficient

The denoising objective can be sufficient for fine-tuning when:

1. **The target embedding captures the task signal**: If embed(next_token) faithfully represents what the model needs to predict, then learning to reconstruct it from context is equivalent to learning next-token prediction. This is true when the embedding table captures token semantics (which pretrained models generally ensure).

2. **The denoising task requires extracting the same features as language modeling**: Each layer must learn "given this context x_embed, what is the next token?" -- whether expressed as an embedding (NoProp) or a probability distribution (standard LM). The information content is the same; only the output representation differs.

3. **The LoRA adapters are capacity-limited**: With rank-8 LoRA, the adapters can only make small perturbations to the base model. Any denoising-useful perturbation must also be LM-useful, because there are not enough degrees of freedom to learn denoising-specific features that are orthogonal to LM features.

### 8.3 When the Proxy FAILS

The denoising objective fails when:

1. **MSE in embedding space does not correlate with cross-entropy**: If two tokens have similar embeddings but very different next-token distributions, the MSE loss is small but the CE loss is large. Example: "the" and "a" might have nearby embeddings but lead to different continuations. The denoiser would conflate them.

2. **The injection pathway creates a shortcut**: If the model learns to pass y_noisy through the injection directly to the output (bypass the transformer), the LoRA adapters learn nothing useful. This is mitigated by: (a) the injection is low-rank (rank-8), (b) early layers have high noise (sigma ~ 1), so the injection carries little target information.

3. **Layer independence hurts**: In a standard transformer, each layer builds on the previous layer's representation. NoProp breaks this chain -- each layer sees x_embed, not the output of the previous layer. The LoRA adapters may learn redundant features (all layers learn the same thing) rather than complementary features.

### 8.4 Theoretical Analysis: Connection to Score Matching

NoProp's denoising objective is connected to denoising score matching (Vincent, 2011). The score function of the data distribution p(y|x) can be estimated by training a denoiser:

```
E[||f(y + sigma*eps) - y||^2] = sigma^2 * E[||nabla_y log p(y|x)||^2] + const
```

If we view each transformer layer as estimating the score at a different noise level, then the 32 layers collectively estimate the score function of the conditional distribution p(next_token_embedding | context). This score function determines the distribution -- so in principle, the NoProp layers contain all the information needed for next-token prediction.

**However**, standard inference does not use the score function. It uses the forward pass to compute logits. The NoProp-trained LoRA adapters encode the score function, but standard inference decodes them as if they encode forward-pass computations. This mismatch is the fundamental risk.

### 8.5 Mitigation: Hybrid Training

To bridge the gap between proxy and true objective, we can use a two-phase approach:

**Phase 1 (NoProp warmup, steps 0-3000)**: Train LoRA adapters with the NoProp denoising objective. This is fast (no cross-layer dependencies) and gets the adapters into a good region of parameter space.

**Phase 2 (Standard fine-tuning, steps 3000-5000)**: Switch to standard backprop-LoRA (or MeZO-LoRA) training with the true cross-entropy loss. This refines the adapters from the NoProp-initialized state.

The hypothesis is that NoProp provides a much better initialization than random (zero) initialization, even if the denoising objective is not directly optimizing perplexity. This is analogous to how pretraining with masked language modeling (a proxy task) provides good initialization for downstream fine-tuning.

### 8.6 Expected Outcomes

| Scenario | NoProp Denoising Loss | Standard Eval CE Loss | Interpretation |
|----------|----------------------|----------------------|----------------|
| **Best case** | Decreasing, converges | Lower than base model | NoProp directly improves LM quality |
| **Good case** | Decreasing, converges | Neutral (same as base) | NoProp learns useful features but MSE != CE |
| **Hybrid win** | Decreasing, converges | Lower after Phase 2 | NoProp provides good initialization |
| **Failure** | Decreasing, converges | Higher than base model | Denoising features hurt LM; injection shortcut |
| **Total failure** | Does not decrease | N/A | LoRA cannot learn denoising; architecture bug |

### 8.7 Comparison to Other Methods

| Method | Objective | Cross-layer? | Local backprop? | Optimizes true loss? | Expected convergence |
|--------|-----------|-------------|----------------|---------------------|---------------------|
| Backprop-LoRA (P16) | CE loss | Yes (full chain) | No (global BP) | Yes | Fastest |
| MeZO-LoRA | CE loss | Yes (full fwd) | No (ZO global) | Yes (unbiased) | Slow, noisy |
| Forward-Forward-LoRA | Goodness | No | Yes | No (proxy) | Unknown |
| **NoProp-LoRA** | MSE denoise | **No** | **Yes** | **No (proxy)** | Unknown |

NoProp-LoRA and Forward-Forward-LoRA occupy the same quadrant (local, proxy). NoProp has the advantage of a theoretically grounded proxy (score matching) versus Forward-Forward's ad-hoc goodness function. NoProp's disadvantage is the injection mechanism, which adds complexity.

---

## 9. Ablation Studies

To validate the design, run these ablations:

1. **Noise schedule**: Linear vs cosine vs reverse-linear. Compare per-layer MSE curves.
2. **Target representation**: Token embedding (DIM=960) vs compact embedding (DIM=64). Compare convergence speed.
3. **Injection rank**: 4 vs 8 vs 16. Does higher rank help denoising but hurt standard inference?
4. **With vs without injection**: Replace inject_A/B with simple element-wise addition (h = x + sigma * y_noisy). Tests whether the injection mechanism matters.
5. **NoProp-only vs NoProp+finetune**: Compare pure NoProp adapters with NoProp-initialized + standard backprop-LoRA fine-tuning.
6. **Layer sampling**: Instead of training all 32 layers per step, randomly sample K layers. Compare K=1, K=8, K=32. The original NoProp paper trains all blocks per step.

---

## 10. Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| Denoising loss does not decrease | Critical | Low | Verify injection mechanism, check gradients |
| Adapters hurt standard inference | High | Medium | Hybrid training (Phase 1 NoProp + Phase 2 standard) |
| Injection shortcut | High | Medium | Low injection rank, high early noise |
| Layer redundancy | Medium | High | Accept: each layer independently useful is OK |
| Memory overhead from activation storage | Low | Low | Only 1 layer's activations at a time (2.4 MB) |
| Implementation complexity | Medium | Low | Reuse backprop_lora.h backward |

---

## 11. Timeline

| Day | Task |
|-----|------|
| 1 | Implement noprop_types.h: data structures, noise schedule, injection alloc/free |
| 1-2 | Implement train_noprop.m: forward pass with injection, MSE loss |
| 2-3 | Implement local backward: reuse backprop_lora.h + new injection backward |
| 3 | Integration: Adam update, logging, checkpoint save/load |
| 4 | Test Hypothesis 1: denoising loss decreases (1000 steps, cpu-only) |
| 4-5 | Test Hypothesis 2: per-layer loss gradient (2000 steps) |
| 5-6 | Test Hypothesis 3: standard eval with NoProp adapters (5000 steps) |
| 6-7 | Ablations: noise schedule, target representation, injection rank |
| 7 | Write results, compare to MeZO and backprop-LoRA baselines |

---

## 12. References

- [NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation](https://arxiv.org/abs/2503.24322) -- Kopitkov & Bhambri, CoLLAs 2025. Core algorithm.
- [NoProp GitHub Implementation](https://github.com/Sid3503/NoProp) -- PyTorch reference implementation for MNIST/CIFAR.
- [Denoising Score Matching](https://www.iro.umontreal.ca/~vin101/publications/smdae_techreport.pdf) -- Vincent, 2011. Theoretical foundation.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) -- Lipman et al., 2023. Linear interpolation schedule.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) -- Hu et al., 2021.
- [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333) -- Malladi et al., 2023.
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2410.21357) -- ICLR 2025. Related work: diffusion for language.
- [DiffusionBlocks: Block-wise Neural Network Training via Denoising](https://arxiv.org/abs/2506.14202) -- Independent parallel work extending NoProp.

---

## Appendix A: Memory Budget

```
Per-layer activation storage (for local backward):
  x_pre     = DIM * SEQ * 4   =  960 * 256 * 4 =  983 KB
  xnorm     = DIM * SEQ * 4   =  983 KB
  Q         = Q_DIM * SEQ * 4 =  960 * 256 * 4 =  983 KB
  K         = KV_DIM * SEQ * 4 = 320 * 256 * 4 =  328 KB
  V         = KV_DIM * SEQ * 4 =  328 KB
  attn_out  = Q_DIM * SEQ * 4 =  983 KB
  x2        = DIM * SEQ * 4   =  983 KB
  x2norm    = DIM * SEQ * 4   =  983 KB
  h1        = HIDDEN * SEQ * 4 = 2560 * 256 * 4 = 2,621 KB
  h3        = HIDDEN * SEQ * 4 = 2,621 KB
  silu_out  = HIDDEN * SEQ * 4 = 2,621 KB
  ──────────────────────────────────────────────
  TOTAL per layer:              ~14.3 MB (for activation storage)

But only 1 layer's activations are alive at a time!
Compare to full backprop: 32 layers * 14.3 MB = 458 MB.

Per-layer LoRA state:
  8 matrices * rank * DIM * 4 bytes ~ 8 * 8 * 960 * 4 = 245 KB
  Adam state (m + v): 2 * 245 KB = 490 KB
  Injection A + B + Adam: ~123 KB
  ──────────────────────────────────────────────
  Total LoRA per layer: ~858 KB

All 32 layers LoRA state: 32 * 858 KB = ~27 MB (all must be in memory)

Working buffers (backward):
  dy, dffn, dsilu, dh1, dh3, dx_ffn, dx2, da, dx_attn, dq, dk, dv, etc.
  ~same as activations: ~14 MB

Grand total memory:
  Activations (1 layer): 14.3 MB
  Working buffers:        14.0 MB
  LoRA state (32 layers): 27.0 MB
  Base weights (frozen):  1,400 MB (in mmap, shared)
  Noise + target buffers: 2 * DIM * SEQ * 4 = 2 MB
  ──────────────────────────────────────────────
  Training overhead:      ~57 MB (vs ~500+ MB for full backprop)
```

## Appendix B: NoProp vs. Standard Forward Pass Comparison

```
STANDARD FORWARD PASS (sequential, layer l depends on layer l-1):
  x_0 = embed(tokens)
  x_1 = layer_0(x_0)
  x_2 = layer_1(x_1)
  ...
  x_32 = layer_31(x_31)
  logits = x_32 @ embed^T

NOPROP TRAINING (independent, no cross-layer dependency):
  x = embed(tokens)
  y = embed(target_tokens)
  eps ~ N(0, I)

  [Can run in any order, or in parallel]
  loss_0  = ||layer_0(x + inject_0(1.00*y + 0.00*eps))  - y||^2   // pure noise
  loss_1  = ||layer_1(x + inject_1(0.97*y + 0.03*eps))  - y||^2
  ...
  loss_15 = ||layer_15(x + inject_15(0.52*y + 0.48*eps)) - y||^2  // half signal
  ...
  loss_31 = ||layer_31(x + inject_31(0.00*y + 1.00*eps)) - y||^2  // clean target

  Each loss_l has its own local backward. No sequential dependency.
```

**Note on the noise schedule direction**: In the listing above, layer 0 receives the NOISIEST input and layer 31 receives the CLEANEST. This mirrors the inference direction: at inference, we start with noise and progressively denoise. During training, layer 0 must learn the hardest denoising task (reconstruct from pure noise) and layer 31 the easiest (pass through clean input).

Wait -- there is an inconsistency to resolve. Let us be precise:

```
sigma_l = 1.0 - l / (NLAYERS - 1)

l=0:  sigma=1.0, alpha=0.0  → y_noisy = 0*y + 1*eps = pure noise
l=31: sigma=0.0, alpha=1.0  → y_noisy = 1*y + 0*eps = clean target
```

At **inference**:
```
z_0 ~ N(0, I)     // start with noise
z_1 = z_0 + eta * (layer_0(x, z_0) - z_0)   // layer 0 denoises the most
...
z_32 = z_31 + eta * (layer_31(x, z_31) - z_31)  // layer 31 does final cleanup
```

This is correct: layer 0 was trained on the noisiest input (sigma=1) and learned to make the biggest denoising step. Layer 31 was trained on the cleanest input (sigma=0) and makes a tiny refinement. The inference chain mirrors the training noise schedule.
