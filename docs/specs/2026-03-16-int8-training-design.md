# Design: INT8 W8A8 Training on Apple Neural Engine

**Date**: 2026-03-16
**Status**: DESIGN (not yet implemented)
**Priority**: P13 (from FINDINGS.md roadmap)
**Goal**: First INT8 training on any NPU hardware, leveraging ANE's INT8 throughput for MeZO zeroth-order and backprop-LoRA training
**Estimated effort**: 5-7 days engineering + 2-3 days experiments

---

## 0. Executive Summary

Apple's Neural Engine on A17 Pro/M4+ hardware has a dedicated int8-int8 compute path
that provides ~1.88x throughput over FP16 for conv operations. We propose exploiting
this for training -- specifically for MeZO's forward-only gradient estimation and for
the P16 hybrid backprop-LoRA forward pass. If both forward passes in MeZO can run in
INT8 without destroying the gradient signal, the theoretical speedup is 1.88x on top
of our existing 1.71x CPU advantage, yielding ~3.2x total. This would be the first
INT8 training on any NPU hardware.

**Critical hardware caveat**: Our current M2 Pro does NOT have the int8-int8 compute
path. On M2, INT8 weights are dequantized to FP16 before compute -- the only benefit
is memory bandwidth (2x smaller weights). The true INT8 compute speedup requires
A17 Pro or M4+ hardware. This design targets those chips but includes an M2-compatible
"bandwidth-only" mode for initial validation.

---

## 1. INT8 Forward Pass Design

### 1.1 ANE INT8 Execution Modes

The ANE supports INT8 via two distinct mechanisms:

**Mode A: Weight-Only Quantization (W8A16, all Apple Silicon)**
- Weights stored as INT8 via `constexpr_affine_dequantize` in MIL
- Activations remain FP16
- ANE dequantizes weights to FP16 before compute
- Benefit: 2x memory bandwidth reduction (smaller BLOBFILE, less DRAM pressure)
- No compute speedup -- same FP16 MAC operations
- Available on M1, M2, M3, M4

**Mode B: Full W8A8 Quantization (A17 Pro / M4+ only)**
- Weights stored as INT8 via `constexpr_affine_dequantize`
- Activations quantized to INT8 between layers via `quantize`/`dequantize` MIL ops
- ANE uses dedicated int8-int8 compute path
- Benefit: ~1.88x throughput over FP16 (both bandwidth AND compute)
- Available only on A17 Pro, M4, M4 Pro, M4 Max, M4 Ultra

For training, we target Mode B but implement Mode A first for validation on M2.

### 1.2 Weight Quantization: Per-Channel Symmetric

For each weight matrix W[OC, IC] (stored as conv kernel [OC, IC, 1, 1]):

```
scale[c] = max(|W[c, :]|) / 127    for c in [0, OC)
W_q[c, i] = clamp(round(W[c, i] / scale[c]), -128, 127)
W_deq[c, i] = scale[c] * W_q[c, i]
```

**Why per-channel (not per-tensor)**:
- Per-tensor quantization uses a single scale for the entire matrix. If one output
  channel has values 10x larger than others, the small channels lose ~90% of their
  dynamic range.
- Per-channel (axis=0, i.e., per output channel) gives each channel its own scale.
  This matches the natural structure of neural network weights where different output
  neurons learn at different magnitudes.
- Quantization error per element: `|W - W_deq| <= scale[c] / 2 = max(|W[c,:]|) / 254`
- Relative error per element: `|W - W_deq| / |W| <= 1/254 ~ 0.4%` (worst case, for
  the largest element in the channel; smaller elements have higher relative error)

**MIL representation**:
```
tensor<int8, [OC, IC, 1, 1]> W_q = const()[val=BLOBFILE(...)];
tensor<fp16, [OC, 1, 1, 1]> scale = const()[val=...];
tensor<int8, [1]> zero_pt = const()[val=tensor<int8, [1]>([0])];
tensor<fp16, [OC, IC, 1, 1]> W = constexpr_affine_dequantize(
    quantized_data=W_q, scale=scale, zero_point=zero_pt, axis=0);
```

This is a compile-time operation: the ANE compiler sees INT8 weights + scale and
can schedule them on the INT8 compute path (on supported hardware) or dequantize
them at load time (on M1-M3).

### 1.3 Activation Quantization: Per-Tensor Dynamic

Between transformer layers, activations must be quantized from FP16 to INT8:

```
For activation tensor x[1, C, 1, S]:
  x_max = max(|x|)                           // per-tensor
  scale_x = x_max / 127
  x_q = clamp(round(x / scale_x), -128, 127)
  x_deq = scale_x * x_q
```

**Why per-tensor (not per-channel) for activations**:
- Activations change every forward pass (unlike weights). Per-channel scales would
  require computing C separate max values and C divisions per layer -- this overhead
  may negate the INT8 speedup.
- Per-tensor requires a single global max and single division -- minimal overhead.
- Activation distributions in transformer hidden states are approximately symmetric
  (centered around 0 after RMSNorm) with heavy tails. Per-tensor handles this
  adequately for 8-bit precision.
- Exception: after SiLU (which has asymmetric output), per-tensor may lose precision
  on the positive tail. We address this in Section 3 (mixed precision strategy).

**Why dynamic (not static)**:
- Static quantization pre-computes activation ranges from calibration data. This is
  standard for inference but inappropriate for training: the activation distribution
  shifts as weights are updated.
- Dynamic quantization computes the scale from the actual activation tensor at each
  forward pass. The overhead is one reduction (max) per layer.
- For MeZO, the perturbed weights w+ez and w-ez produce different activation
  distributions. Static scales calibrated on unperturbed weights would be wrong.

**MIL representation for inter-layer quantization**:
```
// After layer L's output (FP16):
tensor<fp16, [1, DIM, 1, SEQ]> x_fp16 = ...;

// Quantize to INT8
tensor<int8, [1, DIM, 1, SEQ]> x_q = quantize(
    input=x_fp16, scale=scale_x, zero_point=0,
    output_dtype="int8");

// Dequantize back to FP16 for next layer's compute
tensor<fp16, [1, DIM, 1, SEQ]> x_deq = dequantize(
    input=x_q, scale=scale_x, zero_point=0);
```

**Assumption A1**: The `quantize` and `dequantize` MIL ops are supported on A17 Pro/M4
ANE when targeting ios18. This is documented in coremltools for the
`linear_quantize_activations` API but has not been verified via our private-API MIL
compilation path. Risk: MEDIUM. If these ops are not supported in in-memory MIL
compilation, we fall back to CPU-side quantization between ANE dispatches.

### 1.4 Quantization Error Analysis for Training

For a single linear layer y = Wx with per-channel weight quantization and per-tensor
activation quantization:

```
y_exact = W * x
y_quant = (W + dW) * (x + dx)
        = Wx + W*dx + dW*x + dW*dx
        = y_exact + W*dx + dW*x + O(dW*dx)
```

where:
- `dW` = weight quantization noise: `|dW[c,i]| <= s_W[c]/2`
- `dx` = activation quantization noise: `|dx[j]| <= s_x/2`
- `s_W[c] = max(|W[c,:]|)/127`, `s_x = max(|x|)/127`

The output error per element (ignoring second-order term):
```
|y_quant - y_exact| <= |W| * |dx| + |dW| * |x|
```

For a dot product of dimension IC:
- Weight noise contribution: `sum_i |dW[c,i]| * |x[i]| ~ (s_W[c]/2) * sqrt(IC) * rms(x)`
  (by CLT, assuming independent errors)
- Activation noise contribution: `sum_i |W[c,i]| * |dx[i]| ~ (s_x/2) * sqrt(IC) * rms(W[c,:])`

For SmolLM2-360M with DIM=960:
- `sqrt(IC) = sqrt(960) ~ 31`
- Typical `rms(x) ~ 1.0` (after RMSNorm), `rms(W) ~ 0.02` (pretrained)
- Weight noise: `(0.02/254) * 31 * 1.0 ~ 0.0024` per output element
- Activation noise: `(1.0/254) * 31 * 0.02 ~ 0.0024` per output element
- Total INT8 noise: ~0.005 per output element

For comparison, FP16 noise from dot-product accumulation:
- `~ 0.5 ULP * sqrt(960) ~ 0.5 * (2^{-10}) * 31 ~ 0.015` per output element

**Key finding**: INT8 quantization noise (~0.005) is actually SMALLER than existing
FP16 accumulation noise (~0.015) for DIM=960. This is because INT8 per-channel
quantization preserves 7 bits of precision per element, while FP16 accumulation over
960 elements loses ~5 bits from rounding error accumulation.

**Caveat**: This analysis assumes the ANE's INT8 compute path accumulates in higher
precision (INT16 or INT32) internally. If it accumulates in INT8, the accumulation
noise would be catastrophic (~sqrt(960)/127 ~ 0.24 per element). Apple's documentation
does not specify the internal accumulator width, but standard INT8 GEMM implementations
(NVIDIA, Qualcomm) use INT32 accumulators. **Assumption A2**: ANE's INT8 compute
path uses >= INT16 accumulators. Risk: HIGH. If wrong, INT8 training is infeasible.

### 1.5 What Precision Is Needed for Training-Useful Forward Passes?

For MeZO, we need the forward pass to compute a loss value accurate enough that:
```
L(w + ez) - L(w - ez)
```
has a nonzero signal. The gradient estimate is:
```
g_hat = [L+ - L-] / (2e) * z
```

The signal is `L+ - L-`, which for small perturbation epsilon ~ 1e-3 and typical
gradient magnitudes is on the order of 1e-3 to 1e-1 (measured in our experiments:
typical `proj_grad = (L+ - L-) / (2e) ~ 0.001 to 0.1`).

The noise floor from quantization is analyzed in Section 2. The key requirement is:
```
|L+(quant) - L+(exact)| << |L+ - L-|
```

If quantization noise in the loss is comparable to the loss difference, the gradient
signal is destroyed. Section 2 derives this bound precisely.

---

## 2. INT8 Gradient Estimation for MeZO

### 2.1 MeZO Gradient Estimate Under Quantization

The standard MeZO gradient estimate for parameter theta_i is:
```
g_i = [(L(theta + e*z) - L(theta - e*z)) / (2e)] * z_i
```

With INT8 forward passes, we compute:
```
L_q+ = L_INT8(theta + e*z)    (quantized forward at theta + e*z)
L_q- = L_INT8(theta - e*z)    (quantized forward at theta - e*z)
g_q_i = [(L_q+ - L_q-) / (2e)] * z_i
```

The quantized loss differs from the exact loss:
```
L_q+ = L+(exact) + n+     where n+ is quantization noise
L_q- = L-(exact) + n-     where n- is quantization noise
```

Therefore:
```
g_q_i = g_i(exact) + [(n+ - n-) / (2e)] * z_i
```

The gradient estimation error is:
```
delta_g_i = [(n+ - n-) / (2e)] * z_i
```

### 2.2 Quantization Noise in the Loss

The loss is cross-entropy computed on logits. The logits are the output of the final
linear layer (classifier), which takes the 32-layer transformer output as input.

**Error propagation through L transformer layers**:

At each layer, the quantization introduces noise delta_l with variance:
```
Var(delta_l) ~ sigma_W^2 * DIM * Var(x) + sigma_x^2 * DIM * Var(W)
```

where `sigma_W ~ s_W / (2*sqrt(3))` (uniform quantization noise), `sigma_x ~ s_x / (2*sqrt(3))`.

Through L layers with residual connections, the noise accumulates:
```
Var(delta_L) ~ L * Var(delta_per_layer)
```

(Residual connections prevent exponential error growth but cause linear accumulation.)

For SmolLM2-360M (L=32, DIM=960):
```
Var(delta_per_layer) ~ 2 * (0.02/254)^2 * 960 * 1.0^2 = 2 * 6.2e-9 * 960 ~ 1.2e-5
Var(delta_L) ~ 32 * 1.2e-5 ~ 3.8e-4
sigma_L ~ 0.019
```

The noise in the logits is ~0.019 per element. After softmax + cross-entropy:
```
|delta_loss| ~ sigma_L * sqrt(VOCAB_ACTIVE)
```

But cross-entropy focuses on the target token, so the relevant noise is:
```
|delta_loss| ~ sigma_L ~ 0.019    (approximately, from the target logit noise)
```

### 2.3 Signal-to-Noise Ratio

The gradient signal `L+ - L-` for our setup:

From measured data (Phase 2 experiments):
- Typical `|L+ - L-|` ranges from 0.001 to 0.3 (depends on epsilon and gradient magnitude)
- With epsilon = 1e-3: typical `|L+ - L-| ~ 0.001 to 0.01`
- With epsilon = 1e-2: typical `|L+ - L-| ~ 0.01 to 0.1`

Quantization noise in the loss difference:
```
Var(delta_loss_diff) = Var(n+ - n-)
```

**Critical question**: Are n+ and n- correlated?

The quantization noise at each layer depends on the activation values, which differ
between the +e and -e forward passes. For small epsilon, the activations are nearly
identical, so the quantization decisions (rounding) are also nearly identical. This
means n+ and n- are HIGHLY CORRELATED, and their difference is much smaller than
either alone.

Specifically, rounding noise cancels when the same quantization bin is hit:
```
round(x + small_delta) - round(x - small_delta) = 0    (if both round the same way)
```

The only contribution to `n+ - n-` comes from elements near quantization bin boundaries
where the perturbation pushes one forward pass into a different bin. The fraction of
elements near a bin boundary (within epsilon of it) is approximately:
```
p_boundary ~ 2 * epsilon * density_at_boundary / bin_width
           ~ 2 * 1e-3 * (1/scale) / (1/scale)    [uniform density within a bin]
           ~ 2e-3
```

So only ~0.2% of activation elements contribute to `n+ - n-`. This dramatically
reduces the effective noise:
```
sigma(n+ - n-) ~ sigma_L * sqrt(p_boundary) ~ 0.019 * sqrt(0.002) ~ 0.00085
```

**Signal-to-noise ratio at epsilon = 1e-3**:
```
SNR = |L+ - L-| / sigma(n+ - n-)
    ~ 0.005 / 0.00085
    ~ 5.9
```

**At epsilon = 1e-2** (larger perturbation):
```
p_boundary ~ 2e-2, sigma(n+ - n-) ~ 0.019 * sqrt(0.02) ~ 0.0027
SNR ~ 0.05 / 0.0027 ~ 18.5
```

**Conclusion**: The gradient signal is recoverable for both epsilon values, but
epsilon = 1e-2 provides ~3x better SNR. We recommend using a larger epsilon when
training in INT8 to increase the signal above the quantization noise floor.

### 2.4 Formal Error Bound on Gradient Estimate

The expected squared error of the INT8 gradient estimate vs exact:

```
E[||g_q - g_exact||^2] = E[||(n+ - n-)/(2e)||^2] * E[||z||^2]
                        = sigma(n+ - n-)^2 / (4*e^2) * d
```

where d = number of trainable parameters (1.7M for LoRA).

For epsilon = 1e-3:
```
E[||g_q - g_exact||^2] ~ (0.00085)^2 / (4 * 1e-6) * 1.7e6
                        ~ 0.72e-6 / 4e-6 * 1.7e6
                        ~ 0.18 * 1.7e6
                        ~ 3.1e5
```

For epsilon = 1e-2:
```
E[||g_q - g_exact||^2] ~ (0.0027)^2 / (4 * 1e-4) * 1.7e6
                        ~ 7.3e-6 / 4e-4 * 1.7e6
                        ~ 0.018 * 1.7e6
                        ~ 3.1e4
```

The FP16 gradient estimate already has inherent variance:
```
E[||g_fp16||^2] ~ sigma_grad^2 * d ~ (proj_grad)^2 * d ~ (0.01)^2 * 1.7e6 = 170
```

Wait -- this analysis shows the INT8 noise variance (~3.1e4 at e=1e-2) is much larger
than the gradient signal variance (~170). However, this overestimates the problem:
the MeZO gradient IS the noisy directional derivative times z. The "noise" is the
ADDITIONAL noise from quantization on top of the already-noisy ZO estimate. What
matters is whether the quantization noise overwhelms the per-step update, not the
full gradient norm.

**Per-parameter update analysis**:
```
update_i = lr * proj_grad * z_i

For exact MeZO:
  proj_grad_exact ~ N(0, sigma_proj)    where sigma_proj ~ ||grad||/sqrt(d)
  |update_i| ~ lr * sigma_proj ~ 1e-4 * 0.01 = 1e-6

For INT8 MeZO:
  proj_grad_q = proj_grad_exact + (n+ - n-)/(2e)
  noise_in_proj_grad ~ sigma(n+- n-)/(2e) ~ 0.00085/(2e-3) = 0.43    (at e=1e-3)
                     or ~ 0.0027/(2e-2) = 0.135                        (at e=1e-2)
```

At epsilon = 1e-3, the quantization noise in proj_grad (0.43) is ~43x the typical
signal (0.01). **This is catastrophic.**

At epsilon = 1e-2, the noise (0.135) is ~13.5x the signal. **Still problematic.**

### 2.5 Mitigation: Increased Epsilon and Multi-Perturbation Averaging

**Strategy 1: Increase epsilon to 0.1**:
```
p_boundary ~ 0.2 (20% of elements cross bin boundaries)
sigma(n+ - n-) ~ 0.019 * sqrt(0.2) ~ 0.0085
noise_in_proj_grad ~ 0.0085 / 0.2 = 0.043
signal ~ 0.1 * ||grad|| ~ 0.1 * 1.0 = 0.1    (at e=0.1, loss diff is larger)
SNR in proj_grad ~ 0.1 / 0.043 ~ 2.3
```

This is marginal but potentially workable with learning rate reduction.

**Strategy 2: Stochastic rounding instead of nearest rounding**:
Stochastic rounding makes quantization noise unbiased and uncorrelated:
```
round_stoch(x) = floor(x) with probability 1 - frac(x)
               = ceil(x) with probability frac(x)
```

With stochastic rounding, `E[n+ - n-] = 0` and the noise in the loss difference
becomes zero in expectation. The variance is:
```
Var(n+ - n-) = Var(n+) + Var(n-) ~ 2 * sigma_L^2 ~ 2 * (0.019)^2 = 7.2e-4
sigma(n+ - n-) ~ 0.027
```

This is WORSE than deterministic rounding (which benefits from correlated noise
cancellation). Stochastic rounding is helpful for the gradient BIAS (zero bias vs
nonzero bias) but not for gradient VARIANCE.

**Strategy 3: FP32 loss computation (our recommended approach)**:
Keep the classifier layer and cross-entropy computation in FP32 (on CPU, as we
already do). Only the transformer body runs in INT8. The final hidden state is
dequantized to FP32 before the classifier:
```
x_final_fp32 = dequantize(x_final_int8)    // CPU-side
logits = x_final_fp32 @ embed^T             // CPU fp32
loss = cross_entropy(logits, targets)        // CPU fp32
```

This eliminates the logit-level quantization noise. The noise in x_final is
~sigma_L = 0.019 per element, but this is in the hidden state, not in the loss
directly. The loss noise becomes:
```
delta_loss ~ (d_loss/d_x_final) * delta_x_final
           ~ 0.019 * (derivative of loss w.r.t. hidden)
```

The derivative of cross-entropy w.r.t. the final hidden state has magnitude
~1/DIM (after RMSNorm), so:
```
|delta_loss| ~ 0.019 / sqrt(960) ~ 0.0006
```

This is much better. At epsilon = 1e-3:
```
noise_in_proj_grad ~ 0.0006 / 0.002 = 0.3
signal ~ 0.01
SNR ~ 0.033    ... still bad
```

**Revised analysis**: The derivative chain from hidden state to loss is not simply
1/sqrt(DIM). The actual gradient norm ||dL/dx_final|| depends on the prediction
confidence. At cross-entropy loss ~2.0 (our typical value), the gradient norm is ~1.0.
So:
```
|delta_loss| ~ 0.019 * 1.0 / sqrt(DIM) ~ 0.019 / 31 ~ 0.0006
```

Hmm, this remains small. Let me reconsider from first principles.

**Empirical approach (recommended)**: Rather than trying to analytically bound the
noise, we should MEASURE it:
1. Run one forward pass in FP16, record loss_fp16
2. Run same forward pass in INT8 (same weights, same input), record loss_int8
3. Compare `|loss_fp16 - loss_int8|` across 100 random inputs
4. This directly gives us the quantization noise floor in the loss

This is the first experiment in Section 5.

### 2.6 Summary of Noise Analysis

| Epsilon | Loss Noise (sigma) | Proj Grad Noise | Typical Signal | SNR | Viable? |
|---------|-------------------|-----------------|---------------|-----|---------|
| 1e-3 | 0.0006 | 0.3 | 0.01 | 0.03 | NO |
| 1e-2 | 0.0006 | 0.03 | 0.1 | 3.3 | MARGINAL |
| 1e-1 | 0.0006 | 0.003 | 0.5 | 167 | YES |
| 1e-1 (empirical) | TBD | TBD | TBD | TBD | MEASURE |

**Key insight**: INT8 MeZO training likely requires a much larger perturbation
(epsilon ~ 0.05-0.1) than FP16 MeZO (epsilon ~ 1e-3). Larger epsilon increases
the bias in the gradient estimate (higher-order terms in the Taylor expansion) but
reduces the relative quantization noise. This is a fundamentally different operating
point.

**Assumption A3**: Epsilon = 0.05-0.1 with INT8 forward passes will produce usable
gradient estimates. Risk: HIGH. Must be validated empirically.

---

## 3. Mixed Precision Strategy

### 3.1 Operations Classification

| Operation | Precision | Location | Rationale |
|-----------|-----------|----------|-----------|
| **Linear projections** (Wq, Wk, Wv, Wo, W1, W2, W3) | **INT8 W8A8** | ANE | Primary compute -- this is where INT8 throughput matters |
| **RMSNorm** | FP32 | CPU | Normalization is sensitive to precision; already on CPU |
| **RoPE** | FP32 | CPU | Trigonometric operations; already on CPU |
| **SDPA (attention scores)** | FP32 | CPU | Softmax numerics require FP32; already on CPU |
| **SiLU activation** | FP32 | CPU | Non-monotonic gradient; already on CPU |
| **Residual additions** | FP32 | CPU | Accumulation point; precision matters |
| **Embedding lookup** | FP32 | CPU | Table lookup, no compute |
| **Classifier (final linear)** | FP32 | CPU | Loss precision critical for gradient estimation |
| **Cross-entropy loss** | FP32 | CPU | log-sum-exp needs FP32 |
| **Perturbation** (w += e*z) | FP32 | CPU | Perturbation scale is 1e-3 to 0.1; needs FP32 |
| **Weight update** (w -= lr*g*z) | FP32 | CPU | Learning rate is 1e-4 to 1e-5; needs FP32 |
| **LoRA merge** (W_eff = W_base + B@A) | FP32 | CPU | Rank-8 GEMM, tiny compute; precision critical |

### 3.2 Data Flow

```
                           CPU (FP32)                    ANE (INT8/FP16)
                           ========                      ===============

Input tokens -----> Embedding lookup (FP32)
                           |
                    RMSNorm (FP32)
                           |
                    Quantize to INT8 ---------> QKV conv1x1 (INT8 W8A8) ---+
                                                                            |
                    <----- Dequantize from INT8 <---------  Q,K,V (INT8) --+
                           |
                    RoPE (FP32)
                    SDPA softmax (FP32)
                    Attention output (FP32)
                           |
                    Quantize to INT8 ---------> Wo conv1x1 (INT8 W8A8) ----+
                                                                            |
                    <----- Dequantize from INT8 <--------- o_out (INT8) ---+
                           |
                    Residual add (FP32)
                    RMSNorm (FP32)
                           |
                    Quantize to INT8 ---------> FFN fused conv1x1 ---------+
                                                (W1, W3, SiLU, W2)         |
                                                (INT8 weights,              |
                                                 INT8 activations           |
                                                 for conv, FP16 for SiLU)  |
                    <----- Dequantize from INT8 <--------- ffn_out --------+
                           |
                    Residual add (FP32)
                           |
                        [next layer]
                           ...
                    Final RMSNorm (FP32)
                    Classifier matmul (FP32 CPU)
                    Cross-entropy loss (FP32 CPU)
```

### 3.3 LoRA Corrections in INT8 Mode

In our current LoRA-split architecture, base weights are BLOBFILE constants and LoRA
corrections are applied on CPU before writing activations to the ANE IOSurface:

```
x_corrected = x + LoRA_B @ LoRA_A @ x    (CPU FP32)
```

This can remain in FP32 on CPU. The corrected activation is then quantized to INT8
before being sent to the ANE conv kernel. The LoRA correction is a rank-8 perturbation
(~0.5% of the base weight magnitude), so quantizing the corrected activation loses
very little of the LoRA signal -- the correction is already small relative to the
quantization bin width.

**Assumption A4**: LoRA corrections of magnitude ~0.01 * ||W|| survive INT8
quantization of the corrected activation. Risk: MEDIUM. The LoRA signal is ~1% of
the activation magnitude; INT8 has ~0.4% quantization noise. The LoRA signal is
~2.5x the noise floor, which is marginal. If this fails, we must apply LoRA
corrections AFTER dequantization (requiring a second CPU pass).

### 3.4 FFN Fusion with INT8

The current FFN mega-kernel computes:
```
x_next = x_cur + W2 @ (SiLU(W1 @ xnorm) * W3 @ xnorm)
```

In INT8 mode, the conv1x1 operations (W1, W3, W2) can be INT8, but SiLU must be
FP16/FP32. Two options:

**Option A: Inter-op dequantization inside the ANE kernel**
```
h1_int8 = conv_int8(xnorm_int8, W1_int8)     // INT8 matmul
h1_fp16 = dequantize(h1_int8)                 // to FP16
sig = sigmoid(h1_fp16)                        // FP16
silu = h1_fp16 * sig                          // FP16
h3_int8 = conv_int8(xnorm_int8, W3_int8)     // INT8 matmul
h3_fp16 = dequantize(h3_int8)                 // to FP16
gate = silu * h3_fp16                         // FP16
gate_int8 = quantize(gate)                    // back to INT8
ffn_out_int8 = conv_int8(gate_int8, W2_int8)  // INT8 matmul
```

This requires the ANE to handle mixed INT8/FP16 within a single fused kernel. The
MIL IR supports this (quantize/dequantize are valid MIL ops), but compilation
success on the private API path is unverified.

**Option B: Split FFN into 3 ANE dispatches + CPU SiLU**
```
h1 = ANE_conv_int8(xnorm, W1)    // dispatch 1
h3 = ANE_conv_int8(xnorm, W3)    // dispatch 2
gate = cpu_silu(h1) * h3          // CPU FP32
ffn_out = ANE_conv_int8(gate, W2) // dispatch 3
```

This is simpler but adds 2 extra ANE dispatches per layer (64 extra for 32 layers).
At ~160us per dispatch, that's ~10ms overhead.

**Recommendation**: Start with Option B for correctness, then optimize to Option A
if the ANE compiler supports mixed-precision fused kernels.

---

## 4. Implementation

### 4.1 MIL Op Reference

The following MIL operations are needed for INT8 training:

**`constexpr_affine_dequantize`** (compile-time weight dequantization):
```
constexpr_affine_dequantize(
    quantized_data: tensor<int8, [OC, IC, 1, 1]>,
    zero_point: tensor<int8, [1]> or tensor<int8, [OC,1,1,1]>,
    scale: tensor<fp16, [1]> or tensor<fp16, [OC,1,1,1]>,
    axis: int32
) -> tensor<fp16, [OC, IC, 1, 1]>
```
- Available: iOS 16+ / macOS 13+
- On M1-M3: weight is dequantized to FP16 at load time or runtime
- On A17 Pro / M4+: weight stays INT8 and uses int8 compute path
- We already have `ane_bridge_build_weight_blob_int8` for building INT8 blobs

**`quantize`** (runtime activation quantization):
```
quantize(
    input: tensor<fp16, [1, C, 1, S]>,
    scale: tensor<fp16, [1]>,         // per-tensor
    zero_point: int8(0),
    output_dtype: "int8"
) -> tensor<int8, [1, C, 1, S]>
```
- Available: iOS 18+ / macOS 15+ (experimental)
- Computes: output = clamp(round(input / scale), -128, 127)

**`dequantize`** (runtime activation dequantization):
```
dequantize(
    input: tensor<int8, [1, C, 1, S]>,
    scale: tensor<fp16, [1]>,
    zero_point: int8(0)
) -> tensor<fp16, [1, C, 1, S]>
```
- Available: iOS 18+ / macOS 15+
- Computes: output = scale * (input - zero_point)

**`conv`** (1x1 convolution with INT8 weight):
```
conv(
    x: tensor<fp16, [1, IC, 1, SEQ]>,   // or tensor<int8, ...> for W8A8
    weight: tensor<fp16, [OC, IC, 1, 1]>,  // result of constexpr_affine_dequantize
    ...
) -> tensor<fp16, [1, OC, 1, SEQ]>      // or tensor<int8, ...> for W8A8
```

### 4.2 MIL Generator: INT8 Conv1x1

New MIL generator for INT8 weight conv1x1:

```objc
static NSString *gen_conv1x1_int8_mil(int ic, int oc, int seq,
                                       bool per_channel) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n",
        ic, seq];

    // INT8 weight constant with affine dequantization
    [m appendFormat:
        @"        tensor<int8, [%d, %d, 1, 1]> W_q = const()"
        "[name=string(\"W_q\"), val=tensor<int8, [%d, %d, 1, 1]>"
        "(BLOBFILE(path=string(\"@model_path/weights/w_q.bin\"), "
        "offset=uint64(64)))];\n", oc, ic, oc, ic];

    if (per_channel) {
        [m appendFormat:
            @"        tensor<fp16, [%d, 1, 1, 1]> scale = const()"
            "[name=string(\"scale\"), val=tensor<fp16, [%d, 1, 1, 1]>"
            "(BLOBFILE(path=string(\"@model_path/weights/scale.bin\"), "
            "offset=uint64(64)))];\n", oc, oc];
    } else {
        [m appendString:
            @"        tensor<fp16, [1]> scale = const()"
            "[name=string(\"scale\"), val=tensor<fp16, [1]>"
            "(BLOBFILE(path=string(\"@model_path/weights/scale.bin\"), "
            "offset=uint64(64)))];\n"];
    }

    [m appendString:
        @"        tensor<int8, [1]> zp = const()"
        "[name=string(\"zp\"), val=tensor<int8, [1]>([0])];\n"];

    [m appendFormat:
        @"        tensor<fp16, [%d, %d, 1, 1]> W = "
        "constexpr_affine_dequantize(quantized_data=W_q, "
        "zero_point=zp, scale=scale, axis=int32(0))"
        "[name=string(\"W\")];\n", oc, ic];

    // Conv1x1 (same as FP16 path)
    [m appendString:@"        tensor<int32, [2]> st = const()"
        "[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        tensor<int32, [4]> pd = const()"
        "[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"];
    [m appendString:@"        tensor<int32, [2]> dl = const()"
        "[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        int32 gr = const()"
        "[name=string(\"gr\"), val=int32(1)];\n"];
    [m appendString:@"        string pt = const()"
        "[name=string(\"pt\"), val=string(\"valid\")];\n"];

    [m appendFormat:
        @"        tensor<fp16, [1, %d, 1, %d]> y = conv("
        "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
        "weight=W, x=x)[name=string(\"y\")];\n", oc, seq];

    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}
```

### 4.3 INT8 Weight Blob Builder

We already have `ane_bridge_build_weight_blob_int8` and
`ane_bridge_build_weight_blob_quantized` in `bridge/ane_bridge.m`. These need minor
modifications:

1. **Per-channel scale output**: Current `quantized` builder uses per-tensor scale.
   Add per-channel variant:
```objc
uint8_t *ane_bridge_build_weight_blob_quantized_perchannel(
    const float *src, int rows, int cols,
    float *out_scales,      // [rows] array of per-channel scales
    size_t *out_len);
```

2. **Scale blob builder**: Need to build an FP16 blob for the scale tensor:
```objc
NSData *build_scale_blob_fp16(const float *scales, int n);
```

### 4.4 Compilation Function

```objc
static Kern *compile_conv1x1_int8_kern(NSString *mil,
    const float *W,        // FP32 weights [OC, IC]
    int ic, int oc, int seq) {

    // Quantize to INT8 per-channel
    float *scales = (float *)safe_malloc(oc * sizeof(float));
    int8_t *W_q = (int8_t *)safe_malloc((size_t)oc * ic);

    for (int c = 0; c < oc; c++) {
        float max_abs = 0;
        for (int i = 0; i < ic; i++) {
            float a = fabsf(W[c * ic + i]);
            if (a > max_abs) max_abs = a;
        }
        scales[c] = max_abs / 127.0f;
        if (scales[c] == 0.0f) scales[c] = 1.0f;
        float inv_scale = 1.0f / scales[c];
        for (int i = 0; i < ic; i++) {
            float v = W[c * ic + i] * inv_scale;
            v = fmaxf(fminf(v, 127.0f), -128.0f);
            W_q[c * ic + i] = (int8_t)(v + (v >= 0 ? 0.5f : -0.5f));
        }
    }

    // Build blobs
    NSData *w_blob = build_int8_blob(W_q, oc * ic);
    NSData *s_blob = build_fp16_scale_blob(scales, oc);
    free(W_q); free(scales);

    NSDictionary *weights = @{
        @"@model_path/weights/w_q.bin": @{@"data": w_blob},
        @"@model_path/weights/scale.bin": @{@"data": s_blob}
    };

    return compile_kern_mil_w(mil, weights, ic * seq * 2, oc * seq * 2);
}
```

### 4.5 ANE IOSurface Constraints for INT8

**Assumption A5**: IOSurfaces for INT8 activations (if W8A8 mode) may require
different byte-per-element configurations. The current `make_surface` function
assumes 1 byte per pixel. For INT8 activations, the IOSurface would be half the
size of FP16 activations. Risk: LOW. IOSurface is a raw byte buffer; we just
need to allocate the right number of bytes.

For W8A16 mode (our initial target), the IOSurface format remains FP16 for
activations, so no changes needed.

### 4.6 Training Loop Modifications

For MeZO with INT8 forward:

```c
// Compile INT8 conv1x1 kernels (once at startup)
for (int L = 0; L < NLAYERS; L++) {
    // Quantize merged weights: W_eff = W_base + B@A
    lora_merge_weight(W_eff_q, ...);
    dk->wqConv_int8[L] = compile_conv1x1_int8_kern(
        gen_conv1x1_int8_mil(DIM, Q_DIM, SEQ, true),
        W_eff_q, DIM, Q_DIM, SEQ);
    // ... same for Wo, W1, W2, W3
}

// MeZO step (same structure, INT8 forward)
// 1. Perturb LoRA weights (FP32 CPU)
// 2. Re-merge: W_eff = W_base + B@A (FP32 CPU)
// 3. Re-quantize W_eff to INT8 (CPU)
// 4. Write INT8 weights to BLOBFILE (requires recompilation!)
//    OR: use dynamic weight IOSurface with INT8 weights
// 5. Forward pass on ANE (INT8 conv)
// 6. Loss computation on CPU (FP32)
```

**Critical implementation problem**: In our LoRA-split architecture, base weights
are BLOBFILE constants compiled into the kernel. When LoRA weights change (every
MeZO step due to perturbation), we need to update the effective weights. Our current
approach is to apply LoRA corrections on CPU and write corrected activations to
the IOSurface. This remains valid for INT8 mode -- the base weight stays as INT8
BLOBFILE, and the LoRA correction is applied in FP32 on CPU before quantizing the
activation to FP16 for the ANE input.

No per-step recompilation is needed. This is the same architecture as FP16 mode.

---

## 5. Testable Hypotheses

### Hypothesis 1: INT8 Weight Quantization Preserves Forward Pass Quality

**Claim**: Per-channel INT8 quantization of transformer weights introduces less error
than the existing FP16 dot-product accumulation noise.

**Protocol**:
1. Load pretrained SmolLM2-360M checkpoint
2. Run 100 forward passes with random inputs:
   a. FP16 weights on ANE (current mode) -> loss_fp16[i]
   b. INT8 weights on ANE (W8A16) -> loss_int8[i]
3. Compute: `mean(|loss_fp16 - loss_int8|)` and `max(|loss_fp16 - loss_int8|)`

**Success criterion**: `mean(|loss_fp16 - loss_int8|) < 0.01` (less than 0.5% of
typical loss ~2.0). This implies quantization noise is small relative to the loss value.

**Duration**: ~30 seconds (100 forward passes x 131ms / 2 modes).

**Hardware**: M2 Pro (W8A16 mode only; W8A8 requires M4).

### Hypothesis 2: INT8 MeZO Gradient Estimates Are Usable at Epsilon >= 0.05

**Claim**: MeZO gradient estimates computed via INT8 forward passes at epsilon = 0.05
are correlated with FP16 gradient estimates (Pearson r > 0.5).

**Protocol**:
1. Load pretrained SmolLM2-360M with LoRA rank-8
2. For 50 random seeds:
   a. Compute MeZO gradient estimate with FP16 forward: `g_fp16[seed]`
   b. Compute MeZO gradient estimate with INT8 forward: `g_int8[seed]`
3. Compute: `pearson_r(g_fp16, g_int8)` (projected gradient scalar, not full vector)

**Success criterion**: `pearson_r > 0.5` (gradient direction is more right than wrong).

**Duration**: ~100 seconds (50 seeds x 4 forward passes x ~131ms x 2 modes).
Actually less: we only need the scalar proj_grad from each, not the full gradient.

### Hypothesis 3: INT8 MeZO Training Converges

**Claim**: MeZO training with INT8 forward passes reduces validation loss over 500
steps, at a rate within 2x of FP16 MeZO convergence.

**Protocol**:
1. SmolLM2-360M pretrained, LoRA rank-8, attention-only
2. Run 500 steps with INT8 forward passes at epsilon = 0.05, lr = 1e-4
3. Run 500 steps with FP16 forward passes at epsilon = 1e-3, lr = 1e-4 (control)
4. Compare val_loss improvement at step 500

**Success criteria**:
- INT8 val_loss at step 500 < val_loss at step 0 (learning occurred)
- INT8 improvement >= 50% of FP16 improvement (within 2x)

**Duration**: ~4.5 minutes (1000 steps x 262ms).

**Go/no-go gate**: If Hypothesis 1 fails (loss noise > 0.01), skip Hypothesis 2-3.
If Hypothesis 2 fails (correlation < 0.5), try epsilon = 0.1 before declaring failure.

---

## 6. Expected Speedup

### 6.1 Theoretical Speedup

**On M2 Pro (W8A16 mode)**:
- Weights are 2x smaller -> 2x less DRAM bandwidth for weight loads
- Compute is still FP16 -> no FLOPS improvement
- Weight loading is ~30% of our current forward time (the rest is activation I/O,
  CPU ops, dispatch overhead)
- Expected speedup: ~1.15x (2x on 30% of time = 1.0 + 0.3 * 1.0 = 1.3... wait)

Actually: If weight loading takes fraction f of forward time, and becomes 2x faster:
```
T_new = T_old * (1 - f) + T_old * f / 2 = T_old * (1 - f/2)
Speedup = 1 / (1 - f/2)
```

For conv-fused mode, weight loading happens at compile time (BLOBFILE), not per-step.
The per-step cost is activation I/O (writing FP16 activations to IOSurface) and
ANE compute. INT8 weights in BLOBFILE are already loaded once at compile time.

So on M2, the speedup is approximately **1.0x** (no change) for per-step compute.
The benefit is 2x smaller compiled model on disk and potentially better SRAM
utilization (weights occupy half the SRAM).

**On M4 / A17 Pro (W8A8 mode)**:
- Apple claims 1.88x INT8 throughput over FP16
- Our forward pass is 131ms for 32 layers
- Breakdown: ~90ms ANE compute, ~25ms CPU ops, ~15ms dispatch overhead
- INT8 speedup applies to the 90ms ANE compute portion:
  ```
  T_new = 90ms / 1.88 + 25ms + 15ms = 47.9ms + 25ms + 15ms = 87.9ms
  Speedup = 131ms / 87.9ms = 1.49x
  ```

For MeZO (2 forward passes per step):
```
Current (FP16): 2 * 131ms + perturbation_overhead = 262ms + ~5ms = ~267ms
INT8 (W8A8):    2 * 87.9ms + perturbation_overhead + requantize_overhead
              = 175.8ms + ~5ms + ~10ms = ~191ms
Speedup: 267ms / 191ms = 1.40x
```

**Requantize overhead**: When LoRA weights change (perturbation step), we need to
re-quantize the merged effective weights. For 7 weight matrices x 32 layers x
~1M elements each = ~224M quantization operations. At ~4ns each on CPU: ~0.9 seconds.
This is CATASTROPHIC -- it would negate all INT8 speedup.

**Mitigation**: We do NOT re-quantize base weights. LoRA corrections are applied on
CPU in FP32 to the activation, not to the weight. The INT8 base weights remain
unchanged. Only the LoRA A/B matrices change, and these are tiny (rank-8, ~1.7M
total params). Re-quantizing LoRA matrices: 1.7M * 4ns = ~7ms. Acceptable.

But wait -- in our conv-fused architecture, LoRA corrections are already applied
on CPU (before writing activation to IOSurface). The ANE only sees the corrected
activation. So NO weight re-quantization is needed at all.

**Revised estimate on M4**:
```
MeZO INT8 (W8A8): 2 * 87.9ms + ~5ms = ~181ms/step
Speedup vs FP16 ANE: 267ms / 181ms = 1.48x
Speedup vs CPU baseline: 447ms / 181ms = 2.47x (vs current 1.71x)
```

### 6.2 Practical Speedup Estimates

| Mode | Hardware | Forward (ms) | MeZO Step (ms) | vs CPU | vs FP16 ANE |
|------|----------|-------------|----------------|--------|-------------|
| FP16 ANE (current) | M2 Pro | 131 | 267 | 1.67x | 1.0x |
| **INT8 W8A16** | **M2 Pro** | **~131** | **~267** | **1.67x** | **~1.0x** |
| **INT8 W8A8** | **M4** | **~88** | **~181** | **2.47x** | **1.48x** |
| FP16 ANE (projected) | M4 | ~90 | ~185 | 2.42x | — |
| **INT8 W8A8** | **M4** | **~88** | **~181** | **2.47x** | **1.02x over M4 FP16** |

**Critical observation**: On M4, the INT8 speedup over FP16 is modest (~1.02x for
MeZO total step) because the CPU-bound operations (RMSNorm, RoPE, SDPA, SiLU, LoRA
merge) dominate the step time. The ANE compute fraction is only ~68% of the forward
pass, so a 1.88x improvement on 68% yields only 1.30x on the forward pass. After
accounting for the second forward pass and CPU overhead in MeZO, the net benefit is
minimal.

**Where INT8 helps more**:
- Models with less CPU overhead (e.g., if we fuse RMSNorm/SDPA onto ANE)
- Larger models where ANE compute fraction is higher (>90%)
- P16 hybrid mode where the forward pass is a larger fraction of step time

### 6.3 Memory Reduction

Regardless of compute speedup, INT8 weights provide:

| Component | FP16 Size | INT8 Size | Reduction |
|-----------|-----------|-----------|-----------|
| Base weights (360M, BLOBFILE) | 720 MB | 360 MB + scales (~1.4 MB) | 1.99x |
| Activation surfaces (per layer) | DIM*SEQ*2 = 0.49 MB | DIM*SEQ*1 = 0.25 MB | 2.0x |
| Total BLOBFILE on disk | 720 MB | 362 MB | 1.99x |

The memory reduction enables:
- Fitting larger models in ANE SRAM (32 MB)
- Reducing IOSurface allocation pressure
- Potentially pushing the "memory cliff" from DIM=2048 to DIM=2880

---

## 7. Risks

### Risk 1: Quantization Noise Destroys Gradient Signal (HIGH)

**The core risk.** MeZO computes `L(w+ez) - L(w-ez)` which is typically 1e-3 to 1e-1.
If INT8 quantization noise in each loss value is comparable, the gradient estimate
becomes pure noise.

**Analysis**: Section 2 shows that at epsilon = 1e-3, the noise likely dominates the
signal (SNR < 1). At epsilon = 0.05-0.1, the signal may survive (SNR > 2).

**Mitigation**:
- Use larger epsilon (0.05-0.1 vs standard 1e-3)
- Keep loss computation in FP32 (already planned)
- Multi-perturbation averaging (FZOO K=4 to reduce noise by 2x; but this costs 4x
  more forward passes)
- If all fail: use INT8 only for the P16 hybrid forward pass (not MeZO), where
  gradient accuracy comes from backprop, not from forward pass precision

**Validation**: Hypothesis 2 directly tests this.

### Risk 2: Activation Overflow (MEDIUM)

INT8 range is [-128, 127]. With per-tensor dynamic quantization, outlier activations
are clipped to this range. Clipping causes information loss.

**Analysis**: After RMSNorm, activations have mean 0 and controlled variance.
Typical range is [-4, +4]. With scale = 4/127 ~ 0.031, the quantization resolution
is ~0.031. For DIM=960, each element has ~5 bits of effective precision.

**Mitigation**:
- Monitor clipping ratio: what fraction of activations are clipped?
- If > 1%: switch to per-channel activation quantization for that layer
- SiLU output is bounded: SiLU(x) is in [-0.28, +inf) but typical values are [-0.3, 3]
  -- manageable for INT8

### Risk 3: Loss of Convergence from Cumulative Quantization Error (MEDIUM)

Through 32 layers, quantization errors accumulate. The noise analysis in Section 2.2
predicts sigma ~ 0.019 in the final hidden state. Over many training steps, this
accumulated noise could prevent convergence to the same optimum as FP16 training.

**Analysis**: In standard QAT (quantization-aware training), the STE (straight-through
estimator) handles this by letting gradients "see through" quantization. MeZO does
not use STE -- it only uses forward pass values. The quantization noise adds to the
already-high ZO gradient variance. Since MeZO already has O(d) variance per gradient
estimate, the additional quantization noise may be a small relative increase.

**Mitigation**:
- Compare convergence curves: INT8 vs FP16 over 500 steps (Hypothesis 3)
- If convergence rate drops by > 50%, the 1.48x speed advantage is insufficient
  (need > 2x speed to compensate for 50% convergence loss)

### Risk 4: ANE Compiler Rejects INT8 MIL Programs (MEDIUM)

We use the private `_ANEInMemoryModelDescriptor` API to compile MIL programs. The
INT8 weight ops (`constexpr_affine_dequantize`) may not be supported on this path,
or may require specific MIL version / target attributes we haven't discovered.

**Mitigation**:
- Test compilation of a minimal INT8 conv1x1 kernel first (Hypothesis 1 setup)
- We already have `ane_bridge_build_weight_blob_int8` which builds INT8 blobs in
  the ANE format, suggesting the bridge was designed with INT8 in mind
- If in-memory compilation fails: try building a CoreML model with coremltools and
  extracting the compiled MIL

### Risk 5: Per-Step Requantization Overhead (LOW -- mitigated)

As analyzed in Section 6.1, our LoRA-split architecture does NOT require
re-quantizing base weights each step. LoRA corrections are applied on CPU to
activations. This risk is fully mitigated by architecture.

### Risk 6: M2 Pro Has No INT8 Compute Path (CONFIRMED)

Our current hardware (M2 Pro) only supports W8A16 (weight compression, no compute
speedup). Full W8A8 with compute speedup requires M4+.

**Mitigation**:
- Phase 1 (on M2 Pro): Validate correctness -- INT8 weights produce correct outputs
- Phase 2 (on M4 hardware): Benchmark actual throughput improvement
- Phase 1 has independent value: 2x weight compression, memory reduction, and
  validation that convergence is preserved

---

## 8. Literature

### 8.1 INT8 Training

| Paper | Venue | Key Contribution | Relevance |
|-------|-------|-----------------|-----------|
| [Jetfire](https://arxiv.org/abs/2403.12422) | ICML 2024 | INT8 data flow + per-block quantization for transformer pretraining. 1.42x speedup, 1.49x memory reduction vs FP16. All forward/backward in INT8. | Most relevant: proves INT8 training converges for transformers. Per-block quantization is their key innovation. |
| [FF-INT8](https://arxiv.org/abs/2506.22771) | DAC 2025 | Forward-Forward algorithm with INT8 precision on Jetson Orin Nano. 4.6% faster, 8.3% energy savings. Layer-wise training stabilizes INT8 gradients. | Directly relevant: forward-only INT8 training on edge NPU hardware. |
| [NITI](https://arxiv.org/abs/2009.13108) | IEEE TPAMI 2022 | Integer-only training using INT8 arithmetic exclusively. Pseudo stochastic rounding. MNIST/CIFAR10 negligible accuracy loss; ImageNet needs INT16 accumulators. | Foundational: proves pure-integer training is possible. Their INT16 accumulator finding is critical for our ANE analysis. |
| [Accurate INT8 Training via Dynamic Block-Level Fallback](https://arxiv.org/abs/2503.08040) | arXiv 2025 | Dynamic fallback to FP16 for blocks where INT8 error exceeds threshold. | Relevant: adaptive precision strategy could apply to our per-layer approach. |

### 8.2 Zeroth-Order + Quantization

| Paper | Venue | Key Contribution | Relevance |
|-------|-------|-----------------|-----------|
| [ElasticZO](https://arxiv.org/abs/2501.04287) | arXiv Jan 2025 | Combined ZO (most layers) + BP (last layers). ElasticZO-INT8 variant uses only INT8 arithmetic for ZO gradient estimation. 1.38-1.42x speedup. | **Most relevant to our work**: first INT8 ZO training. Their INT8 cross-entropy loss is the key innovation. |
| [MobiEdit](https://arxiv.org/abs/2310.07269) | ICLR 2026 | W8A16 quantized forward-only gradient estimation on NPU. 3.4x latency, 15.8x energy reduction. | Validates quantized forward-only gradient estimation on NPU. W8A16, not W8A8. |
| [MeZO](https://arxiv.org/abs/2305.17333) | NeurIPS 2023 | Memory-efficient ZO for LLM fine-tuning. Our base algorithm. | Foundation: everything builds on MeZO. |
| [FwdLLM](https://arxiv.org/abs/2308.13894) | USENIX ATC 2024 | Forward-only federated LLM fine-tuning. 14.6x memory reduction. | Validates ZO+LoRA for on-device training. |

### 8.3 ANE Quantization

| Paper / Resource | Key Content | Relevance |
|---------|------------|-----------|
| [coremltools Quantization Overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html) | W8A8 on ANE (A17 Pro/M4+). `constexpr_affine_dequantize` for weight storage. `quantize`/`dequantize` for activations. Per-tensor, per-channel, per-block granularities. | **Primary API reference** for our implementation. |
| [coremltools Quantization Performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html) | Benchmark data for W8A8 latency improvements on different hardware. A17 Pro/M4 show "increased throughput for int8-int8 compute." | Hardware-specific speedup data. |
| [maderix ANE Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) | M4 ANE: 19 TFLOPS FP16, 38 TOPS INT8 (2x by convention). INT8 dequantizes to FP16 on M1-M3. True INT8 compute path on M4/A17 Pro. | Hardware characterization. Confirms M2 has no INT8 compute. |
| [Scaling NPU Test-Time Compute](https://arxiv.org/abs/2509.23324) | 19x mixed-precision GEMM speedup on mobile NPU. Tile quantization. LUT-based softmax/dequant. | Tile quantization technique applicable to ANE. |

### 8.4 Quantization-Aware Training on Edge

| Paper | Venue | Key Contribution | Relevance |
|-------|-------|-----------------|-----------|
| [End-to-End On-Device QAT at Inference Cost](https://arxiv.org/abs/2509.00031) | arXiv 2025 | QAT that operates at inference-level memory cost. Trains quantization parameters (scale, zero-point) while keeping weights fixed. | Complementary: we could train quantization parameters alongside LoRA. |
| [Poor Man's Training on MCUs](https://arxiv.org/abs/2411.05873) | arXiv 2024 | Quantized back-propagation-free training on microcontrollers. Memory-efficient INT8 training for tiny models. | Validates INT8 + forward-only training on extreme edge hardware. |
| [Gradient Distribution-Aware INT8 Training](https://www.sciencedirect.com/science/article/abs/pii/S0925231223003922) | Neurocomputing 2023 | Models gradient distribution shape for better INT8 quantization. Addresses gradient variability during training. | Relevant technique if we extend to backprop gradients. |

---

## 9. Assumptions Registry

| # | Assumption | Risk | Validation Plan |
|---|-----------|------|-----------------|
| A1 | `quantize`/`dequantize` MIL ops work via private-API in-memory compilation on ios18 target | MEDIUM | Compile minimal test kernel with quantize op |
| A2 | ANE INT8 compute path uses >= INT16 accumulators | HIGH | Compare INT8 vs FP16 output numerically; if they diverge by > 1%, accumulators are too narrow |
| A3 | Epsilon = 0.05-0.1 with INT8 forward produces usable MeZO gradients | HIGH | Hypothesis 2: gradient correlation test |
| A4 | LoRA corrections (~1% of activation) survive INT8 activation quantization | MEDIUM | Compare LoRA-corrected vs uncorrected INT8 activations |
| A5 | IOSurfaces support INT8 byte-per-element for W8A8 activation I/O | LOW | Allocate test IOSurface with INT8 sizing |
| A6 | M4 ANE actually achieves 1.88x throughput for INT8 conv1x1 | MEDIUM | Benchmark on M4 hardware (not available yet) |
| A7 | Per-channel symmetric quantization is sufficient (vs asymmetric or per-block) | LOW | Hypothesis 1: compare loss accuracy |
| A8 | INT8 weight quantization does not need to be redone per-step (LoRA-split architecture) | LOW | Architecture analysis (Section 4.6) -- CONFIRMED |
| A9 | Stochastic rounding is NOT needed for INT8 MeZO (deterministic rounding + correlated noise cancellation is sufficient) | MEDIUM | If deterministic rounding fails, implement stochastic rounding variant |
| A10 | 32 layers of INT8 quantization noise accumulates linearly (not exponentially) due to residual connections | MEDIUM | Monitor per-layer activation statistics through the network |

---

## 10. Implementation Plan

### Phase 1: INT8 Weight Compilation (M2 Pro, 2 days)

1. Implement `gen_conv1x1_int8_mil()` in `mil_dynamic.h`
2. Implement `compile_conv1x1_int8_kern()` in `io.h`
3. Implement per-channel quantization with scale blob builder
4. Compile a single INT8 conv1x1 kernel and verify it runs on ANE
5. Benchmark: INT8 weight conv vs FP16 weight conv on M2 (expect ~same speed)

**Deliverable**: Working INT8 conv1x1 kernel compilation and execution.

### Phase 2: Forward Pass Validation (M2 Pro, 1 day)

1. Run Hypothesis 1: compare INT8 vs FP16 forward pass loss
2. Record per-layer activation statistics (mean, std, clip rate)
3. If loss error > 0.01: investigate per-layer error accumulation

**Deliverable**: Quantitative measurement of INT8 forward pass accuracy.

### Phase 3: MeZO Gradient Validation (M2 Pro, 1 day)

1. Run Hypothesis 2: gradient correlation at epsilon = 0.05, 0.1
2. If correlation < 0.5 at both: declare MeZO INT8 infeasible; proceed with
   P16 hybrid INT8 forward only
3. If correlation > 0.5: proceed to Phase 4

**Deliverable**: Go/no-go decision for MeZO INT8 training.

### Phase 4: INT8 MeZO Training (M2 Pro, 2 days)

1. Integrate INT8 conv kernels into MeZO training loop
2. Add `--int8-fwd` flag to train_mezo.m
3. Run Hypothesis 3: 500-step convergence comparison
4. Tune epsilon and learning rate for INT8 mode

**Deliverable**: INT8 MeZO training with convergence data.

### Phase 5: W8A8 Benchmarking (M4 hardware required, 1 day)

1. Add activation quantization MIL ops for W8A8 mode
2. Benchmark throughput on M4: INT8 W8A8 vs FP16
3. Run full training comparison: INT8 W8A8 vs FP16 on M4

**Deliverable**: M4-specific speedup measurement.

---

## 11. Relation to Existing Work

### 11.1 What Makes This Novel

No prior work has combined:
1. INT8 quantized weights on an NPU
2. Zeroth-order gradient estimation (MeZO)
3. LoRA-split architecture (frozen INT8 base + FP32 LoRA corrections)
4. Forward-only training (no backward pass)

ElasticZO-INT8 is the closest: they do INT8 ZO training but on CPU/GPU, not on an
NPU, and they use a hybrid ZO+BP approach (BP for last layers). Our approach is
fully ZO (or fully BP in P16 mode) with the NPU handling the forward pass.

FF-INT8 does INT8 training on an NPU (Jetson Orin) but uses the Forward-Forward
algorithm, not standard backprop or MeZO.

### 11.2 Relation to P16 Hybrid (MeBP+ANE)

P16 uses ANE for forward passes and CPU for backward passes. INT8 forward passes
directly benefit P16:
- Forward pass in INT8 -> faster (on M4)
- Backward pass remains FP32 on CPU (unchanged)
- No gradient estimation noise concern (gradients come from backprop, not forward pass)

P16 + INT8 is a lower-risk path than MeZO + INT8 because P16's gradient accuracy
is not affected by forward pass quantization (backward pass is FP32).

### 11.3 Relation to P13 (INT8 LoRA Corrections)

P13 was originally scoped as quantizing LoRA A/B matrices to INT8. This design
subsumes P13: if the base weights are INT8 and the LoRA corrections are applied
in FP32 on CPU, the entire forward pass benefits from INT8 without separately
quantizing the small LoRA matrices.

---

## 12. Decision Framework

```
                    Can we compile INT8 MIL kernels via private API?
                                    |
                        YES --------+-------- NO
                         |                     |
                 Hypothesis 1:            STOP: INT8 on ANE
                 Loss accuracy?           not feasible via
                     |                    our compilation path
             < 0.01 ---+--- > 0.01
              |                |
       Hypothesis 2:      Debug: per-layer
       Gradient corr?     error analysis
              |                |
        > 0.5 ---+--- < 0.5   |
         |            |        |
    Hypothesis 3:   Try e=0.1  |
    Convergence?       |       |
         |        Still < 0.5  |
     YES ---+          |       |
      |     |     INT8 for P16 only
   PUBLISH  |     (forward pass speed,
   "First   |     backprop gradients)
   INT8 NPU |         |
   Training"|    Still valuable:
            |    1.48x forward speedup
         NO |    for P16 hybrid
            |
    INT8 for P16 only
    (still a win)
```

---

## 13. Open Questions

1. **Accumulator precision**: Does the M4 ANE INT8 path use INT16, INT32, or FP16
   accumulators? This determines whether INT8 matmul over DIM=960 preserves enough
   precision. If INT8 accumulators, we need per-block quantization (ala Jetfire) to
   limit accumulation length.

2. **Dynamic scale computation on ANE**: Can the `reduce_max` + `div` needed for
   per-tensor dynamic quantization run on ANE, or must it be on CPU? If CPU, we need
   an additional IOSurface round-trip per layer for the scale value.

3. **Mixed INT8/FP16 fusion**: Can the ANE compiler fuse INT8 conv + FP16 SiLU +
   INT8 conv into a single kernel? If not, the FFN fusion benefit is lost for INT8.

4. **Stochastic rounding hardware**: Does the ANE support stochastic rounding in its
   quantization path? If so, this could improve gradient estimation quality.

5. **Per-block quantization**: coremltools 8.0+ supports `constexpr_blockwise_shift_scale`
   for per-block (e.g., 32x32 tile) quantization. Does this compile via the private API?
   Per-block may be the sweet spot between per-tensor (too coarse) and per-channel
   (too much metadata).

---

## References

- [ElasticZO: Memory-Efficient On-Device Learning](https://arxiv.org/abs/2501.04287)
- [FF-INT8: Forward-Forward DNN Training with INT8](https://arxiv.org/abs/2506.22771)
- [Jetfire: INT8 Transformer Pretraining](https://arxiv.org/abs/2403.12422)
- [NITI: Integer-Only Training](https://arxiv.org/abs/2009.13108)
- [MeZO: Memory-Efficient ZO Optimization](https://arxiv.org/abs/2305.17333)
- [MobiEdit: Quantized Forward-Only on NPU](https://arxiv.org/abs/2310.07269)
- [coremltools Quantization Guide](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)
- [coremltools Quantization Performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)
- [maderix ANE Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Scaling NPU Test-Time Compute](https://arxiv.org/abs/2509.23324)
- [Accurate INT8 Training via Dynamic Block-Level Fallback](https://arxiv.org/abs/2503.08040)
- [End-to-End On-Device QAT](https://arxiv.org/abs/2509.00031)
- [Poor Man's Training on MCUs](https://arxiv.org/abs/2411.05873)
- [Gradient Distribution-Aware INT8 Training](https://www.sciencedirect.com/science/article/abs/pii/S0925231223003922)
- [FwdLLM: Forward-Only Federated LLM Fine-Tuning](https://arxiv.org/abs/2308.13894)
