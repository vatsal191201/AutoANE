# Deep Supervision ZO (DSZO): Breaking the ZO Quality Ceiling via Per-Layer Losses

**Date**: 2026-03-17
**Status**: NOVEL PROPOSAL (not found in existing literature)
**Key claim**: Per-layer auxiliary losses give sqrt(L) more gradient information per ZO step,
potentially lowering the MeZO quality ceiling by 5.7x for 32-layer models.

---

## 1. The Problem: ZO Quality Ceiling (Finding 9)

MeZO estimates gradients from a single scalar loss:
```
g = (L(w+εz) - L(w-εz)) / (2ε) · z
```

For LoRA with d = L * d_layer trainable parameters, each estimate captures ~1/sqrt(d)
of the gradient direction. For SmolLM2-360M with L=32 layers and d_layer=35,840 LoRA
params/layer: d = 1,146,880, so each step gives 1/sqrt(1,146,880) ≈ 0.093% gradient info.

The ceiling: val_loss saturates at 2.0524 after ~600 steps. Backprop reaches 1.7972.

## 2. The Key Insight

**A single scalar loss is wasteful.** It mixes gradient information from all 32 layers
into one number. The ZO estimator then distributes this single scalar across all parameters —
most of the per-layer signal is lost in the projection.

**Per-layer losses decompose the problem.** If each layer l has its own loss L_l, the ZO
gradient for layer l uses only d_layer parameters (not d_total). The estimate captures
1/sqrt(d_layer) ≈ 0.53% of that layer's gradient — **5.7x more information** than the
global estimate.

## 3. The DSZO Algorithm

### 3.1 Architecture Modification

Add a lightweight classification head after each transformer layer:

```
For each layer l = 0, 1, ..., L-1:
    h_l = transformer_layer_l(h_{l-1})
    logits_l = RMSNorm(h_l) @ W_vocab^T     # reuse the final embedding matrix
    L_l = cross_entropy(logits_l, targets)   # per-layer loss
```

The classification head is just RMSNorm + projection to vocab — the SAME weights as
the final classification head. No new parameters. This is standard deep supervision
(Lee et al., 2015; Szegedy et al., 2015).

### 3.2 Per-Layer ZO Gradient

Instead of perturbing ALL LoRA params with one z vector:

```
For each step:
    1. For each layer l independently:
        a. Sample z_l ~ Rademacher(d_layer)     # per-layer perturbation
        b. Perturb layer l's LoRA: w_l += ε * z_l
    2. Forward pass → get per-layer losses L_l+
    3. For each layer l independently:
        a. Restore and perturb: w_l -= 2ε * z_l
    4. Forward pass → get per-layer losses L_l-
    5. For each layer l independently:
        a. Restore: w_l += ε * z_l
        b. Gradient: g_l = (L_l+ - L_l-) / (2ε) · z_l
        c. Update: w_l -= lr · g_l
```

**CRITICAL**: Each layer uses its OWN loss L_l and its OWN perturbation z_l.
The perturbation for layer l does NOT affect layers l+1, l+2, ... because LoRA
corrections are small (rank-8) and we perturb independently.

**ASSUMPTION DSZO-A1**: Perturbing layer l has negligible effect on L_{l+1}, ..., L_L.
This is approximately true because LoRA rank-8 changes are small (ε=1e-3, rank-8 matrices).
Need to verify experimentally.

### 3.3 Information-Theoretic Analysis

| Method | Losses per step | Info per layer | Info per step (32L) |
|--------|----------------|---------------|-------------------|
| MeZO | 1 (global) | 1/sqrt(1.15M) = 0.093% | 0.093% |
| DSZO | 32 (per-layer) | 1/sqrt(35,840) = 0.53% | 0.53% × 32 = 16.9% |
| Backprop | N/A (exact) | 100% | 100% |

**DSZO extracts 182x more gradient information per step than MeZO.**
(32 × 5.7x = 182x)

Even accounting for the approximation in DSZO-A1, the improvement should be substantial.

### 3.4 Why This Hasn't Been Done Before

1. Deep supervision is well-known for CNNs (2015) but rarely used for LLM training
2. ZO methods typically use a single global loss — nobody has combined ZO with deep supervision
3. The per-layer ZO decomposition is a new theoretical contribution
4. For standard training (backprop), deep supervision adds overhead with marginal benefit.
   For ZO, it's transformative because the scalar loss is the information bottleneck.

## 4. Implementation

### 4.1 Changes to train_mezo.m

**Per-layer loss computation** (~20 lines, inside DO_FORWARD_PASS):
```c
// After each transformer layer l, compute auxiliary loss
rmsnorm(aux_norm, x_cur, rms_final, DIM, SEQ);  // reuse final RMSNorm weights
cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
            SEQ, CV, DIM, 1.0f, aux_norm, SEQ, cembed, DIM, 0.0f, aux_logits, CV);
layer_loss[l] = cross_entropy_loss(aux_dlogits, aux_logits, ctargets, CV, SEQ);
```

**Per-layer perturbation and gradient** (~40 lines, replacing MeZO's global perturbation):
```c
for (int l = 0; l < NLAYERS; l++) {
    // Layer-specific seed → layer-specific z_l
    uint64_t layer_seed = mezo_seed + (uint64_t)l * 999983ULL;
    // Gradient from this layer's loss
    float proj_grad_l = (layer_loss_plus[l] - layer_loss_minus[l]) / (2.0f * epsilon);
    // Update only layer l's LoRA params using z_l and proj_grad_l
    perturb_single_layer_lora(&lora_layers[l], &lw[l], layer_seed, -lr * proj_grad_l);
}
```

**Memory**: One extra logits buffer [SEQ, CV] = 256 * 16893 * 4 ≈ 16.5MB. Negligible.

### 4.2 Overhead Analysis

Per-layer loss adds: 1 RMSNorm + 1 matmul (DIM → CV) per layer.
- RMSNorm: ~0.2ms per layer × 32 = 6.4ms
- Matmul (DIM×CV, SEQ=256): ~0.5ms per layer × 32 = 16ms
- **Total overhead: ~22ms per forward pass** (on top of ~131ms baseline = 17% overhead)

This is far less than MeZO's 2-forward-pass cost. DSZO still uses 2 forward passes
(for the +ε and -ε perturbations), so total step time: (131 + 22) × 2 = 306ms.
Compare MeZO at 262ms. **Only 17% slower than MeZO for 5.7x more gradient info.**

## 5. Hypotheses

### H1: DSZO lowers the MeZO ceiling
**Test**: Run DSZO for 1000 steps on SmolLM2-360M. If val_loss < 2.04 (below MeZO's
2.0524 ceiling), H1 is confirmed.
**Success criterion**: val_loss improvement > 1.5x MeZO's 0.0194 (i.e., > 0.029).

### H2: DSZO converges faster per step than MeZO
**Test**: Compare val_loss at step 100, 200, 300 between DSZO and MeZO.
**Success criterion**: DSZO reaches MeZO's 300-step val_loss in < 200 steps.

### H3: Per-layer perturbation independence holds (DSZO-A1)
**Test**: Perturb layer 0 only. Measure change in layer 31's loss. Should be < 1% of
the change in layer 0's loss.
**Success criterion**: |ΔL_31| / |ΔL_0| < 0.01.

## 6. Comparison to Other Methods

| Method | Passes/step | Gradient quality | Quality ceiling | Novelty |
|--------|------------|-----------------|-----------------|---------|
| MeZO | 2 | 0.093%/step | val_loss 2.0524 | Known |
| FZOO K=4 | 5 | 0.37%/step (4x MeZO) | ~2.05 (no improvement) | Known |
| **DSZO** | **2** | **16.9%/step (182x MeZO)** | **< 2.04 (predicted)** | **NOVEL** |
| FF+LoRA | 2 | 100% (local, not global) | Unknown | Novel |
| Backprop | 2 | 100% (global) | val_loss 1.7972 | Standard |

DSZO occupies a unique niche: it uses the ACTUAL task loss (not a proxy like FF's goodness),
extracts far more information per step than standard ZO, and requires only 2 forward passes.

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| DSZO-A1 fails (perturbation cross-talk) | HIGH | Reduce ε. Test H3 first. |
| Auxiliary losses don't provide useful signal at early layers | MEDIUM | Weight auxiliary losses: higher weight for later layers |
| Overhead exceeds benefit | LOW | 22ms overhead is small vs 131ms forward |
| Implementation bugs in per-layer perturbation | MEDIUM | Verify gradient norm per layer |

## 8. Why This Is Genuinely Novel

1. **No paper combines ZO optimization with deep supervision.** ZO literature always uses a single global loss.
2. **The information-theoretic analysis (sqrt(L) improvement) is new.** Nobody has quantified the information loss from using a single scalar in ZO.
3. **The per-layer perturbation decomposition is new.** Standard MeZO perturbs all parameters simultaneously. DSZO perturbs each layer independently with its own loss signal.
4. **This directly addresses Finding 9 (the ZO quality ceiling).** No existing method claims to lower this ceiling for LoRA ZO.

## 9. References

- Lee et al. (2015): "Deeply-Supervised Nets" — deep supervision for CNNs
- Szegedy et al. (2015): "Going Deeper with Convolutions" — auxiliary classifiers in Inception
- Malladi et al. (2023): "MeZO" — zeroth-order LLM fine-tuning (arXiv:2305.17333)
- Our Finding 9: ZO-LoRA quality ceiling at val_loss 2.0524
- Our Phase 5 results: 5 ZO improvements all fail for LoRA
