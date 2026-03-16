# Unconventional Training Methods: Beyond Backpropagation and MeZO

**Date**: 2026-03-17
**Purpose**: Survey the most radical, unconventional approaches to neural network training
that could be applied to NPU/ANE hardware. Focus on methods that don't exist yet or haven't
been combined with LoRA/LLMs.

---

## 1. Methods That Could Break Our ZO Ceiling

### 1.1 NoProp: Diffusion-Inspired Block-Local Training (arXiv:2503.24322, March 2025)

**How it works**: Treats each network block as a denoiser. Instead of propagating gradients,
each block independently learns to denoise a noisy version of the target label. At inference,
blocks sequentially refine noisy labels.

**Why it's radical**: No forward prop chain. No backward prop chain. Each block is trained
completely independently with local backprop only within the block.

**Applicability to our setup**:
- Each transformer layer becomes an independent denoiser
- Training: input = (activations, noisy_label) → output = denoised_label
- Only needs forward pass through single layer + local backward within layer
- **ANE could run the forward passes; CPU does the tiny local backward**
- MUCH less memory than full backprop (no cross-layer activation storage)

**Gap**: Only tested on MNIST/CIFAR. Never applied to transformers or LLMs.
**Novelty if we do it**: First NoProp for LLM fine-tuning. First NoProp on NPU.

**ASSESSMENT**: HIGH potential. The diffusion connection is elegant and the block-local
property perfectly matches ANE's forward-only constraint. The question: does the denoising
objective produce useful fine-tuning for language modeling?

### 1.2 Mono-Forward: Local Errors That Beat Backprop (arXiv:2501.09238, January 2025)

**How it works**: Each layer has its own local loss function. One forward pass per sample.
No backward pass at all. Claims to MATCH OR SURPASS backprop on MLP and CNN tasks.

**Why it's radical**: If true, this is strictly better than backprop for our hardware —
same quality, single forward pass, fully parallelizable across layers.

**Applicability**:
- Tested on MLP and CNN only (NOT transformers)
- If it works for transformers: one ANE forward pass per training step
- Combined with LoRA: train only adapter local losses, freeze base
- **Could be the fastest possible training method on ANE**

**Gap**: Never applied to transformers. Claims need independent verification.
**Novelty**: First Mono-Forward for transformers. First for LLMs. First on NPU.

**ASSESSMENT**: VERY HIGH potential if the claims hold for transformers. The "surpasses backprop"
claim is extraordinary and needs careful verification. But if even MATCHING backprop with
only forward passes, this is revolutionary for NPU training.

### 1.3 Synergistic Information Distillation (SID) (arXiv:2510.03273, October 2025)

**How it works**: Each module refines a probabilistic belief about the target. Balances
fidelity to the actual target with consistency to the preceding module's belief.

**Why it's radical**: Matches or surpasses backprop. Preserves standard feed-forward
inference (drop-in replacement for backprop). Theoretically guaranteed monotonic
improvement with network depth.

**Applicability**:
- Tested on classification tasks
- The probabilistic belief framework could work for next-token prediction
- Each transformer layer refines the token prediction
- **Local training = perfect for ANE forward-only hardware**

**Gap**: Not tested on LLMs. The belief propagation might not work for autoregressive generation.
**Novelty**: First SID for LLMs.

**ASSESSMENT**: MEDIUM-HIGH. The theoretical guarantees are attractive but untested for language.

---

## 2. Evolutionary / Population-Based Methods

### 2.1 Evolution Strategies at Scale (arXiv:2509.24372, September 2025)

**How it works**: Apply ES (population-based optimization) directly to LLM parameters.
No gradients at all. Evaluate fitness of parameter perturbations.

**Key finding**: First successful application to billion-parameter LLMs WITHOUT dimensionality
reduction. Outperforms RL methods for fine-tuning.

**Why it matters for us**: ES is embarrassingly parallel. Each population member only needs
a forward pass. ANE could evaluate N population members simultaneously.

**Gap**: Memory for N copies of a 360M model. Communication overhead between population members.
On a single device, ES is equivalent to MeZO (population size 1). Need multiple devices for benefit.

**ASSESSMENT**: LOW for single-device ANE. HIGH for multi-device federated ANE training.

### 2.2 ESSAM: Competitive ES with Sharpness-Aware Optimization (arXiv:2602.01003, February 2026)

**How it works**: Combines ES with Sharpness-Aware Maximization for better generalization.
Memory-efficient LLM fine-tuning.

**ASSESSMENT**: Interesting but same single-device limitation as ES.

---

## 3. Test-Time Training / Inference-Time Adaptation

### 3.1 Test-Time Learning with LoRA (TLM) (arXiv:2505.20633, May 2025)

**How it works**: Dynamically adapts LLMs at test time using unlabeled test data.
Uses LoRA for lightweight updates. No training phase at all — the model adapts
during inference.

**Why it's radical**: Blurs the line between inference and training. The model continuously
improves as it processes data. 20%+ improvement vs base model on domain adaptation.

**Applicability to ANE**:
- ANE runs the forward pass (inference)
- After each inference, LoRA adapters are updated based on the input
- The "training" IS the inference — no separate training loop
- **ANE hardware is always doing useful training work during every inference**

**Gap**: Needs some form of gradient for LoRA updates. Could use ZO gradients.
**Novelty**: First test-time LoRA training on NPU. Continuous on-device personalization.

**ASSESSMENT**: VERY HIGH impact. This is the killer app for ANE training — the device
learns and personalizes DURING EVERY INFERENCE. No separate training phase needed.
This is what Apple's on-device ML strategy should look like.

### 3.2 Test-Time Training for Long-Context (arXiv:2512.13898, December 2025)

**How it works**: Reframes language modeling as continual learning. Model adapts in
real-time as it processes new information. 2.7x faster than Full-Attention Transformer
at 128k context.

**ASSESSMENT**: MEDIUM. More about long context than on-device training.

---

## 4. Predictive Coding

### 4.1 Predictive Coding Networks for Transformers (2025+)

**How it works**: Each layer predicts the activity of the next layer. The error between
prediction and actual activity drives local weight updates. No global backward pass.

**Why it's relevant**: Local, Hebbian-style updates. Energy-based optimization.
Natural fit for forward-only hardware.

**Recent progress**: New precision-weighting and residual connection modifications
achieve performance comparable to backprop on deep ResNets (2025).

**Gap**: Not yet applied to transformers at LLM scale. Gradient explosion/vanishing
in deep networks (32 layers for SmolLM2-360M).

**ASSESSMENT**: MEDIUM. Promising theory but scaling challenges for deep transformers.

---

## 5. Weight-Space Generative Models

### 5.1 Diffusion-Based Weight Generation (arXiv:2402.18153)

**How it works**: Train a diffusion model that GENERATES neural network weights.
Instead of optimizing weights by gradient descent, sample good weights from a
learned distribution.

**Why it's radical**: Completely replaces training. Given a task description, the
diffusion model generates appropriate weights in one shot.

**Applicability**: Could generate LoRA adapters for specific tasks. One forward pass
through the weight generator → complete LoRA adapter. No iterative training at all.

**Gap**: Currently works for small models only. Generating 1.7M LoRA parameters
would require a substantial weight-space diffusion model.

**ASSESSMENT**: LOW for current capabilities. HIGH for future (if weight generators scale).

### 5.2 Geometric Flow Models over Weights (arXiv:2504.03710, April 2025)

**How it works**: Uses flow matching (continuous normalizing flows) to model the
distribution of neural network weights, respecting the geometric symmetries
(permutations, scaling) of weight space.

**ASSESSMENT**: Theoretical. Not applicable to fine-tuning yet.

---

## 6. PeZO and MeZO Variants

### 6.1 PeZO: Perturbation-Efficient ZO (arXiv:2504.20314, April 2025)

**How it works**: Addresses the PRNG bottleneck in ZO methods. Replaces Gaussian
perturbations with uniform distribution (hardware-friendly). Reuses random numbers
across steps.

**Applicability**: Reduces LUT/FF usage by 48.6% for FPGA. Power savings 86%.
**But**: Our ANE uses xoshiro256+ PRNG which is already fast (21ms vs 700ms for Box-Muller).
The PRNG is not our bottleneck.

**ASSESSMENT**: LOW for ANE (we don't have the PRNG bottleneck).

### 6.2 MeZO-BCD: Block Coordinate Descent

**How it works**: Perturbs and updates only a subset of parameters per step.
2.77x speedup on OPT-13B.

**Applicability**: Could be combined with LoRA — perturb one LoRA matrix per step.
**But**: Our Sparse MeZO results (Phase 5a) showed that reducing active parameters
hurts LoRA ZO. BCD would have the same problem.

**ASSESSMENT**: LOW (same failure mode as Sparse MeZO for LoRA).

---

## 7. THE MOST PROMISING UNCONVENTIONAL DIRECTIONS (Ranked)

| Rank | Method | Novelty | Expected Impact | Feasibility | Key Risk |
|------|--------|---------|-----------------|-------------|----------|
| **1** | **Test-Time LoRA (TLM) on ANE** | VERY HIGH | **Killer app** — continuous personalization | HIGH | Need gradient signal |
| **2** | **NoProp + LoRA for LLMs** | VERY HIGH | Block-local, no cross-layer deps | MEDIUM | Denoising ≠ language modeling |
| **3** | **Mono-Forward for Transformers** | VERY HIGH | Could beat backprop forward-only | MEDIUM | Untested on transformers |
| **4** | **Forward-Forward + LoRA** | HIGH | Proven concept, clean math | HIGH | Local ≠ global objective |
| **5** | **SID for LLMs** | HIGH | Theoretical guarantees | MEDIUM | Belief propagation for LM? |
| **6** | **Predictive Coding Transformers** | MEDIUM | Bio-plausible, local | LOW | Scaling to 32 layers |
| **7** | **Evolution Strategies (multi-device)** | MEDIUM | Embarrassingly parallel | LOW | Need multiple devices |

## 8. THE RADICAL PROPOSAL

The most impactful thing we could build is **Test-Time LoRA Training on ANE**:

Every inference the ANE runs (Siri queries, text predictions, photo processing)
becomes a training step. The model continuously personalizes to the user.
No separate training phase. No explicit fine-tuning. The device just gets
smarter over time by processing your data.

Implementation:
1. ANE runs inference (conv-fused forward pass, ~130ms for SmolLM2-360M)
2. After each inference, compute a ZO gradient estimate (one extra forward pass)
3. Update LoRA adapters (~1ms)
4. Adapter improves predictions for YOUR specific use patterns

Total overhead: ~130ms per inference (2x baseline). In exchange: continuous,
private, on-device personalization.

This is what Apple should be doing with the ANE. We can build the prototype.

## References

- [NoProp](https://arxiv.org/abs/2503.24322) — Training without forward or backward propagation
- [Mono-Forward](https://arxiv.org/abs/2501.09238) — Local errors surpass backprop
- [SID](https://arxiv.org/abs/2510.03273) — Learning without global backpropagation
- [ES at Scale](https://arxiv.org/abs/2509.24372) — Evolution strategies for LLM fine-tuning
- [TLM](https://arxiv.org/abs/2505.20633) — Test-time learning with LoRA
- [TTT-E2E](https://arxiv.org/abs/2512.13898) — Test-time training for long-context LLMs
- [PeZO](https://arxiv.org/abs/2504.20314) — Perturbation-efficient ZO
- [Geometric Flows](https://arxiv.org/abs/2504.03710) — Flow models over weight space
- [Weight Diffusion](https://arxiv.org/abs/2402.18153) — Diffusion-based weight generation
- [Beyond Backprop Survey](https://arxiv.org/abs/2509.19063) — Energy-efficient training
- [Predictive Coding](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881729/) — PC inspires BP alternatives
- [FF-INT8](https://arxiv.org/abs/2506.22771) — Forward-Forward with INT8 on edge
- [ESSAM](https://arxiv.org/abs/2602.01003) — ES with sharpness-aware maximization
- [On-Device ZO](https://arxiv.org/abs/2511.11362) — Backprop-free on-device fine-tuning
- [MobiZO](https://aclanthology.org/2025.emnlp-main.1022.pdf) — Mobile edge LLM fine-tuning
- [Self-Improving Agents](https://arxiv.org/abs/2510.07841) — Test-time self-improvement
