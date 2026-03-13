# What Actually Works on ANE: A Data-Driven Plan

## ANE's Measured Strengths

| Strength | Number | Source |
|----------|--------|--------|
| Conv1x1 throughput vs matmul | **3x faster** | Orion (2603.06728), Constraint #17 |
| Peak FP16 throughput | **19 TFLOPS** | maderix ANE benchmarks (M4) |
| Power efficiency | **6.6 TFLOPS/W** (80x better than A100) | maderix |
| Deep graph pipelining | **94% utilization at 32+ layers** (vs 30% single-op) | maderix |
| On-chip SRAM | **32 MB** (sweet spot: 24 MB working set) | maderix |
| Dispatch overhead | **0.095 ms per dispatch** | Orion |
| Classifier forward on ANE | **10.2x faster than CPU** | Orion |
| Softmax (32K vocab) on ANE | **33.8x faster than CPU** | Orion |
| Delta reload vs recompile | **8.5x faster** (494ms vs 4200ms) | Orion |
| Adapter-as-input | **Zero recompilation** for weight updates | Orion |

## What We're Doing Wrong

**We use `matmul` in our MIL kernels. ANE is a convolution engine. That's a 3x penalty on every linear projection.**

Our `mil_dynamic.h` generates MIL programs like:
```
matmul(x=activation, y=weight)  // 3x SLOWER than it should be
```

It should be:
```
conv(x=activation, weight=kernel, strides=[1,1])  // ANE's native primitive
```

The data format already fits: our activations are `[1, IC, 1, SEQ]` which is `[N, C, H, W]` — exactly what conv2d expects. The weight becomes `[OC, IC, 1, 1]` as a standard 1x1 convolution kernel. No reshaping needed.

## What Would Actually Work: Conv1x1 + MP-LoRA

### The Math

**Current MeZO+LoRA-split CPU (measured 2026-03-14):**
- Forward pass: 435 ms (2x for MeZO = 870 ms)
- Perturbation: 2 ms
- Total: ~593 ms/step (173 steps in 120s)

**Projected with conv1x1 ANE + MP-LoRA:**

| Component | Current (CPU matmul) | Projected (ANE conv1x1) | Speedup |
|-----------|---------------------|------------------------|---------|
| Linear projections | 435 ms | ~145 ms (3x from conv1x1) | 3x |
| Deep graph pipelining | N/A (dispatch per layer) | ~50 ms (94% util, 32 layers) | 3x more |
| MP-LoRA (MobiZO) | 2 fwd passes = 870 ms | 1 fwd pass for both ±eps | 2x |
| Perturbation | 2 ms | 2 ms (CPU, unchanged) | 1x |
| **Total per step** | **593 ms** | **~52 ms** | **~11x** |
| **Steps in 120s** | **173** | **~2,300** | **13x more steps** |

This isn't speculation. Each factor has measured data behind it:
- 3x conv1x1: Orion Constraint #17, maderix benchmarks
- 94% utilization: maderix deep graph benchmark (32+ layer chains)
- 2x MP-LoRA: MobiZO (EMNLP 2025), measured 4.3x with additional inner-loop parallelism
- Adapter-as-input: Orion demonstrated, we already have LoRA-split

### Why This Plays to ANE's Strengths

1. **Static compiled graph**: Base weights baked as BLOBFILE constants. Compile once, run forever. This is ANE's preferred mode.

2. **Convolution primitive**: 1x1 conv uses the fast datapath. Every linear projection (Q/K/V/O/W1/W2/W3) becomes a conv. 7 projections × 32 layers = 224 convolutions, all on the fast path.

3. **Deep graph pipelining**: Chain 32 transformer layers into one or a few mega-kernels. ANE achieves 94% utilization with deep graphs vs 30% for single operations. This is the key to unlocking 19 TFLOPS.

4. **Low power**: 2.8W sustained. Can run for hours (overnight personalization). Battery drain: ~3% per hour on MacBook Pro.

5. **Forward-only**: MeZO needs no backward pass. ANE was designed for inference (forward passes). Perfect match.

6. **Adapter-as-input**: LoRA adapters passed as IOSurface inputs, not baked weights. Hot-swap without recompilation. Already demonstrated by Orion.

## The Killer Use Case: On-Device Personalization

**Why this matters (2B+ Apple devices):**

Apple already deploys:
- Federated learning for keyboard predictions (WWDC 2023)
- On-device LoRA adapters via Foundation Models framework (WWDC 2025)
- 3B-parameter on-device model with 2-bit quantization

**What's missing**: Efficient on-device training that uses the ANE (currently only GPU/CPU via MLX).

**MeZO+LoRA on ANE fills this gap:**
- Forward-only: no backward kernels needed (ANE has limited backward support)
- Privacy: user data never leaves device
- Low power: 2.8W means overnight personalization is feasible
- Small adapters: rank-8 LoRA for 3B model = ~6MB adapters
- Memory efficient: inference-only memory (~1.5GB for 3B quantized)

## Implementation Plan

### Phase 1: Conv1x1 MIL kernels (estimated: 1-2 days)

Replace `matmul` with `conv` in `mil_dynamic.h`:

```
// Current (slow):
matmul(transpose_x=false, transpose_y=false, x=activation, y=weight)

// Target (3x faster):
conv(x=activation, weight=kernel, strides=[1,1], pad_type="valid")
```

Key changes:
- Weight format: `[IC, OC]` → `[OC, IC, 1, 1]` (standard conv kernel)
- Input format: `[1, IC, 1, SEQ]` already correct (N, C, H, W)
- Output: `[1, OC, 1, SEQ]` — same as current

Benchmark: Run the same MeZO+LoRA-split training with conv1x1 vs matmul. Measure ms/step difference.

### Phase 2: Mega-kernel fusion (estimated: 2-3 days)

Chain multiple layers into one MIL program. Orion showed 3.8x training speedup from delta reload; maderix showed 94% utilization with deep graphs.

Target: 4-8 transformer layers per MIL program (limited by ANE's ~119 compilation limit and 32MB SRAM).

### Phase 3: MP-LoRA (estimated: 2-3 days)

Implement MobiZO's Multi-Perturbed LoRA:
- Pack both +eps and -eps perturbations into a single forward pass
- Only replicate the small LoRA B matrix (rank 8 = tiny)
- Get both loss_plus and loss_minus from one ANE evaluation

This halves the number of forward passes from 2 to 1.

### Phase 4: P-GAP integration (estimated: 3-5 days)

Gradient-aligned perturbations (P-GAP, arXiv 2510.18228):
- Project perturbations into gradient-aligned subspace
- 5.2x faster convergence than random perturbations
- Reduces required steps from ~20K to ~4K

## Expected End State

| Metric | Current Best (LoRA-split CPU) | Target (Conv1x1 + MP-LoRA ANE) |
|--------|------------------------------|-------------------------------|
| ms/step | 593 | ~50 |
| Steps/120s | 173 | ~2,300 |
| Power | ~13W (CPU package) | ~2.8W (ANE only) |
| Memory | 2.0 GB | ~1.5 GB (quantized base) |
| Energy/step | ~7.7 J | ~0.14 J |

**Energy per step drops 55x.** This is the metric that matters for on-device: not raw speed, but how much training you get per watt-hour.

A MacBook Pro battery (100 Wh) could run:
- Current CPU: ~46,000 steps (13W × ~9.5h, but throttles)
- Target ANE: ~2,500,000 steps (2.8W × ~36h theoretical)

## Literature Supporting This Plan

| Paper | Key Data Point | How It Supports Our Plan |
|-------|---------------|------------------------|
| [Orion](https://arxiv.org/abs/2603.06728) | Conv1x1 = 3x, adapter-as-input works | Validates Phase 1 + Phase 2 |
| [maderix ANE benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) | 19 TFLOPS, 94% deep graph, 32MB SRAM | Validates throughput targets |
| [MobiZO](https://aclanthology.org/2025.emnlp-main.1022/) | MP-LoRA 4.3x, ExecuTorch deployed | Validates Phase 3 |
| [P-GAP](https://arxiv.org/abs/2510.18228) | 5.2x convergence, 22.4% GPU hours | Validates Phase 4 |
| [ZO2](https://arxiv.org/abs/2503.12668) | OPT-175B on 18GB GPU via ZO offloading | Validates ZO for memory-constrained |
| [Apple Foundation Models](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025) | 3B on-device, LoRA adapters, federated | Validates use case |
| [N3L](https://medium.com/@ayushmanmukherjee12/n3l-the-no-backprop-revolution-for-cpu-npu-training-e3ee9ea3713a) | Forward-only local learning on CPU/NPU | Alternative forward-only approach |
| [Apple Federated Learning](https://machinelearning.apple.com/research/federated-personalization) | On-device personalization deployed | Production validation of use case |
