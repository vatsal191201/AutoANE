# P1: Runtime Weight Injection via Dual-Input IOSurfaces

**Created**: 2026-03-12 | **Status**: Design Complete, Not Yet Implemented

---

## Problem Statement

Our current ANE training pipeline uses **single-surface spatial packing**: weights and activations are concatenated in the spatial dimension of one IOSurface, then separated by `slice_by_size` inside the MIL program. This works but has overhead:

1. **Per-step activation I/O**: Every forward/backward step converts fp32 activations to fp16 and writes them to the activation region of the IOSurface. The weight region is untouched (already staged).
2. **Per-update weight staging**: After every Adam update (every ACCUM_STEPS), all weights are transposed, converted fp32→fp16, and written to the weight region of each layer's IOSurface.
3. **slice_by_size overhead**: The MIL programs emit 2 slice ops per matmul to extract activations and weights from the packed input, plus reshape/transpose before matmul.
4. **Oversized IOSurfaces**: Each surface is `IC * (SEQ + OC) * 2` bytes instead of `IC * SEQ * 2` for activations alone.

The upstream `imperatormk/ane-train` project demonstrated that using **separate IOSurfaces** for weights and activations (dual-input MIL functions) eliminates spatial packing entirely and enables persistent fp16 weight buffers.

## Upstream Findings (imperatormk/ane-train)

### Key Architecture Differences

| Feature | Our approach (spatial packing) | imperatormk (dual-input) |
|---------|-------------------------------|--------------------------|
| MIL function signature | 1 input: `[1, IC, 1, SEQ+OC]` | 2 inputs: W`[1,Cout,Cin]`, X`[1,Cin,S]` |
| Tensor rank | 4D `[1,C,1,S]` | 3D `[1,M,N]` |
| Weight extraction | `slice_by_size` inside MIL | Direct — W is a function parameter |
| IOSurface count per kernel | 1 input + 1 output | 2 inputs + 1 output |
| Weight update | `stage_*_weights()` fp32→fp16 to weight region | `memcpy` to weight IOSurface (~0.001ms) |
| Compile-time weight dependency | None (already runtime) | None |
| MIL target | `ios18` | `ios16` (broader compatibility) |

### Critical Constraints Discovered

1. **Slot ordering**: Input IOSurfaces must be in **ascending byte-size order**. Weight surface (slot0) must be ≤ activation surface (slot1). This means `Cout * Cin ≤ Cin * S`, i.e., **Cout ≤ S**. For our SEQ=128, this means output dimension must be ≤128. Our Q_DIM=512 and HIDDEN=1408 violate this.

2. **3D tensor requirement**: Dual-input matmul uses `[1, M, N]` tensors, not the 4D `[1,C,1,S]` our pipeline uses. This requires changing the MIL generation.

3. **Inner dim alignment**: Contraction dimension must be a multiple of 32. Our DIM=512 satisfies this.

4. **Minimum buffer size**: IOSurface must be ≥2048 bytes. All our weight matrices exceed this.

5. **MIL variable names**: Must avoid reserved words (`var`, `mean`, `x`, `y`, `mul`, `add`, `rsqrt`, `c`, `C`) — violations cause silent zeros.

### The Cout ≤ S Problem

This is the **critical blocker** for direct adoption. Our matmul dimensions:

| Kernel | IC→OC | Cout | S=SEQ=128 | Cout ≤ S? |
|--------|-------|------|-----------|-----------|
| Wq | DIM→Q_DIM | 512 | 128 | **NO** |
| Wk | DIM→KV_DIM | 128 | 128 | YES |
| Wv | DIM→KV_DIM | 128 | 128 | YES |
| Wo | Q_DIM→DIM | 512 | 128 | **NO** |
| W1 | DIM→HIDDEN | 1408 | 128 | **NO** |
| W2 | HIDDEN→DIM | 512 | 128 | **NO** |
| W3 | DIM→HIDDEN | 1408 | 128 | **NO** |

Only Wk and Wv satisfy the constraint. All other kernels would need a workaround.

### Possible Workarounds

**Option A: Transpose the matmul.** Instead of `Y = W @ X` with W as slot0, compute `Y^T = X^T @ W^T` with X^T as slot0 (now the larger tensor). This requires `S * Cin ≤ Cin * Cout`, i.e., `S ≤ Cout` — which is the opposite constraint and is always satisfied for our dimensions. But this changes the output layout and requires pre/post transposition.

**Option B: Increase SEQ.** At SEQ=512, all kernels except W1/W3 (HIDDEN=1408) would satisfy Cout ≤ S. But SEQ is a fundamental hyperparameter affecting training dynamics.

**Option C: Split large matmuls.** Decompose W1 (512→1408) into two matmuls: 512→704 and 512→704, then concat. But this adds complexity and XPC overhead.

**Option D: Hybrid approach.** Use dual-input for kernels that satisfy the constraint (Wk, Wv), keep spatial packing for the rest. Reduces benefit but avoids the constraint entirely.

**Option E: Fall back to spatial packing for violating kernels.** imperatormk notes that when Cout > S, the operation should fall back. Our current spatial packing already works as this fallback.

## Recommended Implementation Plan

Given the Cout ≤ S constraint, a full switch to dual-input is not straightforward for our architecture. The recommended approach is **incremental**:

### Phase 1: Persistent fp16 Weight Buffers (No MIL Changes)

Eliminate redundant fp32→fp16 conversion by maintaining a persistent fp16 copy of weights alongside fp32 master weights:

1. Allocate `_Float16 *Wq_fp16, *Wk_fp16, ...` for each layer
2. After Adam update: convert fp32→fp16 once, store in fp16 buffer
3. During weight staging: `memcpy` from fp16 buffer to IOSurface (no conversion)
4. Net saving: eliminates one fp32→fp16 conversion per staging call

**Risk**: Low. No MIL changes. Memory cost: ~2x weight storage (fp32 + fp16).

### Phase 2: Dual-Input for Eligible Kernels

Switch Wk and Wv kernels to dual-input (they satisfy Cout ≤ S=128):

1. Generate 2-input MIL: `matmul(W[1,128,512], X[1,512,128])`
2. Create separate weight IOSurfaces that persist between steps
3. Use `ane_rewire()` pattern for zero-copy weight swaps

**Risk**: Medium. Requires new MIL generation path and 2-input `_ANERequest` construction.

### Phase 3: Transposed Matmul for Remaining Kernels

Explore Option A (transposed matmul) for Wq, Wo, W1, W2, W3:

1. Generate `Y^T = X^T @ W^T` with activation as slot0
2. Post-transpose the output
3. Benchmark to see if transpose overhead negates the spatial-packing savings

**Risk**: High. Untested pattern. May not work on all ANE chips.

## Decision: Start with Phase 1

Phase 1 provides a concrete speedup with zero risk. It can be implemented in a few hours and benchmarked immediately. Phases 2-3 are research experiments that may or may not pay off.

## Estimated Impact

- **Phase 1**: Saves ~1-2ms per Adam update cycle (one fewer fp32→fp16 pass over all weights). Minor improvement.
- **Phase 2**: Saves ~0.1ms per step for Wk/Wv (no slice_by_size, smaller IOSurface). Marginal.
- **Phase 3**: If it works, could save ~2-4ms per step by eliminating all spatial packing. Significant.

## Comparison to Other Approaches

**P2 (Mega-kernel fusion)**: 3-4x forward speedup by eliminating XPC round-trips. Much higher impact than P1 for forward pass. But conflicts with runtime weights (fused kernels need `const()` weights).

**P3 (_ANEChainingRequest)**: Eliminates CPU round-trips between kernel evaluations. Compatible with both spatial packing and dual-input. Potentially the best single optimization.

**Recommendation**: P3 (chaining) is likely higher impact than P1 for our architecture, because it addresses the XPC round-trip overhead (~160μs × 28 dispatches = ~4.5ms/step) without requiring MIL restructuring.
