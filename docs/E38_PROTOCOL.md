# Experiment 38: ANE Scaling Study — Finding the Crossover Point

**Date**: 2026-03-11
**Status**: IN PROGRESS
**Depends on**: E37 (baseline at DIM=1024)

## Research Question

At what model dimension does ANE-matmul-only outperform CPU-only in training throughput?

## Background

E37 established that at DIM=1024 (95M params), ANE-matmul-only and CPU-only have identical throughput (~105ms/step). V11 showed ANE is 1.88x faster than CPU at 2816-width matmuls but only 0.67x at 1024-width. This suggests a crossover point exists where ANE's matmul advantage overcomes its IOSurface overhead.

### Hypothesis

As model dimension increases:
1. CPU matmul time scales as O(DIM^2) (quadratic in attention dimensions, linear in FFN since HIDDEN ~ 2.75*DIM)
2. ANE matmul time scales similarly but with ~2.5x lower constant factor (V1)
3. IOSurface overhead scales linearly with surface size
4. The ANE advantage should emerge when matmul time dominates IOSurface overhead

At DIM=2048, forward matmul FLOPS are ~4x DIM=1024. If ANE is 2.5x faster for these larger matmuls, the ~30ms saved on forward pass should exceed the ~15ms additional IO overhead.

### Memory Feasibility (computed)

| Config | Params | Total RAM (ANE) | Largest IOSurface | FLOPS/step |
|--------|--------|-----------------|-------------------|------------|
| 4L-1024d | 95M | 1.4 GB | 6.9 MB | 95 GFLOPS |
| 4L-1536d | 177M | 2.7 GB | 14.4 MB | 195 GFLOPS |
| 4L-2048d | 281M | 4.3 GB | 24.8 MB | 329 GFLOPS |

All fit within 16GB. Largest IOSurface (24.8MB) is below the ~32MB SRAM limit (U4).

## Experimental Setup

### Hardware
- MacBook Pro (Mac14,9), Apple M2 Pro, 12 cores (8P+4E), 16GB RAM
- No other compute-intensive processes running during experiments
- Lid open, charger connected
- System restarted before experiment series (clean state)

### Software
- Same training binary, recompiled with different model headers via -D flags
- ane-matmul-only mode: ANE for 7 linear projection matmuls per layer, CPU for everything else
- Random seed: srand48(42) for init and training

### Model Configurations

**4L-1024d** (baseline — E37 verified):
- DIM=1024, HIDDEN=2816, HEADS=16, KV_HEADS=4, HD=64
- 95M params, 4 layers, SEQ=256

**4L-1536d** (intermediate):
- DIM=1536, HIDDEN=4224, HEADS=24, KV_HEADS=6, HD=64
- 177M params, 4 layers, SEQ=256

**4L-2048d** (large):
- DIM=2048, HIDDEN=5632, HEADS=32, KV_HEADS=8, HD=64
- 281M params, 4 layers, SEQ=256

### Methodology
1. Reboot system, wait 5 minutes for thermal baseline
2. For each dimension (1024, 1536, 2048):
   a. Compile binary with appropriate model header
   b. Run CPU-only for 120s: `./train --cpu-only --scratch --time 120`
   c. Wait 120s cooldown
   d. Run ANE-matmul-only for 120s: `./train --ane-matmul-only --scratch --time 120`
   e. Wait 120s cooldown
3. All runs use --scratch (fresh init, no checkpoint resume)
4. Capture full stdout for analysis

### What We Capture
- Per-step: loss, ms/step, activation range, gradient range, thermal state
- Every 10 steps: detailed timing breakdown
- Every 100 steps: val_loss
- End of run: final_loss, val_loss, total_steps, training_seconds, mode

### Analysis Plan
1. **Throughput comparison**: ms/step (median, mean, P95, max) for each dimension × mode
2. **Scaling curve**: plot ms/step vs DIM for both modes, identify crossover
3. **Component breakdown**: where does the time go at each dimension?
4. **IOSurface overhead scaling**: does IO overhead grow linearly with surface size?
5. **Loss quality**: verify ANE matches CPU at matched steps (as in E37)

### Assumptions (stated explicitly)
- **A1**: HIDDEN = 2.75 * DIM is a reasonable ratio (matches Llama-style architectures)
- **A2**: 4 layers is sufficient to characterize per-layer overhead scaling (layer count doesn't change matmul shapes, only multiplies total work linearly)
- **A3**: SEQ=256 is representative (larger SEQ would change the activation-to-weight ratio in IOSurfaces)
- **A4**: IOSurface overhead scales linearly with surface byte count
- **A5**: The 120s budget is sufficient for steady-state measurement (E37 showed stable throughput within 30s)
- **A6**: Fresh init (--scratch) gives representative timing since we're measuring throughput, not training quality

### Controls
- Same binary codebase, same data, same seed, same training hyperparameters
- Only variables: DIM (determines all derived dimensions) and mode (cpu-only vs ane-matmul-only)
- Val evaluation always uses CPU fp32 forward

## Success Criteria
The experiment succeeds if it determines:
1. Whether an ANE throughput advantage exists at DIM=1536 or DIM=2048
2. The approximate crossover dimension (if it exists within our test range)
3. How ms/step scales with DIM for each mode

## Results: NO CROSSOVER — ANE GETS WORSE AT SCALE

### Summary
- **DIM=1536**: ANE and CPU are identical (~159ms/step). IOSurface overhead (5ms) is manageable.
- **DIM=2048**: ANE is **2x SLOWER** than CPU (546-649ms vs 280-288ms). IOSurface memory pressure (379MB) causes cache thrashing that degrades ALL operations.
- **No crossover point exists** in the tested range. ANE gets relatively worse, not better, at larger dimensions.

### A4 DISPROVED
IOSurface overhead does NOT scale linearly. It hits a cliff between 220MB and 379MB total surface allocations, where lock/unlock latency jumps from 5ms to 100-130ms and CPU backward pass degrades by 2.5x.

See EXPERIMENTS.md E38 for full analysis.
