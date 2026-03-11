# Experiment 37: ANE Sustained Throughput Characterization

**Date**: 2026-03-11
**Status**: IN PROGRESS

## Research Question

Does ANE provide a net training speedup over CPU-only when run long enough for thermal behavior to stabilize?

## Background

E36 (120s) showed ane-matmul-only gets 130 steps vs CPU-only's 550 — a 4.2x throughput disadvantage. We observed periodic stalls of 6-8 seconds where ALL system operations freeze (CPU and ANE alike).

### Literature Review Findings (pre-experiment)

Literature review revealed several critical facts that reframe the hypothesis:

1. **ANE is unlikely the thermal bottleneck**: ANE peak power is ~2.8W (vs CPU at 45W+). It has hard power gating (0mW idle). The Orion paper ran 1,000 steps over 22.4 min with 3.3% CV — no thermal degradation.

2. **Single-op ANE utilization is ~30%**: The ANE is a graph execution engine. Single matmul kernels (our unfused approach) get ~30% utilization vs 74-94% for 16-64 op fused graphs. This is a fundamental architectural limitation.

3. **Dispatch overhead is ~0.095ms per call**: Our 7 matmuls per layer x 4 layers = 28 dispatches per step, adding ~2.7ms of fixed overhead.

4. **The original repo uses ANE for BOTH forward and backward dx**: We use ANE only for forward matmuls. The original's generation 3 runs backward dx projections on ANE too (ffnBwdW2t, ffnBwdW13t, wotBwd, qBwd, kvBwd).

### Revised Hypothesis

The stalls are likely caused by CPU thermal throttling (backward pass is CPU-heavy), macOS scheduling, or combined SoC thermal management — NOT ANE-specific throttling. The 4.2x throughput gap may be from:
- 30% ANE utilization on single-op kernels (vs near-100% for CPU AMX on large matmuls)
- IOSurface staging overhead (fp32-to-fp16 conversion + surface locking)
- Dispatch overhead (28 calls/step)
- NOT thermal throttling

## Experimental Setup

### Hardware
- MacBook Pro (Mac14,9), Apple M2 Pro, 12 cores (8P+4E), 16GB RAM
- No other compute-intensive processes running during experiments
- Lid open, charger connected

### Software
- AutoANE training binary compiled from commit after E36 with thermal monitoring added
- Model: 4L/1024d (95.4M params), SEQ=256, ACCUM=10
- Data: tinystories_smollm2_data00.bin (38MB, ~19.7M tokens)
- 90/10 train/val split
- Random seed: srand48(42) for init, srand48(42) for training loop

### Methodology
1. Run **cpu-only** for 600s (10 minutes), capture full stdout
2. Wait **120s** cooldown between runs (let thermals settle)
3. Run **ane-matmul-only** for 600s (10 minutes), capture full stdout
4. Sequential runs to avoid thermal interference (lesson from E35)

### What We Capture
- Per-step: loss, ms/step, activation range, gradient range, thermal state
- Every 10 steps: detailed timing breakdown (ane_fwd, io_fwd, rms, ane_bwd, etc.)
- Every 100 steps: val_loss (CPU fp32 forward on val split)
- End of run: final_loss, val_loss, total_steps, training_seconds, mode

### Analysis Plan
From the captured data, compute:
1. **Throughput over time**: steps/minute in 1-minute windows (6 windows per run)
2. **Step time distribution**: median, P95, P99, max for each 1-minute window
3. **Thermal state transitions**: when does thermal state change from nominal→fair→serious?
4. **Loss at matched steps**: compare loss values at identical step counts
5. **Loss at matched wall-clock**: compare loss values at same elapsed time
6. **Steady-state throughput**: average ms/step over last 5 minutes (excluding first 5 min warmup)

### Assumptions (stated explicitly)
- **A1**: CPU-only mode does not use ANE at all (verified: cblas_sgemm is CPU AMX)
- **A2**: ane-matmul-only uses ANE only for 7 linear projection matmuls per layer (verified in code)
- **A3**: The `ProcessInfo.thermalState` API reflects actual thermal conditions (Apple documentation)
- **A4**: The training data is identical for both runs (same mmap'd file, same seed)
- **A5**: The thermal state at start of each run is "nominal" (enforced by 120s cooldown)
- **A6**: No background processes interfere (best-effort — cannot fully control macOS scheduling)

### Controls
- Same binary, same data, same hyperparameters, same random seed
- Only variable: `--cpu-only` vs `--ane-matmul-only`
- Val evaluation always uses CPU fp32 forward (verified in code)

## Success Criteria
The experiment succeeds if it answers: "At what steady-state throughput (ms/step) does ane-matmul-only settle, and how does it compare to cpu-only?"

## Results: OUTCOME EXCEEDS ALL EXPECTATIONS

ANE-matmul-only achieved 4,690 steps vs CPU-only's 1,262 — a 3.7x throughput advantage.
ANE steady-state: 103ms/step (median), max 165ms. CPU: 163ms median, 16.3s max.
Final val loss: 3.148 (ANE) vs 3.910 (CPU). Both thermal=nominal throughout.

See EXPERIMENTS.md E37 for full analysis.
