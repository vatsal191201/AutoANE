# AutoANE: Next Steps & Research Roadmap

**Created**: 2026-03-12 | **Status**: Active

---

## What Has Been Done

### MeZO+LoRA Training Pipeline (Session 5, 2026-03-13 to 2026-03-16)
- **P9 COMPLETE**: MeZO zeroth-order training with LoRA-split on SmolLM2-360M (pretrained, 32 layers, DIM=960)
- **4 optimization phases** implemented and tested:
  - Phase 1: Conv1x1 hybrid — selective conv1x1 for Wq/Wo/W1/W2/W3 (2x faster), matmul for Wk/Wv. BLOBFILE weight baking. 403-429 ms/step.
  - Phase 2: Fused conv kernels — QKV combined + FFN mega-kernel + Wo conv. 96 IO round-trips (was 224). **262 ms/step = 1.71x faster than CPU.**
  - Phase 3: FZOO multi-perturbation — K=4 one-sided Rademacher. Better gradient quality, 2.5x more passes, zero wall-time benefit.
  - Phase 4: P-GAP gradient-aligned perturbations — tested flat-vector QR (degrades) and faithful per-matrix SVD (diverges with paper params, neutral with standard params). Negative result.
- **First ANE-faster-than-CPU training result** achieved via LoRA-split (no per-step weight staging) + conv-fused (fewer IO round-trips)
- **Gaussian RNG** (Box-Muller from xoshiro256+) and **per-matrix SVD** (LAPACK ssyev_) infrastructure added
- **Faithful P-GAP** implementation: 256 per-matrix SVD bases, PROJECTION constraint, Gaussian perturbations, delta linear decay
- **10 git commits**, design spec, and detailed research log documenting methodology and results
- All claims triple-checked: 50-step timing averages, val_loss within 0.03% of CPU baseline, conv1x1 bit-exact numerics

### Exhaustive Verification Sweep (Session 4, 2026-03-12)
- **6 parallel verification agents** covering: malloc/calloc safety, math ops, MIL kernels/IO, documentation vs code, bridge security, checkpoint/token handling
- **Critical bug found and fixed**: RMSNorm backward gradient had extra `w[i]` factor on correction term. Old: `dx=w[i]*rrms*(dy-x*dot)`. Correct: `dx=rrms*(w[i]*dy-x*dot)`. 7.5% relative error with non-unit weights, masked when w≈1.0.
- **Dead allocations removed (66.7MB)**: `acembed` AdamState (66MB, never used) and `gate_buf` (720KB, never used)
- **GQA dimension validation added**: checkpoint load now validates kv_heads, head_dim, q_dim (were written but never checked on read)
- **Data file validation added**: explicit checks for empty files and odd-byte-count (corrupt) files
- **4 remaining raw calloc calls** converted to safe_calloc (cpu_ops.h:35, mil_dynamic.h:521,536,552)
- **File descriptor leak fixed**: added `close(data_fd)` on mmap failure path
- **UP19 experiment**: confirmed ACCUM_STEPS throughput scaling — CPU 3.0x at accum=50, ANE 2.2x at accum=100 (lower than upstream's claimed 4.74x)
- **3 agent false positives caught**: math agent missed RMSNorm bug; MIL agent falsely claimed sdpaBwd1/2 memory leak (they ARE freed line 1583); bridge agent incorrectly said integer overflow fix was wrong (unary cast precedence)
- **Verified correct (no changes needed)**: Adam optimizer, vDSP_vdiv order, cross-entropy, gradient accumulation, loss scaling, LR schedule, LoRA, GQA, classifier GEMMs, all 14 MIL kernels, IOSurface creation, checkpoint save/load symmetry, token validation
- All 8 regression tests pass after all fixes. Committed as `db3be79`.

### Security Hardening & Bridge Audit (Session 3, 2026-03-12)
- **P10 COMPLETE**: 10 security fixes across config.h, cpu_ops.h, train.m, Makefile
- `safe_malloc()`/`safe_calloc()` wrappers abort on allocation failure — ~60+ call sites converted
- Token OOB bounds checks in `cross_entropy_loss`, `embed_lookup`, `embed_backward` (added vocab param)
- Token range validation on data file load (all tokens < VOCAB)
- `ane_init()` → `bool` return type with dlopen/NSClassFromString validation
- Checkpoint header validation: magic, version, dimensions, step/adam_t bounds
- Compiler hardening: `-fstack-protector-strong -D_FORTIFY_SOURCE=2`
- `io.h`: safe_calloc for blob/kern alloc, NULL check on model creation
- `bridge/ane_bridge.m`: Fixed malloc NULL checks, integer overflow (int→size_t) in blob builders
- Test 7 (gradient health) fixed: now runs with accum=10 steps=20 → `grad_norm=4.2867` (finite)
- All 8 regression tests pass, all 3 model configs compile, bridge compiles
- CPU vs ANE matmul-only divergence verified <0.001% over 21 steps (3 accum cycles)
- Gradient health: grad_norm 3.77→3.07 over 70 steps (monotonically decreasing, finite)

### Systematic Verification (Prior Session)
- Verified all 7 literature references (Kaplan, Chinchilla, Eldan & Li, Wang/DeepNet, Smith, Muennighoff, Nguyen & Salazar) — all accurate
- Verified round-trip ANE→GGUF→ANE is bit-perfect (38 tensors, 36.4M params)
- Verified 1024d/4L timing claims (CPU 97.2ms, ANE full 67.0ms, ANE matmul-only 77.6ms) — all within expected variance
- Discovered autosearch variance problem: "best" config (LR=6.34e-4, ACCUM=7) gives val_loss ~3.8 typical, not 3.288. Baseline config (LR=4e-4, ACCUM=10) reliably gives ~3.5
- Fixed false "first forward/backward pass" claim → credited maderix as pioneer
- Fixed M4 → M2 Pro hardware reference
- Verified text generation, GGUF export, LoRA mode, demo.sh all work

### Upstream Research (This Session)
- Reviewed ALL PRs and issues on maderix/ANE (48 PRs, 12 issues)
- Reviewed ALL PRs and issues on karpathy/autoresearch (200+ PRs, 200+ issues)
- Discovered 3 game-changing upstream developments:
  1. **imperatormk/ane-train**: Runtime weight injection via IOSurface inputs — compile once, train forever
  2. **maderix/ANE PR #24**: Mega-kernel fusion achieves 3-4x forward speedup
  3. **thebasedcapital/ane-infer**: `_ANEChainingRequest` actually works (error was wrong factory method)
- Verified ANE TFLOPS claims against Apple specs (15.8 TOPS M2, 38 TOPS M4)
- Verified Chinchilla 20:1 tokens-to-parameters ratio from Hoffmann et al.
- Web-searched ANE training literature — no other published training comparisons exist

### Code Fixes (This Session)
1. **train.py**: Added data file existence check before compilation (was opaque runtime crash)
2. **gguf_to_ane.py**: Changed silent dtype skip to hard error with clear message
3. **generate.py**: Added validation for temperature and top_k parameters
4. **autoresearch.py**: Confirmed NOT dead code — it's the grid search orchestrator (different from run_autosearch.py)

### Documentation Updates (This Session)
- Fixed TECHNICAL_REPORT.md: "15.8 TFLOPS (FP16) on M4" → "15.8 TOPS on M2, 38 TOPS on M4 (INT8; fp16 ~half)"
- Added 6 new upstream assumptions (UP1-UP6) to ASSUMPTIONS.md
- Updated SA4 with nuance from UP1 (runtime weight injection)
- Updated U12 with UP5 (_ANEChainingRequest confirmed working)
- Added "Related Work (Post-Publication)" section to README
- Added upstream credits (imperatormk, thebasedcapital) to README
- Updated Open Questions with specific upstream references

---

## Stated Assumptions (Complete Registry)

All assumptions are tracked in [ASSUMPTIONS.md](ASSUMPTIONS.md). Summary:

| Category | Count | Key Items |
|----------|-------|-----------|
| Verified | 27 | V1-V27 — all confirmed by experiment |
| Qualified | 1 | V27: autosearch 3.288 was best-of-88, not reproducible |
| Disproved | 8 | D1-D8 — tested and found wrong |
| Unverified/Resolved | 13 | U1-U17 — most resolved or confirmed via literature |
| New from upstream | 23 | UP1-UP23 — most not yet tested; UP19 experimentally confirmed at lower magnitude |

### Implicit Assumptions Found During Audit

| # | Implicit Assumption | Where | Risk |
|---|---------------------|-------|------|
| IA1 | Training data is always available at `../tinystories_smollm2_data00.bin` | train.py, run_experiment.sh | **FIXED**: added existence check |
| IA2 | GGUF files are always F32/F16/BF16 | gguf_to_ane.py | **FIXED**: now errors on unsupported dtypes |
| IA3 | SmolLM2 tokenizer is available for generate.py | generate.py | LOW: graceful fallback exists |
| IA4 | Xcode CLT are installed | Makefile | **FIXED**: Makefile now checks `command -v xcrun` before build |
| IA5 | IOSurface slot sizes are correctly ordered | mil_dynamic.h, io.h | **VERIFIED SAFE (P6 audit, 2026-03-12)**: All 14 kernels use exactly 1 input + 1 output IOSurface. Single-surface spatial packing provides structural immunity. Only `bridge/ane_bridge.m` supports multi-slot (no ordering check). |
| IA6 | Matmul inner dimensions are multiples of 32 | all ANE kernel code | MEDIUM: our DIM=512 is fine, but other configs could silently fail |
| IA7 | Single-process training (no concurrent ANE users) | train.m | CONFIRMED: E18/E37 showed concurrent processes cause false divergence |
| IA8 | macOS 15+ required | README, train.m | LOW: documented requirement |
| IA9 | All weight tensors fit in available unified memory | checkpoint loading | LOW: validate_config() warns above 64GB |
| IA10 | Adam epsilon 1e-8 is sufficient for fp32 CPU training | train.m | LOW: standard value. Note: imperatormk uses eps>=1e-4 for fp16 Adam on ANE |

---

## Research Priorities (Ranked)

### P1: Test Runtime Weight Injection for Transformers [HIGH IMPACT]

**What**: Adapt imperatormk's approach (IOSurface matmul inputs instead of const weights) to our transformer architecture. Compile all ANE kernels once at startup. Pass weights via IOSurface at each step.

**Why**: Eliminates the IOSurface fp32→fp16 conversion overhead that costs us 8.1ms/step. If weights are already fp16 IOSurface inputs, the conversion is unnecessary. This could flip our Finding 2 (CPU beats ANE).

**How**:
1. Study imperatormk's `mil_gen.h` — uses `matmul(W[1,Cout,Cin], X[1,Cin,S])` with both as inputs
2. Verify IOSurface slot ordering in our code (UP2: ascending for inputs, descending for outputs)
3. Verify our dimensions are multiples of 32 (UP3)
4. Implement weight ping-pong buffers (zero-copy weight updates)
5. Benchmark against current dynamic pipeline

**Key constraint**: imperatormk found weights MUST be slot0, activations slot1. `Cout ≤ S` usually required.

**Estimated effort**: 2-3 days engineering, 1 day benchmarking.

### P2: Mega-Kernel Fusion Experiment [HIGH IMPACT]

**What**: Fuse N transformer layers into a single MIL program (as demonstrated in maderix/ANE PR #24).

**Why**: 3-4x forward speedup by eliminating XPC round-trips (~160μs/eval). Our current 28 dispatches/step would become ~4 dispatches.

**Conflict with P1**: Mega-kernels require weights as `const()` (baked), which means recompilation on weight updates. This conflicts with runtime weight injection. The trade-off:
- Runtime weights: no recompile, but 28 XPC round-trips/step
- Mega-kernels: 3-4x faster, but recompile every N steps

**Resolution**: Use ACCUM_STEPS to amortize compile cost. At ACCUM=100, compile overhead drops to ~47% (PR #24 data). Or use partial fusion (4-layer chunks) for 7.7x speedup with faster compile.

**Estimated effort**: 3-5 days engineering.

### P3: _ANEChainingRequest Integration — **INVALIDATED**

**Status (2026-03-12)**: maderix/ANE PR #40 definitively shows `_ANEChainingRequest` requires Espresso IR from disk-compiled `_ANEModel`. Our in-memory MIL path (`_ANEInMemoryModelDescriptor`) cannot produce the required format. **Dead on macOS 15+.**

This was previously identified as "likely highest ROI" but is now confirmed impossible with our architecture. The ~4.5ms/step XPC overhead (28 dispatches × ~160μs) remains addressable only via mega-kernel fusion (P2) or complete CPU fallback.

### P8: Delta Compilation via Unload/Reload [MEDIUM IMPACT — NEW]

**What**: Implement Orion paper's delta compilation approach: `unloadWithQoS(21)` → patch weight BLOBFILE on disk → `loadWithQoS(21)`. Bypasses `ANECCompile()` entirely.

**Why**: 8.5× speedup over full recompilation (4200ms → 494ms). Eliminates the ~119 compilation limit per process. Enables const() weights in MIL (better ANE utilization) while still updating weights at each accumulation step.

**Trade-off vs our approach**: Our runtime weight injection compiles once and never recompiles (0ms overhead), but uses spatial packing with slice_by_size ops and single-input IOSurfaces. Delta compilation allows const() weights (cleaner MIL, potentially higher ANE utilization) but adds 494ms per weight update. At ACCUM=7, that's 70ms/step amortized — vs our 24ms/step total. **Not competitive for our small model (36.4M params)**, but could be valuable for larger models where ANE compute dominates.

**Estimated effort**: 1-2 days. Requires understanding Orion's temp directory management.

### P9: Zeroth-Order Training — **IMPLEMENTED**

**What**: MeZO (Memory-Efficient Zeroth-Order Optimizer) + LoRA-split for ANE fine-tuning of SmolLM2-360M. Forward-only gradient estimation via SPSA: `∇f(w) ≈ [f(w+ε·z) - f(w-ε·z)] / (2ε) · z`.

**Implementation (2026-03-13 to 2026-03-16)**:
- `train_mezo.m`: Complete MeZO training loop with LoRA-split mode
- LoRA rank-8, attention-only adapters (Aq/Bq/Ak/Bk/Av/Bv/Ao/Bo × 32 layers)
- Trainable params: 1,700,800 (1,638,400 LoRA + 62,400 RMS norms)
- Base weights frozen as BLOBFILE constants (no per-step staging)
- 4 optimization phases tested:

| Phase | Speed | Convergence | Status |
|-------|-------|-------------|--------|
| 1: Conv1x1 hybrid | 403-429 ms/step | 1.0x | ✅ Done |
| 2: Fused conv kernels | ~262 ms/step (1.71x CPU) | 1.0x | ✅ Done |
| 3: FZOO multi-perturbation | 2.5x slower/step | no wall-time benefit | ✅ Done |
| 4: P-GAP (simplified + faithful) | same speed | degrades or neutral | ❌ Negative |

**Key result**: Phase 2 (conv-fused) achieves **262ms/step = 1.71x faster than CPU** (447ms baseline). This is the **first demonstration of ANE training faster than CPU** on Apple Silicon. Triple-checked via 50-step average with val_loss within 0.03% of CPU baseline.

**What changed vs previous findings**: LoRA-split mode freezes base weights as BLOBFILE constants, eliminating per-step IOSurface weight staging that previously made ANE slower than CPU. Conv1x1 with fused kernels reduces IO round-trips from 224 to 96 per forward pass.

**Negative results (documented)**:
- FZOO K=4: Better gradient quality but 2.5x more forward passes → zero wall-time benefit
- P-GAP: Tested both flat-vector QR and faithful per-matrix SVD implementations. Paper hyperparams (ε=0.1, lr=1e-2) diverge catastrophically on SmolLM2-360M. Standard hyperparams neutral. Root cause: LoRA rank-8 matrices too small for per-matrix SVD to find useful low-rank structure.

**References**: MeZO (arXiv:2305.17333), FZOO (arXiv:2506.09034), P-GAP (arXiv:2510.18228), ZOSA (arXiv:2511.09156), Orion (arXiv:2603.06728)

### P10: Security Hardening — **IMPLEMENTED**

**What**: Port critical security fixes from maderix/ANE PRs #7, #13, #8, #5, #45.

**Implemented (2026-03-12)**:
1. `config.h`: `ane_init()` → `bool` return type, validates `dlopen()` and `NSClassFromString()` results (PR #7)
2. `config.h`: `safe_malloc()`/`safe_calloc()` wrappers that abort on failure (PR #13)
3. `config.h`: All allocation functions (`layer_weights_alloc`, `layer_acts_alloc`, `layer_grads_alloc`, `adam_alloc`, `lora_layer_alloc`) converted to safe variants
4. `cpu_ops.h`: All malloc/calloc → safe_malloc/safe_calloc (rmsnorm, Adam, SDPA, VocabMap)
5. `cpu_ops.h`: Token OOB bounds checks in `cross_entropy_loss` (targets), `embed_lookup` (tokens), `embed_backward` (tokens) — added vocab parameter (PR #13)
6. `train.m`: `ane_init()` call site checks return value, exits gracefully with actionable message
7. `train.m`: Checkpoint header validation: magic number, version, dimension matching against compiled model, step/adam_t bounds (PR #45)
8. `train.m`: All 40+ malloc/calloc calls → safe variants (including dispatch_async captured buffers)
9. `train.m`: Token range validation on data file load — verifies all tokens < VOCAB before training
10. `Makefile`: Added `-fstack-protector-strong -D_FORTIFY_SOURCE=2` compiler flags (PR #5)

**Verification**: All 7 regression tests pass. All 3 model configs (stories110m, smollm2_360m, autoresearch) compile cleanly.

### P4: Multi-Seed Autosearch [MEDIUM IMPACT] — **IMPLEMENTED**

**What**: Run each autosearch evaluation N times (N=3-5) and use the median val_loss for keep/revert decisions.

**Why**: Our V27/Finding 7 showed run-to-run variance (~0.3 nats) exceeds the optimization signal. Single-evaluation search hill-climbs on noise. The "17% improvement" was seed selection.

**Implementation** (2026-03-12):
- Added `--seed` CLI flag to `train.m` (was hardcoded `srand48(42)`)
- Added `TRAIN_SEED` environment variable support to `train.py`
- Added `--n-seeds` flag to `run_autosearch.py` (default 1 for backward compat)
- Multi-seed runs N experiments with seeds [42, 1042, 2042, ...], takes median val_loss
- Individual run losses are logged to results.tsv for post-hoc variance analysis
- Usage: `python3 run_autosearch.py --experiments 50 --n-seeds 3`

### P5: Larger Dataset Test [MEDIUM IMPACT]

**What**: Train on a larger dataset (100M+ tokens) to test if Finding 1 (step count > model capacity) generalizes beyond the severely data-constrained regime.

**Why**: At 20M tokens, even our smallest model (36.4M params) is 23x below Chinchilla optimal. With more data, the crossover point where larger models win should become visible.

**Data candidates**: OpenWebText (~8B tokens), SlimPajama subset, or simply more TinyStories shards.

**Estimated effort**: 0.5 days setup, 1-2 days training.

### P6: Verify IOSurface Slot Ordering in Our Code — **VERIFIED SAFE**

**Result (2026-03-12)**: Full audit of all 14 ANE kernels in train.m. Every kernel uses exactly 1 input IOSurface and 1 output IOSurface. The single-surface spatial packing architecture (activations + weights packed in spatial dimension, separated by `slice_by_size` inside MIL) provides structural immunity — a 1-element array is trivially sorted. The `Kern` struct (`config.h:63`) has scalar `ioIn`/`ioOut` fields, making multi-slot structurally impossible.

**Only risk surface**: `bridge/ane_bridge.m` supports arbitrary multi-slot configurations with no ordering enforcement. Any Python caller passing non-ascending input sizes or non-descending output sizes will get silent zeros.

### P7: Port Upstream Performance Fixes — **PARTIALLY IMPLEMENTED**

**What**: Integrate applicable fixes from maderix/ANE PRs:
- PR #32: vDSP/cblas vectorization for CPU bottlenecks
- PR #39: Cache-optimized embedding ops (~12x lookup speedup)
- PR #33: ACCUM_STEPS configurable via environment variable

**Implemented (2026-03-12)**:
1. Gradient scaling vectorization (`train.m:1291-1303`): replaced 9 scalar `for` loops per layer with `vDSP_vsmul` — matches the gradient clipping code 70 lines later. Zero risk.
2. `transpose_weight` vectorization (`train.m:64-68`): replaced naive nested loop with `vDSP_mtrans`. Called for 7 matrices × 32 layers at startup and every Adam step.
3. Adam optimizer vectorization (`cpu_ops.h:56-102`): replaced scalar loop with vDSP/vForce pipeline (vDSP_vsmul, vDSP_vsma, vDSP_vmul, vvsqrtf, vDSP_vdiv). Largest buffer is embed at 25.2M floats; called 38+ times per accum step. Lazy-allocated temp buffers (same pattern as g_rms_tmp).
4. `gqa_reduce_kv` vectorization (`io.h:359-370`): replaced scalar inner loop with `vDSP_vadd`. Called 2× per backward step per layer.
5. `vocab_scatter_grads` vectorization (`cpu_ops.h:133-139`): replaced scalar inner loop with `cblas_saxpy`. Called once per accum step over ~16893 tokens.
6. `embed_lookup` vectorization (`cpu_ops.h:147-153`): replaced scalar scatter-write with `cblas_scopy` (strided). Called once per forward step.
7. `embed_backward` vectorization (`cpu_ops.h:155-162`): replaced scalar gather-accumulate with `cblas_saxpy` (strided). Called once per backward step.

**Remaining** (from P7 assessment):
- PR #39 complete compact embed: `embed_lookup` at `train.m:656` still uses full 49152-token table. Switching to compact (16893 tokens) would reduce Adam embed state from 25.2M to 8.6M elements and eliminate `vocab_scatter_grads`. **Deferred**: Requires changes to 8+ locations (embed lookup, backward, gradient norm, clipping, sanitization, Adam, checkpoint save/load) with checkpoint format compatibility concerns. The vectorized Adam already handles 25.2M elements efficiently.
- PR #33 env-var: operational convenience only, 15 min. Already have `--accum` CLI and `train_config.h` defines.

---

## Known Issues (Not Yet Fixed)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | No automated tests | MEDIUM | **FIXED**: Added `tests/test_training.sh` — 8 regression tests covering compilation, forward/backward pass, Adam update, seed sensitivity, checkpoint I/O, ANE mode, gradient health, and Python wrapper. Run: `cd training && bash ../tests/test_training.sh` |
| 2 | Makefile doesn't check for Xcode CLT | LOW | **FIXED**: Added `command -v xcrun` check before build. |
| 3 | Data file symlinks point to absolute paths | LOW | Local-only issue, not tracked by git (.gitignore excludes *.bin). |
| 4 | EXPERIMENTS.md line 380: "2.8W, 6.6 TFLOPS/W" cited from Orion, but our measurement shows ANE peak 1.2W | MEDIUM | **FIXED**: Struck through stale values, added correction note with actual measured data, replaced estimated table with measured powermetrics data. |
| 5 | U3 says "19 TFLOPS FP16" without specifying chip (this is M4-specific) | LOW | **FIXED**: Clarified "19 TFLOPS FP16 on M4 (lower on earlier chips: ~15.8 TOPS INT8 on M2)" |
| 6 | Security: checkpoint loading doesn't validate header fields before allocating memory | MEDIUM | **FIXED**: Added `validate_checkpoint_config()` to generate.py and export_to_gguf.py. Validates all header fields have sane ranges and file size is consistent with claimed dimensions. |

---

## Decision Log

| Date | Decision | Rationale |
|------|---------|-----------|
| 2026-03-12 | CPU-only is the recommended default for all model sizes | E38: ANE never faster, dramatically slower at DIM=2048. V15 verified. |
| 2026-03-12 | Autosearch config (LR=6.34e-4, ACCUM=7) is not the recommended config | V27: variance exceeds signal. Baseline (LR=4e-4, ACCUM=10) is more reliable. |
| 2026-03-12 | 512d/4L is the optimal architecture for quick iteration (120-600s) | V16, V17, V21: confirmed at 120s, 300s, 600s. Gap widens with time. |
| 2026-03-12 | autoresearch.py is NOT dead code | It's the grid search orchestrator, distinct from run_autosearch.py. |
| 2026-03-12 | Runtime weight injection is the correct next research direction | UP1 + UP2 + UP5 from upstream suggest this could change the ANE throughput picture. |
| 2026-03-12 | P3 (_ANEChainingRequest) is DEAD — do not pursue | maderix/ANE PR #40: requires Espresso IR from disk-compiled models, incompatible with in-memory MIL. Dead on macOS 15+. |
| 2026-03-12 | Orion paper (arxiv:2603.06728) is the definitive ANE training reference | 20 documented constraints, delta compilation (8.5x), 3 NaN bugs fixed, Stories110M training in 22.4 min. Must cross-reference all our findings. |
| 2026-03-12 | concat works on ios18 MIL target — contradicts Orion's constraint | Tested: 10 concat ops compile and execute correctly in our sdpaFwd, ffnFused, sdpaBwd1/2. Orion likely used ios16 target. |
| 2026-03-12 | Zeroth-order training (MeZO) is a promising alternative to backprop for ANE | Forward-only training eliminates backward kernels, plays to ANE's strength. ElasticZO-INT8 could leverage ANE's 1.88x INT8 throughput. New P9 research priority. |
| 2026-03-16 | MeZO+LoRA-split+conv-fused achieves 1.71x CPU speedup on ANE | First ANE-faster-than-CPU training result. LoRA-split freezes base weights as BLOBFILE → no per-step staging. Conv-fused reduces IO round-trips 224→96. 262ms/step vs 447ms CPU. |
| 2026-03-16 | P-GAP does not transfer to LoRA ZO training | Tested both simplified (flat-vector QR) and faithful (per-matrix SVD, PROJECTION constraint) implementations. Paper hyperparams diverge; standard hyperparams neutral. Root cause: LoRA rank-8 matrices too small for per-matrix SVD. |
| 2026-03-16 | FZOO multi-perturbation provides no wall-time benefit | K=4 one-sided perturbations give better gradient quality but cost 2.5x more forward passes, netting zero wall-time convergence improvement. |
| 2026-03-16 | Conv1x1 BLOBFILE requires LoRA-split (frozen base weights) | Hard constraint: BLOBFILE constants cannot be updated at runtime. Only viable with LoRA-split where base weights never change. |
| 2026-03-12 | M5 GPU Neural Accelerators (~70 TFLOPS) make ANE training more niche | Apple investing in GPU ML acceleration via MLX, not ANE training. ANE advantage: zero idle power, dedicated silicon. |
| 2026-03-12 | @bearmug's autoresearch PRs (#103, #149, #196) achieve val_loss 3.102 | Most technically ambitious Apple Silicon contribution to karpathy/autoresearch. Uses same dynamic weight pipeline as ours. None merged. |
| 2026-03-12 | All vectorization changes verified correct by first-principles analysis | vDSP_vdiv parameter order confirmed from SDK headers (B,A swapped). Adam matches mathematical definition exactly. CPU vs ANE numerical agreement <0.001% over 70 steps. |
| 2026-03-12 | P10 security hardening complete — 10 fixes across 3 files | safe_malloc/calloc wrappers, token OOB bounds checks, checkpoint validation, ane_init() validation, compiler hardening flags. All 7 tests pass, all 3 model configs compile. |
| 2026-03-12 | RMSNorm backward gradient bug fixed (critical) | Extra `w[i]` on correction term: `dx=w[i]*rrms*(dy-x*dot)` → `dx=rrms*(w[i]*dy-x*dot)`. 7.5% error with non-unit weights. Masked when w≈1.0 (e.g., early training). |
| 2026-03-12 | 66.7MB dead allocations removed | `acembed` AdamState (66MB) and `gate_buf` (720KB) allocated but never used. |
| 2026-03-12 | UP19 experimentally confirmed at lower magnitude | ACCUM_STEPS throughput: CPU 3.0x at accum=50, ANE 2.2x at accum=100. Upstream claims 4.74x — likely model-size/architecture dependent. |
| 2026-03-12 | Verification agents require cross-checking | 3 false positives caught from 6 parallel agents: missed RMSNorm bug, false memory leak claim, wrong integer overflow analysis. Agent outputs must be manually verified. |
| 2026-03-16 | Sparse MeZO and HiZOO both negative for LoRA ZO (Phase 5) | Per-parameter scaling, magnitude-based masking, amplitude-reducing methods all counterproductive for LoRA ZO's 1.7M-param low-rank subspace. |
| 2026-03-16 | Apple MeBP paper does NOT invalidate our MeZO approach | MeBP runs CPU/GPU only (no ANE). "10-100x convergence" claim is for full-param ZO, not LoRA ZO. See docs/2026-03-16-mebp-cross-reference.md. |
| 2026-03-16 | MeBP+ANE hybrid is the most promising unexplored direction (P16) | ANE forward (fast) + CPU backward (checkpointed) combines convergence quality + hardware speed. |
| 2026-03-16 | CoreML stateful models hang at dim>=384 during ct.convert() | Tested in AutoANE/tests/. StateTensorSpec + matmul works at dim<=256 but serialization hangs for larger dims. Not viable for production-scale models via CoreML. |
| 2026-03-16 | NPU training landscape: only Apple ANE and Huawei Ascend have training | All other NPUs (Qualcomm Hexagon, Google Edge TPU, Samsung Exynos, MediaTek APU) are inference-only. Huawei Ascend is the only official vendor-supported NPU training. |

---

## New Research Priorities (2026-03-16)

### P11: Cross-Layer Fusion [HIGH IMPACT]

**What**: Fuse across multiple transformer layers into fewer ANE dispatches. Currently at 96 IO round-trips (3 kernels/layer x 32 layers). Target: 12-16 dispatches (8-layer chunks).

**Why**: Each ANE dispatch has ~160us XPC overhead. 96 dispatches = 15ms overhead. Fusing to 12 dispatches saves ~13ms/step (5% of 262ms).

**Constraint**: LoRA corrections between layers prevent full fusion. Must apply LoRA A/B corrections between fused chunks.

**Estimated effort**: 3-5 days.

### P12: Larger Model Scaling [HIGH IMPACT]

**What**: Test MeZO+LoRA-split+conv-fused on models larger than SmolLM2-360M (e.g., 1B, 3B).

**Why**: Conv1x1 advantage increases with wider dimensions (conv is 2x+ faster than matmul at DIM>=960). Larger models should see even greater ANE speedup.

**Risk**: Memory. 360M requires ~1.5GB. 1B would need ~4GB. 3B would need ~12GB (exceeds M2 Pro 16GB).

**Estimated effort**: 2-3 days.

### P13: INT8 Quantized LoRA [MEDIUM IMPACT]

**What**: Quantize LoRA A/B matrices to INT8 for ANE compute, leveraging ANE's 1.88x INT8 throughput (35 TOPS vs 18.6 TFLOPS FP16).

**Why**: MeZO perturbation noise dominates LoRA precision. INT8 may be sufficient for the perturbation direction.

**Risk**: Rank-8 matrices are very small (8 columns). Quantization may lose critical information.

**Estimated effort**: 2-3 days.

### P14: Mobile Deployment (iOS/iPad) [MEDIUM IMPACT]

**What**: Port MeZO+LoRA-split to iOS for on-device training on iPhone/iPad.

**Why**: ANE may be the only thermally viable compute unit for sustained mobile training. CPU throttles faster.

**Estimated effort**: 1 week (Swift port + iOS app scaffold).

### P15: Variance Reduction ZO [RESEARCH]

**What**: Control variates, antithetic sampling, or other variance reduction techniques that don't reduce perturbation amplitude.

**Why**: Phase 5 ruled out amplitude-reducing methods (Sparse, HiZOO). Variance reduction is the only remaining direction for improving ZO gradient quality.

**Estimated effort**: Research-stage, 1-2 weeks.

### P16: MeBP+ANE Forward Hybrid [HIGH IMPACT — NEW]

**What**: Implement gradient-checkpointed backprop where forward passes (including checkpoint recomputation) run on ANE, and backward matmuls run on CPU via Accelerate BLAS.

**Why**: Combines MeBP's convergence advantage (first-order gradients, 10x fewer steps) with ANE's forward speed (262ms for 32-layer forward). MeBP's lazy weight decompression overhead (32-42%) doesn't apply on ANE (weights already fp16).

**Algorithm**:
1. Forward pass on ANE (conv-fused kernels, 262ms)
2. For each layer backward (on CPU):
   a. Recompute forward activations via ANE (per-layer, ~8ms)
   b. Compute dx, dW via CPU Accelerate BLAS
3. LoRA weight update via Adam on CPU

**Estimated wall-clock**: ~262ms forward + 32 * (8ms ANE recompute + 10ms CPU backward) = ~838ms/step. If MeBP needs 60 steps (10x fewer than MeZO's 600), total = 50s vs 157s for MeZO.

**Risk**: Checkpoint recomputation on ANE requires per-layer model compilation (one-time cost). CPU backward may bottleneck if Accelerate BLAS is slower than expected for backward matmuls.

**Estimated effort**: 3-5 days engineering.

### Updated Priority Ranking (2026-03-16, evening)

| Rank | ID | Direction | Expected Impact | Status |
|------|-----|-----------|----------------|--------|
| 1 | **P16** | **MeBP+ANE hybrid** | **Convergence + speed** | IN PROGRESS (scaffold done) |
| 2 | **P17** | **Autoresearch integration for P16** | Agent loop with backprop quality | After P16 |
| 3 | P11 | Cross-layer fusion | Further reduce IO overhead | Open |
| 4 | P12 | Larger model scaling | Test speedup scaling | Open |
| 5 | P13 | INT8 quantized LoRA | Leverage ANE INT8 | Open |
| 6 | P15 | Variance reduction ZO | Better ZO gradients | Deprioritized (Finding 9: quality ceiling) |
| 7 | P5 | Larger dataset | Generalization test | Open |
| 8 | P14 | Mobile deployment | iOS/iPad training | Open |

### P17: Autoresearch Integration for P16 [HIGH — After P16]

**What**: Wire P16 hybrid mode into the autoresearch agent loop so the AI agent
can autonomously optimize hyperparameters with backprop-quality training on ANE.

**Current state**: The autoresearch infrastructure already exists:
- `training/train.py` — agent-editable config, compiles binary, runs, parses output
- `training/run_autosearch.py` — autonomous search loop (random perturbations, keep/revert)
- `training/autoresearch.py` — grid search orchestrator
- `training/program.md` — agent protocol (edit → run → evaluate → keep/discard)

**Changes needed** (after P16 is validated):
1. Add `MODE = "backprop-lora"` option to `train.py` (passes `--backprop-lora --conv-fused --lora-split`)
2. Update `train.py` to compile `train_mezo` instead of `train` when MODE = "backprop-lora"
3. Add LoRA-specific hyperparams to search space in `run_autosearch.py` (lr, rank, warmup)
4. Update `program.md` with P16 mode instructions
5. Add val_loss parsing from P16 output format

**Estimated effort**: 1 day (train.py changes are ~30 lines).

**Why after P16**: Need validated P16 performance first. The agent loop will only
be effective if the per-step quality is high (backprop-quality gradients) AND
the per-step speed is competitive (ANE forward advantage).

**Integration sequence**:
```
P16 implementation → P16 validation → P17 train.py integration → P17 run_autosearch → agent loop runs
```
