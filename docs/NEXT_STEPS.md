# AutoANE: Next Steps & Research Roadmap

**Created**: 2026-03-12 | **Status**: Active

---

## What Has Been Done (This Session)

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
| New from upstream | 6 | UP1-UP6 — not yet tested by us |

### Implicit Assumptions Found During Audit

| # | Implicit Assumption | Where | Risk |
|---|---------------------|-------|------|
| IA1 | Training data is always available at `../tinystories_smollm2_data00.bin` | train.py, run_experiment.sh | **FIXED**: added existence check |
| IA2 | GGUF files are always F32/F16/BF16 | gguf_to_ane.py | **FIXED**: now errors on unsupported dtypes |
| IA3 | SmolLM2 tokenizer is available for generate.py | generate.py | LOW: graceful fallback exists |
| IA4 | Xcode CLT are installed | Makefile | MEDIUM: could check for `xcrun` before build |
| IA5 | IOSurface slot sizes are correctly ordered | mil_dynamic.h, io.h | **HIGH**: upstream (UP2) shows violations produce silent zeros. Needs verification. |
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

### P3: _ANEChainingRequest Integration [MEDIUM IMPACT]

**What**: Use `_ANEChainingRequest` to chain kernel evaluations without CPU round-trips.

**Why**: Even without mega-kernel fusion, chaining could eliminate the ~160μs XPC overhead between kernels. Combined with runtime weights, this gives the best of both worlds.

**Status**: ane-infer confirmed it works. The correct factory method is `objectWithstatsSurRef:outputBuffer:`. Our U12 is now "confirmed working by third party, untested by us."

**Estimated effort**: 2-3 days engineering (requires understanding ane-infer's implementation).

### P4: Multi-Seed Autosearch [MEDIUM IMPACT]

**What**: Run each autosearch evaluation N times (N=3-5) and use the median val_loss for keep/revert decisions.

**Why**: Our V27/Finding 7 showed run-to-run variance (~0.3 nats) exceeds the optimization signal. Single-evaluation search hill-climbs on noise. The "17% improvement" was seed selection.

**How**: Modify `run_autosearch.py` to run each experiment 3x, take median val_loss. This 3x the runtime (~9 hours for 100 experiments) but produces reproducible results.

**Estimated effort**: 0.5 days code, 9+ hours runtime.

### P5: Larger Dataset Test [MEDIUM IMPACT]

**What**: Train on a larger dataset (100M+ tokens) to test if Finding 1 (step count > model capacity) generalizes beyond the severely data-constrained regime.

**Why**: At 20M tokens, even our smallest model (36.4M params) is 23x below Chinchilla optimal. With more data, the crossover point where larger models win should become visible.

**Data candidates**: OpenWebText (~8B tokens), SlimPajama subset, or simply more TinyStories shards.

**Estimated effort**: 0.5 days setup, 1-2 days training.

### P6: Verify IOSurface Slot Ordering in Our Code [HIGH PRIORITY, LOW EFFORT]

**What**: Audit `mil_dynamic.h` and `io.h` to verify our IOSurface inputs are in ascending size order and outputs in descending size order.

**Why**: UP2 from imperatormk shows violations produce silent zeros with no error. If our code violates this, we could be getting wrong results in ANE modes.

**Estimated effort**: 2-4 hours audit.

### P7: Port Upstream Performance Fixes [LOW EFFORT]

**What**: Integrate applicable fixes from maderix/ANE PRs:
- PR #32: vDSP/cblas vectorization for CPU bottlenecks
- PR #39: Cache-optimized embedding ops (~12x lookup speedup)
- PR #33: ACCUM_STEPS configurable via environment variable

**Estimated effort**: 1-2 days per fix.

---

## Known Issues (Not Yet Fixed)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | No automated tests | MEDIUM | No test infrastructure exists. Would need synthetic data + expected output comparison. |
| 2 | Makefile doesn't check for Xcode CLT | LOW | Could check `which xcrun` before build. |
| 3 | Data file symlinks point to absolute paths | LOW | Local-only issue, not tracked by git (.gitignore excludes *.bin). |
| 4 | EXPERIMENTS.md line 380: "2.8W, 6.6 TFLOPS/W" cited from Orion, but our measurement shows ANE peak 1.2W | MEDIUM | Stale citation from before our power measurements. |
| 5 | U3 says "19 TFLOPS FP16" without specifying chip (this is M4-specific) | LOW | Should clarify "19 TFLOPS FP16 (M4)" |
| 6 | Security: checkpoint loading doesn't validate header fields before allocating memory | MEDIUM | See maderix/ANE PR #45 (OOB write fix). Our `load_checkpoint()` in generate.py trusts header values. |

---

## Decision Log

| Date | Decision | Rationale |
|------|---------|-----------|
| 2026-03-12 | CPU-only is the recommended default for all model sizes | E38: ANE never faster, dramatically slower at DIM=2048. V15 verified. |
| 2026-03-12 | Autosearch config (LR=6.34e-4, ACCUM=7) is not the recommended config | V27: variance exceeds signal. Baseline (LR=4e-4, ACCUM=10) is more reliable. |
| 2026-03-12 | 512d/4L is the optimal architecture for quick iteration (120-600s) | V16, V17, V21: confirmed at 120s, 300s, 600s. Gap widens with time. |
| 2026-03-12 | autoresearch.py is NOT dead code | It's the grid search orchestrator, distinct from run_autosearch.py. |
| 2026-03-12 | Runtime weight injection is the correct next research direction | UP1 + UP2 + UP5 from upstream suggest this could change the ANE throughput picture. |
