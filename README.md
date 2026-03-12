# AutoANE

**Training transformers on Apple's Neural Engine: an empirical study**

AutoANE is an open-source system for training Llama-family transformers on Apple Silicon, building on [maderix/ANE](https://github.com/maderix/ANE)'s pioneering reverse engineering of the Neural Engine's private APIs. It adds systematic ANE vs CPU benchmarking, power measurement, and an autonomous hyperparameter search loop adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The training loop is a compiled Objective-C binary (~1580 lines).

44 controlled experiments, a 100-experiment autonomous search, and first-principles verification produced three central findings:

1. **ANE has a genuine 2.5x matmul speedup over CPU** (Apple AMX/Accelerate), but IOSurface weight-staging overhead negates this advantage end-to-end. CPU-only training wins at every tested model size (36M-281M params).

2. **ANE does not save power.** Direct measurement via `powermetrics` shows package power of 13.3W (CPU-only), 12.6W (ANE matmul), and 12.7W (ANE full). This is the first published power measurement of ANE training workloads.

3. **In a fixed time window, more optimizer steps beats more parameters.** 512d/4L (36M params, ~4300 steps at 120s) achieves val_loss ~3.5 while 1024d/4L (95M params, ~1050 steps) achieves ~4.3. This confirms Kaplan et al.'s observation that the data scaling exponent (0.095) exceeds the parameter exponent (0.076) in the severely data-constrained regime.

**[Complete Findings & Methodology](docs/FINDINGS.md)** | [Experiment Log (E1-E44)](docs/EXPERIMENTS.md) | [Verification Report](docs/VERIFICATION.md) | [Assumptions Registry (71 tracked)](docs/ASSUMPTIONS.md) | [Technical Report](docs/TECHNICAL_REPORT.md) | [Research Roadmap](docs/NEXT_STEPS.md)

---

## Quick Start

**Requirements**: macOS 15+, Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools.

```bash
git clone https://github.com/vatsal191201/AutoANE.git && cd AutoANE

# Download training data (TinyStories, 40MB, 20M tokens, SmolLM2 BPE tokenizer)
bash tools/download_data.sh

# Train from scratch (120s, CPU-only, 36.4M params)
cd training && python3 train.py

# Expected output:
#   val_loss:  ~3.8     (run-to-run variance ~0.3 nats from random init)
#   num_steps: ~4100-4200 (on clean system; step time 22-24ms)
#   36.4M params, 512d/4L, SEQ=128

# Or use demo.sh for the full pipeline (download + train + generate):
cd .. && bash demo.sh
```

All experiments use the same 40MB binary file (`tinystories_smollm2_data00.bin`): 20M tokens, SmolLM2 BPE vocabulary (49152 tokens, 16893 active in this dataset), 90/10 train/val split. Every model trains from random initialization (Xavier/Kaiming) unless stated otherwise.

---

## How to Use This Codebase

### Train a model

```bash
cd training

# CPU-only (recommended — best model quality)
python3 train.py

# ANE matmul-only (ANE for linear projections, CPU for everything else)
# Edit train.py: set USE_ANE = True, ANE_MATMUL_ONLY = True
python3 train.py

# ANE full (all ops on ANE — worse quality, for research only)
# Edit train.py: set USE_ANE = True, ANE_MATMUL_ONLY = False
python3 train.py

# Or compile and run directly (skipping the Python wrapper):
make MODEL=autoresearch
./train --scratch --data ../tinystories_smollm2_data00.bin \
    --lr 4e-4 --warmup 100 --accum 10 --clip 1.0 \
    --steps 999999 --time 120 --scale 256.0 --cpu-only --seed 42
```

### Generate text

```bash
# From a trained checkpoint (uses numpy, no dependencies beyond Python stdlib)
python3 generate.py --prompt "Once upon a time" --tokens 200

# With a specific checkpoint
python3 generate.py --checkpoint training/ane_autoresearch_ckpt.bin --prompt "The cat" --tokens 100
```

### Convert weights

```bash
# HuggingFace → ANE (any llama-family model: llama, qwen2, qwen3, smollm2)
python3 tools/hf_to_ane.py HuggingFaceTB/SmolLM2-135M --output smollm2_135m.bin

# ANE → GGUF (for llama.cpp / Ollama)
python3 tools/export_to_gguf.py training/ane_autoresearch_ckpt.bin --output model.gguf

# GGUF → ANE (e.g., from Ollama)
python3 tools/gguf_to_ane.py model.gguf --output ane_checkpoint.bin
```

RoPE convention is handled automatically: ANE uses paired interleaving `[re0, im0, re1, im1, ...]`; HuggingFace/GGUF uses split halves `[re0, re1, ..., im0, im1, ...]`. Round-trip verified bit-perfect on all 38 tensors (512d/4L) and all 272 tensors (SmolLM2-135M).

### Run the autonomous hyperparameter search

```bash
cd training

# Random perturbation search (no AI agent needed, ~3 hours for 100 experiments)
python3 run_autosearch.py --experiments 100

# With multi-seed evaluation to reduce noise (slower but more reliable)
python3 run_autosearch.py --experiments 50 --n-seeds 3

# Architecture grid search
python3 autoresearch.py --search arch --time 120

# LR sweep across top architectures
python3 autoresearch.py --search lr --time 120
```

### Run tests

```bash
cd training && bash ../tests/test_training.sh

# 8 tests: compilation, forward/backward, Adam update,
# seed sensitivity, checkpoint I/O, ANE mode, gradient health, Python wrapper
```

### Measure power consumption

```bash
# Requires sudo for powermetrics
sudo bash tools/power_benchmark.sh
```

### LoRA fine-tuning

```bash
cd training
# Edit train.py: set USE_LORA = True, LORA_RANK = 8
# Ensure a checkpoint exists to fine-tune from
python3 train.py
```

---

## Navigating the Code

### Where things live

```
AutoANE/
├── training/                    # ── Core training engine ──
│   ├── train.m                  # Main training loop (Obj-C, ~1580 lines)
│   │                            #   Forward pass, backward pass, Adam optimizer,
│   │                            #   checkpoint save/load, CLI argument parsing,
│   │                            #   ANE kernel compilation, gradient accumulation
│   ├── cpu_ops.h                # CPU-side ops: RMSNorm fwd/bwd, cross-entropy,
│   │                            #   AdamW optimizer, embedding fwd/bwd, SDPA,
│   │                            #   vocab compaction (49152 → 16893 active tokens)
│   ├── mil_dynamic.h            # MIL program generators for 10 ANE kernels/layer:
│   │                            #   sdpaFwd, woFwd, ffnFused, sdpaBwd1/2, qBwd,
│   │                            #   kvBwd, wotBwd, ffnBwdW2t, ffnBwdW13t
│   ├── io.h                     # IOSurface helpers, fp32→fp16 weight staging,
│   │                            #   kernel compile/eval wrappers, GQA tile/reduce
│   ├── config.h                 # Model structs, ANE init, safe_malloc/calloc,
│   │                            #   checkpoint header (BLZT v4), layer allocation
│   ├── train_config.h           # Default hyperparameters (#ifndef guards)
│   ├── Makefile                 # Build: xcrun clang -O2, Accelerate, IOSurface, ObjC ARC
│   │
│   ├── train.py                 # Python wrapper: generates C header from hyperparams,
│   │                            #   compiles, runs binary. THE file the agent edits.
│   ├── run_autosearch.py        # Autonomous search: random perturbation + keep/revert
│   ├── autoresearch.py          # Grid search orchestrator (DIM × NLAYERS grids)
│   ├── program.md               # Karpathy-style agent protocol (for Claude Code)
│   ├── run_experiment.sh        # Single experiment runner (used by autoresearch.py)
│   ├── models/                  # 8 model header configs:
│   │   ├── autoresearch.h       #   512d/4L (36.4M) — default/optimal
│   │   ├── autoresearch_1536.h  #   1536d/4L (142M) — scaling study
│   │   ├── autoresearch_2048.h  #   2048d/4L (281M) — scaling study
│   │   ├── stories110m.h        #   768d/12L (110M) — TinyStories baseline
│   │   ├── smollm2_135m.h       #   576d/30L (135M) — SmolLM2 architecture
│   │   ├── smollm2_360m.h       #   960d/32L (362M) — SmolLM2 architecture
│   │   ├── qwen3_06b.h          #   896d/28L (600M) — Qwen3 architecture
│   │   └── qwen3_06b_compact.h  #   (compact vocab variant)
│   ├── results.tsv              # Autosearch results log
│   └── experiments.jsonl        # Grid search results
│
├── generate.py                  # ── Text generation ──
│                                #   Pure numpy inference, top-k sampling,
│                                #   loads BLZT v4 checkpoints, optional HF tokenizer
│
├── tools/                       # ── Utilities ──
│   ├── hf_to_ane.py             # HuggingFace safetensors → ANE checkpoint
│   ├── gguf_to_ane.py           # GGUF (llama.cpp) → ANE checkpoint
│   ├── export_to_gguf.py        # ANE checkpoint → GGUF with tokenizer metadata
│   ├── verify_forward_pass.py   # E2E verification: Python forward vs C binary
│   ├── download_data.sh         # Downloads TinyStories training data
│   └── power_benchmark.sh       # Power measurement script (requires sudo)
│
├── bridge/                      # ── Python ↔ ANE bridge ──
│   ├── ane_bridge.h             # C-callable API for ANE from Python/ctypes
│   ├── ane_bridge.m             # Bridge implementation (IOSurface, compile, eval)
│   └── Makefile                 # Builds libane_bridge.dylib
│
├── tests/
│   └── test_training.sh         # 8 regression tests
│
├── docs/                        # ── Documentation ──
│   ├── FINDINGS.md              # Complete findings, methodology, audit trail
│   ├── TECHNICAL_REPORT.md      # Full technical report (13 sections)
│   ├── EXPERIMENTS.md           # Experiment log (E1-E44)
│   ├── VERIFICATION.md          # First-principles verification (22 sections)
│   ├── ASSUMPTIONS.md           # 71 tracked assumptions (27 verified, 8 disproved, ...)
│   ├── NEXT_STEPS.md            # Research roadmap + session work logs
│   ├── RESEARCH_PLAN.md         # Original research plan (all tasks completed)
│   ├── P1_DESIGN.md             # Design doc: runtime weight injection
│   ├── E37_PROTOCOL.md          # Experiment 37 protocol (sustained throughput)
│   └── E38_PROTOCOL.md          # Experiment 38 protocol (dimension scaling)
│
├── demo.sh                      # One-command: download → train → generate
└── LICENSE                      # MIT
```

### How the training pipeline works

```
train.py (Python)
   │
   ├── 1. Generates autoresearch.h (C header with hyperparameters)
   ├── 2. Runs `make MODEL=autoresearch` to compile train.m
   └── 3. Runs ./train with CLI flags
          │
          train.m (Compiled Obj-C binary)
             │
             ├── Loads data file (mmap'd binary, uint16 tokens)
             ├── Loads/initializes weights (Xavier/Kaiming or from checkpoint)
             ├── IF ANE mode:
             │   ├── ane_init() — dlopen AppleNeuralEngine.framework
             │   ├── Compile 10 MIL kernels per layer (mil_dynamic.h)
             │   └── Create IOSurfaces for weight staging
             │
             └── Training loop (time-budgeted):
                 ├── Forward pass:
                 │   ├── Embed tokens (compact vocab: 49152 → 16893)
                 │   ├── For each layer:
                 │   │   ├── RMSNorm (cpu_ops.h)
                 │   │   ├── Q/K/V projections (ANE matmul or cblas_sgemm)
                 │   │   ├── RoPE (cpu_ops.h)
                 │   │   ├── SDPA + causal mask (ANE or cpu_ops.h)
                 │   │   ├── Output projection (ANE or cblas_sgemm)
                 │   │   └── SwiGLU FFN with residual (ANE or cblas_sgemm)
                 │   └── Classifier head → cross-entropy loss
                 │
                 ├── Backward pass (reverse order, same structure)
                 │
                 └── Every ACCUM_STEPS:
                     ├── Gradient clipping (global L2 norm)
                     ├── AdamW update (cpu_ops.h)
                     ├── LR schedule (cosine with warmup)
                     └── Checkpoint save (BLZT v4 format)
```

### Key design decisions

1. **Compiled C binary, not Python**: The training loop is Obj-C for direct access to ANE private APIs and Accelerate framework. Python is only used as a config generator and orchestrator.

2. **Single IOSurface per kernel**: Each ANE kernel uses exactly 1 input and 1 output IOSurface. Weights and activations are packed into the spatial dimension and separated by `slice_by_size` inside the MIL program. This provides structural immunity to IOSurface slot ordering bugs.

3. **Compact vocabulary**: The full SmolLM2 vocab (49152 tokens) is mapped to the 16893 tokens active in the training data. Reduces classifier from 49152×512 to 16893×512, saving memory and compute.

4. **DeepNet scaling**: Residual connections use `alpha = 1/sqrt(2*n_layers)` for fp16 stability in ANE modes. Required because fp16 residual streams overflow without scaling.

---

## Novel Contributions

We distinguish what is novel from what is adapted from prior work.

**Novel (no prior published equivalent):**
- First quantitative ANE vs CPU training comparison at matched configurations (throughput, loss quality, power)
- First ANE training power measurement (macOS `powermetrics`, 60s per mode, idle-subtracted)
- First IOSurface scaling study across model dimensions (DIM 1024/1536/2048) — demonstrates hard memory pressure ceiling at ~220MB total IOSurface allocation
- First demonstration that selective ANE offloading (matmul-only) matches CPU training quality
- Autonomous hyperparameter search on Apple Silicon with compiled C binary and git keep/revert protocol
- Lossless weight conversion pipeline: HuggingFace <-> ANE checkpoint <-> GGUF (verified bit-perfect on 272-tensor round-trip)

**Adapted from prior work:**
- ANE forward/backward pass via `_ANEClient`/`_ANECompiler` — adapted from [maderix/ANE](https://github.com/maderix/ANE)
- MIL kernel generation and IOSurface spatial packing — line-for-line adaptation from maderix
- Delta compilation investigation — inspired by [Orion](https://github.com/mechramc/Orion) (Murai Labs)
- Keep/revert experiment protocol — adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- DeepNet scaling (`res_alpha = 1/sqrt(2N)`) — from [Wang et al. (2022)](https://arxiv.org/abs/2203.00555)

---

## Research Findings

### Finding 1: Step count dominates model capacity at fixed time budgets

At 120s on Apple Silicon (M2 Pro), 512d/4L (36.4M params) gets ~2500 steps at 42ms/step, achieving val_loss ~3.5. Meanwhile 1024d/4L (95.4M params) gets only ~1050 steps at ~100ms/step, achieving val_loss ~4.3. The smaller model gets ~2.4x more gradient updates and wins by ~0.8 nats.

This advantage **widens** at longer budgets:

| Budget | 512d/4L val_loss | 768d/2L val_loss | Gap |
|--------|------------------|------------------|-----|
| 120s | 3.54 | 3.69 | 0.15 |
| 300s | 3.09 | 3.20 | 0.11 |
| 600s | 2.55 | 2.84 | 0.29 |

At 1800s (4.89 epochs), it reaches val_loss 2.22, consistent with the TinyStories literature baseline of ~2.0 at convergence for 33M-param models (Eldan & Li, 2023).

**Interpretation**: Kaplan et al. (2020) showed loss scales as L(D) ~ D^(-0.095) for data and L(N) ~ N^(-0.076) for parameters. Since 0.095 > 0.076, each additional gradient step contributes more than each additional parameter. In our severely data-constrained regime (23-329x below Chinchilla optimal), this effect is amplified.

Experiments: E39, E40, E41, E42.

### Finding 2: CPU-only training beats ANE at every tested dimension

| Mode | ms/step | Steps@120s | val_loss | Matmul time | IO overhead |
|------|---------|-----------|----------|-------------|-------------|
| CPU-only fp32 | 102.2 | 1041 | 4.20 | 71.9ms | 0.2ms |
| ANE fp16 | 68.7 | 1297 | 4.69 | 29.2ms | 8.1ms |
| ANE matmul-only | ~71 | 1149 | ~4.65 | ~30ms | ~8ms |

(All values for 1024d/4L, 95.4M params, loss_scale=256.)

ANE achieves 2.46x faster raw matmuls but IOSurface weight staging costs 8.1ms/step and fp16 precision degrades loss by ~16% (irreducible). At DIM=2048, ANE is 2x *slower* due to IOSurface memory pressure (379MB causes cache thrashing).

Experiments: E11, E36, E38.

### Finding 3: ANE does not save power

| Mode | Package Power | Energy/step |
|------|--------------|-------------|
| Idle | 8,455 mW | — |
| CPU-only | 13,273 mW | 9.2 mJ |
| ANE matmul | 12,568 mW | 10.9 mJ |
| ANE full | 12,664 mW | 9.7 mJ |

ANE shifts ~1.4W from CPU to ANE subsystem, but total power is unchanged. CPU-only achieves the lowest energy per step.

Experiment: E12.

### Finding 4: fp16 precision gap is irreducible

5 approaches tested to close the ~16% quality gap — all failed. Root cause: fp16 MAC units accumulate sqrt(DIM) ULPs of rounding error per dot product. Hardware limitation, not software-fixable.

Experiments: E14, E15, E19.

### Finding 5: Only use ANE for linear projections

ANE matmul-only mode matches CPU val_loss to 4 decimal places. The precision problem is specifically in non-linear ops (softmax, SiLU, RoPE) where fp16 error compounds. Linear projections tolerate fp16.

Experiment: E36.

### Additional findings

| # | Finding | Evidence |
|---|---------|----------|
| 6 | Delta compilation does not work (5 approaches tested) | E10, E17, E34 |
| 7 | Autonomous search hill-climbs on noise (variance > signal) | E44 |
| 8 | Depth hurts at every width tested (120s budget) | E39 |
| 9 | 2-layer models overfit despite high throughput | E40 |
| 10 | Optimal LR scales with sqrt(model size) | E40 |
| 11 | SEQ=128 optimal (throughput > context at 120s) | E43 |
| 12 | Neither CPU nor ANE diverges at 10+ minutes | E37 |
| 13 | Thermal throttling is 30% single-process (E18's 50% was concurrent) | E24 |

Full findings with methodology: **[docs/FINDINGS.md](docs/FINDINGS.md)**

---

## Comparison to Existing Frameworks

| | AutoANE (CPU-only) | MLX | PyTorch (MPS) |
|--|---------------------|-----|---------------|
| Compute unit | CPU (AMX/Accelerate) | GPU | GPU |
| Mac usable during training? | Yes | No (GPU shared with UI) | No |
| Precision | fp32 | fp32/fp16 | fp32/fp16 |
| Setup | `make && ./train` (zero deps) | `pip install mlx` | `pip install torch` |
| Autonomous research | Built-in | Manual | Manual |
| Export formats | GGUF, ANE checkpoint | Safetensors | PyTorch, ONNX |
| On-device (iOS) path | ANE mode | No | No |

AutoANE's unique property: zero-dependency compiled binary with built-in autonomous search. Training runs on CPU, so the Mac stays fully usable. ANE mode provides a research path toward on-device training on iPhones/iPads (where ANE is the only thermally viable compute unit for sustained workloads).

MLX is better for: raw GPU throughput on large models, broader ecosystem, rapid prototyping in Python.

---

## Known Limitations

1. **fp16 precision gap** (ANE modes): ~16% quality loss from fp16 matmul accumulation rounding. Irreducible via software (5 approaches tested).

2. **DeepNet incompatibility with pretrained weights**: DeepNet scaling (required for fp16 stability) changes residual magnitudes, making ANE fine-tuning from pretrained weights impractical. CPU-only mode works fine.

3. **Single dataset**: All results on TinyStories (20M tokens). Models are 23-329x below Chinchilla optimal data:parameter ratio. Larger datasets would likely change the optimal architecture.

4. **Private APIs**: Uses `_ANEClient`, `_ANECompiler` from `AppleNeuralEngine.framework` — undocumented, subject to change between macOS versions.

5. **Sequence length**: SEQ=128 is optimal for throughput but limits context. SEQ=512+ increases SDPA backward overflow risk on ANE.

---

## Open Questions

1. Does the step-count advantage hold with larger datasets? Chinchilla predicts a crossover where larger models become optimal — but at what data scale?
2. Can zeroth-order training (MeZO/FwdLLM) leverage ANE's fast forward passes while eliminating backward kernels entirely? This is the highest-impact untested direction.
3. Mega-kernel fusion (N transformer layers in one MIL program) achieves 3-4x forward speedup ([maderix/ANE PR #24](https://github.com/maderix/ANE/issues/24)). Can this be combined with runtime weight injection?
4. Function parameter IOSurfaces are 30% faster than our spatial packing ([maderix/ANE PR #22](https://github.com/maderix/ANE/pull/22)). Worth implementing?
5. INT8 quantization halves IOSurface size — does this move the memory pressure ceiling from DIM=1536 to DIM=2048+?

---

## Related Work

| Project | Key Finding | Relevance |
|---------|------------|-----------|
| [imperatormk/ane-train](https://github.com/imperatormk/ane-train) | Runtime weight injection via IOSurface matmul inputs — compile once, train forever. Full Adam on ANE. | Documents critical IOSurface constraints (ascending slot sizes, Ci multiple of 32). |
| [maderix/ANE PR #24](https://github.com/maderix/ANE/issues/24) | Mega-kernel fusion: 4.17x forward speedup. XPC overhead ~160us/eval is the bottleneck. | Fusion + runtime weights is the key open question. |
| [maderix/ANE PR #35](https://github.com/maderix/ANE/pull/35) | M5 ANE support: 128-byte IOSurface alignment. | M5 compatibility for future work. |
| [Orion](https://github.com/mechramc/Orion) (Murai Labs) | Delta compilation: 8.5x recompile speedup. 20 documented ANE constraints. | We could not reproduce the delta compilation mechanism. |
| [thebasedcapital/ane-infer](https://github.com/thebasedcapital/ane-infer) | 25 `_ANEClient` methods documented. `doEvaluateDirectWithModel:` bypasses daemon. | Note: `_ANEChainingRequest` is dead on macOS 15+ (requires Espresso IR from disk-compiled models). |

---

## Credits

- **[maderix](https://github.com/maderix)** — [ANE](https://github.com/maderix/ANE): original reverse engineering of Apple's Neural Engine private APIs and first-ever training on ANE hardware. AutoANE's bridge, IOSurface code, and MIL generation are direct adaptations.
- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch): the autonomous keep/revert experiment protocol.
- **[Orion](https://github.com/mechramc/Orion)** (Murai Labs): ANE training/inference runtime. Delta compilation findings motivated our investigation.
- **[imperatormk](https://github.com/imperatormk)** — [ane-train](https://github.com/imperatormk/ane-train): runtime weight injection approach, comprehensive ANE training cheatsheet.
- **[thebasedcapital](https://github.com/thebasedcapital)** — [ane-infer](https://github.com/thebasedcapital/ane-infer): documented 25 `_ANEClient` methods, achieved 3.6 TFLOPS with fused mega-kernels.

## License

MIT — see [LICENSE](LICENSE)
