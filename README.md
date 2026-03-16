# AutoANE

**Training transformers on Apple's Neural Engine: an empirical study**

AutoANE is an open-source system for training Llama-family transformers on Apple Silicon, building on [maderix/ANE](https://github.com/maderix/ANE)'s pioneering reverse engineering of the Neural Engine's private APIs. It adds systematic ANE vs CPU benchmarking, power measurement, and an autonomous hyperparameter search loop adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The training loop is a compiled Objective-C binary (~1580 lines).

44 controlled experiments (backprop), 37 MeZO conditions, a 4-phase ANE optimization pipeline, a 100-experiment autonomous search, and first-principles verification produced five central findings:

1. **For backprop training, CPU beats ANE at every tested size** (36M-281M params) due to IOSurface weight-staging overhead and irreducible fp16 precision loss.

2. **MeZO+LoRA-split with conv-fused kernels achieves 1.71x faster than CPU** (262ms/step vs 447ms) — the first ANE-faster-than-CPU training result on Apple Silicon. The key: LoRA-split freezes base weights as BLOBFILE constants (no per-step staging), conv1x1 fused kernels reduce IO round-trips from 224 to 96, with zero convergence impact.

3. **ANE does not save power.** Direct measurement via `powermetrics` shows package power of 13.3W (CPU-only), 12.6W (ANE matmul), and 12.7W (ANE full). First published power measurement of ANE training workloads.

4. **In a fixed time window, more optimizer steps beats more parameters.** 512d/4L (36M params, ~4300 steps at 120s) achieves val_loss ~3.5 while 1024d/4L (95M params, ~1050 steps) achieves ~4.3.

5. **Advanced ZO gradient estimators don't help for LoRA ZO.** FZOO multi-perturbation (K=4): zero wall-time benefit. P-GAP gradient-aligned perturbations: diverges with paper hyperparams, neutral with standard hyperparams. Root cause: LoRA rank-8 matrices are too small for per-matrix SVD.

**[Complete Findings & Methodology](docs/FINDINGS.md)** | **[ANE Pipeline Optimization Spec](docs/specs/2026-03-15-ane-training-pipeline-optimization.md)** | **[P-GAP Research Log](docs/specs/2026-03-16-phase4-pgap-research-log.md)** | [MeZO Audit Report](docs/MEZO_AUDIT_REPORT.md) | [Technical Report](docs/TECHNICAL_REPORT.md) | [Experiment Log (E1-E44)](docs/EXPERIMENTS.md) | [Assumptions Registry](docs/ASSUMPTIONS.md) | [Research Roadmap](docs/NEXT_STEPS.md)

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

### MeZO (zeroth-order) training

MeZO eliminates the backward pass — only forward passes + weight perturbation. Uses inference-only memory.

```bash
cd training

# MeZO fine-tune from HuggingFace checkpoint (CPU, SmolLM2-360M)
python3 tools/hf_to_ane.py HuggingFaceTB/SmolLM2-360M ane_smollm2_360m_ckpt.bin
make MODEL=smollm2_360m
./train_mezo --resume ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-4 --eps 1e-3 --time 120 --cpu-only \
    --lora --lora-rank 8 --lora-split

# MeZO+LoRA-split on ANE with conv-fused kernels (FASTEST — 1.71x CPU)
./train_mezo --resume ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-4 --eps 1e-3 --time 120 \
    --lora --lora-rank 8 --lora-split --conv-fused

# MeZO on ANE (full-parameter, slower — for comparison only)
./train_mezo --resume ane_smollm2_360m_ckpt.bin \
    --data ../tinystories_smollm2_data00.bin \
    --lr 1e-5 --eps 1e-3 --time 120
```

### Generate text

```bash
# From a trained checkpoint (uses numpy, no dependencies beyond Python stdlib)
python3 generate.py --prompt "Once upon a time" --tokens 200

# With a specific checkpoint
python3 generate.py training/ane_autoresearch_ckpt.bin --prompt "The cat" --tokens 100

# SmolLM2 checkpoint (requires --rope-theta 100000)
python3 generate.py ane_smollm2_360m_ckpt.bin --rope-theta 100000 --prompt "The little bear"

# From-scratch checkpoint (uses DeepNet residual scaling)
python3 generate.py training/ane_autoresearch_ckpt.bin --from-scratch --prompt "Once upon"
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

### Run tests and verification

```bash
# Training regression tests (8 tests)
cd training && bash ../tests/test_training.sh

# Comprehensive verification (27 automated checks: data, config, checkpoint, architecture)
python3 verify_all.py

# Individual verification scripts
python3 tests/verify_multi_position.py       # Multi-seed HuggingFace comparison
python3 tests/verify_qk_interleave.py        # Q/K weight interleaving (bit-exact)
python3 tests/verify_blas_channel_first.py    # BLAS channel-first layout test
python3 tools/verify_forward_pass.py          # Python forward vs C binary loss
python3 tools/verify_mezo_gradient_bias.py    # MeZO gradient bias quantification
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
│   ├── train.m                  # Backprop training loop (Obj-C, ~1580 lines)
│   │                            #   Forward pass, backward pass, Adam optimizer,
│   │                            #   checkpoint save/load, CLI argument parsing,
│   │                            #   ANE kernel compilation, gradient accumulation
│   ├── train_mezo.m             # MeZO training loop (Obj-C, ~2100 lines)
│   │                            #   Zeroth-order optimization (forward-only),
│   │                            #   MeZO+LoRA, MeZO+LoRA-split, conv1x1 hybrid,
│   │                            #   conv-fused kernels (QKV+FFN), xoshiro256+ RNG,
│   │                            #   Rademacher/Gaussian perturbation, SPSA,
│   │                            #   FZOO multi-perturbation, P-GAP per-matrix SVD
│   ├── cpu_ops.h                # CPU-side ops: RMSNorm fwd/bwd, cross-entropy
│   │                            #   (log-sum-exp), AdamW, embedding fwd/bwd, SDPA,
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
│   ├── verify_mezo_gradient_bias.py # MeZO gradient bias quantification
│   ├── download_data.sh         # Downloads TinyStories training data
│   └── power_benchmark.sh       # Power measurement script (requires sudo)
│
├── bridge/                      # ── Python ↔ ANE bridge ──
│   ├── ane_bridge.h             # C-callable API for ANE from Python/ctypes
│   ├── ane_bridge.m             # Bridge implementation (IOSurface, compile, eval)
│   └── Makefile                 # Builds libane_bridge.dylib
│
├── tests/
│   ├── test_training.sh         # 8 regression tests
│   ├── verify_multi_position.py # Multi-seed HuggingFace forward pass comparison
│   ├── verify_qk_interleave.py  # Q/K weight interleaving verification (bit-exact)
│   └── verify_blas_channel_first.py # BLAS channel-first layout numerical test
│
├── docs/                        # ── Documentation ──
│   ├── FINDINGS.md              # Complete findings, methodology, audit trail (8 findings)
│   ├── MEZO_AUDIT_REPORT.md     # MeZO-on-ANE comprehensive audit (37 conditions, v12)
│   ├── TECHNICAL_REPORT.md      # Full technical report (14 sections)
│   ├── EXPERIMENTS.md           # Experiment log (E1-E44)
│   ├── VERIFICATION.md          # First-principles verification (22 sections)
│   ├── ASSUMPTIONS.md           # 71+ tracked assumptions (27 verified, 8 disproved, ...)
│   ├── NEXT_STEPS.md            # Research roadmap + session work logs (5 sessions)
│   ├── RESEARCH_PLAN.md         # Original research plan (all tasks completed)
│   ├── ANE_STRENGTHS_PLAN.md    # Conv1x1 + MeZO optimization plan (with actual results)
│   ├── specs/                   # Design specs and research logs
│   │   ├── 2026-03-15-ane-training-pipeline-optimization.md  # 4-phase pipeline spec
│   │   ├── 2026-03-16-phase4-pgap-research-log.md           # P-GAP negative result log
│   │   ├── 2026-03-12-mezo-ane-training-design.md            # MeZO design spec
│   │   └── 2026-03-12-mezo-ane-training-plan.md              # MeZO implementation plan
│   ├── P1_DESIGN.md             # Design doc: runtime weight injection
│   ├── E37_PROTOCOL.md          # Experiment 37 protocol (sustained throughput)
│   └── E38_PROTOCOL.md          # Experiment 38 protocol (dimension scaling)
│
├── results/                     # ── MeZO experiment results ──
│   ├── analysis.md              # Full MeZO vs backprop results (37 conditions)
│   ├── research_audit.md        # Detailed v12 research audit
│   ├── condition*.txt           # Raw output from each experimental condition
│   ├── validation_*.c           # C validation programs (perturbation, gradient)
│   └── validate_*/              # Compiled validation binaries
│
├── verify_all.py                # 27 automated verification checks
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
- **First ANE training faster than CPU** — MeZO+LoRA-split with conv-fused kernels: 262ms/step = 1.71x faster than CPU (447ms), zero convergence impact
- First zeroth-order (MeZO) training on any NPU hardware, with 37 controlled conditions across 3 model sizes + 4-phase ANE optimization pipeline
- MeZO+LoRA-split architecture: base weights as BLOBFILE constants, LoRA correction on CPU — eliminates IOSurface weight staging entirely
- Conv1x1 hybrid + fused kernels: selective conv1x1 for 5/7 projections (2x speedup), QKV combined + FFN mega-kernel fusion reduces IO round-trips from 224 to 96
- Negative result for P-GAP on LoRA ZO: faithful per-matrix SVD implementation matching arXiv:2510.18228 — diverges with paper hyperparams, neutral with standard hyperparams. Root cause: LoRA rank-8 matrices too small for SVD
- Negative result for FZOO multi-perturbation: better gradient quality offset by 2.5x computational overhead
- First quantitative ANE vs CPU training comparison at matched configurations (throughput, loss quality, power)
- First ANE training power measurement (macOS `powermetrics`, 60s per mode, idle-subtracted)
- First IOSurface scaling study across model dimensions (DIM 1024/1536/2048) — demonstrates hard memory pressure ceiling at ~220MB
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

### MeZO (Zeroth-Order) Training Findings

37 experimental conditions + 4-phase ANE optimization pipeline. Full audit: **[docs/MEZO_AUDIT_REPORT.md](docs/MEZO_AUDIT_REPORT.md)**. Optimization pipeline: **[docs/specs/2026-03-15-ane-training-pipeline-optimization.md](docs/specs/2026-03-15-ane-training-pipeline-optimization.md)**. P-GAP research log: **[docs/specs/2026-03-16-phase4-pgap-research-log.md](docs/specs/2026-03-16-phase4-pgap-research-log.md)**.

| # | Finding | Key Evidence |
|---|---------|-------------|
| 14 | MeZO works on ANE (first ZO training on any NPU) | Losses match CPU within perturbation noise |
| 15 | Full-parameter MeZO-ANE is 32-47% slower than CPU | IOSurface restaging scales superlinearly with model size |
| 16 | MeZO+LoRA-split eliminates transpose bottleneck | Perturbation 193x faster (579→3ms), transpose 478→0ms |
| 17 | MeZO converges ~100x slower per step than backprop | Val loss delta 0.005 (MeZO) vs 0.30 (BP) in 100 steps |
| 18 | MeZO's sole advantage is 2.0-2.4x lower memory | Only critical at ~1B+ params on 8GB devices |
| 19 | Optimal MeZO LR is ~30x smaller than backprop LR | lr=1e-5 (full), lr=1e-4 (LoRA-split) |
| 20 | Lower LoRA rank = lower ZO variance = better signal | Rank 8 ≥ rank 32 in convergence quality |
| **21** | **Conv-fused MeZO+LoRA-split: 1.71x faster than CPU** | **262ms/step vs 447ms, val_loss within 0.03%** |
| 22 | FZOO K=4 provides zero wall-time convergence benefit | 2.5x more forward passes offsets better gradient quality |
| 23 | P-GAP does not transfer to LoRA ZO training | Paper params diverge; standard params neutral; LoRA rank-8 too small for SVD |

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
2. ~~Can zeroth-order training (MeZO) leverage ANE's fast forward passes?~~ **Answered: YES — 1.71x faster than CPU** with LoRA-split + conv-fused kernels (262ms/step vs 447ms). See [design spec](docs/specs/2026-03-15-ane-training-pipeline-optimization.md).
3. ~~Can FZOO/P-GAP improve MeZO convergence?~~ **Answered: NO for LoRA ZO.** FZOO K=4: zero wall-time benefit (2.5x cost offsets quality). P-GAP: diverges with paper params, neutral with standard params. See [P-GAP research log](docs/specs/2026-03-16-phase4-pgap-research-log.md).
4. **Cross-layer fusion**: Current fusion is intra-layer (QKV combined, FFN mega-kernel). Can multiple transformer layers be fused into fewer ANE dispatches? Challenge: LoRA corrections between layers prevent full fusion.
5. **Larger model scaling**: Does the 1.71x speedup increase at >360M params? Conv1x1 advantages should grow with wider dimensions. IOSurface pressure may be lower in LoRA-split mode (activation-only surfaces).
6. **INT8 LoRA corrections**: Can LoRA A/B matrices be quantized to INT8 for ANE compute? Leverages 1.88x INT8 throughput. Risk: rank-8 matrices may lose precision.
7. **Mobile MeZO deployment**: ANE may be the only thermally viable compute for sustained on-device training on iPhone/iPad. MeZO+LoRA-split is the obvious path.
8. Does MeZO's memory advantage enable training at 1B+ params where backprop doesn't fit in 8GB?

---

## Related Work

### Core References

| Project | Key Finding | Relevance |
|---------|------------|-----------|
| [MeZO](https://arxiv.org/abs/2305.17333) (NeurIPS 2023) | In-place ZO-SGD, inference-only memory for LLM fine-tuning. | Foundation for our MeZO implementation. |
| [MobiZO](https://arxiv.org/abs/2409.15520) (arXiv 2024) | MP-LoRA on Qualcomm NPU, 4.3x speedup via parallelized perturbations. | Same concept (ZO+LoRA on NPU), deployed on Hexagon. |
| [Orion](https://github.com/mechramc/Orion) (Murai Labs) | ANE training, adapter-as-input, 20 ANE constraints, 3x via 1x1 conv, 170+ tok/s GPT-2. | Same hardware. Our LoRA-split is similar to adapter-as-input. |
| [imperatormk/ane-train](https://github.com/imperatormk/ane-train) | Runtime weight injection via IOSurface matmul inputs — compile once, train forever. | Documents critical IOSurface constraints (ascending slot sizes, Ci multiple of 32). |
| [maderix/ANE PR #24](https://github.com/maderix/ANE/issues/24) | Mega-kernel fusion: 4.17x forward speedup. XPC overhead ~160us/eval is the bottleneck. | Fusion + runtime weights is the key open question. |
| [maderix/ANE PR #35](https://github.com/maderix/ANE/pull/35) | M5 ANE support: 128-byte IOSurface alignment. | M5 compatibility for future work. |
| [thebasedcapital/ane-infer](https://github.com/thebasedcapital/ane-infer) | 25 `_ANEClient` methods documented. `doEvaluateDirectWithModel:` bypasses daemon. | Note: `_ANEChainingRequest` is dead on macOS 15+ (requires Espresso IR from disk-compiled models). |

### ZO Optimizer Advances (2025-2026)

| Paper | Key Finding | Relevance |
|-------|------------|-----------|
| [AGZO](https://arxiv.org/abs/2601.17261) (arXiv 2026) | Activation-guided subspace ZO; first ZO benchmarked on NPU (Ascend 910B2, avg 0.709 on Pangu-1B). | Validates ZO+NPU; activation-guided perturbation is complementary to conv1x1. |
| [FZOO](https://arxiv.org/abs/2506.09034) (arXiv 2025) | Rademacher + batched one-sided estimates; 18x fewer forward passes than MeZO on RoBERTa-large. | **Tested: zero wall-time benefit** for LoRA ZO — K=4 gives better gradient quality but 2.5x compute overhead offsets it. |
| [P-GAP](https://arxiv.org/abs/2510.18228) (arXiv 2025) | Gradient-aligned perturbation, 5.2x faster convergence. | **Tested: negative result** for LoRA ZO — paper params diverge, standard params neutral. LoRA rank-8 matrices too small for per-matrix SVD. |
| [ZOSA](https://arxiv.org/abs/2511.09156) (arXiv 2025) | O(ε²) bias proof for one-sided Rademacher perturbations. | Confirmed: one-sided has same asymptotic bias as central-difference. Used in our FZOO implementation. |
| [NoProp](https://arxiv.org/abs/2503.24322) (arXiv 2025) | No full forward/backward pass; diffusion-inspired block-local learning. | Alternative: per-block ANE fine-tuning without full graph compilation. |

### On-Device / Mobile Training (2024-2026)

| Paper | Key Finding | Relevance |
|-------|------------|-----------|
| [FwdLLM](https://arxiv.org/abs/2308.13894) (USENIX ATC 2024) | Forward-only federated LLM fine-tuning; 14.6x memory reduction; LLaMA on commodity mobile. | Same core idea (ZO+LoRA+mobile); validates federated path. |
| [MobileFineTuner](https://arxiv.org/abs/2512.08211) (arXiv 2025) | End-to-end C++ fine-tuning on phones; GPT-2/Gemma 3/Qwen 2.5; energy-aware scheduling. | Peer comparison (backprop-based); energy scheduling applicable. |
| [Scaling NPU Test-Time Compute](https://arxiv.org/abs/2509.23324) (EUROSYS 2026) | 19x mixed-precision GEMM speedup; LUT-based softmax/dequant on mobile NPU. | Tile quantization + LUT ops directly applicable to ANE. |
| [On-Device ZO](https://arxiv.org/abs/2511.11362) (arXiv 2025) | Theory: MeZO enables significantly larger models within on-chip memory vs backprop. | Validates our memory advantage thesis at scale. |

### ANE Tools & Libraries

| Project | Key Finding | Relevance |
|---------|------------|-----------|
| [Anemll](https://github.com/Anemll/Anemll) | Open-source ANE inference: LLaMA/Qwen/Gemma 3; 47-62 tok/s on 1B; ANEMLL-Dedup ~50% size. | Reference ANE implementation. |
| [anemll-bench](https://github.com/Anemll/anemll-bench) | ANE benchmarking with memory bandwidth metrics. | Could validate our conv1x1 speedup claims. |

---

## Credits

- **[maderix](https://github.com/maderix)** — [ANE](https://github.com/maderix/ANE): original reverse engineering of Apple's Neural Engine private APIs and first-ever training on ANE hardware. AutoANE's bridge, IOSurface code, and MIL generation are direct adaptations.
- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch): the autonomous keep/revert experiment protocol.
- **[Orion](https://github.com/mechramc/Orion)** (Murai Labs): ANE training/inference runtime. Delta compilation findings motivated our investigation.
- **[imperatormk](https://github.com/imperatormk)** — [ane-train](https://github.com/imperatormk/ane-train): runtime weight injection approach, comprehensive ANE training cheatsheet.
- **[thebasedcapital](https://github.com/thebasedcapital)** — [ane-infer](https://github.com/thebasedcapital/ane-infer): documented 25 `_ANEClient` methods, achieved 3.6 TFLOPS with fused mega-kernels.

## License

MIT — see [LICENSE](LICENSE)
