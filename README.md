# AutoANE

**Autonomous ML research on Apple Silicon — train transformers from scratch, let an AI agent find the optimal configuration.**

AutoANE is two things:

1. **The first open-source training system for Apple's Neural Engine** — a full forward/backward pass for Llama-family transformers (RMSNorm + GQA + RoPE + SwiGLU) via reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`)
2. **An autonomous research agent** that implements [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — an AI agent iteratively modifies hyperparameters, trains, measures loss, and keeps or reverts changes in a tight loop, no human in the loop

43 experiments have produced [8 verified research findings](#research-findings), a [first-principles verification report](docs/VERIFICATION.md), and a [tracked assumption registry](docs/ASSUMPTIONS.md) with 25 verified, 13 unverified, and 7 disproved assumptions.

## Quick Start

**Requirements:** macOS 15+, Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools

```bash
git clone https://github.com/vatsal191201/AutoANE.git
cd AutoANE

# Download training data (~40MB TinyStories, pre-tokenized)
bash tools/download_data.sh

# Build and train from scratch (2 minutes, CPU-only)
cd training
make MODEL=autoresearch
./train --scratch --time 120 --cpu-only --lr 4e-4

# Or use the autoresearch agent (requires Claude Code)
python3 train.py
```

### What Happens

`train.py` generates a C header from its hyperparameters, compiles the Objective-C training binary (`train.m`, 1571 lines), and runs training for `TIME_BUDGET` seconds. Output includes machine-parseable metrics:

```
final_loss:       3.348118
val_loss:         3.507000
training_seconds: 104.6
total_seconds:    120.0
num_steps:        4074
num_params_M:     36.4
```

### Autonomous Agent Loop

Point Claude Code (or any LLM agent) at `training/program.md`. The agent:

1. Creates a git branch
2. Modifies `training/train.py` (the **only** mutable file — hyperparameters and architecture)
3. Commits, runs `python3 train.py` (compile + train)
4. Parses `val_loss`
5. **Keeps** if improved, **reverts** (`git reset --hard HEAD~1`) if not
6. Loops until interrupted

Results tracked in `training/results.tsv`. In our first session, 13 experiments in ~40 minutes found two improvements (SEQ=128, LR=4e-4), reducing val_loss from 3.533 to 3.507.

## Capabilities

### Training Modes

| Mode | Flag | Precision | Use Case |
|------|------|-----------|----------|
| **CPU-only** (default) | `--cpu-only` | fp32 | Best loss, recommended for all sizes |
| ANE matmul-only | `--ane-matmul-only` | fp16 matmul + fp32 ops | Research: ANE for linear projections, CPU for everything else |
| ANE full | *(default without flags)* | fp16 | Research: full ANE pipeline, ~16% worse loss |

### Supported Architectures

Any Llama-family transformer. Pre-configured models:

| Model | Params | Layers | Attention | Build Command |
|-------|--------|--------|-----------|---------------|
| Stories110M | 109M | 12 | MHA 12/12, dim=768 | `make MODEL=stories110m` |
| SmolLM2-135M | 135M | 30 | GQA 9/3, dim=576 | `make MODEL=smollm2_135m` |
| SmolLM2-360M | 362M | 32 | GQA 15/5, dim=960 | `make MODEL=smollm2_360m` |
| Autoresearch | Variable | Agent-tuned | Auto-configured | `make MODEL=autoresearch` |

### Training Features

- **AdamW optimizer** with cosine LR decay, gradient accumulation, gradient clipping
- **Gradient accumulation** (default 10 steps, effective batch = ACCUM_STEPS x SEQ tokens)
- **DeepNet scaling** (`res_alpha = 1/sqrt(2*N_layers)`) for fp16 stability in ANE modes
- **LoRA fine-tuning** (`--lora --lora-rank 8`) — freeze base weights, train adapters only
- **Checkpointing** — resume training with `--resume`
- **Time-budgeted training** (`--time 120`) — stops cleanly after N seconds

### Weight Conversion

Import pretrained weights from HuggingFace or GGUF format:

```bash
# From HuggingFace (any llama-family model)
python3 tools/hf_to_ane.py HuggingFaceTB/SmolLM2-360M checkpoint.bin

# From GGUF (e.g., from Ollama)
python3 tools/gguf_to_ane.py model.gguf checkpoint.bin

# Then train with imported weights
cd training && ./train --resume --data ../tinystories_smollm2_data00.bin
```

### C Bridge API

`bridge/ane_bridge.h` provides a C-callable interface to ANE private APIs for use from Python (via ctypes) or any language with C FFI:

```c
ane_bridge_init();                          // Load ANE framework
ANEKernelHandle *k = ane_bridge_compile(    // Compile MIL program
    mil_text, mil_len, weight_data, weight_len,
    n_inputs, input_sizes, n_outputs, output_sizes);
ane_bridge_write_input(k, 0, data, bytes);  // Stage input
ane_bridge_eval(k);                         // Run on ANE
ane_bridge_read_output(k, 0, out, bytes);   // Read result
ane_bridge_free(k);                         // Cleanup
```

Includes INT8 quantization helpers (`ane_bridge_build_weight_blob_quantized`) and transposed weight blob construction for backward passes.

### Autoresearch Modes

| Mode | Script | Use Case |
|------|--------|----------|
| **Agent Loop** | `program.md` + `train.py` | Autonomous Karpathy-style keep/revert via Claude Code |
| **Grid Search** | `autoresearch.py` | Pre-defined architecture/LR sweeps, no agent needed |

## How It Works

### ANE Kernel Pipeline

The system compiles **10 MIL (Machine Learning Intermediate Language) kernels per layer** at startup:

| Kernel | Direction | Purpose |
|--------|-----------|---------|
| sdpaFwd | Forward | Scaled dot-product attention |
| woFwd | Forward | Output projection (GQA-aware) |
| ffnFused | Forward | SwiGLU FFN with residual |
| sdpaBwd1/2 | Backward | Attention backward (two-pass) |
| qBwd | Backward | Query projection backward |
| kvBwd | Backward | Key/value projection backward |
| wotBwd | Backward | Output projection backward (transpose) |
| ffnBwdW2t | Backward | FFN W2 backward |
| ffnBwdW13t | Backward | FFN W1/W3 backward |

Weights are staged via **IOSurface spatial dimensions** — the compiled MIL program never changes, only the weight data is patched each step via NEON-vectorized fp32-to-fp16 conversion. This avoids the 4.2s recompilation cost per step that naive ANE training would require.

### Training Loop (per step)

1. **Forward**: ANE (attention + FFN matmuls) or CPU (cblas_sgemm) + CPU (RMSNorm, RoPE, residual)
2. **Loss**: CPU cross-entropy with fp32 accumulation
3. **Backward**: ANE or CPU matmuls + CPU weight gradients (cblas_sgemm)
4. **Optimizer**: AdamW with bias correction, cosine LR decay, gradient accumulation

### fp16 Stability (ANE modes)

ANE computes in fp16. DeepNet scaling keeps activations bounded:
- Without scaling: activations overflow to inf at layer ~20
- With scaling: stable training for 32+ layers, activations in [-6, +7] range
- Known limitation: SDPA backward produces near-zero dq/dk/dv due to fp16 underflow. Attention layers learn slowly; embedding + FFN gradients carry training.

## Research Findings

43 experiments across ANE characterization, architecture search, LR sweeps, budget scaling, verification runs, and autonomous agent loop. All results independently verified by re-running key configurations ([docs/VERIFICATION.md](docs/VERIFICATION.md)). Full experiment log: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

### Key Results

| # | Finding | Evidence | Literature |
|---|---------|----------|-----------|
| 1 | **Smaller models win at fixed time budgets** | 512d/4L (val 3.507) vs 1024d/4L (val 4.30) at 120s | Chinchilla in data-constrained regime |
| 2 | **Step count dominates model capacity** | 4074 steps x 36M params > 1050 steps x 95M params | Kaplan scaling: data exponent (0.095) > param exponent (0.076) |
| 3 | **Advantage widens with longer training** | Gap grows 0.15 to 0.29 from 120s to 600s | Muennighoff: repeated data effective to ~4 epochs |
| 4 | **CPU-only beats ANE for all tested sizes** | IOSurface overhead negates ANE matmul speedup | Novel (no prior ANE training comparisons exist) |
| 5 | **Depth aids generalization but costs throughput** | 4L beats 2L at same step count, loses at same wall-clock | Nguyen & Salazar (regime-specific) |
| 6 | **LR scales with sqrt(model size)** | 5e-4 for 36M, 3e-4 for 73M | LAMB heuristic |
| 7 | **Shorter sequences trade context for throughput** | SEQ=128 gives 1.75x steps vs SEQ=256, val improves | Throughput-dominated regime |
| 8 | **Optimal LR co-varies with batch size** | Halving SEQ requires LR 5e-4 to 4e-4 | Smith et al. (2018) linear scaling rule |

### The Central Finding

**"In a fixed time window, more optimizer steps beats more parameters."**

This is Kaplan et al.'s observation that the data scaling exponent (beta = 0.095) exceeds the parameter exponent (alpha = 0.076) — each additional training step improves loss more than each additional parameter. At 120 seconds on Apple Silicon, 512d/4L (36M params, 4074 steps) achieves val_loss **3.507** while 1024d/4L (95M params, 1050 steps) achieves only **4.30**. The smaller model gets 3.9x more gradient updates and wins decisively. This advantage **widens** at longer budgets.

### Best Configuration

```
512d/4L, SEQ=128, LR=4e-4, CPU-only, 120s budget
  val_loss 3.507, 4074 steps, 24.2ms/step, 36.4M params
  At 1800s: val_loss 2.22 (still improving at ~5 epochs on 20M token dataset)
```

### ANE Characterization (Novel Findings)

These results are novel — no prior work compares ANE and CPU training quality at matched configurations:

- **ANE has genuine 2.5x matmul speedup** over CPU AMX (cblas_sgemm), but IOSurface weight staging overhead negates the advantage end-to-end
- **IOSurface memory pressure** creates a hard ceiling: at DIM=1536 (220MB surfaces), ANE reaches parity; at DIM=2048 (379MB), ANE is 2x *slower* due to cache thrashing
- **fp16 precision gap is irreducible** (~16% worse loss) — not fixable by clamping, LR tuning, or weight decay. The root cause is fp16 matmul accumulation error (sqrt(DIM) ULPs of rounding per dot product)
- **Delta compilation does not work** via any tested API path (5 approaches tried). ANE loads from compiled cache, not source BLOBFILEs
- **Fusing non-linear ops into ANE fp16 causes overfitting** (val gap 1.22 vs 0.60). ANE should only handle linear projections (matmul), with CPU doing attention scores, SiLU, RoPE, and residuals

## Project Structure

```
AutoANE/
├── training/
│   ├── train.m              # Training binary (Objective-C, 1571 lines)
│   ├── train.py             # Agent-editable config (the ONLY mutable file)
│   ├── program.md           # Karpathy-style agent protocol
│   ├── autoresearch.py      # Grid search orchestrator
│   ├── run_experiment.sh    # Experiment runner (compile + run + log)
│   ├── train_config.h       # Default hyperparameters (#ifndef guards)
│   ├── mil_dynamic.h        # MIL kernel generator (10 kernels/layer)
│   ├── io.h                 # IOSurface I/O, weight staging, fp32-to-fp16
│   ├── config.h             # Derived sizes, memory allocation
│   ├── cpu_ops.h            # RMSNorm, RoPE, SDPA, loss, AdamW (CPU fp32)
│   ├── Makefile             # Build system (xcrun clang + Accelerate + IOSurface)
│   ├── experiments.jsonl    # Grid search results (JSON lines)
│   ├── results.tsv          # Agent loop results (keep/discard/crash)
│   └── models/              # Model header configs (8 architectures)
├── tools/
│   ├── hf_to_ane.py         # HuggingFace to ANE checkpoint converter
│   ├── gguf_to_ane.py       # GGUF to ANE checkpoint converter
│   └── download_data.sh     # Training data download (TinyStories)
├── bridge/
│   ├── ane_bridge.h         # C-callable ANE API (compile, eval, I/O)
│   ├── ane_bridge.m         # Bridge implementation (Objective-C)
│   └── Makefile
└── docs/
    ├── VERIFICATION.md      # First-principles verification of all claims
    ├── EXPERIMENTS.md       # Full experiment log (E1-E43)
    ├── ASSUMPTIONS.md       # Tracked assumptions: 25 verified, 7 disproved
    └── RESEARCH_PLAN.md     # Research roadmap and findings
```

## Known Limitations

- **Attention gradient underflow**: SDPA backward on ANE produces near-zero dq/dk/dv due to fp16 underflow. Training works (embedding + FFN gradients flow), but attention layers learn slowly.
- **No pretrained fine-tuning**: DeepNet scaling (required for fp16 stability) is incompatible with pretrained weight magnitudes. Training from scratch works; fine-tuning requires the CPU-only path.
- **Sequence length**: Default SEQ=128 (optimal for throughput at 120s). SEQ=64 degrades due to insufficient context for coherent gradients. SEQ=512+ increases SDPA backward overflow risk on ANE.
- **Private APIs**: Uses `_ANEClient`, `_ANECompiler` from `AppleNeuralEngine.framework` — undocumented and subject to change between macOS versions.
- **Single dataset**: Currently only TinyStories (20M tokens, SmolLM2 tokenizer). All models are severely data-starved (23x below Chinchilla optimal ratio).

## Roadmap

- [ ] CPU fp32 SDPA backward (fix attention gradient underflow)
- [ ] Mixed precision pipeline (ANE fp16 forward, CPU fp32 backward)
- [ ] Background training daemon (train while you work, ANE is a separate compute unit)
- [ ] GGUF/CoreML export pipeline
- [ ] Multi-shard dataset support (escape 20M token ceiling)
- [ ] Longer sequence lengths (requires fp32 attention backward)

## Credits

- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch): the autonomous keep/revert experiment loop protocol. AutoANE adapts this for Apple Silicon with compiled C training and ANE-specific constraints.
- **[Manjeet Singh (maderix)](https://github.com/maderix)** — [ANE](https://github.com/maderix/ANE): the original reverse engineering of Apple's Neural Engine private APIs and first-ever training on ANE hardware.
- **Orion** — [arXiv:2603.06728](https://arxiv.org/abs/2603.06728): first open end-to-end system for ANE LLM training and inference, demonstrating delta compilation and LoRA adapter hot-swap.

## License

MIT — see [LICENSE](LICENSE)
