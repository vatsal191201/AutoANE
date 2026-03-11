# AutoANE

**Train neural networks on Apple Neural Engine — the 19 TFLOPS your Mac isn't using.**

AutoANE combines [maderix/ANE](https://github.com/maderix/ANE) (the first neural network training system for Apple's Neural Engine) with [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (automated hyperparameter search via AI agents) to enable autonomous model training on hardware that sits idle in every Mac, iPad, and iPhone.

## Why This Exists

Your Mac has a Neural Engine delivering up to 38 TOPS / 19 TFLOPS FP16 — and it does nothing 99.9% of the time. MLX and PyTorch use the GPU, which means:

- **Your Mac becomes unusable during training** (GPU renders UI, video, 3D — same chip)
- **MLX has memory leaks** that crash 128GB machines during fine-tuning ([mlx#1406](https://github.com/ml-explore/mlx/issues/1406), [mlx#2254](https://github.com/ml-explore/mlx/issues/2254))
- **19 TFLOPS of compute sits completely idle**

ANE training uses a **separate compute unit** from the GPU. Train models while editing video, rendering 3D, or just using your computer normally.

### Energy Efficiency

| Hardware | FP16 TFLOPS | Power | Efficiency (TFLOPS/W) |
|----------|-------------|-------|-----------------------|
| M4 ANE | 19 | ~2.8W | **6.6** |
| A100 GPU | 312 | ~400W | 0.08 |

ANE is **80x more power-efficient per FLOP** than an A100. Hard power gating means zero leakage when idle.

## What It Does

1. **ANE Training Engine** — Trains Llama-family transformers (RMSNorm + GQA + RoPE + SwiGLU) entirely on the Neural Engine via reverse-engineered private APIs
2. **Autoresearch** — AI agent automatically searches for optimal model architecture and hyperparameters within your time/hardware budget
3. **No ML expertise required** — Give it data and a time budget, autoresearch figures out the rest

### Architecture

```
User: "Here's my data. Train something."
         |
    [Autoresearch Agent]
    - Profiles your hardware
    - Picks starting architecture
    - LOOP: tweak params -> train on ANE -> measure loss -> keep/revert
    - Returns best model for YOUR Mac + YOUR data
         |
    [ANE Training Engine]
    - 10 MIL kernels compiled once at startup
    - Weights staged via IOSurface (no recompilation per step)
    - fp16 ANE matmuls + fp32 CPU accumulation
    - DeepNet scaling for fp16 stability
         |
    Best model -> export to GGUF/CoreML
```

## Quick Start

**Requirements:** macOS 15+, Apple Silicon (M1/M2/M3/M4)

```bash
# Clone
git clone https://github.com/vatsal191201/AutoANE.git
cd AutoANE/training

# Download training data (TinyStories, ~40MB)
# Place tinystories_smollm2_data00.bin in training/ directory

# Build for SmolLM2-360M (default)
make MODEL=smollm2_360m

# Train from scratch (5 minutes)
./train --scratch --time 300

# Or use autoresearch (requires Claude Code or similar agent)
python3 train.py
```

### Supported Models

| Model | Params | Layers | Config |
|-------|--------|--------|--------|
| Stories110M | 109M | 12 | MHA 12/12, dim=768 |
| SmolLM2-135M | 135M | 30 | GQA 9/3, dim=576 |
| SmolLM2-360M | 362M | 32 | GQA 15/5, dim=960 |
| Autoresearch | Variable | Agent-tuned | Auto-configured |

### Build Options

```bash
make MODEL=stories110m      # 109M params, fastest
make MODEL=smollm2_135m     # 135M params
make MODEL=smollm2_360m     # 362M params (default baseline)
make MODEL=autoresearch     # Agent-configured (via train.py)
```

## How Training Works

### ANE Kernel Pipeline

The system compiles **10 MIL (Machine Learning Intermediate Language) kernels per layer** at startup:

| Kernel | Purpose |
|--------|---------|
| sdpaFwd | Scaled dot-product attention (forward) |
| ffnFused | SwiGLU FFN with residual connection (forward) |
| woFwd | Output projection (forward, GQA) |
| sdpaBwd1/2 | Attention backward (two-pass) |
| qBwd | Query projection backward |
| kvBwd | Key/value projection backward |
| wotBwd | Output projection backward (transpose) |
| ffnBwdW2t | FFN W2 backward |
| ffnBwdW13t | FFN W1/W3 backward |

Weights are staged via **IOSurface spatial dimensions** — the compiled program never changes, only the weight data gets patched each step. This avoids the 4.2s recompilation cost that naive ANE training requires.

### Training Loop

Each step:
1. **Forward pass**: ANE (attention + FFN) + CPU (RMSNorm, residual add)
2. **Loss**: CPU cross-entropy with fp32 accumulation
3. **Backward pass**: ANE (activation gradients) + CPU (weight gradients via cblas)
4. **Optimizer**: AdamW with gradient accumulation and cosine LR decay

### fp16 Stability

ANE computes in fp16. DeepNet scaling (`res_alpha = 1/sqrt(2*N_layers)`) keeps activations bounded:
- Without scaling: activations overflow to inf at layer ~20
- With scaling: stable training for 32+ layers, activations in [-6, +7] range

## Autoresearch Integration

AutoANE implements the [karpathy/autoresearch](https://github.com/karpathy/autoresearch) protocol: an AI agent autonomously optimizes a training configuration through a keep/revert loop.

### How It Works

The agent follows `training/program.md`:

1. Creates a git branch (`autoresearch/<tag>`)
2. Modifies `training/train.py` (the ONLY mutable file)
3. Commits, runs `python3 train.py` (compiles C binary → trains for N seconds)
4. Parses `val_loss` from output
5. **Keep** if val_loss improved, **revert** (`git reset --hard HEAD~1`) if not
6. Loops forever until human interrupts (~30 experiments/hour at 2min budget)

Results are tracked in `training/results.tsv` (commit, val_loss, status: keep/discard/crash).

### Two Modes

| Mode | Script | Use Case |
|------|--------|----------|
| **Agent Loop** | `program.md` + `train.py` | Autonomous Karpathy-style keep/revert via Claude Code |
| **Grid Search** | `autoresearch.py` | Pre-defined architecture/LR sweeps, no agent needed |

### Key Finding

**"In a fixed time window, more optimizer steps beats more parameters."**

At 120s: 512d/4L with SEQ=128 (36.4M params, 4074 steps, val_loss **3.507**) beats 1024d/4L (95.4M params, 1050 steps, val_loss 4.30). The smaller model with shorter sequences gets 3.9× more gradient updates and wins decisively. This advantage **widens** at longer budgets (V21).

### Agent-Editable Hyperparameters

```python
# Architecture (E39: 512d/4L optimal for ≤10min budgets)
DEPTH = 4           # transformer layers
DIM = 512           # model dimension
HIDDEN = 1408       # FFN hidden dim (2.75× DIM)
HEADS = 8           # query attention heads
KV_HEADS = 2        # key/value heads (GQA)
HEAD_DIM = 64       # per-head dimension
SEQ = 128           # sequence length (E43: shorter = more steps)

# Training (E43: LR=4e-4 optimal at SEQ=128)
LR = 4e-4           # peak learning rate
WARMUP_STEPS = 100  # linear warmup
ACCUM_STEPS = 10    # gradient accumulation
GRAD_CLIP = 1.0     # gradient norm clipping
```

## Project Structure

```
AutoANE/
├── training/
│   ├── train.m              # Main training binary (Obj-C, 1571 lines)
│   ├── train.py             # Agent-editable config (the ONLY mutable file)
│   ├── program.md           # Karpathy-style agent protocol
│   ├── autoresearch.py      # Grid search orchestrator
│   ├── run_experiment.sh    # Experiment runner (compile + run + log)
│   ├── train_config.h       # Default hyperparameters (#ifndef guards)
│   ├── mil_dynamic.h        # MIL kernel generator (10 kernels/layer)
│   ├── io.h                 # IOSurface I/O, weight staging
│   ├── config.h             # Derived sizes, memory allocation
│   ├── cpu_ops.h            # RMSNorm, residual, loss, AdamW
│   ├── Makefile             # Build system
│   ├── experiments.jsonl    # Grid search results (JSON lines)
│   ├── results.tsv          # Agent loop results (keep/discard/crash)
│   └── models/              # Model configurations
├── tools/
│   ├── hf_to_ane.py         # HuggingFace -> ANE weight converter
│   ├── gguf_to_ane.py       # GGUF -> ANE weight converter
│   └── download_data.sh     # Training data download
├── bridge/
│   ├── ane_bridge.h         # C-callable ANE API
│   ├── ane_bridge.m         # Bridge implementation
│   └── Makefile
└── docs/
    ├── VERIFICATION.md      # First-principles verification of all claims
    ├── EXPERIMENTS.md       # Full experiment log (E1-E43)
    ├── ASSUMPTIONS.md       # Tracked assumptions with evidence
    └── RESEARCH_PLAN.md     # Research roadmap and findings
```

## Research Results

43 experiments across architecture search, LR sweeps, budget scaling, ANE characterization, and autonomous agent loop. All results independently verified ([docs/VERIFICATION.md](docs/VERIFICATION.md)).

### Key Findings

| # | Finding | Evidence | Literature |
|---|---------|----------|-----------|
| 1 | Smaller models win at fixed time budgets | 512d/4L (val 3.54) beats 1024d/4L (val 4.30) at 120s | Consistent with Chinchilla in data-constrained regime |
| 2 | Step count dominates model capacity | 2500 steps × 36M > 1050 steps × 95M | Kaplan: data β > param α |
| 3 | Advantage widens at longer training | Gap grows 0.15→0.29 from 120s→600s | Muennighoff: repeated data effective to ~4 epochs |
| 4 | CPU-only beats ANE for all tested sizes | IOSurface overhead negates ANE speedup | Novel finding (no prior ANE training comparisons) |
| 5 | Depth aids generalization but costs throughput | 4L beats 2L at same step count but loses at same wall-clock | Nguyen & Salazar (regime-specific) |
| 6 | LR scales with sqrt(model size) | 5e-4 for 36M, 3e-4 for 73M | Consistent with LAMB heuristic |
| 7 | Shorter sequences trade context for throughput | SEQ=128 gives 1.75× steps vs SEQ=256, val improves | Throughput-dominated regime |
| 8 | Optimal LR co-varies with batch size | Halving SEQ → LR 5e-4→4e-4 optimal | Smith et al. (2018) linear scaling rule |

### Best Configuration

```
512d/4L, SEQ=128, LR=4e-4, CPU-only, 120s budget
→ val_loss 3.507, 4074 steps, 24.2ms/step, 36.4M params
→ At 1800s: val_loss 2.22 (still improving at ~5 epochs)
```

## Known Limitations

- **Attention gradients**: SDPA backward on ANE produces near-zero dq/dk/dv due to fp16 underflow. Training still works (embedding + FFN gradients flow), but attention layers learn slowly. CPU fp32 SDPA backward is planned.
- **Fine-tuning pretrained models**: DeepNet scaling (required for fp16 stability) is incompatible with pretrained weight magnitudes. Training from scratch works; fine-tuning is an active research area.
- **Sequence length**: Default SEQ=128 (optimal for throughput). SEQ=64 degrades due to insufficient context. Longer sequences (512+) increase SDPA backward overflow risk.
- **Private APIs**: Uses `_ANEClient`, `_ANECompiler` — undocumented Apple frameworks that could change.

## Roadmap

- [ ] CPU fp32 SDPA backward (fix attention gradient underflow)
- [ ] LoRA fine-tuning support (freeze base weights, train adapters)
- [ ] Mixed precision backward (ANE fp16 forward, CPU fp32 backward)
- [ ] Background training daemon (train while you work)
- [ ] GGUF/CoreML export pipeline
- [ ] Validation/eval on held-out data
- [ ] SimpleStories dataset support
- [ ] Longer sequence lengths

## Credits

- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch): the autonomous keep/revert experiment loop protocol. AutoANE adapts this for Apple Silicon with compiled C training and ANE-specific constraints.
- **[Manjeet Singh (maderix)](https://github.com/maderix)** — [ANE](https://github.com/maderix/ANE): the original reverse engineering of Apple's Neural Engine private APIs and first-ever training on ANE hardware.
- **Orion** — [arXiv:2603.06728](https://arxiv.org/abs/2603.06728): first open end-to-end system for ANE LLM training and inference, demonstrating delta compilation and LoRA adapter hot-swap.

## License

MIT — see [LICENSE](LICENSE)
