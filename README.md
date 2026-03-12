# AutoANE

**Training transformers on Apple's Neural Engine: an empirical study**

AutoANE is an open-source system for training Llama-family transformers on Apple Silicon, including the first full forward/backward pass on the Neural Engine (ANE) via reverse-engineered private APIs. It pairs a compiled Objective-C training binary (1571 lines) with an autonomous hyperparameter search loop adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

43 controlled experiments, a 100-experiment autonomous search, and first-principles verification produced three central findings:

1. **ANE has a genuine 2.5x matmul speedup over CPU** (Apple AMX/Accelerate), but IOSurface weight-staging overhead negates this advantage end-to-end. CPU-only training wins at every tested model size (36M-281M params).

2. **ANE does not save power.** Direct measurement via `powermetrics` shows package power of 13.3W (CPU-only), 12.6W (ANE matmul), and 12.7W (ANE full). This is the first published power measurement of ANE training workloads.

3. **In a fixed time window, more optimizer steps beats more parameters.** 512d/4L (36M params, 4074 steps at 120s) achieves val_loss 3.51 while 1024d/4L (95M params, 1050 steps) achieves 4.30. This confirms Kaplan et al.'s observation that the data scaling exponent (0.095) exceeds the parameter exponent (0.076) in the severely data-constrained regime.

Full experiment log: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Verification report: [docs/VERIFICATION.md](docs/VERIFICATION.md) | Assumption registry: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Technical report: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## Reproducing Results

**Requirements**: macOS 15+, Apple Silicon (M1/M2/M3/M4), Xcode Command Line Tools.

```bash
git clone https://github.com/vatsal191201/AutoANE.git && cd AutoANE

# Download training data (TinyStories, 40MB, 20M tokens, SmolLM2 BPE tokenizer)
bash tools/download_data.sh

# Train from scratch (120s, CPU-only, best known config)
cd training && python3 train.py

# Expected output:
#   val_loss:  3.288    (±0.05, run-to-run variance from random init)
#   num_steps: ~4100
#   36.4M params, 512d/4L, SEQ=128

# Generate text from trained checkpoint
cd .. && python3 generate.py --prompt "Once upon a time" --tokens 200

# Run autonomous hyperparameter search (88 experiments, ~3 hours)
cd training && python3 run_autosearch.py --experiments 100
```

All experiments use the same 40MB binary file (`tinystories_smollm2_data00.bin`): 20M tokens, SmolLM2 BPE vocabulary (49152 tokens, 16893 active in this dataset), 90/10 train/val split. Every model trains from random initialization (Xavier/Kaiming) unless stated otherwise.

---

## Novel Contributions

We distinguish what is novel from what is adapted from prior work.

**Novel (no prior published equivalent):**
- First quantitative ANE vs CPU training comparison at matched configurations (throughput, loss quality, power)
- First ANE training power measurement (macOS `powermetrics`, 60s per mode, idle-subtracted)
- First IOSurface scaling study across model dimensions (DIM 1024/1536/2048) — demonstrates hard memory pressure ceiling at ~220MB total IOSurface allocation
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

At 120s on Apple Silicon (M4), 512d/4L (36.4M params) gets 2542 steps at 42ms/step, achieving val_loss 3.54. Meanwhile 1024d/4L (95.4M params) gets only 1050 steps at 99ms/step, achieving val_loss 4.30. The smaller model gets 2.4x more gradient updates and wins by 0.76 nats.

This advantage **widens** at longer budgets:

| Budget | 512d/4L val_loss | 768d/2L val_loss | Gap |
|--------|------------------|------------------|-----|
| 120s | 3.54 | 3.69 | 0.15 |
| 300s | 3.09 | 3.20 | 0.11 |
| 600s | 2.55 | 2.84 | 0.29 |

At 600s, the 512d/4L model has seen 1.57 epochs of data (31M token-encounters on a 20M token dataset). It is still underfitting. At 1800s (4.89 epochs), it reaches val_loss 2.22, consistent with the TinyStories literature baseline of ~2.0 at convergence for 33M-param models (Eldan & Li, 2023).

**Interpretation**: Kaplan et al. (2020) showed loss scales as L(D) ~ D^(-0.095) for data and L(N) ~ N^(-0.076) for parameters. Since 0.095 > 0.076, each additional gradient step contributes more than each additional parameter. In our severely data-constrained regime (23-329x below Chinchilla optimal), this effect is amplified: the data exponent dominates.

Experiments: E39 (11-config architecture grid), E40 (per-architecture LR sweep), E41 (budget scaling 120-1800s), E42 (independent verification, all results reproduced within 0.3%).

### Finding 2: CPU-only training beats ANE at every tested dimension

| Mode | ms/step | Steps@120s | val_loss | Matmul time | IO overhead |
|------|---------|-----------|----------|-------------|-------------|
| CPU-only fp32 | 102.2 | 1041 | 4.20 | 71.9ms | 0.2ms |
| ANE fp16 | 68.7 | 1297 | 4.69 | 29.2ms | 8.1ms |
| ANE matmul-only | ~71 | 1149 | ~4.65 | ~30ms | ~8ms |

(All values for 1024d/4L, 95.4M params, loss_scale=256. Loss values are 200-step rolling averages.)

ANE achieves 2.46x faster raw matmuls (29.2ms vs 71.9ms via cblas_sgemm). But the end-to-end picture is different: IOSurface weight staging costs 8.1ms/step (fp32->fp16 conversion + spatial packing via NEON), and fp16 precision degrades loss by ~16% (irreducible — tested via clamping, LR tuning, weight decay tuning). CPU fp32 produces better models despite fewer steps.

At larger dimensions, ANE gets *worse*:

| DIM | IOSurface size | ANE vs CPU | Reason |
|-----|---------------|------------|--------|
| 1024 | 60MB | 1.5x faster (step) | IOSurface fits L2 cache |
| 1536 | 220MB | Parity | Cache pressure begins |
| 2048 | 379MB | 2x *slower* | Cache thrashing in all operations |

Experiments: E11 (CPU vs ANE comparison), E36 (matmul-only mode), E38 (dimension scaling study, novel).

### Finding 3: ANE does not save power

Measured via `sudo powermetrics --samplers cpu_power,gpu_power,ane_power` for 60 seconds per mode:

| Mode | CPU Power | ANE Power | Package Power | Energy/step |
|------|-----------|-----------|---------------|-------------|
| Idle | — | — | 8455 mW | — |
| CPU-only | 13241 mW | 9 mW | 13273 mW | 9.2 mJ |
| ANE matmul | 12132 mW | 384 mW | 12568 mW | 10.9 mJ |
| ANE full | 11821 mW | 765 mW | 12664 mW | 9.7 mJ |

Package power is ~12.5-13.3W across all modes. ANE shifts ~1.4W from CPU to ANE subsystem, but total power is unchanged. CPU-only achieves the lowest energy per step (9.2 mJ) because it completes more steps in the same time at the same power draw.

Orion emphasizes ANE as "dedicated silicon that's idle in most workloads" but does not quantify power efficiency for training. Our measurement shows no power benefit for training workloads.

Experiment: E12.

### Finding 4: fp16 precision gap is irreducible

ANE computes in fp16. The loss gap vs fp32 is ~16% and cannot be closed:

| Approach tried | Result | Why it fails |
|----------------|--------|-------------|
| Activation clamping [-4, +4] | No change | Precision loss is from matmul accumulation (sqrt(DIM) ULP rounding per dot product), not magnitude |
| Lower LR (1e-4) | Worse | Underfitting |
| Higher weight decay (0.3) | No change | WD doesn't affect per-step precision |
| Loss scaling (256x) | Required but doesn't close gap | Prevents underflow but doesn't improve accumulation precision |
| DeepNet scaling | Required for stability | Prevents overflow but gap remains |

At DIM=1024, each fp16 dot product accumulates sqrt(1024) = 32 ULPs of rounding error. This is a hardware limitation of fp16 MAC units.

Experiments: E14 (clamping), E15 (LR/WD tuning), E19 (extended training with zero NaN/Inf confirms gap is genuine).

### Finding 5: Fusing non-linear ops into ANE hurts generalization

| Mode | Train-val gap | Description |
|------|--------------|-------------|
| ANE full (fused) | 1.22 | RoPE, attention scores, SiLU all in fp16 |
| ANE matmul-only | 0.60 | Only 7 linear projections per layer on ANE, everything else on CPU fp32 |
| CPU-only | 0.60 | All fp32 |

ANE matmul-only achieves identical val_loss to CPU at matched step counts (to 4 decimal places). The precision problem is specifically in non-linear operations (softmax, SiLU, RoPE) where fp16 error compounds. Linear projections tolerate fp16 because error doesn't compound across the accumulation.

Experiment: E36 (novel finding — first demonstration that selective ANE offloading matches CPU quality).

### Finding 6: Delta compilation does not work

Tested 5 approaches to avoid recompiling ANE kernels when weights change:

1. Unload -> write new BLOBFILE -> reload: **Output unchanged** (ANE caches compiled binary)
2. tmpDir weight patching: **Output unchanged**
3. e5bundlecache investigation: Only small metadata (~96 bytes per entry)
4. _ANEInMemoryModel reload: **API not functional**
5. Fresh recompile with cached graph: ~60ms/kernel (too expensive)

ANE loads from an inaccessible memory-mapped compiled binary, not from the source BLOBFILE. Orion claims 8.5x faster recompilation via unload/reload (494ms for 60 kernels vs 4200ms full compile). We could not reproduce this — our reload attempts returned unchanged outputs.

Experiments: E10, E17 (5 approaches, all fail), E34 (re-confirmed with newer APIs).

### Finding 7: Autonomous search finds 17% improvement

100 experiments via `run_autosearch.py` (random perturbation with keep/revert, no AI agent):

| Experiment | Change | val_loss | Cumulative improvement |
|------------|--------|----------|----------------------|
| Baseline | — | 3.952 | — |
| Keep 1 | HIDDEN 1408->1152, WARMUP 100->71 | 3.906 | 1.2% |
| Keep 2 | WEIGHT_DECAY 0.1->0.098 | 3.676 | 7.0% |
| Keep 3 | LR 4e-4->4.19e-4 | 3.671 | 7.1% |
| Keep 5 | ADAM_B2 0.95->0.959, LR->6.34e-4 | 3.505 | 11.3% |
| Keep 15 | WEIGHT_DECAY 0.098->0.076 | 3.480 | 11.9% |
| Keep 87 | ACCUM_STEPS 10->7 | 3.288 | 16.8% |

Best configuration: `512d/4L, SEQ=128, LR=6.34e-4, ACCUM=7, WD=0.076, ADAM_B2=0.959`.

Diminishing returns after ~60 experiments — 5 of 6 improvements found in the first 30. The final improvement (ACCUM 10->7) at experiment 87 was the only late-stage discovery.

### Additional findings

| # | Finding | Evidence |
|---|---------|----------|
| 8 | Depth hurts at every width tested (120s budget) | E39: 512d 4L->8L costs +0.61 val_loss |
| 9 | 2-layer models overfit despite high throughput | E40: 768d/2L train-val gap +0.83 |
| 10 | Optimal LR scales with sqrt(model size) | E40: 5e-4 for 36M, 3e-4 for 73M |
| 11 | SEQ=128 is optimal (throughput > context at 120s) | E43: 1.75x steps vs SEQ=256, val improves |
| 12 | Optimal LR co-varies with batch size | E43: halving SEQ requires LR 5e-4->4e-4 (Smith et al. linear scaling rule) |
| 13 | Minimum useful sequence length is ~128 for TinyStories | E43: SEQ=64 catastrophically worse despite 40% more steps |
| 14 | Thermal throttling is 30% at 10 min (single process) | E24: E18's "50% throttling" was from concurrent processes |
| 15 | Neither CPU nor ANE diverges at 10+ minutes (clean system) | E37: 5777 steps stable for both modes |

---

## System Architecture

### Training Binary (`training/train.m`, 1571 lines, Objective-C)

The training loop is a compiled binary, not Python. `train.py` generates a C header file from its hyperparameters, invokes `make`, and runs the resulting binary. This separation means the agent can only modify hyperparameters and architecture — not the optimizer, loss function, or data pipeline.

### ANE Kernel Pipeline

10 MIL (Machine Learning Intermediate Language) kernels compiled per layer at startup:

| Kernel | Direction | Operation |
|--------|-----------|-----------|
| sdpaFwd | Forward | Scaled dot-product attention |
| woFwd | Forward | Output projection (GQA-aware) |
| ffnFused | Forward | SwiGLU FFN with residual |
| sdpaBwd1/2 | Backward | Attention backward (two-pass) |
| qBwd | Backward | Query projection backward |
| kvBwd | Backward | Key/value projection backward |
| wotBwd | Backward | Output projection backward |
| ffnBwdW2t | Backward | FFN W2 backward |
| ffnBwdW13t | Backward | FFN W1/W3 backward |

Weights are staged via IOSurface spatial dimensions at each step: the compiled MIL program is fixed, and only the weight data is patched via NEON-vectorized fp32->fp16 conversion. This avoids the 4.2s recompilation cost per step that static compilation would require.

### Training Modes

| Mode | Flag | Precision | Best for |
|------|------|-----------|----------|
| **CPU-only** (recommended) | `--cpu-only` | fp32 | All training |
| ANE matmul-only | `--ane-matmul-only` | fp16 matmul + fp32 ops | ANE research |
| ANE full | *(default)* | fp16 | ANE characterization |

### Training Features

- **AdamW** with bias correction, cosine LR decay (min_lr = 0.1 * max_lr), gradient clipping
- **Gradient accumulation** (effective batch = ACCUM_STEPS x SEQ tokens)
- **DeepNet scaling** for fp16 stability in ANE modes
- **LoRA fine-tuning** (`--lora --lora-rank 8`)
- **Time-budgeted training** (`--time 120`)
- **Checkpointing** (BLZT v4 format: 96-byte header, per-layer weights + Adam state)

### Weight Conversion Pipeline

```
HuggingFace (safetensors) <-> ANE checkpoint (BLZT v4) <-> GGUF (llama.cpp)
       hf_to_ane.py              export_to_gguf.py / gguf_to_ane.py
```

- **RoPE convention**: ANE uses paired interleaving `[re0, im0, re1, im1, ...]`; HuggingFace/GGUF uses split halves `[re0, re1, ..., im0, im1, ...]`. Converters handle this automatically.
- **Round-trip verified**: ANE -> GGUF -> ANE produces bit-perfect results on all 38 tensors (512d/4L) and all 272 tensors (SmolLM2-135M).
- **GGUF export** includes full tokenizer metadata (49152 BPE tokens, 48900 merge rules) for llama.cpp compatibility. Generates at 316.5 t/s in llama-cli.

### Autonomous Search

Two modes:

| Mode | Script | How it works |
|------|--------|-------------|
| Agent loop | `program.md` + `train.py` | LLM agent (Claude Code) modifies train.py, commits, trains, keeps/reverts |
| Automated search | `run_autosearch.py` | Random perturbation with keep/revert, no AI agent needed |

Both produce `results.tsv` (commit, val_loss, steps, params, status, description).

---

## Project Structure

```
AutoANE/
├── training/
│   ├── train.m              # Training binary (Obj-C, 1571 lines, READ ONLY)
│   ├── train.py             # Agent-editable hyperparameters (the ONLY mutable file)
│   ├── program.md           # Karpathy-style agent protocol
│   ├── run_autosearch.py    # Autonomous search (no AI agent)
│   ├── autoresearch.py      # Grid search orchestrator
│   ├── run_experiment.sh    # Single experiment runner
│   ├── mil_dynamic.h        # MIL kernel generator (10 kernels/layer)
│   ├── io.h                 # IOSurface I/O, weight staging, fp32->fp16
│   ├── cpu_ops.h            # RMSNorm, RoPE, SDPA, loss, AdamW (CPU fp32)
│   ├── config.h             # Derived sizes, memory allocation
│   ├── train_config.h       # Default hyperparameters (#ifndef guards)
│   ├── Makefile             # Build: xcrun clang + Accelerate + IOSurface
│   ├── results.tsv          # Agent loop results
│   ├── experiments.jsonl    # Grid search results
│   └── models/              # Model header configs (8 architectures)
├── generate.py              # Text generation from checkpoint (numpy)
├── demo.sh                  # One-command demo: download + train + generate
├── tools/
│   ├── hf_to_ane.py         # HuggingFace -> ANE converter
│   ├── gguf_to_ane.py       # GGUF -> ANE converter
│   ├── export_to_gguf.py    # ANE -> GGUF exporter (with tokenizer)
│   ├── power_benchmark.sh   # Power measurement (requires sudo)
│   └── download_data.sh     # TinyStories download
├── bridge/
│   ├── ane_bridge.h         # C-callable ANE API
│   ├── ane_bridge.m         # Bridge implementation
│   └── Makefile
└── docs/
    ├── TECHNICAL_REPORT.md  # Full technical report
    ├── VERIFICATION.md      # First-principles verification (21 sections)
    ├── EXPERIMENTS.md       # Experiment log (E1-E43 + E44 autosearch)
    ├── ASSUMPTIONS.md       # 27 verified, 8 disproved, 13 unverified
    └── RESEARCH_PLAN.md     # Research roadmap with completed results
```

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

1. **Attention gradient underflow** (ANE modes): SDPA backward produces dq/dk magnitudes ~100x smaller than dv due to fp16 underflow. Training works (embedding + FFN gradients carry it), but attention layers learn slowly.

2. **DeepNet incompatibility with pretrained weights**: DeepNet scaling (required for fp16 stability) changes residual connection magnitudes, making fine-tuning from pretrained weights impractical in ANE modes. CPU-only mode works for fine-tuning.

3. **Single dataset**: All results are on TinyStories (20M tokens). Models are 23-329x below Chinchilla optimal data:parameter ratio. Larger datasets would likely change the optimal architecture.

4. **Private APIs**: Uses `_ANEClient`, `_ANECompiler` from `AppleNeuralEngine.framework` — undocumented and subject to change between macOS versions.

5. **Sequence length**: SEQ=128 is optimal for throughput but limits context. SEQ=512+ increases SDPA backward overflow risk on ANE.

---

## Open Questions

1. Does the step-count advantage hold with larger datasets? Chinchilla predicts a crossover where larger models become optimal — but at what data scale?
2. Can `_ANEChainingRequest` (firmware-level chained execution) eliminate CPU round-trips between layers? This is unexplored and could fundamentally change the ANE throughput picture.
3. INT8 quantization halves IOSurface size — does this move the memory pressure ceiling from DIM=1536 to DIM=2048+?
4. Mixed precision (ANE fp16 forward, CPU fp32 backward) is theoretically sound but untested as a complete pipeline.

---

## Credits

- **[maderix](https://github.com/maderix)** — [ANE](https://github.com/maderix/ANE): original reverse engineering of Apple's Neural Engine private APIs and first-ever training on ANE hardware. AutoANE's bridge, IOSurface code, and MIL generation are direct adaptations.
- **[Andrej Karpathy](https://github.com/karpathy)** — [autoresearch](https://github.com/karpathy/autoresearch): the autonomous keep/revert experiment protocol. Agent edits a single `train.py` containing the full model, optimizer (Muon + AdamW), and training loop.
- **[Orion](https://github.com/mechramc/Orion)** (Murai Labs): ANE training/inference runtime. Claims 8.5x faster weight reload vs full compilation. Delta compilation findings from Orion motivated our investigation (which could not reproduce their reload mechanism).

## License

MIT — see [LICENSE](LICENSE)
