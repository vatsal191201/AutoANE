# ANE Autoresearch Agent Protocol

## Overview

You are optimizing a transformer language model trained **from scratch** on Apple Neural Engine (ANE).
The model trains on TinyStories (children's stories) using fp16 ANE compute with fp32 CPU accumulation.
Your goal: **minimize `final_loss`** within a fixed **60-second** training budget per experiment.

## System Architecture

- **Llama-family transformer**: RMSNorm, GQA attention with RoPE, SwiGLU FFN
- **ANE kernels**: 10 MIL kernels compiled once at startup, weights staged via IOSurface each step
- **fp16 compute**: all ANE matmuls in fp16, CPU accumulation in fp32
- **DeepNet scaling**: residual connections scaled by `1/sqrt(2*NLAYERS)` for fp16 stability
- **Loss scaling**: gradients scaled by 256x to prevent fp16 underflow in ANE backward pass
- **Gradient sanitization**: optional NaN/Inf cleanup (per Orion paper Bug #3 fix)
- **AdamW optimizer**: with bias correction, cosine LR schedule, linear warmup

## How to Run an Experiment

Use `run_experiment.sh` to compile and run training with custom hyperparameters:

```bash
# Default config (4 layers, dim=1024, lr=3e-4)
./run_experiment.sh

# Override specific hyperparameters via JSON
./run_experiment.sh '{"lr": "5e-4", "wd": "0.2", "accum": "5"}'

# Override architecture
./run_experiment.sh '{"nlayers": "8", "dim": "512", "heads": "8", "kv_heads": "4", "hd": "64", "hidden": "1408"}'

# Pass a config file
./run_experiment.sh my_config.json
```

Each experiment:
1. Compiles `train.m` with `-D` overrides from `train_config.h`
2. Runs training for exactly 60 seconds (`--time 60`)
3. Captures machine-parseable output (final_loss, total_seconds, num_steps, etc.)
4. Appends result as a JSON line to `experiments.jsonl`
5. Prints result to stdout

## What You Can Modify

All hyperparameters are set via JSON keys passed to `run_experiment.sh`.

### Architecture (triggers full recompile + new random init)

| Key | Default | Description |
|-----|---------|-------------|
| `dim` | 1024 | Model embedding dimension |
| `nlayers` | 4 | Number of transformer layers (depth) |
| `heads` | 16 | Query attention heads |
| `kv_heads` | 4 | Key/value heads (GQA: heads/kv_heads groups) |
| `hd` | 64 | Per-head dimension (dim must equal heads * hd) |
| `hidden` | 2816 | SwiGLU FFN hidden dimension |
| `seq` | 256 | Sequence length |

### Training Hyperparameters

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | 3e-4 | Peak learning rate (cosine schedule) |
| `adam_b1` | 0.9 | Adam beta1 (momentum decay) |
| `adam_b2` | 0.95 | Adam beta2 (squared gradient decay) |
| `adam_eps` | 1e-8 | Adam epsilon (numerical stability) |
| `wd` | 0.1 | AdamW weight decay |
| `accum` | 10 | Gradient accumulation steps (effective batch = accum * seq tokens) |
| `warmup` | 100 | Linear LR warmup steps |
| `grad_clip` | 1.0 | Global gradient norm clipping threshold |
| `loss_scale` | 256.0 | fp16 gradient scaling factor |
| `min_lr_frac` | 0.1 | Minimum LR as fraction of peak (cosine floor) |

### Boolean Flags

| Key | Default | Description |
|-----|---------|-------------|
| `cpu_attn_bwd` | false | Use CPU fp32 for SDPA backward (more accurate, slower) |
| `sanitize` | false | Replace NaN with 0, clip Inf to +/-65504 in gradients |

## Architecture Constraints

- `dim` must equal `heads * hd`
- `heads` must be divisible by `kv_heads`
- Total memory is approximately 3x model params in float32 (weights + Adam m + Adam v)
- Keep total params under ~500M for reasonable step throughput on ANE
- Larger models get fewer training steps in the 60s budget

## What You CANNOT Modify

- `vocab` (49152, fixed by SmolLM2 tokenizer)
- `data_path` (TinyStories dataset, fixed)
- The C training code (`train.m`, `mil_dynamic.h`, `cpu_ops.h`, `io.h`, `config.h`)
- The kernel compilation pipeline
- The 60-second time budget

## Reading Results

Results are appended as JSON lines to `experiments.jsonl`:

```json
{"timestamp": "2026-03-11T10:30:00Z", "config": {"lr": "5e-4", "accum": "5"}, "status": "ok", "final_loss": 7.123456, "training_seconds": 55.2, "total_seconds": 58.1, "total_tokens_M": 3.5, "num_steps": 54, "num_params_M": 63.2, "depth": 4, "compile_seconds": 3, "time_budget": 60}
```

Key fields:
- `final_loss` (float): The metric to minimize. Lower is better.
- `num_steps` (int): How many gradient steps completed in the budget.
- `num_params_M` (float): Model size in millions of parameters.
- `total_tokens_M` (float): Total tokens processed.
- `status` (string): "ok", "compile_error", "crash", or "no_output".

Failed experiments have `"final_loss": null` and a non-ok status.

## Key Findings from Prior Research

### fp16 Precision Gap
ANE computes all matmuls in fp16. This introduces a precision gap vs. fp32 CPU training:
- Attention score computation loses precision with large head dimensions
- Gradient underflow in backward pass (mitigated by loss_scale=256)
- DeepNet residual scaling (`1/sqrt(2*NLAYERS)`) prevents activation overflow
- Very deep models (32+ layers) amplify fp16 rounding errors across layers

### Thermal Throttling
Apple Silicon throttles ANE under sustained load:
- First 30-60 seconds run at full speed
- Performance may degrade 10-20% after sustained workload
- Shorter experiments (60s) are less affected than long runs
- Step times reported in output help detect throttling mid-run

### Architecture vs. Hyperparameter Sensitivity
- **Depth vs. width tradeoff**: More layers with smaller width often beats shallow+wide at same param count, but deeper models get fewer steps in budget
- **GQA ratio**: 3-4x grouping (e.g., 15Q/5KV, 16Q/4KV) balances quality and throughput
- **Learning rate**: Higher LR (5e-4 to 1e-3) can speed convergence in short runs but risks fp16 overflow
- **Gradient accumulation**: Larger accum = larger effective batch = fewer steps but stabler gradients
- **Warmup**: Critical for stability with higher LR; 50-200 steps typical

### Practical Tips
- In 60 seconds, a 4-layer 1024d model gets ~50-80 steps depending on accum
- Reducing accum from 10 to 5 doubles the number of weight updates (more data seen)
- Very small models (2L, 512d) train fast but may underfit
- The step counter includes accumulation microsteps; actual weight updates = steps / accum
- Sequence length 256 is the default; shorter sequences are faster but see less context

## Experiment Loop (Recommended Workflow)

1. **Baseline**: Run with no config to establish baseline loss
2. **Sweep one variable at a time**: Change lr, accum, or architecture individually
3. **Read `experiments.jsonl`** to compare results across runs
4. **Combine best findings**: Once you identify individually good settings, combine them
5. **Iterate**: Keep running experiments, using results to guide next choices

## Files

| File | Purpose |
|------|---------|
| `train.m` | C/Obj-C training loop (DO NOT MODIFY) |
| `train_config.h` | Default hyperparameters with #ifndef guards |
| `run_experiment.sh` | Experiment runner (compile + run + log) |
| `experiments.jsonl` | Accumulated experiment results |
| `config.h` | Model-agnostic structs and ANE init |
| `mil_dynamic.h` | MIL kernel generation for ANE |
| `cpu_ops.h` | CPU fallback operations (RMSNorm, softmax, etc.) |
| `io.h` | IOSurface read/write helpers |
| `models/*.h` | Pre-defined model architectures |
| `program.md` | This file (agent instructions) |
