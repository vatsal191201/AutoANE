# AutoANE Autoresearch Agent Protocol

> Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for Apple Neural Engine training.

## Overview

You are an autonomous ML research agent optimizing a transformer language model trained on Apple Silicon. Your goal: **minimize `val_loss`** through iterative modification of `train.py`, using a git-based keep/revert loop.

You run experiments in a tight loop: modify → commit → train → evaluate → keep or revert. You do NOT pause to ask the human. The loop runs until the human interrupts you.

## Setup (Run Once)

1. Read these files to understand the system:
   - `README.md` (project overview)
   - `training/program.md` (this file — your protocol)
   - `training/train.py` (the ONLY file you modify)
   - `docs/ASSUMPTIONS.md` (verified findings from prior research)

2. Create a fresh git branch:
   ```bash
   git checkout -b autoresearch/<tag>   # e.g., autoresearch/mar11
   ```

3. Verify training data exists:
   ```bash
   ls -la tinystories_smollm2_data00.bin   # should be ~40MB
   ```

4. Initialize `training/results.tsv`:
   ```bash
   echo "commit\tval_loss\tsteps\tparams_M\tms_step\tstatus\tdescription" > training/results.tsv
   ```

5. Run baseline experiment (no changes):
   ```bash
   cd training && python3 train.py
   ```
   Record the baseline val_loss. This is your "best so far."

6. Confirm to the human that setup is complete and the loop is starting.

## The Experiment Loop

Repeat forever:

### 1. Propose a Change

Examine `train.py` and decide what to modify. You may change ONLY the `AGENT-EDITABLE HYPERPARAMETERS` section:

**Architecture** (triggers recompile + new random init):
- `DEPTH` — transformer layers (2-12 typical)
- `DIM` — model dimension (256-2048)
- `HIDDEN` — FFN hidden dim (usually 2.75× DIM, rounded to 64)
- `HEADS` — query attention heads (DIM / HEAD_DIM)
- `KV_HEADS` — key/value heads (HEADS must be divisible by KV_HEADS)
- `HEAD_DIM` — per-head dimension (64 typical)
- `SEQ` — sequence length (128-512)

**Training hyperparameters** (no recompile needed if architecture unchanged):
- `LR` — peak learning rate
- `WARMUP_STEPS` — linear warmup steps
- `ACCUM_STEPS` — gradient accumulation (effective batch = ACCUM_STEPS × SEQ tokens)
- `GRAD_CLIP` — gradient norm clipping
- `WEIGHT_DECAY` — AdamW weight decay
- `ADAM_B1`, `ADAM_B2` — Adam momentum parameters

**Training mode**:
- `CPU_ONLY` — pure CPU fp32 (recommended default, see V15)
- `ANE_MATMUL_ONLY` — ANE for linear projections only (viable up to DIM=1536)
- `LOSS_SCALE` — fp16 gradient scaling (only for ANE modes)

**Budget**:
- `TIME_BUDGET` — wall-clock seconds per experiment

### 2. Commit

```bash
git add training/train.py
git commit -m "<short description of what you changed and why>"
```

### 3. Run

```bash
cd training && python3 train.py
```

The script generates a C header, compiles the binary (~2s), and trains for `TIME_BUDGET` seconds. Output ends with machine-parseable metrics:
```
final_loss:       3.348118
val_loss:         3.543304
training_seconds: 104.6
total_seconds:    120.0
num_steps:        2542
num_params_M:     36.4
```

### 4. Evaluate

Parse `val_loss` from the output. This is the metric to minimize.

### 5. Keep or Revert

- **If val_loss IMPROVED** (lower than best so far):
  ```bash
  # Keep the commit. Update best.
  echo "<commit>\t<val_loss>\t<steps>\t<params_M>\t<ms_step>\tkeep\t<description>" >> training/results.tsv
  ```
  Update your "best so far" val_loss.

- **If val_loss DID NOT IMPROVE** (equal or higher):
  ```bash
  # Revert the commit. Go back to previous state.
  git reset --hard HEAD~1
  echo "<commit>\t<val_loss>\t<steps>\t<params_M>\t<ms_step>\tdiscard\t<description>" >> training/results.tsv
  ```

- **If the run CRASHED** (compile error, runtime crash, NaN loss):
  ```bash
  git reset --hard HEAD~1
  echo "<commit>\tnull\t0\t0\t0\tcrash\t<description>" >> training/results.tsv
  ```
  Debug if obvious; otherwise skip and try a different change.

### 6. Go to Step 1

Do NOT pause to ask the human. Do NOT stop to analyze. Run the next experiment immediately. The loop runs until interrupted.

## What You CANNOT Modify

- `train.m` — the C/Obj-C training binary
- `mil_dynamic.h`, `cpu_ops.h`, `io.h`, `config.h` — kernel/ops code
- `train_config.h` — defaults (your changes go through `train.py`)
- `VOCAB` (49152, fixed by tokenizer)
- `DATA_PATH` (fixed dataset)

## Constraints

- `DIM` must equal `HEADS × HEAD_DIM`
- `HEADS` must be divisible by `KV_HEADS`
- Total memory ≈ 3× params in fp32 (weights + Adam m + Adam v). Keep under ~10GB.
- A small improvement that adds ugly complexity is not worth it.
- Simplicity wins. A clean, small change that works beats a complex one.

## Key Findings from Prior Research

Read `docs/ASSUMPTIONS.md` for the full registry. Critical findings:

### Architecture (E39, E40, E41)
- **Smaller/faster models win at fixed time budgets** (V16). At 120s, 512d/4L (val 3.54) crushes 1024d/4L (val 4.30).
- **Depth is strictly harmful at short budgets** (V17). More layers = slower steps = fewer updates.
- **Optimal LR scales with model size** (V18). 5e-4 for 512d, 3e-4 for 1024d+.
- **512d/4L advantage widens at longer budgets** (V21). No crossover through 600s.
- **Data volume is the bottleneck, not model capacity** (V22). 20M token dataset limits all models.

### ANE-Specific
- **CPU-only is correct default for all model sizes** (V15). ANE never faster with dynamic weight approach.
- **IOSurface overhead scales dramatically** at DIM>1536 (V14). Don't go there.
- **ANE matmul-only mode matches CPU at small dimensions** (V13). Only use for DIM≤1536.
- **fp16 precision gap is real (~16%)** (SA1). Not fixable by software.

### Training Dynamics
- **2-layer models overfit heavily** (V20). Train-val gap +0.83 at optimal LR.
- **4 layers is the sweet spot** for generalization at our data scale.
- **Loss scaling 256.0 essential** for ANE fp16 modes (V6).

## Strategy Guidance

**Promising directions** (based on research gaps):
1. Sequence length variation (SEQ: 128 vs 256 vs 512) — untested
2. Weight decay tuning (0.05-0.3) — shallow models may benefit from higher WD
3. Warmup reduction (50 vs 100) — current 100 may be suboptimal at high LR
4. Gradient accumulation (5 vs 10 vs 20) — trades batch size vs step count
5. Learning rate schedules above 5e-4 (7e-4, 1e-3) — risky but could work for small models
6. `ADAM_B2` variation (0.95 vs 0.99) — affects gradient noise

**Dead ends** (don't waste experiments):
- DIM > 1024 with ANE modes (IOSurface scaling ceiling)
- Deep models (8+ layers) at short budgets
- Large models (>100M params) with 20M token dataset
- `LOSS_SCALE` changes in CPU-only mode (irrelevant)
- `ANE_MATMUL_ONLY` at DIM > 1536

## Differences from karpathy/autoresearch

| Aspect | Karpathy | AutoANE |
|--------|----------|---------|
| Hardware | NVIDIA H100 GPU | Apple Neural Engine / Apple Silicon CPU |
| Training code | Python (train.py IS the model) | Objective-C (train.m, not modifiable) |
| Agent modifies | train.py (model + optimizer + loop) | train.py (hyperparameters + architecture only) |
| Metric | val_bpb (bits per byte) | val_loss (cross-entropy in nats) |
| Budget | 5 min (GPU) | 2 min default (CPU, configurable) |
| Optimizer | Muon + AdamW | AdamW only |
| Data | Not specified in README (large-scale) | TinyStories (20M tokens) |
| Scope of changes | Arbitrary code modifications | Hyperparameter + architecture search |

The core protocol (modify → commit → train → keep/revert) is identical. The scope of modification is narrower because our training binary is compiled C, not interpreted Python.

## Files

| File | Purpose |
|------|---------|
| `train.py` | **The ONLY file you modify.** Hyperparameters + architecture config. |
| `train.m` | C/Obj-C training loop (READ ONLY) |
| `train_config.h` | Default hyperparameters with #ifndef guards (READ ONLY) |
| `run_experiment.sh` | Low-level experiment runner (used by autoresearch.py, not the agent loop) |
| `autoresearch.py` | Grid search orchestrator (alternative to agent loop) |
| `experiments.jsonl` | Results from grid search runs |
| `results.tsv` | Results from agent loop runs (keep/discard/crash) |
| `program.md` | This file (your instructions) |
