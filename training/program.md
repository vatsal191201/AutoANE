# ANE Autoresearch — Agent Protocol

## Overview
You are optimizing a transformer language model trained from scratch on Apple Neural Engine (ANE).
The model trains on TinyStories (children's stories) using fp16 ANE compute with fp32 CPU operations.
Your goal: minimize `final_loss` within a fixed 5-minute training budget.

## Architecture
- **Llama-family transformer**: RMSNorm → Attention (GQA + RoPE) → RMSNorm → SwiGLU FFN
- **ANE kernels**: 10 kernels compiled once, weights staged via IOSurface each step
- **fp16 compute**: all ANE matmuls in fp16, CPU accumulation in fp32
- **DeepNet scaling**: residual connections scaled by 1/sqrt(2*NLAYERS) for fp16 stability

## Setup
1. Create branch `autoresearch/<tag>`
2. Read this file + `train.py`
3. Initialize `results.tsv` with header
4. First run is ALWAYS baseline (unmodified train.py)

## Experiment Loop (LOOP FOREVER)
1. Check git status
2. Edit `train.py` hyperparameters section
3. `git commit`
4. Run: `python3 train.py > run.log 2>&1`
5. Parse: `grep "^final_loss:" run.log`
6. If empty → crash. Check `tail -50 run.log`. Fix or abandon.
7. Log to results.tsv
8. If final_loss improved → keep commit
9. If final_loss equal or worse → `git reset` back

## results.tsv Format
```
commit	final_loss	status	description
a1b2c3d	7.123456	keep	baseline
```

## What You Can Edit (in train.py)
ONLY modify values in the `AGENT-EDITABLE HYPERPARAMETERS` section:

### Architecture (triggers recompile ~2s)
- `DEPTH`: number of layers (more depth = more capacity but slower)
- `DIM`: model dimension (must equal HEADS * HEAD_DIM)
- `HIDDEN`: FFN hidden dim (typically 2-4x DIM)
- `HEADS`: query heads (more = finer attention but more memory)
- `KV_HEADS`: key/value heads (GQA ratio = HEADS/KV_HEADS)
- `HEAD_DIM`: per-head dimension (typically 64 or 128)
- `SEQ`: sequence length (longer = more context but slower)

### Training
- `LR`: peak learning rate
- `WARMUP_STEPS`: linear LR warmup
- `ACCUM_STEPS`: gradient accumulation (effective batch = ACCUM * SEQ tokens)
- `GRAD_CLIP`: gradient norm clipping
- `WEIGHT_DECAY`: AdamW weight decay

### Constraints
- `DIM` must equal `HEADS * HEAD_DIM`
- `HEADS` must be divisible by `KV_HEADS`
- Total memory ~3x model params in float32 (weights + adam m + adam v)
- Keep total params under ~500M for reasonable training speed on ANE

## What You CANNOT Edit
- `VOCAB`, `DATA_PATH` (data is fixed)
- `TIME_BUDGET` (fixed at 300 seconds)
- The C training code (train.m, mil_dynamic.h, etc.)
- The data preparation pipeline

## Tips
- More depth with smaller width often beats shallow+wide at same param count
- GQA ratio of 3 (e.g., 15Q/5KV) works well for efficiency
- Higher LR with warmup can speed convergence
- gradient accumulation increases effective batch size without more memory
- fp16 ANE can overflow with very large activations — DeepNet scaling helps
- Each step processes SEQ tokens; more steps in budget = more data seen
