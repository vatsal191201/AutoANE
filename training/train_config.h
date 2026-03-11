// train_config.h — Agent-modifiable training configuration for AutoANE autoresearch
// All values use #ifndef guards so they can be overridden at compile time via -D flags.
// Example: clang -DMAX_LR=5e-4 -DACCUM_STEPS=5 -DNLAYERS=8 ...
//
// Architecture defaults match the autoresearch model (4L-1024d).
// Training defaults match train.m hardcoded values.
#pragma once

// ===== Architecture (model dimensions) =====
// These define the transformer shape. Changing them requires recompilation
// and produces a different model architecture (new random init).

#ifndef DIM
#define DIM 1024               // Model embedding dimension
#endif

#ifndef NLAYERS
#define NLAYERS 4              // Number of transformer layers (depth)
#endif

#ifndef HEADS
#define HEADS 16               // Number of query attention heads
#endif

#ifndef KV_HEADS
#define KV_HEADS 4             // Number of key/value heads (GQA grouping)
#endif

#ifndef HD
#define HD 64                  // Per-head dimension (DIM must equal HEADS * HD)
#endif

#ifndef HIDDEN
#define HIDDEN 2816            // FFN hidden dimension (SwiGLU intermediate size)
#endif

#ifndef VOCAB
#define VOCAB 49152            // Vocabulary size (fixed by tokenizer, do not change)
#endif

#ifndef SEQ
#define SEQ 256                // Sequence length (tokens per training sample)
#endif

// Derived dimensions (do not override directly)
#ifndef GQA_RATIO
#define GQA_RATIO (HEADS/KV_HEADS)
#endif

#ifndef Q_DIM
#define Q_DIM (HEADS*HD)
#endif

#ifndef KV_DIM
#define KV_DIM (KV_HEADS*HD)
#endif

// Model name for logging (auto-generated from architecture)
#ifndef MODEL_NAME
#define MODEL_NAME "autoresearch"
#endif

#ifndef DEFAULT_DATA_PATH
#define DEFAULT_DATA_PATH "../tinystories_smollm2_data00.bin"
#endif

#ifndef CKPT_PATH
#define CKPT_PATH "ane_autoresearch_ckpt.bin"
#endif

// ===== Training hyperparameters =====
// These are defaults used in train.m main() and can be overridden
// either via -D flags at compile time or via --flag at runtime.
// Runtime flags (--lr, --accum, etc.) take precedence over compile-time -D values.

#ifndef MAX_LR
#define MAX_LR 3e-4f           // Peak learning rate (cosine schedule)
#endif

#ifndef ADAM_B1
#define ADAM_B1 0.9f           // Adam beta1 (momentum)
#endif

#ifndef ADAM_B2
#define ADAM_B2 0.95f          // Adam beta2 (RMS of gradients)
#endif

#ifndef ADAM_EPS
#define ADAM_EPS 1e-8f         // Adam epsilon (numerical stability)
#endif

#ifndef WD
#define WD 0.1f               // AdamW weight decay (applied to all non-RMSNorm params)
#endif

#ifndef ACCUM_STEPS
#define ACCUM_STEPS 10         // Gradient accumulation steps (effective batch = ACCUM * SEQ tokens)
#endif

#ifndef WARMUP_STEPS
#define WARMUP_STEPS 100       // Linear LR warmup steps before cosine decay
#endif

#ifndef GRAD_CLIP
#define GRAD_CLIP 1.0f         // Global gradient norm clipping threshold
#endif

#ifndef LOSS_SCALE
#define LOSS_SCALE 256.0f      // Loss scaling for fp16 ANE backward (prevents gradient underflow)
#endif

#ifndef MIN_LR_FRAC
#define MIN_LR_FRAC 0.1f      // Minimum LR as fraction of MAX_LR (cosine schedule floor)
#endif

// ===== Validation =====
// Compile-time checks for architecture constraints
#if (HEADS % KV_HEADS) != 0
#error "HEADS must be divisible by KV_HEADS"
#endif
