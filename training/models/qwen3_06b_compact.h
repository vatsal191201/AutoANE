// qwen3_06b_compact.h — Qwen3-0.6B with compacted vocab for TinyStories
// Original vocab 151936 -> 18516 active tokens
#pragma once

#define MODEL_NAME "Qwen3-0.6B-compact"
#define DIM 1024
#define HIDDEN 3072
#define HEADS 16
#define KV_HEADS 8
#define HD 128               // explicit head_dim (NOT DIM/HEADS)
#define GQA_RATIO (HEADS / KV_HEADS)  // = 2
#define Q_DIM (HEADS * HD)            // = 2048
#define KV_DIM (KV_HEADS * HD)        // = 1024 (= DIM for this model)
#define SEQ 256
#define NLAYERS 28
#define VOCAB 18516

#define CKPT_PATH "ane_qwen3_06b_compact_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_qwen3_compact.bin"
