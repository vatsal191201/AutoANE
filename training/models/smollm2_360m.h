// smollm2_360m.h — SmolLM2-360M (HuggingFace) config for ANE dynamic training
// Architecture: llama, 32 layers, GQA 15Q/5KV heads
#pragma once

#define MODEL_NAME "SmolLM2-360M"
#define DIM 960
#define HIDDEN 2560
#define HEADS 15
#define KV_HEADS 5
#define HD 64
#define GQA_RATIO (HEADS/KV_HEADS)  // 3
#define Q_DIM (HEADS*HD)            // 960
#define KV_DIM (KV_HEADS*HD)        // 320
#define SEQ 256
#define NLAYERS 32
#define VOCAB 49152

#define DEFAULT_DATA_PATH "../tinystories_smollm2_data00.bin"
#define CKPT_PATH "ane_smollm2_360m_ckpt.bin"
