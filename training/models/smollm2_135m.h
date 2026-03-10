// smollm2_135m.h — SmolLM2-135M (HuggingFace) config for ANE dynamic training
// Architecture: llama, 30 layers, GQA 9Q/3KV heads
#pragma once

#define MODEL_NAME "SmolLM2-135M"
#define DIM 576
#define HIDDEN 1536
#define HEADS 9
#define KV_HEADS 3
#define HD 64
#define GQA_RATIO (HEADS/KV_HEADS)  // 3
#define Q_DIM (HEADS*HD)            // 576
#define KV_DIM (KV_HEADS*HD)        // 192
#define SEQ 256
#define NLAYERS 30
#define VOCAB 49152

#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
#define CKPT_PATH "ane_smollm2_135m_ckpt.bin"
