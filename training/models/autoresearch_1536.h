// autoresearch_1536.h — 4L-1536d scaling experiment config
// Llama-style architecture with GQA, for E38 ANE scaling study
#pragma once

#define MODEL_NAME "autoresearch-4L-1536d"
#define DIM 1536
#define HIDDEN 4224
#define HEADS 24
#define KV_HEADS 6
#define HD 64
#define GQA_RATIO (HEADS/KV_HEADS)
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)
#define SEQ 256
#define NLAYERS 4
#define VOCAB 49152

#define DEFAULT_DATA_PATH "../tinystories_smollm2_data00.bin"
#define CKPT_PATH "ane_autoresearch_1536_ckpt.bin"
