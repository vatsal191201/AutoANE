// autoresearch_2048.h — 4L-2048d scaling experiment config
// Llama-style architecture with GQA, for E38 ANE scaling study
#pragma once

#define MODEL_NAME "autoresearch-4L-2048d"
#define DIM 2048
#define HIDDEN 5632
#define HEADS 32
#define KV_HEADS 8
#define HD 64
#define GQA_RATIO (HEADS/KV_HEADS)
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)
#define SEQ 256
#define NLAYERS 4
#define VOCAB 49152

#define ROPE_THETA 10000.0f

#define DEFAULT_DATA_PATH "../tinystories_smollm2_data00.bin"
#define CKPT_PATH "ane_autoresearch_2048_ckpt.bin"
