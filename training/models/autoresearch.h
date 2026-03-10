#pragma once
#define MODEL_NAME "test-8L-512d"
#define DIM 512
#define HIDDEN 1536
#define HEADS 8
#define KV_HEADS 4
#define HD 64
#define GQA_RATIO (HEADS/KV_HEADS)
#define Q_DIM (HEADS*HD)
#define KV_DIM (KV_HEADS*HD)
#define SEQ 256
#define NLAYERS 8
#define VOCAB 49152
#define DEFAULT_DATA_PATH "../tinystories_smollm2_data00.bin"
#define CKPT_PATH "ane_autoresearch_ckpt.bin"
