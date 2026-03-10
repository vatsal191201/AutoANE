#!/usr/bin/env python3
"""Convert HuggingFace model to ANE training checkpoint format.
Supports any llama-family architecture (llama, qwen2, qwen3, smollm2, etc.)
Outputs float32 binary checkpoint matching train.m's CkptHdr format (v4).

Usage: python hf_to_ane.py <hf_model_name> <output_ckpt.bin>
"""

import struct
import numpy as np
import sys
import os

def interleave_weights(W, n_heads, head_dim):
    """Convert Q/K weight matrix from HF (non-interleaved) to GGUF/ANE (interleaved) RoPE ordering.
    HF:   [re0, re1, ..., re31, im0, im1, ..., im31] per head
    ANE:  [re0, im0, re1, im1, ..., re31, im31] per head
    W shape: [n_heads * head_dim, dim]
    """
    W_out = np.zeros_like(W)
    half = head_dim // 2
    for h in range(n_heads):
        for i in range(half):
            src_re = h * head_dim + i
            src_im = h * head_dim + half + i
            dst_re = h * head_dim + 2 * i
            dst_im = h * head_dim + 2 * i + 1
            W_out[dst_re] = W[src_re]
            W_out[dst_im] = W[src_im]
    return W_out

def convert_hf_to_ane(model_name, output_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading HuggingFace model: {model_name}")
    config = AutoConfig.from_pretrained(model_name)

    # Extract architecture params
    dim = config.hidden_size
    hidden = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    hd = getattr(config, 'head_dim', dim // n_heads)
    q_dim = n_heads * hd
    kv_dim = n_kv_heads * hd
    vocab = config.vocab_size

    print(f"Architecture: {config.model_type}")
    print(f"  dim={dim} hidden={hidden} layers={n_layers}")
    print(f"  heads={n_heads}/{n_kv_heads} hd={hd} q_dim={q_dim} kv_dim={kv_dim}")
    print(f"  vocab={vocab}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    state = {k: v.float().numpy() for k, v in model.state_dict().items()}

    # Detect layer key prefix
    sample_key = next(k for k in state.keys() if 'layers.0' in k)
    layer_prefix = sample_key.split('layers.0')[0] + 'layers'

    # Write ANE checkpoint
    with open(output_path, 'wb') as f:
        header = struct.pack('<iiiiiiiiiiffdddiiiiii',
            0x424C5A54,  # magic "BLZT"
            4,           # version
            0,           # step
            10000,       # total_steps
            n_layers,
            vocab,
            dim,
            hidden,
            n_heads,
            256,         # seq_len
            3e-4,        # lr
            0.0,         # loss
            0.0, 0.0, 0.0,  # cum_compile, cum_train, cum_wall
            0, 0, 0,     # cum_steps, cum_batches, adam_t
            n_kv_heads,
            hd,
            q_dim
        )
        f.write(header)

        wq_sz = q_dim * dim
        wk_sz = kv_dim * dim
        wv_sz = kv_dim * dim
        wo_sz = dim * q_dim
        w1_sz = hidden * dim
        w2_sz = dim * hidden
        w3_sz = hidden * dim

        for L in range(n_layers):
            prefix = f'{layer_prefix}.{L}'

            # Get weights
            Wq = state[f'{prefix}.self_attn.q_proj.weight']  # [q_dim, dim]
            Wk = state[f'{prefix}.self_attn.k_proj.weight']  # [kv_dim, dim]
            Wv = state[f'{prefix}.self_attn.v_proj.weight']  # [kv_dim, dim]
            Wo = state[f'{prefix}.self_attn.o_proj.weight']  # [dim, q_dim]
            W1 = state[f'{prefix}.mlp.gate_proj.weight']     # [hidden, dim]
            W2 = state[f'{prefix}.mlp.down_proj.weight']     # [dim, hidden]
            W3 = state[f'{prefix}.mlp.up_proj.weight']       # [hidden, dim]

            # RMS norms
            rms_att = state[f'{prefix}.input_layernorm.weight']  # [dim]
            rms_ffn = state[f'{prefix}.post_attention_layernorm.weight']  # [dim]

            # Interleave Q/K for ANE's RoPE convention
            Wq = interleave_weights(Wq, n_heads, hd)
            Wk = interleave_weights(Wk, n_kv_heads, hd)

            # Write in checkpoint order
            f.write(Wq.astype(np.float32).tobytes())
            f.write(Wk.astype(np.float32).tobytes())
            f.write(Wv.astype(np.float32).tobytes())
            f.write(Wo.astype(np.float32).tobytes())
            f.write(W1.astype(np.float32).tobytes())
            f.write(W2.astype(np.float32).tobytes())
            f.write(W3.astype(np.float32).tobytes())
            f.write(rms_att.astype(np.float32).tobytes())
            f.write(rms_ffn.astype(np.float32).tobytes())

            # Adam state (zeros for fresh start)
            layer_params = wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2 * dim
            f.write(np.zeros(layer_params * 2, dtype=np.float32).tobytes())

            if L % 5 == 0:
                print(f"  Layer {L}/{n_layers} written")

        # rms_final
        # Try different key names
        rms_final_key = None
        for key in ['model.norm.weight', 'model.final_layernorm.weight']:
            if key in state:
                rms_final_key = key
                break
        if rms_final_key is None:
            raise KeyError(f"Cannot find final norm weight. Keys: {[k for k in state.keys() if 'norm' in k.lower()]}")

        rms_final = state[rms_final_key]
        f.write(rms_final.astype(np.float32).tobytes())
        f.write(np.zeros(dim * 2, dtype=np.float32).tobytes())

        # Embedding
        embed = state['model.embed_tokens.weight']
        f.write(embed.astype(np.float32).tobytes())
        f.write(np.zeros(vocab * dim * 2, dtype=np.float32).tobytes())

    file_size = os.path.getsize(output_path)
    total_params = n_layers * (wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2*dim) + dim + vocab*dim
    print(f"\nWrote {output_path}: {file_size/1e6:.1f} MB")
    print(f"Total params: {total_params/1e6:.1f}M")

    # Print model config for header file
    print(f"\n// Model config for ANE training:")
    print(f'#define MODEL_NAME "{model_name.split("/")[-1]}"')
    print(f"#define DIM {dim}")
    print(f"#define HIDDEN {hidden}")
    print(f"#define HEADS {n_heads}")
    print(f"#define KV_HEADS {n_kv_heads}")
    print(f"#define HD {hd}")
    print(f'#define GQA_RATIO (HEADS/KV_HEADS)  // {n_heads//n_kv_heads}')
    print(f'#define Q_DIM (HEADS*HD)            // {q_dim}')
    print(f'#define KV_DIM (KV_HEADS*HD)        // {kv_dim}')
    print(f"#define SEQ 256")
    print(f"#define NLAYERS {n_layers}")
    print(f"#define VOCAB {vocab}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hf_model_name> <output_ckpt.bin>")
        print(f"Example: {sys.argv[0]} Qwen/Qwen3-0.6B ane_qwen3_06b_ckpt.bin")
        sys.exit(1)
    convert_hf_to_ane(sys.argv[1], sys.argv[2])
