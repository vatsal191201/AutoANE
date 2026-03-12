#!/usr/bin/env python3
"""Convert GGUF model (from ollama) to ANE training checkpoint format.
Supports F16 and F32 GGUF tensors. Outputs float32 binary checkpoint."""

import struct
import numpy as np
import sys
import os

def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')

def read_value(f, vtype):
    if vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7: return struct.unpack('<?', f.read(1))[0]
    elif vtype == 8: return read_string(f)
    elif vtype == 9:
        atype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, atype) for _ in range(alen)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<H', f.read(2))[0]
    else: return None

GGML_TYPES = {0: ('F32', 4), 1: ('F16', 2), 29: ('BF16', 2)}


def interleave_weights(W, n_heads, head_dim):
    """Convert Q/K from HF/GGUF (non-interleaved) to ANE (interleaved) RoPE ordering.
    HF/GGUF:  [re0, re1, ..., re31, im0, im1, ..., im31] per head
    ANE:      [re0, im0, re1, im1, ..., re31, im31] per head
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

def load_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not GGUF: {magic}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        metadata = {}
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            metadata[key] = read_value(f, vtype)

        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': offset}

        # Data starts at aligned position after header
        header_end = f.tell()
        alignment = metadata.get('general.alignment', 32)
        data_start = (header_end + alignment - 1) // alignment * alignment

        # Load tensor data
        for name, info in tensors.items():
            dtype_id = info['dtype']
            if dtype_id not in GGML_TYPES:
                print(f"  ERROR: Unsupported dtype {dtype_id} for tensor '{name}'")
                print(f"  Supported dtypes: {list(GGML_TYPES.keys())} ({', '.join(n for n,_ in GGML_TYPES.values())})")
                print(f"  Quantized GGUF files must be dequantized first (use llama.cpp quantize --dequantize)")
                raise ValueError(f"Unsupported GGML dtype {dtype_id} for tensor '{name}'")
            dtype_name, elem_size = GGML_TYPES[dtype_id]
            n_elements = 1
            for d in info['dims']:
                n_elements *= d
            f.seek(data_start + info['offset'])
            raw = f.read(n_elements * elem_size)
            if dtype_name == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(info['dims'][::-1])
            elif dtype_name == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(info['dims'][::-1])
            elif dtype_name == 'BF16':
                raw16 = np.frombuffer(raw, dtype=np.uint16)
                arr = np.zeros(len(raw16), dtype=np.float32)
                arr.view(np.uint32)[:] = raw16.astype(np.uint32) << 16
                arr = arr.reshape(info['dims'][::-1])
            info['data'] = arr

    return metadata, tensors

def convert_to_ane_ckpt(gguf_path, output_path):
    print(f"Loading GGUF: {gguf_path}")
    meta, tensors = load_gguf(gguf_path)

    arch = meta.get('general.architecture', 'llama')
    dim = meta.get(f'{arch}.embedding_length')
    hidden = meta.get(f'{arch}.feed_forward_length')
    n_layers = meta.get(f'{arch}.block_count')
    n_heads = meta.get(f'{arch}.attention.head_count')
    n_kv_heads = meta.get(f'{arch}.attention.head_count_kv', n_heads)
    vocab = meta.get(f'{arch}.vocab_size')
    hd = dim // n_heads
    q_dim = n_heads * hd
    kv_dim = n_kv_heads * hd

    print(f"Architecture: {arch}")
    print(f"  dim={dim} hidden={hidden} layers={n_layers}")
    print(f"  heads={n_heads}/{n_kv_heads} hd={hd} q_dim={q_dim} kv_dim={kv_dim}")
    print(f"  vocab={vocab}")

    # Write ANE checkpoint (v4 format matching training_dynamic/train.m)
    with open(output_path, 'wb') as f:
        # Header — must match C struct CkptHdr exactly (96 bytes)
        # 10 ints + 2 floats + 3 doubles + 6 ints
        header = struct.pack('<iiiiiiiiiiffdddiiiiii',
            0x424C5A54,  # magic "BLZT"
            4,           # version
            0,           # step
            10000,       # total_steps
            n_layers,    # n_layers
            vocab,       # vocab_size
            dim,         # dim
            hidden,      # hidden_dim
            n_heads,     # n_heads
            256,         # seq_len
            3e-4,        # lr
            0.0,         # loss
            0.0,         # cum_compile
            0.0,         # cum_train
            0.0,         # cum_wall
            0,           # cum_steps
            0,           # cum_batches
            0,           # adam_t
            n_kv_heads,  # kv_heads
            hd,          # head_dim
            q_dim        # q_dim
        )
        f.write(header)

        # Per-layer weights
        wq_sz = q_dim * dim
        wk_sz = kv_dim * dim
        wv_sz = kv_dim * dim
        wo_sz = dim * q_dim
        w1_sz = hidden * dim
        w2_sz = dim * hidden
        w3_sz = hidden * dim

        def get_tensor(name):
            if name in tensors and 'data' in tensors[name]:
                return tensors[name]['data']
            print(f"  WARNING: tensor {name} not found")
            return None

        for L in range(n_layers):
            # Weights
            wq = get_tensor(f'blk.{L}.attn_q.weight')
            wk = get_tensor(f'blk.{L}.attn_k.weight')
            wv = get_tensor(f'blk.{L}.attn_v.weight')
            wo = get_tensor(f'blk.{L}.attn_output.weight')
            w1 = get_tensor(f'blk.{L}.ffn_gate.weight')
            w2 = get_tensor(f'blk.{L}.ffn_down.weight')
            w3 = get_tensor(f'blk.{L}.ffn_up.weight')
            rms_att = get_tensor(f'blk.{L}.attn_norm.weight')
            rms_ffn = get_tensor(f'blk.{L}.ffn_norm.weight')

            # Interleave Q/K for ANE's RoPE convention
            # GGUF stores [re0..re31, im0..im31] per head (HF convention)
            # ANE needs [re0, im0, re1, im1, ...] per head (interleaved)
            wq = interleave_weights(wq, n_heads, hd)
            wk = interleave_weights(wk, n_kv_heads, hd)

            f.write(wq.flatten().astype(np.float32).tobytes())   # Wq [q_dim, dim]
            f.write(wk.flatten().astype(np.float32).tobytes())   # Wk [kv_dim, dim]
            f.write(wv.flatten().astype(np.float32).tobytes())   # Wv [kv_dim, dim]
            f.write(wo.flatten().astype(np.float32).tobytes())   # Wo [dim, q_dim]
            f.write(w1.flatten().astype(np.float32).tobytes())   # W1 [hidden, dim]
            f.write(w2.flatten().astype(np.float32).tobytes())   # W2 [dim, hidden]
            f.write(w3.flatten().astype(np.float32).tobytes())   # W3 [hidden, dim]
            f.write(rms_att.flatten().astype(np.float32).tobytes())  # rms_att [dim]
            f.write(rms_ffn.flatten().astype(np.float32).tobytes())  # rms_ffn [dim]

            # Adam state (zeros for fresh start)
            layer_params = wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2*dim
            zeros = np.zeros(layer_params * 2, dtype=np.float32)  # m and v
            f.write(zeros.tobytes())

        # rms_final
        rms_final = get_tensor('output_norm.weight')
        f.write(rms_final.flatten().astype(np.float32).tobytes())
        # Adam state for rms_final
        f.write(np.zeros(dim * 2, dtype=np.float32).tobytes())

        # Embedding
        embed = get_tensor('token_embd.weight')
        f.write(embed.flatten().astype(np.float32).tobytes())
        # Adam state for embedding
        f.write(np.zeros(vocab * dim * 2, dtype=np.float32).tobytes())

    file_size = os.path.getsize(output_path)
    print(f"\nWrote {output_path}: {file_size/1e6:.1f} MB")
    total_params = n_layers * (wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2*dim) + dim + vocab*dim
    print(f"Total params: {total_params/1e6:.1f}M")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output_ckpt.bin>")
        sys.exit(1)
    convert_to_ane_ckpt(sys.argv[1], sys.argv[2])
