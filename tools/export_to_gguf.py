#!/usr/bin/env python3
"""Export an AutoANE checkpoint to GGUF format (compatible with llama.cpp).

Writes F32 GGUF with llama-family tensor naming convention.

Usage:
    python3 tools/export_to_gguf.py training/ane_autoresearch_ckpt.bin model.gguf
"""

import struct, sys, os, math
import numpy as np


GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGML_TYPE_F32 = 0

# GGUF value types
GV_UINT32 = 4
GV_INT32 = 5
GV_FLOAT32 = 6
GV_STRING = 8
GV_ARRAY = 9
GV_UINT64 = 10


def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)


def write_kv(f, key, vtype, value):
    write_string(f, key)
    f.write(struct.pack('<I', vtype))
    if vtype == GV_UINT32:
        f.write(struct.pack('<I', value))
    elif vtype == GV_INT32:
        f.write(struct.pack('<i', value))
    elif vtype == GV_FLOAT32:
        f.write(struct.pack('<f', value))
    elif vtype == GV_STRING:
        write_string(f, value)
    elif vtype == GV_UINT64:
        f.write(struct.pack('<Q', value))
    elif vtype == GV_ARRAY:
        atype, items = value
        f.write(struct.pack('<I', atype))
        f.write(struct.pack('<Q', len(items)))
        for item in items:
            if atype == GV_STRING:
                write_string(f, item)
            elif atype == GV_INT32:
                f.write(struct.pack('<i', item))
            elif atype == GV_UINT32:
                f.write(struct.pack('<I', item))
            elif atype == GV_FLOAT32:
                f.write(struct.pack('<f', item))


def write_tensor_info(f, name, shape, dtype, offset):
    write_string(f, name)
    f.write(struct.pack('<I', len(shape)))
    for d in shape:
        f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', dtype))
    f.write(struct.pack('<Q', offset))


def de_interleave_weights(W, n_heads, head_dim):
    """Convert from ANE interleaved (re0,im0,re1,im1) to HF (re0,re1,...,im0,im1,...) per head.
    Reverse of hf_to_ane.py interleave_weights."""
    W_out = np.zeros_like(W)
    half = head_dim // 2
    for h in range(n_heads):
        for i in range(half):
            src_re = h * head_dim + 2 * i
            src_im = h * head_dim + 2 * i + 1
            dst_re = h * head_dim + i
            dst_im = h * head_dim + half + i
            W_out[dst_re] = W[src_re]
            W_out[dst_im] = W[src_im]
    return W_out


def validate_checkpoint_config(c, file_size):
    """Validate checkpoint header fields before allocating memory."""
    checks = [
        (1 <= c['n_layers'] <= 256, f"n_layers={c['n_layers']} out of range [1, 256]"),
        (64 <= c['dim'] <= 16384, f"dim={c['dim']} out of range [64, 16384]"),
        (64 <= c['hidden'] <= 65536, f"hidden={c['hidden']} out of range [64, 65536]"),
        (1 <= c['vocab'] <= 500000, f"vocab={c['vocab']} out of range [1, 500000]"),
        (1 <= c['n_heads'] <= 256, f"n_heads={c['n_heads']} out of range [1, 256]"),
        (1 <= c['n_kv_heads'] <= c['n_heads'],
         f"n_kv_heads={c['n_kv_heads']} out of range [1, n_heads={c['n_heads']}]"),
        (1 <= c['hd'] <= 1024, f"hd={c['hd']} out of range [1, 1024]"),
        (c['dim'] % c['n_heads'] == 0, f"dim={c['dim']} not divisible by n_heads={c['n_heads']}"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"Checkpoint header validation failed: {msg}")

    kv_dim = c['n_kv_heads'] * c['hd']
    layer_params = (c['q_dim'] * c['dim'] + kv_dim * c['dim'] * 2 +
                    c['dim'] * c['q_dim'] + c['hidden'] * c['dim'] * 2 +
                    c['dim'] * c['hidden'] + c['dim'] * 2)
    layer_bytes = layer_params * 3 * 4
    other_bytes = c['dim'] * 3 * 4 + c['vocab'] * c['dim'] * 4
    min_size = 96 + c['n_layers'] * layer_bytes + other_bytes
    if file_size < min_size:
        raise ValueError(
            f"Checkpoint too small: {file_size} bytes, expected >= {min_size} bytes "
            f"for {c['n_layers']}L dim={c['dim']} vocab={c['vocab']}")


def load_ane_checkpoint(path):
    """Load ANE BLZT v4 checkpoint."""
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        hdr = struct.unpack('<iiiiiiiiiiffdddiiiiii', f.read(96))

        magic, version = hdr[0], hdr[1]
        if magic != 0x424C5A54:
            raise ValueError("Not a BLZT checkpoint")
        if version != 4:
            raise ValueError(f"Unsupported checkpoint version {version}")

        c = {
            'step': hdr[2], 'n_layers': hdr[4], 'vocab': hdr[5],
            'dim': hdr[6], 'hidden': hdr[7], 'n_heads': hdr[8],
            'n_kv_heads': hdr[18], 'hd': hdr[19], 'q_dim': hdr[20],
        }
        validate_checkpoint_config(c, file_size)
        kv_dim = c['n_kv_heads'] * c['hd']

        sizes = {
            'Wq': (c['q_dim'], c['dim']),   'Wk': (kv_dim, c['dim']),
            'Wv': (kv_dim, c['dim']),        'Wo': (c['dim'], c['q_dim']),
            'W1': (c['hidden'], c['dim']),   'W2': (c['dim'], c['hidden']),
            'W3': (c['hidden'], c['dim']),
            'rms_att': (c['dim'],),          'rms_ffn': (c['dim'],),
        }
        layer_param_count = sum(math.prod(s) for s in sizes.values())

        layers = []
        for L in range(c['n_layers']):
            layer = {}
            for name, shape in sizes.items():
                n = math.prod(shape)
                layer[name] = np.frombuffer(f.read(n * 4), np.float32).reshape(shape).copy()
            f.seek(layer_param_count * 2 * 4, 1)  # skip Adam
            layers.append(layer)

        rms_final = np.frombuffer(f.read(c['dim'] * 4), np.float32).copy()
        f.seek(c['dim'] * 2 * 4, 1)

        embed = np.frombuffer(f.read(c['vocab'] * c['dim'] * 4), np.float32) \
                    .reshape(c['vocab'], c['dim']).copy()

    return c, layers, rms_final, embed


def load_tokenizer_metadata(vocab_size):
    """Load SmolLM2 tokenizer and return GGUF metadata entries for it."""
    try:
        import json, glob
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                            trust_remote_code=True)
        tokens = []
        scores = []
        token_types = []
        for i in range(vocab_size):
            try:
                t = tok.convert_ids_to_tokens(i)
                if t is None:
                    t = f"[UNK{i}]"
            except Exception:
                t = f"[UNK{i}]"
            tokens.append(t)
            scores.append(0.0)
            token_types.append(1)  # normal token

        # Load BPE merges from tokenizer.json
        merges = []
        cache_pattern = os.path.expanduser(
            "~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/*/tokenizer.json")
        for tf in glob.glob(cache_pattern):
            with open(tf) as fh:
                data = json.load(fh)
            if 'model' in data and 'merges' in data['model']:
                merges = data['model']['merges']
            break

        bos_id = tok.bos_token_id if tok.bos_token_id is not None else 1
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2

        print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges, bos={bos_id}, eos={eos_id}")
        result = [
            ('tokenizer.ggml.model', GV_STRING, 'gpt2'),
            ('tokenizer.ggml.tokens', GV_ARRAY, (GV_STRING, tokens)),
            ('tokenizer.ggml.scores', GV_ARRAY, (GV_FLOAT32, scores)),
            ('tokenizer.ggml.token_type', GV_ARRAY, (GV_INT32, token_types)),
            ('tokenizer.ggml.bos_token_id', GV_UINT32, bos_id),
            ('tokenizer.ggml.eos_token_id', GV_UINT32, eos_id),
        ]
        if merges:
            result.append(('tokenizer.ggml.merges', GV_ARRAY, (GV_STRING, merges)))
        return result
    except ImportError:
        print("  WARNING: transformers not installed, skipping tokenizer metadata")
        return []


def export_gguf(ane_path, gguf_path):
    print(f"Loading ANE checkpoint: {ane_path}")
    c, layers, rms_final, embed = load_ane_checkpoint(ane_path)

    print(f"  {c['n_layers']}L dim={c['dim']} hidden={c['hidden']} "
          f"heads={c['n_heads']}/{c['n_kv_heads']} hd={c['hd']} vocab={c['vocab']}")

    # Build tensor list: (name, data_array_in_gguf_shape)
    # GGUF uses reversed dimension order (row-major to col-major convention)
    tensors = []

    # Embedding
    tensors.append(('token_embd.weight', embed))

    # Per-layer
    for L in range(c['n_layers']):
        lw = layers[L]

        # De-interleave Q/K for standard RoPE convention
        Wq = de_interleave_weights(lw['Wq'], c['n_heads'], c['hd'])
        Wk = de_interleave_weights(lw['Wk'], c['n_kv_heads'], c['hd'])

        tensors.append((f'blk.{L}.attn_q.weight', Wq))
        tensors.append((f'blk.{L}.attn_k.weight', Wk))
        tensors.append((f'blk.{L}.attn_v.weight', lw['Wv']))
        tensors.append((f'blk.{L}.attn_output.weight', lw['Wo']))
        tensors.append((f'blk.{L}.ffn_gate.weight', lw['W1']))
        tensors.append((f'blk.{L}.ffn_down.weight', lw['W2']))
        tensors.append((f'blk.{L}.ffn_up.weight', lw['W3']))
        tensors.append((f'blk.{L}.attn_norm.weight', lw['rms_att']))
        tensors.append((f'blk.{L}.ffn_norm.weight', lw['rms_ffn']))

    # Final norm
    tensors.append(('output_norm.weight', rms_final))

    # Load tokenizer metadata for llama.cpp compatibility
    tokenizer_meta = load_tokenizer_metadata(c['vocab'])

    # Metadata KV pairs
    metadata = [
        ('general.architecture', GV_STRING, 'llama'),
        ('general.name', GV_STRING, f'AutoANE-{c["n_layers"]}L-{c["dim"]}d'),
        ('general.file_type', GV_UINT32, 0),  # F32
        ('llama.block_count', GV_UINT32, c['n_layers']),
        ('llama.embedding_length', GV_UINT32, c['dim']),
        ('llama.feed_forward_length', GV_UINT32, c['hidden']),
        ('llama.attention.head_count', GV_UINT32, c['n_heads']),
        ('llama.attention.head_count_kv', GV_UINT32, c['n_kv_heads']),
        ('llama.attention.layer_norm_rms_epsilon', GV_FLOAT32, 1e-5),
        ('llama.context_length', GV_UINT32, 128),
        ('llama.vocab_size', GV_UINT32, c['vocab']),
        ('llama.rope.dimension_count', GV_UINT32, c['hd']),
        ('llama.rope.freq_base', GV_FLOAT32, 10000.0),
    ] + tokenizer_meta

    alignment = 32

    print(f"Writing GGUF: {gguf_path}")
    with open(gguf_path, 'wb') as f:
        # Header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', len(metadata)))

        # Metadata
        for key, vtype, value in metadata:
            write_kv(f, key, vtype, value)

        # Tensor info (compute offsets)
        data_offset = 0
        tensor_data_list = []
        for name, arr in tensors:
            arr = arr.astype(np.float32)
            # GGUF dims are reversed from numpy shape
            gguf_dims = list(reversed(arr.shape))
            write_tensor_info(f, name, gguf_dims, GGML_TYPE_F32, data_offset)
            tensor_data_list.append(arr)
            # Align next tensor
            nbytes = arr.nbytes
            padded = (nbytes + alignment - 1) // alignment * alignment
            data_offset += padded

        # Pad to alignment before tensor data
        pos = f.tell()
        pad = (alignment - pos % alignment) % alignment
        f.write(b'\x00' * pad)

        # Tensor data
        for arr in tensor_data_list:
            f.write(arr.tobytes())
            # Pad to alignment
            pad = (alignment - arr.nbytes % alignment) % alignment
            if pad:
                f.write(b'\x00' * pad)

    total_params = sum(arr.size for arr in tensor_data_list)
    file_size = os.path.getsize(gguf_path)
    print(f"  {total_params/1e6:.1f}M params, {file_size/1e6:.1f} MB")
    print(f"  Load with: llama.cpp/llama-cli -m {gguf_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <checkpoint.bin> <output.gguf>")
        print(f"Example: {sys.argv[0]} training/ane_autoresearch_ckpt.bin model.gguf")
        sys.exit(1)
    export_gguf(sys.argv[1], sys.argv[2])
