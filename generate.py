#!/usr/bin/env python3
"""Generate text from a trained AutoANE checkpoint.

Pure numpy inference — no dependencies beyond numpy.
Optionally uses HuggingFace tokenizer for text encoding/decoding.

Usage:
    python3 generate.py [checkpoint] [--prompt "Once upon a time"] [--tokens 200]
    python3 generate.py training/ane_autoresearch_ckpt.bin --temperature 0.8 --top-k 40
"""

import struct, sys, os, math, time
import numpy as np


def load_checkpoint(path):
    """Load ANE checkpoint (v4 BLZT format). Returns config, layers, rms_final, embed."""
    with open(path, 'rb') as f:
        # Header: 10 ints + 2 floats + 3 doubles + 6 ints = 96 bytes
        hdr = struct.unpack('<iiiiiiiiiiffdddiiiiii', f.read(96))

        magic, version = hdr[0], hdr[1]
        assert magic == 0x424C5A54, f"Not a BLZT checkpoint (magic={hex(magic)})"
        assert version == 4, f"Unsupported checkpoint version {version}"

        config = {
            'step': hdr[2], 'n_layers': hdr[4], 'vocab': hdr[5],
            'dim': hdr[6], 'hidden': hdr[7], 'n_heads': hdr[8],
            'n_kv_heads': hdr[18], 'hd': hdr[19], 'q_dim': hdr[20],
        }
        c = config
        kv_dim = c['n_kv_heads'] * c['hd']

        # Per-layer sizes
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
            # Skip Adam m + v state
            f.seek(layer_param_count * 2 * 4, 1)
            layers.append(layer)

        # Final RMSNorm
        rms_final = np.frombuffer(f.read(c['dim'] * 4), np.float32).copy()
        f.seek(c['dim'] * 2 * 4, 1)  # skip Adam state

        # Embedding (tied with classifier)
        embed = np.frombuffer(f.read(c['vocab'] * c['dim'] * 4), np.float32) \
                    .reshape(c['vocab'], c['dim']).copy()

    return config, layers, rms_final, embed


# ===== Forward pass (matches train.m CPU-only path exactly) =====

def rmsnorm(x, w):
    """x: [seq, dim], w: [dim] -> [seq, dim]"""
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)) * w


def apply_rope(x, n_heads, hd):
    """RoPE with interleaved (re, im) pairs. x: [seq, n_heads*hd] -> [seq, n_heads*hd]"""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)

    freqs = 1.0 / (10000.0 ** (2.0 * np.arange(hd // 2, dtype=np.float32) / hd))
    theta = np.arange(seq, dtype=np.float32)[:, None] * freqs[None, :]  # [seq, hd//2]
    cos_t = np.cos(theta)[:, None, :]  # [seq, 1, hd//2]
    sin_t = np.sin(theta)[:, None, :]

    x_re, x_im = x[:, :, 0::2], x[:, :, 1::2]
    out = np.empty_like(x)
    out[:, :, 0::2] = x_re * cos_t - x_im * sin_t
    out[:, :, 1::2] = x_re * sin_t + x_im * cos_t
    return out.reshape(seq, -1)


def attention(Q, K, V, n_heads, n_kv_heads, hd):
    """GQA causal attention. Q: [seq, q_dim], K/V: [seq, kv_dim]"""
    seq = Q.shape[0]
    gqa_ratio = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(hd)

    Q = Q.reshape(seq, n_heads, hd).transpose(1, 0, 2)      # [heads, seq, hd]
    K = K.reshape(seq, n_kv_heads, hd).transpose(1, 0, 2)    # [kv_heads, seq, hd]
    V = V.reshape(seq, n_kv_heads, hd).transpose(1, 0, 2)    # [kv_heads, seq, hd]

    # Tile KV heads for GQA
    K = np.repeat(K, gqa_ratio, axis=0)  # [heads, seq, hd]
    V = np.repeat(V, gqa_ratio, axis=0)

    # Scaled dot-product attention
    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale  # [heads, seq, seq]

    # Causal mask
    mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
    scores += mask[None, :, :]

    # Softmax (numerically stable)
    scores -= scores.max(axis=-1, keepdims=True)
    exp_s = np.exp(scores)
    attn = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9)

    out = np.matmul(attn, V)  # [heads, seq, hd]
    return out.transpose(1, 0, 2).reshape(seq, -1)  # [seq, q_dim]


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def forward_last(tokens, config, layers, rms_final, embed):
    """Forward pass returning logits for the LAST token only."""
    c = config
    res_alpha = 1.0 / math.sqrt(2.0 * c['n_layers'])

    x = embed[tokens]  # [seq, dim]

    for lw in layers:
        xn = rmsnorm(x, lw['rms_att'])
        Q = xn @ lw['Wq'].T
        K = xn @ lw['Wk'].T
        V = xn @ lw['Wv'].T
        Q = apply_rope(Q, c['n_heads'], c['hd'])
        K = apply_rope(K, c['n_kv_heads'], c['hd'])
        o = attention(Q, K, V, c['n_heads'], c['n_kv_heads'], c['hd']) @ lw['Wo'].T
        x = x + res_alpha * o

        x2 = rmsnorm(x, lw['rms_ffn'])
        ffn = (silu(x2 @ lw['W1'].T) * (x2 @ lw['W3'].T)) @ lw['W2'].T
        x = x + res_alpha * ffn

    x = rmsnorm(x, rms_final)
    return x[-1] @ embed.T  # [vocab]


# ===== Sampling =====

def sample_top_k(logits, temperature=1.0, top_k=40):
    if temperature < 1e-8:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / temperature

    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        vals = logits[idx]
        vals -= vals.max()
        probs = np.exp(vals)
        probs /= probs.sum()
        return int(idx[np.random.choice(len(idx), p=probs)])

    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(logits), p=probs))


# ===== Generation loop =====

def generate(config, layers, rms_final, embed, prompt_tokens,
             n_tokens=200, temperature=0.8, top_k=40, stream_fn=None):
    """Autoregressive generation. Returns list of all tokens (prompt + generated)."""
    tokens = list(prompt_tokens)
    max_ctx = 128  # match training SEQ

    t0 = time.time()
    for i in range(n_tokens):
        ctx = tokens[-max_ctx:]
        logits = forward_last(np.array(ctx, dtype=np.int64), config, layers, rms_final, embed)
        tok = sample_top_k(logits, temperature, top_k)
        tokens.append(tok)

        if stream_fn:
            stream_fn(tok, i, n_tokens)

    elapsed = time.time() - t0
    tps = n_tokens / elapsed if elapsed > 0 else 0
    print(f"\n[{n_tokens} tokens in {elapsed:.1f}s = {tps:.1f} tok/s]", file=sys.stderr)
    return tokens


# ===== Tokenizer helpers =====

def load_tokenizer():
    """Try to load SmolLM2 tokenizer. Returns (tokenizer, encode_fn, decode_fn) or None."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                            trust_remote_code=True)
        return tok, tok.encode, lambda ids: tok.decode(ids, skip_special_tokens=True)
    except Exception:
        return None, None, None


def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate text from a trained AutoANE checkpoint')
    p.add_argument('checkpoint', nargs='?', help='Path to .bin checkpoint')
    p.add_argument('--prompt', default='Once upon a time', help='Text prompt')
    p.add_argument('--tokens', type=int, default=200, help='Tokens to generate')
    p.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (0=greedy)')
    p.add_argument('--top-k', type=int, default=40, help='Top-k sampling (0=full)')
    p.add_argument('--seed', type=int, default=None, help='Random seed')
    p.add_argument('--raw', action='store_true', help='Print raw token IDs instead of text')
    args = p.parse_args()

    # Find checkpoint
    ckpt = args.checkpoint
    if not ckpt:
        candidates = [
            'training/ane_autoresearch_ckpt.bin',
            'ane_autoresearch_ckpt.bin',
            'training/ane_smollm2_360m_ckpt.bin',
            'training/ane_stories110m_ckpt.bin',
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt = c
                break
        if not ckpt:
            print("No checkpoint found. Train first:\n  cd training && python3 train.py",
                  file=sys.stderr)
            sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load model
    print(f"Loading {ckpt}...", file=sys.stderr)
    config, layers, rms_final, embed = load_checkpoint(ckpt)
    n_params = sum(w.size for L in layers for w in L.values()) + rms_final.size + embed.size
    print(f"  {config['n_layers']}L dim={config['dim']} | {n_params/1e6:.1f}M params | "
          f"step {config['step']}", file=sys.stderr)

    # Tokenize
    tok, encode_fn, decode_fn = load_tokenizer()
    if encode_fn and not args.raw:
        prompt_tokens = encode_fn(args.prompt)
        print(f"  Prompt: \"{args.prompt}\" -> {len(prompt_tokens)} tokens", file=sys.stderr)
    else:
        prompt_tokens = [1]  # BOS
        if not args.raw:
            print("  No tokenizer (pip install transformers). Using BOS token.", file=sys.stderr)

    # Streaming callback
    partial = []
    last_len = 0
    def stream(tok_id, i, total):
        nonlocal last_len
        if decode_fn and not args.raw:
            partial.append(tok_id)
            text = decode_fn(prompt_tokens + partial)
            new_text = text[last_len:]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                last_len = len(text)

    # Generate
    print(f"  Generating {args.tokens} tokens (temp={args.temperature}, top_k={args.top_k})...",
          file=sys.stderr)
    all_tokens = generate(config, layers, rms_final, embed, prompt_tokens,
                          n_tokens=args.tokens, temperature=args.temperature,
                          top_k=args.top_k, stream_fn=stream if not args.raw else None)

    # Output
    if args.raw:
        print(all_tokens)
    elif decode_fn:
        print(f"\n{'='*60}")
        print(decode_fn(all_tokens))
    else:
        print(f"\nToken IDs: {all_tokens}")
        print("Install transformers for text: pip install transformers", file=sys.stderr)


if __name__ == '__main__':
    main()
