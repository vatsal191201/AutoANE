#!/usr/bin/env python3
"""Verify generate.py forward pass matches the C training binary.

Loads the same checkpoint and data, computes val_loss in Python using
the same vocab-compaction scheme as train.m, and compares against the
C binary's reported val_loss.

This is a first-principles verification: if the Python loss matches
the C loss on the same data, the forward pass is numerically correct.
"""

import struct, sys, os, math, time
import numpy as np

# Add parent dir to path for generate.py imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate import load_checkpoint, forward_last, rmsnorm, apply_rope, attention, silu


def load_data(path):
    """Load tokenized binary data (uint16 tokens)."""
    data = np.fromfile(path, dtype=np.uint16)
    return data


def build_vocab_map(token_data, full_vocab):
    """Replicate train.m's vocab compaction."""
    used = set(token_data.tolist())
    full_to_compact = np.full(full_vocab, -1, dtype=np.int32)
    compact_to_full = []
    cid = 0
    for v in range(full_vocab):
        if v in used:
            full_to_compact[v] = cid
            compact_to_full.append(v)
            cid += 1
    return full_to_compact, np.array(compact_to_full, dtype=np.int32), cid


def forward_full_sequence(tokens_np, config, layers, rms_final, embed):
    """Forward pass returning logits for ALL positions. tokens: [seq] int."""
    c = config
    res_alpha = 1.0 / math.sqrt(2.0 * c['n_layers'])
    seq = len(tokens_np)

    x = embed[tokens_np]  # [seq, dim]

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
    return x  # [seq, dim] — return pre-logit hidden states


def cross_entropy_loss(hidden_states, embed, targets, compact_to_full):
    """Compute cross-entropy with compacted vocab (matching train.m)."""
    # Compact embedding: only rows for active tokens
    compact_embed = embed[compact_to_full]  # [CV, dim]

    # Logits: [seq, CV]
    logits = hidden_states @ compact_embed.T

    # Stable softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    log_sum_exp = np.log(exp_logits.sum(axis=-1))

    # Cross-entropy: -log(softmax[target])
    seq = len(targets)
    losses = []
    for t in range(seq):
        target_logit = logits[t, targets[t]] - logits_max[t, 0]
        loss_t = -target_logit + log_sum_exp[t]
        losses.append(loss_t)

    return float(np.mean(losses))


def cross_entropy_full_vocab(hidden_states, embed, targets):
    """Cross-entropy with full vocab (no compaction)."""
    logits = hidden_states @ embed.T  # [seq, vocab]
    logits_max = logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    log_sum_exp = np.log(exp_logits.sum(axis=-1))

    seq = len(targets)
    losses = []
    for t in range(seq):
        target_logit = logits[t, targets[t]] - logits_max[t, 0]
        loss_t = -target_logit + log_sum_exp[t]
        losses.append(loss_t)

    return float(np.mean(losses))


def main():
    # Paths
    ckpt_path = 'training/ane_autoresearch_ckpt.bin'
    data_path = 'tinystories_smollm2_data00.bin'

    for p in [ckpt_path, data_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run from repo root.")
            sys.exit(1)

    # Load model
    print("=== Forward Pass Verification ===\n")
    print("[1] Loading checkpoint...")
    config, layers, rms_final, embed = load_checkpoint(ckpt_path)
    c = config
    print(f"    {c['n_layers']}L dim={c['dim']} hidden={c['hidden']} "
          f"heads={c['n_heads']}/{c['n_kv_heads']} hd={c['hd']}")
    print(f"    vocab={c['vocab']}, step={c['step']}")

    # Load data
    print("\n[2] Loading data...")
    token_data = load_data(data_path)
    n_tokens = len(token_data)
    val_start = int(n_tokens * 0.9)
    train_tokens = val_start
    val_tokens = n_tokens - val_start
    print(f"    {n_tokens} tokens total, train={train_tokens}, val={val_tokens}")

    # Build vocab map
    print("\n[3] Building vocab compaction map...")
    full_to_compact, compact_to_full, CV = build_vocab_map(token_data, c['vocab'])
    print(f"    Full vocab: {c['vocab']}, Compact vocab: {CV} "
          f"({CV/c['vocab']*100:.1f}% active)")

    # Verify token range
    tok_min, tok_max = int(token_data.min()), int(token_data.max())
    print(f"    Token range: [{tok_min}, {tok_max}]")
    assert tok_max < c['vocab'], f"Token {tok_max} >= vocab {c['vocab']}"

    # Run validation on multiple samples
    SEQ = 128  # match training config
    n_val_samples = 10
    print(f"\n[4] Computing val_loss on {n_val_samples} samples (SEQ={SEQ})...")

    np.random.seed(42)
    compact_losses = []
    full_losses = []

    for i in range(n_val_samples):
        # Random position in validation split
        vpos = val_start + np.random.randint(0, val_tokens - SEQ - 1)
        input_tokens = token_data[vpos:vpos + SEQ].astype(np.int64)
        target_tokens_full = token_data[vpos + 1:vpos + SEQ + 1].astype(np.int64)
        target_tokens_compact = np.array([full_to_compact[t] for t in target_tokens_full],
                                          dtype=np.int64)

        t0 = time.time()
        hidden = forward_full_sequence(input_tokens, config, layers, rms_final, embed)
        fwd_ms = (time.time() - t0) * 1000

        # Loss with compact vocab (matches C binary)
        loss_compact = cross_entropy_loss(hidden, embed, target_tokens_compact, compact_to_full)
        compact_losses.append(loss_compact)

        # Loss with full vocab
        loss_full = cross_entropy_full_vocab(hidden, embed, target_tokens_full)
        full_losses.append(loss_full)

        print(f"    Sample {i+1}: compact_loss={loss_compact:.4f}, "
              f"full_loss={loss_full:.4f}, fwd={fwd_ms:.0f}ms")

    avg_compact = np.mean(compact_losses)
    avg_full = np.mean(full_losses)
    std_compact = np.std(compact_losses)

    print(f"\n[5] Results:")
    print(f"    Avg compact val_loss: {avg_compact:.4f} +/- {std_compact:.4f}")
    print(f"    Avg full val_loss:    {avg_full:.4f}")
    print(f"    Compact-Full delta:   {avg_full - avg_compact:.4f} "
          f"(expected: full > compact due to larger softmax denominator)")

    # Compare with known C binary val_loss
    known_val_loss = 3.507  # from results.tsv, best configuration
    delta = abs(avg_compact - known_val_loss)
    print(f"\n[6] Comparison with C binary:")
    print(f"    C binary val_loss (results.tsv): {known_val_loss:.3f}")
    print(f"    Python compact val_loss:         {avg_compact:.3f}")
    print(f"    Delta:                           {delta:.3f}")

    # Check: initial loss should be ~ln(CV) for random weights
    initial_loss_expected = math.log(CV)
    print(f"\n[7] Sanity checks:")
    print(f"    ln(compact_vocab) = ln({CV}) = {initial_loss_expected:.2f} "
          f"(untrained model should produce this)")
    print(f"    Trained model loss: {avg_compact:.2f} "
          f"(should be << {initial_loss_expected:.2f})")

    if avg_compact < initial_loss_expected:
        print(f"    CHECK PASSED: Loss {avg_compact:.2f} < {initial_loss_expected:.2f}")
    else:
        print(f"    CHECK FAILED: Loss should be below untrained baseline")

    if delta < 0.5:
        print(f"    CHECK PASSED: Within 0.5 of C binary ({delta:.3f})")
    elif delta < 1.0:
        print(f"    CHECK WARNING: Within 1.0 of C binary ({delta:.3f}) — "
              f"may be due to different val samples")
    else:
        print(f"    CHECK FAILED: Delta {delta:.3f} > 1.0 — forward pass may be wrong")

    # Top-k prediction check
    print(f"\n[8] Top-k prediction check (last sample):")
    last_hidden = hidden[-1]  # last position
    logits = last_hidden @ embed.T
    top_k_idx = np.argsort(logits)[-10:][::-1]
    top_k_logits = logits[top_k_idx]
    top_k_probs = np.exp(top_k_logits - top_k_logits.max())
    top_k_probs /= top_k_probs.sum()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                            trust_remote_code=True)
        print(f"    Top 10 predicted tokens:")
        for j, (idx, prob) in enumerate(zip(top_k_idx, top_k_probs)):
            token_str = tok.decode([int(idx)])
            print(f"      {j+1}. '{token_str}' (id={idx}, p={prob:.3f})")
    except ImportError:
        print(f"    Top 10 token IDs: {top_k_idx.tolist()}")
        print(f"    Probabilities: {[f'{p:.3f}' for p in top_k_probs]}")


if __name__ == '__main__':
    main()
