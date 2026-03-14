#!/usr/bin/env python3
"""Head-to-head logit comparison: AutoANE numpy forward pass vs HuggingFace transformers.

Loads SmolLM2-360M from both HuggingFace and the ANE checkpoint, runs one
forward pass on a fixed token sequence, and compares output logits position
by position. If discrepancies exceed tolerance, traces layer-by-layer to
find the divergence point.

Expected result: max absolute logit diff < 0.01 (float32 precision).
"""

import sys, os, math, time
import numpy as np

# Add repo root so we can import generate.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from generate import load_checkpoint, rmsnorm, apply_rope, attention, silu

# ============================================================================
# Config
# ============================================================================
ANE_CKPT = os.path.join(REPO_ROOT, "training", "ane_smollm2_360m_clean.bin")
HF_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
ROPE_THETA = 100000.0

# ============================================================================
# ANE full forward pass (returns logits at ALL positions)
# ============================================================================
def ane_forward_all(tokens, config, layers, rms_final, embed):
    """Full forward pass returning [seq, vocab] logits."""
    c = config
    rope_theta = c.get('rope_theta', 10000.0)
    x = embed[tokens]  # [seq, dim]

    for lw in layers:
        xn = rmsnorm(x, lw['rms_att'])
        Q = xn @ lw['Wq'].T
        K = xn @ lw['Wk'].T
        V = xn @ lw['Wv'].T
        Q = apply_rope(Q, c['n_heads'], c['hd'], rope_theta)
        K = apply_rope(K, c['n_kv_heads'], c['hd'], rope_theta)
        o = attention(Q, K, V, c['n_heads'], c['n_kv_heads'], c['hd']) @ lw['Wo'].T
        x = x + o

        x2 = rmsnorm(x, lw['rms_ffn'])
        ffn = (silu(x2 @ lw['W1'].T) * (x2 @ lw['W3'].T)) @ lw['W2'].T
        x = x + ffn

    x = rmsnorm(x, rms_final)
    logits = x @ embed.T  # [seq, vocab]
    return logits


def ane_forward_layerwise(tokens, config, layers, rms_final, embed):
    """Forward pass that records intermediate states after each layer."""
    c = config
    rope_theta = c.get('rope_theta', 10000.0)
    x = embed[tokens]  # [seq, dim]
    traces = {'embed': x.copy()}

    for li, lw in enumerate(layers):
        xn = rmsnorm(x, lw['rms_att'])
        if li == 0:
            traces['L0_post_rms_att'] = xn.copy()

        Q = xn @ lw['Wq'].T
        K = xn @ lw['Wk'].T
        V = xn @ lw['Wv'].T
        if li == 0:
            traces['L0_Q_pre_rope'] = Q.copy()
            traces['L0_K_pre_rope'] = K.copy()
            traces['L0_V'] = V.copy()

        Q = apply_rope(Q, c['n_heads'], c['hd'], rope_theta)
        K = apply_rope(K, c['n_kv_heads'], c['hd'], rope_theta)
        if li == 0:
            traces['L0_Q_post_rope'] = Q.copy()
            traces['L0_K_post_rope'] = K.copy()

        attn_out = attention(Q, K, V, c['n_heads'], c['n_kv_heads'], c['hd'])
        if li == 0:
            traces['L0_attn_out'] = attn_out.copy()

        o = attn_out @ lw['Wo'].T
        if li == 0:
            traces['L0_o_proj'] = o.copy()

        x = x + o

        x2 = rmsnorm(x, lw['rms_ffn'])
        ffn = (silu(x2 @ lw['W1'].T) * (x2 @ lw['W3'].T)) @ lw['W2'].T
        x = x + ffn

        traces[f'L{li}_out'] = x.copy()

    x = rmsnorm(x, rms_final)
    traces['final_norm'] = x.copy()
    logits = x @ embed.T
    traces['logits'] = logits
    return traces


# ============================================================================
# HuggingFace forward pass with layer-by-layer tracing
# ============================================================================
def hf_forward_layerwise(model, input_ids_tensor):
    """Run HF model and capture intermediate hidden states via hooks."""
    import torch
    traces = {}

    # Embedding
    embed_out = model.model.embed_tokens(input_ids_tensor)
    traces['embed'] = embed_out.detach().cpu().numpy().squeeze(0)

    # Hook into layer 0 internals
    layer0 = model.model.layers[0]

    # Pre-attention norm
    xn = layer0.input_layernorm(embed_out)
    traces['L0_post_rms_att'] = xn.detach().cpu().numpy().squeeze(0)

    # Q, K, V projections (HF native split-halves format)
    bsz, seq_len, _ = xn.shape

    Q_hf = layer0.self_attn.q_proj(xn)  # [1, seq, q_dim]
    K_hf = layer0.self_attn.k_proj(xn)  # [1, seq, kv_dim]
    V_hf = layer0.self_attn.v_proj(xn)  # [1, seq, kv_dim]

    traces['L0_Q_pre_rope_hf_format'] = Q_hf.detach().cpu().numpy().squeeze(0)
    traces['L0_K_pre_rope_hf_format'] = K_hf.detach().cpu().numpy().squeeze(0)
    traces['L0_V'] = V_hf.detach().cpu().numpy().squeeze(0)

    # Full forward pass with output_hidden_states
    with torch.no_grad():
        out = model(input_ids_tensor, output_hidden_states=True)

    # Hidden states: tuple of (n_layers+1) tensors, each [1, seq, dim]
    for li, hs in enumerate(out.hidden_states):
        if li == 0:
            pass  # embedding, already captured
        else:
            traces[f'L{li-1}_out'] = hs.detach().cpu().numpy().squeeze(0)

    # Final logits
    traces['logits'] = out.logits.detach().cpu().numpy().squeeze(0)  # [seq, vocab]

    return traces


# ============================================================================
# Conversion helpers
# ============================================================================
def interleaved_to_split(x, n_heads, hd):
    """Convert interleaved [re0,im0,re1,im1,...] to split [re0,re1,...,im0,im1,...] per head."""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2
    out = np.empty_like(x)
    out[:, :, :half] = x[:, :, 0::2]
    out[:, :, half:] = x[:, :, 1::2]
    return out.reshape(seq, -1)


def split_to_interleaved(x, n_heads, hd):
    """Convert split [re0,re1,...,im0,im1,...] to interleaved [re0,im0,re1,im1,...] per head."""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2
    out = np.empty_like(x)
    out[:, :, 0::2] = x[:, :, :half]
    out[:, :, 1::2] = x[:, :, half:]
    return out.reshape(seq, -1)


# ============================================================================
# Comparison utilities
# ============================================================================
def compare(name, a, b, atol=1e-4):
    """Compare two arrays, print stats."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False

    diff = np.abs(a - b)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    # Relative diff (avoid div by zero)
    denom = np.maximum(np.abs(a), np.abs(b)) + 1e-30
    max_rel = float((diff / denom).max())

    ok = max_diff < atol
    status = "OK" if ok else "MISMATCH"
    print(f"  {name:40s}  max={max_diff:.6e}  mean={mean_diff:.6e}  "
          f"rel={max_rel:.6e}  [{status}]")
    return ok


def top_k_agreement(logits_a, logits_b, k=5):
    """Check if top-k predicted tokens agree at each position."""
    seq = logits_a.shape[0]
    agreements = []
    for pos in range(seq):
        top_a = set(np.argsort(logits_a[pos])[-k:])
        top_b = set(np.argsort(logits_b[pos])[-k:])
        agreements.append(len(top_a & top_b) / k)
    return agreements


# ============================================================================
# Main
# ============================================================================
def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 78)
    print("LOGIT VERIFICATION: AutoANE numpy vs HuggingFace transformers")
    print("Model: SmolLM2-360M")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: Tokenize a fixed prompt
    # ------------------------------------------------------------------
    print("\n[1] Tokenizing prompt...")
    tok = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    prompt = "Once upon a time there was"
    tokens = tok.encode(prompt)
    print(f"    Prompt: \"{prompt}\"")
    print(f"    Tokens: {tokens}  (len={len(tokens)})")

    tokens_np = np.array(tokens, dtype=np.int64)
    tokens_pt = torch.tensor([tokens], dtype=torch.long)

    # ------------------------------------------------------------------
    # Step 2: Load HuggingFace model
    # ------------------------------------------------------------------
    print("\n[2] Loading HuggingFace model...")
    t0 = time.time()
    hf_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME,
                                                      dtype=torch.float32)
    hf_model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s")

    hf_config = hf_model.config
    print(f"    dim={hf_config.hidden_size} hidden={hf_config.intermediate_size} "
          f"layers={hf_config.num_hidden_layers}")
    print(f"    heads={hf_config.num_attention_heads} "
          f"kv_heads={hf_config.num_key_value_heads} "
          f"hd={hf_config.hidden_size // hf_config.num_attention_heads}")
    hf_rope_theta = getattr(hf_config, 'rope_theta',
                            hf_config.rope_scaling.get('rope_theta', 10000.0)
                            if hasattr(hf_config, 'rope_scaling') and hf_config.rope_scaling
                            else 10000.0)
    print(f"    vocab={hf_config.vocab_size} rope_theta={hf_rope_theta}")

    # ------------------------------------------------------------------
    # Step 3: Load ANE checkpoint
    # ------------------------------------------------------------------
    print("\n[3] Loading ANE checkpoint...")
    t0 = time.time()
    ane_config, ane_layers, ane_rms_final, ane_embed = load_checkpoint(ANE_CKPT)
    ane_config['rope_theta'] = ROPE_THETA
    print(f"    Loaded in {time.time()-t0:.1f}s")
    c = ane_config
    print(f"    dim={c['dim']} hidden={c['hidden']} layers={c['n_layers']}")
    print(f"    heads={c['n_heads']} kv_heads={c['n_kv_heads']} hd={c['hd']}")
    print(f"    vocab={c['vocab']} step={c['step']}")

    n_params = sum(w.size for L in ane_layers for w in L.values()) + \
               ane_rms_final.size + ane_embed.size
    print(f"    Total params: {n_params/1e6:.1f}M")

    # ------------------------------------------------------------------
    # Step 4: Verify weight equivalence between HF and ANE
    # ------------------------------------------------------------------
    print("\n[4] Verifying weight equivalence (HF vs ANE checkpoint)...")
    hf_state = {k: v.float().numpy() for k, v in hf_model.state_dict().items()}

    # Embedding
    hf_embed = hf_state['model.embed_tokens.weight']
    embed_diff = np.max(np.abs(hf_embed - ane_embed))
    print(f"    Embedding max diff: {embed_diff:.6e}")

    # Final RMSNorm
    hf_rms_final = hf_state['model.norm.weight']
    rms_final_diff = np.max(np.abs(hf_rms_final - ane_rms_final))
    print(f"    Final RMSNorm max diff: {rms_final_diff:.6e}")

    # Layer 0 weights
    L = 0
    prefix = f'model.layers.{L}'

    # Wv, Wo, W1, W2, W3 should match exactly (no interleaving)
    for ane_name, hf_suffix in [('Wv', 'self_attn.v_proj.weight'),
                                 ('Wo', 'self_attn.o_proj.weight'),
                                 ('W1', 'mlp.gate_proj.weight'),
                                 ('W2', 'mlp.down_proj.weight'),
                                 ('W3', 'mlp.up_proj.weight'),
                                 ('rms_att', 'input_layernorm.weight'),
                                 ('rms_ffn', 'post_attention_layernorm.weight')]:
        hf_w = hf_state[f'{prefix}.{hf_suffix}']
        ane_w = ane_layers[L][ane_name]
        wd = np.max(np.abs(hf_w - ane_w))
        status = "OK" if wd < 1e-6 else "MISMATCH"
        print(f"    L0.{ane_name:8s} max diff: {wd:.6e}  [{status}]")

    # Wq, Wk should differ by interleaving permutation
    hf_Wq = hf_state[f'{prefix}.self_attn.q_proj.weight']
    ane_Wq = ane_layers[L]['Wq']
    n_heads = c['n_heads']
    hd = c['hd']

    def deinterleave_weight(W, n_h, head_dim):
        """ANE interleaved -> HF split-halves for weight rows."""
        W_out = np.zeros_like(W)
        h2 = head_dim // 2
        for h in range(n_h):
            for i in range(h2):
                src_re = h * head_dim + 2 * i
                src_im = h * head_dim + 2 * i + 1
                dst_re = h * head_dim + i
                dst_im = h * head_dim + h2 + i
                W_out[dst_re] = W[src_re]
                W_out[dst_im] = W[src_im]
        return W_out

    ane_Wq_deinterleaved = deinterleave_weight(ane_Wq, n_heads, hd)
    wq_diff = np.max(np.abs(hf_Wq - ane_Wq_deinterleaved))
    print(f"    L0.Wq       max diff (after deinterleave): {wq_diff:.6e}  "
          f"[{'OK' if wq_diff < 1e-6 else 'MISMATCH'}]")

    hf_Wk = hf_state[f'{prefix}.self_attn.k_proj.weight']
    ane_Wk = ane_layers[L]['Wk']
    ane_Wk_deinterleaved = deinterleave_weight(ane_Wk, c['n_kv_heads'], hd)
    wk_diff = np.max(np.abs(hf_Wk - ane_Wk_deinterleaved))
    print(f"    L0.Wk       max diff (after deinterleave): {wk_diff:.6e}  "
          f"[{'OK' if wk_diff < 1e-6 else 'MISMATCH'}]")

    weights_ok = all(d < 1e-6 for d in [embed_diff, rms_final_diff, wq_diff, wk_diff])
    if not weights_ok:
        print("\n    WARNING: Weight mismatch detected. Logit comparison will be unreliable.")
        print("    The checkpoint may not have been converted from this exact HF model.")

    # ------------------------------------------------------------------
    # Step 5: Run both forward passes
    # ------------------------------------------------------------------
    print("\n[5] Running forward passes...")

    # ANE forward
    t0 = time.time()
    ane_logits = ane_forward_all(tokens_np, ane_config, ane_layers,
                                 ane_rms_final, ane_embed)
    ane_ms = (time.time() - t0) * 1000
    print(f"    ANE numpy:  {ane_ms:.0f}ms  logits shape={ane_logits.shape}")

    # HF forward
    t0 = time.time()
    with torch.no_grad():
        hf_out = hf_model(tokens_pt)
    hf_ms = (time.time() - t0) * 1000
    hf_logits = hf_out.logits.detach().cpu().numpy().squeeze(0)  # [seq, vocab]
    print(f"    HF torch:   {hf_ms:.0f}ms  logits shape={hf_logits.shape}")

    # ------------------------------------------------------------------
    # Step 6: Compare logits
    # ------------------------------------------------------------------
    print("\n[6] Logit comparison (all positions)...")

    diff = np.abs(ane_logits - hf_logits)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    max_pos = np.unravel_index(diff.argmax(), diff.shape)

    print(f"    Max absolute diff:  {max_diff:.6e}  (at pos={max_pos[0]}, vocab_idx={max_pos[1]})")
    print(f"    Mean absolute diff: {mean_diff:.6e}")

    # Per-position stats
    print(f"\n    Per-position max abs diff:")
    for pos in range(len(tokens)):
        pos_diff = diff[pos]
        pos_max = float(pos_diff.max())
        pos_mean = float(pos_diff.mean())
        tok_str = tok.decode([tokens[pos]])
        print(f"      pos={pos} token='{tok_str}' (id={tokens[pos]}):  "
              f"max={pos_max:.6e}  mean={pos_mean:.6e}")

    # ------------------------------------------------------------------
    # Step 7: Top-k token agreement
    # ------------------------------------------------------------------
    print(f"\n[7] Top-k token agreement (k=5)...")
    agreements = top_k_agreement(ane_logits, hf_logits, k=5)
    for pos in range(len(tokens)):
        tok_str = tok.decode([tokens[pos]])
        ane_top5 = np.argsort(ane_logits[pos])[-5:][::-1]
        hf_top5 = np.argsort(hf_logits[pos])[-5:][::-1]
        ane_top5_str = [tok.decode([int(t)]) for t in ane_top5]
        hf_top5_str = [tok.decode([int(t)]) for t in hf_top5]
        match_pct = agreements[pos] * 100
        print(f"    pos={pos} '{tok_str}': {match_pct:.0f}% overlap")
        print(f"      ANE top5: {ane_top5.tolist()} {ane_top5_str}")
        print(f"      HF  top5: {hf_top5.tolist()} {hf_top5_str}")

    # ------------------------------------------------------------------
    # Step 8: Layer-by-layer tracing (if discrepancy found)
    # ------------------------------------------------------------------
    TOLERANCE = 0.01
    if max_diff > TOLERANCE:
        print(f"\n[8] DISCREPANCY DETECTED (max_diff={max_diff:.6e} > {TOLERANCE})")
        print("    Tracing layer by layer to find divergence point...")

        # Run both with tracing
        print("\n    Running ANE layerwise forward...")
        t0 = time.time()
        ane_traces = ane_forward_layerwise(tokens_np, ane_config, ane_layers,
                                            ane_rms_final, ane_embed)
        print(f"    Done in {time.time()-t0:.1f}s")

        print("    Running HF layerwise forward...")
        t0 = time.time()
        hf_traces = hf_forward_layerwise(hf_model, tokens_pt)
        print(f"    Done in {time.time()-t0:.1f}s")

        print("\n    Comparison at each stage:")

        # Embedding
        compare("Embedding lookup", ane_traces['embed'], hf_traces['embed'])

        # Layer 0 RMSNorm
        compare("L0 post-RMSNorm(att)", ane_traces['L0_post_rms_att'],
                hf_traces['L0_post_rms_att'])

        # Q, K, V projections (need format conversion for Q, K)
        # ANE Q is in interleaved format; HF Q is in split-halves format
        ane_Q_as_split = interleaved_to_split(ane_traces['L0_Q_pre_rope'],
                                               c['n_heads'], c['hd'])
        compare("L0 Q projection (ANE->split vs HF)",
                ane_Q_as_split, hf_traces['L0_Q_pre_rope_hf_format'])

        ane_K_as_split = interleaved_to_split(ane_traces['L0_K_pre_rope'],
                                               c['n_kv_heads'], c['hd'])
        compare("L0 K projection (ANE->split vs HF)",
                ane_K_as_split, hf_traces['L0_K_pre_rope_hf_format'])

        compare("L0 V projection", ane_traces['L0_V'], hf_traces['L0_V'])

        # Layer outputs
        for li in range(min(c['n_layers'], 32)):
            key = f'L{li}_out'
            if key in ane_traces and key in hf_traces:
                ok = compare(f"Layer {li} output", ane_traces[key], hf_traces[key])
                if not ok and li > 0:
                    # Check if previous layer was OK
                    prev_key = f'L{li-1}_out'
                    if prev_key in ane_traces:
                        prev_diff = np.max(np.abs(
                            ane_traces[prev_key] - hf_traces[prev_key]))
                        if prev_diff < TOLERANCE:
                            print(f"    >>> Divergence begins at layer {li} "
                                  f"(L{li-1} diff={prev_diff:.6e})")
                            break

        compare("Final logits", ane_traces['logits'], hf_traces['logits'])

    else:
        print(f"\n[8] No layer tracing needed (max_diff={max_diff:.6e} < {TOLERANCE})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Max absolute logit diff:  {max_diff:.6e}")
    print(f"  Mean absolute logit diff: {mean_diff:.6e}")
    print(f"  Top-5 agreement:          {np.mean(agreements)*100:.1f}%")

    if max_diff < TOLERANCE:
        print(f"\n  PASS: Logits match within tolerance ({TOLERANCE})")
        print(f"  AutoANE numpy forward pass is numerically equivalent to HuggingFace.")
    else:
        print(f"\n  FAIL: Logits differ beyond tolerance ({TOLERANCE})")
        print(f"  See layer-by-layer trace above to identify divergence point.")

    return 0 if max_diff < TOLERANCE else 1


if __name__ == '__main__':
    exit(main())
