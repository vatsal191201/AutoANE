#!/usr/bin/env python3
"""Verify that HuggingFace → ANE checkpoint conversion preserves all weights exactly.

Loads the HuggingFace model (HuggingFaceTB/SmolLM2-360M) and the ANE binary checkpoint,
then compares every weight matrix, reporting max absolute error per weight and overall verdict.
"""

import struct
import numpy as np
import sys
import os

# ── Model constants (from smollm2_360m.h) ──
DIM      = 960
HIDDEN   = 2560
HEADS    = 15
KV_HEADS = 5
HD       = 64
Q_DIM    = HEADS * HD    # 960
KV_DIM   = KV_HEADS * HD # 320
SEQ      = 256
NLAYERS  = 32
VOCAB    = 49152

# Derived weight sizes
WQ_SZ = Q_DIM * DIM      # 960*960 = 921600
WK_SZ = KV_DIM * DIM     # 320*960 = 307200
WV_SZ = KV_DIM * DIM     # 320*960 = 307200
WO_SZ = DIM * Q_DIM      # 960*960 = 921600
W1_SZ = HIDDEN * DIM     # 2560*960 = 2457600
W2_SZ = DIM * HIDDEN     # 960*2560 = 2457600
W3_SZ = HIDDEN * DIM     # 2560*960 = 2457600
LAYER_PARAMS = WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2 * DIM

# Header format: '<iiiiiiiiiiffdddiiiiii' = 96 bytes
HEADER_FMT = '<iiiiiiiiiiffdddiiiiii'
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 96


def interleave_weights(W, n_heads, head_dim):
    """Convert Q/K weight matrix from HF (non-interleaved) to ANE (interleaved) RoPE ordering.
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


def compare_weights(name, hf_w, ane_w):
    """Compare two weight arrays. Returns (max_abs_error, match_bool)."""
    if hf_w.shape != ane_w.shape:
        print(f"  SHAPE MISMATCH {name}: HF {hf_w.shape} vs ANE {ane_w.shape}")
        return float('inf'), False
    diff = np.abs(hf_w - ane_w)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    exact = (max_err == 0.0)
    status = "EXACT" if exact else f"MAX_ERR={max_err:.2e} MEAN_ERR={mean_err:.2e}"
    print(f"  {name:30s}  shape={str(hf_w.shape):20s}  {status}")
    return max_err, exact


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_model_name = "HuggingFaceTB/SmolLM2-360M"
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "training", "ane_smollm2_360m_clean.bin")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: ANE checkpoint not found at {ckpt_path}")
        sys.exit(1)

    # ── 1. Load HuggingFace model ──
    print(f"Loading HuggingFace model: {hf_model_name}")
    config = AutoConfig.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float32)
    state = {k: v.float().numpy() for k, v in model.state_dict().items()}

    # Verify architecture matches
    assert config.hidden_size == DIM, f"DIM mismatch: {config.hidden_size} vs {DIM}"
    assert config.intermediate_size == HIDDEN, f"HIDDEN mismatch: {config.intermediate_size} vs {HIDDEN}"
    assert config.num_hidden_layers == NLAYERS, f"NLAYERS mismatch: {config.num_hidden_layers} vs {NLAYERS}"
    assert config.num_attention_heads == HEADS, f"HEADS mismatch: {config.num_attention_heads} vs {HEADS}"
    assert config.num_key_value_heads == KV_HEADS, f"KV_HEADS mismatch: {config.num_key_value_heads} vs {KV_HEADS}"
    assert config.vocab_size == VOCAB, f"VOCAB mismatch: {config.vocab_size} vs {VOCAB}"
    print(f"Architecture verified: dim={DIM} hidden={HIDDEN} layers={NLAYERS} "
          f"heads={HEADS}/{KV_HEADS} hd={HD} vocab={VOCAB}")

    # ── 2. Load ANE checkpoint binary ──
    print(f"\nLoading ANE checkpoint: {ckpt_path}")
    file_size = os.path.getsize(ckpt_path)
    print(f"  File size: {file_size / 1e6:.1f} MB")

    with open(ckpt_path, 'rb') as f:
        # Read header
        header_bytes = f.read(HEADER_SIZE)
        header = struct.unpack(HEADER_FMT, header_bytes)
        magic = header[0]
        version = header[1]
        step = header[2]
        n_layers_h = header[4]
        vocab_h = header[5]
        dim_h = header[6]
        hidden_h = header[7]
        n_heads_h = header[8]
        seq_h = header[9]
        kv_heads_h = header[18]
        hd_h = header[19]
        q_dim_h = header[20]

        print(f"  Header: magic=0x{magic:08X} version={version} step={step}")
        print(f"  n_layers={n_layers_h} vocab={vocab_h} dim={dim_h} hidden={hidden_h}")
        print(f"  heads={n_heads_h} kv_heads={kv_heads_h} hd={hd_h} q_dim={q_dim_h}")

        assert magic == 0x424C5A54, f"Bad magic: 0x{magic:08X}"
        assert version == 4, f"Bad version: {version}"
        assert n_layers_h == NLAYERS
        assert vocab_h == VOCAB
        assert dim_h == DIM
        assert hidden_h == HIDDEN
        assert n_heads_h == HEADS
        assert kv_heads_h == KV_HEADS
        assert hd_h == HD
        assert q_dim_h == Q_DIM

        # ── 3. Read per-layer weights from checkpoint ──
        # Layout per layer (matching mezo_save_checkpoint / mezo_load_checkpoint):
        #   Weights: Wq, Wk, Wv, Wo, W1, W2, W3, rms_att, rms_ffn
        #   Adam m/v: (Wq_m, Wq_v, Wk_m, Wk_v, Wv_m, Wv_v, Wo_m, Wo_v,
        #              W1_m, W1_v, W2_m, W2_v, W3_m, W3_v,
        #              rms_att_m, rms_att_v, rms_ffn_m, rms_ffn_v)
        # Note: hf_to_ane.py writes adam zeros as contiguous block of layer_params*2,
        # but since all are zeros, total byte count is the same:
        #   mezo format: 2*(WQ_SZ+WK_SZ+WV_SZ+WO_SZ+W1_SZ+W2_SZ+W3_SZ+DIM+DIM) = 2*LAYER_PARAMS
        #   hf_to_ane:   layer_params*2 where layer_params = LAYER_PARAMS → same total

        adam_per_layer = 2 * LAYER_PARAMS  # total adam floats per layer

        all_errors = []
        all_exact = True

        print(f"\n{'='*80}")
        print("LAYER-BY-LAYER WEIGHT COMPARISON")
        print(f"{'='*80}")

        for L in range(NLAYERS):
            print(f"\n--- Layer {L} ---")
            prefix = f'model.layers.{L}'

            # Read weights from checkpoint
            ane_Wq = np.frombuffer(f.read(WQ_SZ * 4), dtype=np.float32).reshape(Q_DIM, DIM)
            ane_Wk = np.frombuffer(f.read(WK_SZ * 4), dtype=np.float32).reshape(KV_DIM, DIM)
            ane_Wv = np.frombuffer(f.read(WV_SZ * 4), dtype=np.float32).reshape(KV_DIM, DIM)
            ane_Wo = np.frombuffer(f.read(WO_SZ * 4), dtype=np.float32).reshape(DIM, Q_DIM)
            ane_W1 = np.frombuffer(f.read(W1_SZ * 4), dtype=np.float32).reshape(HIDDEN, DIM)
            ane_W2 = np.frombuffer(f.read(W2_SZ * 4), dtype=np.float32).reshape(DIM, HIDDEN)
            ane_W3 = np.frombuffer(f.read(W3_SZ * 4), dtype=np.float32).reshape(HIDDEN, DIM)
            ane_rms_att = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
            ane_rms_ffn = np.frombuffer(f.read(DIM * 4), dtype=np.float32)

            # Skip Adam state
            f.read(adam_per_layer * 4)

            # Get HF weights
            hf_Wq = state[f'{prefix}.self_attn.q_proj.weight']
            hf_Wk = state[f'{prefix}.self_attn.k_proj.weight']
            hf_Wv = state[f'{prefix}.self_attn.v_proj.weight']
            hf_Wo = state[f'{prefix}.self_attn.o_proj.weight']
            hf_W1 = state[f'{prefix}.mlp.gate_proj.weight']
            hf_W2 = state[f'{prefix}.mlp.down_proj.weight']
            hf_W3 = state[f'{prefix}.mlp.up_proj.weight']
            hf_rms_att = state[f'{prefix}.input_layernorm.weight']
            hf_rms_ffn = state[f'{prefix}.post_attention_layernorm.weight']

            # Apply interleaving to Q/K (same transform as hf_to_ane.py)
            hf_Wq_interleaved = interleave_weights(hf_Wq, HEADS, HD)
            hf_Wk_interleaved = interleave_weights(hf_Wk, KV_HEADS, HD)

            # Compare: Wq and Wk use interleaved HF weight
            err, exact = compare_weights(f"L{L}.Wq (interleaved)", hf_Wq_interleaved, ane_Wq)
            all_errors.append((f"L{L}.Wq", err)); all_exact &= exact

            err, exact = compare_weights(f"L{L}.Wk (interleaved)", hf_Wk_interleaved, ane_Wk)
            all_errors.append((f"L{L}.Wk", err)); all_exact &= exact

            # Compare: Wv, Wo, W1, W2, W3, rms_att, rms_ffn — direct match
            for name, hf_w, ane_w in [
                (f"L{L}.Wv", hf_Wv, ane_Wv),
                (f"L{L}.Wo", hf_Wo, ane_Wo),
                (f"L{L}.W1", hf_W1, ane_W1),
                (f"L{L}.W2", hf_W2, ane_W2),
                (f"L{L}.W3", hf_W3, ane_W3),
                (f"L{L}.rms_att", hf_rms_att, ane_rms_att),
                (f"L{L}.rms_ffn", hf_rms_ffn, ane_rms_ffn),
            ]:
                err, exact = compare_weights(name, hf_w, ane_w)
                all_errors.append((name, err)); all_exact &= exact

        # ── 4. Read rms_final ──
        print(f"\n--- Final weights ---")
        ane_rms_final = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
        f.read(DIM * 2 * 4)  # skip adam m/v for rms_final

        hf_rms_final = state['model.norm.weight']
        err, exact = compare_weights("rms_final", hf_rms_final, ane_rms_final)
        all_errors.append(("rms_final", err)); all_exact &= exact

        # ── 5. Read embed_tokens ──
        ane_embed = np.frombuffer(f.read(VOCAB * DIM * 4), dtype=np.float32).reshape(VOCAB, DIM)
        # Don't need to read remaining adam state

        hf_embed = state['model.embed_tokens.weight']
        err, exact = compare_weights("embed_tokens", hf_embed, ane_embed)
        all_errors.append(("embed_tokens", err)); all_exact &= exact

        # Check we consumed the right amount of data
        pos = f.tell()
        remaining_expected = VOCAB * DIM * 2 * 4  # adam m/v for embed
        expected_total = HEADER_SIZE + NLAYERS * (LAYER_PARAMS + adam_per_layer) * 4 + \
                         (DIM + DIM * 2) * 4 + (VOCAB * DIM + VOCAB * DIM * 2) * 4
        print(f"\n  File position after weights: {pos} / {file_size}")
        print(f"  Remaining (adam for embed): {file_size - pos} bytes "
              f"(expected {remaining_expected} bytes)")

    # ── 6. Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    global_max_err = max(e for _, e in all_errors)
    n_exact = sum(1 for _, e in all_errors if e == 0.0)
    n_total = len(all_errors)

    print(f"Total weight matrices compared: {n_total}")
    print(f"Exact matches (max_err == 0.0): {n_exact}/{n_total}")
    print(f"Global max absolute error:      {global_max_err:.2e}")

    if not all_exact:
        print("\nNon-exact weights:")
        for name, err in all_errors:
            if err > 0.0:
                print(f"  {name}: max_err = {err:.2e}")

    print(f"\n{'='*80}")
    if all_exact:
        print("VERDICT: PASS -- All weights match EXACTLY (bitwise identical)")
    else:
        print(f"VERDICT: FAIL -- {n_total - n_exact} weight(s) have non-zero error")
    print(f"{'='*80}")

    return 0 if all_exact else 1


if __name__ == '__main__':
    sys.exit(main())
