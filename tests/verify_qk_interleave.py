#!/usr/bin/env python3
"""Verify Q/K weight interleaving done by hf_to_ane.py is correct.

Two tests:
  1. Element-wise: load HF Wq, interleave manually, compare to checkpoint Wq
  2. Functional: verify that HF-RoPE(Wq_hf @ x) == ANE-RoPE(Wq_interleaved @ x)

Config (SmolLM2-360M): n_heads=15, kv_heads=5, hd=64, dim=960, q_dim=960
"""

import struct
import numpy as np
import sys
import os

# ─── Config ───────────────────────────────────────────────────────────────────
N_HEADS   = 15
KV_HEADS  = 5
HD        = 64
DIM       = 960
Q_DIM     = N_HEADS * HD   # 960
KV_DIM    = KV_HEADS * HD  # 320
HIDDEN    = 2560  # SmolLM2-360M intermediate_size
HEADER_SZ = 96
ROPE_THETA = 100000.0  # SmolLM2 uses 100000

# ─── Interleave function (copied from hf_to_ane.py) ──────────────────────────
def interleave_weights(W, n_heads, head_dim):
    """HF (non-interleaved) -> ANE (interleaved) RoPE ordering.
    HF:  [re0, re1, ..., re31, im0, im1, ..., im31] per head
    ANE: [re0, im0, re1, im1, ..., re31, im31] per head
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

# ─── Interleave vectors (for comparing outputs) ─────────────────────────────
def interleave_vectors(v, n_heads, hd):
    """Convert Q/K vectors from HF layout to ANE layout.
    HF:  [re0, re1, ..., re31, im0, im1, ..., im31] per head
    ANE: [re0, im0, re1, im1, ..., re31, im31] per head
    v shape: [seq, n_heads * hd]
    """
    seq = v.shape[0]
    half = hd // 2
    v = v.reshape(seq, n_heads, hd)
    out = np.empty_like(v)
    out[:, :, 0::2] = v[:, :, :half]
    out[:, :, 1::2] = v[:, :, half:]
    return out.reshape(seq, -1)

# ─── RoPE: HF style (non-interleaved pairs) ─────────────────────────────────
def apply_rope_hf(x, n_heads, hd, rope_theta=ROPE_THETA):
    """RoPE with HF convention: first half = re, second half = im per head.
    x: [seq, n_heads*hd] -> [seq, n_heads*hd]
    """
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2

    freqs = 1.0 / (rope_theta ** (2.0 * np.arange(half, dtype=np.float64) / hd))
    theta = np.arange(seq, dtype=np.float64)[:, None] * freqs[None, :]  # [seq, half]
    cos_t = np.cos(theta)[:, None, :]  # [seq, 1, half]
    sin_t = np.sin(theta)[:, None, :]

    # HF convention: x[:, :, :half] = re, x[:, :, half:] = im
    x_re = x[:, :, :half]
    x_im = x[:, :, half:]
    out = np.empty_like(x)
    out[:, :, :half] = x_re * cos_t - x_im * sin_t
    out[:, :, half:] = x_re * sin_t + x_im * cos_t
    return out.reshape(seq, -1)

# ─── RoPE: ANE style (interleaved pairs) ────────────────────────────────────
def apply_rope_ane(x, n_heads, hd, rope_theta=ROPE_THETA):
    """RoPE with ANE convention: interleaved (re, im) pairs per head.
    x: [seq, n_heads*hd] -> [seq, n_heads*hd]
    """
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2

    freqs = 1.0 / (rope_theta ** (2.0 * np.arange(half, dtype=np.float64) / hd))
    theta = np.arange(seq, dtype=np.float64)[:, None] * freqs[None, :]  # [seq, half]
    cos_t = np.cos(theta)[:, None, :]  # [seq, 1, half]
    sin_t = np.sin(theta)[:, None, :]

    # ANE convention: even indices = re, odd indices = im
    x_re = x[:, :, 0::2]   # [seq, n_heads, half]
    x_im = x[:, :, 1::2]   # [seq, n_heads, half]
    out = np.empty_like(x)
    out[:, :, 0::2] = x_re * cos_t - x_im * sin_t
    out[:, :, 1::2] = x_re * sin_t + x_im * cos_t
    return out.reshape(seq, -1)


def main():
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'training', 'ane_smollm2_360m_ckpt.bin')
    ckpt_path = os.path.abspath(ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Load HF Wq for layer 0
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Loading SmolLM2-360M from HuggingFace...")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                                  torch_dtype=torch.float32)
    state = model.state_dict()

    Wq_hf = state['model.layers.0.self_attn.q_proj.weight'].float().numpy()
    Wk_hf = state['model.layers.0.self_attn.k_proj.weight'].float().numpy()
    print(f"  HF Wq shape: {Wq_hf.shape} (expected [{Q_DIM}, {DIM}])")
    print(f"  HF Wk shape: {Wk_hf.shape} (expected [{KV_DIM}, {DIM}])")
    assert Wq_hf.shape == (Q_DIM, DIM), f"Unexpected Wq shape: {Wq_hf.shape}"
    assert Wk_hf.shape == (KV_DIM, DIM), f"Unexpected Wk shape: {Wk_hf.shape}"

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Manually interleave HF Wq
    # ═══════════════════════════════════════════════════════════════════════
    print("\nInterleaving HF Wq with our formula...")
    Wq_interleaved = interleave_weights(Wq_hf, N_HEADS, HD)
    Wk_interleaved = interleave_weights(Wk_hf, KV_HEADS, HD)
    print(f"  Interleaved Wq shape: {Wq_interleaved.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Load Wq from ANE checkpoint (layer 0)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\nLoading Wq from checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        # Read header
        hdr = struct.unpack('<iiiiiiiiiiffdddiiiiii', f.read(HEADER_SZ))
        magic = hdr[0]
        assert magic == 0x424C5A54, f"Bad magic: {hex(magic)}"
        print(f"  Header: magic=0x{magic:08X} version={hdr[1]} step={hdr[2]}")
        print(f"  n_layers={hdr[4]} vocab={hdr[5]} dim={hdr[6]} hidden={hdr[7]}")
        print(f"  n_heads={hdr[8]} kv_heads={hdr[18]} hd={hdr[19]} q_dim={hdr[20]}")

        # Verify config matches
        assert hdr[6] == DIM, f"dim mismatch: {hdr[6]} vs {DIM}"
        assert hdr[8] == N_HEADS, f"n_heads mismatch: {hdr[8]} vs {N_HEADS}"
        assert hdr[18] == KV_HEADS, f"kv_heads mismatch: {hdr[18]} vs {KV_HEADS}"
        assert hdr[19] == HD, f"hd mismatch: {hdr[19]} vs {HD}"
        assert hdr[20] == Q_DIM, f"q_dim mismatch: {hdr[20]} vs {Q_DIM}"

        # Layer 0 Wq is immediately after header
        Wq_ckpt = np.frombuffer(f.read(Q_DIM * DIM * 4), np.float32).reshape(Q_DIM, DIM).copy()
        Wk_ckpt = np.frombuffer(f.read(KV_DIM * DIM * 4), np.float32).reshape(KV_DIM, DIM).copy()

    print(f"  Checkpoint Wq shape: {Wq_ckpt.shape}")
    print(f"  Checkpoint Wk shape: {Wk_ckpt.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Element-wise comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: Element-wise comparison (interleaved HF vs checkpoint)")
    print("=" * 70)

    diff_q = np.abs(Wq_interleaved - Wq_ckpt)
    max_diff_q = diff_q.max()
    mean_diff_q = diff_q.mean()
    print(f"  Wq max |diff|:  {max_diff_q:.2e}")
    print(f"  Wq mean |diff|: {mean_diff_q:.2e}")

    diff_k = np.abs(Wk_interleaved - Wk_ckpt)
    max_diff_k = diff_k.max()
    mean_diff_k = diff_k.mean()
    print(f"  Wk max |diff|:  {max_diff_k:.2e}")
    print(f"  Wk mean |diff|: {mean_diff_k:.2e}")

    if max_diff_q == 0.0 and max_diff_k == 0.0:
        print("  >>> PASS: Wq and Wk are BIT-EXACT matches!")
    elif max_diff_q < 1e-6 and max_diff_k < 1e-6:
        print("  >>> PASS: Wq and Wk match within float32 tolerance")
    else:
        print("  >>> FAIL: Significant difference detected!")
        # Debug: show where differences are
        bad_q = np.where(diff_q > 1e-6)
        if len(bad_q[0]) > 0:
            print(f"  First 5 mismatched Wq positions: rows={bad_q[0][:5]}, cols={bad_q[1][:5]}")
            for i in range(min(5, len(bad_q[0]))):
                r, c = bad_q[0][i], bad_q[1][i]
                print(f"    [{r},{c}]: interleaved={Wq_interleaved[r,c]:.8f}, ckpt={Wq_ckpt[r,c]:.8f}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Functional test — RoPE equivalence
    # ═══════════════════════════════════════════════════════════════════════
    #
    # The key insight: HF and ANE use DIFFERENT layouts for the Q/K vectors.
    #   HF:  [re0, re1, ..., re31, im0, im1, ..., im31] per head
    #   ANE: [re0, im0, re1, im1, ..., re31, im31] per head
    #
    # After applying RoPE in each convention, the RESULTS are also in
    # different layouts. So we must convert to a common layout before
    # comparing. We convert both to a canonical (re, im) pair list.
    #
    # The test verifies the END-TO-END equivalence:
    #   interleave_to_ane(HF_RoPE(Wq_hf @ x)) == ANE_RoPE(Wq_ane @ x)
    #
    print("\n" + "=" * 70)
    print("TEST 2: Functional test — HF-RoPE(Wq_hf @ x) vs ANE-RoPE(Wq_ane @ x)")
    print("=" * 70)

    np.random.seed(42)
    seq_len = 4  # test with a few positions

    # Random input: [seq_len, DIM]
    x = np.random.randn(seq_len, DIM).astype(np.float64)

    # Path A: HF convention
    # Q_hf = x @ Wq_hf^T  (using HF non-interleaved weights)
    Q_hf = (x @ Wq_hf.astype(np.float64).T)  # [seq_len, Q_DIM]
    Q_hf_roped = apply_rope_hf(Q_hf, N_HEADS, HD, ROPE_THETA)

    # Path B: ANE convention
    # Q_ane = x @ Wq_interleaved^T  (using interleaved weights from checkpoint)
    Q_ane = (x @ Wq_ckpt.astype(np.float64).T)  # [seq_len, Q_DIM]
    Q_ane_roped = apply_rope_ane(Q_ane, N_HEADS, HD, ROPE_THETA)

    # Convert HF-RoPE output to ANE layout for comparison
    # HF layout per head: [re0..re31, im0..im31]
    # ANE layout per head: [re0, im0, re1, im1, ..., re31, im31]
    Q_hf_roped_as_ane = interleave_vectors(Q_hf_roped, N_HEADS, HD)

    diff_func = np.abs(Q_hf_roped_as_ane - Q_ane_roped)
    max_diff_func = diff_func.max()
    mean_diff_func = diff_func.mean()

    print(f"  Max |interleave(HF_RoPE(Wq_hf@x)) - ANE_RoPE(Wq_ane@x)|:  {max_diff_func:.2e}")
    print(f"  Mean:                                                       {mean_diff_func:.2e}")

    # Also compare pre-RoPE Q values to check the interleaving is just a permutation
    Q_hf_as_ane = interleave_vectors(Q_hf, N_HEADS, HD)
    diff_pre_rope = np.abs(Q_hf_as_ane - Q_ane).max()
    print(f"\n  Sanity check (pre-RoPE, interleave(Wq_hf@x) vs Wq_ane@x): {diff_pre_rope:.2e}")
    print(f"    Q_hf norm:  {np.linalg.norm(Q_hf):.6f}")
    print(f"    Q_ane norm: {np.linalg.norm(Q_ane):.6f}")

    # Verify the interleaving is a row permutation: for each head,
    # the set of row values should be the same
    print(f"\n  Verifying row permutation property (head 0):")
    hf_head0 = Wq_hf[:HD, :]      # first head, HF layout
    ane_head0 = Wq_ckpt[:HD, :]    # first head, ANE layout
    # In HF: rows 0..31 are re, rows 32..63 are im
    # In ANE: rows 0,2,4,...,62 are re, rows 1,3,5,...,63 are im
    hf_re_rows = set(tuple(hf_head0[i].tolist()) for i in range(HD//2))
    ane_re_rows = set(tuple(ane_head0[2*i].tolist()) for i in range(HD//2))
    hf_im_rows = set(tuple(hf_head0[HD//2 + i].tolist()) for i in range(HD//2))
    ane_im_rows = set(tuple(ane_head0[2*i + 1].tolist()) for i in range(HD//2))
    print(f"    RE rows match: {hf_re_rows == ane_re_rows}")
    print(f"    IM rows match: {hf_im_rows == ane_im_rows}")

    if max_diff_func < 1e-10:
        print(f"\n  >>> PASS: RoPE outputs are numerically identical (diff < 1e-10)")
    elif max_diff_func < 1e-6:
        print(f"\n  >>> PASS: RoPE outputs match within float64 tolerance")
    else:
        print(f"\n  >>> FAIL: RoPE outputs differ significantly!")

    # ═══════════════════════════════════════════════════════════════════════
    # Also test K weights
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("TEST 2b: Same functional test for K weights")
    print("-" * 70)

    K_hf = (x @ Wk_hf.astype(np.float64).T)
    K_hf_roped = apply_rope_hf(K_hf, KV_HEADS, HD, ROPE_THETA)

    K_ane = (x @ Wk_ckpt.astype(np.float64).T)
    K_ane_roped = apply_rope_ane(K_ane, KV_HEADS, HD, ROPE_THETA)

    K_hf_roped_as_ane = interleave_vectors(K_hf_roped, KV_HEADS, HD)

    diff_k_func = np.abs(K_hf_roped_as_ane - K_ane_roped)
    max_diff_k_func = diff_k_func.max()
    mean_diff_k_func = diff_k_func.mean()

    print(f"  Max |interleave(HF_RoPE(Wk_hf@x)) - ANE_RoPE(Wk_ane@x)|:  {max_diff_k_func:.2e}")
    print(f"  Mean:                                                       {mean_diff_k_func:.2e}")

    if max_diff_k_func < 1e-10:
        print(f"  >>> PASS: K RoPE outputs are numerically identical")
    elif max_diff_k_func < 1e-6:
        print(f"  >>> PASS: K RoPE outputs match within float64 tolerance")
    else:
        print(f"  >>> FAIL: K RoPE outputs differ significantly!")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test 1 (element-wise):  Wq max diff = {max_diff_q:.2e}, Wk max diff = {max_diff_k:.2e}")
    print(f"  Test 2 (functional Q):  max diff = {max_diff_func:.2e}")
    print(f"  Test 2b (functional K): max diff = {max_diff_k_func:.2e}")

    all_pass = (max_diff_q < 1e-6 and max_diff_k < 1e-6 and
                max_diff_func < 1e-6 and max_diff_k_func < 1e-6)
    if all_pass:
        print("\n  ALL TESTS PASSED — interleaving is correct!")
    else:
        print("\n  SOME TESTS FAILED — investigate above details")
        sys.exit(1)


if __name__ == '__main__':
    main()
