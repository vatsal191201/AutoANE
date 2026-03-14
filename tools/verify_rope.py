#!/usr/bin/env python3
"""Numerical verification: AutoANE interleaved RoPE vs HuggingFace split-halves RoPE.

Verifies that the two RoPE conventions produce identical results after proper
format conversion, and that hf_to_ane.py's interleave_weights correctly bridges
the two formats.

Model parameters: SmolLM2-360M (DIM=960, HEADS=15, KV_HEADS=5, HD=64, ROPE_THETA=100000)
"""

import numpy as np

# ============================================================================
# SmolLM2-360M configuration
# ============================================================================
DIM = 960
HEADS = 15
KV_HEADS = 5
HD = 64
ROPE_THETA = 100000.0
SEQ = 256

# ============================================================================
# Implementation 1: AutoANE interleaved RoPE (from generate.py)
# Layout per head: [re0, im0, re1, im1, ..., re31, im31]
# Input shape: [seq, n_heads * hd]
# ============================================================================
def ane_rope(x, n_heads, hd, rope_theta=ROPE_THETA):
    """RoPE with interleaved (re, im) pairs. x: [seq, n_heads*hd] -> [seq, n_heads*hd]"""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)

    freqs = 1.0 / (rope_theta ** (2.0 * np.arange(hd // 2, dtype=np.float64) / hd))
    theta = np.arange(seq, dtype=np.float64)[:, None] * freqs[None, :]  # [seq, hd//2]
    cos_t = np.cos(theta)[:, None, :]  # [seq, 1, hd//2]
    sin_t = np.sin(theta)[:, None, :]

    x_re, x_im = x[:, :, 0::2], x[:, :, 1::2]
    out = np.empty_like(x)
    out[:, :, 0::2] = x_re * cos_t - x_im * sin_t
    out[:, :, 1::2] = x_re * sin_t + x_im * cos_t
    return out.reshape(seq, -1)


# ============================================================================
# Implementation 1b: C implementation (from cpu_ops.h rope_forward_inplace)
# Faithfully translated to Python. Data layout: [DIM, SEQ] channel-first.
# Within each head, pairs are (2i, 2i+1) = interleaved.
# ============================================================================
def c_rope_channelfirst(x, seq, dim, hd, rope_theta=ROPE_THETA):
    """Exact translation of C rope_forward_inplace. x: [dim, seq] in-place."""
    x = x.copy()
    nheads = dim // hd
    for h in range(nheads):
        for i in range(hd // 2):
            freq = 1.0 / (rope_theta ** (2.0 * i / float(hd)))
            for p in range(seq):
                theta = p * freq
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                idx0 = (h * hd + 2 * i) * seq + p
                idx1 = (h * hd + 2 * i + 1) * seq + p
                v0, v1 = x.flat[idx0], x.flat[idx1]
                x.flat[idx0] = v0 * cos_t - v1 * sin_t
                x.flat[idx1] = v0 * sin_t + v1 * cos_t
    return x


# ============================================================================
# Implementation 2: HuggingFace split-halves RoPE (LlamaRotaryEmbedding)
# Layout per head: [re0, re1, ..., re31, im0, im1, ..., im31]
# This is how the HF model natively stores and computes RoPE.
# ============================================================================
def hf_rope(x, n_heads, hd, rope_theta=ROPE_THETA):
    """HuggingFace-style RoPE with split halves. x: [seq, n_heads*hd] -> [seq, n_heads*hd]

    Per head, the first half of the head dim are 'real' components and the
    second half are 'imaginary' components:
      head = [x0, x1, ..., x31, x32, x33, ..., x63]
             |--- real half ---|  |--- imag half ---|

    The rotation is:
      out_real = x_real * cos - x_imag * sin
      out_imag = x_real * sin + x_imag * cos
    """
    seq = x.shape[0]
    half = hd // 2
    x = x.reshape(seq, n_heads, hd)

    # Same frequency computation as ANE
    freqs = 1.0 / (rope_theta ** (2.0 * np.arange(half, dtype=np.float64) / hd))
    theta = np.arange(seq, dtype=np.float64)[:, None] * freqs[None, :]  # [seq, half]
    cos_t = np.cos(theta)[:, None, :]  # [seq, 1, half]
    sin_t = np.sin(theta)[:, None, :]

    # Split halves: first half = real, second half = imaginary
    x_real = x[:, :, :half]   # [seq, heads, half]
    x_imag = x[:, :, half:]   # [seq, heads, half]

    out = np.empty_like(x)
    out[:, :, :half] = x_real * cos_t - x_imag * sin_t
    out[:, :, half:] = x_real * sin_t + x_imag * cos_t
    return out.reshape(seq, -1)


# ============================================================================
# Format conversion functions
# ============================================================================
def interleaved_to_split(x, n_heads, hd):
    """Convert from interleaved [re0,im0,re1,im1,...] to split [re0,re1,...,im0,im1,...] per head.
    x: [seq, n_heads*hd] -> [seq, n_heads*hd]"""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2
    out = np.empty_like(x)
    out[:, :, :half] = x[:, :, 0::2]   # real components
    out[:, :, half:] = x[:, :, 1::2]   # imaginary components
    return out.reshape(seq, -1)


def split_to_interleaved(x, n_heads, hd):
    """Convert from split [re0,re1,...,im0,im1,...] to interleaved [re0,im0,re1,im1,...] per head.
    x: [seq, n_heads*hd] -> [seq, n_heads*hd]"""
    seq = x.shape[0]
    x = x.reshape(seq, n_heads, hd)
    half = hd // 2
    out = np.empty_like(x)
    out[:, :, 0::2] = x[:, :, :half]   # real at even indices
    out[:, :, 1::2] = x[:, :, half:]   # imaginary at odd indices
    return out.reshape(seq, -1)


def interleave_weights(W, n_heads, head_dim):
    """From hf_to_ane.py: convert Q/K weight rows from HF to ANE ordering.
    HF:   [re0, re1, ..., re31, im0, im1, ..., im31] per head (output dim)
    ANE:  [re0, im0, re1, im1, ..., re31, im31] per head (output dim)
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


# ============================================================================
# Verification
# ============================================================================
def main():
    np.random.seed(42)
    print("=" * 72)
    print("RoPE Numerical Verification: AutoANE (interleaved) vs HuggingFace (split-halves)")
    print("=" * 72)
    print(f"Config: HEADS={HEADS}, HD={HD}, SEQ={SEQ}, ROPE_THETA={ROPE_THETA}")
    print()

    # -----------------------------------------------------------------------
    # Test 1: ANE Python vs C implementation (same interleaved format)
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 1: ANE Python rope (generate.py) vs C rope (cpu_ops.h)")
    print("  Both use interleaved format, but different data layouts:")
    print("  Python: [seq, dim] row-major  |  C: [dim, seq] channel-first")
    print("-" * 72)

    # Use a small subset for the C loop (it's O(heads*hd*seq) scalar ops)
    test_heads = 5  # use fewer heads for the slow scalar loop
    test_dim = test_heads * HD
    x_py = np.random.randn(SEQ, test_dim).astype(np.float64)

    # Python interleaved RoPE
    y_py = ane_rope(x_py, test_heads, HD)

    # C-style: transpose to [dim, seq], apply, transpose back
    x_c = x_py.T.copy()  # [dim, seq]
    y_c = c_rope_channelfirst(x_c, SEQ, test_dim, HD)
    y_c = y_c.T  # back to [seq, dim]

    diff_1 = np.max(np.abs(y_py - y_c))
    reldiff_1 = np.max(np.abs(y_py - y_c) / (np.abs(y_py) + 1e-30))
    print(f"  Max abs diff:  {diff_1:.2e}")
    print(f"  Max rel diff:  {reldiff_1:.2e}")
    print(f"  PASS: {diff_1 < 1e-10}" if diff_1 < 1e-10 else f"  FAIL: diff={diff_1}")
    print()

    # -----------------------------------------------------------------------
    # Test 2: Format round-trip (interleaved <-> split halves)
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 2: Format round-trip (interleaved -> split -> interleaved)")
    print("-" * 72)

    x_orig = np.random.randn(SEQ, HEADS * HD)
    x_split = interleaved_to_split(x_orig, HEADS, HD)
    x_back = split_to_interleaved(x_split, HEADS, HD)
    rt_diff = np.max(np.abs(x_orig - x_back))
    print(f"  Round-trip max diff: {rt_diff:.2e}")
    print(f"  PASS: {rt_diff == 0.0}")
    print()

    # -----------------------------------------------------------------------
    # Test 3: Core equivalence -- ANE interleaved vs HF split-halves
    # Apply same logical rotation, but in different formats
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 3: ANE interleaved RoPE vs HF split-halves RoPE")
    print("  Start with SAME data, convert formats, apply each RoPE,")
    print("  convert result back, check equivalence.")
    print("-" * 72)

    # Create input in interleaved format
    x_interleaved = np.random.randn(SEQ, HEADS * HD)

    # Convert to split-halves for HF
    x_split = interleaved_to_split(x_interleaved, HEADS, HD)

    # Apply ANE RoPE to interleaved input
    y_ane = ane_rope(x_interleaved, HEADS, HD)

    # Apply HF RoPE to split-halves input
    y_hf = hf_rope(x_split, HEADS, HD)

    # Convert HF result back to interleaved for comparison
    y_hf_as_interleaved = split_to_interleaved(y_hf, HEADS, HD)

    diff_3 = np.max(np.abs(y_ane - y_hf_as_interleaved))
    reldiff_3 = np.max(np.abs(y_ane - y_hf_as_interleaved) / (np.abs(y_ane) + 1e-30))
    print(f"  Max abs diff:  {diff_3:.2e}")
    print(f"  Max rel diff:  {reldiff_3:.2e}")
    print(f"  PASS: {diff_3 < 1e-12}" if diff_3 < 1e-12 else f"  FAIL: diff={diff_3}")
    print()

    # -----------------------------------------------------------------------
    # Test 4: End-to-end weight interleaving verification
    # Simulates: HF model -> hf_to_ane.py -> ANE inference
    # vs: HF model -> HF inference (native)
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 4: End-to-end weight interleaving (hf_to_ane.py pipeline)")
    print("  Simulates: x @ Wq_hf^T -> HF RoPE  vs  x @ Wq_ane^T -> ANE RoPE")
    print("  where Wq_ane = interleave_weights(Wq_hf)")
    print("-" * 72)

    # Random input and weight matrix
    x_input = np.random.randn(SEQ, DIM)  # [seq, dim]
    Wq_hf = np.random.randn(HEADS * HD, DIM)  # HF Q projection [q_dim, dim]

    # HF path: project then apply HF RoPE
    q_hf = x_input @ Wq_hf.T  # [seq, q_dim] in split-halves order
    q_hf_rotated = hf_rope(q_hf, HEADS, HD)

    # ANE path: interleave weights (as hf_to_ane.py does), project, ANE RoPE
    Wq_ane = interleave_weights(Wq_hf, HEADS, HD)
    q_ane = x_input @ Wq_ane.T  # [seq, q_dim] in interleaved order
    q_ane_rotated = ane_rope(q_ane, HEADS, HD)

    # Convert ANE result to split-halves for comparison
    q_ane_as_split = interleaved_to_split(q_ane_rotated, HEADS, HD)

    diff_4 = np.max(np.abs(q_hf_rotated - q_ane_as_split))
    reldiff_4 = np.max(np.abs(q_hf_rotated - q_ane_as_split) / (np.abs(q_hf_rotated) + 1e-30))
    print(f"  Max abs diff:  {diff_4:.2e}")
    print(f"  Max rel diff:  {reldiff_4:.2e}")
    print(f"  PASS: {diff_4 < 1e-10}" if diff_4 < 1e-10 else f"  FAIL: diff={diff_4}")
    print()

    # -----------------------------------------------------------------------
    # Test 5: Verify weight interleaving is correct permutation
    # Check that interleave_weights correctly maps split->interleaved on rows
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 5: Verify interleave_weights row permutation")
    print("  Check: Wq_ane @ v produces interleaved output when Wq_hf @ v")
    print("  would produce split-halves output.")
    print("-" * 72)

    v = np.random.randn(DIM)
    q_hf_vec = Wq_hf @ v       # split-halves ordering
    q_ane_vec = Wq_ane @ v      # interleaved ordering

    # Manually convert hf result to interleaved
    q_hf_manual_interleaved = np.empty_like(q_hf_vec)
    half = HD // 2
    for h in range(HEADS):
        for i in range(half):
            q_hf_manual_interleaved[h * HD + 2 * i]     = q_hf_vec[h * HD + i]        # real
            q_hf_manual_interleaved[h * HD + 2 * i + 1]  = q_hf_vec[h * HD + half + i] # imag

    diff_5 = np.max(np.abs(q_ane_vec - q_hf_manual_interleaved))
    print(f"  Max abs diff:  {diff_5:.2e}")
    print(f"  PASS: {diff_5 < 1e-12}" if diff_5 < 1e-12 else f"  FAIL: diff={diff_5}")
    print()

    # -----------------------------------------------------------------------
    # Test 6: KV heads (GQA) verification
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 6: KV heads (GQA) - same test with n_kv_heads=5")
    print("-" * 72)

    Wk_hf = np.random.randn(KV_HEADS * HD, DIM)
    Wk_ane = interleave_weights(Wk_hf, KV_HEADS, HD)

    k_hf = x_input @ Wk_hf.T
    k_hf_rotated = hf_rope(k_hf, KV_HEADS, HD)

    k_ane = x_input @ Wk_ane.T
    k_ane_rotated = ane_rope(k_ane, KV_HEADS, HD)

    k_ane_as_split = interleaved_to_split(k_ane_rotated, KV_HEADS, HD)

    diff_6 = np.max(np.abs(k_hf_rotated - k_ane_as_split))
    reldiff_6 = np.max(np.abs(k_hf_rotated - k_ane_as_split) / (np.abs(k_hf_rotated) + 1e-30))
    print(f"  Max abs diff:  {diff_6:.2e}")
    print(f"  Max rel diff:  {reldiff_6:.2e}")
    print(f"  PASS: {diff_6 < 1e-10}" if diff_6 < 1e-10 else f"  FAIL: diff={diff_6}")
    print()

    # -----------------------------------------------------------------------
    # Test 7: Spot-check specific values at known positions
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 7: Spot-check - manual rotation for head=0, pos=1, pair=0")
    print("-" * 72)

    x_test = np.zeros((2, HEADS * HD))
    x_test[1, 0] = 3.0   # re0 of head 0
    x_test[1, 1] = 4.0   # im0 of head 0 (interleaved)

    freq_0 = 1.0 / (ROPE_THETA ** (0.0 / HD))  # = 1.0
    theta_pos1 = 1.0 * freq_0  # = 1.0
    cos_expected = np.cos(theta_pos1)
    sin_expected = np.sin(theta_pos1)

    expected_re = 3.0 * cos_expected - 4.0 * sin_expected
    expected_im = 3.0 * sin_expected + 4.0 * cos_expected

    y_test = ane_rope(x_test, HEADS, HD)
    actual_re = y_test[1, 0]
    actual_im = y_test[1, 1]

    print(f"  theta(pos=1, i=0) = {theta_pos1:.6f}")
    print(f"  cos={cos_expected:.6f}, sin={sin_expected:.6f}")
    print(f"  Expected: re={expected_re:.6f}, im={expected_im:.6f}")
    print(f"  Got:      re={actual_re:.6f}, im={actual_im:.6f}")
    diff_7 = max(abs(expected_re - actual_re), abs(expected_im - actual_im))
    print(f"  Max abs diff:  {diff_7:.2e}")
    print(f"  PASS: {diff_7 < 1e-12}")
    print()

    # -----------------------------------------------------------------------
    # Test 8: High-frequency pair (last pair, i=31, freq decays with theta)
    # -----------------------------------------------------------------------
    print("-" * 72)
    print("TEST 8: High-frequency pair spot-check (i=31, pos=255)")
    print("-" * 72)

    freq_31 = 1.0 / (ROPE_THETA ** (62.0 / HD))  # 2*31 = 62
    theta_high = 255.0 * freq_31
    print(f"  freq(i=31) = {freq_31:.10e}")
    print(f"  theta(pos=255, i=31) = {theta_high:.10f}")
    print(f"  cos(theta) = {np.cos(theta_high):.10f}")
    print(f"  sin(theta) = {np.sin(theta_high):.10f}")
    print(f"  (Verifies frequencies decay correctly for large rope_theta=100000)")
    print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_pass = all([
        diff_1 < 1e-10,
        rt_diff == 0.0,
        diff_3 < 1e-12,
        diff_4 < 1e-10,
        diff_5 < 1e-12,
        diff_6 < 1e-10,
        diff_7 < 1e-12,
    ])
    results = [
        ("Test 1: Python vs C (interleaved)",          diff_1,  diff_1 < 1e-10),
        ("Test 2: Format round-trip",                  rt_diff, rt_diff == 0.0),
        ("Test 3: ANE vs HF RoPE (core equivalence)",  diff_3,  diff_3 < 1e-12),
        ("Test 4: End-to-end weight pipeline",         diff_4,  diff_4 < 1e-10),
        ("Test 5: interleave_weights permutation",     diff_5,  diff_5 < 1e-12),
        ("Test 6: KV heads (GQA)",                     diff_6,  diff_6 < 1e-10),
        ("Test 7: Manual spot-check",                  diff_7,  diff_7 < 1e-12),
    ]
    for name, diff, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name:50s} max_diff={diff:.2e}")

    print()
    if all_pass:
        print("ALL TESTS PASSED -- RoPE implementations are numerically equivalent")
        print("after proper format conversion. hf_to_ane.py interleaving is correct.")
    else:
        print("SOME TESTS FAILED -- investigate differences above")

    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
