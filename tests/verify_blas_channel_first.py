#!/usr/bin/env python3
"""Numerical verification: BLAS call ordering and channel-first layout.

For LAYER 0 only, verifies:
  a) Embedding lookup: channel-first x_cf[d, t] = embed[tok[t], d]
  b) Wq matmul: Q_cf[q_dim, seq] = Wq[q_dim, dim] @ x_cf[dim, seq]
  c) RMSNorm: channel-first vs row-major give identical normalized output

Uses ACTUAL weights from the SmolLM2-360M checkpoint AND the HuggingFace model.
"""

import struct, os, sys, math
import numpy as np

# ---------------------------------------------------------------------------
# Config (SmolLM2-360M)
# ---------------------------------------------------------------------------
DIM      = 960
Q_DIM    = 960
KV_DIM   = 320
HIDDEN   = 2560
HEADS    = 15
KV_HEADS = 5
HD       = 64
NLAYERS  = 32
VOCAB    = 49152
SEQ      = 256

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(REPO, "ane_smollm2_360m_ckpt.bin")
DATA_PATH = os.path.join(REPO, "tinystories_smollm2_data00.bin")

# ---------------------------------------------------------------------------
# Load checkpoint layer-0 weights (binary format)
# ---------------------------------------------------------------------------
def load_layer0_from_checkpoint(path):
    """Read the 96-byte header, then layer-0 weights only."""
    with open(path, "rb") as f:
        hdr = struct.unpack("<iiiiiiiiiiffdddiiiiii", f.read(96))
        magic = hdr[0]
        assert magic == 0x424C5A54, f"Bad magic: {hex(magic)}"

        # Layer 0 weights immediately after header
        wq_sz  = Q_DIM  * DIM
        wk_sz  = KV_DIM * DIM
        wv_sz  = KV_DIM * DIM
        wo_sz  = DIM    * Q_DIM
        w1_sz  = HIDDEN * DIM
        w2_sz  = DIM    * HIDDEN
        w3_sz  = HIDDEN * DIM

        Wq = np.frombuffer(f.read(wq_sz * 4), np.float32).reshape(Q_DIM, DIM).copy()
        Wk = np.frombuffer(f.read(wk_sz * 4), np.float32).reshape(KV_DIM, DIM).copy()
        Wv = np.frombuffer(f.read(wv_sz * 4), np.float32).reshape(KV_DIM, DIM).copy()
        Wo = np.frombuffer(f.read(wo_sz * 4), np.float32).reshape(DIM, Q_DIM).copy()
        W1 = np.frombuffer(f.read(w1_sz * 4), np.float32).reshape(HIDDEN, DIM).copy()
        W2 = np.frombuffer(f.read(w2_sz * 4), np.float32).reshape(DIM, HIDDEN).copy()
        W3 = np.frombuffer(f.read(w3_sz * 4), np.float32).reshape(HIDDEN, DIM).copy()
        rms_att = np.frombuffer(f.read(DIM * 4), np.float32).copy()
        rms_ffn = np.frombuffer(f.read(DIM * 4), np.float32).copy()

        # Skip to end-of-layers to read rms_final and embed
        # Per-layer total params
        layer_params = wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2 * DIM

        # We already read layer 0 weights; skip its Adam state + remaining 31 layers
        f.seek(layer_params * 2 * 4, 1)  # layer-0 Adam m+v
        for _ in range(NLAYERS - 1):
            f.seek(layer_params * 4, 1)      # weights
            f.seek(layer_params * 2 * 4, 1)  # Adam m+v

        rms_final = np.frombuffer(f.read(DIM * 4), np.float32).copy()
        f.seek(DIM * 2 * 4, 1)  # skip Adam state for rms_final

        embed = np.frombuffer(f.read(VOCAB * DIM * 4), np.float32).reshape(VOCAB, DIM).copy()

    return {
        "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
        "W1": W1, "W2": W2, "W3": W3,
        "rms_att": rms_att, "rms_ffn": rms_ffn,
        "rms_final": rms_final, "embed": embed,
    }


# ---------------------------------------------------------------------------
# De-interleave helper (reverse of interleave_weights in hf_to_ane.py)
# ---------------------------------------------------------------------------
def deinterleave_weights(W, n_heads, head_dim):
    """Convert ANE interleaved RoPE ordering back to HF non-interleaved.
    ANE:  [re0, im0, re1, im1, ..., re31, im31] per head
    HF:   [re0, re1, ..., re31, im0, im1, ..., im31] per head
    W shape: [n_heads * head_dim, dim]
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  BLAS Call Ordering & Channel-First Layout Verification")
    print("  SmolLM2-360M  |  Layer 0  |  SEQ=256")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data tokens
    # ------------------------------------------------------------------
    print("\n[1] Loading data tokens...")
    token_data = np.fromfile(DATA_PATH, dtype=np.uint16)
    tokens = token_data[:SEQ].astype(np.int64)
    print(f"    First 8 tokens: {tokens[:8].tolist()}")
    assert len(tokens) == SEQ

    # ------------------------------------------------------------------
    # 2. Load checkpoint layer 0
    # ------------------------------------------------------------------
    print("\n[2] Loading checkpoint layer 0...")
    ckpt = load_layer0_from_checkpoint(CKPT_PATH)
    print(f"    Wq shape: {ckpt['Wq'].shape}")
    print(f"    embed shape: {ckpt['embed'].shape}")

    # ------------------------------------------------------------------
    # 3. Load HuggingFace model
    # ------------------------------------------------------------------
    print("\n[3] Loading HuggingFace SmolLM2-360M...")
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M", torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_embed_weight = hf_model.model.embed_tokens.weight.detach().numpy()  # [VOCAB, DIM]
    hf_Wq = hf_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()  # [Q_DIM, DIM]
    hf_rms_att_w = hf_model.model.layers[0].input_layernorm.weight.detach().numpy()  # [DIM]

    print(f"    HF embed shape:   {hf_embed_weight.shape}")
    print(f"    HF Wq shape:      {hf_Wq.shape}")
    print(f"    HF rms_att shape: {hf_rms_att_w.shape}")

    # ------------------------------------------------------------------
    # Verify checkpoint embed matches HF embed (sanity check)
    # ------------------------------------------------------------------
    print("\n[Sanity] Checkpoint embed vs HF embed...")
    embed_diff = np.max(np.abs(ckpt["embed"] - hf_embed_weight))
    print(f"    Max |ckpt_embed - hf_embed| = {embed_diff:.2e}")
    assert embed_diff < 1e-5, "Embeddings don't match!"

    # ------------------------------------------------------------------
    # Verify Wq: checkpoint stores interleaved; HF stores non-interleaved
    # ------------------------------------------------------------------
    print("\n[Sanity] Checkpoint Wq vs HF Wq (after de-interleaving)...")
    ckpt_Wq_deinterleaved = deinterleave_weights(ckpt["Wq"], HEADS, HD)
    wq_diff = np.max(np.abs(ckpt_Wq_deinterleaved - hf_Wq))
    print(f"    Max |deinterleave(ckpt_Wq) - hf_Wq| = {wq_diff:.2e}")
    assert wq_diff < 1e-5, "Wq weights don't match after de-interleaving!"

    # ==================================================================
    # (a) EMBEDDING LOOKUP: channel-first convention
    # ==================================================================
    print("\n" + "=" * 70)
    print("  (a) Embedding Lookup: Channel-First Convention")
    print("=" * 70)

    # Row-major (HF convention): x_rm[t, d] = embed[tok[t], d]
    # Shape: [SEQ, DIM]
    x_rm = hf_embed_weight[tokens]   # [SEQ, DIM]

    # HF model's actual embedding output
    with torch.no_grad():
        hf_input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, SEQ]
        hf_embed_out = hf_model.model.embed_tokens(hf_input_ids)  # [1, SEQ, DIM]
    hf_embed_np = hf_embed_out.squeeze(0).numpy()  # [SEQ, DIM]

    # Channel-first: x_cf[d, t] = embed[tok[t], d]
    # Shape: [DIM, SEQ]
    x_cf = np.zeros((DIM, SEQ), dtype=np.float32)
    for t in range(SEQ):
        for d in range(DIM):
            x_cf[d, t] = ckpt["embed"][tokens[t], d]

    # Verify: x_cf should be the transpose of x_rm
    diff_embed_transpose = np.max(np.abs(x_cf - x_rm.T))
    print(f"\n    x_cf[d,t] = embed[tok[t], d]")
    print(f"    x_rm[t,d] = embed[tok[t], d]  (HF convention)")
    print(f"    Max |x_cf - x_rm^T| = {diff_embed_transpose:.2e}")

    # Verify our row-major matches HF's actual embed output
    diff_embed_vs_hf = np.max(np.abs(x_rm - hf_embed_np))
    print(f"    Max |x_rm - hf_embed_tokens_output| = {diff_embed_vs_hf:.2e}")

    # Verify channel-first transpose matches HF output
    diff_cf_vs_hf = np.max(np.abs(x_cf.T - hf_embed_np))
    print(f"    Max |x_cf^T - hf_embed_tokens_output| = {diff_cf_vs_hf:.2e}")

    status_a = "PASS" if diff_cf_vs_hf < 1e-6 else "FAIL"
    print(f"\n    [{status_a}] Embedding lookup: channel-first layout is correct")

    # ==================================================================
    # (b) Wq MATMUL: channel-first vs row-major
    # ==================================================================
    print("\n" + "=" * 70)
    print("  (b) Wq Matmul: Channel-First vs Row-Major")
    print("=" * 70)

    # First apply RMSNorm (layer 0 input_layernorm) to get the actual input to Wq
    # Row-major: xn_rm = rmsnorm(x_rm, rms_att)  [SEQ, DIM]
    def rmsnorm_rm(x, w):
        """Row-major: x [SEQ, DIM], w [DIM] -> [SEQ, DIM]"""
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
        return (x / rms) * w

    xn_rm = rmsnorm_rm(x_rm, ckpt["rms_att"])  # [SEQ, DIM]

    # Channel-first RMSNorm input: x_cf [DIM, SEQ]
    def rmsnorm_cf(x, w):
        """Channel-first: x [DIM, SEQ], w [DIM] -> [DIM, SEQ]"""
        rms = np.sqrt(np.mean(x * x, axis=0, keepdims=True) + 1e-5)  # [1, SEQ]
        return (x / rms) * w[:, None]  # broadcast w[DIM] -> [DIM, 1]
    xn_cf = rmsnorm_cf(x_cf, ckpt["rms_att"])  # [DIM, SEQ]

    # -- Row-major Wq matmul --
    # HF convention: Q_rm = xn_rm @ Wq^T, shape [SEQ, Q_DIM]
    # Note: checkpoint Wq is interleaved; HF Wq is non-interleaved
    # For comparing the matmul itself, we use the CHECKPOINT Wq (interleaved)
    # since that's what the ANE binary uses
    Q_rm = xn_rm @ ckpt["Wq"].T  # [SEQ, Q_DIM]

    # -- Channel-first Wq matmul --
    # Q_cf[q_dim, seq] = Wq[q_dim, dim] @ xn_cf[dim, seq]
    Q_cf = ckpt["Wq"] @ xn_cf  # [Q_DIM, SEQ]

    # Verify: Q_cf should be the transpose of Q_rm
    diff_q_transpose = np.max(np.abs(Q_cf - Q_rm.T))
    print(f"\n    Row-major:     Q_rm[seq, q_dim] = xn @ Wq^T         shape {Q_rm.shape}")
    print(f"    Channel-first: Q_cf[q_dim, seq] = Wq @ xn_cf        shape {Q_cf.shape}")
    print(f"    Max |Q_cf - Q_rm^T| = {diff_q_transpose:.2e}")

    # Also verify against HF's actual Q computation (need to use HF's non-interleaved Wq)
    Q_hf = xn_rm @ hf_Wq.T  # [SEQ, Q_DIM] using HF weights (non-interleaved)
    Q_ckpt_deint = xn_rm @ ckpt_Wq_deinterleaved.T  # should match Q_hf
    diff_q_vs_hf = np.max(np.abs(Q_ckpt_deint - Q_hf))
    print(f"    Max |Q_ckpt_deinterleaved - Q_hf| = {diff_q_vs_hf:.2e}")

    status_b = "PASS" if diff_q_transpose < 1e-3 else "FAIL"
    print(f"\n    [{status_b}] Wq matmul: channel-first BLAS ordering is correct")
    print(f"          (tolerance allows float32 accumulation differences)")

    # ==================================================================
    # (c) RMSNorm: channel-first vs row-major
    # ==================================================================
    print("\n" + "=" * 70)
    print("  (c) RMSNorm: Channel-First vs Row-Major")
    print("=" * 70)

    # Row-major: mean(x^2) over axis=-1 (DIM axis), per sequence position
    rms_sq_rm = np.mean(x_rm * x_rm, axis=-1)  # [SEQ]

    # Channel-first: mean(x^2) over axis=0 (DIM axis), per sequence position
    rms_sq_cf = np.mean(x_cf * x_cf, axis=0)   # [SEQ]

    diff_rms_sq = np.max(np.abs(rms_sq_rm - rms_sq_cf))
    print(f"\n    Row-major:     mean(x^2) over axis=-1 (dim)  shape {rms_sq_rm.shape}")
    print(f"    Channel-first: mean(x^2) over axis=0  (dim)  shape {rms_sq_cf.shape}")
    print(f"    Max |rms_sq_rm - rms_sq_cf| = {diff_rms_sq:.2e}")

    # Verify normalized output
    diff_norm = np.max(np.abs(xn_cf - xn_rm.T))
    print(f"\n    Normalized output:")
    print(f"    Max |xn_cf[d,t] - xn_rm[t,d]| = {diff_norm:.2e}")

    # Compare against HF's actual RMSNorm output
    with torch.no_grad():
        hf_rms_out = hf_model.model.layers[0].input_layernorm(hf_embed_out)
    hf_rms_np = hf_rms_out.squeeze(0).numpy()  # [SEQ, DIM]

    # Our row-major RMSNorm uses checkpoint rms_att weights which should match HF
    rms_w_diff = np.max(np.abs(ckpt["rms_att"] - hf_rms_att_w))
    print(f"\n    RMSNorm weight diff (ckpt vs HF): {rms_w_diff:.2e}")

    diff_norm_vs_hf = np.max(np.abs(xn_rm - hf_rms_np))
    print(f"    Max |our_rmsnorm - hf_rmsnorm| = {diff_norm_vs_hf:.2e}")

    diff_cf_norm_vs_hf = np.max(np.abs(xn_cf.T - hf_rms_np))
    print(f"    Max |cf_rmsnorm^T - hf_rmsnorm| = {diff_cf_norm_vs_hf:.2e}")

    status_c = "PASS" if diff_norm < 1e-5 else "FAIL"
    print(f"\n    [{status_c}] RMSNorm: channel-first layout is correct")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n    (a) Embedding lookup (channel-first):  {status_a}  "
          f"(max diff = {diff_cf_vs_hf:.2e})")
    print(f"    (b) Wq matmul (BLAS ordering):         {status_b}  "
          f"(max diff = {diff_q_transpose:.2e})")
    print(f"    (c) RMSNorm (axis convention):          {status_c}  "
          f"(max diff = {diff_norm:.2e})")

    all_pass = all(s == "PASS" for s in [status_a, status_b, status_c])
    print(f"\n    Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print()

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
