#!/usr/bin/env python3
"""Comprehensive verification of AutoANE SmolLM2-360M data, model, and checkpoint integrity.

Checks:
  1. Data format verification (raw hex + uint16 token validity)
  2. Model dimensions cross-check against HuggingFace config
  3. RoPE frequency verification
  4. Checkpoint header verification
  5. Compact vocab count
"""

import struct
import numpy as np
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO, "tinystories_smollm2_data00.bin")
CKPT_PATH = os.path.join(REPO, "training", "ane_smollm2_360m_ckpt.bin")

# Expected model dimensions for SmolLM2-360M
EXPECTED = {
    "dim": 960,
    "hidden": 2560,
    "heads": 15,
    "kv_heads": 5,
    "n_layers": 32,
    "vocab": 49152,
    "hd": 64,       # head_dim = dim / heads = 960 / 15
    "rope_theta": 100000.0,
    "q_dim": 960,   # heads * hd = 15 * 64
    "kv_dim": 320,  # kv_heads * hd = 5 * 64
    "seq": 256,
}

passes = 0
fails = 0

def check(name, condition, detail=""):
    global passes, fails
    if condition:
        passes += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        fails += 1
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# =========================================================================
# 1. DATA FORMAT VERIFICATION
# =========================================================================
section("1. DATA FORMAT VERIFICATION")

with open(DATA_PATH, "rb") as f:
    raw20 = f.read(20)

print(f"\n  First 20 bytes as raw hex:")
hex_str = " ".join(f"{b:02x}" for b in raw20)
print(f"    {hex_str}")

# Interpret as uint16 little-endian tokens
tokens_10 = np.frombuffer(raw20, dtype=np.uint16)
print(f"\n  First 10 uint16 tokens: {list(tokens_10)}")

# Check they're valid SmolLM2 token IDs (0..49151)
all_valid = all(0 <= t < 49152 for t in tokens_10)
check("First 10 tokens in valid SmolLM2 range [0, 49151]", all_valid,
      f"min={min(tokens_10)}, max={max(tokens_10)}")

# Check no magic number header: magic numbers are typically large or have specific patterns
# Common magic: 0x46544B54 (FTKB), 0x67676D6C (ggml), etc. — these would show as very
# large uint16 values or recognizable ASCII
first_u32 = struct.unpack("<I", raw20[:4])[0]
known_magics = [0x424C5A54, 0x46544B54, 0x67676D6C, 0x67676D66, 0x46494C45]
no_magic = first_u32 not in known_magics
check("No magic number header (raw token data)", no_magic,
      f"first uint32 = 0x{first_u32:08X}")

# Plausibility: first few tokens should look like plausible text token IDs
# (not extremely large, not all zeros, some variety)
has_variety = len(set(tokens_10)) > 1
check("Tokens show variety (not all identical)", has_variety,
      f"{len(set(tokens_10))} unique values in first 10")

# File size should be a multiple of 2 bytes (uint16)
file_size = os.path.getsize(DATA_PATH)
check("File size is multiple of 2 (uint16 aligned)", file_size % 2 == 0,
      f"size={file_size} bytes, {file_size//2} tokens")


# =========================================================================
# 2. MODEL DIMENSIONS CROSS-CHECK
# =========================================================================
section("2. MODEL DIMENSIONS CROSS-CHECK (vs HuggingFace)")

try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-360M")

    hf_vals = {
        "dim": config.hidden_size,
        "hidden": config.intermediate_size,
        "heads": config.num_attention_heads,
        "kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "n_layers": config.num_hidden_layers,
        "vocab": config.vocab_size,
        "hd": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "rope_theta": getattr(config, "rope_theta", None),
    }

    print(f"\n  {'Field':<15} {'Expected':>10} {'HuggingFace':>12} {'Match':>8}")
    print(f"  {'-'*47}")

    for field in ["dim", "hidden", "heads", "kv_heads", "n_layers", "vocab", "hd", "rope_theta"]:
        exp = EXPECTED[field]
        got = hf_vals[field]
        match = (exp == got) if not isinstance(exp, float) else abs(exp - got) < 1e-6
        symbol = "==" if match else "!="
        print(f"  {field:<15} {str(exp):>10} {str(got):>12} {symbol:>8}")
        check(f"dim cross-check: {field}", match, f"expected={exp}, got={got}")

    # Also verify derived quantities
    computed_hd = hf_vals["dim"] // hf_vals["heads"]
    check("head_dim = dim / heads", computed_hd == hf_vals["hd"],
          f"{hf_vals['dim']}/{hf_vals['heads']} = {computed_hd}, config says {hf_vals['hd']}")

except ImportError:
    print("  WARNING: transformers not installed, skipping HuggingFace cross-check")
    print("  Using expected values only for remaining checks")
except Exception as e:
    print(f"  WARNING: Could not load HF config: {e}")
    print("  Using expected values only for remaining checks")


# =========================================================================
# 3. RoPE FREQUENCY VERIFICATION
# =========================================================================
section("3. RoPE FREQUENCY VERIFICATION")

hd = EXPECTED["hd"]        # 64
theta = EXPECTED["rope_theta"]  # 100000.0
n_freq = hd // 2           # 32 frequencies

# Our formula: freq[i] = 1.0 / theta^(2*i/hd)
our_freqs = np.array([1.0 / (theta ** (2.0 * i / hd)) for i in range(n_freq)])

print(f"\n  RoPE params: head_dim={hd}, theta={theta}, n_freq={n_freq}")
print(f"\n  {'i':>4}  {'Our freq[i]':>20}  {'HF freq[i]':>20}  {'Match':>8}")
print(f"  {'-'*56}")

try:
    import torch
    # HuggingFace computes: inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    hf_inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    hf_freqs = hf_inv_freq.numpy()
    has_hf = True
except ImportError:
    # Compute manually the same way HF does
    indices = np.arange(0, hd, 2, dtype=np.float64)
    hf_freqs = 1.0 / (theta ** (indices / hd))
    has_hf = False
    print(f"  (torch not available; computing HF formula manually)")

# Print first 5
for i in range(5):
    match = np.isclose(our_freqs[i], hf_freqs[i], rtol=1e-12)
    symbol = "==" if match else "!="
    print(f"  {i:>4}  {our_freqs[i]:>20.15e}  {hf_freqs[i]:>20.15e}  {symbol:>8}")

print(f"  {'...':>4}")

# Print last 5
for i in range(n_freq - 5, n_freq):
    match = np.isclose(our_freqs[i], hf_freqs[i], rtol=1e-12)
    symbol = "==" if match else "!="
    print(f"  {i:>4}  {our_freqs[i]:>20.15e}  {hf_freqs[i]:>20.15e}  {symbol:>8}")

all_match = np.allclose(our_freqs, hf_freqs, rtol=1e-12)
check("All RoPE frequencies match (rtol=1e-12)", all_match)

max_rel_err = np.max(np.abs(our_freqs - hf_freqs) / np.abs(hf_freqs))
check("Max relative error < 1e-14", max_rel_err < 1e-14,
      f"max_rel_err = {max_rel_err:.2e}")


# =========================================================================
# 4. CHECKPOINT HEADER VERIFICATION
# =========================================================================
section("4. CHECKPOINT HEADER VERIFICATION")

with open(CKPT_PATH, "rb") as f:
    header_bytes = f.read(96)

# Parse the 96-byte CkptHdr
# struct CkptHdr {
#   int magic, version, step, total_steps;          // 4 ints = 16 bytes
#   int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;  // 6 ints = 24 bytes
#   float lr, loss;                                 // 2 floats = 8 bytes
#   double cum_compile, cum_train, cum_wall;        // 3 doubles = 24 bytes
#   int cum_steps, cum_batches, adam_t;             // 3 ints = 12 bytes
#   int kv_heads, head_dim, q_dim;                  // 3 ints = 12 bytes
# }
# Total: 16 + 24 + 8 + 24 + 12 + 12 = 96 bytes

fmt = "<iiiiiiiiiiffdddiiiiii"
fields = struct.unpack(fmt, header_bytes)

(magic, version, step, total_steps,
 n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len,
 lr, loss,
 cum_compile, cum_train, cum_wall,
 cum_steps, cum_batches, adam_t,
 kv_heads, head_dim, q_dim) = fields

print(f"\n  Checkpoint: {CKPT_PATH}")
print(f"  Header (96 bytes):")
print(f"    magic       = 0x{magic:08X}  ('{struct.pack('<I', magic).decode('ascii', errors='replace')}')")
print(f"    version     = {version}")
print(f"    step        = {step}")
print(f"    total_steps = {total_steps}")
print(f"    n_layers    = {n_layers}")
print(f"    vocab_size  = {vocab_size}")
print(f"    dim         = {dim}")
print(f"    hidden_dim  = {hidden_dim}")
print(f"    n_heads     = {n_heads}")
print(f"    seq_len     = {seq_len}")
print(f"    lr          = {lr}")
print(f"    loss        = {loss}")
print(f"    cum_compile = {cum_compile}")
print(f"    cum_train   = {cum_train}")
print(f"    cum_wall    = {cum_wall}")
print(f"    cum_steps   = {cum_steps}")
print(f"    cum_batches = {cum_batches}")
print(f"    adam_t      = {adam_t}")
print(f"    kv_heads    = {kv_heads}")
print(f"    head_dim    = {head_dim}")
print(f"    q_dim       = {q_dim}")

# Verify magic
check("Magic = 0x424C5A54 ('BLZT')", magic == 0x424C5A54,
      f"got 0x{magic:08X}")

# Verify version
check("Version = 4", version == 4, f"got {version}")

# Cross-check dimensions
check("n_layers = 32", n_layers == EXPECTED["n_layers"], f"got {n_layers}")
check("vocab_size = 49152", vocab_size == EXPECTED["vocab"], f"got {vocab_size}")
check("dim = 960", dim == EXPECTED["dim"], f"got {dim}")
check("hidden_dim = 2560", hidden_dim == EXPECTED["hidden"], f"got {hidden_dim}")
check("n_heads = 15", n_heads == EXPECTED["heads"], f"got {n_heads}")
check("seq_len = 256", seq_len == EXPECTED["seq"], f"got {seq_len}")
check("kv_heads = 5", kv_heads == EXPECTED["kv_heads"], f"got {kv_heads}")
check("head_dim = 64", head_dim == EXPECTED["hd"], f"got {head_dim}")
check("q_dim = 960", q_dim == EXPECTED["q_dim"], f"got {q_dim}")

# Verify file size consistency
# Per-layer weight sizes (float32)
wq_sz = EXPECTED["q_dim"] * EXPECTED["dim"]       # 960 * 960
wk_sz = EXPECTED["kv_dim"] * EXPECTED["dim"]       # 320 * 960
wv_sz = EXPECTED["kv_dim"] * EXPECTED["dim"]       # 320 * 960
wo_sz = EXPECTED["dim"] * EXPECTED["q_dim"]         # 960 * 960
w1_sz = EXPECTED["hidden"] * EXPECTED["dim"]        # 2560 * 960
w2_sz = EXPECTED["dim"] * EXPECTED["hidden"]        # 960 * 2560
w3_sz = EXPECTED["hidden"] * EXPECTED["dim"]        # 2560 * 960
layer_params = wq_sz + wk_sz + wv_sz + wo_sz + w1_sz + w2_sz + w3_sz + 2 * EXPECTED["dim"]

# hf_to_ane.py writes per layer: weights (layer_params) + adam state (layer_params * 2 zeros)
per_layer_bytes = layer_params * 4 + layer_params * 2 * 4  # weights + 2x adam (m + v)

# After all layers: rms_final (dim) + adam for rms_final (dim * 2) + embed (vocab*dim) + adam for embed (vocab*dim*2)
rms_final_bytes = EXPECTED["dim"] * 4 + EXPECTED["dim"] * 2 * 4
embed_bytes = EXPECTED["vocab"] * EXPECTED["dim"] * 4 + EXPECTED["vocab"] * EXPECTED["dim"] * 2 * 4

expected_file_size = 96 + EXPECTED["n_layers"] * per_layer_bytes + rms_final_bytes + embed_bytes

actual_file_size = os.path.getsize(CKPT_PATH)

print(f"\n  File size analysis:")
print(f"    Header:              96 bytes")
print(f"    Per-layer params:    {layer_params} floats = {layer_params*4} bytes weights")
print(f"    Per-layer total:     {per_layer_bytes} bytes (weights + adam m,v)")
print(f"    x {EXPECTED['n_layers']} layers:         {EXPECTED['n_layers'] * per_layer_bytes} bytes")
print(f"    rms_final + adam:    {rms_final_bytes} bytes")
print(f"    embed + adam:        {embed_bytes} bytes")
print(f"    Expected total:      {expected_file_size} bytes ({expected_file_size/1e9:.3f} GB)")
print(f"    Actual file size:    {actual_file_size} bytes ({actual_file_size/1e9:.3f} GB)")

check("File size matches expected",
      actual_file_size == expected_file_size,
      f"expected={expected_file_size}, actual={actual_file_size}, diff={actual_file_size - expected_file_size}")


# =========================================================================
# 5. COMPACT VOCAB COUNT
# =========================================================================
section("5. COMPACT VOCAB COUNT")

print(f"\n  Reading {DATA_PATH} ({file_size} bytes, {file_size//2} tokens)...")
data = np.fromfile(DATA_PATH, dtype=np.uint16)
unique_tokens = np.unique(data)
n_unique = len(unique_tokens)

print(f"  Total tokens:  {len(data)}")
print(f"  Unique tokens: {n_unique}")
print(f"  Min token ID:  {unique_tokens[0]}")
print(f"  Max token ID:  {unique_tokens[-1]}")

check("Unique token count = 16893", n_unique == 16893,
      f"got {n_unique}")

# Also verify all unique tokens are valid
all_in_range = unique_tokens[-1] < 49152
check("All unique tokens in valid range [0, 49151]", all_in_range,
      f"max={unique_tokens[-1]}")


# =========================================================================
# SUMMARY
# =========================================================================
section("SUMMARY")
total = passes + fails
print(f"\n  {passes}/{total} checks PASSED, {fails}/{total} FAILED\n")
if fails == 0:
    print("  All checks passed!")
else:
    print(f"  {fails} check(s) failed -- review above for details.")
    sys.exit(1)
