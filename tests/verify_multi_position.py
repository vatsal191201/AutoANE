#!/usr/bin/env python3
"""Verify C training code matches HuggingFace at MULTIPLE data positions.

Loads SmolLM2-360M from HuggingFace and computes CE loss with compact vocab
+ log-sum-exp at 5 different random data positions (replicating the C code's
drand48-based data sampling). Compares against the C binary's reported loss
for seed=42.
"""

import ctypes
import math
import numpy as np
import os
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. Replicate C's drand48-based data sampling
# ============================================================
libc = ctypes.CDLL('libSystem.B.dylib')
libc.srand48.argtypes = [ctypes.c_long]
libc.drand48.restype = ctypes.c_double


def get_data_position(seed, n_tokens, step=0):
    """Replicate C code: srand48(init_seed + step * 7919); pos = drand48() * max_pos"""
    libc.srand48(seed + step * 7919)
    r = libc.drand48()
    n_train = int(n_tokens * 0.9)
    max_pos = n_train - 256 - 1
    return int(r * max_pos)


# ============================================================
# 2. Build compact vocab map (matching C's vocab_map_build)
# ============================================================
def build_vocab_map(token_data, full_vocab):
    """Replicate train.m's vocab compaction over the ENTIRE dataset."""
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


# ============================================================
# 3. Compute CE loss with compact vocab + log-sum-exp
#    (matching C's cross_entropy_loss exactly)
# ============================================================
def compact_ce_loss(logits_full, targets_full, compact_to_full, full_to_compact):
    """
    logits_full: [SEQ, full_vocab] -- full-vocab logits from HF model
    targets_full: [SEQ] -- full-vocab target token IDs
    compact_to_full: [CV] -- mapping compact -> full vocab
    full_to_compact: [full_vocab] -- mapping full -> compact vocab (-1 if unused)

    Returns: CE loss matching C's implementation (compact vocab, log-sum-exp)
    """
    # Extract only the compact vocab columns from logits
    logits_compact = logits_full[:, compact_to_full]  # [SEQ, CV]

    SEQ = logits_compact.shape[0]
    total_loss = 0.0

    for t in range(SEQ):
        row = logits_compact[t]  # [CV]
        maxv = row.max()
        shifted = row - maxv
        exp_shifted = np.exp(shifted)
        sum_exp = exp_shifted.sum()

        # target in compact vocab
        tgt_compact = full_to_compact[targets_full[t]]
        assert tgt_compact >= 0, f"Target token {targets_full[t]} not in compact vocab!"

        # loss_t = -(logit[target] - max) + log(sum(exp(logit - max)))
        loss_t = -shifted[tgt_compact] + math.log(sum_exp)
        total_loss += loss_t

    return total_loss / SEQ


def main():
    DATA_PATH = '/Users/vatsalb/Desktop/AutoANE_repo/tinystories_smollm2_data00.bin'
    CKPT_PATH = '/Users/vatsalb/Desktop/AutoANE_repo/training/ane_smollm2_360m_ckpt.bin'
    TRAIN_DIR = '/Users/vatsalb/Desktop/AutoANE_repo/training'
    HF_MODEL = 'HuggingFaceTB/SmolLM2-360M'
    SEQ = 256
    FULL_VOCAB = 49152

    seeds = [42, 100, 200, 300, 400]

    # ---- Load data ----
    print("=" * 70)
    print("MULTI-POSITION VERIFICATION: HuggingFace vs C Training Code")
    print("=" * 70)
    print()

    print("[1] Loading training data...")
    token_data = np.fromfile(DATA_PATH, dtype=np.uint16)
    n_tokens = len(token_data)
    n_train = int(n_tokens * 0.9)
    print(f"    {n_tokens} tokens, train split: {n_train}")

    # ---- Build compact vocab map from ENTIRE dataset ----
    print("\n[2] Building compact vocab map from entire dataset...")
    t0 = time.time()
    full_to_compact, compact_to_full, CV = build_vocab_map(token_data, FULL_VOCAB)
    print(f"    Full vocab: {FULL_VOCAB}, Compact vocab: {CV} ({CV/FULL_VOCAB*100:.1f}% active)")
    print(f"    Built in {time.time()-t0:.1f}s")

    # ---- Compute data positions ----
    print("\n[3] Computing data positions (replicating C drand48)...")
    positions = {}
    for seed in seeds:
        pos = get_data_position(seed, n_tokens, step=0)
        positions[seed] = pos
        print(f"    seed={seed:>3d} -> pos={pos:>8d}  "
              f"(tokens [{pos}:{pos+SEQ}] -> [{pos+1}:{pos+SEQ+1}])")

    # ---- Load HuggingFace model ----
    print(f"\n[4] Loading HuggingFace model: {HF_MODEL}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.float32)
    model.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s")
    print(f"    Config: {model.config.num_hidden_layers}L, dim={model.config.hidden_size}, "
          f"heads={model.config.num_attention_heads}/{model.config.num_key_value_heads}, "
          f"hidden={model.config.intermediate_size}, vocab={model.config.vocab_size}")

    # ---- Compute HF loss at each position ----
    print(f"\n[5] Computing HF compact CE loss at {len(seeds)} positions...")
    hf_losses = {}
    for seed in seeds:
        pos = positions[seed]
        input_ids = token_data[pos:pos + SEQ].astype(np.int64)
        target_ids = token_data[pos + 1:pos + SEQ + 1].astype(np.int64)

        # HF forward pass
        with torch.no_grad():
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            outputs = model(input_tensor)
            logits_hf = outputs.logits[0].float().numpy()  # [SEQ, full_vocab]

        # Compute compact CE loss (matching C code)
        loss = compact_ce_loss(logits_hf, target_ids, compact_to_full, full_to_compact)
        hf_losses[seed] = loss
        print(f"    seed={seed:>3d}  pos={pos:>8d}  HF compact CE = {loss:.6f}")

    # ---- Run C binary for seed=42 ----
    print(f"\n[6] Running C binary (seed=42, lr=0, steps=1)...")
    c_loss_plus = None
    c_loss_minus = None
    c_midpoint = None

    train_mezo_path = os.path.join(TRAIN_DIR, 'train_mezo')
    if os.path.exists(train_mezo_path):
        try:
            result = subprocess.run(
                [train_mezo_path, '--resume', CKPT_PATH,
                 '--steps', '1', '--lr', '0', '--seed', '42',
                 '--cpu-only', '--val-every', '999999'],
                capture_output=True, text=True, timeout=300,
                cwd=TRAIN_DIR
            )
            output = result.stdout + result.stderr
            print(f"    C binary output:")
            for line in output.strip().split('\n'):
                print(f"      {line}")

            # Parse loss values from output (prefer final_loss_* for full precision)
            for line in output.split('\n'):
                stripped = line.strip()
                if stripped.startswith('final_loss_plus:'):
                    c_loss_plus = float(stripped.split(':')[1].strip())
                elif stripped.startswith('final_loss_minus:'):
                    c_loss_minus = float(stripped.split(':')[1].strip())
            # Fallback: parse from step log line if final_loss not found
            if c_loss_plus is None or c_loss_minus is None:
                for line in output.split('\n'):
                    if 'loss_plus=' in line:
                        parts = line.split()
                        for p in parts:
                            if p.startswith('loss_plus='):
                                c_loss_plus = float(p.split('=')[1])
                            elif p.startswith('loss_minus='):
                                c_loss_minus = float(p.split('=')[1])

            if c_loss_plus is not None and c_loss_minus is not None:
                c_midpoint = (c_loss_plus + c_loss_minus) / 2.0
                print(f"\n    Parsed: loss_plus={c_loss_plus:.6f}, loss_minus={c_loss_minus:.6f}")
                print(f"    Midpoint (avg): {c_midpoint:.6f}")
            else:
                print("\n    WARNING: Could not parse loss values from C binary output")
        except subprocess.TimeoutExpired:
            print("    WARNING: C binary timed out (300s)")
        except Exception as e:
            print(f"    WARNING: C binary error: {e}")
    else:
        print(f"    WARNING: {train_mezo_path} not found. Skipping C binary comparison.")
        print(f"    Build with: cd training && make mezo MODEL=smollm2_360m")

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Seed':>6s}  {'Position':>10s}  {'HF Compact CE':>14s}  {'Notes':s}")
    print(f"{'----':>6s}  {'--------':>10s}  {'-------------':>14s}  {'-----':s}")
    for seed in seeds:
        pos = positions[seed]
        loss = hf_losses[seed]
        notes = ""
        if seed == 42 and c_midpoint is not None:
            delta = abs(loss - c_midpoint)
            notes = f"C midpoint={c_midpoint:.6f}, delta={delta:.6f}"
        print(f"{seed:>6d}  {pos:>10d}  {loss:>14.6f}  {notes}")

    if c_loss_plus is not None and c_loss_minus is not None:
        print()
        print(f"C binary (seed=42):")
        print(f"  loss_plus  = {c_loss_plus:.6f}  (weights + epsilon)")
        print(f"  loss_minus = {c_loss_minus:.6f}  (weights - epsilon)")
        print(f"  midpoint   = {c_midpoint:.6f}  (avg, approximates unperturbed)")
        print(f"  epsilon effect: {abs(c_loss_plus - c_loss_minus):.6f}")
        print()
        delta = abs(hf_losses[42] - c_midpoint)
        print(f"HF vs C midpoint delta (seed=42): {delta:.6f}")
        if delta < 0.01:
            print("  --> EXCELLENT: losses match within 0.01")
        elif delta < 0.05:
            print("  --> GOOD: losses match within 0.05 (expected fp32 vs fp32 variance)")
        elif delta < 0.1:
            print("  --> OK: small delta, likely numerical differences")
        else:
            print("  --> WARNING: significant delta, investigate forward pass differences")

    # ---- Loss statistics ----
    losses_arr = np.array([hf_losses[s] for s in seeds])
    print()
    print(f"HF Compact CE statistics across {len(seeds)} positions:")
    print(f"  Mean:   {losses_arr.mean():.6f}")
    print(f"  Std:    {losses_arr.std():.6f}")
    print(f"  Min:    {losses_arr.min():.6f}")
    print(f"  Max:    {losses_arr.max():.6f}")
    print(f"  ln(CV) = ln({CV}) = {math.log(CV):.4f} (untrained baseline)")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
