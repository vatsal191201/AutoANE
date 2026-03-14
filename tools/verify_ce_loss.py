#!/usr/bin/env python3
"""
Numerical verification of cross_entropy_loss() from cpu_ops.h (lines 108-133).

The C function computes:
  - For each row t in [0, S):
      1. maxv = max(logits[t, :])
      2. drow = logits[t, :] - maxv          (subtract max for stability)
      3. drow = exp(drow)                     (elementwise exp)
      4. sum  = sum(drow)                     (sum of exponentials)
      5. drow = drow / sum                    (normalize -> softmax probs)
      6. loss_t = -(logits[t, tgt] - maxv) + log(sum)
      7. drow[tgt] -= 1.0                    (gradient: softmax - one_hot)
      8. drow *= 1/S                          (scale gradient by 1/S)
  - return total_loss / S

This script replicates that EXACTLY in float32, then cross-checks against
a textbook implementation using scipy.special.log_softmax.
"""

import numpy as np
from scipy.special import log_softmax

np.random.seed(42)

FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6


def ce_loss_c_matching(logits_f32, targets):
    """
    Exact replica of the C cross_entropy_loss(), operating in float32.
    logits_f32: np.float32 array of shape [S, V]
    targets:    np.uint16 (or int) array of shape [S]
    Returns: (loss, dlogits)  where dlogits has shape [S, V]
    """
    S, V = logits_f32.shape
    assert logits_f32.dtype == np.float32
    invS = np.float32(1.0 / S)
    dlogits = np.empty_like(logits_f32)
    total_loss = np.float32(0.0)

    for t in range(S):
        row = logits_f32[t, :]                           # const float *row
        # Step 1: maxv = max(row)
        maxv = np.float32(np.max(row))                   # vDSP_maxv
        # Step 2: drow = row - maxv
        neg_max = np.float32(-maxv)
        drow = (row + neg_max).astype(np.float32)        # vDSP_vsadd
        # Step 3: drow = exp(drow)
        drow = np.exp(drow).astype(np.float32)           # vvexpf
        # Step 4: sum = sum(drow)
        sum_exp = np.float32(np.sum(drow))               # vDSP_sve
        # Step 5: drow = drow / sum  (softmax probabilities)
        inv_sum = np.float32(1.0 / sum_exp)
        drow = (drow * inv_sum).astype(np.float32)       # vDSP_vsmul

        tgt = int(targets[t])
        assert 0 <= tgt < V, f"target[{t}]={tgt} OOB (vocab={V})"

        # Step 6: loss_t = -(row[tgt] - maxv) + log(sum)
        loss_t = np.float32(-(row[tgt] - maxv)) + np.float32(np.log(np.float32(sum_exp)))
        total_loss = np.float32(total_loss + loss_t)

        # Step 7: drow[tgt] -= 1.0
        drow[tgt] = np.float32(drow[tgt] - np.float32(1.0))

        # Step 8: drow *= invS
        drow = (drow * invS).astype(np.float32)          # vDSP_vsmul

        dlogits[t, :] = drow

    loss = np.float32(total_loss / np.float32(S))
    return loss, dlogits


def ce_loss_textbook(logits_f64, targets):
    """
    Textbook cross-entropy loss using float64 for reference accuracy.
    loss = mean over t of -log_softmax(logits[t, :])[target[t]]
    grad = (softmax(logits) - one_hot(targets)) / S
    """
    S, V = logits_f64.shape
    log_probs = log_softmax(logits_f64, axis=1)  # [S, V], float64
    losses = np.array([-log_probs[t, targets[t]] for t in range(S)])
    loss = np.mean(losses)

    # Gradient: softmax - one_hot, divided by S
    probs = np.exp(log_probs)  # softmax
    one_hot = np.zeros_like(probs)
    for t in range(S):
        one_hot[t, targets[t]] = 1.0
    dlogits = (probs - one_hot) / S

    return loss, dlogits


def verify_gradient(dlogits, logits_f32, targets, label):
    """
    Verify that dlogits == (softmax(logits) - one_hot(targets)) / S
    computed independently in float64 for reference.
    """
    S, V = logits_f32.shape
    logits_f64 = logits_f32.astype(np.float64)
    log_probs = log_softmax(logits_f64, axis=1)
    probs = np.exp(log_probs)
    one_hot = np.zeros((S, V), dtype=np.float64)
    for t in range(S):
        one_hot[t, targets[t]] = 1.0
    expected_grad = (probs - one_hot) / S

    # Compare in float32
    expected_grad_f32 = expected_grad.astype(np.float32)
    max_abs_err = np.max(np.abs(dlogits - expected_grad_f32))
    mean_abs_err = np.mean(np.abs(dlogits - expected_grad_f32))

    # Use a relative tolerance scaled to the magnitude of the gradient
    max_expected = np.max(np.abs(expected_grad_f32))
    rel_err = max_abs_err / max(max_expected, 1e-30)

    ok = rel_err < 1e-4  # very generous for float32 accumulation differences
    status = "PASS" if ok else "FAIL"
    print(f"  Gradient check [{label}]: {status}")
    print(f"    max|dlogits - expected|   = {max_abs_err:.6e}")
    print(f"    mean|dlogits - expected|  = {mean_abs_err:.6e}")
    print(f"    max|expected|             = {max_expected:.6e}")
    print(f"    relative error            = {rel_err:.6e}")
    return ok


def run_test(name, logits_f32, targets):
    S, V = logits_f32.shape
    print(f"\n{'='*70}")
    print(f"TEST: {name}  (S={S}, V={V})")
    print(f"{'='*70}")

    # C-matching (float32)
    loss_c, dlogits_c = ce_loss_c_matching(logits_f32, targets)

    # Textbook (float64)
    loss_tb, dlogits_tb = ce_loss_textbook(logits_f32.astype(np.float64), targets)

    # Compare losses
    abs_loss_err = abs(float(loss_c) - float(loss_tb))
    rel_loss_err = abs_loss_err / max(abs(float(loss_tb)), 1e-30)
    loss_ok = rel_loss_err < 1e-5
    status = "PASS" if loss_ok else "FAIL"
    print(f"  Loss (C-matching, f32): {loss_c:.8f}")
    print(f"  Loss (textbook, f64):   {loss_tb:.8f}")
    print(f"  Absolute error:         {abs_loss_err:.6e}")
    print(f"  Relative error:         {rel_loss_err:.6e}")
    print(f"  Loss agreement: {status}")

    # Compare gradients: C-matching vs textbook
    dlogits_tb_f32 = dlogits_tb.astype(np.float32)
    grad_max_err = np.max(np.abs(dlogits_c - dlogits_tb_f32))
    grad_mean_err = np.mean(np.abs(dlogits_c - dlogits_tb_f32))
    grad_max_val = max(np.max(np.abs(dlogits_c)), 1e-30)
    grad_rel_err = grad_max_err / grad_max_val
    grad_ok = grad_rel_err < 1e-4
    status = "PASS" if grad_ok else "FAIL"
    print(f"  Gradient max abs error: {grad_max_err:.6e}")
    print(f"  Gradient mean abs err:  {grad_mean_err:.6e}")
    print(f"  Gradient max |value|:   {grad_max_val:.6e}")
    print(f"  Gradient rel error:     {grad_rel_err:.6e}")
    print(f"  Gradient C vs textbook: {status}")

    # Independent gradient verification
    grad_verify_ok = verify_gradient(dlogits_c, logits_f32, targets, name)

    # Verify specific gradient properties
    print(f"  --- Gradient structural checks ---")
    for t in range(min(S, 3)):  # check first few rows
        tgt = targets[t]
        # softmax prob at target
        row = logits_f32[t, :].astype(np.float64)
        row_shifted = row - np.max(row)
        exp_row = np.exp(row_shifted)
        softmax_prob = exp_row[tgt] / np.sum(exp_row)
        expected_grad_at_tgt = (softmax_prob - 1.0) / S
        actual_grad_at_tgt = float(dlogits_c[t, tgt])
        err = abs(actual_grad_at_tgt - expected_grad_at_tgt)
        print(f"    row {t}: dlogits[{t},{tgt}] = {actual_grad_at_tgt:.8f}, "
              f"expected = {expected_grad_at_tgt:.8f}, err = {err:.2e}")

    all_ok = loss_ok and grad_ok and grad_verify_ok
    return all_ok


# ============================================================
# Test cases
# ============================================================

all_pass = True

# --- (a) Small case: V=5, S=3, random logits, known targets ---
logits_a = np.random.randn(3, 5).astype(np.float32)
targets_a = np.array([0, 2, 4], dtype=np.uint16)
all_pass &= run_test("Small (V=5, S=3)", logits_a, targets_a)

# --- (b) Medium case: V=16893, S=256 ---
logits_b = np.random.randn(256, 16893).astype(np.float32)
targets_b = np.random.randint(0, 16893, size=256).astype(np.uint16)
all_pass &= run_test("Medium (V=16893, S=256)", logits_b, targets_b)

# --- (c) Edge case: very large logits (numerical stability) ---
logits_c1 = np.random.randn(4, 10).astype(np.float32)
logits_c1[0, :] *= 100     # scale row 0 to ~100
logits_c1[1, :] += 500     # shift row 1 by +500
logits_c1[2, :] -= 500     # shift row 2 by -500
logits_c1[3, 0] = 88.0     # one dominant logit near exp overflow boundary
logits_c1[3, 1:] = -88.0
targets_c1 = np.array([3, 7, 1, 0], dtype=np.uint16)
all_pass &= run_test("Large logits (stability)", logits_c1, targets_c1)

# --- (c) Edge case: all-same logits ---
logits_c2 = np.full((4, 10), 3.14, dtype=np.float32)
targets_c2 = np.array([0, 5, 9, 3], dtype=np.uint16)
all_pass &= run_test("All-same logits", logits_c2, targets_c2)

# Verify: when all logits are equal, loss = log(V) for each sample
expected_loss_uniform = np.log(10.0)
loss_c2, _ = ce_loss_c_matching(logits_c2, targets_c2)
uniform_err = abs(float(loss_c2) - expected_loss_uniform)
uniform_ok = uniform_err < 1e-5
status = "PASS" if uniform_ok else "FAIL"
print(f"  Uniform logits -> loss should be log(V)={expected_loss_uniform:.6f}, "
      f"got {loss_c2:.6f}, err={uniform_err:.2e}: {status}")
all_pass &= uniform_ok

# --- (c) Edge case: target at index 0 ---
logits_c3 = np.random.randn(2, 20).astype(np.float32)
targets_c3 = np.array([0, 0], dtype=np.uint16)
all_pass &= run_test("Target at index 0", logits_c3, targets_c3)

# --- (c) Edge case: target at last index ---
logits_c4 = np.random.randn(2, 20).astype(np.float32)
targets_c4 = np.array([19, 19], dtype=np.uint16)
all_pass &= run_test("Target at last index (V-1)", logits_c4, targets_c4)

# --- (c) Edge case: single sample (S=1) ---
logits_c5 = np.random.randn(1, 100).astype(np.float32)
targets_c5 = np.array([50], dtype=np.uint16)
all_pass &= run_test("Single sample (S=1)", logits_c5, targets_c5)

# --- Cross-check: loss formula equivalence ---
print(f"\n{'='*70}")
print("ALGEBRAIC VERIFICATION: loss_t = -log(softmax(logits)[tgt])")
print(f"{'='*70}")
logits_v = np.random.randn(8, 50).astype(np.float32)
targets_v = np.random.randint(0, 50, size=8).astype(np.uint16)
loss_c, _ = ce_loss_c_matching(logits_v, targets_v)

# The C code computes: loss_t = -(logits[tgt] - max) + log(sum(exp(logits - max)))
#                              = -logits[tgt] + max + log(sum(exp(logits - max)))
#                              = -logits[tgt] + log(exp(max) * sum(exp(logits - max)))
#                              = -logits[tgt] + log(sum(exp(logits)))
#                              = -log(exp(logits[tgt]) / sum(exp(logits)))
#                              = -log(softmax(logits)[tgt])
# This is exactly the standard cross-entropy loss.
# Verify:
manual_losses = []
for t in range(8):
    row = logits_v[t, :].astype(np.float64)
    maxv = np.max(row)
    log_sum_exp = maxv + np.log(np.sum(np.exp(row - maxv)))
    loss_t = -row[targets_v[t]] + log_sum_exp
    manual_losses.append(loss_t)
manual_loss = np.mean(manual_losses)
err = abs(float(loss_c) - manual_loss)
ok = err < 1e-5
status = "PASS" if ok else "FAIL"
print(f"  C-matching loss:  {loss_c:.8f}")
print(f"  Manual LSE loss:  {manual_loss:.8f}")
print(f"  Error:            {err:.2e}")
print(f"  Equivalence:      {status}")
all_pass &= ok

# --- Final summary ---
print(f"\n{'='*70}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print(f"{'='*70}")
