#!/usr/bin/env python3
"""Verify whether the CE loss epsilon guard bug affected MeZO gradient estimates.

MeZO computes: projected_grad = (loss_plus - loss_minus) / (2 * epsilon)

The old CE code used: -mean(log(softmax_prob + 1e-10))
The new CE code uses: mean(log-sum-exp)  (numerically stable, no guard)

If the bias from the epsilon guard is approximately constant across +/- perturbations,
it cancels in the gradient estimate. This script quantifies whether that cancellation
actually occurs.

Uses HuggingFace SmolLM2-360M and the same data/batch as the C training code
(seed=42, step=0).
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. Replicate macOS drand48 (48-bit LCG)
# ============================================================
# macOS drand48 uses: X_{n+1} = (a * X_n + c) mod m
# a = 0x5DEECE66D, c = 0xB, m = 2^48
# srand48(seed) sets the state to (seed << 16) | 0x330E

DRAND48_A = 0x5DEECE66D
DRAND48_C = 0xB
DRAND48_M = 1 << 48

class Drand48:
    def __init__(self):
        self.state = 0

    def srand48(self, seed):
        # macOS srand48: state = (seed << 16) | 0x330E
        self.state = ((seed & 0xFFFFFFFF) << 16) | 0x330E

    def drand48(self):
        self.state = (DRAND48_A * self.state + DRAND48_C) % DRAND48_M
        return self.state / float(DRAND48_M)


# ============================================================
# 2. Load data and build vocab map
# ============================================================
def load_data_and_vocab(data_path, full_vocab=49152):
    """Load tokenized binary data and build vocab compaction map."""
    token_data = np.fromfile(data_path, dtype=np.uint16)
    n_tokens = len(token_data)
    val_start = int(n_tokens * 0.9)
    train_tokens = val_start

    # Build compact vocab map (same as C code)
    used = set(token_data.tolist())
    full_to_compact = np.full(full_vocab, -1, dtype=np.int32)
    compact_to_full = []
    cid = 0
    for v in range(full_vocab):
        if v in used:
            full_to_compact[v] = cid
            compact_to_full.append(v)
            cid += 1
    compact_to_full = np.array(compact_to_full, dtype=np.int32)
    CV = cid

    return token_data, train_tokens, full_to_compact, compact_to_full, CV


# ============================================================
# 3. Get data position for seed=42, step=0
# ============================================================
def get_data_position(init_seed, step, train_tokens, seq_len):
    """Replicate C code: srand48(init_seed + step * 7919); pos = drand48() * max_pos"""
    rng = Drand48()
    rng.srand48(init_seed + step * 7919)
    max_pos = train_tokens - seq_len - 1
    pos = int(rng.drand48() * max_pos)
    return pos


# ============================================================
# 4. CE loss functions (old with epsilon guard vs new log-sum-exp)
# ============================================================
def ce_loss_old(logits, targets):
    """OLD formula: -mean(log(softmax[target] + 1e-10))
    logits: [seq, vocab], targets: [seq] (compact vocab IDs)
    """
    # Stable softmax
    logits_max = logits.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits - logits_max)
    softmax_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

    # Gather target probabilities
    seq = logits.shape[0]
    target_probs = softmax_probs[torch.arange(seq), targets]

    # Old formula with epsilon guard
    loss = -torch.log(target_probs + 1e-10).mean()
    return loss


def ce_loss_new(logits, targets):
    """NEW formula: mean(log-sum-exp - target_logit)
    Numerically stable, no epsilon guard.
    logits: [seq, vocab], targets: [seq] (compact vocab IDs)
    """
    # log-sum-exp formulation
    logits_max = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(logits - logits_max).sum(dim=-1))

    seq = logits.shape[0]
    target_logits = logits[torch.arange(seq), targets] - logits_max.squeeze(-1)

    loss = (-target_logits + log_sum_exp).mean()
    return loss


# ============================================================
# MAIN
# ============================================================
def main():
    repo_dir = "/Users/vatsalb/Desktop/AutoANE_repo"
    data_path = os.path.join(repo_dir, "tinystories_smollm2_data00.bin")

    if not os.path.exists(data_path):
        print("ERROR: Data file not found: " + data_path)
        sys.exit(1)

    # Configuration (matching C code)
    INIT_SEED = 42
    STEP = 0
    SEQ = 128
    EPSILON = 0.001
    FULL_VOCAB = 49152

    print("=" * 70)
    print("MeZO Gradient Bias Verification")
    print("Does the CE epsilon guard bug affect gradient estimates?")
    print("=" * 70)

    # ---- Step 1: Load data and build vocab map ----
    print("\n[1] Loading data...")
    token_data, train_tokens, full_to_compact, compact_to_full, CV = \
        load_data_and_vocab(data_path, FULL_VOCAB)
    print("    Tokens: {}, Train: {}, Compact vocab: {}".format(len(token_data), train_tokens, CV))

    # ---- Step 2: Get batch position for seed=42, step=0 ----
    print("\n[2] Computing data position (seed={}, step={})...".format(INIT_SEED, STEP))
    pos = get_data_position(INIT_SEED, STEP, train_tokens, SEQ)
    print("    Data position: {}".format(pos))

    input_tokens = token_data[pos:pos + SEQ].astype(np.int64)
    target_tokens_full = token_data[pos + 1:pos + SEQ + 1].astype(np.int64)
    target_tokens_compact = np.array([full_to_compact[t] for t in target_tokens_full],
                                      dtype=np.int64)
    print("    Input tokens[0:5]: {}".format(input_tokens[:5]))
    print("    Target tokens (compact)[0:5]: {}".format(target_tokens_compact[:5]))

    # ---- Step 3: Load SmolLM2-360M ----
    print("\n[3] Loading SmolLM2-360M from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-360M",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("    Model loaded. Parameters: {:.1f}M".format(n_params / 1e6))

    # Convert data to torch
    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)  # [1, SEQ]
    targets_compact = torch.tensor(target_tokens_compact, dtype=torch.long)
    compact_to_full_tensor = torch.tensor(compact_to_full, dtype=torch.long)

    # ---- Step 4: Compute baseline logits (unperturbed) ----
    print("\n[4] Computing unperturbed logits...")
    with torch.no_grad():
        hidden_base = model.model(input_ids).last_hidden_state[0]  # [SEQ, dim]
    embed_weight = model.get_input_embeddings().weight.detach()  # [vocab, dim]
    compact_embed = embed_weight[compact_to_full_tensor]  # [CV, dim]
    logits_base = hidden_base @ compact_embed.T  # [SEQ, CV]

    old_loss_base = ce_loss_old(logits_base, targets_compact).item()
    new_loss_base = ce_loss_new(logits_base, targets_compact).item()
    bias_base = old_loss_base - new_loss_base

    print("    OLD CE loss (with epsilon guard): {:.8f}".format(old_loss_base))
    print("    NEW CE loss (log-sum-exp):        {:.8f}".format(new_loss_base))
    print("    Bias (old - new):                 {:.8f}".format(bias_base))
    print("    Relative bias:                    {:.4f}%".format(bias_base / new_loss_base * 100))

    # Per-position analysis
    logits_max = logits_base.max(dim=-1, keepdim=True).values
    exp_logits = torch.exp(logits_base - logits_max)
    softmax_probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    target_probs = softmax_probs[torch.arange(SEQ), targets_compact]

    print("\n    Per-position target probability stats:")
    print("      min:    {:.6e}".format(target_probs.min().item()))
    print("      max:    {:.6e}".format(target_probs.max().item()))
    print("      median: {:.6e}".format(target_probs.median().item()))
    print("      mean:   {:.6e}".format(target_probs.mean().item()))

    # Find positions where epsilon guard matters most
    eps_contribution = torch.abs(torch.log(target_probs + 1e-10) - torch.log(target_probs))
    worst_pos = torch.argmax(eps_contribution).item()
    print("\n    Worst epsilon-guard position: {}".format(worst_pos))
    print("      Target prob:     {:.6e}".format(target_probs[worst_pos].item()))
    print("      log(p):          {:.6f}".format(torch.log(target_probs[worst_pos]).item()))
    print("      log(p + 1e-10):  {:.6f}".format(torch.log(target_probs[worst_pos] + 1e-10).item()))
    print("      Position error:  {:.6f}".format(eps_contribution[worst_pos].item()))

    # ---- Step 5: MeZO perturbation ----
    print("\n[5] Simulating MeZO perturbation (epsilon={})...".format(EPSILON))
    print("    Perturbing layer 0 Wq weights only (single-layer MeZO simulation)")

    # Get the specific parameter to perturb: layer 0, self_attn.q_proj.weight
    param_name = "model.layers.0.self_attn.q_proj.weight"
    param = None
    for name, p in model.named_parameters():
        if name == param_name:
            param = p
            break

    if param is None:
        print("    ERROR: Could not find parameter " + param_name)
        sys.exit(1)

    print("    Parameter: {}, shape: {}, numel: {}".format(param_name, param.shape, param.numel()))

    # Generate random perturbation direction (Rademacher: +1/-1)
    torch.manual_seed(12345)
    z = torch.randint(0, 2, param.shape, dtype=torch.float32) * 2 - 1  # {-1, +1}

    # ---- Step 5a: Perturb +epsilon ----
    print("\n    [5a] Perturbation +epsilon...")
    with torch.no_grad():
        param.add_(EPSILON * z)

    with torch.no_grad():
        hidden_plus = model.model(input_ids).last_hidden_state[0]
    # Recompute compact embed since embedding layer is NOT perturbed
    logits_plus = hidden_plus @ compact_embed.T

    old_loss_plus = ce_loss_old(logits_plus, targets_compact).item()
    new_loss_plus = ce_loss_new(logits_plus, targets_compact).item()

    print("    OLD loss+: {:.8f}".format(old_loss_plus))
    print("    NEW loss+: {:.8f}".format(new_loss_plus))

    # ---- Step 5b: Perturb -2*epsilon (to get theta - epsilon*z) ----
    print("\n    [5b] Perturbation -epsilon...")
    with torch.no_grad():
        param.add_(-2 * EPSILON * z)  # from +eps to -eps

    with torch.no_grad():
        hidden_minus = model.model(input_ids).last_hidden_state[0]
    logits_minus = hidden_minus @ compact_embed.T

    old_loss_minus = ce_loss_old(logits_minus, targets_compact).item()
    new_loss_minus = ce_loss_new(logits_minus, targets_compact).item()

    print("    OLD loss-: {:.8f}".format(old_loss_minus))
    print("    NEW loss-: {:.8f}".format(new_loss_minus))

    # ---- Step 5c: Restore weights ----
    with torch.no_grad():
        param.add_(EPSILON * z)  # restore to original

    # ---- Step 6: Compute MeZO gradient estimates ----
    print("\n" + "=" * 70)
    print("[6] MeZO Gradient Estimates")
    print("=" * 70)

    old_grad = (old_loss_plus - old_loss_minus) / (2 * EPSILON)
    new_grad = (new_loss_plus - new_loss_minus) / (2 * EPSILON)
    gradient_bias = old_grad - new_grad

    print("\n    old_grad = (old_loss+ - old_loss-) / (2*eps)")
    print("             = ({:.8f} - {:.8f}) / {:.4f}".format(old_loss_plus, old_loss_minus, 2 * EPSILON))
    print("             = {:.8f}".format(old_grad))

    print("\n    new_grad = (new_loss+ - new_loss-) / (2*eps)")
    print("             = ({:.8f} - {:.8f}) / {:.4f}".format(new_loss_plus, new_loss_minus, 2 * EPSILON))
    print("             = {:.8f}".format(new_grad))

    print("\n    gradient_bias = old_grad - new_grad")
    print("                  = {:.8f}".format(gradient_bias))

    if abs(new_grad) > 1e-15:
        relative_bias = gradient_bias / new_grad
        print("\n    relative_gradient_bias = gradient_bias / new_grad")
        print("                          = {:.8f}".format(relative_bias))
        print("                          = {:.4f}%".format(relative_bias * 100))
    else:
        relative_bias = float('inf') if gradient_bias != 0 else 0
        print("\n    new_grad is ~0, relative bias is undefined (new_grad={:.2e})".format(new_grad))

    # ---- Step 7: Detailed breakdown ----
    print("\n" + "=" * 70)
    print("[7] Detailed Analysis")
    print("=" * 70)

    old_bias_plus = old_loss_plus - new_loss_plus
    old_bias_minus = old_loss_minus - new_loss_minus

    print("\n    Bias in loss+ (old - new):  {:.10f}".format(old_bias_plus))
    print("    Bias in loss- (old - new):  {:.10f}".format(old_bias_minus))
    print("    Bias difference:            {:.10f}".format(old_bias_plus - old_bias_minus))
    print("    (This is what feeds into gradient_bias * 2*eps)")

    print("\n    Key insight: The gradient bias depends on how much the")
    print("    epsilon-guard bias CHANGES between +/- perturbations.")
    print("    If the bias is constant, it cancels perfectly in the gradient.")

    bias_change_pct = abs(old_bias_plus - old_bias_minus) / max(abs(old_bias_plus), 1e-20) * 100
    print("\n    Bias change as % of bias magnitude: {:.4f}%".format(bias_change_pct))

    # ---- Step 8: Multiple perturbation directions ----
    print("\n" + "=" * 70)
    print("[8] Statistical Analysis: 10 Random Perturbation Directions")
    print("=" * 70)

    old_grads = []
    new_grads = []
    grad_biases = []
    rel_biases = []

    for trial in range(10):
        torch.manual_seed(trial * 1000 + 42)
        z_trial = torch.randint(0, 2, param.shape, dtype=torch.float32) * 2 - 1

        # +epsilon
        with torch.no_grad():
            param.add_(EPSILON * z_trial)
        with torch.no_grad():
            h_plus = model.model(input_ids).last_hidden_state[0]
        l_plus = h_plus @ compact_embed.T
        old_lp = ce_loss_old(l_plus, targets_compact).item()
        new_lp = ce_loss_new(l_plus, targets_compact).item()

        # -2*epsilon
        with torch.no_grad():
            param.add_(-2 * EPSILON * z_trial)
        with torch.no_grad():
            h_minus = model.model(input_ids).last_hidden_state[0]
        l_minus = h_minus @ compact_embed.T
        old_lm = ce_loss_old(l_minus, targets_compact).item()
        new_lm = ce_loss_new(l_minus, targets_compact).item()

        # Restore
        with torch.no_grad():
            param.add_(EPSILON * z_trial)

        og = (old_lp - old_lm) / (2 * EPSILON)
        ng = (new_lp - new_lm) / (2 * EPSILON)
        gb = og - ng
        rb = gb / ng if abs(ng) > 1e-15 else float('nan')

        old_grads.append(og)
        new_grads.append(ng)
        grad_biases.append(gb)
        rel_biases.append(rb)

        print("    Trial {:2d}: old_grad={:+.8f}  new_grad={:+.8f}  "
              "bias={:+.2e}  rel_bias={:+.4f}".format(trial, og, ng, gb, rb))

    # Summary statistics
    mean_og = np.mean(old_grads)
    mean_ng = np.mean(new_grads)
    mean_gb = np.mean(grad_biases)
    std_gb = np.std(grad_biases)
    mean_rb = np.nanmean(rel_biases)
    std_rb = np.nanstd(rel_biases)

    print("\n    Mean old_grad:        {:+.8f}".format(mean_og))
    print("    Mean new_grad:        {:+.8f}".format(mean_ng))
    print("    Mean gradient_bias:   {:+.2e}".format(mean_gb))
    print("    Std  gradient_bias:   {:.2e}".format(std_gb))
    print("    Mean relative_bias:   {:+.6f} ({:+.4f}%)".format(mean_rb, mean_rb * 100))
    print("    Std  relative_bias:   {:.6f} ({:.4f}%)".format(std_rb, std_rb * 100))

    # ---- Step 9: Conclusion ----
    print("\n" + "=" * 70)
    print("[9] Conclusion")
    print("=" * 70)

    # Compare gradient bias to gradient magnitude
    mean_grad_magnitude = np.mean(np.abs(new_grads))
    mean_bias_magnitude = np.mean(np.abs(grad_biases))
    ratio = mean_bias_magnitude / mean_grad_magnitude if mean_grad_magnitude > 0 else float('inf')

    print("\n    Mean |gradient|:       {:.2e}".format(mean_grad_magnitude))
    print("    Mean |gradient_bias|:  {:.2e}".format(mean_bias_magnitude))
    print("    |bias| / |gradient|:   {:.2e} ({:.4f}%)".format(ratio, ratio * 100))

    if ratio < 0.01:
        print("\n    VERDICT: The CE epsilon guard bug has NEGLIGIBLE effect on MeZO")
        print("    gradient estimates ({:.4f}% bias). The bias largely cancels".format(ratio * 100))
        print("    between +/- perturbations because it is nearly constant.")
        print("    The claim that 'training gradients were unaffected' is CONFIRMED.")
    elif ratio < 0.10:
        print("\n    VERDICT: The CE epsilon guard bug has SMALL but non-negligible")
        print("    effect on MeZO gradient estimates ({:.2f}% bias).".format(ratio * 100))
        print("    The bias does NOT fully cancel between +/- perturbations.")
    else:
        print("\n    VERDICT: The CE epsilon guard bug has SIGNIFICANT effect on MeZO")
        print("    gradient estimates ({:.2f}% bias).".format(ratio * 100))
        print("    The bias does NOT cancel between +/- perturbations.")

    # Cross-check: does the loss bias itself change between perturbations?
    print("\n    Additional context:")
    print("    - Unperturbed loss bias (old-new): {:.8f}".format(bias_base))
    print("    - This bias is {:.4f}% of the loss".format(bias_base / new_loss_base * 100))
    print("    - But gradient bias is only {:.4f}% of gradient magnitude".format(ratio * 100))
    if ratio < 0.01:
        print("    - The loss bias is ~constant, so it cancels in (L+ - L-)/(2*eps)")


if __name__ == "__main__":
    main()
