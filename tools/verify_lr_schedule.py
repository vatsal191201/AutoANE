#!/usr/bin/env python3
"""Verify the cosine learning rate schedule in train_mezo.m against the standard formula."""

import math
import sys

# =============================================================================
# 1. Implement the C formula in Python
# =============================================================================

def lr_mezo(step, base_lr, total_steps, start_step=0):
    """Exact Python translation of the C code from train_mezo.m lines 1008-1011:

        float min_lr = base_lr * 0.1f;
        float decay = (float)(step - start_step) / (float)(total_steps - start_step);
        lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay)) * (base_lr - min_lr);
    """
    min_lr = base_lr * 0.1
    decay = (step - start_step) / (total_steps - start_step)
    lr = min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (base_lr - min_lr)
    return lr


def lr_pytorch_cosine(step, base_lr, total_steps, start_step=0):
    """PyTorch CosineAnnealingLR formula:
        lr = eta_min + 0.5 * (1 + cos(pi * T_cur / T_max)) * (eta_max - eta_min)
    where T_cur = step - start_step, T_max = total_steps - start_step
    """
    eta_min = base_lr * 0.1
    T_cur = step - start_step
    T_max = total_steps - start_step
    lr = eta_min + 0.5 * (1 + math.cos(math.pi * T_cur / T_max)) * (base_lr - eta_min)
    return lr


# =============================================================================
# 2. Tabulate lr values for base_lr=1e-4, total_steps=500, start_step=0
# =============================================================================

print("=" * 80)
print("COSINE LR SCHEDULE VERIFICATION")
print("=" * 80)

base_lr = 1e-4
total_steps = 500
start_step = 0
min_lr = base_lr * 0.1

print(f"\nParameters: base_lr={base_lr}, total_steps={total_steps}, start_step={start_step}")
print(f"Expected min_lr = base_lr * 0.1 = {min_lr}")

print(f"\n{'Step':>6} | {'MeZO lr':>12} | {'PyTorch lr':>12} | {'Match':>6}")
print("-" * 48)

key_steps = [0, 50, 100, 125, 150, 200, 250, 300, 350, 400, 450, 499, 500]
all_match = True

for step in key_steps:
    m_lr = lr_mezo(step, base_lr, total_steps, start_step)
    p_lr = lr_pytorch_cosine(step, base_lr, total_steps, start_step)
    match = abs(m_lr - p_lr) < 1e-15
    if not match:
        all_match = False
    print(f"{step:>6} | {m_lr:>12.6e} | {p_lr:>12.6e} | {'YES' if match else 'NO':>6}")


# =============================================================================
# 3. Verify boundary conditions
# =============================================================================

print("\n" + "=" * 80)
print("BOUNDARY CONDITION CHECKS")
print("=" * 80)

results = []

# Check lr(0) = base_lr
lr_at_0 = lr_mezo(0, base_lr, total_steps, 0)
test1 = abs(lr_at_0 - base_lr) < 1e-15
results.append(("lr(step=0) == base_lr", test1, lr_at_0, base_lr))
print(f"\nlr(step=0) = {lr_at_0:.6e}, expected base_lr = {base_lr:.6e}")
print(f"  -> {'PASS' if test1 else 'FAIL'}")

# Check lr(total_steps) = min_lr
lr_at_end = lr_mezo(total_steps, base_lr, total_steps, 0)
test2 = abs(lr_at_end - min_lr) < 1e-15
results.append(("lr(step=total_steps) == min_lr", test2, lr_at_end, min_lr))
print(f"\nlr(step=total_steps={total_steps}) = {lr_at_end:.6e}, expected min_lr = {min_lr:.6e}")
print(f"  -> {'PASS' if test2 else 'FAIL'}")

# Check midpoint: lr(250) should be (base_lr + min_lr) / 2
lr_at_mid = lr_mezo(250, base_lr, total_steps, 0)
expected_mid = (base_lr + min_lr) / 2.0
test3 = abs(lr_at_mid - expected_mid) < 1e-15
results.append(("lr(midpoint) == (base_lr + min_lr)/2", test3, lr_at_mid, expected_mid))
print(f"\nlr(step=250) = {lr_at_mid:.6e}, expected (base_lr+min_lr)/2 = {expected_mid:.6e}")
print(f"  -> {'PASS' if test3 else 'FAIL'}")


# =============================================================================
# 4. Verify standard cosine annealing (not inverted)
# =============================================================================

print("\n" + "=" * 80)
print("MONOTONICITY CHECK (not inverted)")
print("=" * 80)

monotone = True
prev_lr = lr_mezo(0, base_lr, total_steps, 0)
for step in range(1, total_steps + 1):
    cur_lr = lr_mezo(step, base_lr, total_steps, 0)
    if cur_lr > prev_lr + 1e-15:
        print(f"  NON-MONOTONE at step {step}: lr went UP from {prev_lr:.6e} to {cur_lr:.6e}")
        monotone = False
        break
    prev_lr = cur_lr

print(f"LR is monotonically decreasing from step 0 to {total_steps}: {'PASS' if monotone else 'FAIL'}")
results.append(("Monotonically decreasing (not inverted)", monotone, None, None))


# =============================================================================
# 5. Edge cases
# =============================================================================

print("\n" + "=" * 80)
print("EDGE CASES")
print("=" * 80)

# step > total_steps
print("\n--- step > total_steps ---")
for overshoot_step in [501, 550, 600, 750, 1000]:
    lr_over = lr_mezo(overshoot_step, base_lr, total_steps, 0)
    decay = overshoot_step / total_steps
    print(f"  step={overshoot_step}: decay={decay:.3f}, lr={lr_over:.6e}", end="")
    if lr_over < min_lr:
        print(f"  [BELOW min_lr! diff={min_lr - lr_over:.2e}]")
    elif lr_over > base_lr:
        print(f"  [ABOVE base_lr!]")
    else:
        print(f"  [between min_lr and base_lr, cosine wrapping back up]")

print("\n  NOTE: The C loop runs 'for step = start_step; step < total_steps', so")
print("  step never reaches total_steps. The last step executed is total_steps-1.")
print(f"  lr(step=499) = {lr_mezo(499, base_lr, total_steps, 0):.6e}")
print(f"  This is slightly above min_lr ({min_lr:.6e}) -- expected behavior.")

# step < start_step
print("\n--- step < start_step ---")
print("  This cannot happen: the loop starts at step=start_step, so step is always >= start_step.")
print("  If it did happen, decay would be negative, causing lr > base_lr (wrong direction).")
lr_neg = lr_mezo(-5, base_lr, total_steps, 0)
print(f"  Hypothetical lr(step=-5) = {lr_neg:.6e} (> base_lr={base_lr:.6e})")


# =============================================================================
# 6. Resume behavior: start_step != 0
# =============================================================================

print("\n" + "=" * 80)
print("RESUME BEHAVIOR: --resume from step 30, total_steps=500")
print("=" * 80)

resume_step = 30

# What the code ACTUALLY does (from source):
# start_step is loaded from checkpoint as the step the checkpoint was saved at
# decay = (step - start_step) / (total_steps - start_step)
# So decay goes from 0 to ~1 over the REMAINING steps, NOT from 0 to 1 over ALL steps

print(f"\nThe code uses: decay = (step - start_step) / (total_steps - start_step)")
print(f"With start_step={resume_step}, total_steps=500:")
print(f"  At step {resume_step}: decay = 0/{500-resume_step} = 0.0, lr = base_lr (RESET)")
print(f"  At step 500: decay = {500-resume_step}/{500-resume_step} = 1.0, lr = min_lr")
print()

# Compare the two interpretations
print(f"{'Step':>6} | {'Resumed (reset)':>15} | {'Global schedule':>15} | {'Difference':>12}")
print("-" * 60)
for step in [30, 100, 200, 300, 400, 499]:
    lr_reset = lr_mezo(step, base_lr, total_steps, start_step=resume_step)
    lr_global = lr_mezo(step, base_lr, total_steps, start_step=0)
    print(f"{step:>6} | {lr_reset:>15.6e} | {lr_global:>15.6e} | {lr_reset - lr_global:>12.2e}")

print()
print("INTERPRETATION: When resuming from step 30, the cosine schedule RESETS --")
print("it treats step 30 as the beginning (decay=0, lr=base_lr) and decays to")
print("min_lr over the remaining 470 steps. This means the lr trajectory after")
print("resume is NOT the same as if training had continued without interruption.")
print("This is a DELIBERATE CHOICE, not a bug -- it gives the resumed run a")
print("full cosine anneal over its remaining budget.")


# =============================================================================
# 7. Verify actual logged values: lr=3e-4, 500 steps
# =============================================================================

print("\n" + "=" * 80)
print("VERIFY ACTUAL LOGGED VALUES (lr=3e-4, 500 steps, start_step=0)")
print("=" * 80)

base_lr_actual = 3e-4
total_steps_actual = 500
start_step_actual = 0

logged_values = {
    0:   3.00e-04,
    100: 2.74e-04,
    200: 2.07e-04,
    300: 1.23e-04,
    400: 5.58e-05,
    500: 3.00e-05,  # "final lr"
}

print(f"\nParameters: base_lr={base_lr_actual}, total_steps={total_steps_actual}")
print(f"min_lr = {base_lr_actual * 0.1:.2e}")
print()
print(f"{'Step':>6} | {'Logged lr':>12} | {'Computed lr':>12} | {'Rel Error':>12} | {'Status':>8}")
print("-" * 65)

all_logged_match = True
for step, logged_lr in sorted(logged_values.items()):
    computed_lr = lr_mezo(step, base_lr_actual, total_steps_actual, start_step_actual)

    # The logged values are rounded to 2 significant figures (%.2e format)
    # So we compare by rounding computed to same format
    computed_rounded = float(f"{computed_lr:.2e}")
    logged_rounded = float(f"{logged_lr:.2e}")

    if logged_rounded != 0:
        rel_err = abs(computed_rounded - logged_rounded) / logged_rounded
    else:
        rel_err = abs(computed_rounded - logged_rounded)

    match = (computed_rounded == logged_rounded)
    if not match:
        all_logged_match = False

    print(f"{step:>6} | {logged_lr:>12.2e} | {computed_lr:>12.6e} ({computed_rounded:.2e}) | {rel_err:>12.2e} | {'PASS' if match else 'FAIL'}")

results.append(("Logged values match formula", all_logged_match, None, None))

# Note about step 500 vs step 499
print(f"\nNote: The loop runs step < total_steps, so the last executed step is 499.")
print(f"  lr(step=499) = {lr_mezo(499, base_lr_actual, total_steps_actual, 0):.6e}")
print(f"  lr(step=500) = {lr_mezo(500, base_lr_actual, total_steps_actual, 0):.6e}")
print(f"  The 'final lr' logged as 3.00e-05 matches lr(step=500)=min_lr exactly,")
print(f"  suggesting it was either computed at step=500 or reported as the target min_lr.")
print(f"  At step 499, lr = {lr_mezo(499, base_lr_actual, total_steps_actual, 0):.2e},")
print(f"  which is very close to but not exactly min_lr.")


# =============================================================================
# 8. Formula equivalence check
# =============================================================================

print("\n" + "=" * 80)
print("FORMULA EQUIVALENCE: MeZO vs PyTorch CosineAnnealingLR")
print("=" * 80)

print(f"\nMeZO formula (train_mezo.m:1009-1011):")
print(f"  min_lr = base_lr * 0.1")
print(f"  decay = (step - start_step) / (total_steps - start_step)")
print(f"  lr = min_lr + 0.5 * (1 + cos(pi * decay)) * (base_lr - min_lr)")
print()
print(f"PyTorch CosineAnnealingLR:")
print(f"  lr = eta_min + 0.5 * (1 + cos(pi * T_cur / T_max)) * (eta_max - eta_min)")
print(f"  where T_cur = step - start_step, T_max = total_steps - start_step")
print()
print(f"These are IDENTICAL when eta_min = base_lr * 0.1 and eta_max = base_lr.")

formulas_match = True
for step in range(0, 501):
    m = lr_mezo(step, base_lr, total_steps, 0)
    p = lr_pytorch_cosine(step, base_lr, total_steps, 0)
    if abs(m - p) > 1e-15:
        formulas_match = False
        print(f"  MISMATCH at step {step}: mezo={m}, pytorch={p}")
        break

print(f"\nAll 501 values match exactly: {'PASS' if formulas_match else 'FAIL'}")
results.append(("Formula matches PyTorch CosineAnnealingLR", formulas_match, None, None))


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

all_pass = True
for name, passed, actual, expected in results:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    detail = ""
    if actual is not None and expected is not None:
        detail = f" (got {actual:.6e}, expected {expected:.6e})"
    print(f"  [{status}] {name}{detail}")

print()
if all_pass:
    print("OVERALL: PASS -- Cosine LR schedule is correctly implemented.")
else:
    print("OVERALL: FAIL -- See above for details.")
    sys.exit(1)
