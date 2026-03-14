#!/usr/bin/env python3
"""
MeZO Convergence Analysis: Empirical vs Theoretical
Compares training curves against zeroth-order optimization convergence theory.
"""

import re
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Parse training output
# ============================================================

with open('/tmp/mezo_convergence_output.txt', 'r') as f:
    output = f.read()

# Parse step logs: step N  loss_plus=X  loss_minus=Y  proj_grad=Z  lr=W
step_pattern = r'step (\d+)\s+loss_plus=([\d.]+)\s+loss_minus=([\d.]+)\s+proj_grad=([\-\d.]+)\s+lr=([\d.e\-+]+)'
step_matches = re.findall(step_pattern, output)

# Parse val_loss: [val_loss=X at step Y]
val_pattern = r'\[val_loss=([\d.]+) at step (\d+)\]'
val_matches = re.findall(val_pattern, output)

# Build data arrays
train_steps = []
loss_plus_arr = []
loss_minus_arr = []
proj_grad_arr = []
lr_arr = []

for m in step_matches:
    train_steps.append(int(m[0]))
    loss_plus_arr.append(float(m[1]))
    loss_minus_arr.append(float(m[2]))
    proj_grad_arr.append(float(m[3]))
    lr_arr.append(float(m[4]))

val_steps = []
val_losses = []
for m in val_matches:
    val_losses.append(float(m[0]))
    val_steps.append(int(m[1]))

train_steps = np.array(train_steps)
loss_plus = np.array(loss_plus_arr)
loss_minus = np.array(loss_minus_arr)
proj_grad = np.array(proj_grad_arr)
lr_values = np.array(lr_arr)
val_steps = np.array(val_steps)
val_losses = np.array(val_losses)

# Average of loss_plus and loss_minus as the "train loss" per logged step
train_loss = (loss_plus + loss_minus) / 2.0

print("=" * 70)
print("MeZO CONVERGENCE ANALYSIS: EMPIRICAL vs THEORY")
print("=" * 70)

# ============================================================
# 2. Basic Statistics
# ============================================================

print("\n--- RAW DATA ---")
print(f"{'Step':>6}  {'loss+':>8}  {'loss-':>8}  {'train_avg':>10}  {'proj_grad':>10}  {'lr':>10}")
for i in range(len(train_steps)):
    print(f"{train_steps[i]:>6}  {loss_plus[i]:>8.4f}  {loss_minus[i]:>8.4f}  "
          f"{train_loss[i]:>10.4f}  {proj_grad[i]:>10.6f}  {lr_values[i]:>10.2e}")

print(f"\n{'Step':>6}  {'val_loss':>8}")
for i in range(len(val_steps)):
    print(f"{val_steps[i]:>6}  {val_losses[i]:>8.4f}")

baseline_val = 2.0718  # Given baseline
initial_val = val_losses[0]
final_val = val_losses[-1]
best_val = np.min(val_losses)
best_val_step = val_steps[np.argmin(val_losses)]

print(f"\n--- CONVERGENCE SUMMARY ---")
print(f"Baseline val_loss (pretrained):   {baseline_val:.4f}")
print(f"Initial val_loss (step {val_steps[0]:d}):     {initial_val:.4f}")
print(f"Final val_loss (step {val_steps[-1]:d}):      {final_val:.4f}")
print(f"Best val_loss (step {best_val_step:d}):       {best_val:.4f}")
print(f"Absolute reduction (baseline):    {baseline_val - final_val:.4f}")
print(f"Relative reduction (baseline):    {(baseline_val - final_val)/baseline_val*100:.2f}%")
print(f"Absolute reduction (initial):     {initial_val - final_val:.4f}")
print(f"Relative reduction (initial):     {(initial_val - final_val)/initial_val*100:.2f}%")

# ============================================================
# 3. Early vs Late convergence rate
# ============================================================

# Use val_loss for reliable comparison
# "First 200 steps" = val loss at step 100 to step 300 (first few val points)
# "Last 200 steps" = val loss at step 800 to step 1000

# Early: steps 100-300 (indices depending on data)
early_mask = (val_steps >= 100) & (val_steps <= 300)
late_mask = (val_steps >= 800) & (val_steps <= 1000)

early_steps = val_steps[early_mask]
early_losses = val_losses[early_mask]
late_steps = val_steps[late_mask]
late_losses = val_losses[late_mask]

if len(early_steps) >= 2:
    early_rate = (early_losses[-1] - early_losses[0]) / (early_steps[-1] - early_steps[0])
else:
    early_rate = 0

if len(late_steps) >= 2:
    late_rate = (late_losses[-1] - late_losses[0]) / (late_steps[-1] - late_steps[0])
else:
    late_rate = 0

print(f"\n--- CONVERGENCE RATE: EARLY vs LATE ---")
print(f"Early (steps {early_steps[0]}-{early_steps[-1]}): "
      f"val_loss {early_losses[0]:.4f} -> {early_losses[-1]:.4f}, "
      f"rate = {early_rate:.6f} per step")
print(f"Late  (steps {late_steps[0]}-{late_steps[-1]}):  "
      f"val_loss {late_losses[0]:.4f} -> {late_losses[-1]:.4f}, "
      f"rate = {late_rate:.6f} per step")
ratio = abs(early_rate) / max(abs(late_rate), 1e-10)
print(f"Early/Late rate ratio:            {ratio:.2f}x")

# ============================================================
# 4. Monotonicity Analysis
# ============================================================

val_diffs = np.diff(val_losses)
n_decreases = np.sum(val_diffs < 0)
n_increases = np.sum(val_diffs > 0)
n_flat = np.sum(val_diffs == 0)

print(f"\n--- MONOTONICITY (val_loss) ---")
print(f"Step-to-step changes: {n_decreases} decreases, {n_increases} increases, {n_flat} flat")
print(f"Monotonic:            {'Yes' if n_increases == 0 else 'No (oscillatory)'}")
if n_increases > 0:
    inc_indices = np.where(val_diffs > 0)[0]
    print(f"Increases at steps:   {[f'{val_steps[i]}->{val_steps[i+1]}' for i in inc_indices]}")
    print(f"Max increase:         {np.max(val_diffs):.4f}")
print(f"Max decrease:         {np.min(val_diffs):.4f}")

# Train loss monotonicity
train_diffs = np.diff(train_loss)
t_dec = np.sum(train_diffs < 0)
t_inc = np.sum(train_diffs > 0)
print(f"\n--- MONOTONICITY (train_loss avg) ---")
print(f"Step-to-step changes: {t_dec} decreases, {t_inc} increases")
print(f"Highly oscillatory:   {'Yes' if t_inc > len(train_diffs)*0.3 else 'No'}")
print(f"Train loss std:       {np.std(train_loss):.4f}")
print(f"Train loss range:     [{np.min(train_loss):.4f}, {np.max(train_loss):.4f}]")

# ============================================================
# 5. Gradient SNR Approximation
# ============================================================

# proj_grad = (loss+ - loss-) / (2*epsilon) gives the directional derivative
# Its variance across steps approximates gradient noise
# SNR = |mean(proj_grad)| / std(proj_grad)

mean_pg = np.mean(proj_grad)
std_pg = np.std(proj_grad)
snr = abs(mean_pg) / std_pg if std_pg > 0 else float('inf')

print(f"\n--- GRADIENT SNR (from projected gradient) ---")
print(f"proj_grad values:     {proj_grad}")
print(f"Mean proj_grad:       {mean_pg:.6f}")
print(f"Std proj_grad:        {std_pg:.6f}")
print(f"SNR (|mean|/std):     {snr:.4f}")
print(f"Interpretation:       {'Very noisy (SNR<1)' if snr < 1 else 'Moderate noise' if snr < 3 else 'Clean signal'}")

# Also compute from loss_plus - loss_minus differences directly
loss_diff = loss_plus - loss_minus
print(f"\nloss+ - loss- diffs:  {loss_diff}")
print(f"Mean |diff|:          {np.mean(np.abs(loss_diff)):.6f}")
print(f"Std diff:             {np.std(loss_diff):.6f}")

# ============================================================
# 6. Curve Fitting: val_loss vs theoretical forms
# ============================================================

print("\n" + "=" * 70)
print("CURVE FITTING: val_loss(t) vs THEORETICAL CONVERGENCE FORMS")
print("=" * 70)

# Use val_loss as it's the more reliable metric
t = val_steps.astype(float)
y = val_losses.copy()

# We need t > 0 for 1/sqrt(t) and 1/t fits
# All our val steps start at 100, so this is fine

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

results = {}

# Model 1: O(1/sqrt(T)) -- loss(t) = a/sqrt(t) + c
def model_inv_sqrt(t, a, c):
    return a / np.sqrt(t) + c

try:
    popt, _ = curve_fit(model_inv_sqrt, t, y, p0=[0.1, 2.0], maxfev=10000)
    y_pred = model_inv_sqrt(t, *popt)
    r2 = r_squared(y, y_pred)
    results['O(1/sqrt(T))'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                                'formula': f'loss(t) = {popt[0]:.6f}/sqrt(t) + {popt[1]:.6f}'}
    print(f"\n1. O(1/sqrt(T)):  loss(t) = {popt[0]:.6f}/sqrt(t) + {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n1. O(1/sqrt(T)):  Fit failed: {e}")

# Model 2: O(1/T) -- loss(t) = a/t + c
def model_inv_t(t, a, c):
    return a / t + c

try:
    popt, _ = curve_fit(model_inv_t, t, y, p0=[1.0, 2.0], maxfev=10000)
    y_pred = model_inv_t(t, *popt)
    r2 = r_squared(y, y_pred)
    results['O(1/T)'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                          'formula': f'loss(t) = {popt[0]:.6f}/t + {popt[1]:.6f}'}
    print(f"\n2. O(1/T):        loss(t) = {popt[0]:.6f}/t + {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n2. O(1/T):        Fit failed: {e}")

# Model 3: Exponential -- loss(t) = a*exp(-b*t) + c
def model_exp(t, a, b, c):
    return a * np.exp(-b * t) + c

try:
    popt, _ = curve_fit(model_exp, t, y, p0=[0.05, 0.001, 2.05], maxfev=10000)
    y_pred = model_exp(t, *popt)
    r2 = r_squared(y, y_pred)
    results['Exponential'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                               'formula': f'loss(t) = {popt[0]:.6f}*exp(-{popt[1]:.6f}*t) + {popt[2]:.6f}'}
    print(f"\n3. Exponential:   loss(t) = {popt[0]:.6f}*exp(-{popt[1]:.6f}*t) + {popt[2]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n3. Exponential:   Fit failed: {e}")

# Model 4: Linear -- loss(t) = a*t + c
def model_linear(t, a, c):
    return a * t + c

try:
    popt, _ = curve_fit(model_linear, t, y, p0=[-1e-5, 2.07], maxfev=10000)
    y_pred = model_linear(t, *popt)
    r2 = r_squared(y, y_pred)
    results['Linear'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                          'formula': f'loss(t) = {popt[0]:.8f}*t + {popt[1]:.6f}'}
    print(f"\n4. Linear:        loss(t) = {popt[0]:.8f}*t + {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n4. Linear:        Fit failed: {e}")

# Model 5: Logarithmic -- loss(t) = a*log(t) + c (common in practice)
def model_log(t, a, c):
    return a * np.log(t) + c

try:
    popt, _ = curve_fit(model_log, t, y, p0=[-0.01, 2.1], maxfev=10000)
    y_pred = model_log(t, *popt)
    r2 = r_squared(y, y_pred)
    results['Logarithmic'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                               'formula': f'loss(t) = {popt[0]:.6f}*ln(t) + {popt[1]:.6f}'}
    print(f"\n5. Logarithmic:   loss(t) = {popt[0]:.6f}*ln(t) + {popt[1]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n5. Logarithmic:   Fit failed: {e}")

# Model 6: Power law -- loss(t) = a * t^(-b) + c
def model_power(t, a, b, c):
    return a * np.power(t, -b) + c

try:
    popt, _ = curve_fit(model_power, t, y, p0=[0.5, 0.5, 2.05], maxfev=10000,
                        bounds=([0, 0, 1.5], [100, 5, 3.0]))
    y_pred = model_power(t, *popt)
    r2 = r_squared(y, y_pred)
    results['Power law'] = {'params': popt, 'r2': r2, 'pred': y_pred,
                             'formula': f'loss(t) = {popt[0]:.6f}*t^(-{popt[1]:.6f}) + {popt[2]:.6f}'}
    print(f"\n6. Power law:     loss(t) = {popt[0]:.6f}*t^(-{popt[1]:.6f}) + {popt[2]:.6f}")
    print(f"   R² = {r2:.6f}")
except Exception as e:
    print(f"\n6. Power law:     Fit failed: {e}")

# ============================================================
# 7. Best Fit Ranking
# ============================================================

print("\n" + "-" * 50)
print("RANKING BY R²:")
ranked = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
for i, (name, res) in enumerate(ranked):
    print(f"  {i+1}. {name:20s}  R² = {res['r2']:.6f}  |  {res['formula']}")

best_name, best_res = ranked[0]
print(f"\nBest fit: {best_name} (R² = {best_res['r2']:.6f})")

# Residual analysis for best fit
residuals = y - best_res['pred']
print(f"\nResiduals for best fit ({best_name}):")
print(f"  Mean residual:    {np.mean(residuals):.6f}")
print(f"  Std residual:     {np.std(residuals):.6f}")
print(f"  Max |residual|:   {np.max(np.abs(residuals)):.6f}")

# ============================================================
# 8. Theoretical Comparison
# ============================================================

print("\n" + "=" * 70)
print("THEORETICAL COMPARISON: MeZO ZO-SGD")
print("=" * 70)

d = 1700800  # trainable params (~1.7M for LoRA-split)
d_full = 361800000  # full model params
T = 1000  # total steps
n = 1  # batch size (single sample MeZO)
lr_init = 1e-4
epsilon = 1e-3

print(f"\nProblem dimensions:")
print(f"  d (trainable params):     {d:,}")
print(f"  d_full (all params):      {d_full:,}")
print(f"  T (steps):                {T}")
print(f"  n (batch size):           {n}")
print(f"  lr (initial):             {lr_init}")
print(f"  epsilon:                  {epsilon}")

# Theoretical convergence rate for ZO-SGD (Malladi et al. Theorem 1)
# E[||grad f||^2] <= O(d / (n*T))
# With cosine LR schedule, effective T is reduced

# Empirical loss reduction rate
total_reduction = val_losses[0] - val_losses[-1]
per_step_reduction = total_reduction / (val_steps[-1] - val_steps[0])

print(f"\nEmpirical convergence:")
print(f"  Total val_loss reduction:    {total_reduction:.4f}")
print(f"  Per-step reduction:          {per_step_reduction:.8f}")
print(f"  Reduction per 100 steps:     {per_step_reduction * 100:.6f}")

# ZO vs SGD expected slowdown
# Worst case: ZO is d times slower than SGD
# MeZO paper: effective slowdown is r (Hessian rank), not d
# Empirical: loss_reduction_per_step ≈ lr^2/d for ZO
zo_slowdown_worst = d  # factor of d worse than SGD
zo_expected_lr_eff = lr_init**2 / d  # worst-case effective lr for ZO

print(f"\nTheoretical predictions (ZO-SGD):")
print(f"  Worst-case slowdown (d):     {zo_slowdown_worst:,}x vs SGD")
print(f"  Expected lr_eff (lr²/d):     {zo_expected_lr_eff:.2e}")
print(f"  Empirical per-step rate:     {abs(per_step_reduction):.2e}")

# Compare to SGD expected rate
sgd_expected_rate = lr_init  # SGD: loss reduction ~ lr * ||grad||
print(f"  SGD expected rate (lr):      {sgd_expected_rate:.2e}")

empirical_slowdown = sgd_expected_rate / max(abs(per_step_reduction), 1e-20)
print(f"  Empirical slowdown vs SGD:   {empirical_slowdown:.0f}x")

# Check if slowdown is closer to d or to sqrt(d) or to r
import math
print(f"\n  sqrt(d) = {math.sqrt(d):.0f}")
print(f"  d = {d:,}")
print(f"  Empirical slowdown = {empirical_slowdown:.0f}")

# Estimate effective Hessian rank from convergence
# If convergence ~ r/T, then r ≈ T * per_step_rate / lr
# From Theorem 1: E[||grad||^2] <= (f0-f*)/T + d*sigma^2/(nT) + d*eps^2*L
# The d factor appears as multiplier on variance

# MeZO gradient variance analysis
# proj_grad = (f(x+eps*z) - f(x-eps*z)) / (2*eps) estimates z^T grad
# True gradient magnitude ~ sqrt(d) * E[|proj_grad|]
est_grad_mag = np.mean(np.abs(proj_grad))  # E[|z^T grad|]
est_true_grad = math.sqrt(d) * est_grad_mag  # ||grad|| estimate
print(f"\n  Estimated |proj_grad|:       {est_grad_mag:.6f}")
print(f"  Estimated ||grad|| (sqrt(d)*|pg|): {est_true_grad:.2f}")

# MeZO Lemma 2: Var[ZO-grad] = (d+n-1)/n * Var[true grad]
# With n=1: Var[ZO-grad] = d * Var[true grad]
zo_var_factor = (d + n - 1) / n
print(f"\n  ZO variance amplification:   {zo_var_factor:.0f}x (=(d+n-1)/n)")
print(f"  This equals d since n=1:     {d:,}")

# ============================================================
# 9. Loss landscape curvature estimation
# ============================================================

print(f"\n--- LOSS LANDSCAPE CURVATURE ---")
# From consecutive loss differences, estimate local curvature
# loss+ and loss- are evaluations at w+eps*z and w-eps*z
# (loss+ + loss-)/2 ≈ f(w) + eps^2/2 * z^T H z (Hessian quadratic)
# (loss+ - loss-) / (2*eps) ≈ z^T grad

for i in range(len(train_steps)):
    f_avg = (loss_plus[i] + loss_minus[i]) / 2.0
    # The Hessian trace can be estimated: Tr(H) ≈ E[(loss+ + loss-)/2 - f(w)] * 2 / eps^2
    # But we don't have f(w) directly. We approximate f(w) ≈ val_loss at nearest step
    pass

# Instead, look at proj_grad sign changes and magnitude
pg_signs = np.sign(proj_grad)
sign_changes = np.sum(np.diff(pg_signs) != 0)
print(f"proj_grad sign changes:        {sign_changes} / {len(proj_grad)-1}")
print(f"Mean |proj_grad| early:        {np.mean(np.abs(proj_grad[:5])):.6f}")
print(f"Mean |proj_grad| late:         {np.mean(np.abs(proj_grad[5:])):.6f}")

# ============================================================
# 10. Effective dimension analysis
# ============================================================

print(f"\n--- EFFECTIVE DIMENSION ANALYSIS ---")
# If the loss follows L(t) = a/t^b + c, the exponent b tells us:
# b = 0.5 -> O(1/sqrt(T)): standard ZO non-convex rate (d-dependent)
# b = 1.0 -> O(1/T): strongly convex or variance-reduced ZO rate
# b between -> interpolating behavior

if 'Power law' in results:
    b_exp = results['Power law']['params'][1]
    print(f"Power law exponent b:          {b_exp:.4f}")
    if b_exp < 0.3:
        print(f"Interpretation: Very slow convergence (sub-sqrt(T))")
    elif b_exp < 0.7:
        print(f"Interpretation: Near O(1/sqrt(T)) - standard non-convex ZO rate")
    elif b_exp < 1.3:
        print(f"Interpretation: Near O(1/T) - faster than standard ZO (possible strong convexity)")
    else:
        print(f"Interpretation: Super-linear convergence")

# ============================================================
# 11. Summary Table
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
Configuration:
  Model:                SmolLM2-360M with LoRA-split (rank 8)
  Trainable params:     {d:,} (~1.7M)
  Full model params:    {d_full:,} (~361.8M)
  Steps:                {T}
  Initial LR:           {lr_init} (cosine decay)
  Epsilon:              {epsilon}
  Batch size:           {n}

Convergence Results:
  Baseline val_loss:    {baseline_val:.4f}
  Final val_loss:       {final_val:.4f}
  Best val_loss:        {best_val:.4f} (step {best_val_step})
  Total reduction:      {baseline_val - final_val:.4f} ({(baseline_val - final_val)/baseline_val*100:.2f}%)
  Early rate (per step): {abs(early_rate):.8f}
  Late rate (per step):  {abs(late_rate):.8f}
  Early/Late ratio:     {ratio:.2f}x
  Monotonic:            {'Yes' if n_increases == 0 else 'No'}
  Gradient SNR:         {snr:.4f}

Best Theoretical Fit:
  Form:                 {best_name}
  R²:                   {best_res['r2']:.6f}
  Formula:              {best_res['formula']}

ZO vs SGD Comparison:
  ZO variance factor:   {zo_var_factor:.0f}x
  Empirical slowdown:   {empirical_slowdown:.0f}x vs SGD
  Theoretical worst:    {d:,}x (factor of d)
""")

# ============================================================
# 12. Try to plot (optional)
# ============================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Val loss convergence
    ax = axes[0, 0]
    ax.plot(val_steps, val_losses, 'bo-', label='val_loss', markersize=6)
    ax.axhline(y=baseline_val, color='r', linestyle='--', alpha=0.7, label=f'baseline={baseline_val}')
    # Plot best fits
    t_smooth = np.linspace(val_steps[0], val_steps[-1], 200)
    for name, res in ranked[:3]:
        if name == 'O(1/sqrt(T))':
            ax.plot(t_smooth, model_inv_sqrt(t_smooth, *res['params'][:2]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
        elif name == 'O(1/T)':
            ax.plot(t_smooth, model_inv_t(t_smooth, *res['params'][:2]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
        elif name == 'Exponential':
            ax.plot(t_smooth, model_exp(t_smooth, *res['params'][:3]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
        elif name == 'Logarithmic':
            ax.plot(t_smooth, model_log(t_smooth, *res['params'][:2]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
        elif name == 'Power law':
            ax.plot(t_smooth, model_power(t_smooth, *res['params'][:3]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
        elif name == 'Linear':
            ax.plot(t_smooth, model_linear(t_smooth, *res['params'][:2]), '--', alpha=0.5, label=f'{name} (R²={res["r2"]:.4f})')
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('MeZO Convergence: Val Loss vs Theoretical Fits')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Train loss (loss+ and loss-)
    ax = axes[0, 1]
    ax.plot(train_steps, loss_plus, 'r^-', label='loss+', markersize=5, alpha=0.7)
    ax.plot(train_steps, loss_minus, 'bv-', label='loss-', markersize=5, alpha=0.7)
    ax.plot(train_steps, train_loss, 'g.-', label='avg', markersize=7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('MeZO Train Loss: loss+ and loss-')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Projected gradient
    ax = axes[1, 0]
    ax.bar(train_steps, proj_grad, width=50, alpha=0.7, color='steelblue')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=np.mean(proj_grad), color='r', linestyle='--', alpha=0.5, label=f'mean={np.mean(proj_grad):.3f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Projected Gradient')
    ax.set_title('MeZO Projected Gradient (z^T * grad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning rate schedule
    ax = axes[1, 1]
    ax.plot(train_steps, lr_values, 'k-o', markersize=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Cosine LR Schedule')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/mezo_convergence_analysis.png', dpi=150)
    print("Plot saved to /tmp/mezo_convergence_analysis.png")

except ImportError:
    print("matplotlib not available, skipping plot generation")
except Exception as e:
    print(f"Plotting failed: {e}")
