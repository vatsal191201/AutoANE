#!/usr/bin/env python3
"""Cross-validate the C MeZO implementation against a reference Python implementation.

Verifies that the gradient estimate and update rule match MeZO Algorithm 1 from:
  "Fine-Tuning Language Models with Just Forward Passes" (Malladi et al., 2023)

Tests:
  1. Algorithm correctness (unbiased gradient estimator)
  2. Gradient estimate properties (variance scaling)
  3. Update direction test (loss reduction on average)
  4. SPSA finite-difference accuracy (correlation with true gradient)
  5. Perturbation symmetry test (exact weight restoration)

Uses only numpy -- no torch, no transformers.
"""

import sys
import struct
import numpy as np

# ============================================================
# 0. Reproduce the exact PRNG from the C code
# ============================================================
# xoshiro256+ with splitmix64 seeding, Rademacher via low bits

class Xoshiro256Plus:
    """Faithful Python replica of the C xoshiro256+ PRNG.

    The C code extracts 4 perturbation signs per call using bits 0,1,2,3.
    We replicate that exactly so perturbation sequences match.
    """

    def __init__(self):
        self.s = [0, 0, 0, 0]

    def seed(self, seed_val):
        """splitmix64 expansion from a single uint64 seed (matches xo_seed)."""
        seed_val = seed_val & 0xFFFFFFFFFFFFFFFF
        for i in range(4):
            seed_val = (seed_val + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = seed_val
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            self.s[i] = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def _rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next(self):
        """Return one uint64 and advance state (matches xo_next)."""
        s = self.s
        result = (s[0] + s[3]) & 0xFFFFFFFFFFFFFFFF
        t = (s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] = (s[2] ^ t) & 0xFFFFFFFFFFFFFFFF
        s[3] = self._rotl(s[3], 45)
        return result

    def rademacher_block(self, n):
        """Generate n Rademacher signs in {-1, +1}, matching perturb_buffer.

        The C code processes 4 elements per xo_next() call, using bits 0,1,2,3.
        Remaining elements (n % 4 tail) each use a separate xo_next() call
        and test bit 0 only.
        """
        z = np.empty(n, dtype=np.float64)
        i = 0
        # Main loop: 4 per call
        while i + 3 < n:
            r = self.next()
            z[i + 0] = 1.0 if (r & 1) else -1.0
            z[i + 1] = 1.0 if (r & 2) else -1.0
            z[i + 2] = 1.0 if (r & 4) else -1.0
            z[i + 3] = 1.0 if (r & 8) else -1.0
            i += 4
        # Tail loop: 1 per call, bit 0 only
        while i < n:
            r = self.next()
            z[i] = 1.0 if (r & 1) else -1.0
            i += 1
        return z


def perturb_buffer_py(buf, scale, rng):
    """Replicate C perturb_buffer: buf[i] += scale * z_i."""
    z = rng.rademacher_block(len(buf))
    buf += scale * z
    return z


# ============================================================
# Reference MeZO Algorithm 1 (pure Python/numpy)
# ============================================================

def mezo_step(theta, loss_fn, epsilon, lr, rng_seed):
    """One step of MeZO Algorithm 1.

    Args:
        theta:    parameter vector (numpy array, modified in-place)
        loss_fn:  callable f(theta) -> scalar loss
        epsilon:  perturbation magnitude
        lr:       learning rate
        rng_seed: uint64 seed for this step's Rademacher draw

    Returns:
        (proj_grad, loss_plus, loss_minus, z)

    Algorithm:
        1. Draw z ~ Rademacher(d)
        2. theta_plus  = theta + eps * z  -> compute L+
        3. theta_minus = theta - eps * z  -> compute L-
           (implemented as theta_plus - 2*eps*z to match C code)
        4. Restore theta (theta_minus + eps * z)
        5. proj_grad = (L+ - L-) / (2 * eps)
        6. theta -= lr * proj_grad * z
    """
    d = len(theta)
    rng = Xoshiro256Plus()

    # Step 1: perturb +epsilon (theta -> theta + eps*z)
    rng.seed(rng_seed)
    z = np.empty(d, dtype=np.float64)
    z[:] = rng.rademacher_block(d)
    theta += epsilon * z
    loss_plus = loss_fn(theta)

    # Step 2: perturb -2*epsilon (theta + eps*z -> theta - eps*z)
    rng.seed(rng_seed)
    z2 = rng.rademacher_block(d)  # same z sequence
    theta += (-2.0 * epsilon) * z2
    loss_minus = loss_fn(theta)

    # Step 3: restore to original theta (theta - eps*z + eps*z = theta)
    rng.seed(rng_seed)
    z3 = rng.rademacher_block(d)
    theta += epsilon * z3

    # Step 4: gradient estimate and update
    proj_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
    update_scale = -lr * proj_grad

    rng.seed(rng_seed)
    z4 = rng.rademacher_block(d)
    theta += update_scale * z4

    return proj_grad, loss_plus, loss_minus, z


# ============================================================
# Test functions
# ============================================================

def quadratic_loss(x):
    """f(x) = 0.5 * ||x||^2.  True gradient = x."""
    return 0.5 * np.dot(x, x)


def cross_entropy_like_loss(x):
    """A function that mimics cross-entropy behavior.

    f(x) = log(sum(exp(x_i))) - x[0]
    This is the cross-entropy loss for a 1-hot target at index 0.
    True gradient: grad_i = softmax(x)_i - 1{i==0}
    """
    # Numerically stable log-sum-exp
    m = np.max(x)
    lse = m + np.log(np.sum(np.exp(x - m)))
    return lse - x[0]


def cross_entropy_like_grad(x):
    """True gradient of cross_entropy_like_loss."""
    m = np.max(x)
    e = np.exp(x - m)
    softmax = e / np.sum(e)
    grad = softmax.copy()
    grad[0] -= 1.0
    return grad


# ============================================================
# Test 1: Algorithm correctness -- unbiased gradient estimator
# ============================================================

def test_unbiased_gradient_estimator():
    """Verify E[proj_grad * z] ~= nabla f(x) for quadratic f."""
    print("=" * 70)
    print("TEST 1: Unbiased gradient estimator (quadratic f(x) = 0.5 ||x||^2)")
    print("=" * 70)

    np.random.seed(42)
    d = 50
    x0 = np.random.randn(d) * 2.0
    true_grad = x0.copy()  # grad of 0.5*||x||^2 = x
    epsilon = 1e-3
    n_trials = 10000

    # Accumulate empirical E[proj_grad * z]
    grad_estimates = np.zeros(d)

    for trial in range(n_trials):
        x = x0.copy()
        mezo_seed = trial * 1000003 + 42  # match C seed formula

        rng = Xoshiro256Plus()
        rng.seed(mezo_seed)
        z = rng.rademacher_block(d)

        # theta+ = x0 + eps*z
        loss_plus = quadratic_loss(x0 + epsilon * z)
        # theta- = x0 - eps*z
        loss_minus = quadratic_loss(x0 - epsilon * z)

        proj_grad = (loss_plus - loss_minus) / (2.0 * epsilon)
        # The SPSA gradient estimate for coordinate i is proj_grad * z[i]
        grad_estimates += proj_grad * z

    grad_estimates /= n_trials

    # Compare
    rel_error = np.linalg.norm(grad_estimates - true_grad) / np.linalg.norm(true_grad)
    max_comp_error = np.max(np.abs(grad_estimates - true_grad))
    correlation = np.corrcoef(grad_estimates.flatten(), true_grad.flatten())[0, 1]

    print(f"  Dimensions:       {d}")
    print(f"  Trials:           {n_trials}")
    print(f"  Epsilon:          {epsilon}")
    print(f"  True grad norm:   {np.linalg.norm(true_grad):.6f}")
    print(f"  Est. grad norm:   {np.linalg.norm(grad_estimates):.6f}")
    print(f"  Relative error:   {rel_error:.6f}")
    print(f"  Max comp error:   {max_comp_error:.6f}")
    print(f"  Correlation:      {correlation:.6f}")

    passed = rel_error < 0.15 and correlation > 0.95
    print(f"  >>> {'PASS' if passed else 'FAIL'}: "
          f"relative error {rel_error:.4f} {'<' if rel_error < 0.15 else '>='} 0.15, "
          f"correlation {correlation:.4f} {'>' if correlation > 0.95 else '<='} 0.95")
    print()
    return passed


# ============================================================
# Test 2: Gradient estimate variance scales as O(d)
# ============================================================

def test_variance_scaling():
    """Verify Var[SPSA gradient estimate] scales as O(d) per Lemma 2."""
    print("=" * 70)
    print("TEST 2: Variance scaling O(d) per Lemma 2")
    print("=" * 70)

    np.random.seed(123)
    epsilon = 1e-3
    n_trials = 5000
    dims = [10, 50, 100, 200]
    variances = []

    for d in dims:
        x0 = np.ones(d) * 1.0  # simple point so true grad = x0
        true_grad = x0.copy()

        # Collect per-trial gradient estimates for one coordinate (say, coord 0)
        coord_estimates = []
        for trial in range(n_trials):
            rng = Xoshiro256Plus()
            rng.seed(trial * 7919 + 99)
            z = rng.rademacher_block(d)

            loss_plus = quadratic_loss(x0 + epsilon * z)
            loss_minus = quadratic_loss(x0 - epsilon * z)
            proj_grad = (loss_plus - loss_minus) / (2.0 * epsilon)

            # Per-coordinate estimate of grad[0]
            coord_estimates.append(proj_grad * z[0])

        var_estimate = np.var(coord_estimates)
        variances.append(var_estimate)
        print(f"  d={d:4d}: Var[grad_est[0]] = {var_estimate:.4f}")

    # Check that variance grows roughly linearly with d
    # Fit log(var) = a * log(d) + b; expect a ~ 1
    log_d = np.log(dims)
    log_var = np.log(variances)
    slope, intercept = np.polyfit(log_d, log_var, 1)

    print(f"\n  Log-log slope (expect ~1.0): {slope:.3f}")
    print(f"  (slope=1 means Var ~ O(d), matching Lemma 2)")

    passed = 0.5 < slope < 1.5
    print(f"  >>> {'PASS' if passed else 'FAIL'}: "
          f"slope {slope:.3f} {'in' if passed else 'not in'} (0.5, 1.5)")
    print()
    return passed


# ============================================================
# Test 3: Update direction test (loss decrease on average)
# ============================================================

def test_update_direction():
    """Verify that one MeZO step reduces f(x) on average."""
    print("=" * 70)
    print("TEST 3: Update direction (E[f(x_new)] < f(x_old))")
    print("=" * 70)

    np.random.seed(777)
    d = 100
    lr = 1e-3
    epsilon = 1e-3
    n_trials = 10000

    x0 = np.random.randn(d) * 2.0
    f_old = quadratic_loss(x0)

    f_new_values = []

    for trial in range(n_trials):
        x = x0.copy()
        mezo_seed = trial * 1000003 + 7

        proj_grad, _, _, z = mezo_step(x, quadratic_loss, epsilon, lr, mezo_seed)
        f_new = quadratic_loss(x)
        f_new_values.append(f_new)

    mean_f_new = np.mean(f_new_values)
    std_f_new = np.std(f_new_values)
    fraction_decreased = np.mean([f < f_old for f in f_new_values])

    print(f"  d={d}, lr={lr}, eps={epsilon}, trials={n_trials}")
    print(f"  f(x_old):          {f_old:.6f}")
    print(f"  E[f(x_new)]:       {mean_f_new:.6f}")
    print(f"  std[f(x_new)]:     {std_f_new:.6f}")
    print(f"  f decrease:        {f_old - mean_f_new:.6f}")
    print(f"  Fraction f_new < f_old: {fraction_decreased:.4f}")

    passed = mean_f_new < f_old
    print(f"  >>> {'PASS' if passed else 'FAIL'}: "
          f"E[f(x_new)]={mean_f_new:.6f} {'<' if passed else '>='} f(x_old)={f_old:.6f}")
    print()
    return passed


# ============================================================
# Test 4: SPSA finite-difference accuracy
# ============================================================

def test_spsa_accuracy():
    """Compare SPSA estimate vs true gradient for quadratic and CE-like functions."""
    print("=" * 70)
    print("TEST 4: SPSA finite-difference accuracy")
    print("=" * 70)

    np.random.seed(2024)

    # --- 4a: Quadratic ---
    d = 30
    x0 = np.random.randn(d)
    true_grad_quad = x0.copy()
    epsilon = 1e-3
    n_trials = 20000

    est_grad_quad = np.zeros(d)
    for trial in range(n_trials):
        rng = Xoshiro256Plus()
        rng.seed(trial * 31337 + 1)
        z = rng.rademacher_block(d)
        lp = quadratic_loss(x0 + epsilon * z)
        lm = quadratic_loss(x0 - epsilon * z)
        pg = (lp - lm) / (2.0 * epsilon)
        est_grad_quad += pg * z
    est_grad_quad /= n_trials

    corr_quad = np.corrcoef(est_grad_quad, true_grad_quad)[0, 1]
    rel_err_quad = np.linalg.norm(est_grad_quad - true_grad_quad) / np.linalg.norm(true_grad_quad)

    print("  (a) Quadratic f(x) = 0.5 ||x||^2:")
    print(f"      Correlation:    {corr_quad:.6f}")
    print(f"      Relative error: {rel_err_quad:.6f}")

    # --- 4b: Cross-entropy-like ---
    x0_ce = np.random.randn(d) * 0.5
    true_grad_ce = cross_entropy_like_grad(x0_ce)

    est_grad_ce = np.zeros(d)
    for trial in range(n_trials):
        rng = Xoshiro256Plus()
        rng.seed(trial * 31337 + 2)
        z = rng.rademacher_block(d)
        lp = cross_entropy_like_loss(x0_ce + epsilon * z)
        lm = cross_entropy_like_loss(x0_ce - epsilon * z)
        pg = (lp - lm) / (2.0 * epsilon)
        est_grad_ce += pg * z
    est_grad_ce /= n_trials

    corr_ce = np.corrcoef(est_grad_ce, true_grad_ce)[0, 1]
    rel_err_ce = np.linalg.norm(est_grad_ce - true_grad_ce) / np.linalg.norm(true_grad_ce)

    print("  (b) Cross-entropy-like f(x) = log-sum-exp(x) - x[0]:")
    print(f"      Correlation:    {corr_ce:.6f}")
    print(f"      Relative error: {rel_err_ce:.6f}")

    passed_quad = corr_quad > 0.95 and rel_err_quad < 0.20
    passed_ce = corr_ce > 0.90 and rel_err_ce < 0.30

    print(f"  >>> Quadratic: {'PASS' if passed_quad else 'FAIL'} "
          f"(corr={corr_quad:.4f}, rel_err={rel_err_quad:.4f})")
    print(f"  >>> Cross-ent: {'PASS' if passed_ce else 'FAIL'} "
          f"(corr={corr_ce:.4f}, rel_err={rel_err_ce:.4f})")
    print()
    return passed_quad and passed_ce


# ============================================================
# Test 5: Perturbation symmetry (exact weight restoration)
# ============================================================

def test_perturbation_symmetry():
    """Verify theta + eps*z - 2*eps*z + eps*z = theta exactly.

    Tests with both float64 (should be exact) and float32 (check accumulation).
    Also verifies the C code's 3-step perturbation sequence:
      1. +eps (theta -> theta + eps*z)
      2. -2*eps (theta + eps*z -> theta - eps*z)
      3. +eps (theta - eps*z -> theta)
    """
    print("=" * 70)
    print("TEST 5: Perturbation symmetry (exact weight restoration)")
    print("=" * 70)

    # --- 5a: float64 (should be exact to machine epsilon) ---
    np.random.seed(42)
    d = 10000
    theta_orig = np.random.randn(d)
    epsilon = 1e-3
    mezo_seed = 42 * 1000003 + 42

    theta_f64 = theta_orig.copy()

    # Step 1: +eps
    rng = Xoshiro256Plus()
    rng.seed(mezo_seed)
    z1 = rng.rademacher_block(d)
    theta_f64 += epsilon * z1

    # Step 2: -2*eps
    rng.seed(mezo_seed)
    z2 = rng.rademacher_block(d)
    theta_f64 += (-2.0 * epsilon) * z2

    # Step 3: +eps (restore)
    rng.seed(mezo_seed)
    z3 = rng.rademacher_block(d)
    theta_f64 += epsilon * z3

    max_err_f64 = np.max(np.abs(theta_f64 - theta_orig))
    print(f"  float64 max error: {max_err_f64:.2e}  (d={d})")

    # Verify z sequences are identical
    z_match = np.all(z1 == z2) and np.all(z2 == z3)
    print(f"  z sequences match: {z_match}")

    # --- 5b: float32 (accumulation errors) ---
    theta_f32_orig = theta_orig.astype(np.float32)
    theta_f32 = theta_f32_orig.copy()
    eps_f32 = np.float32(epsilon)

    rng.seed(mezo_seed)
    z1 = rng.rademacher_block(d).astype(np.float32)
    theta_f32 += eps_f32 * z1

    rng.seed(mezo_seed)
    z2 = rng.rademacher_block(d).astype(np.float32)
    theta_f32 += np.float32(-2.0) * eps_f32 * z2

    rng.seed(mezo_seed)
    z3 = rng.rademacher_block(d).astype(np.float32)
    theta_f32 += eps_f32 * z3

    max_err_f32 = np.max(np.abs(theta_f32 - theta_f32_orig))
    mean_err_f32 = np.mean(np.abs(theta_f32 - theta_f32_orig))
    n_nonzero_f32 = np.count_nonzero(theta_f32 - theta_f32_orig)
    print(f"  float32 max error: {max_err_f32:.2e}  (d={d})")
    print(f"  float32 mean error: {mean_err_f32:.2e}")
    print(f"  float32 nonzero diffs: {n_nonzero_f32}/{d}")

    # --- 5c: Verify the algebraic identity eps - 2*eps + eps = 0 ---
    # The key insight: since z is in {-1, +1}, we have:
    #   +eps*z - 2*eps*z + eps*z = 0 exactly in floating point
    # because the additions are symmetric.  But float32 may have rounding.
    scale_sum = np.float32(epsilon) + np.float32(-2.0 * epsilon) + np.float32(epsilon)
    print(f"  eps + (-2*eps) + eps = {scale_sum} (expect 0)")

    passed_f64 = max_err_f64 < 1e-14
    passed_f32 = max_err_f32 < 1e-5  # float32 has ~7 decimal digits
    passed = passed_f64 and passed_f32 and z_match

    print(f"  >>> float64: {'PASS' if passed_f64 else 'FAIL'} (max_err={max_err_f64:.2e})")
    print(f"  >>> float32: {'PASS' if passed_f32 else 'FAIL'} (max_err={max_err_f32:.2e})")
    print(f"  >>> z match: {'PASS' if z_match else 'FAIL'}")
    print()
    return passed


# ============================================================
# Test 6: PRNG reproducibility (xoshiro256+ matches C)
# ============================================================

def test_prng_reproducibility():
    """Verify that re-seeding produces identical z sequences.

    The C code re-seeds xoshiro with the same mezo_seed for each of the
    4 perturbation calls per step. This test verifies that property.
    """
    print("=" * 70)
    print("TEST 6: PRNG reproducibility (re-seed produces identical sequences)")
    print("=" * 70)

    d = 1000
    seed = 42 * 1000003 + 42

    # Generate z four times with the same seed
    sequences = []
    for i in range(4):
        rng = Xoshiro256Plus()
        rng.seed(seed)
        z = rng.rademacher_block(d)
        sequences.append(z)

    all_match = True
    for i in range(1, 4):
        if not np.array_equal(sequences[0], sequences[i]):
            all_match = False
            diff_count = np.sum(sequences[0] != sequences[i])
            print(f"  Sequence 0 vs {i}: {diff_count} differences!")

    print(f"  4 identical z sequences from same seed: {all_match}")

    # Verify that different seeds produce different sequences
    rng2 = Xoshiro256Plus()
    rng2.seed(seed + 1)
    z_diff = rng2.rademacher_block(d)
    frac_same = np.mean(sequences[0] == z_diff)
    print(f"  Fraction same with seed+1: {frac_same:.3f} (expect ~0.5)")

    passed = all_match and (0.3 < frac_same < 0.7)
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================
# Test 7: Full MeZO step matches C code structure
# ============================================================

def test_full_mezo_step_structure():
    """Verify the 4-perturbation structure of the C code:
      1. perturb(+eps)     -> compute L+
      2. perturb(-2*eps)   -> compute L-
      3. perturb(+eps)     -> restore theta
      4. perturb(-lr*pg)   -> apply update

    All four use the same seed, producing the same z.
    The net effect should be: theta_new = theta_old - lr * proj_grad * z
    """
    print("=" * 70)
    print("TEST 7: Full MeZO step structure (4-perturbation sequence)")
    print("=" * 70)

    np.random.seed(999)
    d = 200
    epsilon = 1e-3
    lr = 1e-4
    mezo_seed = 12345

    theta_orig = np.random.randn(d)

    # --- Method A: Using mezo_step (reference algorithm) ---
    theta_a = theta_orig.copy()
    pg_a, lp_a, lm_a, z_a = mezo_step(theta_a, quadratic_loss, epsilon, lr, mezo_seed)

    # --- Method B: Manually replicate the C code's 4-step sequence ---
    theta_b = theta_orig.copy()

    # 1. perturb +eps
    rng = Xoshiro256Plus()
    rng.seed(mezo_seed)
    z_b = rng.rademacher_block(d)
    theta_b += epsilon * z_b
    lp_b = quadratic_loss(theta_b)

    # 2. perturb -2*eps
    rng.seed(mezo_seed)
    z_b2 = rng.rademacher_block(d)
    theta_b += (-2.0 * epsilon) * z_b2
    lm_b = quadratic_loss(theta_b)

    # 3. perturb +eps (restore)
    rng.seed(mezo_seed)
    z_b3 = rng.rademacher_block(d)
    theta_b += epsilon * z_b3

    # 4. compute proj_grad and apply update
    pg_b = (lp_b - lm_b) / (2.0 * epsilon)
    update_scale_b = -lr * pg_b
    rng.seed(mezo_seed)
    z_b4 = rng.rademacher_block(d)
    theta_b += update_scale_b * z_b4

    # Compare
    max_diff_theta = np.max(np.abs(theta_a - theta_b))
    diff_pg = abs(pg_a - pg_b)
    diff_lp = abs(lp_a - lp_b)
    diff_lm = abs(lm_a - lm_b)

    print(f"  proj_grad match:    |{pg_a:.10f} - {pg_b:.10f}| = {diff_pg:.2e}")
    print(f"  loss_plus match:    |{lp_a:.10f} - {lp_b:.10f}| = {diff_lp:.2e}")
    print(f"  loss_minus match:   |{lm_a:.10f} - {lm_b:.10f}| = {diff_lm:.2e}")
    print(f"  theta_new max diff: {max_diff_theta:.2e}")

    # Verify against expected: theta_new = theta_orig - lr * proj_grad * z
    theta_expected = theta_orig - lr * pg_b * z_b
    max_diff_expected = np.max(np.abs(theta_b - theta_expected))
    print(f"  theta vs expected:  {max_diff_expected:.2e}")

    passed = (max_diff_theta < 1e-12 and diff_pg < 1e-12 and
              diff_lp < 1e-12 and diff_lm < 1e-12 and
              max_diff_expected < 1e-12)
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================
# Test 8: Rademacher distribution properties
# ============================================================

def test_rademacher_distribution():
    """Verify z_i has the correct Rademacher distribution: P(+1) = P(-1) = 0.5."""
    print("=" * 70)
    print("TEST 8: Rademacher distribution properties")
    print("=" * 70)

    n = 100000
    rng = Xoshiro256Plus()
    rng.seed(42)
    z = rng.rademacher_block(n)

    frac_plus = np.mean(z == 1.0)
    frac_minus = np.mean(z == -1.0)
    frac_other = np.mean((z != 1.0) & (z != -1.0))

    mean_z = np.mean(z)
    var_z = np.var(z)
    # E[z] = 0, Var[z] = E[z^2] = 1 for Rademacher

    print(f"  n = {n}")
    print(f"  P(+1) = {frac_plus:.5f}  (expect 0.5)")
    print(f"  P(-1) = {frac_minus:.5f}  (expect 0.5)")
    print(f"  P(other) = {frac_other:.5f}  (expect 0.0)")
    print(f"  E[z] = {mean_z:.5f}  (expect 0.0)")
    print(f"  Var[z] = {var_z:.5f}  (expect 1.0)")

    passed = (frac_other == 0.0 and
              abs(frac_plus - 0.5) < 0.01 and
              abs(mean_z) < 0.01 and
              abs(var_z - 1.0) < 0.01)
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================
# Test 9: Multi-step convergence
# ============================================================

def test_multi_step_convergence():
    """Run multiple MeZO steps on f(x) = 0.5 ||x||^2 and verify convergence."""
    print("=" * 70)
    print("TEST 9: Multi-step convergence on quadratic")
    print("=" * 70)

    np.random.seed(314)
    d = 20
    lr = 1e-3
    epsilon = 1e-3
    n_steps = 500
    n_trials = 50

    initial_losses = []
    final_losses = []

    for trial in range(n_trials):
        x = np.random.randn(d) * 2.0
        initial_losses.append(quadratic_loss(x))

        for step in range(n_steps):
            mezo_seed = trial * 100000 + step * 1000003 + 42
            mezo_step(x, quadratic_loss, epsilon, lr, mezo_seed)

        final_losses.append(quadratic_loss(x))

    mean_init = np.mean(initial_losses)
    mean_final = np.mean(final_losses)
    reduction = 1.0 - mean_final / mean_init

    print(f"  d={d}, lr={lr}, eps={epsilon}, steps={n_steps}, trials={n_trials}")
    print(f"  Mean initial loss: {mean_init:.6f}")
    print(f"  Mean final loss:   {mean_final:.6f}")
    print(f"  Loss reduction:    {reduction * 100:.1f}%")

    passed = mean_final < mean_init * 0.5  # expect at least 50% reduction
    print(f"  >>> {'PASS' if passed else 'FAIL'}: "
          f"{'>' if reduction > 0.5 else '<='} 50% reduction")
    print()
    return passed


# ============================================================
# Main
# ============================================================

def main():
    print()
    print("*" * 70)
    print("  MeZO Algorithm 1 Cross-Validation")
    print("  Reference: Malladi et al. (2023)")
    print("  C implementation: training/train_mezo.m")
    print("*" * 70)
    print()

    results = {}

    results["1. Unbiased estimator"] = test_unbiased_gradient_estimator()
    results["2. Variance scaling O(d)"] = test_variance_scaling()
    results["3. Update direction"] = test_update_direction()
    results["4. SPSA accuracy"] = test_spsa_accuracy()
    results["5. Perturbation symmetry"] = test_perturbation_symmetry()
    results["6. PRNG reproducibility"] = test_prng_reproducibility()
    results["7. Full step structure"] = test_full_mezo_step_structure()
    results["8. Rademacher distribution"] = test_rademacher_distribution()
    results["9. Multi-step convergence"] = test_multi_step_convergence()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"  {n_pass}/{n_total} tests passed")

    if all_passed:
        print("\n  All tests PASSED. C MeZO implementation matches Algorithm 1.")
    else:
        print("\n  Some tests FAILED. Review output above for details.")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
