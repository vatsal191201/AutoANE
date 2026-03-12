// validation_perturbation_cancel.c — Verify perturbation cancel property
// Tests that perturb(+eps) followed by perturb(-eps) with same seed returns to original weights
// This is the mathematical foundation of MeZO correctness.
//
// Build: clang -O2 -o validate_cancel validation_perturbation_cancel.c -lm
// Expected: max_error = 0.0 (exact cancellation due to deterministic PRNG)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// ===== xoshiro256+ (copy from train_mezo.m for standalone test) =====
static uint64_t xo_s[4];
static inline uint64_t xo_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t xo_next(void) {
    uint64_t result = xo_s[0] + xo_s[3];
    uint64_t t = xo_s[1] << 17;
    xo_s[2] ^= xo_s[0]; xo_s[3] ^= xo_s[1]; xo_s[1] ^= xo_s[2]; xo_s[0] ^= xo_s[3];
    xo_s[2] ^= t; xo_s[3] = xo_rotl(xo_s[3], 45);
    return result;
}
static void xo_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        xo_s[i] = z ^ (z >> 31);
    }
}

static void perturb_buffer(float *buf, size_t n, float scale) {
    float neg_scale = -scale;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        uint64_t r = xo_next();
        buf[i+0] += (r & 1) ? scale : neg_scale;
        buf[i+1] += (r & 2) ? scale : neg_scale;
        buf[i+2] += (r & 4) ? scale : neg_scale;
        buf[i+3] += (r & 8) ? scale : neg_scale;
    }
    for (; i < n; i++) {
        uint64_t r = xo_next();
        buf[i] += (r & 1) ? scale : neg_scale;
    }
}

int main(void) {
    printf("=== Perturbation Cancel Validation ===\n\n");

    // Test 1: Basic cancel (perturb +eps then -eps)
    {
        size_t n = 1000000;  // 1M floats
        float *buf = (float*)malloc(n * sizeof(float));
        float *original = (float*)malloc(n * sizeof(float));

        // Initialize with known values
        for (size_t i = 0; i < n; i++) buf[i] = original[i] = (float)i * 0.001f;

        uint64_t seed = 12345;
        float eps = 1e-3f;

        // Perturb +eps
        xo_seed(seed);
        perturb_buffer(buf, n, +eps);

        // Perturb -eps (same seed)
        xo_seed(seed);
        perturb_buffer(buf, n, -eps);

        // Check: should be exactly original
        double max_err = 0;
        for (size_t i = 0; i < n; i++) {
            double err = fabs((double)buf[i] - (double)original[i]);
            if (err > max_err) max_err = err;
        }
        printf("Test 1 (cancel +eps/-eps, n=%zu): max_error = %.15e\n", n, max_err);
        printf("  Result: %s\n\n", max_err == 0.0 ? "EXACT CANCEL (PASS)" : "FAIL - nonzero error");

        free(buf); free(original);
    }

    // Test 2: MeZO sequence (+eps, -2eps, +eps = original)
    {
        size_t n = 500000;
        float *buf = (float*)malloc(n * sizeof(float));
        float *original = (float*)malloc(n * sizeof(float));

        for (size_t i = 0; i < n; i++) buf[i] = original[i] = sinf(i * 0.01f);

        uint64_t seed = 99999;
        float eps = 1e-3f;

        // MeZO sequence: +eps, -2eps, +eps
        xo_seed(seed); perturb_buffer(buf, n, +eps);
        xo_seed(seed); perturb_buffer(buf, n, -2.0f * eps);
        xo_seed(seed); perturb_buffer(buf, n, +eps);

        double max_err = 0;
        for (size_t i = 0; i < n; i++) {
            double err = fabs((double)buf[i] - (double)original[i]);
            if (err > max_err) max_err = err;
        }
        printf("Test 2 (MeZO sequence +eps/-2eps/+eps, n=%zu): max_error = %.15e\n", n, max_err);
        printf("  Result: %s\n\n", max_err == 0.0 ? "EXACT CANCEL (PASS)" : "FAIL - nonzero error");

        free(buf); free(original);
    }

    // Test 3: Rademacher distribution check (z_i in {-1,+1} with ~50/50 split)
    {
        size_t n = 10000000;  // 10M samples for statistical power
        float *buf = (float*)calloc(n, sizeof(float));

        uint64_t seed = 42;
        xo_seed(seed);
        perturb_buffer(buf, n, 1.0f);  // scale=1.0 so buf[i] = z_i

        long count_plus = 0, count_minus = 0;
        double sum = 0, sum_sq = 0;
        for (size_t i = 0; i < n; i++) {
            if (buf[i] > 0) count_plus++;
            else count_minus++;
            sum += buf[i];
            sum_sq += buf[i] * buf[i];
        }

        double mean = sum / n;
        double var = sum_sq / n - mean * mean;
        double ratio = (double)count_plus / n;

        printf("Test 3 (Rademacher distribution, n=%zu):\n", n);
        printf("  +1 count: %ld (%.4f%%)\n", count_plus, 100.0 * ratio);
        printf("  -1 count: %ld (%.4f%%)\n", count_minus, 100.0 * (1.0 - ratio));
        printf("  E[z] = %.6f (expected: 0.0)\n", mean);
        printf("  E[z^2] = %.6f (expected: 1.0)\n", sum_sq / n);
        printf("  Var[z] = %.6f (expected: 1.0)\n", var);
        printf("  Result: %s\n\n",
               (fabs(mean) < 0.001 && fabs(var - 1.0) < 0.001 && fabs(ratio - 0.5) < 0.001)
               ? "PASS" : "FAIL");

        free(buf);
    }

    // Test 4: Verify gradient estimate is unbiased for simple quadratic
    // L(theta) = 0.5 * ||theta||^2, so grad L = theta
    // SPSA estimate: g_hat = [(L(theta+eps*z) - L(theta-eps*z)) / (2*eps)] * z
    // E[g_hat] should equal theta
    {
        int d = 100;  // dimension
        int n_trials = 100000;
        float *theta = (float*)malloc(d * sizeof(float));
        float *grad_sum = (float*)calloc(d, sizeof(float));
        float *perturbed = (float*)malloc(d * sizeof(float));

        // Set theta = [1, 2, 3, ..., d]
        for (int i = 0; i < d; i++) theta[i] = (float)(i + 1);

        float eps = 1e-3f;

        for (int trial = 0; trial < n_trials; trial++) {
            uint64_t seed = (uint64_t)trial * 1000003ULL + 42;

            // Compute L(theta + eps*z)
            memcpy(perturbed, theta, d * sizeof(float));
            xo_seed(seed);
            perturb_buffer(perturbed, d, +eps);
            float loss_plus = 0;
            for (int i = 0; i < d; i++) loss_plus += 0.5f * perturbed[i] * perturbed[i];

            // Compute L(theta - eps*z)
            memcpy(perturbed, theta, d * sizeof(float));
            xo_seed(seed);
            perturb_buffer(perturbed, d, -eps);
            float loss_minus = 0;
            for (int i = 0; i < d; i++) loss_minus += 0.5f * perturbed[i] * perturbed[i];

            // Projected gradient
            float proj_grad = (loss_plus - loss_minus) / (2.0f * eps);

            // Accumulate gradient estimate: g_hat_i = proj_grad * z_i
            // Regenerate z_i with same seed
            xo_seed(seed);
            float neg_one = -1.0f;
            size_t j = 0;
            for (; j + 3 < (size_t)d; j += 4) {
                uint64_t r = xo_next();
                float z0 = (r & 1) ? 1.0f : neg_one;
                float z1 = (r & 2) ? 1.0f : neg_one;
                float z2 = (r & 4) ? 1.0f : neg_one;
                float z3 = (r & 8) ? 1.0f : neg_one;
                grad_sum[j+0] += proj_grad * z0;
                grad_sum[j+1] += proj_grad * z1;
                grad_sum[j+2] += proj_grad * z2;
                grad_sum[j+3] += proj_grad * z3;
            }
            for (; j < (size_t)d; j++) {
                uint64_t r = xo_next();
                float z = (r & 1) ? 1.0f : neg_one;
                grad_sum[j] += proj_grad * z;
            }
        }

        // Average and compare to true gradient (theta)
        double max_rel_err = 0;
        for (int i = 0; i < d; i++) {
            float estimated = grad_sum[i] / n_trials;
            float true_grad = theta[i];
            double rel_err = fabs(estimated - true_grad) / fabs(true_grad);
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }

        printf("Test 4 (Gradient unbiasedness, d=%d, n_trials=%d):\n", d, n_trials);
        printf("  Max relative error (E[g_hat] vs true grad): %.4f%%\n", max_rel_err * 100);
        printf("  Result: %s\n\n",
               max_rel_err < 0.05 ? "PASS (unbiased within 5%%)" : "FAIL");

        free(theta); free(grad_sum); free(perturbed);
    }

    // Test 5: Verify 4-bit extraction gives independent bits
    {
        size_t n_calls = 1000000;
        long bit_counts[4] = {0, 0, 0, 0};
        long pair_counts[6] = {0, 0, 0, 0, 0, 0};  // 01, 02, 03, 12, 13, 23

        xo_seed(42);
        for (size_t c = 0; c < n_calls; c++) {
            uint64_t r = xo_next();
            int b0 = (r & 1) ? 1 : 0;
            int b1 = (r & 2) ? 1 : 0;
            int b2 = (r & 4) ? 1 : 0;
            int b3 = (r & 8) ? 1 : 0;
            bit_counts[0] += b0;
            bit_counts[1] += b1;
            bit_counts[2] += b2;
            bit_counts[3] += b3;
            pair_counts[0] += b0 & b1;
            pair_counts[1] += b0 & b2;
            pair_counts[2] += b0 & b3;
            pair_counts[3] += b1 & b2;
            pair_counts[4] += b1 & b3;
            pair_counts[5] += b2 & b3;
        }

        printf("Test 5 (Bit independence, n=%zu):\n", n_calls);
        printf("  Bit 0 frequency: %.4f (expected: 0.5)\n", (double)bit_counts[0] / n_calls);
        printf("  Bit 1 frequency: %.4f (expected: 0.5)\n", (double)bit_counts[1] / n_calls);
        printf("  Bit 2 frequency: %.4f (expected: 0.5)\n", (double)bit_counts[2] / n_calls);
        printf("  Bit 3 frequency: %.4f (expected: 0.5)\n", (double)bit_counts[3] / n_calls);

        int pass = 1;
        for (int i = 0; i < 4; i++)
            if (fabs((double)bit_counts[i] / n_calls - 0.5) > 0.002) pass = 0;

        // For independent bits, P(b_i & b_j) = P(b_i) * P(b_j) = 0.25
        printf("  Pair 01 joint: %.4f (expected: 0.25)\n", (double)pair_counts[0] / n_calls);
        printf("  Pair 02 joint: %.4f (expected: 0.25)\n", (double)pair_counts[1] / n_calls);
        printf("  Pair 03 joint: %.4f (expected: 0.25)\n", (double)pair_counts[2] / n_calls);
        for (int i = 0; i < 6; i++)
            if (fabs((double)pair_counts[i] / n_calls - 0.25) > 0.002) pass = 0;

        printf("  Result: %s\n", pass ? "PASS (bits are independent)" : "FAIL");
    }

    return 0;
}
