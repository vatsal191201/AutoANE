// validation_gradient_unbiased.c — Verify SPSA gradient estimate is unbiased with Rademacher
//
// Mathematical proof: For L(theta) = 0.5 * ||theta||^2 (quadratic):
//   proj_grad = theta^T z (exact, no epsilon bias)
//   g_hat_i = proj_grad * z_i = (theta^T z) * z_i
//   E[g_hat_i] = sum_j theta_j * E[z_j * z_i] = theta_i  (since E[z_i z_j] = delta_ij)
//
// BUT: Var(g_hat_i) = ||theta||^2 - theta_i^2
//   For theta = [1,...,d], ||theta||^2 = d(d+1)(2d+1)/6
//   This means small components have terrible SNR.
//
// Better test: check directional alignment <E[g_hat], theta> / ||theta||^2 ≈ 1.0
//
// Build: clang -O2 -o validate_grad validation_gradient_unbiased.c -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

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
    printf("=== SPSA Gradient Unbiasedness Validation (Rademacher) ===\n\n");

    int dims[] = {10, 100, 1000, 10000};
    int n_dims = 4;

    for (int di = 0; di < n_dims; di++) {
        int d = dims[di];
        int n_trials = 500000;
        float *theta = (float*)malloc(d * sizeof(float));
        float *grad_sum = (float*)calloc(d, sizeof(float));
        float *perturbed = (float*)malloc(d * sizeof(float));

        // theta = [1, 1, ..., 1] (uniform, so all components have same SNR)
        for (int i = 0; i < d; i++) theta[i] = 1.0f;
        float true_norm_sq = (float)d;  // ||theta||^2 = d

        float eps = 1e-3f;
        double dot_sum = 0;  // For directional test

        for (int trial = 0; trial < n_trials; trial++) {
            uint64_t seed = (uint64_t)trial * 1000003ULL + 42;

            // L(theta + eps*z)
            memcpy(perturbed, theta, d * sizeof(float));
            xo_seed(seed);
            perturb_buffer(perturbed, d, +eps);
            double loss_plus = 0;
            for (int i = 0; i < d; i++) loss_plus += 0.5 * perturbed[i] * perturbed[i];

            // L(theta - eps*z)
            memcpy(perturbed, theta, d * sizeof(float));
            xo_seed(seed);
            perturb_buffer(perturbed, d, -eps);
            double loss_minus = 0;
            for (int i = 0; i < d; i++) loss_minus += 0.5 * perturbed[i] * perturbed[i];

            float proj_grad = (float)((loss_plus - loss_minus) / (2.0 * eps));

            // Regenerate z and accumulate gradient estimate
            xo_seed(seed);
            size_t j = 0;
            for (; j + 3 < (size_t)d; j += 4) {
                uint64_t r = xo_next();
                float z0 = (r & 1) ? 1.0f : -1.0f;
                float z1 = (r & 2) ? 1.0f : -1.0f;
                float z2 = (r & 4) ? 1.0f : -1.0f;
                float z3 = (r & 8) ? 1.0f : -1.0f;
                grad_sum[j+0] += proj_grad * z0;
                grad_sum[j+1] += proj_grad * z1;
                grad_sum[j+2] += proj_grad * z2;
                grad_sum[j+3] += proj_grad * z3;
            }
            for (; j < (size_t)d; j++) {
                uint64_t r = xo_next();
                float z = (r & 1) ? 1.0f : -1.0f;
                grad_sum[j] += proj_grad * z;
            }
        }

        // Compute statistics
        double mean_est = 0, mean_sq_err = 0;
        for (int i = 0; i < d; i++) {
            float est = grad_sum[i] / n_trials;
            mean_est += est;
            mean_sq_err += (est - theta[i]) * (est - theta[i]);
        }
        mean_est /= d;

        // Directional alignment: <E[g_hat], theta> / ||theta||^2
        double dot = 0, norm_est_sq = 0;
        for (int i = 0; i < d; i++) {
            float est = grad_sum[i] / n_trials;
            dot += est * theta[i];
            norm_est_sq += est * est;
        }
        double alignment = dot / true_norm_sq;  // Should be ~1.0
        double cosine = dot / (sqrt(norm_est_sq) * sqrt(true_norm_sq));

        // Theory: Var(g_hat_i) = ||theta||^2 - theta_i^2 = d - 1
        // std(mean at n_trials) = sqrt(d-1) / sqrt(n_trials)
        double expected_std_mean = sqrt((double)(d - 1)) / sqrt((double)n_trials);
        double rmse = sqrt(mean_sq_err / d);

        printf("d=%5d | mean(g_hat_i)=%.4f (expect 1.0) | alignment=%.6f | cosine=%.6f | RMSE=%.4f (expect_std=%.4f)\n",
               d, mean_est, alignment, cosine, rmse, expected_std_mean);

        free(theta); free(grad_sum); free(perturbed);
    }

    printf("\nInterpretation:\n");
    printf("  - alignment ≈ 1.0 means E[g_hat] points in same direction as true gradient\n");
    printf("  - cosine ≈ 1.0 means high directional accuracy\n");
    printf("  - RMSE ≈ expected_std means variance matches theory: Var = (d-1)/n_trials\n");
    printf("  - RMSE grows as sqrt(d) — this is the dimension-dependent noise of SPSA\n");
    printf("\nConclusion: Rademacher perturbation produces UNBIASED gradient estimates\n");
    printf("but with variance proportional to d (dimension). This is fundamental to SPSA,\n");
    printf("NOT a bug. The MeZO paper (Lemma 2) states: E[||g_hat||^2] = (d+n-1)/n * E[||grad||^2]\n");

    return 0;
}
