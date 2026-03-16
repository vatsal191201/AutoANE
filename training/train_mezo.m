// train_mezo.m — Zeroth-Order (MeZO/SPSA) training on Apple Neural Engine
// Forward-pass only: no backward kernels, no gradients, no Adam state.
// Memory = inference memory (seed trick eliminates perturbation storage).
//
// Build: make mezo MODEL=autoresearch (or smollm2_135m, smollm2_360m)
// Usage: ./train_mezo --scratch --data data.bin --cpu-only --steps 1000
#include "mil_dynamic.h"
#include "cpu_ops.h"
#include "backprop_lora.h"
#include <math.h>

// Dynamic kernel set per layer (forward-only subset for MeZO)
typedef struct {
    // Fused forward kernels (legacy — fp16 RoPE+attention+SiLU+residual)
    Kern *sdpaFwd;     // QKV matmul + RoPE + GQA tile + SDPA (no Wo)
    Kern *woFwd;       // attn_out @ Wo^T → o_out (Q_DIM → DIM)
    Kern *ffnFused;    // W1,W3 + SiLU + W2 + residual (fused)
    // Unfused forward kernels (matmul-only — RoPE+attention+SiLU+residual on CPU fp32)
    Kern *wqFwd;       // xnorm @ Wq → Q (DIM → Q_DIM)
    Kern *wkvFwd;      // xnorm @ Wk/Wv → K or V (DIM → KV_DIM) — shared kernel, separate surfaces
    Kern *w13Fwd;      // x2norm @ W1/W3 → h1 or h3 (DIM → HIDDEN) — shared kernel, separate surfaces
    Kern *w2Fwd;       // silu_out @ W2 → ffn_out (HIDDEN → DIM)
    // Conv1x1 hybrid kernels (weights baked as BLOBFILE, activation-only IOSurface)
    // Used for Wq, Wo, W1, W2, W3 when --conv-hybrid + --lora-split
    // Per-layer: each layer gets its own compiled kernel with baked weights
    Kern *wqConv[NLAYERS];   // DIM → Q_DIM conv1x1 (per layer, baked Wq^T)
    Kern *woConv[NLAYERS];   // Q_DIM → DIM conv1x1 (per layer, baked Wo^T)
    Kern *w1Conv[NLAYERS];   // DIM → HIDDEN conv1x1 (per layer, baked W1^T)
    Kern *w2Conv[NLAYERS];   // HIDDEN → DIM conv1x1 (per layer, baked W2^T)
    Kern *w3Conv[NLAYERS];   // DIM → HIDDEN conv1x1 (per layer, baked W3^T)
    // Conv1x1 FUSED kernels (--conv-fused mode)
    // QKV fused: 3 conv1x1 ops in one kernel (Wq+Wk+Wv baked, shared input)
    Kern *qkvConv[NLAYERS];  // DIM → Q_DIM+2*KV_DIM fused conv1x1 (per layer)
    // FFN fused: conv(W1)+conv(W3)+SiLU+conv(W2)+residual in one kernel
    Kern *ffnConv[NLAYERS];  // DIM×2SEQ → DIM×SEQ fused conv1x1 (per layer)
    // Backward kernels (unused in MeZO but kept for struct compatibility)
    Kern *ffnBwdW2t;
    Kern *ffnBwdW13t;
    Kern *wotBwd;
    Kern *sdpaBwd1;
    Kern *sdpaBwd2;
    Kern *qBwd;
    Kern *kvBwd;
} DynLayerKernels;

// Transpose W[rows,cols] → W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, (vDSP_Length)cols, (vDSP_Length)rows);
}

// Spatial (last dim) sizes for unfused matmul kernels: input is [1, IC, 1, SEQ+OC]
#define WQ_FWD_SP  (SEQ + Q_DIM)
#define WKV_FWD_SP (SEQ + KV_DIM)
#define W13_FWD_SP (SEQ + HIDDEN)
#define W2_FWD_SP  (SEQ + DIM)

// ===== Compile forward-only dynamic kernels (ONCE) =====
static bool compile_dynamic_kernels(DynLayerKernels *dk, float res_alpha, bool unfused_fwd, bool compile_bwd) {
    (void)compile_bwd;  // MeZO never compiles backward kernels

    if (unfused_fwd) {
        // --- Unfused forward: individual matmul kernels ---
        printf("  Compiling wqFwd (DIM->Q_DIM)...\n");
        dk->wqFwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, Q_DIM, SEQ), @{},
            DIM*WQ_FWD_SP*2, Q_DIM*SEQ*2);
        if (!dk->wqFwd) return false;

        printf("  Compiling wkvFwd (DIM->KV_DIM)...\n");
        dk->wkvFwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, KV_DIM, SEQ), @{},
            DIM*WKV_FWD_SP*2, KV_DIM*SEQ*2);
        if (!dk->wkvFwd) return false;

        printf("  Compiling woFwd (Q_DIM->DIM)...\n");
        dk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
            Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
        if (!dk->woFwd) return false;

        printf("  Compiling w13Fwd (DIM->HIDDEN)...\n");
        dk->w13Fwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, HIDDEN, SEQ), @{},
            DIM*W13_FWD_SP*2, HIDDEN*SEQ*2);
        if (!dk->w13Fwd) return false;

        printf("  Compiling w2Fwd (HIDDEN->DIM)...\n");
        dk->w2Fwd = compile_kern_mil_w(gen_dyn_matmul_mil(HIDDEN, DIM, SEQ), @{},
            HIDDEN*W2_FWD_SP*2, DIM*SEQ*2);
        if (!dk->w2Fwd) return false;
    }

    return true;
}

// ===== xoshiro256+ PRNG (fast, deterministic, high-quality) =====
// Used for Rademacher perturbation: z_i in {-1, +1}
// 33x faster than Box-Muller+drand48 (21ms vs 700ms per 36.4M params)
static uint64_t xo_s[4];

static inline uint64_t xo_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t xo_next(void) {
    uint64_t result = xo_s[0] + xo_s[3];
    uint64_t t = xo_s[1] << 17;
    xo_s[2] ^= xo_s[0]; xo_s[3] ^= xo_s[1]; xo_s[1] ^= xo_s[2]; xo_s[0] ^= xo_s[3];
    xo_s[2] ^= t; xo_s[3] = xo_rotl(xo_s[3], 45);
    return result;
}

// Initialize xoshiro from a single seed (splitmix64 expansion)
static void xo_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        xo_s[i] = z ^ (z >> 31);
    }
}

// ===== Rademacher perturbation: buf[i] += scale * z_i, z_i in {-1,+1} =====
// Extracts 4 bits per xoshiro call for maximum throughput
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

// ===== Extract Rademacher direction z into buffer (for P-GAP gradient estimation) =====
// Fills z_out[n] with ±1 values using the same PRNG sequence as perturb_buffer
static void extract_z_buffer(float *z_out, size_t n) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        uint64_t r = xo_next();
        z_out[i+0] = (r & 1) ? 1.0f : -1.0f;
        z_out[i+1] = (r & 2) ? 1.0f : -1.0f;
        z_out[i+2] = (r & 4) ? 1.0f : -1.0f;
        z_out[i+3] = (r & 8) ? 1.0f : -1.0f;
    }
    for (; i < n; i++) {
        uint64_t r = xo_next();
        z_out[i] = (r & 1) ? 1.0f : -1.0f;
    }
}

// Extract full LoRA z-vector: same ordering as perturb_lora_weights
// z_out must be pre-allocated to total_trainable_params size
static void extract_lora_z(LoRALayer *ll, int nlayers, uint64_t seed, float *z_out) {
    xo_seed(seed);
    size_t off = 0;
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        extract_z_buffer(z_out + off, (size_t)r * DIM); off += (size_t)r * DIM;     // Aq
        extract_z_buffer(z_out + off, (size_t)Q_DIM * r); off += (size_t)Q_DIM * r; // Bq
        extract_z_buffer(z_out + off, (size_t)r * DIM); off += (size_t)r * DIM;     // Ak
        extract_z_buffer(z_out + off, (size_t)KV_DIM * r); off += (size_t)KV_DIM * r; // Bk
        extract_z_buffer(z_out + off, (size_t)r * DIM); off += (size_t)r * DIM;     // Av
        extract_z_buffer(z_out + off, (size_t)KV_DIM * r); off += (size_t)KV_DIM * r; // Bv
        extract_z_buffer(z_out + off, (size_t)r * Q_DIM); off += (size_t)r * Q_DIM; // Ao
        extract_z_buffer(z_out + off, (size_t)DIM * r); off += (size_t)DIM * r;     // Bo
        if (ll[L].has_ffn) {
            extract_z_buffer(z_out + off, (size_t)r * DIM); off += (size_t)r * DIM;       // A1
            extract_z_buffer(z_out + off, (size_t)HIDDEN * r); off += (size_t)HIDDEN * r;  // B1
            extract_z_buffer(z_out + off, (size_t)r * HIDDEN); off += (size_t)r * HIDDEN;  // A2
            extract_z_buffer(z_out + off, (size_t)DIM * r); off += (size_t)DIM * r;        // B2
            extract_z_buffer(z_out + off, (size_t)r * DIM); off += (size_t)r * DIM;        // A3
            extract_z_buffer(z_out + off, (size_t)HIDDEN * r); off += (size_t)HIDDEN * r;  // B3
        }
        extract_z_buffer(z_out + off, DIM); off += DIM;  // rms_att
        extract_z_buffer(z_out + off, DIM); off += DIM;  // rms_ffn
    }
    extract_z_buffer(z_out + off, DIM); off += DIM;      // rms_final
}

// ===== P-GAP: Perturb LoRA weights using pre-computed z-vector =====
// Unlike perturb_lora_weights (which regenerates z from seed), this uses a stored z buffer.
// The z buffer must follow the same parameter ordering as perturb_lora_weights.
static void perturb_lora_with_z(LoRALayer *ll, LayerWeights *lw, float *rms_final,
                                 int nlayers, const float *z, float scale) {
    size_t off = 0;
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        cblas_saxpy((int)((size_t)r * DIM), scale, z + off, 1, ll[L].Aq, 1); off += (size_t)r * DIM;
        cblas_saxpy((int)((size_t)Q_DIM * r), scale, z + off, 1, ll[L].Bq, 1); off += (size_t)Q_DIM * r;
        cblas_saxpy((int)((size_t)r * DIM), scale, z + off, 1, ll[L].Ak, 1); off += (size_t)r * DIM;
        cblas_saxpy((int)((size_t)KV_DIM * r), scale, z + off, 1, ll[L].Bk, 1); off += (size_t)KV_DIM * r;
        cblas_saxpy((int)((size_t)r * DIM), scale, z + off, 1, ll[L].Av, 1); off += (size_t)r * DIM;
        cblas_saxpy((int)((size_t)KV_DIM * r), scale, z + off, 1, ll[L].Bv, 1); off += (size_t)KV_DIM * r;
        cblas_saxpy((int)((size_t)r * Q_DIM), scale, z + off, 1, ll[L].Ao, 1); off += (size_t)r * Q_DIM;
        cblas_saxpy((int)((size_t)DIM * r), scale, z + off, 1, ll[L].Bo, 1); off += (size_t)DIM * r;
        if (ll[L].has_ffn) {
            cblas_saxpy((int)((size_t)r * DIM), scale, z + off, 1, ll[L].A1, 1); off += (size_t)r * DIM;
            cblas_saxpy((int)((size_t)HIDDEN * r), scale, z + off, 1, ll[L].B1, 1); off += (size_t)HIDDEN * r;
            cblas_saxpy((int)((size_t)r * HIDDEN), scale, z + off, 1, ll[L].A2, 1); off += (size_t)r * HIDDEN;
            cblas_saxpy((int)((size_t)DIM * r), scale, z + off, 1, ll[L].B2, 1); off += (size_t)DIM * r;
            cblas_saxpy((int)((size_t)r * DIM), scale, z + off, 1, ll[L].A3, 1); off += (size_t)r * DIM;
            cblas_saxpy((int)((size_t)HIDDEN * r), scale, z + off, 1, ll[L].B3, 1); off += (size_t)HIDDEN * r;
        }
        cblas_saxpy(DIM, scale, z + off, 1, lw[L].rms_att, 1); off += DIM;
        cblas_saxpy(DIM, scale, z + off, 1, lw[L].rms_ffn, 1); off += DIM;
    }
    cblas_saxpy(DIM, scale, z + off, 1, rms_final, 1);
}

// ===== Modified Gram-Schmidt QR: orthonormalize columns of Q[d × r] in-place =====
static void gram_schmidt_qr(float *Q_mat, size_t d, int r) {
    for (int j = 0; j < r; j++) {
        float *qj = Q_mat + (size_t)j * d;
        for (int i = 0; i < j; i++) {
            float *qi = Q_mat + (size_t)i * d;
            float dot;
            vDSP_dotpr(qi, 1, qj, 1, &dot, (vDSP_Length)d);
            cblas_saxpy((int)d, -dot, qi, 1, qj, 1);
        }
        float norm_sq;
        vDSP_dotpr(qj, 1, qj, 1, &norm_sq, (vDSP_Length)d);
        float inv_norm = 1.0f / sqrtf(norm_sq + 1e-10f);
        vDSP_vsmul(qj, 1, &inv_norm, qj, 1, (vDSP_Length)d);
    }
}

// ===== Count total trainable params for P-GAP =====
static size_t count_lora_params(LoRALayer *ll, int nlayers) {
    size_t total = 0;
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        total += (size_t)r * DIM * 3 + (size_t)Q_DIM * r + (size_t)KV_DIM * r * 2
               + (size_t)r * Q_DIM + (size_t)DIM * r;
        if (ll[L].has_ffn) {
            total += (size_t)r * DIM * 2 + (size_t)HIDDEN * r * 2
                   + (size_t)r * HIDDEN + (size_t)DIM * r;
        }
        total += DIM * 2;  // rms_att + rms_ffn
    }
    total += DIM;  // rms_final
    return total;
}

// ===== Sparse MeZO: exclude large-magnitude params from perturbation =====
static int cmp_float_abs(const void *a, const void *b) {
    float fa = fabsf(*(const float *)a), fb = fabsf(*(const float *)b);
    return (fa > fb) - (fa < fb);
}

static void compute_sparse_mask(LoRALayer *ll, LayerWeights *lw, float *rms_final,
                                int nlayers, uint8_t *mask, float sparse_ratio,
                                size_t total_params) {
    if (sparse_ratio <= 0.0f) { memset(mask, 1, total_params); return; }
    float *mags = (float *)safe_malloc(total_params * sizeof(float));
    size_t idx = 0;
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Aq[j]);
        for (size_t j = 0; j < (size_t)Q_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bq[j]);
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Ak[j]);
        for (size_t j = 0; j < (size_t)KV_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bk[j]);
        for (size_t j = 0; j < (size_t)r * DIM; j++) mags[idx++] = fabsf(ll[L].Av[j]);
        for (size_t j = 0; j < (size_t)KV_DIM * r; j++) mags[idx++] = fabsf(ll[L].Bv[j]);
        for (size_t j = 0; j < (size_t)r * Q_DIM; j++) mags[idx++] = fabsf(ll[L].Ao[j]);
        for (size_t j = 0; j < (size_t)DIM * r; j++) mags[idx++] = fabsf(ll[L].Bo[j]);
        for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(lw[L].rms_att[j]);
        for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(lw[L].rms_ffn[j]);
    }
    for (size_t j = 0; j < DIM; j++) mags[idx++] = fabsf(rms_final[j]);
    assert(idx == total_params);
    float *sorted = (float *)safe_malloc(total_params * sizeof(float));
    memcpy(sorted, mags, total_params * sizeof(float));
    qsort(sorted, total_params, sizeof(float), cmp_float_abs);
    size_t keep_count = (size_t)((1.0f - sparse_ratio) * total_params);
    if (keep_count == 0) keep_count = 1;
    if (keep_count > total_params) keep_count = total_params;
    float threshold = sorted[keep_count - 1];
    for (size_t i = 0; i < total_params; i++) mask[i] = (mags[i] <= threshold) ? 1 : 0;
    size_t n_active = 0;
    for (size_t i = 0; i < total_params; i++) n_active += mask[i];
    printf("  [Sparse mask] ratio=%.3f  threshold=%.6f  active=%zu/%zu (%.1f%%)\n",
           sparse_ratio, threshold, n_active, total_params, 100.0f * n_active / total_params);
    free(mags); free(sorted);
}

// ===== HiZOO: Diagonal Hessian preconditioning =====
// z_vals: raw Gaussian z values from the perturbation step. z[i]² varies per
// element, enabling per-parameter Hessian differentiation. This is why HiZOO
// requires Gaussian (not Rademacher) perturbations.
static void update_hessian(float *H, const float *z_vals, size_t n,
                           float loss_plus, float loss_minus, float loss_0,
                           float epsilon, float alpha) {
    float delta_L = loss_plus + loss_minus - 2.0f * loss_0;
    float curvature = fabsf(delta_L) / (epsilon * epsilon);
    for (size_t i = 0; i < n; i++) {
        float h_est = curvature * z_vals[i] * z_vals[i];  // z[i]² varies per element
        H[i] = (1.0f - alpha) * H[i] + alpha * h_est;
        if (H[i] < 1e-8f) H[i] = 1e-8f;
        if (H[i] > 1e6f) H[i] = 1e6f;
    }
}

static void print_hessian_stats(const float *H, size_t n, int step) {
    float h_min = H[0], h_max = H[0];
    double h_sum = 0, h_sum2 = 0;
    for (size_t i = 0; i < n; i++) {
        if (H[i] < h_min) h_min = H[i];
        if (H[i] > h_max) h_max = H[i];
        h_sum += H[i]; h_sum2 += (double)H[i] * H[i];
    }
    double h_mean = h_sum / n, h_var = h_sum2 / n - h_mean * h_mean;
    printf("  [Hessian@%d] min=%.4e max=%.4e mean=%.4e std=%.4e ratio=%.1f\n",
           step, h_min, h_max, (float)h_mean, (float)sqrt(fmax(h_var,0)), h_max/(h_min+1e-30f));
}

// ===== Gaussian RNG: Box-Muller from xoshiro256+ =====
// Returns two N(0,1) samples per call. Uses global xo_s[] state.
static inline void gaussian_pair(float *out1, float *out2) {
    // Generate two uniform (0,1) from xoshiro
    uint64_t r1 = xo_next(), r2 = xo_next();
    // Map to (0,1) — avoid exact 0 for log safety
    double u1 = ((double)(r1 >> 11) + 0.5) / (double)(1ULL << 53);
    double u2 = ((double)(r2 >> 11) + 0.5) / (double)(1ULL << 53);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;
    *out1 = (float)(r * cos(theta));
    *out2 = (float)(r * sin(theta));
}

// Fill buffer with N(0,1) Gaussian samples
static void gaussian_fill(float *buf, size_t n) {
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        gaussian_pair(&buf[i], &buf[i+1]);
    }
    if (i < n) {
        float tmp;
        gaussian_pair(&buf[i], &tmp);
    }
}

// ===== Faithful P-GAP: Per-matrix SVD basis =====
// Each LoRA matrix gets its own SVD basis (U, S, V) for gradient-aligned perturbation.
// For a matrix W ∈ R^{m×n}, the probe accumulates G ∈ R^{m×n}, then:
//   SVD(G) = U_r × diag(S_r) × V_r^T  (truncated to rank pgap_svd_r)
// Training perturbation: Z_init ~ N(0, I_{r×r}), PROJECTION, Z_f = U × Z × V^T
typedef struct {
    int rows, cols;       // matrix dimensions (m, n)
    int svd_r;            // SVD truncation rank (min(rows,cols,pgap_svd_r))
    float *G;             // Accumulated gradient [rows × cols]
    float *U;             // Left singular vectors [rows × svd_r]
    float *S;             // Singular values [svd_r]
    float *Vt;            // Right singular vectors transposed [svd_r × cols]
    float *Z_f;           // Current perturbation for this matrix [rows × cols]
    bool has_basis;       // Whether SVD has been computed
} PGAPMatrixBasis;

// Number of LoRA matrices per layer (attention-only: 8, with FFN: 14)
#define PGAP_ATTN_MATS 8
#define PGAP_FFN_MATS  6
#define PGAP_RMS_PER_LAYER 2  // rms_att, rms_ffn (1D vectors, no SVD)

// Initialize P-GAP basis for one LoRA matrix
static PGAPMatrixBasis pgap_basis_init(int rows, int cols, int max_svd_r) {
    PGAPMatrixBasis b;
    b.rows = rows;
    b.cols = cols;
    b.svd_r = max_svd_r;
    if (b.svd_r > rows) b.svd_r = rows;
    if (b.svd_r > cols) b.svd_r = cols;
    b.G  = (float*)safe_calloc((size_t)rows * cols, 4);
    b.U  = (float*)safe_calloc((size_t)rows * b.svd_r, 4);
    b.S  = (float*)safe_calloc(b.svd_r, 4);
    b.Vt = (float*)safe_calloc((size_t)b.svd_r * cols, 4);
    b.Z_f = (float*)safe_calloc((size_t)rows * cols, 4);
    b.has_basis = false;
    return b;
}

static void pgap_basis_free(PGAPMatrixBasis *b) {
    free(b->G); free(b->U); free(b->S); free(b->Vt); free(b->Z_f);
    memset(b, 0, sizeof(*b));
}

// Per-matrix SVD via eigendecomposition of the smaller Gram matrix.
// For G[m×n]: if m <= n, compute G·G^T [m×m], else G^T·G [n×n].
// Eigendecompose to get singular values and vectors.
static void pgap_compute_svd(PGAPMatrixBasis *b) {
    int m = b->rows, n = b->cols, r = b->svd_r;
    // Determine which Gram matrix to use
    bool use_GGt = (m <= n);  // G·G^T is m×m
    int gram_dim = use_GGt ? m : n;

    float *gram = (float*)safe_calloc((size_t)gram_dim * gram_dim, 4);

    if (use_GGt) {
        // gram = G · G^T [m×m], G is [m×n] row-major
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, m, n, 1.0f, b->G, n, b->G, n, 0.0f, gram, m);
    } else {
        // gram = G^T · G [n×n]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    n, n, m, 1.0f, b->G, n, b->G, n, 0.0f, gram, n);
    }

    // Eigendecomposition: gram = V_g · diag(eig) · V_g^T
    float *eigenvalues = (float*)safe_malloc(gram_dim * 4);
    {
        char jobz = 'V', uplo = 'U';
        __LAPACK_int nn = gram_dim, lda = gram_dim, info = 0;
        __LAPACK_int lwork = -1;
        float work_query;
        ssyev_(&jobz, &uplo, &nn, gram, &lda, eigenvalues, &work_query, &lwork, &info);
        lwork = (__LAPACK_int)work_query;
        float *work = (float*)safe_malloc(lwork * 4);
        ssyev_(&jobz, &uplo, &nn, gram, &lda, eigenvalues, work, &lwork, &info);
        free(work);
        if (info != 0) {
            fprintf(stderr, "pgap_compute_svd: ssyev failed info=%d\n", info);
            free(gram); free(eigenvalues);
            return;
        }
    }

    // Eigenvalues are ascending. Take top-r (from end).
    // Singular values = sqrt(eigenvalues). Store descending.
    for (int i = 0; i < r; i++) {
        float ev = eigenvalues[gram_dim - 1 - i];
        b->S[i] = (ev > 0) ? sqrtf(ev) : 0.0f;
    }

    if (use_GGt) {
        // gram columns (LAPACK column-major) are eigenvectors of G·G^T = left singular vectors U
        // ssyev stores eigenvectors in columns of gram (column-major, gram_dim=m)
        // Top-r eigenvectors are last r columns (ascending eigenvalue order)
        // U[m × r] — extract last r columns, reverse order (largest first)
        for (int i = 0; i < r; i++) {
            // Column (gram_dim - 1 - i) of gram (column-major) → column i of U
            float *src_col = gram + (size_t)(gram_dim - 1 - i) * m;
            for (int row = 0; row < m; row++) {
                b->U[row * r + i] = src_col[row];  // Store U row-major
            }
        }
        // Recover V^T: V^T = diag(1/S) · U^T · G
        // V^T[r × n] = diag(1/S)[r×r] · U^T[r×m] · G[m×n]
        // First: tmp[r×n] = U^T · G
        float *tmp = (float*)safe_malloc((size_t)r * n * 4);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    r, n, m, 1.0f, b->U, r, b->G, n, 0.0f, tmp, n);
        // Scale each row i by 1/S[i]
        for (int i = 0; i < r; i++) {
            float inv_s = (b->S[i] > 1e-10f) ? 1.0f / b->S[i] : 0.0f;
            for (int j = 0; j < n; j++) {
                b->Vt[i * n + j] = tmp[i * n + j] * inv_s;
            }
        }
        free(tmp);
    } else {
        // gram columns are eigenvectors of G^T·G = right singular vectors V
        // Vt[r × n] — extract last r columns, reverse, transpose
        for (int i = 0; i < r; i++) {
            float *src_col = gram + (size_t)(gram_dim - 1 - i) * n;
            for (int col = 0; col < n; col++) {
                b->Vt[i * n + col] = src_col[col];  // Store V^T row-major
            }
        }
        // Recover U: U = G · V · diag(1/S)
        // U[m × r] = G[m×n] · V[n×r] · diag(1/S)
        // V[n×r] is Vt transposed
        float *V_mat = (float*)safe_malloc((size_t)n * r * 4);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < n; j++)
                V_mat[j * r + i] = b->Vt[i * n + j];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, r, n, 1.0f, b->G, n, V_mat, r, 0.0f, b->U, r);
        // Scale each column i by 1/S[i]
        for (int i = 0; i < r; i++) {
            float inv_s = (b->S[i] > 1e-10f) ? 1.0f / b->S[i] : 0.0f;
            for (int row = 0; row < m; row++) {
                b->U[row * r + i] *= inv_s;
            }
        }
        free(V_mat);
    }

    b->has_basis = true;
    free(gram);
    free(eigenvalues);
}

// P-GAP PROJECTION constraint:
// Given Z_init[r×r], S[r] (diagonal as matrix), compute:
//   alpha = (<S, Z_init>_F - xi*sqrt(delta)*||S||_F) / (||S||^2_F + 1e-12)
//   Z = Z_init - alpha * S
// This ensures <S, Z>_F = xi*sqrt(delta)*||S||_F (gradient alignment)
static void pgap_project(float *Z_init, const float *S, int r, float xi, float delta) {
    // S is diagonal stored as vector S[r]. In the r×r matrix view:
    //   <S_mat, Z>_F = sum_i S[i] * Z[i*r+i]  (only diagonal contributes)
    //   ||S_mat||_F = sqrt(sum_i S[i]^2)
    float S_dot_Z = 0, S_norm_sq = 0;
    for (int i = 0; i < r; i++) {
        S_dot_Z += S[i] * Z_init[i * r + i];  // diagonal element
        S_norm_sq += S[i] * S[i];
    }
    float S_norm = sqrtf(S_norm_sq);
    float target = xi * sqrtf(fabsf(delta)) * S_norm;
    float alpha = (S_dot_Z - target) / (S_norm_sq + 1e-12f);

    // Z_init[i,j] -= alpha * S_mat[i,j]
    // S_mat is diagonal, so only Z_init[i*r+i] -= alpha * S[i]
    for (int i = 0; i < r; i++) {
        Z_init[i * r + i] -= alpha * S[i];
    }
}

// Generate perturbation Z_f = U × Z × V^T for one matrix
// Z_init is r×r, U is [m×r], V^T is [r×n], Z_f is [m×n]
static void pgap_gen_perturbation(PGAPMatrixBasis *b, float *Z_init) {
    int m = b->rows, n = b->cols, r = b->svd_r;
    // tmp[m×r] = U[m×r] × Z_init[r×r]
    float *tmp = (float*)safe_malloc((size_t)m * r * 4);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, r, r, 1.0f, b->U, r, Z_init, r, 0.0f, tmp, r);
    // Z_f[m×n] = tmp[m×r] × Vt[r×n]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, r, 1.0f, tmp, r, b->Vt, n, 0.0f, b->Z_f, n);
    free(tmp);
}

// ===== Perturb ALL model weights using deterministic seed =====
// NOTE: perturbs `embed` (full VOCAB*DIM), not cembed. Caller must rebuild
// cembed = vocab_compact_embed(embed, &vm, DIM) before any forward pass.
static void perturb_all_weights(LayerWeights *lw, float *embed, float *rms_final,
                                uint64_t seed, float scale) {
    xo_seed(seed);
    perturb_buffer(embed, (size_t)VOCAB * DIM, scale);
    for (int L = 0; L < NLAYERS; L++) {
        perturb_buffer(lw[L].rms_att, DIM, scale);
        perturb_buffer(lw[L].Wq, WQ_SZ, scale);
        perturb_buffer(lw[L].Wk, WK_SZ, scale);
        perturb_buffer(lw[L].Wv, WV_SZ, scale);
        perturb_buffer(lw[L].Wo, WO_SZ, scale);
        perturb_buffer(lw[L].rms_ffn, DIM, scale);
        perturb_buffer(lw[L].W1, W1_SZ, scale);
        perturb_buffer(lw[L].W2, W2_SZ, scale);
        perturb_buffer(lw[L].W3, W3_SZ, scale);
    }
    perturb_buffer(rms_final, DIM, scale);
}

// ===== MeZO + LoRA: perturb ONLY adapter matrices =====
static void perturb_lora_weights(LoRALayer *ll, LayerWeights *lw,
                                 float *rms_final, int nlayers, uint64_t seed, float scale) {
    xo_seed(seed);
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        // Attention adapters
        perturb_buffer(ll[L].Aq, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bq, (size_t)Q_DIM * r, scale);
        perturb_buffer(ll[L].Ak, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bk, (size_t)KV_DIM * r, scale);
        perturb_buffer(ll[L].Av, (size_t)r * DIM, scale);
        perturb_buffer(ll[L].Bv, (size_t)KV_DIM * r, scale);
        perturb_buffer(ll[L].Ao, (size_t)r * Q_DIM, scale);
        perturb_buffer(ll[L].Bo, (size_t)DIM * r, scale);
        // FFN adapters (if present)
        if (ll[L].has_ffn) {
            perturb_buffer(ll[L].A1, (size_t)r * DIM, scale);
            perturb_buffer(ll[L].B1, (size_t)HIDDEN * r, scale);
            perturb_buffer(ll[L].A2, (size_t)r * HIDDEN, scale);
            perturb_buffer(ll[L].B2, (size_t)DIM * r, scale);
            perturb_buffer(ll[L].A3, (size_t)r * DIM, scale);
            perturb_buffer(ll[L].B3, (size_t)HIDDEN * r, scale);
        }
        // RMS norms are still trainable (small, always perturbed)
        perturb_buffer(lw[L].rms_att, DIM, scale);
        perturb_buffer(lw[L].rms_ffn, DIM, scale);
    }
    perturb_buffer(rms_final, DIM, scale);
}

// ===== HiZOO perturbation: Gaussian z with sparse mask + Hessian scaling =====
// When H != NULL (HiZOO mode): uses GAUSSIAN z (Box-Muller) so z[i]² varies
// per element, enabling per-parameter Hessian differentiation.
// When H == NULL (sparse-only): uses Rademacher z (faster, sufficient).
// z_out: if non-NULL, stores the raw z values for Hessian update.
// CRITICAL: PRNG sequence differs from perturb_lora_weights — do not mix.
static void perturb_lora_hizoo(LoRALayer *ll, LayerWeights *lw,
                                float *rms_final, int nlayers, uint64_t seed,
                                float scale, const uint8_t *mask, const float *H,
                                float *z_out) {
    xo_seed(seed);
    size_t idx = 0;
    bool use_gaussian = (H != NULL);  // Gaussian needed for Hessian differentiation

    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        #define PERTURB_HIZOO(buf, count) do { \
            size_t _n = (count); \
            if (use_gaussian) { \
                /* Generate Gaussian z via Box-Muller, then apply mask+Hessian */ \
                for (size_t _i = 0; _i < _n; _i += 2) { \
                    float g1, g2; \
                    gaussian_pair(&g1, &g2); \
                    for (int _k = 0; _k < 2 && (_i + _k) < _n; _k++) { \
                        float z_i = (_k == 0) ? g1 : g2; \
                        if (z_out) z_out[idx + _i + _k] = z_i; \
                        float s = scale; \
                        if (mask && !mask[idx + _i + _k]) s = 0.0f; \
                        if (H) s /= sqrtf(H[idx + _i + _k]); \
                        (buf)[_i + _k] += s * z_i; \
                    } \
                } \
            } else { \
                /* Rademacher z (faster, for sparse-only mode) */ \
                for (size_t _i = 0; _i < _n; _i++) { \
                    uint64_t _r = xo_next(); \
                    float z_i = (_r & 1) ? 1.0f : -1.0f; \
                    if (z_out) z_out[idx + _i] = z_i; \
                    float s = scale; \
                    if (mask && !mask[idx + _i]) s = 0.0f; \
                    (buf)[_i] += s * z_i; \
                } \
            } \
            idx += _n; \
        } while(0)
        PERTURB_HIZOO(ll[L].Aq, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bq, (size_t)Q_DIM * r);
        PERTURB_HIZOO(ll[L].Ak, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bk, (size_t)KV_DIM * r);
        PERTURB_HIZOO(ll[L].Av, (size_t)r * DIM);
        PERTURB_HIZOO(ll[L].Bv, (size_t)KV_DIM * r);
        PERTURB_HIZOO(ll[L].Ao, (size_t)r * Q_DIM);
        PERTURB_HIZOO(ll[L].Bo, (size_t)DIM * r);
        PERTURB_HIZOO(lw[L].rms_att, DIM);
        PERTURB_HIZOO(lw[L].rms_ffn, DIM);
        #undef PERTURB_HIZOO
    }
    {
        size_t _n = DIM;
        if (use_gaussian) {
            for (size_t _i = 0; _i < _n; _i += 2) {
                float g1, g2;
                gaussian_pair(&g1, &g2);
                for (int _k = 0; _k < 2 && (_i + _k) < _n; _k++) {
                    float z_i = (_k == 0) ? g1 : g2;
                    if (z_out) z_out[idx + _i + _k] = z_i;
                    float s = scale;
                    if (mask && !mask[idx + _i + _k]) s = 0.0f;
                    if (H) s /= sqrtf(H[idx + _i + _k]);
                    rms_final[_i + _k] += s * z_i;
                }
            }
        } else {
            for (size_t _i = 0; _i < _n; _i++) {
                uint64_t _r = xo_next();
                float z_i = (_r & 1) ? 1.0f : -1.0f;
                if (z_out) z_out[idx + _i] = z_i;
                float s = scale;
                if (mask && !mask[idx + _i]) s = 0.0f;
                rms_final[_i] += s * z_i;
            }
        }
        idx += _n;
    }
}

// Merge LoRA adapters into effective weights: W_eff = W_base + B @ A
static void lora_merge_all(LayerWeights *lw, LoRALayer *ll, int nlayers) {
    for (int L = 0; L < nlayers; L++) {
        int r = ll[L].rank;
        lora_merge_weight(lw[L].Wq, ll[L].Wq_base, ll[L].Bq, ll[L].Aq, Q_DIM, r, DIM);
        lora_merge_weight(lw[L].Wk, ll[L].Wk_base, ll[L].Bk, ll[L].Ak, KV_DIM, r, DIM);
        lora_merge_weight(lw[L].Wv, ll[L].Wv_base, ll[L].Bv, ll[L].Av, KV_DIM, r, DIM);
        lora_merge_weight(lw[L].Wo, ll[L].Wo_base, ll[L].Bo, ll[L].Ao, DIM, r, Q_DIM);
        if (ll[L].has_ffn) {
            lora_merge_weight(lw[L].W1, ll[L].W1_base, ll[L].B1, ll[L].A1, HIDDEN, r, DIM);
            lora_merge_weight(lw[L].W2, ll[L].W2_base, ll[L].B2, ll[L].A2, DIM, r, HIDDEN);
            lora_merge_weight(lw[L].W3, ll[L].W3_base, ll[L].B3, ll[L].A3, HIDDEN, r, DIM);
        }
    }
}

// ===== Adapter-as-input: compute LoRA correction CPU-side, add to ANE output =====
// out += B @ (A @ x), where A[rank,in_dim], B[out_dim,rank], x[in_dim,SEQ]
static void lora_addmm(float *out, const float *A, const float *B,
                        const float *x, float *tmp_r,
                        int out_dim, int rank, int in_dim) {
    // tmp_r[rank, SEQ] = A[rank, in_dim] @ x[in_dim, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rank, SEQ, in_dim, 1.0f, A, in_dim, x, SEQ, 0.0f, tmp_r, SEQ);
    // out[out_dim, SEQ] += B[out_dim, rank] @ tmp_r[rank, SEQ]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                out_dim, SEQ, rank, 1.0f, B, rank, tmp_r, SEQ, 1.0f, out, SEQ);
}

// ===== MeZO checkpoint (BLZT v4, zeros for Adam state) =====
static void mezo_save_checkpoint(const char *path, int step, int total_steps, float lr, float loss,
                                 double ct, double cw, int cs,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write checkpoint %s\n", path); return; }
    CkptHdr h = {0};
    h.magic = 0x424C5A54; h.version = 4;
    h.step = step; h.total_steps = total_steps; h.lr = lr; h.loss = loss;
    h.n_layers = NLAYERS; h.vocab_size = VOCAB; h.dim = DIM; h.hidden_dim = HIDDEN;
    h.n_heads = HEADS; h.seq_len = SEQ;
    h.cum_train = ct; h.cum_wall = cw; h.cum_steps = cs; h.adam_t = 0;
    h.kv_heads = KV_HEADS; h.head_dim = HD; h.q_dim = Q_DIM;
    fwrite(&h, sizeof(h), 1, f);
    // Write weights + zeros for Adam state (layout matches train.m exactly)
    size_t max_sz = WQ_SZ > W1_SZ ? WQ_SZ : W1_SZ;
    if ((size_t)VOCAB * DIM > max_sz) max_sz = (size_t)VOCAB * DIM;
    float *zeros_big = (float*)safe_calloc(max_sz, 4);
    for (int L = 0; L < NLAYERS; L++) {
        // All weights first (same order as train.m)
        fwrite(lw[L].Wq,4,WQ_SZ,f); fwrite(lw[L].Wk,4,WK_SZ,f);
        fwrite(lw[L].Wv,4,WV_SZ,f); fwrite(lw[L].Wo,4,WO_SZ,f);
        fwrite(lw[L].W1,4,W1_SZ,f); fwrite(lw[L].W2,4,W2_SZ,f); fwrite(lw[L].W3,4,W3_SZ,f);
        fwrite(lw[L].rms_att,4,DIM,f); fwrite(lw[L].rms_ffn,4,DIM,f);
        // All Adam m/v states as zeros (same order as train.m)
        fwrite(zeros_big,4,WQ_SZ,f); fwrite(zeros_big,4,WQ_SZ,f);
        fwrite(zeros_big,4,WK_SZ,f); fwrite(zeros_big,4,WK_SZ,f);
        fwrite(zeros_big,4,WV_SZ,f); fwrite(zeros_big,4,WV_SZ,f);
        fwrite(zeros_big,4,WO_SZ,f); fwrite(zeros_big,4,WO_SZ,f);
        fwrite(zeros_big,4,W1_SZ,f); fwrite(zeros_big,4,W1_SZ,f);
        fwrite(zeros_big,4,W2_SZ,f); fwrite(zeros_big,4,W2_SZ,f);
        fwrite(zeros_big,4,W3_SZ,f); fwrite(zeros_big,4,W3_SZ,f);
        fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
        fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
    }
    fwrite(rms_final,4,DIM,f); fwrite(zeros_big,4,DIM,f); fwrite(zeros_big,4,DIM,f);
    fwrite(embed,4,(size_t)VOCAB*DIM,f); fwrite(zeros_big,4,(size_t)VOCAB*DIM,f); fwrite(zeros_big,4,(size_t)VOCAB*DIM,f);
    free(zeros_big);
    fclose(f);
}

static bool mezo_load_checkpoint(const char *path, int *step, float *lr, float *loss,
                                 LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return false; }
    if (h.magic != 0x424C5A54 || h.version != 4) { fclose(f); return false; }
    if (h.n_layers != NLAYERS || h.dim != DIM || h.vocab_size != VOCAB ||
        h.hidden_dim != HIDDEN || h.seq_len != SEQ || h.n_heads != HEADS ||
        h.kv_heads != KV_HEADS || h.head_dim != HD || h.q_dim != Q_DIM) {
        fprintf(stderr, "MeZO checkpoint mismatch\n"); fclose(f); return false;
    }
    *step = h.step; *lr = h.lr; *loss = h.loss;
    if (h.step < 0 || h.step > 10000000) {
        fprintf(stderr, "MeZO checkpoint has invalid step value\n"); fclose(f); return false;
    }
    // Read weights, skip Adam m/v (layout matches train.m exactly)
    size_t max_sz = WQ_SZ > W1_SZ ? WQ_SZ : W1_SZ;
    if ((size_t)VOCAB * DIM > max_sz) max_sz = (size_t)VOCAB * DIM;
    float *skip = (float*)safe_malloc(max_sz * 4);
    for (int L = 0; L < NLAYERS; L++) {
        // All weights first (same order as train.m)
        fread(lw[L].Wq,4,WQ_SZ,f); fread(lw[L].Wk,4,WK_SZ,f);
        fread(lw[L].Wv,4,WV_SZ,f); fread(lw[L].Wo,4,WO_SZ,f);
        fread(lw[L].W1,4,W1_SZ,f); fread(lw[L].W2,4,W2_SZ,f); fread(lw[L].W3,4,W3_SZ,f);
        fread(lw[L].rms_att,4,DIM,f); fread(lw[L].rms_ffn,4,DIM,f);
        // Skip all Adam m/v states
        fread(skip,4,WQ_SZ,f); fread(skip,4,WQ_SZ,f);
        fread(skip,4,WK_SZ,f); fread(skip,4,WK_SZ,f);
        fread(skip,4,WV_SZ,f); fread(skip,4,WV_SZ,f);
        fread(skip,4,WO_SZ,f); fread(skip,4,WO_SZ,f);
        fread(skip,4,W1_SZ,f); fread(skip,4,W1_SZ,f);
        fread(skip,4,W2_SZ,f); fread(skip,4,W2_SZ,f);
        fread(skip,4,W3_SZ,f); fread(skip,4,W3_SZ,f);
        fread(skip,4,DIM,f); fread(skip,4,DIM,f);
        fread(skip,4,DIM,f); fread(skip,4,DIM,f);
    }
    fread(rms_final,4,DIM,f); fread(skip,4,DIM,f); fread(skip,4,DIM,f);
    fread(embed,4,(size_t)VOCAB*DIM,f); fread(skip,4,(size_t)VOCAB*DIM,f); fread(skip,4,(size_t)VOCAB*DIM,f);
    free(skip);
    fclose(f);
    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&g_tb);

        // Defaults
        int total_steps = 999999;
        float lr = 1e-5f, base_lr = 1e-5f;
        float epsilon = 1e-3f;
        double time_budget_sec = 0;
        bool from_scratch = false, cpu_only = false, ane_matmul_only = false;
        bool conv_hybrid = false; // conv1x1 for Wq,Wo,W1,W2,W3; matmul for Wk,Wv
        bool conv_fused = false;  // fused QKV + fused FFN conv1x1 kernels (Phase 2)
        bool backprop_lora = false;  // P16: ANE conv-fused forward + CPU backward + LoRA gradients
        bool lr_from_cli = false;
        bool use_lora = false;
        bool lora_split = false;  // adapter-as-input: no merge, no restage
        bool lora_ffn = false;    // also apply LoRA to W1, W2, W3
        int lora_rank = 8;
        int fzoo_K = 0;  // FZOO multi-perturbation: 0=disabled (standard MeZO), K>=1=use K directions
        int pgap_r = 0;  // P-GAP: subspace rank (0=disabled, r>0=use r-rank subspace)
        int pgap_k = 100; // P-GAP: subspace refresh interval (every k steps) — paper default 100
        int pgap_h = 10;  // P-GAP: probe perturbations for subspace estimation — paper default 10
        float pgap_xi = 1.0f;    // P-GAP: alignment strength parameter
        float pgap_delta0 = 2.0f; // P-GAP: initial delta (decays linearly to 0)
        int probe_gradient = 0; // Diagnostic: probe gradient subspace and report SVD spectrum
        float sparse_ratio = 0.0f;
        float hessian_alpha = 0.0f;
        int mask_refresh = 100;
        long init_seed = 42;
        int val_every = 500;
        const char *data_path = DEFAULT_DATA_PATH;
        const char *ckpt_load_path = NULL;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--scratch") == 0) from_scratch = true;
            else if (strcmp(argv[i], "--cpu-only") == 0) cpu_only = true;
            else if (strcmp(argv[i], "--ane-matmul-only") == 0) ane_matmul_only = true;
            else if (strcmp(argv[i], "--lr") == 0 && i+1 < argc) { lr = atof(argv[++i]); base_lr = lr; lr_from_cli = true; }
            else if (strcmp(argv[i], "--epsilon") == 0 && i+1 < argc) epsilon = atof(argv[++i]);
            else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) total_steps = atoi(argv[++i]);
            else if (strcmp(argv[i], "--time") == 0 && i+1 < argc) time_budget_sec = atof(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) init_seed = atol(argv[++i]);
            else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) data_path = argv[++i];
            else if (strcmp(argv[i], "--val-every") == 0 && i+1 < argc) val_every = atoi(argv[++i]);
            else if (strcmp(argv[i], "--resume") == 0 && i+1 < argc) ckpt_load_path = argv[++i];
            else if (strcmp(argv[i], "--lora") == 0) use_lora = true;
            else if (strcmp(argv[i], "--lora-split") == 0) { use_lora = true; lora_split = true; }
            else if (strcmp(argv[i], "--lora-rank") == 0 && i+1 < argc) lora_rank = atoi(argv[++i]);
            else if (strcmp(argv[i], "--lora-ffn") == 0) { lora_ffn = true; if (!use_lora) { use_lora = true; lora_split = true; } }
            else if (strcmp(argv[i], "--conv-hybrid") == 0) conv_hybrid = true;
            else if (strcmp(argv[i], "--conv-fused") == 0) conv_fused = true;
            else if (strcmp(argv[i], "--backprop-lora") == 0) backprop_lora = true;
            else if (strcmp(argv[i], "--fzoo") == 0 && i+1 < argc) { fzoo_K = atoi(argv[++i]); }
            else if (strcmp(argv[i], "--pgap") == 0 && i+1 < argc) { pgap_r = atoi(argv[++i]); }
            else if (strcmp(argv[i], "--pgap-k") == 0 && i+1 < argc) { pgap_k = atoi(argv[++i]); }
            else if (strcmp(argv[i], "--pgap-h") == 0 && i+1 < argc) { pgap_h = atoi(argv[++i]); }
            else if (strcmp(argv[i], "--pgap-xi") == 0 && i+1 < argc) { pgap_xi = atof(argv[++i]); }
            else if (strcmp(argv[i], "--pgap-delta0") == 0 && i+1 < argc) { pgap_delta0 = atof(argv[++i]); }
            else if (strcmp(argv[i], "--probe-gradient") == 0 && i+1 < argc) { probe_gradient = atoi(argv[++i]); }
            else if (strcmp(argv[i], "--sparse-ratio") == 0 && i+1 < argc) { sparse_ratio = atof(argv[++i]); }
            else if (strcmp(argv[i], "--hessian-alpha") == 0 && i+1 < argc) { hessian_alpha = atof(argv[++i]); }
            else if (strcmp(argv[i], "--mask-refresh") == 0 && i+1 < argc) { mask_refresh = atoi(argv[++i]); }
        }

        if (fzoo_K < 0) { fprintf(stderr, "ERROR: --fzoo K must be >= 1\n"); return 1; }
        if (fzoo_K > 0 && fzoo_K < 1) { fprintf(stderr, "ERROR: --fzoo K must be >= 1\n"); return 1; }
        if (sparse_ratio < 0.0f || sparse_ratio >= 1.0f) {
            fprintf(stderr, "ERROR: --sparse-ratio must be in [0, 1)\n"); return 1;
        }
        if (hessian_alpha < 0.0f) {
            fprintf(stderr, "ERROR: --hessian-alpha must be >= 0\n"); return 1;
        }

        // --conv-hybrid and --conv-fused require --lora-split (base weights frozen, baked at compile time)
        if (conv_hybrid && !lora_split) {
            fprintf(stderr, "ERROR: --conv-hybrid requires --lora-split (base weights must be frozen)\n");
            return 1;
        }
        if (conv_fused && !lora_split) {
            fprintf(stderr, "ERROR: --conv-fused requires --lora-split (base weights must be frozen)\n");
            return 1;
        }
        // --conv-fused FFN kernel bakes W1/W2/W3 — incompatible with LoRA on FFN
        if (conv_fused && lora_ffn) {
            fprintf(stderr, "ERROR: --conv-fused is incompatible with --lora-ffn (FFN weights are baked)\n");
            return 1;
        }
        // --conv-fused implies conv-hybrid (superset)
        if (conv_fused) conv_hybrid = true;

        // --backprop-lora requires --lora-split and implies --conv-fused for ANE forward
        if (backprop_lora) {
            if (!lora_split) { use_lora = true; lora_split = true; }
            if (!conv_fused && !cpu_only) { conv_fused = true; conv_hybrid = true; }
        }

        if (conv_hybrid) { ane_matmul_only = true; cpu_only = false; }  // conv-hybrid implies ANE mode
        if (!cpu_only && !ane_matmul_only) cpu_only = true;  // Default to CPU-only
        if (!cpu_only) {
            if (!ane_init()) {
                fprintf(stderr, "ANE init failed. Use --cpu-only.\n");
                return 1;
            }
        }

        // === Print config ===
        printf("=== MeZO (Zeroth-Order) Training: %s (%d layers, GQA %d/%d) ===\n",
               MODEL_NAME, NLAYERS, HEADS, KV_HEADS);
        printf("dim=%d q_dim=%d kv_dim=%d hd=%d hidden=%d seq=%d vocab=%d\n",
               DIM, Q_DIM, KV_DIM, HD, HIDDEN, SEQ, VOCAB);
        double total_p = (double)NLAYERS * LAYER_PARAMS + DIM + (double)VOCAB * DIM;
        const char *mode_str = cpu_only ? "CPU-only" : (conv_fused ? "ANE-conv-fused" : (conv_hybrid ? "ANE-conv-hybrid" : "ANE-matmul-only"));
        printf("Params: %.1fM | Mode: %s%s\n", total_p / 1e6, mode_str,
               backprop_lora ? " + BACKPROP-LORA (P16 hybrid)" : "");
        if (backprop_lora) {
            printf("P16 Hybrid: ANE conv-fused forward + CPU fp32 backward + LoRA gradients\n");
            printf("  lr=%g seed=%ld val_every=%d rank=%d\n", lr, init_seed, val_every, lora_rank);
        } else if (use_lora) {
            printf("MeZO+LoRA: lr=%g epsilon=%g seed=%ld val_every=%d rank=%d\n", lr, epsilon, init_seed, val_every, lora_rank);
        } else {
            printf("MeZO: lr=%g epsilon=%g seed=%ld val_every=%d\n", lr, epsilon, init_seed, val_every);
        }
        if (sparse_ratio > 0) printf("  sparse_ratio=%.3f  mask_refresh=%d\n", sparse_ratio, mask_refresh);
        if (hessian_alpha > 0) printf("  hessian_alpha=%.2e\n", hessian_alpha);
        if (fzoo_K > 0) printf("FZOO: K=%d directions (%d forward passes/step, adaptive step size)\n", fzoo_K, fzoo_K + 1);
        printf("Memory: ~%.0fMB (inference only, no gradients/optimizer)\n",
               (total_p * 4 + SEQ * DIM * 4 * 10) / 1e6);

        // === Allocate weights ===
        LayerWeights lw[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) lw[L] = layer_weights_alloc();
        float *rms_final = (float*)safe_malloc(DIM * 4);
        float *embed = (float*)safe_malloc((size_t)VOCAB * DIM * 4);

        int start_step = 0;
        float resume_loss = 0;

        if (ckpt_load_path) {
            if (mezo_load_checkpoint(ckpt_load_path, &start_step, &lr, &resume_loss, lw, rms_final, embed)) {
                printf("[RESUMED from step %d, loss=%.4f]\n", start_step, resume_loss);
                if (lr_from_cli) {
                    lr = base_lr;  // CLI --lr overrides checkpoint lr
                    printf("  (using CLI lr=%g instead of checkpoint lr)\n", lr);
                } else {
                    base_lr = lr;
                }
            } else {
                fprintf(stderr, "Failed to load checkpoint %s\n", ckpt_load_path);
                return 1;
            }
        } else if (from_scratch) {
            printf("Initializing from scratch (seed=%ld)\n", init_seed);
            srand48(init_seed);
            float scale_d = 1.0f / sqrtf(DIM), scale_qd = 1.0f / sqrtf(Q_DIM);
            float scale_h = 1.0f / sqrtf(HIDDEN);
            float res_scale = 1.0f / sqrtf(2.0f * NLAYERS);
            for (int L = 0; L < NLAYERS; L++) {
                for (size_t i = 0; i < WQ_SZ; i++) lw[L].Wq[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WK_SZ; i++) lw[L].Wk[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WV_SZ; i++) lw[L].Wv[i] = scale_d * (2 * drand48() - 1);
                for (size_t i = 0; i < WO_SZ; i++) lw[L].Wo[i] = scale_qd * res_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < W1_SZ; i++) lw[L].W1[i] = scale_h * (2 * drand48() - 1);
                for (size_t i = 0; i < W2_SZ; i++) lw[L].W2[i] = scale_d * res_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < W3_SZ; i++) lw[L].W3[i] = scale_h * (2 * drand48() - 1);
                for (int i = 0; i < DIM; i++) { lw[L].rms_att[i] = 1.0f; lw[L].rms_ffn[i] = 1.0f; }
            }
            for (int i = 0; i < DIM; i++) rms_final[i] = 1.0f;
            float escale = 0.02f;
            for (size_t i = 0; i < (size_t)VOCAB * DIM; i++) embed[i] = escale * (2 * drand48() - 1);
        } else {
            fprintf(stderr, "Must specify --scratch or --resume <path>\n");
            return 1;
        }

        // === LoRA initialization ===
        LoRALayer lora_layers[NLAYERS];
        (void)0;  // FFN LoRA uses same lora_tmp buffer (rank*SEQ)
        if (use_lora) {
            int r = lora_rank;
            float a_scale = 1.0f / sqrtf((float)r);
            srand48(init_seed + 12345);  // Separate seed for LoRA init
            size_t lora_params = 0;
            for (int L = 0; L < NLAYERS; L++) {
                lora_layers[L] = lora_layer_alloc(r, lora_ffn);
                // Copy base weights (frozen)
                memcpy(lora_layers[L].Wq_base, lw[L].Wq, WQ_SZ * 4);
                memcpy(lora_layers[L].Wk_base, lw[L].Wk, WK_SZ * 4);
                memcpy(lora_layers[L].Wv_base, lw[L].Wv, WV_SZ * 4);
                memcpy(lora_layers[L].Wo_base, lw[L].Wo, WO_SZ * 4);
                // Init A with small random, B with zero (LoRA starts as identity)
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Aq[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Ak[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].Av[i] = a_scale * (2 * drand48() - 1);
                for (size_t i = 0; i < (size_t)r * Q_DIM; i++) lora_layers[L].Ao[i] = a_scale * (2 * drand48() - 1);
                // B matrices stay zero (calloc)
                // Aq[r,DIM]+Bq[Q_DIM,r] + Ak[r,DIM]+Bk[KV_DIM,r] + Av[r,DIM]+Bv[KV_DIM,r] + Ao[r,Q_DIM]+Bo[DIM,r]
                lora_params += (size_t)r * DIM * 3 + (size_t)Q_DIM * r + (size_t)KV_DIM * r * 2 + (size_t)r * Q_DIM + (size_t)DIM * r;
                if (lora_ffn) {
                    memcpy(lora_layers[L].W1_base, lw[L].W1, W1_SZ * 4);
                    memcpy(lora_layers[L].W2_base, lw[L].W2, W2_SZ * 4);
                    memcpy(lora_layers[L].W3_base, lw[L].W3, W3_SZ * 4);
                    // A1[rank,DIM], B1[HIDDEN,rank], A3[rank,DIM], B3[HIDDEN,rank]
                    for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].A1[i] = a_scale * (2 * drand48() - 1);
                    for (size_t i = 0; i < (size_t)r * DIM; i++) lora_layers[L].A3[i] = a_scale * (2 * drand48() - 1);
                    // A2[rank,HIDDEN]
                    float a2_scale = 1.0f / sqrtf((float)r);
                    for (size_t i = 0; i < (size_t)r * HIDDEN; i++) lora_layers[L].A2[i] = a2_scale * (2 * drand48() - 1);
                    // B matrices stay zero (calloc)
                    // A1[r,DIM]+B1[HIDDEN,r] + A2[r,HIDDEN]+B2[DIM,r] + A3[r,DIM]+B3[HIDDEN,r]
                    lora_params += (size_t)r * DIM * 2 + (size_t)HIDDEN * r * 2 + (size_t)r * HIDDEN + (size_t)DIM * r;
                }
            }
            size_t rms_params = (size_t)NLAYERS * 2 * DIM + DIM;
            printf("LoRA: rank=%d, adapter params=%.1fK, trainable RMS params=%.1fK\n",
                   r, (float)lora_params / 1e3, (float)rms_params / 1e3);
            if (lora_ffn) printf("  Adapters on: Wq, Wk, Wv, Wo, W1, W2, W3 | Frozen: embed\n");
            else printf("  Adapters on: Wq, Wk, Wv, Wo | Frozen: W1, W2, W3, embed\n");
            printf("  Perturbation: LoRA A/B + RMS only (~%.1fK params vs %.1fM full)\n",
                   (float)(lora_params + rms_params) / 1e3, total_p / 1e6);
            if (lora_split) printf("  Mode: adapter-as-input (zero restaging, CPU-side LoRA correction)\n");
            // FFN LoRA split uses same lora_tmp buffer (allocated later)
        }

        // === P16 backprop-lora: allocate LoRA gradients + Adam state ===
        LoRAGrads lora_grads_arr[NLAYERS];
        LoRAAdam lora_adam_arr[NLAYERS];
        float *grms_att[NLAYERS], *grms_ffn[NLAYERS];
        float *grms_final = NULL;
        AdamState la_rms_att[NLAYERS], la_rms_ffn[NLAYERS];
        AdamState la_rms_final = {0};
        int adam_t_bp = 0;
        if (backprop_lora && use_lora) {
            for (int L = 0; L < NLAYERS; L++) {
                lora_grads_arr[L] = lora_grads_alloc(lora_rank);
                lora_adam_arr[L] = lora_adam_alloc(lora_rank);
                grms_att[L] = (float*)safe_calloc(DIM, 4);
                grms_ffn[L] = (float*)safe_calloc(DIM, 4);
                la_rms_att[L] = adam_alloc(DIM);
                la_rms_ffn[L] = adam_alloc(DIM);
            }
            grms_final = (float*)safe_calloc(DIM, 4);
            la_rms_final = adam_alloc(DIM);
            printf("P16: Allocated LoRA gradient + Adam state for backprop mode\n");
        }

        // === Sparse-HiZOO buffer allocation ===
        bool use_hizoo = (sparse_ratio > 0.0f || hessian_alpha > 0.0f) && lora_split;
        float *diag_hessian = NULL;
        uint8_t *sparse_mask = NULL;
        size_t hizoo_n_params = 0;
        if (use_hizoo) {
            hizoo_n_params = count_lora_params(lora_layers, NLAYERS);
            // Print parameter magnitude stats before masking
            {
                double lora_sum = 0; size_t lora_cnt = 0;
                double rms_sum = 0; size_t rms_cnt = 0;
                for (int L = 0; L < NLAYERS; L++) {
                    int r = lora_layers[L].rank;
                    size_t la_params = (size_t)r * DIM * 3 + (size_t)Q_DIM * r
                                     + (size_t)KV_DIM * r * 2 + (size_t)r * Q_DIM + (size_t)DIM * r;
                    // Sum LoRA magnitudes (rough sample: just Aq)
                    for (size_t j = 0; j < (size_t)r * DIM; j++) lora_sum += fabsf(lora_layers[L].Aq[j]);
                    lora_cnt += (size_t)r * DIM;
                    for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(lw[L].rms_att[j]);
                    for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(lw[L].rms_ffn[j]);
                    rms_cnt += DIM * 2;
                    (void)la_params;
                }
                for (size_t j = 0; j < DIM; j++) rms_sum += fabsf(rms_final[j]);
                rms_cnt += DIM;
                printf("  [Sparse-HiZOO] total_params=%zu  LoRA_mean_mag=%.6f  RMS_mean_mag=%.6f\n",
                       hizoo_n_params, lora_sum / fmax(1, lora_cnt), rms_sum / fmax(1, rms_cnt));
            }
            // Allocate diagonal Hessian (init to 1.0) + z_vals buffer for Gaussian z
            if (hessian_alpha > 0.0f) {
                diag_hessian = (float *)safe_malloc(hizoo_n_params * sizeof(float));
                for (size_t i = 0; i < hizoo_n_params; i++) diag_hessian[i] = 1.0f;
                printf("  Hessian buffer: %.1f MB (uses Gaussian z for per-element differentiation)\n",
                       hizoo_n_params * 4.0f / (1024*1024));
            }
            // Allocate and compute sparse mask
            if (sparse_ratio > 0.0f) {
                sparse_mask = (uint8_t *)safe_malloc(hizoo_n_params * sizeof(uint8_t));
                compute_sparse_mask(lora_layers, lw, rms_final, NLAYERS, sparse_mask, sparse_ratio, hizoo_n_params);
            }
        }

        // === mmap token data ===
        int data_fd = open(data_path, O_RDONLY);
        if (data_fd < 0) { fprintf(stderr, "Cannot open %s\n", data_path); return 1; }
        struct stat st; fstat(data_fd, &st);
        size_t data_len = st.st_size;
        if (data_len == 0 || data_len % 2 != 0) { fprintf(stderr, "FATAL: invalid data file\n"); close(data_fd); return 1; }
        uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (token_data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); close(data_fd); return 1; }
        size_t n_tokens = data_len / 2;
        for (size_t i = 0; i < n_tokens; i++) {
            if (token_data[i] >= VOCAB) {
                fprintf(stderr, "FATAL: token[%zu]=%d >= VOCAB=%d\n", i, token_data[i], VOCAB);
                return 1;
            }
        }
        size_t val_start = (size_t)(n_tokens * 0.9);
        size_t train_tokens = val_start;
        size_t val_tokens = n_tokens - val_start;
        printf("Tokens: %zu (train: %zu, val: %zu)\n", n_tokens, train_tokens, val_tokens);
        if (train_tokens < (size_t)SEQ + 2) {
            fprintf(stderr, "FATAL: training split too small (%zu tokens < SEQ+2=%d)\n",
                    train_tokens, SEQ + 2);
            return 1;
        }

        // Vocab compaction
        VocabMap vm = vocab_map_build(token_data, n_tokens, VOCAB);
        int CV = vm.compact_vocab;
        printf("Vocab compaction: %d -> %d active\n", VOCAB, CV);
        float *cembed = vocab_compact_embed(embed, &vm, DIM);

        // Residual scaling: DeepNet for from-scratch, standard (1.0) for pretrained
        // SmolLM2/Llama models use alpha=1.0; DeepNet scaling only valid when
        // weights are initialized with matching 1/sqrt(2L) scale on Wo/W2
        float res_alpha = from_scratch ? 1.0f / sqrtf(2.0f * NLAYERS) : 1.0f;

        // === Forward buffers (reused across layers, no per-layer caching) ===
        float *x_cur = (float*)safe_malloc(SEQ * DIM * 4);
        float *xnorm_buf = (float*)safe_malloc(SEQ * DIM * 4);
        float *Q = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *K = (float*)safe_malloc(SEQ * KV_DIM * 4);
        float *V = (float*)safe_malloc(SEQ * KV_DIM * 4);
        float *attn_out = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *o_out = (float*)safe_malloc(SEQ * DIM * 4);
        float *h1 = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *h3 = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *silu_out = (float*)safe_malloc(SEQ * HIDDEN * 4);
        float *k_tiled = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *v_tiled = (float*)safe_malloc(SEQ * Q_DIM * 4);
        float *logits = (float*)safe_malloc(SEQ * CV * 4);
        float *dlogits = (float*)safe_malloc(SEQ * CV * 4);  // throwaway for cross_entropy_loss
        // Temp buffer for lora_split mode: A @ x intermediate result [rank, SEQ]
        float *lora_tmp = NULL;
        if (lora_split) {
            lora_tmp = (float*)safe_malloc((size_t)lora_rank * SEQ * 4);
        }

        // === Compile ANE kernels (forward only) ===
        DynLayerKernels dk;
        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        memset(&dk, 0, sizeof(dk));
        memset(pls, 0, sizeof(pls));
        memset(plr, 0, sizeof(plr));

        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];

        if (!cpu_only && conv_fused) {
            // Conv-fused mode: QKV fused (3 convs → 1 kernel) + Wo conv + FFN fused (3 convs+SiLU+residual → 1 kernel)
            // 3 kernels/layer instead of 7, 96 total round-trips instead of 224
            printf("\nCompiling ANE conv-fused kernels...\n");
            uint64_t t_compile = mach_absolute_time();

            // Generate MIL templates (same MIL per shape, different baked weights per layer)
            NSString *mil_qkv = gen_qkv_fused_conv1x1_mil(DIM, Q_DIM, KV_DIM, SEQ);
            NSString *mil_wo_conv = gen_conv1x1_mil(Q_DIM, DIM, SEQ);
            NSString *mil_ffn = gen_ffn_fused_conv1x1_mil(DIM, HIDDEN, SEQ);

            for (int L = 0; L < NLAYERS; L++) {
                printf("  Layer %d: compiling 3 fused kernels (QKV + Wo + FFN)...\n", L);

                // QKV fused: Wq+Wk+Wv baked, input xnorm, output concat(Q,K,V)
                dk.qkvConv[L] = compile_qkv_fused_conv1x1_kern(mil_qkv,
                    lw[L].Wq, lw[L].Wk, lw[L].Wv, DIM, Q_DIM, KV_DIM, SEQ);
                if (!dk.qkvConv[L]) { fprintf(stderr, "qkvConv[%d] compile failed\n", L); return 1; }

                // Wo conv (unchanged from conv-hybrid): Q_DIM → DIM
                dk.woConv[L] = compile_conv1x1_kern(mil_wo_conv, lw[L].Wo, Q_DIM, DIM, SEQ);
                if (!dk.woConv[L]) { fprintf(stderr, "woConv[%d] compile failed\n", L); return 1; }

                // FFN fused: W1+W3+SiLU+W2+residual, input concat(xnorm,x_cur), output x_next
                dk.ffnConv[L] = compile_ffn_fused_conv1x1_kern(mil_ffn,
                    lw[L].W1, lw[L].W3, lw[L].W2, DIM, HIDDEN, SEQ);
                if (!dk.ffnConv[L]) { fprintf(stderr, "ffnConv[%d] compile failed\n", L); return 1; }

                // No matmul buffers needed — all projections use conv
                Wqt_buf[L] = NULL; Wkt_buf[L] = NULL; Wvt_buf[L] = NULL; Wot_buf[L] = NULL;
                W1t_buf[L] = NULL; W2t_buf[L] = NULL; W3t_buf[L] = NULL;
            }

            printf("Compiled %d kernels in %.0fms\n", g_compile_count,
                   tb_ms(mach_absolute_time() - t_compile));

        } else if (!cpu_only && conv_hybrid) {
            // Conv-hybrid mode: conv1x1 for Wq,Wo,W1,W2,W3 (baked weights, activation-only I/O)
            //                   matmul for Wk,Wv (960→320 is 2.8x slower with conv)
            printf("\nCompiling ANE conv-hybrid kernels...\n");
            uint64_t t_compile = mach_absolute_time();

            // Compile shared matmul kernel for Wk/Wv only (DIM→KV_DIM)
            printf("  Compiling wkvFwd matmul (DIM->KV_DIM)...\n");
            dk.wkvFwd = compile_kern_mil_w(gen_dyn_matmul_mil(DIM, KV_DIM, SEQ), @{},
                DIM*WKV_FWD_SP*2, KV_DIM*SEQ*2);
            if (!dk.wkvFwd) { fprintf(stderr, "wkvFwd compile failed\n"); return 1; }

            // Generate MIL templates for conv kernels (same MIL per shape, different baked weights)
            NSString *mil_wq_conv = gen_conv1x1_mil(DIM, Q_DIM, SEQ);
            NSString *mil_wo_conv = gen_conv1x1_mil(Q_DIM, DIM, SEQ);
            NSString *mil_w1_conv = gen_conv1x1_mil(DIM, HIDDEN, SEQ);
            NSString *mil_w2_conv = gen_conv1x1_mil(HIDDEN, DIM, SEQ);
            NSString *mil_w3_conv = gen_conv1x1_mil(DIM, HIDDEN, SEQ);

            for (int L = 0; L < NLAYERS; L++) {
                printf("  Layer %d: compiling 5 conv + matmul surfaces...\n", L);

                dk.wqConv[L] = compile_conv1x1_kern(mil_wq_conv, lw[L].Wq, DIM, Q_DIM, SEQ);
                if (!dk.wqConv[L]) { fprintf(stderr, "wqConv[%d] compile failed\n", L); return 1; }
                dk.woConv[L] = compile_conv1x1_kern(mil_wo_conv, lw[L].Wo, Q_DIM, DIM, SEQ);
                if (!dk.woConv[L]) { fprintf(stderr, "woConv[%d] compile failed\n", L); return 1; }
                dk.w1Conv[L] = compile_conv1x1_kern(mil_w1_conv, lw[L].W1, DIM, HIDDEN, SEQ);
                if (!dk.w1Conv[L]) { fprintf(stderr, "w1Conv[%d] compile failed\n", L); return 1; }
                dk.w2Conv[L] = compile_conv1x1_kern(mil_w2_conv, lw[L].W2, HIDDEN, DIM, SEQ);
                if (!dk.w2Conv[L]) { fprintf(stderr, "w2Conv[%d] compile failed\n", L); return 1; }
                dk.w3Conv[L] = compile_conv1x1_kern(mil_w3_conv, lw[L].W3, DIM, HIDDEN, SEQ);
                if (!dk.w3Conv[L]) { fprintf(stderr, "w3Conv[%d] compile failed\n", L); return 1; }

                Wkt_buf[L] = (float*)safe_malloc(WK_SZ * 4);
                Wvt_buf[L] = (float*)safe_malloc(WV_SZ * 4);
                Wqt_buf[L] = NULL; Wot_buf[L] = NULL;
                W1t_buf[L] = NULL; W2t_buf[L] = NULL; W3t_buf[L] = NULL;

                pls[L].wkFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                pls[L].wvFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                plr[L].wkFwd = make_request(dk.wkvFwd, pls[L].wkFwd_in);
                plr[L].wvFwd = make_request(dk.wkvFwd, pls[L].wvFwd_in);
            }

            printf("Compiled %d kernels in %.0fms\n", g_compile_count,
                   tb_ms(mach_absolute_time() - t_compile));

        } else if (!cpu_only) {
            printf("\nCompiling ANE forward kernels...\n");
            uint64_t t_compile = mach_absolute_time();
            if (!compile_dynamic_kernels(&dk, 1.0f / sqrtf(2.0f * NLAYERS), true, false)) {
                fprintf(stderr, "ANE kernel compilation failed\n"); return 1;
            }
            printf("Compiled %d kernels in %.0fms\n", g_compile_count,
                   tb_ms(mach_absolute_time() - t_compile));

            // Allocate transposed weight buffers + IOSurfaces
            for (int L = 0; L < NLAYERS; L++) {
                Wqt_buf[L] = (float*)safe_malloc(WQ_SZ * 4);
                Wkt_buf[L] = (float*)safe_malloc(WK_SZ * 4);
                Wvt_buf[L] = (float*)safe_malloc(WV_SZ * 4);
                Wot_buf[L] = (float*)safe_malloc(WO_SZ * 4);
                W1t_buf[L] = (float*)safe_malloc(W1_SZ * 4);
                W2t_buf[L] = (float*)safe_malloc(W2_SZ * 4);
                W3t_buf[L] = (float*)safe_malloc(W3_SZ * 4);

                // Per-layer IOSurfaces for unfused forward kernels
                pls[L].wqFwd_in = make_surface(DIM * WQ_FWD_SP * 2);
                pls[L].wkFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                pls[L].wvFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
                pls[L].woFwd_in = make_surface(Q_DIM * WO_FWD_SP * 2);
                pls[L].w1Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
                pls[L].w3Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
                pls[L].w2Fwd_in = make_surface(HIDDEN * W2_FWD_SP * 2);

                // Per-layer requests
                plr[L].wqFwd = make_request(dk.wqFwd, pls[L].wqFwd_in);
                plr[L].wkFwd = make_request(dk.wkvFwd, pls[L].wkFwd_in);
                plr[L].wvFwd = make_request(dk.wkvFwd, pls[L].wvFwd_in);
                plr[L].woFwd = make_request(dk.woFwd, pls[L].woFwd_in);
                plr[L].w1Fwd = make_request(dk.w13Fwd, pls[L].w1Fwd_in);
                plr[L].w3Fwd = make_request(dk.w13Fwd, pls[L].w3Fwd_in);
                plr[L].w2Fwd = make_request(dk.w2Fwd, pls[L].w2Fwd_in);
            }
        } else {
            for (int L = 0; L < NLAYERS; L++) {
                Wqt_buf[L] = NULL; Wkt_buf[L] = NULL; Wvt_buf[L] = NULL; Wot_buf[L] = NULL;
                W1t_buf[L] = NULL; W2t_buf[L] = NULL; W3t_buf[L] = NULL;
            }
            printf("CPU-only mode: skipping ANE kernel compilation\n");
        }

        // ===== Transpose + stage weights into IOSurfaces (ANE only) =====
        // Extracted as a function for reuse after perturbation
        // (inline for now — called 2x per step in ANE mode)
        #define RETRANSPOSE_AND_STAGE() do { \
            for (int L = 0; L < NLAYERS; L++) { \
                transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM); \
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM); \
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM); \
                transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM); \
                transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM); \
                transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN); \
                transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM); \
                { IOSurfaceLock(pls[L].wqFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wqFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt_buf[L] + d*Q_DIM, Q_DIM); \
                  IOSurfaceUnlock(pls[L].wqFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wkFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wkFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wkFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wvFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wvFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wvFwd_in, 0, NULL); } \
                stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]); \
                { IOSurfaceLock(pls[L].w1Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w1Fwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W1t_buf[L] + d*HIDDEN, HIDDEN); \
                  IOSurfaceUnlock(pls[L].w1Fwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].w3Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w3Fwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W3t_buf[L] + d*HIDDEN, HIDDEN); \
                  IOSurfaceUnlock(pls[L].w3Fwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].w2Fwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].w2Fwd_in); \
                  for (int h = 0; h < HIDDEN; h++) \
                      cvt_f32_f16(buf + h*W2_FWD_SP + SEQ, W2t_buf[L] + h*DIM, DIM); \
                  IOSurfaceUnlock(pls[L].w2Fwd_in, 0, NULL); } \
            } \
        } while(0)

        // Attention-only restage: for MeZO+LoRA, FFN weights (W1/W2/W3) never change
        #define RETRANSPOSE_ATTN_ONLY() do { \
            for (int L = 0; L < NLAYERS; L++) { \
                transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM); \
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM); \
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM); \
                transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM); \
                { IOSurfaceLock(pls[L].wqFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wqFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt_buf[L] + d*Q_DIM, Q_DIM); \
                  IOSurfaceUnlock(pls[L].wqFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wkFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wkFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wkFwd_in, 0, NULL); } \
                { IOSurfaceLock(pls[L].wvFwd_in, 0, NULL); \
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wvFwd_in); \
                  for (int d = 0; d < DIM; d++) \
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt_buf[L] + d*KV_DIM, KV_DIM); \
                  IOSurfaceUnlock(pls[L].wvFwd_in, 0, NULL); } \
                stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]); \
            } \
        } while(0)

        // Initial transpose + staging
        if (!cpu_only && conv_fused) {
            // Conv-fused: all weights baked at compile time — no staging needed
            printf("Initial weight staging complete (conv-fused: all weights baked)\n");
        } else if (!cpu_only && conv_hybrid) {
            // Conv-hybrid: only stage Wk/Wv matmul weights (frozen, one-time)
            for (int L = 0; L < NLAYERS; L++) {
                transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
                transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
                { IOSurfaceLock(pls[L].wkFwd_in, 0, NULL);
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wkFwd_in);
                  for (int d = 0; d < DIM; d++)
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt_buf[L] + d*KV_DIM, KV_DIM);
                  IOSurfaceUnlock(pls[L].wkFwd_in, 0, NULL); }
                { IOSurfaceLock(pls[L].wvFwd_in, 0, NULL);
                  _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(pls[L].wvFwd_in);
                  for (int d = 0; d < DIM; d++)
                      cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt_buf[L] + d*KV_DIM, KV_DIM);
                  IOSurfaceUnlock(pls[L].wvFwd_in, 0, NULL); }
            }
            printf("Initial weight staging complete (conv-hybrid: Wk/Wv matmul only)\n");
        } else if (!cpu_only) {
            RETRANSPOSE_AND_STAGE();
            printf("Initial weight staging complete\n");
        }

        // ===== Forward pass macro (shared by MeZO, FZOO, and validation) =====
        // DO_FORWARD_PASS(input_toks, ctargets_arr, loss_var):
        //   Runs full transformer forward pass and computes cross-entropy loss.
        //   input_toks: uint16_t* input token array (length SEQ)
        //   ctargets_arr: uint16_t* compact target token array (length SEQ)
        //   loss_var: float variable to store the resulting loss
        #define DO_FORWARD_PASS(input_toks, ctargets_arr, loss_var) do { \
            embed_lookup(x_cur, embed, input_toks, DIM, SEQ, VOCAB); \
            for (int L = 0; L < NLAYERS; L++) { \
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ); \
                if (cpu_only) { \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, Q, SEQ); \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, K, SEQ); \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, V, SEQ); \
                    if (lora_split) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM); \
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                    } \
                } else if (conv_fused) { \
                    /* QKV fused: single kernel, 1 eval instead of 3 */ \
                    io_write_conv_acts(dk.qkvConv[L]->ioIn, xnorm_buf, DIM, SEQ); \
                    ane_eval(dk.qkvConv[L]); \
                    /* Read concat(Q[Q_DIM], K[KV_DIM], V[KV_DIM]) and split */ \
                    { IOSurfaceLock(dk.qkvConv[L]->ioOut, kIOSurfaceLockReadOnly, NULL); \
                      _Float16 *qkv_buf = (_Float16*)IOSurfaceGetBaseAddress(dk.qkvConv[L]->ioOut); \
                      int qkv_ch = Q_DIM + 2*KV_DIM; \
                      cvt_f16_f32(Q, qkv_buf, Q_DIM * SEQ); \
                      cvt_f16_f32(K, qkv_buf + Q_DIM * SEQ, KV_DIM * SEQ); \
                      cvt_f16_f32(V, qkv_buf + (Q_DIM + KV_DIM) * SEQ, KV_DIM * SEQ); \
                      IOSurfaceUnlock(dk.qkvConv[L]->ioOut, kIOSurfaceLockReadOnly, NULL); } \
                    { LoRALayer *ll = &lora_layers[L]; \
                    lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM); \
                    lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                    lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); } \
                } else if (conv_hybrid) { \
                    io_write_conv_acts(dk.wqConv[L]->ioIn, xnorm_buf, DIM, SEQ); \
                    ane_eval(dk.wqConv[L]); \
                    io_read_dyn(dk.wqConv[L]->ioOut, Q, Q_DIM, SEQ); \
                    io_write_dyn_acts(pls[L].wkFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP); \
                    ane_eval_req(dk.wkvFwd, plr[L].wkFwd); \
                    io_read_dyn(dk.wkvFwd->ioOut, K, KV_DIM, SEQ); \
                    io_write_dyn_acts(pls[L].wvFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP); \
                    ane_eval_req(dk.wkvFwd, plr[L].wvFwd); \
                    io_read_dyn(dk.wkvFwd->ioOut, V, KV_DIM, SEQ); \
                    { LoRALayer *ll = &lora_layers[L]; \
                    lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM); \
                    lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                    lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); } \
                } else { \
                    io_write_dyn_acts(pls[L].wqFwd_in, xnorm_buf, DIM, SEQ, WQ_FWD_SP); \
                    ane_eval_req(dk.wqFwd, plr[L].wqFwd); \
                    io_read_dyn(dk.wqFwd->ioOut, Q, Q_DIM, SEQ); \
                    io_write_dyn_acts(pls[L].wkFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP); \
                    ane_eval_req(dk.wkvFwd, plr[L].wkFwd); \
                    io_read_dyn(dk.wkvFwd->ioOut, K, KV_DIM, SEQ); \
                    io_write_dyn_acts(pls[L].wvFwd_in, xnorm_buf, DIM, SEQ, WKV_FWD_SP); \
                    ane_eval_req(dk.wkvFwd, plr[L].wvFwd); \
                    io_read_dyn(dk.wkvFwd->ioOut, V, KV_DIM, SEQ); \
                    if (lora_split) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM); \
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM); \
                    } \
                } \
                rope_forward_inplace(Q, SEQ, Q_DIM, HD); \
                rope_forward_inplace(K, SEQ, KV_DIM, HD); \
                gqa_tile_kv(k_tiled, K, SEQ); \
                gqa_tile_kv(v_tiled, V, SEQ); \
                cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ); \
                if (cpu_only) { \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, attn_out, SEQ, 0.0f, o_out, SEQ); \
                    if (lora_split) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM); \
                    } \
                } else if (conv_hybrid) { \
                    io_write_conv_acts(dk.woConv[L]->ioIn, attn_out, Q_DIM, SEQ); \
                    ane_eval(dk.woConv[L]); \
                    io_read_dyn(dk.woConv[L]->ioOut, o_out, DIM, SEQ); \
                    { LoRALayer *ll = &lora_layers[L]; \
                    lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM); } \
                } else { \
                    write_wo_fwd_acts(pls[L].woFwd_in, attn_out); \
                    ane_eval_req(dk.woFwd, plr[L].woFwd); \
                    io_read_dyn(dk.woFwd->ioOut, o_out, DIM, SEQ); \
                    if (lora_split) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM); \
                    } \
                } \
                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM)); \
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ); \
                if (conv_fused) { \
                    /* FFN fused: single kernel does W1+W3+SiLU+W2+residual */ \
                    /* Input: concat(xnorm, x_cur) [1, DIM, 1, 2*SEQ] */ \
                    /* Output: x_next [1, DIM, 1, SEQ] — written directly to x_cur */ \
                    io_write_ffn_fused_conv_input(dk.ffnConv[L]->ioIn, xnorm_buf, x_cur, DIM, SEQ); \
                    ane_eval(dk.ffnConv[L]); \
                    io_read_dyn(dk.ffnConv[L]->ioOut, x_cur, DIM, SEQ); \
                } else { \
                if (cpu_only) { \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, h1, SEQ); \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, h3, SEQ); \
                    if (lora_split && lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                    } \
                } else if (conv_hybrid) { \
                    io_write_conv_acts(dk.w1Conv[L]->ioIn, xnorm_buf, DIM, SEQ); \
                    ane_eval(dk.w1Conv[L]); \
                    io_read_dyn(dk.w1Conv[L]->ioOut, h1, HIDDEN, SEQ); \
                    io_write_conv_acts(dk.w3Conv[L]->ioIn, xnorm_buf, DIM, SEQ); \
                    ane_eval(dk.w3Conv[L]); \
                    io_read_dyn(dk.w3Conv[L]->ioOut, h3, HIDDEN, SEQ); \
                    if (lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                    } \
                } else { \
                    io_write_dyn_acts(pls[L].w1Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP); \
                    ane_eval_req(dk.w13Fwd, plr[L].w1Fwd); \
                    io_read_dyn(dk.w13Fwd->ioOut, h1, HIDDEN, SEQ); \
                    io_write_dyn_acts(pls[L].w3Fwd_in, xnorm_buf, DIM, SEQ, W13_FWD_SP); \
                    ane_eval_req(dk.w13Fwd, plr[L].w3Fwd); \
                    io_read_dyn(dk.w13Fwd->ioOut, h3, HIDDEN, SEQ); \
                    if (lora_split && lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(h1, ll->A1, ll->B1, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                        lora_addmm(h3, ll->A3, ll->B3, xnorm_buf, lora_tmp, HIDDEN, ll->rank, DIM); \
                    } \
                } \
                for (int i = 0; i < HIDDEN * SEQ; i++) { \
                    float s = h1[i] / (1.0f + expf(-h1[i])); \
                    silu_out[i] = s * h3[i]; \
                } \
                if (cpu_only) { \
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                                DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, silu_out, SEQ, 0.0f, o_out, SEQ); \
                    if (lora_split && lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN); \
                    } \
                } else if (conv_hybrid) { \
                    io_write_conv_acts(dk.w2Conv[L]->ioIn, silu_out, HIDDEN, SEQ); \
                    ane_eval(dk.w2Conv[L]); \
                    io_read_dyn(dk.w2Conv[L]->ioOut, o_out, DIM, SEQ); \
                    if (lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN); \
                    } \
                } else { \
                    io_write_dyn_acts(pls[L].w2Fwd_in, silu_out, HIDDEN, SEQ, W2_FWD_SP); \
                    ane_eval_req(dk.w2Fwd, plr[L].w2Fwd); \
                    io_read_dyn(dk.w2Fwd->ioOut, o_out, DIM, SEQ); \
                    if (lora_split && lora_ffn) { \
                        LoRALayer *ll = &lora_layers[L]; \
                        lora_addmm(o_out, ll->A2, ll->B2, silu_out, lora_tmp, DIM, ll->rank, HIDDEN); \
                    } \
                } \
                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM)); \
                } /* end !conv_fused FFN */ \
            } \
            rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ); \
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, \
                        SEQ, CV, DIM, 1.0f, xnorm_buf, SEQ, cembed, DIM, 0.0f, logits, CV); \
            (loss_var) = cross_entropy_loss(dlogits, logits, ctargets_arr, CV, SEQ); \
        } while(0)

        // ===== Gradient Subspace Probe (diagnostic) =====
        // Measures gradient low-rankness via SVD of gradient estimate matrix
        if (probe_gradient > 0 && lora_split) {
            int h = probe_gradient;  // number of probe perturbations
            printf("\n=== Gradient Subspace Probe: h=%d perturbations ===\n", h);

            // Count total trainable params (LoRA + RMS)
            size_t total_params = 0;
            for (int L = 0; L < NLAYERS; L++) {
                int r = lora_layers[L].rank;
                total_params += (size_t)r * DIM * 3 + (size_t)Q_DIM * r + (size_t)KV_DIM * r * 2 + (size_t)r * Q_DIM + (size_t)DIM * r;
                if (lora_layers[L].has_ffn) {
                    total_params += (size_t)r * DIM * 2 + (size_t)HIDDEN * r * 2 + (size_t)r * HIDDEN + (size_t)DIM * r;
                }
                total_params += DIM * 2;  // rms_att + rms_ffn
            }
            total_params += DIM;  // rms_final
            printf("Total trainable params: %zu\n", total_params);

            // Allocate: G matrix [total_params × h] stored column-major
            // Plus z vector [total_params] for extracting perturbation direction
            float *G = (float*)safe_calloc(total_params * (size_t)h, 4);
            float *z_vec = (float*)safe_malloc(total_params * 4);

            // Sample one data batch for all probes (fixed data)
            size_t max_pos = train_tokens - SEQ - 1;
            srand48(init_seed);
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *probe_tokens = token_data + pos;
            uint16_t *probe_target = token_data + pos + 1;
            uint16_t probe_ct[SEQ];
            for (int t = 0; t < SEQ; t++) probe_ct[t] = (uint16_t)vm.full_to_compact[probe_target[t]];

            printf("Data batch: pos=%zu\n", pos);

            for (int hi = 0; hi < h; hi++) {
                uint64_t probe_seed = (uint64_t)hi * 1000003ULL + 77777ULL;

                // Perturb +epsilon
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, probe_seed, +epsilon);
                float loss_plus_p = 0;
                DO_FORWARD_PASS(probe_tokens, probe_ct, loss_plus_p);

                // Perturb -2*epsilon (to -epsilon)
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, probe_seed, -2.0f * epsilon);
                float loss_minus_p = 0;
                DO_FORWARD_PASS(probe_tokens, probe_ct, loss_minus_p);

                // Restore (perturb +epsilon)
                perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, probe_seed, +epsilon);

                // Scalar gradient estimate
                float pg = (loss_plus_p - loss_minus_p) / (2.0f * epsilon);

                // Extract z vector
                extract_lora_z(lora_layers, NLAYERS, probe_seed, z_vec);

                // g_hi = pg * z → store as column hi of G
                float *col = G + (size_t)hi * total_params;
                for (size_t j = 0; j < total_params; j++) {
                    col[j] = pg * z_vec[j];
                }

                printf("  Probe %d/%d: loss+=%.6f loss-=%.6f proj_grad=%.6f\n",
                       hi+1, h, loss_plus_p, loss_minus_p, pg);
            }
            free(z_vec);

            // ===== SVD of G [total_params × h] =====
            // We compute the economy SVD: G = U S V^T
            // Since total_params >> h, we use G^T G (h×h) eigendecomposition instead
            // G^T G = V S^2 V^T, then singular values = sqrt(eigenvalues)
            // This is O(total_params * h^2) instead of O(total_params^2 * h)
            printf("\nComputing SVD via G^T G (h×h = %d×%d)...\n", h, h);

            // Compute G^T G [h × h]
            float *GtG = (float*)safe_calloc((size_t)h * h, 4);
            // GtG[i,j] = sum_k G[k,i] * G[k,j] = dot(col_i, col_j)
            for (int i = 0; i < h; i++) {
                for (int j = i; j < h; j++) {
                    float dot = 0;
                    float *ci = G + (size_t)i * total_params;
                    float *cj = G + (size_t)j * total_params;
                    vDSP_dotpr(ci, 1, cj, 1, &dot, (vDSP_Length)total_params);
                    GtG[i * h + j] = dot;
                    GtG[j * h + i] = dot;  // symmetric
                }
            }

            // Eigendecomposition of GtG using LAPACK dsyev
            // GtG is overwritten with eigenvectors, eigenvalues in ascending order
            float *eigenvalues = (float*)safe_malloc(h * 4);
            {
                char jobz = 'V', uplo = 'U';
                __LAPACK_int n = h, lda = h, info = 0;
                __LAPACK_int lwork = -1;
                float work_query;
                ssyev_(&jobz, &uplo, &n, GtG, &lda, eigenvalues, &work_query, &lwork, &info);
                lwork = (__LAPACK_int)work_query;
                float *work = (float*)safe_malloc(lwork * 4);
                ssyev_(&jobz, &uplo, &n, GtG, &lda, eigenvalues, work, &lwork, &info);
                free(work);
                if (info != 0) { fprintf(stderr, "ssyev failed with info=%d\n", info); }
            }

            // Eigenvalues are in ascending order; singular values = sqrt(eigenvalues)
            // Print in descending order
            printf("\n=== Gradient Subspace SVD Spectrum ===\n");
            printf("%-6s  %-12s  %-12s  %-12s\n", "Rank", "Sing.Value", "Variance%", "Cumulative%");
            float total_var = 0;
            for (int i = 0; i < h; i++) {
                float ev = eigenvalues[i];
                if (ev < 0) ev = 0;  // numerical noise
                total_var += ev;
            }
            float cum_var = 0;
            for (int i = h - 1; i >= 0; i--) {
                float ev = eigenvalues[i];
                if (ev < 0) ev = 0;
                float sv = sqrtf(ev);
                float var_pct = (total_var > 0) ? 100.0f * ev / total_var : 0;
                cum_var += var_pct;
                printf("%-6d  %-12.6f  %-12.2f  %-12.2f\n", h - i, sv, var_pct, cum_var);
            }

            // Summary stats
            printf("\nTotal variance: %.6f\n", total_var);
            cum_var = 0;
            for (int i = h - 1; i >= 0; i--) {
                float ev = eigenvalues[i];
                if (ev < 0) ev = 0;
                cum_var += ev;
                float pct = 100.0f * cum_var / total_var;
                if (pct >= 90.0f) {
                    printf("90%% variance captured by top %d/%d directions\n", h - i, h);
                    break;
                }
            }
            cum_var = 0;
            for (int i = h - 1; i >= 0; i--) {
                float ev = eigenvalues[i];
                if (ev < 0) ev = 0;
                cum_var += ev;
                float pct = 100.0f * cum_var / total_var;
                if (pct >= 99.0f) {
                    printf("99%% variance captured by top %d/%d directions\n", h - i, h);
                    break;
                }
            }

            free(eigenvalues);
            free(GtG);
            free(G);
            printf("\n=== Gradient probe complete ===\n");
            return 0;  // Exit after probe
        }

        // P-GAP defaults and validation
        if (pgap_r > 0 && !lora_split) {
            fprintf(stderr, "ERROR: --pgap requires --lora-split\n");
            return 1;
        }

        // Faithful P-GAP state: per-matrix SVD bases for all LoRA matrices
        // Each layer has 8 attention LoRA matrices (Aq,Bq,Ak,Bk,Av,Bv,Ao,Bo)
        // Plus 2 RMS norm vectors (1D, perturbed with Gaussian, no SVD needed)
        // Plus rms_final (1D)
        int pgap_n_mats_per_layer = PGAP_ATTN_MATS;  // 8 for attention-only
        int pgap_total_mats = pgap_n_mats_per_layer * NLAYERS;
        PGAPMatrixBasis *pgap_bases = NULL;
        float *pgap_z_rms = NULL;       // Gaussian perturbation for RMS norms [NLAYERS*2*DIM + DIM]
        size_t pgap_rms_size = 0;
        bool pgap_basis_ready = false;
        int pgap_probe_count = 0;       // How many probes accumulated so far
        size_t pgap_d = 0;              // Total trainable params
        float *pgap_z_flat = NULL;      // Full flat perturbation vector [pgap_d] for perturb_lora_with_z

        if (pgap_r > 0 && lora_split) {
            pgap_d = count_lora_params(lora_layers, NLAYERS);
            pgap_rms_size = (size_t)NLAYERS * 2 * DIM + DIM;
            pgap_z_rms = (float*)safe_malloc(pgap_rms_size * 4);
            pgap_z_flat = (float*)safe_malloc(pgap_d * 4);

            // Allocate per-matrix bases
            pgap_bases = (PGAPMatrixBasis*)safe_calloc(pgap_total_mats, sizeof(PGAPMatrixBasis));
            size_t total_basis_mem = 0;
            for (int L = 0; L < NLAYERS; L++) {
                int r = lora_layers[L].rank;
                int base = L * PGAP_ATTN_MATS;
                // Matrix shapes: [rows, cols] for each LoRA matrix
                int shapes[][2] = {
                    {r, DIM},    // Aq
                    {Q_DIM, r},  // Bq
                    {r, DIM},    // Ak
                    {KV_DIM, r}, // Bk
                    {r, DIM},    // Av
                    {KV_DIM, r}, // Bv
                    {r, Q_DIM},  // Ao
                    {DIM, r},    // Bo
                };
                for (int mi = 0; mi < PGAP_ATTN_MATS; mi++) {
                    pgap_bases[base + mi] = pgap_basis_init(shapes[mi][0], shapes[mi][1], pgap_r);
                    PGAPMatrixBasis *b = &pgap_bases[base + mi];
                    total_basis_mem += (size_t)b->rows * b->cols * 4  // G
                        + (size_t)b->rows * b->svd_r * 4             // U
                        + (size_t)b->svd_r * 4                       // S
                        + (size_t)b->svd_r * b->cols * 4             // Vt
                        + (size_t)b->rows * b->cols * 4;             // Z_f
                }
            }
            printf("P-GAP (faithful): svd_r=%d, k=%d, h=%d, xi=%.1f, delta0=%.1f\n",
                   pgap_r, pgap_k, pgap_h, pgap_xi, pgap_delta0);
            printf("  %d per-matrix bases (%d layers × %d mats), total_params=%zu\n",
                   pgap_total_mats, NLAYERS, PGAP_ATTN_MATS, pgap_d);
            printf("  Basis memory: %.1f MB\n", (double)total_basis_mem / 1e6);
            // Paper hyperparameters for LoRA: epsilon=0.1, lr=1e-2..5e-2
            if (epsilon < 0.01f) {
                printf("  WARNING: P-GAP paper uses epsilon=0.1 for LoRA (current: %g)\n", epsilon);
            }
        }

        // ===== MeZO Training Loop =====
        float last_loss_plus = 999.0f, last_loss_minus = 999.0f;
        float best_loss = resume_loss > 0 ? resume_loss : 999.0f;
        double total_train_ms = 0;
        int total_steps_done = 0;
        uint64_t t_wall_start = mach_absolute_time();

        printf("\nStarting MeZO training...\n\n");

        for (int step = start_step; step < total_steps; step++) {
            // Time budget check
            if (time_budget_sec > 0 && step > start_step + 2) {
                double elapsed_sec = tb_ms(mach_absolute_time() - t_wall_start) / 1000.0;
                if (elapsed_sec >= time_budget_sec) {
                    printf("Time budget %.0fs reached at step %d (%.1fs elapsed)\n",
                           time_budget_sec, step, elapsed_sec);
                    total_steps = step;
                    break;
                }
            }

            uint64_t t_step = mach_absolute_time();
            double t_perturb = 0, t_fwd = 0, t_transpose = 0;

            // Sample data (from training split)
            size_t max_pos = train_tokens - SEQ - 1;
            srand48(init_seed + step * 7919LL);  // Deterministic data sampling per step
            size_t pos = (size_t)(drand48() * max_pos);
            uint16_t *input_tokens = token_data + pos;
            uint16_t *target_raw = token_data + pos + 1;
            uint16_t ctargets[SEQ];
            for (int t = 0; t < SEQ; t++) ctargets[t] = (uint16_t)vm.full_to_compact[target_raw[t]];

            // MeZO seed for this step
            uint64_t mezo_seed = (uint64_t)step * 1000003ULL + (uint64_t)init_seed;

            float loss_plus = 0, loss_minus = 0, proj_grad = 0;
            uint64_t t0;

            if (backprop_lora) {
                // ===== P16 HYBRID: ANE conv-fused forward + CPU fp32 backward =====
                // Forward: conv_fused ANE or CPU, saving activations per layer.
                // Backward: CPU fp32 dx chain + LoRA gradient projection.
                // Optimizer: Adam on LoRA A/B + RMS norms.
                double t_bp_fwd = 0, t_bp_bwd = 0, t_bp_opt = 0;

                // Allocate activation/work buffers (once on first step)
                static BP_LayerActs bp_acts[NLAYERS];
                static BP_WorkBufs bp_work;
                static bool bp_init_done = false;
                if (!bp_init_done) {
                    for (int L = 0; L < NLAYERS; L++) bp_acts[L] = bp_layer_acts_alloc();
                    bp_work = bp_work_alloc();
                    bp_init_done = true;
                    size_t act_mem = (size_t)NLAYERS * ((size_t)DIM*SEQ*5 + (size_t)Q_DIM*SEQ*2 + (size_t)KV_DIM*SEQ*2 + (size_t)HIDDEN*SEQ*3) * 4;
                    size_t work_mem = ((size_t)DIM*SEQ*5 + (size_t)Q_DIM*SEQ*4 + (size_t)KV_DIM*SEQ*2 + (size_t)HIDDEN*SEQ*4) * 4;
                    printf("P16: Allocated %.0f MB activations + %.1f MB work buffers\n",
                           act_mem/1e6, work_mem/1e6);
                }

                // ===== FORWARD PASS (with activation saves) =====
                t0 = mach_absolute_time();
                embed_lookup(x_cur, embed, input_tokens, DIM, SEQ, VOCAB);
                for (int L = 0; L < NLAYERS; L++) {
                    BP_LayerActs *ac = &bp_acts[L];
                    memcpy(ac->x_pre, x_cur, (size_t)DIM * SEQ * 4);
                    rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                    memcpy(ac->xnorm, xnorm_buf, (size_t)DIM * SEQ * 4);
                    // QKV: cpu_only or conv_fused
                    if (cpu_only) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    Q_DIM, SEQ, DIM, 1.0f, lw[L].Wq, DIM, xnorm_buf, SEQ, 0.0f, Q, SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    KV_DIM, SEQ, DIM, 1.0f, lw[L].Wk, DIM, xnorm_buf, SEQ, 0.0f, K, SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    KV_DIM, SEQ, DIM, 1.0f, lw[L].Wv, DIM, xnorm_buf, SEQ, 0.0f, V, SEQ);
                    } else if (conv_fused) {
                        io_write_conv_acts(dk.qkvConv[L]->ioIn, xnorm_buf, DIM, SEQ);
                        ane_eval(dk.qkvConv[L]);
                        { IOSurfaceLock(dk.qkvConv[L]->ioOut, kIOSurfaceLockReadOnly, NULL);
                          _Float16 *qkv_buf = (_Float16*)IOSurfaceGetBaseAddress(dk.qkvConv[L]->ioOut);
                          cvt_f16_f32(Q, qkv_buf, Q_DIM * SEQ);
                          cvt_f16_f32(K, qkv_buf + Q_DIM * SEQ, KV_DIM * SEQ);
                          cvt_f16_f32(V, qkv_buf + (Q_DIM + KV_DIM) * SEQ, KV_DIM * SEQ);
                          IOSurfaceUnlock(dk.qkvConv[L]->ioOut, kIOSurfaceLockReadOnly, NULL); }
                    }
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(Q, ll->Aq, ll->Bq, xnorm_buf, lora_tmp, Q_DIM, ll->rank, DIM);
                        lora_addmm(K, ll->Ak, ll->Bk, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                        lora_addmm(V, ll->Av, ll->Bv, xnorm_buf, lora_tmp, KV_DIM, ll->rank, DIM);
                    }
                    memcpy(ac->Q, Q, (size_t)Q_DIM * SEQ * 4);
                    memcpy(ac->K, K, (size_t)KV_DIM * SEQ * 4);
                    memcpy(ac->V, V, (size_t)KV_DIM * SEQ * 4);
                    rope_forward_inplace(Q, SEQ, Q_DIM, HD);
                    rope_forward_inplace(K, SEQ, KV_DIM, HD);
                    gqa_tile_kv(k_tiled, K, SEQ);
                    gqa_tile_kv(v_tiled, V, SEQ);
                    cpu_sdpa_forward(Q, k_tiled, v_tiled, attn_out, HEADS, HD, SEQ);
                    memcpy(ac->attn_out, attn_out, (size_t)Q_DIM * SEQ * 4);
                    // Wo
                    if (cpu_only) {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    DIM, SEQ, Q_DIM, 1.0f, lw[L].Wo, Q_DIM, attn_out, SEQ, 0.0f, o_out, SEQ);
                    } else if (conv_fused) {
                        io_write_conv_acts(dk.woConv[L]->ioIn, attn_out, Q_DIM, SEQ);
                        ane_eval(dk.woConv[L]);
                        io_read_dyn(dk.woConv[L]->ioOut, o_out, DIM, SEQ);
                    }
                    if (lora_split) {
                        LoRALayer *ll = &lora_layers[L];
                        lora_addmm(o_out, ll->Ao, ll->Bo, attn_out, lora_tmp, DIM, ll->rank, Q_DIM);
                    }
                    vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
                    memcpy(ac->x2, x_cur, (size_t)DIM * SEQ * 4);
                    rmsnorm(xnorm_buf, x_cur, lw[L].rms_ffn, DIM, SEQ);
                    memcpy(ac->x2norm, xnorm_buf, (size_t)DIM * SEQ * 4);
                    // FFN
                    if (conv_fused) {
                        io_write_ffn_fused_conv_input(dk.ffnConv[L]->ioIn, xnorm_buf, x_cur, DIM, SEQ);
                        ane_eval(dk.ffnConv[L]);
                        // Recompute h1/h3/silu on CPU for backward (conv_fused doesn't expose them)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, ac->h1, SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, ac->h3, SEQ);
                        for (int i = 0; i < HIDDEN * SEQ; i++) {
                            float s = ac->h1[i] / (1.0f + expf(-ac->h1[i]));
                            ac->silu_out[i] = s * ac->h3[i];
                        }
                        io_read_dyn(dk.ffnConv[L]->ioOut, x_cur, DIM, SEQ);
                    } else {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    HIDDEN, SEQ, DIM, 1.0f, lw[L].W1, DIM, xnorm_buf, SEQ, 0.0f, h1, SEQ);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    HIDDEN, SEQ, DIM, 1.0f, lw[L].W3, DIM, xnorm_buf, SEQ, 0.0f, h3, SEQ);
                        memcpy(ac->h1, h1, (size_t)HIDDEN * SEQ * 4);
                        memcpy(ac->h3, h3, (size_t)HIDDEN * SEQ * 4);
                        for (int i = 0; i < HIDDEN * SEQ; i++) {
                            float s = h1[i] / (1.0f + expf(-h1[i]));
                            silu_out[i] = s * h3[i];
                        }
                        memcpy(ac->silu_out, silu_out, (size_t)HIDDEN * SEQ * 4);
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                    DIM, SEQ, HIDDEN, 1.0f, lw[L].W2, HIDDEN, silu_out, SEQ, 0.0f, o_out, SEQ);
                        vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x_cur, 1, (vDSP_Length)(SEQ * DIM));
                    }
                }
                // Save x_cur (post-last-layer) for final RMSNorm backward
                float *x_pre_final = (float*)safe_malloc((size_t)DIM * SEQ * 4);
                memcpy(x_pre_final, x_cur, (size_t)DIM * SEQ * 4);
                rmsnorm(xnorm_buf, x_cur, rms_final, DIM, SEQ);
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans,
                            SEQ, CV, DIM, 1.0f, xnorm_buf, SEQ, cembed, DIM, 0.0f, logits, CV);
                float train_loss = cross_entropy_loss(dlogits, logits, ctargets, CV, SEQ);
                t_bp_fwd = tb_ms(mach_absolute_time() - t0);
                loss_plus = train_loss;

                // ===== BACKWARD PASS =====
                t0 = mach_absolute_time();
                bp_lora_backward(dlogits, x_pre_final, cembed, rms_final,
                                 lw, bp_acts, lora_layers, lora_grads_arr,
                                 grms_att, grms_ffn, grms_final,
                                 res_alpha, &bp_work, lora_rank, CV);
                free(x_pre_final);
                t_bp_bwd = tb_ms(mach_absolute_time() - t0);

                // ===== OPTIMIZER (Adam on LoRA A/B + RMS norms) =====
                t0 = mach_absolute_time();
                adam_t_bp++;
                float adam_b1 = 0.9f, adam_b2 = 0.95f, adam_eps = 1e-8f;
                float wd = 0.0f;
                for (int L = 0; L < NLAYERS; L++) {
                    LoRALayer *ll = &lora_layers[L];
                    LoRAGrads *lg = &lora_grads_arr[L];
                    LoRAAdam *la_l = &lora_adam_arr[L];
                    adam_update(ll->Aq, lg->Aq, &la_l->Aq, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Bq, lg->Bq, &la_l->Bq, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Ak, lg->Ak, &la_l->Ak, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Bk, lg->Bk, &la_l->Bk, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Av, lg->Av, &la_l->Av, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Bv, lg->Bv, &la_l->Bv, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Ao, lg->Ao, &la_l->Ao, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    adam_update(ll->Bo, lg->Bo, &la_l->Bo, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, wd);
                    lora_merge_weight(lw[L].Wq, ll->Wq_base, ll->Bq, ll->Aq, Q_DIM, lora_rank, DIM);
                    lora_merge_weight(lw[L].Wk, ll->Wk_base, ll->Bk, ll->Ak, KV_DIM, lora_rank, DIM);
                    lora_merge_weight(lw[L].Wv, ll->Wv_base, ll->Bv, ll->Av, KV_DIM, lora_rank, DIM);
                    lora_merge_weight(lw[L].Wo, ll->Wo_base, ll->Bo, ll->Ao, DIM, lora_rank, Q_DIM);
                    adam_update(lw[L].rms_att, grms_att[L], &la_rms_att[L], adam_t_bp, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    adam_update(lw[L].rms_ffn, grms_ffn[L], &la_rms_ffn[L], adam_t_bp, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                    lora_grads_zero(&lora_grads_arr[L], lora_rank);
                    memset(grms_att[L], 0, DIM * 4);
                    memset(grms_ffn[L], 0, DIM * 4);
                }
                adam_update(rms_final, grms_final, &la_rms_final, adam_t_bp, lr, adam_b1, adam_b2, adam_eps, 0.0f);
                memset(grms_final, 0, DIM * 4);
                t_bp_opt = tb_ms(mach_absolute_time() - t0);

                // Logging
                double step_ms = tb_ms(mach_absolute_time() - t_step);
                if (step % 10 == 0 || step == start_step) {
                    printf("step %d  loss=%.4f  lr=%.2e  %.1fms/step (fwd=%.0f bwd=%.0f opt=%.0f)\n",
                           step, train_loss, lr, step_ms, t_bp_fwd, t_bp_bwd, t_bp_opt);
                }

                // LR schedule: cosine decay
                float progress = (float)(step - start_step) / (float)(total_steps - start_step);
                lr = base_lr * 0.5f * (1.0f + cosf(3.14159f * progress));

            } else if (fzoo_K > 0) {
                // ===== FZOO: Multi-perturbation one-sided gradient estimation =====
                // Step 1: Unperturbed forward pass -> loss_0
                t0 = mach_absolute_time();
                float fzoo_losses[fzoo_K + 1];  // losses[0]=loss_0, losses[1..K]=perturbed
                DO_FORWARD_PASS(input_tokens, ctargets, fzoo_losses[0]);
                t_fwd += tb_ms(mach_absolute_time() - t0);

                // Step 2: K perturbed forward passes
                for (int fk = 0; fk < fzoo_K; fk++) {
                    uint64_t mezo_seed_k = mezo_seed + (uint64_t)fk * 999983ULL;

                    // Perturb +epsilon with direction k
                    t0 = mach_absolute_time();
                    if (use_lora) {
                        perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed_k, +epsilon);
                        if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                    } else {
                        perturb_all_weights(lw, embed, rms_final, mezo_seed_k, +epsilon);
                    }
                    if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
                    t_perturb += tb_ms(mach_absolute_time() - t0);

                    if (!cpu_only && !lora_split) {
                        t0 = mach_absolute_time();
                        if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                        else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                        else { RETRANSPOSE_AND_STAGE(); }
                        t_transpose += tb_ms(mach_absolute_time() - t0);
                    }

                    // Forward pass -> loss_k
                    t0 = mach_absolute_time();
                    DO_FORWARD_PASS(input_tokens, ctargets, fzoo_losses[fk + 1]);
                    t_fwd += tb_ms(mach_absolute_time() - t0);

                    // Restore: perturb -epsilon with same seed
                    t0 = mach_absolute_time();
                    if (use_lora) {
                        perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed_k, -epsilon);
                        if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                    } else {
                        perturb_all_weights(lw, embed, rms_final, mezo_seed_k, -epsilon);
                    }
                    if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
                    t_perturb += tb_ms(mach_absolute_time() - t0);
                }

                // Step 3: Compute sigma = std(loss_0, loss_1, ..., loss_K) for adaptive step size
                float fzoo_mean = 0;
                for (int fk = 0; fk <= fzoo_K; fk++) fzoo_mean += fzoo_losses[fk];
                fzoo_mean /= (float)(fzoo_K + 1);
                float fzoo_var = 0;
                for (int fk = 0; fk <= fzoo_K; fk++) {
                    float d = fzoo_losses[fk] - fzoo_mean;
                    fzoo_var += d * d;
                }
                float fzoo_sigma = sqrtf(fzoo_var / (float)(fzoo_K + 1));

                // Step 4: Accumulate K gradient updates
                float loss_avg = 0;
                float proj_grad_sum = 0;
                t0 = mach_absolute_time();
                for (int fk = 0; fk < fzoo_K; fk++) {
                    uint64_t mezo_seed_k = mezo_seed + (uint64_t)fk * 999983ULL;
                    float grad_k = (fzoo_losses[fk + 1] - fzoo_losses[0]) / epsilon;
                    proj_grad_sum += grad_k;
                    loss_avg += fzoo_losses[fk + 1];

                    float update_scale_k = -lr * grad_k / (float)fzoo_K;
                    if (use_lora) {
                        perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed_k, update_scale_k);
                    } else {
                        perturb_all_weights(lw, embed, rms_final, mezo_seed_k, update_scale_k);
                    }
                }
                if (!lora_split) {
                    if (use_lora) lora_merge_all(lw, lora_layers, NLAYERS);
                }
                t_perturb += tb_ms(mach_absolute_time() - t0);
                loss_avg /= (float)fzoo_K;
                proj_grad = proj_grad_sum / (float)fzoo_K;

                // Re-build compact embedding after weight update
                if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }

                // Defer re-transpose for validation
                if (!cpu_only && !conv_hybrid && val_every > 0 && (step + 1) % val_every == 0 && val_tokens > SEQ + 1) {
                    t0 = mach_absolute_time();
                    if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                    else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                    else { RETRANSPOSE_AND_STAGE(); }
                    t_transpose += tb_ms(mach_absolute_time() - t0);
                }

                // Set loss_plus/loss_minus for compatibility with reporting
                loss_plus = fzoo_losses[0];
                loss_minus = loss_avg;  // use loss_avg for the "minus" slot

                // LR schedule (cosine decay, no warmup)
                float min_lr = base_lr * 0.1f;
                float decay = (float)(step - start_step) / (float)(total_steps - start_step);
                lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay)) * (base_lr - min_lr);

                double step_ms = tb_ms(mach_absolute_time() - t_step);
                total_train_ms += step_ms;
                total_steps_done++;
                last_loss_plus = loss_plus;
                last_loss_minus = loss_minus;

                // Log FZOO
                if (step % 100 == 0 || step == start_step) {
                    printf("step %d  fzoo_K=%d  loss_0=%.4f  loss_avg=%.4f  sigma=%.4f  "
                           "proj_grad=%.6f  lr=%.2e  step_ms=%.0f\n",
                           step, fzoo_K, fzoo_losses[0], loss_avg, fzoo_sigma,
                           proj_grad, lr, step_ms);
                }

            } else if (pgap_r > 0 && pgap_basis_ready) {
                // ===== Faithful P-GAP: Per-matrix projected perturbation step =====
                // For each LoRA matrix: Z_init ~ N(0, I_{r×r}), PROJECTION, Z_f = U×Z×V^T
                // For RMS norms: z ~ N(0, I)
                // Concatenate all into flat z vector, SPSA with epsilon

                t0 = mach_absolute_time();
                // Compute delta for this step (linear decay from delta0 to 0)
                float pgap_delta = pgap_delta0 * (1.0f - (float)(step - start_step) / (float)(total_steps - start_step));
                if (pgap_delta < 0) pgap_delta = 0;

                // Generate per-matrix perturbations
                xo_seed(mezo_seed + 31337ULL);
                for (int L = 0; L < NLAYERS; L++) {
                    int base = L * PGAP_ATTN_MATS;
                    for (int mi = 0; mi < PGAP_ATTN_MATS; mi++) {
                        PGAPMatrixBasis *b = &pgap_bases[base + mi];
                        int r = b->svd_r;
                        // Z_init ~ N(0, I_{r×r})
                        float Z_init[64 * 64];  // max svd_r=8, so 8×8=64 elements
                        gaussian_fill(Z_init, (size_t)r * r);
                        // PROJECTION constraint
                        pgap_project(Z_init, b->S, r, pgap_xi, pgap_delta);
                        // Z_f = U × Z × V^T
                        pgap_gen_perturbation(b, Z_init);
                    }
                }
                // Generate Gaussian perturbation for RMS norms
                gaussian_fill(pgap_z_rms, pgap_rms_size);

                // Assemble flat z vector from per-matrix Z_f + RMS z
                {
                    size_t off = 0;
                    for (int L = 0; L < NLAYERS; L++) {
                        int base = L * PGAP_ATTN_MATS;
                        for (int mi = 0; mi < PGAP_ATTN_MATS; mi++) {
                            PGAPMatrixBasis *b = &pgap_bases[base + mi];
                            size_t sz = (size_t)b->rows * b->cols;
                            memcpy(pgap_z_flat + off, b->Z_f, sz * 4);
                            off += sz;
                        }
                        // rms_att, rms_ffn
                        memcpy(pgap_z_flat + off, pgap_z_rms + (size_t)L * 2 * DIM, DIM * 4);
                        off += DIM;
                        memcpy(pgap_z_flat + off, pgap_z_rms + (size_t)L * 2 * DIM + DIM, DIM * 4);
                        off += DIM;
                    }
                    // rms_final
                    memcpy(pgap_z_flat + off, pgap_z_rms + (size_t)NLAYERS * 2 * DIM, DIM * 4);
                    off += DIM;
                }
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 1. Perturb +epsilon using z_flat
                t0 = mach_absolute_time();
                perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, +epsilon);
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 2. Forward pass -> loss_plus
                t0 = mach_absolute_time();
                DO_FORWARD_PASS(input_tokens, ctargets, loss_plus);
                t_fwd += tb_ms(mach_absolute_time() - t0);

                // 3. Perturb -2*epsilon
                t0 = mach_absolute_time();
                perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, -2.0f * epsilon);
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 4. Forward pass -> loss_minus
                t0 = mach_absolute_time();
                DO_FORWARD_PASS(input_tokens, ctargets, loss_minus);
                t_fwd += tb_ms(mach_absolute_time() - t0);

                // 5. Restore to original theta
                t0 = mach_absolute_time();
                perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, +epsilon);
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 6. Gradient estimate (scalar) + update using same z_flat
                proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
                float update_scale = -lr * proj_grad;

                t0 = mach_absolute_time();
                perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, update_scale);
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 7. LR schedule
                float min_lr = base_lr * 0.1f;
                float decay_frac = (float)(step - start_step) / (float)(total_steps - start_step);
                lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay_frac)) * (base_lr - min_lr);

                double step_ms = tb_ms(mach_absolute_time() - t_step);
                total_train_ms += step_ms;
                total_steps_done++;
                last_loss_plus = loss_plus;
                last_loss_minus = loss_minus;

                // 8. Log
                if (step % 100 == 0 || step == start_step) {
                    printf("step %d  [P-GAP proj] loss+=%.4f  loss-=%.4f  proj_grad=%.6f  "
                           "lr=%.2e  delta=%.3f  step_ms=%.0f (fwd=%.0f perturb=%.0f)\n",
                           step, loss_plus, loss_minus, proj_grad, lr, pgap_delta,
                           step_ms, t_fwd, t_perturb);
                }

            } else {
                // ===== Standard MeZO: Central-difference gradient estimation =====
                // Also handles P-GAP collection steps (unprojected, stores gradient for basis)
                // When use_hizoo is true, uses perturb_lora_hizoo (1-bit PRNG + sparse mask + Hessian)

                // 0. L0 forward pass (HiZOO only: needed for curvature estimate)
                float loss_0 = 0.0f;
                if (hessian_alpha > 0.0f) {
                    t0 = mach_absolute_time();
                    DO_FORWARD_PASS(input_tokens, ctargets, loss_0);
                    t_fwd += tb_ms(mach_absolute_time() - t0);
                }

                // Allocate z_vals on stack for Hessian update (Gaussian z[i]² varies per element)
                // Only needed when hessian_alpha > 0 (HiZOO mode uses Gaussian z)
                float *z_vals = NULL;
                if (hessian_alpha > 0.0f) {
                    z_vals = (float *)safe_malloc(hizoo_n_params * sizeof(float));
                }

                // 1. Perturb +epsilon (store z values for Hessian update)
                t0 = mach_absolute_time();
                if (use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon, sparse_mask, diag_hessian, z_vals);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon);
                    if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
                }
                if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
                t_perturb += tb_ms(mach_absolute_time() - t0);

                if (!cpu_only && !lora_split) {
                    t0 = mach_absolute_time();
                    if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                    else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                    else { RETRANSPOSE_AND_STAGE(); }
                    t_transpose += tb_ms(mach_absolute_time() - t0);
                }

                // 2. Forward pass -> loss_plus
                t0 = mach_absolute_time();
                DO_FORWARD_PASS(input_tokens, ctargets, loss_plus);
                t_fwd += tb_ms(mach_absolute_time() - t0);

                // 3. Perturb -2*epsilon (to theta - epsilon*z)
                t0 = mach_absolute_time();
                if (use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed, -2.0f * epsilon, sparse_mask, diag_hessian, NULL);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, -2.0f * epsilon);
                    if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, -2.0f * epsilon);
                }
                if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }
                t_perturb += tb_ms(mach_absolute_time() - t0);

                if (!cpu_only && !lora_split) {
                    t0 = mach_absolute_time();
                    if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                    else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                    else { RETRANSPOSE_AND_STAGE(); }
                    t_transpose += tb_ms(mach_absolute_time() - t0);
                }

                // 4. Forward pass -> loss_minus
                t0 = mach_absolute_time();
                DO_FORWARD_PASS(input_tokens, ctargets, loss_minus);
                t_fwd += tb_ms(mach_absolute_time() - t0);

                // 5. Restore to original theta
                t0 = mach_absolute_time();
                if (use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon, sparse_mask, diag_hessian, NULL);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, +epsilon);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, +epsilon);
                }
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // 6. Gradient estimate + Hessian update + update step
                proj_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

                // Hessian EMA update (before weight update) — uses z_vals with Gaussian z[i]²
                if (hessian_alpha > 0.0f && diag_hessian && z_vals) {
                    update_hessian(diag_hessian, z_vals, hizoo_n_params, loss_plus, loss_minus, loss_0, epsilon, hessian_alpha);
                    if (step % 100 == 0 || step <= start_step + 10)
                        print_hessian_stats(diag_hessian, hizoo_n_params, step);
                }

                float update_scale = -lr * proj_grad;

                t0 = mach_absolute_time();
                if (use_hizoo) {
                    perturb_lora_hizoo(lora_layers, lw, rms_final, NLAYERS, mezo_seed, update_scale, sparse_mask, diag_hessian, NULL);
                } else if (use_lora) {
                    perturb_lora_weights(lora_layers, lw, rms_final, NLAYERS, mezo_seed, update_scale);
                    if (!lora_split) lora_merge_all(lw, lora_layers, NLAYERS);
                } else {
                    perturb_all_weights(lw, embed, rms_final, mezo_seed, update_scale);
                }
                t_perturb += tb_ms(mach_absolute_time() - t0);

                // Re-build compact embedding after weight update
                if (!lora_split) { free(cembed); cembed = vocab_compact_embed(embed, &vm, DIM); }

                // Free z_vals buffer (allocated per step for Hessian update)
                if (z_vals) { free(z_vals); z_vals = NULL; }

                // Sparse mask refresh
                if (sparse_ratio > 0.0f && sparse_mask && step > start_step && (step % mask_refresh) == 0)
                    compute_sparse_mask(lora_layers, lw, rms_final, NLAYERS, sparse_mask, sparse_ratio, hizoo_n_params);

                // Faithful P-GAP probe phase: every pgap_k steps, run h probes
                // to estimate gradient and build per-matrix SVD bases.
                // Probes happen on the SAME data batch as the current step.
                if (pgap_r > 0 && lora_split && (step % pgap_k) == 0 && step > start_step) {
                    t0 = mach_absolute_time();
                    printf("  [P-GAP probe phase at step %d: h=%d probes...]\n", step, pgap_h);

                    // Zero all G matrices
                    for (int mi = 0; mi < pgap_total_mats; mi++) {
                        memset(pgap_bases[mi].G, 0, (size_t)pgap_bases[mi].rows * pgap_bases[mi].cols * 4);
                    }

                    // Run h probe perturbations (Gaussian, per paper)
                    for (int hi = 0; hi < pgap_h; hi++) {
                        uint64_t probe_seed = mezo_seed + (uint64_t)(hi + 1) * 7777777ULL;

                        // Generate Gaussian perturbation and extract z vector
                        xo_seed(probe_seed);
                        size_t off = 0;
                        for (int L = 0; L < NLAYERS; L++) {
                            int base_idx = L * PGAP_ATTN_MATS;
                            for (int mi = 0; mi < PGAP_ATTN_MATS; mi++) {
                                PGAPMatrixBasis *b = &pgap_bases[base_idx + mi];
                                size_t sz = (size_t)b->rows * b->cols;
                                gaussian_fill(pgap_z_flat + off, sz);
                                off += sz;
                            }
                            // RMS norms: Gaussian
                            gaussian_fill(pgap_z_flat + off, DIM); off += DIM;
                            gaussian_fill(pgap_z_flat + off, DIM); off += DIM;
                        }
                        gaussian_fill(pgap_z_flat + off, DIM); off += DIM;

                        // Perturb +epsilon
                        perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, +epsilon);

                        // Forward pass -> loss_probe_plus
                        float loss_probe_plus = 0;
                        DO_FORWARD_PASS(input_tokens, ctargets, loss_probe_plus);

                        // Perturb -2*epsilon
                        perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, -2.0f * epsilon);

                        // Forward pass -> loss_probe_minus
                        float loss_probe_minus = 0;
                        DO_FORWARD_PASS(input_tokens, ctargets, loss_probe_minus);

                        // Restore
                        perturb_lora_with_z(lora_layers, lw, rms_final, NLAYERS, pgap_z_flat, +epsilon);

                        // Scalar gradient estimate
                        float rho = (loss_probe_plus - loss_probe_minus) / (2.0f * epsilon);

                        // Accumulate per-matrix: G_l += (rho / h) * Q_l
                        float scale_rho = rho / (float)pgap_h;
                        off = 0;
                        for (int L = 0; L < NLAYERS; L++) {
                            int base_idx = L * PGAP_ATTN_MATS;
                            for (int mi = 0; mi < PGAP_ATTN_MATS; mi++) {
                                PGAPMatrixBasis *b = &pgap_bases[base_idx + mi];
                                size_t sz = (size_t)b->rows * b->cols;
                                cblas_saxpy((int)sz, scale_rho, pgap_z_flat + off, 1, b->G, 1);
                                off += sz;
                            }
                            off += DIM * 2;  // skip RMS
                        }
                        off += DIM;  // skip rms_final

                        if (hi == 0 || hi == pgap_h - 1) {
                            printf("    probe %d/%d: loss+=%.4f loss-=%.4f rho=%.6f\n",
                                   hi+1, pgap_h, loss_probe_plus, loss_probe_minus, rho);
                        }
                    }

                    // SVD each per-matrix G to get basis (U, S, Vt)
                    int svd_ok = 0;
                    for (int mi = 0; mi < pgap_total_mats; mi++) {
                        pgap_compute_svd(&pgap_bases[mi]);
                        if (pgap_bases[mi].has_basis) svd_ok++;
                    }
                    pgap_basis_ready = (svd_ok == pgap_total_mats);

                    double probe_ms = tb_ms(mach_absolute_time() - t0);
                    printf("  [P-GAP probe complete: %d/%d SVDs ok, %.0fms, S[0] range: ",
                           svd_ok, pgap_total_mats, probe_ms);
                    // Print S[0] range across all bases
                    float s_min = 1e30f, s_max = 0;
                    for (int mi = 0; mi < pgap_total_mats; mi++) {
                        if (pgap_bases[mi].S[0] < s_min) s_min = pgap_bases[mi].S[0];
                        if (pgap_bases[mi].S[0] > s_max) s_max = pgap_bases[mi].S[0];
                    }
                    printf("%.4f..%.4f]\n", s_min, s_max);
                }

                // 7. Defer re-transpose
                if (!cpu_only && !conv_hybrid && val_every > 0 && (step + 1) % val_every == 0 && val_tokens > SEQ + 1) {
                    t0 = mach_absolute_time();
                    if (use_lora && !lora_ffn) { RETRANSPOSE_ATTN_ONLY(); }
                    else if (use_lora && lora_ffn) { RETRANSPOSE_AND_STAGE(); }
                    else { RETRANSPOSE_AND_STAGE(); }
                    t_transpose += tb_ms(mach_absolute_time() - t0);
                }

                // 8. LR schedule (cosine decay, no warmup)
                float min_lr = base_lr * 0.1f;
                float decay = (float)(step - start_step) / (float)(total_steps - start_step);
                lr = min_lr + 0.5f * (1.0f + cosf(M_PI * decay)) * (base_lr - min_lr);

                double step_ms = tb_ms(mach_absolute_time() - t_step);
                total_train_ms += step_ms;
                total_steps_done++;
                last_loss_plus = loss_plus;
                last_loss_minus = loss_minus;

                // 9. Log
                if (step % 100 == 0 || step == start_step) {
                    bool is_pgap_collect = (pgap_r > 0 && (step % pgap_k) < pgap_r);
                    printf("step %d  %sloss_plus=%.4f  loss_minus=%.4f  proj_grad=%.6f  lr=%.2e  "
                           "step_ms=%.0f (fwd=%.0f perturb=%.0f transpose=%.0f)",
                           step, is_pgap_collect ? "[P-GAP collect] " : "",
                           loss_plus, loss_minus, proj_grad, lr,
                           step_ms, t_fwd, t_perturb, t_transpose);
                    if (hessian_alpha > 0.0f)
                        printf("  L0=%.4f dL=%.4e", loss_0, loss_plus + loss_minus - 2.0f * loss_0);
                    printf("\n");
                }
            }

            // 10. Validation
            if (val_every > 0 && (step + 1) % val_every == 0 && val_tokens > SEQ + 1) {
                float val_loss_sum = 0;
                int val_batches = 10;
                srand48(999);  // Fixed val seed
                for (int vb = 0; vb < val_batches; vb++) {
                    size_t vpos = val_start + (size_t)(drand48() * (val_tokens - SEQ - 1));
                    uint16_t *vinput = token_data + vpos;
                    uint16_t *vtarget_raw = token_data + vpos + 1;
                    uint16_t vctargets[SEQ];
                    for (int t = 0; t < SEQ; t++) vctargets[t] = (uint16_t)vm.full_to_compact[vtarget_raw[t]];

                    float vb_loss;
                    DO_FORWARD_PASS(vinput, vctargets, vb_loss);
                    val_loss_sum += vb_loss;
                }
                float val_loss = val_loss_sum / val_batches;
                printf("  [val_loss=%.4f at step %d]\n", val_loss, step + 1);

                // Checkpoint on best val
                if (val_loss < best_loss) {
                    best_loss = val_loss;
                    double wall = tb_ms(mach_absolute_time() - t_wall_start);
                    mezo_save_checkpoint(CKPT_PATH, step + 1, total_steps, lr, val_loss,
                                        total_train_ms, wall, total_steps_done,
                                        lw, rms_final, embed);
                    printf("  [ckpt saved, best_val=%.4f]\n", best_loss);
                }
            }
        }

        // ===== Final report =====
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        printf("\n=== MeZO Efficiency Report ===\n");
        printf("Total steps:  %d\n", total_steps_done);
        printf("Train time:   %.0fms (%.1fms/step)\n", total_train_ms, total_train_ms / fmax(1, total_steps_done));
        printf("Wall time:    %.1fs\n", wall / 1000.0);
        printf("\n---\n");
        printf("final_loss_plus:  %.6f\n", last_loss_plus);
        printf("final_loss_minus: %.6f\n", last_loss_minus);
        printf("training_seconds: %.1f\n", total_train_ms / 1000.0);
        printf("total_seconds:    %.1f\n", wall / 1000.0);
        printf("num_steps:        %d\n", total_steps_done);
        printf("num_params_M:     %.1f\n", ((double)NLAYERS * LAYER_PARAMS + DIM + (double)VOCAB * DIM) / 1e6);
        printf("mode:             mezo-%s%s%s\n", cpu_only ? "cpu" : "ane",
               conv_hybrid ? "-conv-hybrid" : "",
               lora_split ? "-lora-split" : (use_lora ? "-lora" : ""));
        printf("epsilon:          %g\n", epsilon);
        printf("lr:               %g\n", lr);
        if (use_lora) printf("lora_rank:        %d\n", lora_rank);
        if (fzoo_K > 0) printf("fzoo_K:           %d\n", fzoo_K);
        if (pgap_r > 0) printf("pgap_r:           %d\npgap_k:           %d\npgap_h:           %d\npgap_xi:          %g\npgap_delta0:      %g\n", pgap_r, pgap_k, pgap_h, pgap_xi, pgap_delta0);
        if (sparse_ratio > 0) printf("sparse_ratio:     %g\nmask_refresh:     %d\n", sparse_ratio, mask_refresh);
        if (hessian_alpha > 0) printf("hessian_alpha:    %g\n", hessian_alpha);

        // Cleanup
        if (pgap_bases) {
            for (int mi = 0; mi < pgap_total_mats; mi++) pgap_basis_free(&pgap_bases[mi]);
            free(pgap_bases);
        }
        if (pgap_z_rms) free(pgap_z_rms);
        if (pgap_z_flat) free(pgap_z_flat);
        if (diag_hessian) free(diag_hessian);
        if (sparse_mask) free(sparse_mask);
        if (lora_tmp) free(lora_tmp);
        if (use_lora) {
            for (int L = 0; L < NLAYERS; L++) lora_layer_free(&lora_layers[L]);
        }
        for (int L = 0; L < NLAYERS; L++) {
            layer_weights_free(&lw[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
        }
        free(rms_final); free(embed); free(cembed);
        free(x_cur); free(xnorm_buf);
        free(Q); free(K); free(V); free(attn_out); free(o_out);
        free(h1); free(h3); free(silu_out);
        free(k_tiled); free(v_tiled); free(logits); free(dlogits);
        munmap(token_data, data_len); close(data_fd);

        if (!cpu_only && conv_fused) {
            for (int L = 0; L < NLAYERS; L++) {
                free_kern(dk.qkvConv[L]); free_kern(dk.woConv[L]); free_kern(dk.ffnConv[L]);
            }
        } else if (!cpu_only && conv_hybrid) {
            free_per_layer(pls, plr);
            free_kern(dk.wkvFwd);
            for (int L = 0; L < NLAYERS; L++) {
                free_kern(dk.wqConv[L]); free_kern(dk.woConv[L]);
                free_kern(dk.w1Conv[L]); free_kern(dk.w2Conv[L]); free_kern(dk.w3Conv[L]);
            }
        } else if (!cpu_only) {
            free_per_layer(pls, plr);
            free_kern(dk.wqFwd); free_kern(dk.wkvFwd); free_kern(dk.w13Fwd);
            free_kern(dk.w2Fwd); free_kern(dk.woFwd);
        }
    }
    return 0;
}
