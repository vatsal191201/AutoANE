// benchmark_transpose.m — Microbenchmark: decompose RETRANSPOSE_AND_STAGE overhead
// Measures: vDSP_mtrans, cvt_f32_f16 per-channel staging, IOSurface lock/unlock
// Build: clang -O2 -framework Foundation -framework IOSurface -framework Accelerate \
//        -include models/smollm2_135m.h -o benchmark_transpose benchmark_transpose.m
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

// Timer helper
static double tb_ms(uint64_t elapsed) {
    static mach_timebase_info_data_t info = {0,0};
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)elapsed * info.numer / info.denom / 1e6;
}

// Model dimensions (from smollm2_135m.h)
#ifndef DIM
#define DIM 576
#define HIDDEN 1536
#define Q_DIM 576
#define KV_DIM 192
#define SEQ 256
#define NLAYERS 30
#endif

#define WQ_FWD_SP  (SEQ + Q_DIM)
#define WKV_FWD_SP (SEQ + KV_DIM)
#define W13_FWD_SP (SEQ + HIDDEN)
#define W2_FWD_SP  (SEQ + DIM)

// Transpose
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    vDSP_mtrans(src, 1, dst, 1, (vDSP_Length)cols, (vDSP_Length)rows);
}

// NEON fp32->fp16
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

int main(void) {
    printf("=== RETRANSPOSE_AND_STAGE Microbenchmark ===\n");
    printf("Model: SmolLM2-135M | DIM=%d HIDDEN=%d Q_DIM=%d KV_DIM=%d NLAYERS=%d SEQ=%d\n\n",
           DIM, HIDDEN, Q_DIM, KV_DIM, NLAYERS, SEQ);

    // Allocate weight buffers (1 layer)
    float *Wq = (float*)calloc(Q_DIM * DIM, sizeof(float));
    float *Wk = (float*)calloc(KV_DIM * DIM, sizeof(float));
    float *Wv = (float*)calloc(KV_DIM * DIM, sizeof(float));
    float *Wo = (float*)calloc(DIM * Q_DIM, sizeof(float));
    float *W1 = (float*)calloc(HIDDEN * DIM, sizeof(float));
    float *W2 = (float*)calloc(DIM * HIDDEN, sizeof(float));
    float *W3 = (float*)calloc(HIDDEN * DIM, sizeof(float));

    // Transposed buffers
    float *Wqt = (float*)calloc(Q_DIM * DIM, sizeof(float));
    float *Wkt = (float*)calloc(KV_DIM * DIM, sizeof(float));
    float *Wvt = (float*)calloc(KV_DIM * DIM, sizeof(float));
    float *Wot = (float*)calloc(DIM * Q_DIM, sizeof(float));
    float *W1t = (float*)calloc(HIDDEN * DIM, sizeof(float));
    float *W2t = (float*)calloc(DIM * HIDDEN, sizeof(float));
    float *W3t = (float*)calloc(HIDDEN * DIM, sizeof(float));

    // Fill with random data
    srand48(42);
    for (int i = 0; i < Q_DIM*DIM; i++) Wq[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < KV_DIM*DIM; i++) Wk[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < KV_DIM*DIM; i++) Wv[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < DIM*Q_DIM; i++) Wo[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < HIDDEN*DIM; i++) W1[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < DIM*HIDDEN; i++) W2[i] = (float)drand48() - 0.5f;
    for (int i = 0; i < HIDDEN*DIM; i++) W3[i] = (float)drand48() - 0.5f;

    // IOSurface buffers (per-layer)
    IOSurfaceRef wqFwd_in = make_surface(DIM * WQ_FWD_SP * 2);
    IOSurfaceRef wkFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
    IOSurfaceRef wvFwd_in = make_surface(DIM * WKV_FWD_SP * 2);
    IOSurfaceRef woFwd_in = make_surface(Q_DIM * (SEQ + DIM) * 2);
    IOSurfaceRef w1Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
    IOSurfaceRef w3Fwd_in = make_surface(DIM * W13_FWD_SP * 2);
    IOSurfaceRef w2Fwd_in = make_surface(HIDDEN * W2_FWD_SP * 2);

    int N_TRIALS = 10;

    // ===== Phase 1: Measure transpose alone (all 7 matrices, 1 layer) =====
    printf("--- Phase 1: vDSP_mtrans per layer ---\n");
    double t_transpose_single = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();
        transpose_weight(Wqt, Wq, Q_DIM, DIM);
        transpose_weight(Wkt, Wk, KV_DIM, DIM);
        transpose_weight(Wvt, Wv, KV_DIM, DIM);
        transpose_weight(Wot, Wo, DIM, Q_DIM);
        transpose_weight(W1t, W1, HIDDEN, DIM);
        transpose_weight(W2t, W2, DIM, HIDDEN);
        transpose_weight(W3t, W3, HIDDEN, DIM);
        t_transpose_single += tb_ms(mach_absolute_time() - t0);
    }
    t_transpose_single /= N_TRIALS;
    printf("  Per-layer transpose (7 matrices): %.2f ms\n", t_transpose_single);
    printf("  All %d layers:                    %.2f ms\n", NLAYERS, t_transpose_single * NLAYERS);

    // ===== Phase 2: Measure IOSurface staging alone (per-channel cvt_f32_f16) =====
    printf("\n--- Phase 2: IOSurface staging per layer ---\n");
    double t_stage_single = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();

        // Wq staging
        IOSurfaceLock(wqFwd_in, 0, NULL);
        _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(wqFwd_in);
        for (int d = 0; d < DIM; d++)
            cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt + d*Q_DIM, Q_DIM);
        IOSurfaceUnlock(wqFwd_in, 0, NULL);

        // Wk staging
        IOSurfaceLock(wkFwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(wkFwd_in);
        for (int d = 0; d < DIM; d++)
            cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt + d*KV_DIM, KV_DIM);
        IOSurfaceUnlock(wkFwd_in, 0, NULL);

        // Wv staging
        IOSurfaceLock(wvFwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(wvFwd_in);
        for (int d = 0; d < DIM; d++)
            cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt + d*KV_DIM, KV_DIM);
        IOSurfaceUnlock(wvFwd_in, 0, NULL);

        // Wo staging (simplified)
        IOSurfaceLock(woFwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(woFwd_in);
        int WO_FWD_SP = SEQ + DIM;
        for (int d = 0; d < Q_DIM; d++)
            cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wot + d*DIM, DIM);
        IOSurfaceUnlock(woFwd_in, 0, NULL);

        // W1 staging
        IOSurfaceLock(w1Fwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(w1Fwd_in);
        for (int d = 0; d < DIM; d++)
            cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W1t + d*HIDDEN, HIDDEN);
        IOSurfaceUnlock(w1Fwd_in, 0, NULL);

        // W3 staging
        IOSurfaceLock(w3Fwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(w3Fwd_in);
        for (int d = 0; d < DIM; d++)
            cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W3t + d*HIDDEN, HIDDEN);
        IOSurfaceUnlock(w3Fwd_in, 0, NULL);

        // W2 staging (manual element-wise — the slowest part)
        IOSurfaceLock(w2Fwd_in, 0, NULL);
        buf = (_Float16*)IOSurfaceGetBaseAddress(w2Fwd_in);
        for (int h = 0; h < HIDDEN; h++)
            for (int d = 0; d < DIM; d++)
                buf[h*W2_FWD_SP + SEQ + d] = (_Float16)W2[d*HIDDEN + h];
        IOSurfaceUnlock(w2Fwd_in, 0, NULL);

        t_stage_single += tb_ms(mach_absolute_time() - t0);
    }
    t_stage_single /= N_TRIALS;
    printf("  Per-layer staging (7 matrices): %.2f ms\n", t_stage_single);
    printf("  All %d layers:                  %.2f ms\n", NLAYERS, t_stage_single * NLAYERS);

    // ===== Phase 3: W2 staging alone (suspected bottleneck) =====
    printf("\n--- Phase 3: W2 element-wise staging (suspected bottleneck) ---\n");
    double t_w2_stage = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();
        IOSurfaceLock(w2Fwd_in, 0, NULL);
        _Float16 *buf2 = (_Float16*)IOSurfaceGetBaseAddress(w2Fwd_in);
        for (int h = 0; h < HIDDEN; h++)
            for (int d = 0; d < DIM; d++)
                buf2[h*W2_FWD_SP + SEQ + d] = (_Float16)W2[d*HIDDEN + h];
        IOSurfaceUnlock(w2Fwd_in, 0, NULL);
        t_w2_stage += tb_ms(mach_absolute_time() - t0);
    }
    t_w2_stage /= N_TRIALS;
    printf("  W2 staging per layer: %.2f ms\n", t_w2_stage);
    printf("  W2 staging all %d layers: %.2f ms\n", NLAYERS, t_w2_stage * NLAYERS);

    // ===== Phase 4: W2 staging with pre-transpose (optimization candidate) =====
    printf("\n--- Phase 4: W2 optimized — pre-transpose then bulk cvt_f32_f16 ---\n");
    double t_w2_opt = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();
        // First transpose W2[DIM,HIDDEN] -> W2t[HIDDEN,DIM]
        transpose_weight(W2t, W2, DIM, HIDDEN);
        // Then stage like the other matrices
        IOSurfaceLock(w2Fwd_in, 0, NULL);
        _Float16 *buf2 = (_Float16*)IOSurfaceGetBaseAddress(w2Fwd_in);
        for (int h = 0; h < HIDDEN; h++)
            cvt_f32_f16(buf2 + h*W2_FWD_SP + SEQ, W2t + h*DIM, DIM);
        IOSurfaceUnlock(w2Fwd_in, 0, NULL);
        t_w2_opt += tb_ms(mach_absolute_time() - t0);
    }
    t_w2_opt /= N_TRIALS;
    printf("  W2 optimized per layer: %.2f ms (%.1fx speedup)\n",
           t_w2_opt, t_w2_stage / t_w2_opt);

    // ===== Phase 5: IOSurface lock/unlock overhead alone =====
    printf("\n--- Phase 5: IOSurface lock/unlock overhead ---\n");
    double t_lock = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < 7; i++) {
            IOSurfaceLock(wqFwd_in, 0, NULL);
            IOSurfaceUnlock(wqFwd_in, 0, NULL);
        }
        t_lock += tb_ms(mach_absolute_time() - t0);
    }
    t_lock /= N_TRIALS;
    printf("  7 lock/unlock pairs per layer: %.3f ms\n", t_lock);
    printf("  All %d layers:                 %.3f ms\n", NLAYERS, t_lock * NLAYERS);

    // ===== Phase 6: Full RETRANSPOSE_AND_STAGE equivalent (all 30 layers) =====
    printf("\n--- Phase 6: Full RETRANSPOSE_AND_STAGE (all layers) ---\n");
    double t_full = 0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        uint64_t t0 = mach_absolute_time();
        for (int L = 0; L < NLAYERS; L++) {
            // Transpose
            transpose_weight(Wqt, Wq, Q_DIM, DIM);
            transpose_weight(Wkt, Wk, KV_DIM, DIM);
            transpose_weight(Wvt, Wv, KV_DIM, DIM);
            transpose_weight(Wot, Wo, DIM, Q_DIM);
            transpose_weight(W1t, W1, HIDDEN, DIM);
            transpose_weight(W2t, W2, DIM, HIDDEN);
            transpose_weight(W3t, W3, HIDDEN, DIM);

            // Staging (same as Phase 2)
            IOSurfaceLock(wqFwd_in, 0, NULL);
            _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(wqFwd_in);
            for (int d = 0; d < DIM; d++)
                cvt_f32_f16(buf + d*WQ_FWD_SP + SEQ, Wqt + d*Q_DIM, Q_DIM);
            IOSurfaceUnlock(wqFwd_in, 0, NULL);

            IOSurfaceLock(wkFwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(wkFwd_in);
            for (int d = 0; d < DIM; d++)
                cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wkt + d*KV_DIM, KV_DIM);
            IOSurfaceUnlock(wkFwd_in, 0, NULL);

            IOSurfaceLock(wvFwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(wvFwd_in);
            for (int d = 0; d < DIM; d++)
                cvt_f32_f16(buf + d*WKV_FWD_SP + SEQ, Wvt + d*KV_DIM, KV_DIM);
            IOSurfaceUnlock(wvFwd_in, 0, NULL);

            IOSurfaceLock(woFwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(woFwd_in);
            int WO_FWD_SP = SEQ + DIM;
            for (int d = 0; d < Q_DIM; d++)
                cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wot + d*DIM, DIM);
            IOSurfaceUnlock(woFwd_in, 0, NULL);

            IOSurfaceLock(w1Fwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(w1Fwd_in);
            for (int d = 0; d < DIM; d++)
                cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W1t + d*HIDDEN, HIDDEN);
            IOSurfaceUnlock(w1Fwd_in, 0, NULL);

            IOSurfaceLock(w3Fwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(w3Fwd_in);
            for (int d = 0; d < DIM; d++)
                cvt_f32_f16(buf + d*W13_FWD_SP + SEQ, W3t + d*HIDDEN, HIDDEN);
            IOSurfaceUnlock(w3Fwd_in, 0, NULL);

            IOSurfaceLock(w2Fwd_in, 0, NULL);
            buf = (_Float16*)IOSurfaceGetBaseAddress(w2Fwd_in);
            for (int h = 0; h < HIDDEN; h++)
                for (int d = 0; d < DIM; d++)
                    buf[h*W2_FWD_SP + SEQ + d] = (_Float16)W2[d*HIDDEN + h];
            IOSurfaceUnlock(w2Fwd_in, 0, NULL);
        }
        t_full += tb_ms(mach_absolute_time() - t0);
    }
    t_full /= N_TRIALS;
    printf("  Full RETRANSPOSE_AND_STAGE: %.2f ms\n", t_full);
    printf("  3x per MeZO step:          %.2f ms\n", t_full * 3);
    printf("  2x (optimized, defer 3rd): %.2f ms (saving %.2f ms/step)\n",
           t_full * 2, t_full);

    // ===== Summary =====
    printf("\n=== Summary ===\n");
    double t_trans_total = t_transpose_single * NLAYERS;
    double t_stage_total = t_stage_single * NLAYERS;
    printf("Transpose (vDSP_mtrans):     %.2f ms (%.0f%%)\n",
           t_trans_total, 100.0*t_trans_total/t_full);
    printf("Staging (cvt+lock/unlock):   %.2f ms (%.0f%%)\n",
           t_stage_total, 100.0*t_stage_total/t_full);
    printf("  of which W2 element-wise:  %.2f ms\n", t_w2_stage * NLAYERS);
    printf("Lock/unlock alone:           %.3f ms\n", t_lock * NLAYERS);
    printf("\nOptimization opportunities:\n");
    printf("  1. Defer 3rd RETRANSPOSE: save %.2f ms/step (%.0f%%)\n",
           t_full, 100.0*t_full/(t_full*3));
    printf("  2. Optimize W2 staging:   save %.2f ms per full restage\n",
           (t_w2_stage - t_w2_opt) * NLAYERS);

    // Cleanup
    free(Wq); free(Wk); free(Wv); free(Wo); free(W1); free(W2); free(W3);
    free(Wqt); free(Wkt); free(Wvt); free(Wot); free(W1t); free(W2t); free(W3t);
    CFRelease(wqFwd_in); CFRelease(wkFwd_in); CFRelease(wvFwd_in);
    CFRelease(woFwd_in); CFRelease(w1Fwd_in); CFRelease(w3Fwd_in); CFRelease(w2Fwd_in);

    return 0;
}
