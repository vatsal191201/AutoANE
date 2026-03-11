// bench_cpu_matmul.m — Benchmark cblas_sgemm for AutoANE matrix shapes
// Build: xcrun clang -O2 -framework Accelerate -framework Foundation -o bench_cpu_matmul bench_cpu_matmul.m
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

static double tb_ms(uint64_t dt) {
    static mach_timebase_info_data_t tb = {0};
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)dt * tb.numer / tb.denom / 1e6;
}

// Benchmark: C = A @ B where A is [M,K], B is [K,N], C is [M,N]
static void bench_sgemm(const char *name, int M, int K, int N, int iters) {
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    // Random init
    for (int i = 0; i < M*K; i++) A[i] = (float)drand48() * 0.01f;
    for (int i = 0; i < K*N; i++) B[i] = (float)drand48() * 0.01f;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }

    // Benchmark
    uint64_t t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double total_ms = tb_ms(mach_absolute_time() - t0);
    double per_iter_ms = total_ms / iters;

    // FLOPS: 2*M*N*K per matmul
    double flops = 2.0 * M * N * K;
    double gflops = flops * iters / total_ms / 1e6;  // GFLOPS = flops / (ms * 1e6)

    printf("%-40s  %4dx%-4d @ %-4dx%-4d  %.3f ms  %.1f GFLOPS\n",
           name, M, K, K, N, per_iter_ms, gflops);

    free(A); free(B); free(C);
}

int main() {
    printf("=== CPU Matmul Benchmark (cblas_sgemm, fp32) ===\n");
    printf("%-40s  %-20s  %-8s  %s\n", "Operation", "Shape", "Time", "Throughput");
    printf("-------------------------------------------------------------------------------------\n");

    int SEQ = 256;
    int DIM = 1024;
    int Q_DIM = 1024;  // HEADS*HD = 16*64
    int KV_DIM = 256;  // KV_HEADS*HD = 4*64
    int HIDDEN = 2816;
    int HEADS = 16;
    int HD = 64;
    int iters = 1000;

    // === Forward pass matmuls ===
    printf("\n--- Forward Pass ---\n");
    // Wq: [SEQ,DIM] @ [DIM,Q_DIM] -> [SEQ,Q_DIM]
    bench_sgemm("Wq forward (x @ Wq)", SEQ, DIM, Q_DIM, iters);
    // Wk: [SEQ,DIM] @ [DIM,KV_DIM] -> [SEQ,KV_DIM]
    bench_sgemm("Wk forward (x @ Wk)", SEQ, DIM, KV_DIM, iters);
    // Wv: same as Wk
    bench_sgemm("Wv forward (x @ Wv)", SEQ, DIM, KV_DIM, iters);
    // Wo: [SEQ,Q_DIM] @ [Q_DIM,DIM] -> [SEQ,DIM]
    bench_sgemm("Wo forward (attn @ Wo)", SEQ, Q_DIM, DIM, iters);
    // W1: [SEQ,DIM] @ [DIM,HIDDEN] -> [SEQ,HIDDEN]
    bench_sgemm("W1 forward (x @ W1)", SEQ, DIM, HIDDEN, iters);
    // W3: same as W1
    bench_sgemm("W3 forward (x @ W3)", SEQ, DIM, HIDDEN, iters);
    // W2: [SEQ,HIDDEN] @ [HIDDEN,DIM] -> [SEQ,DIM]
    bench_sgemm("W2 forward (silu @ W2)", SEQ, HIDDEN, DIM, iters);

    // Attention: Q@K^T and S@V
    // Q@K^T: per head [SEQ,HD] @ [HD,SEQ] -> [SEQ,SEQ], done HEADS times
    bench_sgemm("Q@K^T per head", SEQ, HD, SEQ, iters);
    // scores@V: [SEQ,SEQ] @ [SEQ,HD] -> [SEQ,HD], done HEADS times
    bench_sgemm("scores@V per head", SEQ, SEQ, HD, iters);

    // === Backward pass matmuls (dX = dY @ W^T, same shapes) ===
    printf("\n--- Backward Pass (dX) ---\n");
    // dWq: dq @ Wq^T -> dx_q
    bench_sgemm("Wq^T backward (dq @ Wq^T)", SEQ, Q_DIM, DIM, iters);
    // dWk: dk @ Wk^T -> dx_k
    bench_sgemm("Wk^T backward (dk @ Wk^T)", SEQ, KV_DIM, DIM, iters);
    // dWv: dv @ Wv^T -> dx_v
    bench_sgemm("Wv^T backward (dv @ Wv^T)", SEQ, KV_DIM, DIM, iters);
    // dWo: da @ Wo^T -> dx_o
    bench_sgemm("Wo^T backward (da @ Wo^T)", SEQ, DIM, Q_DIM, iters);
    // dW2: dffn @ W2^T -> dh
    bench_sgemm("W2^T backward (dffn @ W2^T)", SEQ, DIM, HIDDEN, iters);
    // dW1: dh1 @ W1^T -> dx_1
    bench_sgemm("W1^T backward (dh1 @ W1^T)", SEQ, HIDDEN, DIM, iters);
    // dW3: dh3 @ W3^T -> dx_3
    bench_sgemm("W3^T backward (dh3 @ W3^T)", SEQ, HIDDEN, DIM, iters);

    // === dW computation (already on CPU in current pipeline) ===
    printf("\n--- dW Computation (weight gradients) ---\n");
    // dWq: x^T @ dq -> [DIM,Q_DIM]
    bench_sgemm("dWq (x^T @ dq)", DIM, SEQ, Q_DIM, iters);
    // dWk: x^T @ dk -> [DIM,KV_DIM]
    bench_sgemm("dWk (x^T @ dk)", DIM, SEQ, KV_DIM, iters);
    // dWv: x^T @ dv -> [DIM,KV_DIM]
    bench_sgemm("dWv (x^T @ dv)", DIM, SEQ, KV_DIM, iters);
    // dWo: attn^T @ da -> [Q_DIM,DIM]
    bench_sgemm("dWo (attn^T @ da)", Q_DIM, SEQ, DIM, iters);
    // dW1: xnorm^T @ dh1 -> [DIM,HIDDEN]
    bench_sgemm("dW1 (x^T @ dh1)", DIM, SEQ, HIDDEN, iters);
    // dW2: silu^T @ dffn -> [HIDDEN,DIM]
    bench_sgemm("dW2 (silu^T @ dffn)", HIDDEN, SEQ, DIM, iters);
    // dW3: xnorm^T @ dh3 -> [DIM,HIDDEN]
    bench_sgemm("dW3 (x^T @ dh3)", DIM, SEQ, HIDDEN, iters);

    // === Summary ===
    printf("\n--- Per-Layer Summary (1 layer) ---\n");
    // Forward: 7 linear matmuls + HEADS*2 attention matmuls
    {
        int shapes[][3] = {
            {SEQ,DIM,Q_DIM}, {SEQ,DIM,KV_DIM}, {SEQ,DIM,KV_DIM},  // Wq,Wk,Wv
            {SEQ,Q_DIM,DIM},                                         // Wo
            {SEQ,DIM,HIDDEN}, {SEQ,DIM,HIDDEN}, {SEQ,HIDDEN,DIM},   // W1,W3,W2
        };
        double fwd_ms = 0;
        for (int i = 0; i < 7; i++) {
            float *A = calloc(shapes[i][0]*shapes[i][1], 4);
            float *B = calloc(shapes[i][1]*shapes[i][2], 4);
            float *C = calloc(shapes[i][0]*shapes[i][2], 4);
            uint64_t t0 = mach_absolute_time();
            for (int j = 0; j < 100; j++)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            shapes[i][0], shapes[i][2], shapes[i][1],
                            1.0f, A, shapes[i][1], B, shapes[i][2], 0.0f, C, shapes[i][2]);
            fwd_ms += tb_ms(mach_absolute_time() - t0) / 100.0;
            free(A); free(B); free(C);
        }
        // Attention matmuls (per head)
        {
            float *A = calloc(SEQ*HD, 4);
            float *B = calloc(HD*SEQ, 4);
            float *C = calloc(SEQ*SEQ, 4);
            float *D = calloc(SEQ*HD, 4);
            uint64_t t0 = mach_absolute_time();
            for (int j = 0; j < 100; j++) {
                for (int h = 0; h < HEADS; h++) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                SEQ, SEQ, HD, 1.0f, A, HD, B, HD, 0.0f, C, SEQ);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                SEQ, HD, SEQ, 1.0f, C, SEQ, A, HD, 0.0f, D, HD);
                }
            }
            fwd_ms += tb_ms(mach_absolute_time() - t0) / 100.0;
            free(A); free(B); free(C); free(D);
        }
        printf("Forward (linear + attn):  %.2f ms / layer\n", fwd_ms);

        // Backward dX: similar to forward (7 matmuls + attention backward)
        double bwd_dx_ms = fwd_ms;  // approximately same
        printf("Backward dX (approx):     %.2f ms / layer\n", bwd_dx_ms);

        // Backward dW: 7 matmuls [DIM/Q_DIM/HIDDEN, SEQ, *]
        double bwd_dw_ms = 0;
        int dw_shapes[][3] = {
            {DIM,SEQ,Q_DIM}, {DIM,SEQ,KV_DIM}, {DIM,SEQ,KV_DIM},  // dWq,dWk,dWv
            {Q_DIM,SEQ,DIM},                                         // dWo
            {DIM,SEQ,HIDDEN}, {HIDDEN,SEQ,DIM}, {DIM,SEQ,HIDDEN},   // dW1,dW2,dW3
        };
        for (int i = 0; i < 7; i++) {
            float *A = calloc(dw_shapes[i][0]*dw_shapes[i][1], 4);
            float *B = calloc(dw_shapes[i][1]*dw_shapes[i][2], 4);
            float *C = calloc(dw_shapes[i][0]*dw_shapes[i][2], 4);
            uint64_t t0 = mach_absolute_time();
            for (int j = 0; j < 100; j++)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            dw_shapes[i][0], dw_shapes[i][2], dw_shapes[i][1],
                            1.0f, A, dw_shapes[i][1], B, dw_shapes[i][2], 0.0f, C, dw_shapes[i][2]);
            bwd_dw_ms += tb_ms(mach_absolute_time() - t0) / 100.0;
            free(A); free(B); free(C);
        }
        printf("Backward dW:              %.2f ms / layer\n", bwd_dw_ms);

        double total_per_layer = fwd_ms + bwd_dx_ms + bwd_dw_ms;
        printf("Total per layer:          %.2f ms\n", total_per_layer);
        printf("Total 4 layers:           %.2f ms\n", total_per_layer * 4);
        printf("\n(Does not include: RMSNorm, SiLU, softmax, embedding, loss, Adam)\n");
    }

    return 0;
}
