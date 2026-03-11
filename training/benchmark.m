// benchmark.m — Automated ANE characterization benchmark suite for AutoANE
// Build: xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate -isysroot $(xcrun --show-sdk-path) -fobjc-arc -include dlfcn.h -o benchmark benchmark.m
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>

// ===== Minimal ANE runtime (copied from bench_conv_vs_matmul.m) =====
static Class g_D, g_I, g_AR, g_AIO;

static void ane_init(void) {
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) { printf("FATAL: Cannot load AppleNeuralEngine.framework\n"); exit(1); }
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
}

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *props = @{@"IOSurfaceWidth": @(bytes), @"IOSurfaceHeight": @1,
                            @"IOSurfaceBytesPerElement": @1, @"IOSurfacePixelFormat": @0};
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

typedef struct {
    void *model;
    IOSurfaceRef ioIn, ioOut;
    void *request;
    void *tmpDir;
} Kern;

static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int sz = cnt * 2;
    int total = 128 + sz;
    uint8_t *b = (uint8_t*)calloc(total, 1);
    b[0]=1; b[4]=2;
    b[64]=0xEF; b[65]=0xBE; b[66]=0xAD; b[67]=0xDE;
    b[68]=1;
    *(uint32_t*)(b+72) = sz;
    *(uint32_t*)(b+80) = 128;
    memcpy(b+128, d, sz);
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

static Kern *compile_kern(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
                                withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  compile FAIL: %s\n", e ? [[e description] UTF8String] : "?"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  load FAIL\n"); return NULL;
    }
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    k->model = (void*)CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}

static void ane_run(Kern *k) {
    id mdl = (__bridge id)k->model;
    id req = (__bridge id)k->request;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}

static double tb_ms(uint64_t dt) {
    static mach_timebase_info_data_t tb = {0};
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)dt * tb.numer / tb.denom / 1e6;
}

// ===== MIL Generator (dynamic matmul) =====

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Dynamic matmul: y = x @ W, input [1,IC,1,SEQ+OC] (acts+weights packed), output [1,OC,1,SEQ]
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    int sp = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];

    // Slice activation [1,IC,1,SEQ]
    [m appendFormat:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];

    // Slice weight [1,IC,1,OC]
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];

    // Reshape for matmul
    [m appendFormat:@"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", seq, ic];
    [m appendFormat:@"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", oc, seq];
    [m appendFormat:@"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== JSON result accumulator =====
static NSMutableDictionary *g_results;

static void json_add_double(NSString *key, double val) {
    g_results[key] = @(val);
}

static void json_add_string(NSString *key, NSString *val) {
    g_results[key] = val;
}

// ===== Benchmark 1: ANE Matmul =====
typedef struct {
    const char *name;
    int ic;
    int oc;
    int seq;
} MatmulShape;

static void bench_ane_matmul(int iters) {
    printf("\n========================================\n");
    printf("  Benchmark 1: ANE Dynamic Matmul\n");
    printf("  %d iterations per shape\n", iters);
    printf("========================================\n");

    int seq = 256;
    MatmulShape shapes[] = {
        {"1024x1024", 1024, 1024, seq},
        {"1024x256",  1024,  256, seq},
        {"1024x2816", 1024, 2816, seq},
        {"2816x1024", 2816, 1024, seq},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < n_shapes; s++) {
        MatmulShape *sh = &shapes[s];
        printf("\n--- %s: IC=%d OC=%d SEQ=%d ---\n", sh->name, sh->ic, sh->oc, sh->seq);

        // Compile kernel
        NSString *mil = gen_dyn_matmul_mil(sh->ic, sh->oc, sh->seq);
        int in_bytes  = sh->ic * (sh->seq + sh->oc) * 2;
        int out_bytes = sh->oc * sh->seq * 2;

        uint64_t t0 = mach_absolute_time();
        Kern *k = compile_kern(mil, @{}, in_bytes, out_bytes);
        double compile_ms = tb_ms(mach_absolute_time() - t0);

        if (!k) {
            printf("COMPILE FAILED for %s\n", sh->name);
            json_add_double([NSString stringWithFormat:@"ane_matmul_%s_ms", sh->name], -1);
            continue;
        }
        printf("Compiled in %.1f ms\n", compile_ms);
        json_add_double([NSString stringWithFormat:@"ane_compile_%s_ms", sh->name], compile_ms);

        // Stage random input data
        IOSurfaceLock(k->ioIn, 0, NULL);
        _Float16 *in_buf = (_Float16*)IOSurfaceGetBaseAddress(k->ioIn);
        int in_elems = sh->ic * (sh->seq + sh->oc);
        for (int i = 0; i < in_elems; i++) in_buf[i] = (_Float16)(drand48() * 0.02 - 0.01);
        IOSurfaceUnlock(k->ioIn, 0, NULL);

        // Warmup
        for (int i = 0; i < 20; i++) ane_run(k);

        // Benchmark
        t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++) ane_run(k);
        double total_ms = tb_ms(mach_absolute_time() - t0);
        double per_iter = total_ms / iters;

        double flops = 2.0 * sh->ic * sh->oc * sh->seq;
        double gflops = flops / per_iter / 1e6;

        printf("Time: %.3f ms/eval  (%.1f GFLOPS)\n", per_iter, gflops);

        json_add_double([NSString stringWithFormat:@"ane_matmul_%s_ms", sh->name], per_iter);
        json_add_double([NSString stringWithFormat:@"ane_matmul_%s_gflops", sh->name], gflops);

        free_kern(k);
    }
}

// ===== Benchmark 2: CPU Matmul (cblas_sgemm) =====
static void bench_cpu_matmul(int iters) {
    printf("\n========================================\n");
    printf("  Benchmark 2: CPU Matmul (cblas_sgemm)\n");
    printf("  %d iterations per shape\n", iters);
    printf("========================================\n");

    int seq = 256;
    MatmulShape shapes[] = {
        {"1024x1024", 1024, 1024, seq},
        {"1024x256",  1024,  256, seq},
        {"1024x2816", 1024, 2816, seq},
        {"2816x1024", 2816, 1024, seq},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < n_shapes; s++) {
        MatmulShape *sh = &shapes[s];
        int M = sh->seq;
        int K = sh->ic;
        int N = sh->oc;
        printf("\n--- %s: [%d,%d] @ [%d,%d] ---\n", sh->name, M, K, K, N);

        float *A = (float*)malloc(M * K * sizeof(float));
        float *B = (float*)malloc(K * N * sizeof(float));
        float *C = (float*)malloc(M * N * sizeof(float));

        for (int i = 0; i < M*K; i++) A[i] = (float)drand48() * 0.01f;
        for (int i = 0; i < K*N; i++) B[i] = (float)drand48() * 0.01f;

        // Warmup
        for (int i = 0; i < 10; i++) {
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
        double per_iter = total_ms / iters;

        double flops = 2.0 * M * N * K;
        double gflops = flops * iters / total_ms / 1e6;

        printf("Time: %.3f ms/eval  (%.1f GFLOPS)\n", per_iter, gflops);

        json_add_double([NSString stringWithFormat:@"cpu_matmul_%s_ms", sh->name], per_iter);
        json_add_double([NSString stringWithFormat:@"cpu_matmul_%s_gflops", sh->name], gflops);

        free(A); free(B); free(C);
    }
}

// ===== Benchmark 3: IO Overhead Test =====
static void bench_io_overhead(void) {
    printf("\n========================================\n");
    printf("  Benchmark 3: IOSurface IO Overhead\n");
    printf("========================================\n");

    // Typical tensor sizes in bytes (fp16)
    typedef struct { const char *name; size_t bytes; } TensorSize;
    TensorSize sizes[] = {
        {"256x1024 (acts)",       256 * 1024 * 2},
        {"1024x1024 (weights)",  1024 * 1024 * 2},
        {"256x2816 (hidden)",     256 * 2816 * 2},
        {"1024x2816 (W1)",       1024 * 2816 * 2},
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int io_iters = 1000;

    for (int s = 0; s < n_sizes; s++) {
        TensorSize *ts = &sizes[s];
        int n_elems = (int)(ts->bytes / 2);
        printf("\n--- %s (%zu bytes) ---\n", ts->name, ts->bytes);

        IOSurfaceRef surf = make_surface(ts->bytes);
        _Float16 *src = (_Float16*)malloc(ts->bytes);
        _Float16 *dst = (_Float16*)malloc(ts->bytes);
        for (int i = 0; i < n_elems; i++) src[i] = (_Float16)(drand48() * 0.01);

        // Benchmark: lock + write + unlock
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < io_iters; i++) {
            IOSurfaceLock(surf, 0, NULL);
            _Float16 *base = (_Float16*)IOSurfaceGetBaseAddress(surf);
            memcpy(base, src, ts->bytes);
            IOSurfaceUnlock(surf, 0, NULL);
        }
        double write_ms = tb_ms(mach_absolute_time() - t0) / io_iters;

        // Benchmark: lock + read + unlock
        t0 = mach_absolute_time();
        for (int i = 0; i < io_iters; i++) {
            IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
            _Float16 *base = (_Float16*)IOSurfaceGetBaseAddress(surf);
            memcpy(dst, base, ts->bytes);
            IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
        }
        double read_ms = tb_ms(mach_absolute_time() - t0) / io_iters;

        // Benchmark: full write+read cycle
        t0 = mach_absolute_time();
        for (int i = 0; i < io_iters; i++) {
            IOSurfaceLock(surf, 0, NULL);
            _Float16 *base = (_Float16*)IOSurfaceGetBaseAddress(surf);
            memcpy(base, src, ts->bytes);
            IOSurfaceUnlock(surf, 0, NULL);

            IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
            base = (_Float16*)IOSurfaceGetBaseAddress(surf);
            memcpy(dst, base, ts->bytes);
            IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
        }
        double cycle_ms = tb_ms(mach_absolute_time() - t0) / io_iters;

        double bw_write = (double)ts->bytes / write_ms / 1e6;  // GB/s
        double bw_read  = (double)ts->bytes / read_ms  / 1e6;

        printf("Write: %.4f ms  (%.1f GB/s)\n", write_ms, bw_write);
        printf("Read:  %.4f ms  (%.1f GB/s)\n", read_ms, bw_read);
        printf("Cycle: %.4f ms\n", cycle_ms);

        NSString *key_prefix = [NSString stringWithFormat:@"io_%zu", ts->bytes];
        json_add_double([NSString stringWithFormat:@"%@_write_ms", key_prefix], write_ms);
        json_add_double([NSString stringWithFormat:@"%@_read_ms", key_prefix], read_ms);
        json_add_double([NSString stringWithFormat:@"%@_cycle_ms", key_prefix], cycle_ms);
        json_add_double([NSString stringWithFormat:@"%@_write_gbps", key_prefix], bw_write);
        json_add_double([NSString stringWithFormat:@"%@_read_gbps", key_prefix], bw_read);

        CFRelease(surf);
        free(src);
        free(dst);
    }
}

// ===== Benchmark 4: Thermal Profile =====
static void bench_thermal_profile(void) {
    printf("\n========================================\n");
    printf("  Benchmark 4: Thermal Profile (60s)\n");
    printf("  Continuous ANE matmul 1024x1024\n");
    printf("  Recording ms/eval every 5 seconds\n");
    printf("========================================\n");

    int ic = 1024, oc = 1024, seq = 256;
    NSString *mil = gen_dyn_matmul_mil(ic, oc, seq);
    int in_bytes  = ic * (seq + oc) * 2;
    int out_bytes = oc * seq * 2;

    Kern *k = compile_kern(mil, @{}, in_bytes, out_bytes);
    if (!k) { printf("COMPILE FAILED for thermal test\n"); return; }

    // Stage data
    IOSurfaceLock(k->ioIn, 0, NULL);
    _Float16 *in_buf = (_Float16*)IOSurfaceGetBaseAddress(k->ioIn);
    int in_elems = ic * (seq + oc);
    for (int i = 0; i < in_elems; i++) in_buf[i] = (_Float16)(drand48() * 0.02 - 0.01);
    IOSurfaceUnlock(k->ioIn, 0, NULL);

    // Warmup
    for (int i = 0; i < 20; i++) ane_run(k);

    double duration_s = 60.0;
    double sample_interval_s = 5.0;
    int sample_iters = 200;  // iters per sample for averaging

    NSMutableArray *thermal_samples = [NSMutableArray array];

    printf("\n%8s  %10s  %10s  %10s\n", "Time(s)", "ms/eval", "GFLOPS", "Status");
    printf("-----------------------------------------------\n");

    uint64_t wall_start = mach_absolute_time();
    double elapsed_s = 0;
    int sample_idx = 0;

    while (elapsed_s < duration_s) {
        // Measure a batch of iterations
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < sample_iters; i++) ane_run(k);
        double batch_ms = tb_ms(mach_absolute_time() - t0);
        double per_eval = batch_ms / sample_iters;

        elapsed_s = tb_ms(mach_absolute_time() - wall_start) / 1000.0;

        double flops = 2.0 * ic * oc * seq;
        double gflops = flops / per_eval / 1e6;

        // Detect throttling: if ms/eval > 2x the first sample
        const char *status = "OK";
        if (sample_idx > 0) {
            double first_ms = [thermal_samples[0] doubleValue];
            if (per_eval > first_ms * 2.0) status = "THROTTLED";
            else if (per_eval > first_ms * 1.3) status = "WARMING";
        }

        printf("%8.1f  %10.3f  %10.1f  %10s\n", elapsed_s, per_eval, gflops, status);

        [thermal_samples addObject:@(per_eval)];

        NSString *key = [NSString stringWithFormat:@"thermal_%ds_ms", (int)(sample_idx * sample_interval_s)];
        json_add_double(key, per_eval);

        sample_idx++;

        // Fill remaining time in this interval with continuous runs
        double target_s = sample_idx * sample_interval_s;
        while (tb_ms(mach_absolute_time() - wall_start) / 1000.0 < target_s &&
               tb_ms(mach_absolute_time() - wall_start) / 1000.0 < duration_s) {
            ane_run(k);
        }
        elapsed_s = tb_ms(mach_absolute_time() - wall_start) / 1000.0;
    }

    // Compute summary stats
    double min_ms = 1e9, max_ms = 0, sum_ms = 0;
    for (NSNumber *n in thermal_samples) {
        double v = [n doubleValue];
        if (v < min_ms) min_ms = v;
        if (v > max_ms) max_ms = v;
        sum_ms += v;
    }
    double avg_ms = sum_ms / [thermal_samples count];
    double drift_pct = (max_ms - min_ms) / min_ms * 100.0;

    printf("\nThermal Summary:\n");
    printf("  Min: %.3f ms  Max: %.3f ms  Avg: %.3f ms\n", min_ms, max_ms, avg_ms);
    printf("  Drift: %.1f%% %s\n", drift_pct,
           drift_pct > 100 ? "(SEVERE THROTTLING)" :
           drift_pct > 30 ? "(MODERATE THROTTLING)" :
           "(STABLE)");

    json_add_double(@"thermal_min_ms", min_ms);
    json_add_double(@"thermal_max_ms", max_ms);
    json_add_double(@"thermal_avg_ms", avg_ms);
    json_add_double(@"thermal_drift_pct", drift_pct);
    json_add_string(@"thermal_status",
                    drift_pct > 100 ? @"SEVERE_THROTTLING" :
                    drift_pct > 30 ? @"MODERATE_THROTTLING" :
                    @"STABLE");

    free_kern(k);
}

// ===== JSON Output =====
static void print_json_results(void) {
    printf("\n========================================\n");
    printf("  JSON Results\n");
    printf("========================================\n");

    // Build timestamp
    NSDateFormatter *fmt = [[NSDateFormatter alloc] init];
    [fmt setDateFormat:@"yyyy-MM-dd'T'HH:mm:ssZZZZZ"];
    NSString *timestamp = [fmt stringFromDate:[NSDate date]];

    // Get machine info
    NSProcessInfo *pi = [NSProcessInfo processInfo];
    NSString *os_version = [pi operatingSystemVersionString];
    NSString *host = [pi hostName];

    // Build top-level JSON
    NSMutableDictionary *top = [NSMutableDictionary dictionary];
    top[@"benchmark"] = @"autoane-v1";
    top[@"timestamp"] = timestamp;
    top[@"os_version"] = os_version;
    top[@"hostname"] = host;
    top[@"results"] = g_results;

    NSError *e = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:top
                                                  options:NSJSONWritingSortedKeys | NSJSONWritingPrettyPrinted
                                                    error:&e];
    if (json) {
        NSString *str = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
        printf("\n%s\n", [str UTF8String]);
    } else {
        printf("JSON serialization error: %s\n", [[e description] UTF8String]);
    }
}

// ===== Main =====
int main(int argc, char *argv[]) {
    @autoreleasepool {
    g_results = [NSMutableDictionary dictionary];

    printf("=== AutoANE Benchmark Suite v1 ===\n");
    printf("Running all characterization tests...\n");

    ane_init();
    srand48(42);

    // Parse optional flags
    BOOL skip_thermal = NO;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-thermal") == 0) skip_thermal = YES;
    }

    // Benchmark 1: ANE Matmul
    bench_ane_matmul(500);

    // Benchmark 2: CPU Matmul
    bench_cpu_matmul(1000);

    // Benchmark 3: IO Overhead
    bench_io_overhead();

    // Benchmark 4: Thermal Profile
    if (!skip_thermal) {
        bench_thermal_profile();
    } else {
        printf("\n[Skipping thermal profile (--skip-thermal)]\n");
    }

    // Print JSON output
    print_json_results();

    printf("\n=== AutoANE Benchmark Complete ===\n");
    }
    return 0;
}
