// bench_conv_vs_matmul.m — ANE microbenchmark: conv 1x1 vs matmul
// Tests the Orion paper claim that conv 1x1 is ~3x faster than matmul on ANE.
// Build: xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate -isysroot $(xcrun --show-sdk-path) -fobjc-arc -o bench_conv bench_conv_vs_matmul.m
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>

// ===== Minimal ANE runtime =====
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

// ===== MIL Generators =====

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

// Conv 1x1: y = conv(x, W), weight baked as const BLOBFILE
static NSString *gen_conv1x1_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];

    // Weight as baked const
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", oc, ic, oc, ic];

    // Conv params
    [m appendString:@"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"];
    [m appendString:@"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [m appendString:@"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"];

    // Conv 1x1
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x)[name=string(\"y\")];\n", oc, seq];

    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== Delta compilation test =====
static void test_delta_reload(int ic, int oc, int seq) {
    printf("\n=== Delta Compilation Test (unload -> write BLOBFILE -> reload) ===\n");

    int w_cnt = oc * ic;
    _Float16 *w1 = (_Float16*)malloc(w_cnt * 2);
    _Float16 *w2 = (_Float16*)malloc(w_cnt * 2);
    for (int i = 0; i < w_cnt; i++) {
        w1[i] = (_Float16)(drand48() * 0.02 - 0.01);
        w2[i] = (_Float16)(drand48() * 0.1 - 0.05);
    }

    NSString *mil = gen_conv1x1_mil(ic, oc, seq);
    NSData *blob1 = build_blob_fp16(w1, w_cnt);
    NSDictionary *weights1 = @{@"@model_path/weights/w.bin": @{@"data": blob1}};

    printf("Compiling conv 1x1 kernel [%d,%d,1,1]...\n", oc, ic);
    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern(mil, weights1, ic * seq * 2, oc * seq * 2);
    double compile_ms = tb_ms(mach_absolute_time() - t0);
    if (!k) { printf("COMPILE FAILED\n"); free(w1); free(w2); return; }
    printf("Compiled in %.1f ms\n", compile_ms);

    // Write test input
    IOSurfaceLock(k->ioIn, 0, NULL);
    _Float16 *in_buf = (_Float16*)IOSurfaceGetBaseAddress(k->ioIn);
    for (int i = 0; i < ic * seq; i++) in_buf[i] = (_Float16)0.5;
    IOSurfaceUnlock(k->ioIn, 0, NULL);

    // Run with w1
    ane_run(k);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    _Float16 *out_raw = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
    float out1_sum = 0;
    for (int i = 0; i < oc * seq; i++) out1_sum += (float)out_raw[i];
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Output with w1: sum = %.4f\n", out1_sum);

    // === Delta reload ===
    id mdl = (__bridge id)k->model;
    NSError *e = nil;

    // Step 1: Unload
    t0 = mach_absolute_time();
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    double unload_ms = tb_ms(mach_absolute_time() - t0);

    // Step 2: Write new weights
    t0 = mach_absolute_time();
    NSData *blob2 = build_blob_fp16(w2, w_cnt);
    NSString *td = (__bridge NSString*)k->tmpDir;
    NSString *wpath = [td stringByAppendingPathComponent:@"weights/w.bin"];
    [blob2 writeToFile:wpath atomically:NO];
    double write_ms = tb_ms(mach_absolute_time() - t0);

    // Step 3: Reload
    t0 = mach_absolute_time();
    e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double reload_ms = tb_ms(mach_absolute_time() - t0);

    if (!ok) {
        printf("RELOAD FAILED: %s\n", e ? [[e description] UTF8String] : "?");
        printf("Attempting recompile as fallback...\n");
        t0 = mach_absolute_time();
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (ok) {
            e = nil;
            ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        double recompile_ms = tb_ms(mach_absolute_time() - t0);
        printf("Recompile: %.1f ms (ok=%d)\n", recompile_ms, ok);
    }

    printf("Delta: unload=%.2fms write=%.2fms reload=%.2fms total=%.2fms\n",
           unload_ms, write_ms, reload_ms, unload_ms + write_ms + reload_ms);

    // Run with w2
    ane_run(k);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    out_raw = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
    float out2_sum = 0;
    for (int i = 0; i < oc * seq; i++) out2_sum += (float)out_raw[i];
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Output with w2: sum = %.4f\n", out2_sum);

    if (fabsf(out1_sum - out2_sum) > 0.01f) {
        printf("SUCCESS: Weights changed (%.4f vs %.4f)\n", out1_sum, out2_sum);
    } else {
        printf("WARNING: Outputs similar — new weights may not have loaded\n");
    }

    free_kern(k);
    free(w1); free(w2);
}

// ===== Enhanced Delta Compilation Test =====
// Tests multiple approaches to delta weight update:
// 1. Original: unload → write tmpDir → reload (known to fail)
// 2. New: unload → write tmpDir → recompile → reload (measure overhead)
// 3. New: enumerate files in tmpDir and ANE cache to understand structure
// 4. New: try creating fresh model with same MIL + new weights

static void enumerate_dir(NSString *path, int depth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *e = nil;
    NSArray *items = [fm contentsOfDirectoryAtPath:path error:&e];
    if (!items) return;
    for (NSString *item in items) {
        NSString *full = [path stringByAppendingPathComponent:item];
        BOOL isDir = NO;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        NSDictionary *attrs = [fm attributesOfItemAtPath:full error:nil];
        unsigned long long sz = [attrs fileSize];
        for (int i = 0; i < depth; i++) printf("  ");
        printf("%s%s (%llu bytes)\n", [item UTF8String], isDir ? "/" : "", sz);
        if (isDir && depth < 3) enumerate_dir(full, depth + 1);
    }
}

static void test_delta_enhanced(int ic, int oc, int seq) {
    printf("\n=== Enhanced Delta Compilation Test ===\n");
    printf("Shape: IC=%d OC=%d SEQ=%d\n", ic, oc, seq);

    int w_cnt = oc * ic;
    _Float16 *w1 = (_Float16*)malloc(w_cnt * 2);
    _Float16 *w2 = (_Float16*)malloc(w_cnt * 2);
    for (int i = 0; i < w_cnt; i++) {
        w1[i] = (_Float16)1.0;  // All 1s — easy to verify
        w2[i] = (_Float16)2.0;  // All 2s — output should double
    }

    NSString *mil = gen_conv1x1_mil(ic, oc, seq);
    NSData *blob1 = build_blob_fp16(w1, w_cnt);
    NSDictionary *weights1 = @{@"@model_path/weights/w.bin": @{@"data": blob1}};

    printf("\n--- Step 1: Initial compile with w1 (all 1.0) ---\n");
    uint64_t t0 = mach_absolute_time();
    Kern *k = compile_kern(mil, weights1, ic * seq * 2, oc * seq * 2);
    double compile1_ms = tb_ms(mach_absolute_time() - t0);
    if (!k) { printf("COMPILE FAILED\n"); free(w1); free(w2); return; }
    printf("Initial compile: %.1f ms\n", compile1_ms);

    // Enumerate tmpDir contents
    NSString *td = (__bridge NSString*)k->tmpDir;
    printf("\ntmpDir = %s\n", [td UTF8String]);
    printf("tmpDir contents after compile:\n");
    enumerate_dir(td, 1);

    // Search for e5bundlecache
    printf("\nSearching for ANE cache directories...\n");
    NSString *cacheDir = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    printf("User cache dir: %s\n", [cacheDir UTF8String]);
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *cacheDirs = [fm contentsOfDirectoryAtPath:cacheDir error:nil];
    for (NSString *d in cacheDirs) {
        if ([d containsString:@"e5"] || [d containsString:@"ane"] || [d containsString:@"ANE"] || [d containsString:@"neural"]) {
            printf("  Found: %s/\n", [d UTF8String]);
            enumerate_dir([cacheDir stringByAppendingPathComponent:d], 2);
        }
    }
    // Also check ~/Library/Caches explicitly
    NSString *libCache = [@"~/Library/Caches" stringByExpandingTildeInPath];
    NSArray *libCacheDirs = [fm contentsOfDirectoryAtPath:libCache error:nil];
    for (NSString *d in libCacheDirs) {
        if ([d containsString:@"e5"] || [d containsString:@"ane"] || [d containsString:@"ANE"] || [d containsString:@"neural"] || [d containsString:@"apple"]) {
            printf("  Found in ~/Library/Caches/: %s/\n", [d UTF8String]);
            NSString *subpath = [libCache stringByAppendingPathComponent:d];
            enumerate_dir(subpath, 2);
        }
    }

    // Write test input (all 0.5)
    IOSurfaceLock(k->ioIn, 0, NULL);
    _Float16 *in_buf = (_Float16*)IOSurfaceGetBaseAddress(k->ioIn);
    for (int i = 0; i < ic * seq; i++) in_buf[i] = (_Float16)0.5;
    IOSurfaceUnlock(k->ioIn, 0, NULL);

    // Run with w1
    ane_run(k);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    _Float16 *out = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
    float sum1 = 0;
    for (int i = 0; i < oc * seq; i++) sum1 += (float)out[i];
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("\n--- Step 2: Run with w1 ---\n");
    printf("Output sum (w1=1.0): %.4f (expected: %.4f)\n", sum1, (float)ic * 0.5f * oc * seq);

    // === Approach 1: unload → write → reload (known to fail) ===
    printf("\n--- Step 3: Approach 1 — unload → write tmpDir → reload ---\n");
    id mdl = (__bridge id)k->model;
    NSError *e = nil;

    t0 = mach_absolute_time();
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    double unload_ms = tb_ms(mach_absolute_time() - t0);

    NSData *blob2 = build_blob_fp16(w2, w_cnt);
    NSString *wpath = [td stringByAppendingPathComponent:@"weights/w.bin"];
    [blob2 writeToFile:wpath atomically:NO];

    t0 = mach_absolute_time();
    e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double reload_ms = tb_ms(mach_absolute_time() - t0);
    printf("unload=%.2fms reload=%.2fms ok=%d\n", unload_ms, reload_ms, ok);

    ane_run(k);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    out = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
    float sum_a1 = 0;
    for (int i = 0; i < oc * seq; i++) sum_a1 += (float)out[i];
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Output sum after reload: %.4f %s\n", sum_a1,
           fabsf(sum_a1 - sum1) > 0.01f ? "CHANGED!" : "(unchanged — reload didn't pick up new weights)");

    // === Approach 2: unload → write → recompile → load ===
    printf("\n--- Step 4: Approach 2 — unload → write tmpDir → recompile → load ---\n");
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);

    // w2 already written above, write again to be sure
    [blob2 writeToFile:wpath atomically:NO];

    t0 = mach_absolute_time();
    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    double recompile_ms = tb_ms(mach_absolute_time() - t0);
    if (!ok) printf("  recompile FAIL: %s\n", e ? [[e description] UTF8String] : "?");

    t0 = mach_absolute_time();
    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double load2_ms = tb_ms(mach_absolute_time() - t0);
    printf("recompile=%.2fms load=%.2fms ok=%d\n", recompile_ms, load2_ms, ok);
    printf("Recompile overhead vs initial: %.1f%% (%.1fms vs %.1fms)\n",
           recompile_ms / compile1_ms * 100, recompile_ms, compile1_ms);

    ane_run(k);
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    out = (_Float16*)IOSurfaceGetBaseAddress(k->ioOut);
    float sum_a2 = 0;
    for (int i = 0; i < oc * seq; i++) sum_a2 += (float)out[i];
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Output sum after recompile: %.4f (expected with w2=2.0: %.4f) %s\n",
           sum_a2, (float)ic * 0.5f * 2.0f * oc * seq,
           fabsf(sum_a2 - sum1) > 0.01f ? "WEIGHTS UPDATED!" : "SAME — recompile didn't help");

    // === Approach 3: Full fresh compile with new weights ===
    printf("\n--- Step 5: Approach 3 — fresh compile from scratch with w2 ---\n");
    free_kern(k);

    NSDictionary *weights2 = @{@"@model_path/weights/w.bin": @{@"data": blob2}};
    t0 = mach_absolute_time();
    Kern *k2 = compile_kern(mil, weights2, ic * seq * 2, oc * seq * 2);
    double compile2_ms = tb_ms(mach_absolute_time() - t0);
    if (!k2) { printf("Fresh compile FAILED\n"); free(w1); free(w2); return; }
    printf("Fresh compile: %.1f ms\n", compile2_ms);

    IOSurfaceLock(k2->ioIn, 0, NULL);
    in_buf = (_Float16*)IOSurfaceGetBaseAddress(k2->ioIn);
    for (int i = 0; i < ic * seq; i++) in_buf[i] = (_Float16)0.5;
    IOSurfaceUnlock(k2->ioIn, 0, NULL);

    ane_run(k2);
    IOSurfaceLock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
    out = (_Float16*)IOSurfaceGetBaseAddress(k2->ioOut);
    float sum_fresh = 0;
    for (int i = 0; i < oc * seq; i++) sum_fresh += (float)out[i];
    IOSurfaceUnlock(k2->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Output sum (fresh w2): %.4f (expected: %.4f) %s\n",
           sum_fresh, (float)ic * 0.5f * 2.0f * oc * seq,
           fabsf(sum_fresh - sum1 * 2.0f) < sum1 * 0.01f ? "CORRECT — 2x of w1" : "UNEXPECTED");

    // === Approach 4: Patch 'data' file in tmpDir ===
    // The compilation creates a 'data' file in tmpDir that's the same size as the weight blob.
    // If we overwrite this file and reload, the ANE might pick up new weights.
    printf("\n--- Step 6: Approach 4 — patch tmpDir/data + reload ---\n");

    // First, compile fresh with w1 again to get a clean baseline
    NSDictionary *weights1_fresh = @{@"@model_path/weights/w.bin": @{@"data": blob1}};
    Kern *k3 = compile_kern(mil, weights1_fresh, ic * seq * 2, oc * seq * 2);
    if (!k3) { printf("Fresh w1 compile FAILED\n"); free_kern(k2); free(w1); free(w2); return; }

    IOSurfaceLock(k3->ioIn, 0, NULL);
    in_buf = (_Float16*)IOSurfaceGetBaseAddress(k3->ioIn);
    for (int i = 0; i < ic * seq; i++) in_buf[i] = (_Float16)0.5;
    IOSurfaceUnlock(k3->ioIn, 0, NULL);

    // Verify baseline
    ane_run(k3);
    IOSurfaceLock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
    out = (_Float16*)IOSurfaceGetBaseAddress(k3->ioOut);
    float sum_baseline = 0;
    for (int i = 0; i < oc * seq; i++) sum_baseline += (float)out[i];
    IOSurfaceUnlock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
    printf("Baseline (w1): %.4f\n", sum_baseline);

    NSString *td3 = (__bridge NSString*)k3->tmpDir;

    // List files in tmpDir
    printf("tmpDir contents:\n");
    enumerate_dir(td3, 1);

    // Read existing data file to understand format
    NSString *dataPath = [td3 stringByAppendingPathComponent:@"data"];
    NSData *origData = [NSData dataWithContentsOfFile:dataPath];
    printf("data file size: %lu bytes\n", (unsigned long)[origData length]);

    // Unload
    id mdl3 = (__bridge id)k3->model;
    e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl3, @selector(unloadWithQoS:error:), 21, &e);

    // Try 4a: Overwrite 'data' with w2 blob (raw fp16 without BLOBFILE header)
    printf("\n  4a: Write raw fp16 w2 to tmpDir/data...\n");
    NSMutableData *rawW2 = [NSMutableData dataWithLength:w_cnt * 2];
    memcpy([rawW2 mutableBytes], w2, w_cnt * 2);
    // But the data file might have a specific format. Let's preserve the header if any.
    if (origData && [origData length] >= (NSUInteger)(w_cnt * 2)) {
        // Check if data starts with BLOBFILE header
        const uint8_t *bytes = (const uint8_t*)[origData bytes];
        printf("  data[0..7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
               bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
        printf("  data[64..67]: %02x %02x %02x %02x (0xDEADBEEF = BLOB marker)\n",
               bytes[64], bytes[65], bytes[66], bytes[67]);

        // If it has a BLOBFILE header (0xDEADBEEF at offset 64), write blob2 format
        if (bytes[64] == 0xEF && bytes[65] == 0xBE && bytes[66] == 0xAD && bytes[67] == 0xDE) {
            printf("  data file has BLOBFILE header — writing blob2 format\n");
            [blob2 writeToFile:dataPath atomically:NO];
        } else {
            printf("  data file has unknown format — writing raw fp16\n");
            [rawW2 writeToFile:dataPath atomically:NO];
        }
    }

    t0 = mach_absolute_time();
    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl3, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double reload4a_ms = tb_ms(mach_absolute_time() - t0);
    printf("  reload: %.2fms ok=%d\n", reload4a_ms, ok);

    if (ok) {
        ane_run(k3);
        IOSurfaceLock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
        out = (_Float16*)IOSurfaceGetBaseAddress(k3->ioOut);
        float sum_4a = 0;
        for (int i = 0; i < oc * seq; i++) sum_4a += (float)out[i];
        IOSurfaceUnlock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
        printf("  Output sum: %.4f %s\n", sum_4a,
               fabsf(sum_4a - sum_baseline) > 0.01f ? "WEIGHTS CHANGED!" : "(unchanged)");
    }

    // Try 4b: Also overwrite weights/w.bin AND data, then reload
    printf("\n  4b: Write w2 to BOTH tmpDir/data AND tmpDir/weights/w.bin...\n");
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl3, @selector(unloadWithQoS:error:), 21, &e);
    [blob2 writeToFile:dataPath atomically:NO];
    [blob2 writeToFile:[td3 stringByAppendingPathComponent:@"weights/w.bin"] atomically:NO];

    t0 = mach_absolute_time();
    e = nil;
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl3, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    double reload4b_ms = tb_ms(mach_absolute_time() - t0);
    printf("  reload: %.2fms ok=%d\n", reload4b_ms, ok);

    if (ok) {
        ane_run(k3);
        IOSurfaceLock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
        out = (_Float16*)IOSurfaceGetBaseAddress(k3->ioOut);
        float sum_4b = 0;
        for (int i = 0; i < oc * seq; i++) sum_4b += (float)out[i];
        IOSurfaceUnlock(k3->ioOut, kIOSurfaceLockReadOnly, NULL);
        printf("  Output sum: %.4f %s\n", sum_4b,
               fabsf(sum_4b - sum_baseline) > 0.01f ? "WEIGHTS CHANGED!" : "(unchanged)");
    }

    // Try 4c: Also check e5bundlecache — find our kernel's entry and try patching there
    printf("\n  4c: Searching e5bundlecache for our compiled kernel...\n");
    NSString *e5cache = [@"~/Library/Caches/com.apple.e5rt.e5bundlecache" stringByExpandingTildeInPath];
    NSArray *e5versions = [fm contentsOfDirectoryAtPath:e5cache error:nil];
    for (NSString *ver in e5versions) {
        NSString *verPath = [e5cache stringByAppendingPathComponent:ver];
        NSArray *entries = [fm contentsOfDirectoryAtPath:verPath error:nil];
        printf("  e5bundlecache/%s: %lu entries\n", [ver UTF8String], (unsigned long)[entries count]);
        for (NSString *entry in entries) {
            NSString *entryPath = [verPath stringByAppendingPathComponent:entry];
            NSArray *files = [fm contentsOfDirectoryAtPath:entryPath error:nil];
            for (NSString *f in files) {
                NSString *fpath = [entryPath stringByAppendingPathComponent:f];
                NSDictionary *attrs = [fm attributesOfItemAtPath:fpath error:nil];
                unsigned long long sz = [attrs fileSize];
                if (sz > 100000) { // Only show large files (likely weight data)
                    printf("    %s/%s: %llu bytes\n", [entry UTF8String], [f UTF8String], sz);
                }
            }
        }
    }

    free_kern(k3);

    printf("\n--- Summary ---\n");
    printf("Initial compile:        %.1f ms\n", compile1_ms);
    printf("Reload only (no help):  %.1f ms\n", reload_ms);
    printf("Recompile (same model): %.1f ms (FAILED — sandbox)\n", recompile_ms);
    printf("Fresh compile (new):    %.1f ms\n", compile2_ms);
    printf("Patch data + reload:    %.1f ms\n", reload4a_ms);
    printf("w1 output sum:          %.4f\n", sum1);
    printf("After reload:           %.4f %s\n", sum_a1, fabsf(sum_a1-sum1)<0.01f ? "(no change)" : "(CHANGED)");
    printf("After recompile:        %.4f %s\n", sum_a2, fabsf(sum_a2-sum1)<0.01f ? "(no change)" : "(CHANGED)");
    printf("Fresh w2:               %.4f\n", sum_fresh);

    free_kern(k2);
    free(w1); free(w2);
}

// ===== Shape benchmark =====
static void bench_shape(const char *name, int ic, int oc, int seq, int iters) {
    printf("\n--- %s: IC=%d OC=%d SEQ=%d ---\n", name, ic, oc, seq);

    int w_cnt = oc * ic;
    _Float16 *w = (_Float16*)malloc(w_cnt * 2);
    for (int i = 0; i < w_cnt; i++) w[i] = (_Float16)(drand48() * 0.02 - 0.01);

    // Compile matmul kernel
    NSString *matmul_mil = gen_dyn_matmul_mil(ic, oc, seq);
    uint64_t t0 = mach_absolute_time();
    Kern *k_mm = compile_kern(matmul_mil, @{}, ic * (seq + oc) * 2, oc * seq * 2);
    double mm_compile = tb_ms(mach_absolute_time() - t0);
    if (!k_mm) { printf("Matmul compile FAILED\n"); free(w); return; }

    // Compile conv kernel
    NSString *conv_mil = gen_conv1x1_mil(ic, oc, seq);
    NSData *blob = build_blob_fp16(w, w_cnt);
    NSDictionary *weights = @{@"@model_path/weights/w.bin": @{@"data": blob}};
    t0 = mach_absolute_time();
    Kern *k_cv = compile_kern(conv_mil, weights, ic * seq * 2, oc * seq * 2);
    double cv_compile = tb_ms(mach_absolute_time() - t0);
    if (!k_cv) { printf("Conv compile FAILED\n"); free_kern(k_mm); free(w); return; }

    printf("Compile: matmul=%.1fms conv=%.1fms\n", mm_compile, cv_compile);

    // Stage data
    IOSurfaceLock(k_mm->ioIn, 0, NULL);
    _Float16 *mm_in = (_Float16*)IOSurfaceGetBaseAddress(k_mm->ioIn);
    for (int i = 0; i < ic * (seq + oc); i++) mm_in[i] = (_Float16)0.01;
    for (int d = 0; d < ic; d++)
        for (int o = 0; o < oc; o++)
            mm_in[d * (seq + oc) + seq + o] = w[d * oc + o];
    IOSurfaceUnlock(k_mm->ioIn, 0, NULL);

    IOSurfaceLock(k_cv->ioIn, 0, NULL);
    _Float16 *cv_in = (_Float16*)IOSurfaceGetBaseAddress(k_cv->ioIn);
    for (int i = 0; i < ic * seq; i++) cv_in[i] = (_Float16)0.01;
    IOSurfaceUnlock(k_cv->ioIn, 0, NULL);

    // Warmup
    for (int i = 0; i < 10; i++) { ane_run(k_mm); ane_run(k_cv); }

    // Bench matmul
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_run(k_mm);
    double mm_per = tb_ms(mach_absolute_time() - t0) / iters;

    // Bench conv
    t0 = mach_absolute_time();
    for (int i = 0; i < iters; i++) ane_run(k_cv);
    double cv_per = tb_ms(mach_absolute_time() - t0) / iters;

    double speedup = mm_per / cv_per;
    double flops = 2.0 * ic * oc * seq;

    printf("Matmul: %.3f ms  (%.1f GFLOPS)\n", mm_per, flops / mm_per / 1e6);
    printf("Conv:   %.3f ms  (%.1f GFLOPS)\n", cv_per, flops / cv_per / 1e6);
    printf("Speedup: %.2fx (%s)\n", speedup, speedup > 1.0 ? "conv wins" : "matmul wins");

    free_kern(k_mm);
    free_kern(k_cv);
    free(w);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
    ane_init();
    srand48(42);

    int iters = 500;
    printf("=== ANE Conv 1x1 vs Matmul Benchmark (%d iters) ===\n", iters);

    // SmolLM2-360M shapes: DIM=960, Q_DIM=960, KV_DIM=320, HIDDEN=2560, SEQ=256
    bench_shape("Wq (DIM->Q_DIM)",    960,  960,  256, iters);
    bench_shape("Wk (DIM->KV_DIM)",   960,  320,  256, iters);
    bench_shape("Wo (Q_DIM->DIM)",    960,  960,  256, iters);
    bench_shape("W1 (DIM->HIDDEN)",   960,  2560, 256, iters);
    bench_shape("W2 (HIDDEN->DIM)",   2560, 960,  256, iters);

    // Run enhanced delta compilation test
    test_delta_enhanced(1024, 1024, 256);

    printf("\n=== Done ===\n");
    }
    return 0;
}
