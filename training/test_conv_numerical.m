// test_conv_numerical.m — Verify conv1x1 and matmul produce identical numerical outputs
// on ANE for the SAME mathematical operation: y = W @ x
// Build: xcrun clang -O2 -framework Foundation -framework IOSurface -framework Accelerate -isysroot $(xcrun --show-sdk-path) -fobjc-arc -ldl -include dlfcn.h -o test_conv_num test_conv_numerical.m
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

// ===== MIL Generators =====
#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"

// Dynamic matmul: input [1,IC,1,SEQ+OC], weight packed in spatial dim
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq) {
    int sp = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sp];
    [m appendFormat:@"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, seq];
    [m appendFormat:@"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", seq];
    [m appendFormat:@"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc];
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

// Conv 1x1: weight baked as BLOBFILE [OC, IC, 1, 1]
static NSString *gen_conv1x1_mil(int ic, int oc, int seq) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, seq];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n", oc, ic, oc, ic];
    [m appendString:@"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"];
    [m appendString:@"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"];
    [m appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];
    [m appendString:@"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x)[name=string(\"y\")];\n", oc, seq];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ===== Numerical comparison =====
static void compare_outputs(const char *name, int ic, int oc, int seq) {
    printf("\n=== Numerical Comparison: %s (IC=%d, OC=%d, SEQ=%d) ===\n", name, ic, oc, seq);

    // Generate weight W in [OC, IC] layout (row-major)
    // Conv1x1 expects [OC, IC, 1, 1] — direct match
    // Matmul expects [IC, OC] in spatial dim — needs transpose
    int w_cnt = oc * ic;
    _Float16 *w_oc_ic = (_Float16*)malloc(w_cnt * 2);  // [OC, IC] layout
    for (int i = 0; i < w_cnt; i++) w_oc_ic[i] = (_Float16)(drand48() * 0.02 - 0.01);

    // Transpose to [IC, OC] for matmul
    _Float16 *w_ic_oc = (_Float16*)malloc(w_cnt * 2);
    for (int o = 0; o < oc; o++)
        for (int d = 0; d < ic; d++)
            w_ic_oc[d * oc + o] = w_oc_ic[o * ic + d];

    // Generate random activations
    _Float16 *acts = (_Float16*)malloc(ic * seq * 2);
    for (int i = 0; i < ic * seq; i++) acts[i] = (_Float16)(drand48() * 0.1 - 0.05);

    // Compile matmul kernel
    NSString *mm_mil = gen_dyn_matmul_mil(ic, oc, seq);
    Kern *k_mm = compile_kern(mm_mil, @{}, ic * (seq + oc) * 2, oc * seq * 2);
    if (!k_mm) { printf("Matmul compile FAILED\n"); free(w_oc_ic); free(w_ic_oc); free(acts); return; }

    // Compile conv kernel with weight as BLOBFILE
    NSString *cv_mil = gen_conv1x1_mil(ic, oc, seq);
    NSData *blob = build_blob_fp16(w_oc_ic, w_cnt);
    NSDictionary *weights = @{@"@model_path/weights/w.bin": @{@"data": blob}};
    Kern *k_cv = compile_kern(cv_mil, weights, ic * seq * 2, oc * seq * 2);
    if (!k_cv) { printf("Conv compile FAILED\n"); free_kern(k_mm); free(w_oc_ic); free(w_ic_oc); free(acts); return; }

    // Stage matmul input: [1, IC, 1, SEQ+OC] with acts and TRANSPOSED weight
    IOSurfaceLock(k_mm->ioIn, 0, NULL);
    {
        _Float16 *mm_in = (_Float16*)IOSurfaceGetBaseAddress(k_mm->ioIn);
        // Activations: [IC, SEQ] in channels-first
        for (int d = 0; d < ic; d++)
            for (int s = 0; s < seq; s++)
                mm_in[d * (seq + oc) + s] = acts[d * seq + s];
        // Weight: [IC, OC] transposed
        for (int d = 0; d < ic; d++)
            for (int o = 0; o < oc; o++)
                mm_in[d * (seq + oc) + seq + o] = w_ic_oc[d * oc + o];
    }
    IOSurfaceUnlock(k_mm->ioIn, 0, NULL);

    // Stage conv input: [1, IC, 1, SEQ] activations only
    IOSurfaceLock(k_cv->ioIn, 0, NULL);
    {
        _Float16 *cv_in = (_Float16*)IOSurfaceGetBaseAddress(k_cv->ioIn);
        memcpy(cv_in, acts, ic * seq * 2);
    }
    IOSurfaceUnlock(k_cv->ioIn, 0, NULL);

    // Run both
    ane_run(k_mm);
    ane_run(k_cv);

    // Compare outputs [1, OC, 1, SEQ]
    IOSurfaceLock(k_mm->ioOut, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(k_cv->ioOut, kIOSurfaceLockReadOnly, NULL);
    {
        _Float16 *mm_out = (_Float16*)IOSurfaceGetBaseAddress(k_mm->ioOut);
        _Float16 *cv_out = (_Float16*)IOSurfaceGetBaseAddress(k_cv->ioOut);

        float max_abs_diff = 0, sum_abs_diff = 0, max_rel_diff = 0;
        int n = oc * seq;
        int mismatch_count = 0;

        for (int i = 0; i < n; i++) {
            float mm_val = (float)mm_out[i];
            float cv_val = (float)cv_out[i];
            float abs_diff = fabsf(mm_val - cv_val);
            sum_abs_diff += abs_diff;
            if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
            float denom = fmaxf(fabsf(mm_val), fabsf(cv_val));
            if (denom > 1e-6f) {
                float rel = abs_diff / denom;
                if (rel > max_rel_diff) max_rel_diff = rel;
            }
            if (abs_diff > 0.01f) mismatch_count++;
        }

        // Also compute CPU reference using Accelerate
        float *cpu_acts = (float*)malloc(ic * seq * 4);
        float *cpu_w = (float*)malloc(w_cnt * 4);
        float *cpu_out = (float*)malloc(oc * seq * 4);
        for (int i = 0; i < ic * seq; i++) cpu_acts[i] = (float)acts[i];
        for (int i = 0; i < w_cnt; i++) cpu_w[i] = (float)w_oc_ic[i];

        // CPU: y = W @ x, W=[OC,IC], x=[IC,SEQ] → y=[OC,SEQ]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    oc, seq, ic, 1.0f, cpu_w, ic, cpu_acts, seq, 0.0f, cpu_out, seq);

        float max_cpu_mm = 0, max_cpu_cv = 0;
        for (int i = 0; i < n; i++) {
            float d1 = fabsf(cpu_out[i] - (float)mm_out[i]);
            float d2 = fabsf(cpu_out[i] - (float)cv_out[i]);
            if (d1 > max_cpu_mm) max_cpu_mm = d1;
            if (d2 > max_cpu_cv) max_cpu_cv = d2;
        }

        printf("Output elements: %d\n", n);
        printf("Matmul vs Conv:  max_abs=%.6f  mean_abs=%.6f  max_rel=%.6f  mismatches(>0.01)=%d\n",
               max_abs_diff, sum_abs_diff / n, max_rel_diff, mismatch_count);
        printf("CPU vs Matmul:   max_abs=%.6f\n", max_cpu_mm);
        printf("CPU vs Conv:     max_abs=%.6f\n", max_cpu_cv);
        printf("VERDICT: %s\n",
               max_abs_diff < 0.05f ? "PASS (within fp16 tolerance)" : "FAIL (significant numerical difference)");

        // Print first few values for sanity
        printf("First 5 values:\n");
        for (int i = 0; i < 5 && i < n; i++) {
            printf("  [%d] matmul=%.4f  conv=%.4f  cpu=%.4f  diff=%.6f\n",
                   i, (float)mm_out[i], (float)cv_out[i], cpu_out[i],
                   fabsf((float)mm_out[i] - (float)cv_out[i]));
        }

        free(cpu_acts); free(cpu_w); free(cpu_out);
    }
    IOSurfaceUnlock(k_mm->ioOut, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(k_cv->ioOut, kIOSurfaceLockReadOnly, NULL);

    free_kern(k_mm);
    free_kern(k_cv);
    free(w_oc_ic);
    free(w_ic_oc);
    free(acts);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
    ane_init();
    srand48(42);

    printf("=== Conv1x1 vs Matmul Numerical Correctness Test ===\n");
    printf("Tests that both ANE primitives produce the same result\n");
    printf("for the same mathematical operation y = W @ x\n\n");

    // SmolLM2-360M projection shapes
    compare_outputs("Wq (DIM->Q_DIM)",   960,  960,  256);
    compare_outputs("Wk (DIM->KV_DIM)",  960,  320,  256);
    compare_outputs("Wo (Q_DIM->DIM)",   960,  960,  256);
    compare_outputs("W1 (DIM->HIDDEN)",  960,  2560, 256);
    compare_outputs("W2 (HIDDEN->DIM)",  2560, 960,  256);

    printf("\n=== Done ===\n");
    }
    return 0;
}
