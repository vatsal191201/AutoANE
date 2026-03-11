// test_aneclient.m — Investigate _ANEClient API for delta compilation (E34)
// Build: xcrun clang -O2 -framework Foundation -framework IOSurface -isysroot $(xcrun --show-sdk-path) -fobjc-arc -include dlfcn.h -o test_aneclient test_aneclient.m
//
// This tests whether _ANEClient's loadModelNewInstance can update weights
// without recompilation (delta compilation hypothesis from Orion paper).

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <mach/mach_time.h>

static double tb_ms(uint64_t dt) {
    static mach_timebase_info_data_t tb = {0};
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)dt * tb.numer / tb.denom / 1e6;
}

// Generate a simple matmul MIL program for testing
static NSString *gen_test_mil(int ic, int oc, int seq) {
    int sp = seq + oc;
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
     "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
     "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
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

static IOSurfaceRef make_surface(size_t bytes) {
    NSDictionary *props = @{@"IOSurfaceWidth": @(bytes), @"IOSurfaceHeight": @1,
                            @"IOSurfaceBytesPerElement": @1, @"IOSurfacePixelFormat": @0};
    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
    printf("=== E34: _ANEClient API Investigation ===\n\n");

    // Load framework
    void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) { printf("FATAL: Cannot load ANE framework\n"); return 1; }

    // Get classes
    Class ANEClient = NSClassFromString(@"_ANEClient");
    Class ANEModel = NSClassFromString(@"_ANEModel");
    Class ANEInMemModel = NSClassFromString(@"_ANEInMemoryModel");
    Class ANEInMemDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class ANERequest = NSClassFromString(@"_ANERequest");
    Class ANEIOSurf = NSClassFromString(@"_ANEIOSurfaceObject");

    printf("Classes found:\n");
    printf("  _ANEClient: %s\n", ANEClient ? "YES" : "NO");
    printf("  _ANEModel: %s\n", ANEModel ? "YES" : "NO");
    printf("  _ANEInMemoryModel: %s\n", ANEInMemModel ? "YES" : "NO");
    printf("  _ANEInMemoryModelDescriptor: %s\n", ANEInMemDesc ? "YES" : "NO");
    printf("  _ANERequest: %s\n", ANERequest ? "YES" : "NO");
    printf("  _ANEIOSurfaceObject: %s\n", ANEIOSurf ? "YES" : "NO");

    // === Test 1: Get _ANEClient shared connection ===
    printf("\n--- Test 1: _ANEClient shared connection ---\n");
    id client = ((id(*)(Class,SEL))objc_msgSend)(ANEClient, @selector(sharedConnection));
    printf("  ANEClient: %s\n", client ? [[client description] UTF8String] : "NULL");

    if (!client) {
        printf("FATAL: Cannot get ANEClient shared connection\n");
        return 1;
    }

    // === Test 2: Compile via _ANEInMemoryModel (baseline) ===
    printf("\n--- Test 2: Baseline compile via _ANEInMemoryModel ---\n");
    int ic = 256, oc = 256, seq = 64;  // Small for quick testing
    NSString *mil = gen_test_mil(ic, oc, seq);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        ANEInMemDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{}, nil);
    printf("  Descriptor: %s\n", desc ? "OK" : "NULL");

    id inMemModel = ((id(*)(Class,SEL,id))objc_msgSend)(ANEInMemModel, @selector(inMemoryModelWithDescriptor:), desc);
    printf("  InMemoryModel: %s\n", inMemModel ? "OK" : "NULL");

    // Get hex identifier
    id hexId = ((id(*)(id,SEL))objc_msgSend)(inMemModel, @selector(hexStringIdentifier));
    printf("  Hex ID: %s\n", hexId ? [hexId UTF8String] : "NULL");

    // Write model files to temp dir
    NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
    [[NSFileManager defaultManager] createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
                                withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    // Compile
    uint64_t t0 = mach_absolute_time();
    NSError *err = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        inMemModel, @selector(compileWithQoS:options:error:), 21, @{}, &err);
    double compile_ms = tb_ms(mach_absolute_time() - t0);
    printf("  Compile: %s (%.1f ms)\n", ok ? "OK" : "FAIL", compile_ms);
    if (!ok) printf("  Error: %s\n", [[err description] UTF8String]);

    // Load
    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        inMemModel, @selector(loadWithQoS:options:error:), 21, @{}, &err);
    double load_ms = tb_ms(mach_absolute_time() - t0);
    printf("  Load: %s (%.1f ms)\n", ok ? "OK" : "FAIL", load_ms);

    // Setup IO surfaces and run a test evaluation
    int in_bytes  = ic * (seq + oc) * 2;
    int out_bytes = oc * seq * 2;
    IOSurfaceRef ioIn = make_surface(in_bytes);
    IOSurfaceRef ioOut = make_surface(out_bytes);

    // Fill input with test data
    IOSurfaceLock(ioIn, 0, NULL);
    _Float16 *in_buf = (_Float16*)IOSurfaceGetBaseAddress(ioIn);
    for (int i = 0; i < ic * (seq + oc); i++) in_buf[i] = (_Float16)0.01;
    IOSurfaceUnlock(ioIn, 0, NULL);

    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ANEIOSurf, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(ANEIOSurf, @selector(objectWithIOSurface:), ioOut);
    id request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(ANERequest,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        inMemModel, @selector(evaluateWithQoS:options:request:error:), 21, @{}, request, &err);
    double eval_ms = tb_ms(mach_absolute_time() - t0);
    printf("  Evaluate: %s (%.3f ms)\n", ok ? "OK" : "FAIL", eval_ms);

    // === Test 3: Try _ANEClient's compileModel directly ===
    printf("\n--- Test 3: _ANEClient.compileModel ---\n");

    // Get the _ANEModel from the InMemoryModel
    id aneModel = ((id(*)(id,SEL))objc_msgSend)(inMemModel, @selector(model));
    printf("  _ANEModel from InMemModel: %s\n", aneModel ? "OK" : "NULL");

    if (aneModel) {
        // Try compiling via client
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(compileModel:options:qos:error:), aneModel, @{}, 21, &err);
        double client_compile_ms = tb_ms(mach_absolute_time() - t0);
        printf("  Client compile: %s (%.1f ms)\n", ok ? "OK" : "FAIL", client_compile_ms);
        if (!ok && err) printf("  Error: %s\n", [[err description] UTF8String]);
    }

    // === Test 4: Try loadModelNewInstance (delta compilation hypothesis) ===
    printf("\n--- Test 4: loadModelNewInstance (delta compilation test) ---\n");

    // First, unload the current model
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        inMemModel, @selector(unloadWithQoS:error:), 21, &err);

    // Create a slightly different MIL (same topology, different shape to test)
    // For true delta compilation, we'd want same topology + different weights
    // But dynamic matmul packs weights in input, so topology IS the same

    // Re-compile with same MIL (simulates weight update with same topology)
    id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        ANEInMemDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{}, nil);
    id inMemModel2 = ((id(*)(Class,SEL,id))objc_msgSend)(ANEInMemModel, @selector(inMemoryModelWithDescriptor:), desc2);

    id hexId2 = ((id(*)(id,SEL))objc_msgSend)(inMemModel2, @selector(hexStringIdentifier));
    printf("  New hex ID: %s\n", hexId2 ? [hexId2 UTF8String] : "NULL");
    printf("  Same as original: %s\n", [hexId isEqualToString:hexId2] ? "YES (topology cached!)" : "NO (different)");

    NSString *tmpDir2 = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId2];
    [[NSFileManager defaultManager] createDirectoryAtPath:[tmpDir2 stringByAppendingPathComponent:@"weights"]
                                withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[tmpDir2 stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        inMemModel2, @selector(compileWithQoS:options:error:), 21, @{}, &err);
    double recompile_ms = tb_ms(mach_absolute_time() - t0);
    printf("  Recompile (cached topology): %s (%.1f ms vs original %.1f ms)\n",
           ok ? "OK" : "FAIL", recompile_ms, compile_ms);

    t0 = mach_absolute_time();
    ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        inMemModel2, @selector(loadWithQoS:options:error:), 21, @{}, &err);
    double reload_ms = tb_ms(mach_absolute_time() - t0);
    printf("  Reload: %s (%.1f ms)\n", ok ? "OK" : "FAIL", reload_ms);

    // Try loadModelNewInstance on the client
    if (aneModel) {
        printf("\n  Trying _ANEClient.loadModelNewInstance...\n");
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(loadModelNewInstance:options:modelInstParams:qos:error:),
            aneModel, @{}, nil, 21, &err);
        double new_inst_ms = tb_ms(mach_absolute_time() - t0);
        printf("  loadModelNewInstance: %s (%.1f ms)\n", ok ? "OK" : "FAIL", new_inst_ms);
        if (!ok && err) printf("  Error: %s\n", [[err description] UTF8String]);
    }

    // === Test 5: Check compiledModelExistsFor ===
    printf("\n--- Test 5: Compiled model cache check ---\n");
    BOOL exists = ((BOOL(*)(id,SEL,id))objc_msgSend)(
        client, @selector(compiledModelExistsFor:), aneModel);
    printf("  compiledModelExistsFor: %s\n", exists ? "YES (cached)" : "NO");

    // === Summary ===
    printf("\n=== Summary ===\n");
    printf("First compile: %.1f ms\n", compile_ms);
    printf("Cached recompile: %.1f ms (%.1fx speedup)\n", recompile_ms, compile_ms / recompile_ms);
    printf("Load: %.1f ms\n", load_ms);
    printf("Reload: %.1f ms\n", reload_ms);
    printf("Eval: %.3f ms\n", eval_ms);

    // Cleanup
    CFRelease(ioIn);
    CFRelease(ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:tmpDir2 error:nil];

    printf("\n=== E34 Complete ===\n");
    }
    return 0;
}
