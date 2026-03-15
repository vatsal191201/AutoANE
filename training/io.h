// io.h — IOSurface helpers, NEON conversion, kernel compile/eval
// Updated for GQA (Qwen3-0.6B): Q_DIM != DIM, separate KV heads
#pragma once
#include "config.h"

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

// Blob builders for const weights (mask, rms)
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)safe_calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)safe_calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout, fp16 on surface)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// fp16 IOSurface I/O (for dynamic matmul kernels with fp16 input/output)
static void io_write_dyn(IOSurfaceRef s, const float *act, int ic, int seq,
                         const float *W, int oc) {
    int sp = seq + oc;
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < ic; d++) {
        cvt_f32_f16(buf + d*sp, act + d*seq, seq);
        cvt_f32_f16(buf + d*sp + seq, W + d*oc, oc);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// Write only activations to dynamic matmul surface (weights already staged)
static void io_write_dyn_acts(IOSurfaceRef s, const float *act, int ic, int seq, int sp) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < ic; d++)
        cvt_f32_f16(buf + d*sp, act + d*seq, seq);
    IOSurfaceUnlock(s, 0, NULL);
}

// Write ONLY activations to conv1x1 input surface: [1, IC, 1, SEQ] contiguous fp16
// No weights in the IOSurface — they are baked as BLOBFILE constants at compile time.
static void io_write_conv_acts(IOSurfaceRef s, const float *act, int ic, int seq) {
    IOSurfaceLock(s, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), act, ic * seq);
    IOSurfaceUnlock(s, 0, NULL);
}

// Read output from dynamic matmul kernel: [1, OC, 1, SEQ]
static void io_read_dyn(IOSurfaceRef s, float *out, int oc, int seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(out, (_Float16*)IOSurfaceGetBaseAddress(s), oc * seq);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Compile MIL to ANE kernel
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { printf("  [compile] model=NULL\n"); return NULL; }
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)safe_calloc(1, sizeof(Kern));
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
static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}

// Compile a conv1x1 kernel with weights baked as BLOBFILE constants.
// Weight layout: [OC, IC, 1, 1] as fp16.
// The returned Kern has IOSurfaces sized for activation-only I/O.
static Kern *compile_conv1x1_kern(NSString *mil, const float *W, int ic, int oc, int seq) {
    // Convert weight from fp32 [OC, IC] row-major to fp16 [OC, IC, 1, 1]
    // The [OC, IC, 1, 1] layout is just [OC, IC] contiguous — same memory layout
    int w_cnt = oc * ic;
    _Float16 *w_fp16 = (_Float16*)safe_malloc(w_cnt * 2);
    cvt_f32_f16(w_fp16, W, w_cnt);
    NSData *blob = build_blob_fp16(w_fp16, w_cnt);
    free(w_fp16);
    NSDictionary *weights = @{@"@model_path/weights/w.bin": @{@"data": blob}};
    // Input: [1, IC, 1, SEQ] fp16 = IC*SEQ*2 bytes (activation only)
    // Output: [1, OC, 1, SEQ] fp16 = OC*SEQ*2 bytes
    return compile_kern_mil_w(mil, weights, ic * seq * 2, oc * seq * 2);
}
static void ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void ane_eval_req(Kern *k, void *request) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}
static void *make_request(Kern *k, IOSurfaceRef ioIn) {
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);
    return (void*)CFBridgingRetain(req);
}

// ===== Per-layer weight staging for GQA =====
// sdpaFwd: [1, DIM, 1, SEQ + Q_DIM + KV_DIM + KV_DIM] fp16 — no Wo (separate kernel)
//   Wq: [DIM, Q_DIM], Wk: [DIM, KV_DIM], Wv: [DIM, KV_DIM]
#define SDPA_FWD_SP (SEQ + Q_DIM + KV_DIM + KV_DIM)
static void stage_sdpa_fwd_weights(IOSurfaceRef s, const float *Wq, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ,                   Wq + d*Q_DIM, Q_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM,             Wk + d*KV_DIM, KV_DIM);
        cvt_f32_f16(buf + d*SDPA_FWD_SP + SEQ+Q_DIM+KV_DIM,     Wv + d*KV_DIM, KV_DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_sdpa_fwd_acts(IOSurfaceRef s, const float *xnorm) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*SDPA_FWD_SP, xnorm + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// woFwd: [1, Q_DIM, 1, SEQ + DIM] fp16 — Wo: [Q_DIM, DIM]
#define WO_FWD_SP (SEQ + DIM)
static void stage_wo_fwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP + SEQ, Wo + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wo_fwd_acts(IOSurfaceRef s, const float *attn_out) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*WO_FWD_SP, attn_out + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnFused: [1, DIM, 1, 2*SEQ+3*HIDDEN] fp16
#define FFN_FUSED_SP (2*SEQ + 3*HIDDEN)
static void stage_ffn_fused_weights(IOSurfaceRef s,
                                     const float *W1t, const float *W3t, const float *W2_orig) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ,          W1t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+HIDDEN,   W3t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + 2*SEQ+2*HIDDEN, W2_orig + d*HIDDEN, HIDDEN);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_fused_acts(IOSurfaceRef s, const float *x2norm, const float *x2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP,       x2norm + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ, x2 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW2t: [1, DIM, 1, SEQ+HIDDEN] fp16
#define FFN_BWD_W2T_SP (SEQ + HIDDEN)
static void stage_ffn_bwd_w2t_weights(IOSurfaceRef s, const float *W2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP + SEQ, W2 + d*HIDDEN, HIDDEN);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w2t_acts(IOSurfaceRef s, const float *dffn) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_BWD_W2T_SP, dffn + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// ffnBwdW13t: [1, HIDDEN, 1, 2*SEQ+2*DIM] fp16
#define FFN_BWD_W13T_SP (2*SEQ + 2*DIM)
static void stage_ffn_bwd_w13t_weights(IOSurfaceRef s, const float *W1, const float *W3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ,       W1 + d*DIM, DIM);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + 2*SEQ + DIM, W3 + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_ffn_bwd_w13t_acts(IOSurfaceRef s, const float *dh1, const float *dh3) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < HIDDEN; d++) {
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP,       dh1 + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*FFN_BWD_W13T_SP + SEQ, dh3 + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// wotBwd: [1, DIM, 1, SEQ+Q_DIM] fp16 — Wo is [DIM, Q_DIM], matmul gives Wo^T @ dy
#define WOT_BWD_SP (SEQ + Q_DIM)
static void stage_wot_bwd_weights(IOSurfaceRef s, const float *Wo) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP + SEQ, Wo + d*Q_DIM, Q_DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_wot_bwd_acts(IOSurfaceRef s, const float *dy) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*WOT_BWD_SP, dy + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// qBwd: [1, Q_DIM, 1, SEQ+DIM] fp16 — Wq is [Q_DIM, DIM], matmul gives Wq^T @ dq
#define Q_BWD_SP (SEQ + DIM)
static void stage_q_bwd_weights(IOSurfaceRef s, const float *Wq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP + SEQ, Wq + d*DIM, DIM);
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_q_bwd_acts(IOSurfaceRef s, const float *dq) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < Q_DIM; d++)
        cvt_f32_f16(buf + d*Q_BWD_SP, dq + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// kvBwd: [1, KV_DIM, 1, 2*SEQ+2*DIM] fp16 — dk @ Wk + dv @ Wv → dx_kv
#define KV_BWD_SP (2*SEQ + 2*DIM)
static void stage_kv_bwd_weights(IOSurfaceRef s, const float *Wk, const float *Wv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ,       Wk + d*DIM, DIM);
        cvt_f32_f16(buf + d*KV_BWD_SP + 2*SEQ + DIM, Wv + d*DIM, DIM);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
static void write_kv_bwd_acts(IOSurfaceRef s, const float *dk, const float *dv) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < KV_DIM; d++) {
        cvt_f32_f16(buf + d*KV_BWD_SP,       dk + d*SEQ, SEQ);
        cvt_f32_f16(buf + d*KV_BWD_SP + SEQ, dv + d*SEQ, SEQ);
    }
    IOSurfaceUnlock(s, 0, NULL);
}

// Safe CFRelease that handles NULL (CFRelease(NULL) is undefined behavior)
static inline void safe_cfrelease(CFTypeRef ref) { if (ref) CFRelease(ref); }

// Free per-layer surfaces and requests
static void free_per_layer(PerLayerSurfaces *pls, PerLayerRequests *plr) {
    for (int L = 0; L < NLAYERS; L++) {
        // Fused forward surfaces/requests
        safe_cfrelease(pls[L].sdpaFwd_in); safe_cfrelease(pls[L].woFwd_in); safe_cfrelease(pls[L].ffnFused_in);
        safe_cfrelease(plr[L].sdpaFwd); safe_cfrelease(plr[L].woFwd); safe_cfrelease(plr[L].ffnFused);
        // Unfused forward surfaces/requests
        safe_cfrelease(pls[L].wqFwd_in); safe_cfrelease(pls[L].wkFwd_in); safe_cfrelease(pls[L].wvFwd_in);
        safe_cfrelease(pls[L].w1Fwd_in); safe_cfrelease(pls[L].w3Fwd_in); safe_cfrelease(pls[L].w2Fwd_in);
        safe_cfrelease(plr[L].wqFwd); safe_cfrelease(plr[L].wkFwd); safe_cfrelease(plr[L].wvFwd);
        safe_cfrelease(plr[L].w1Fwd); safe_cfrelease(plr[L].w3Fwd); safe_cfrelease(plr[L].w2Fwd);
        // Backward surfaces/requests
        safe_cfrelease(pls[L].ffnBwdW2t_in); safe_cfrelease(pls[L].ffnBwdW13t_in);
        safe_cfrelease(pls[L].wotBwd_in); safe_cfrelease(pls[L].qBwd_in); safe_cfrelease(pls[L].kvBwd_in);
        safe_cfrelease(plr[L].ffnBwdW2t); safe_cfrelease(plr[L].ffnBwdW13t);
        safe_cfrelease(plr[L].wotBwd); safe_cfrelease(plr[L].qBwd); safe_cfrelease(plr[L].kvBwd);
    }
}

// GQA helpers: tile KV from KV_HEADS to HEADS, and reduce HEADS to KV_HEADS
// tile_kv: input [KV_DIM, SEQ], output [Q_DIM, SEQ]
// Each KV head is duplicated GQA_RATIO times
static void gqa_tile_kv(float *out, const float *in, int seq) {
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            memcpy(out + q_head * HD * seq, in + kv * HD * seq, HD * seq * sizeof(float));
        }
    }
}
// reduce_kv: input [Q_DIM, SEQ], output [KV_DIM, SEQ]
// Sum contributions from Q heads sharing each KV head
static void gqa_reduce_kv(float *out, const float *in, int seq) {
    memset(out, 0, KV_DIM * seq * sizeof(float));
    for (int kv = 0; kv < KV_HEADS; kv++) {
        for (int r = 0; r < GQA_RATIO; r++) {
            int q_head = kv * GQA_RATIO + r;
            const float *src = in + q_head * HD * seq;
            float *dst = out + kv * HD * seq;
            vDSP_vadd(src, 1, dst, 1, dst, 1, (vDSP_Length)(HD * seq));
        }
    }
}
