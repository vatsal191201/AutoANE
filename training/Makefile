CC = xcrun clang
CFLAGS = -O2 -DACCELERATE_NEW_LAPACK -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc \
         -fstack-protector-strong -D_FORTIFY_SOURCE=2

# Model selection: make MODEL=smollm2_360m
#   Available models:
#     stories110m     - 109M params, 12 layers, MHA 12/12, fastest
#     smollm2_135m    - 135M params, 30 layers, GQA 9/3
#     smollm2_360m    - 362M params, 32 layers, GQA 15/5 (default)
#     autoresearch    - variable, auto-configured by train.py
MODEL ?= smollm2_360m
MODEL_HDR = models/$(MODEL).h

train: train.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@command -v xcrun >/dev/null 2>&1 || { echo "ERROR: Xcode Command Line Tools not found. Install with: xcode-select --install"; exit 1; }
	@echo "Building for model: $(MODEL)"
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -o train train.m

mezo: train_mezo.m config.h io.h cpu_ops.h mil_dynamic.h $(MODEL_HDR)
	@command -v xcrun >/dev/null 2>&1 || { echo "ERROR: Xcode Command Line Tools not found. Install with: xcode-select --install"; exit 1; }
	@echo "Building MeZO for model: $(MODEL)"
	$(CC) $(CFLAGS) -include $(MODEL_HDR) -o train_mezo train_mezo.m

benchmark: benchmark.m
	$(CC) $(CFLAGS) -include dlfcn.h -o benchmark benchmark.m

clean:
	rm -f train train_mezo benchmark

.PHONY: clean
