# What Can You Do With ANE That You Cannot Do (Or Cannot Do Well) With CPU/GPU Alone?

**Date**: 2026-03-17
**Author**: Brutally honest assessment based on extensive literature review and web research

---

## Executive Summary

ANE uniquely enables exactly **one thing**: sustained, low-power ML inference that runs independently of CPU/GPU without competing for their resources or draining the battery. This single property creates a small but genuine set of use cases that are impractical (not impossible) on CPU/GPU alone. For *training* specifically, ANE does not uniquely enable anything -- it offers a modest speedup (1.71x measured) at potentially lower power, but GPU/CPU can do the same job. The honest framing is: **ANE is a power-efficiency and resource-isolation play, not a capability play.**

---

## Part 1: What People Actually Use ANE For Today

### Shipping in Production (Real Products, Billions of Devices)

| Use Case | Runs on ANE? | Could GPU/CPU Do It? | Why ANE? |
|----------|-------------|---------------------|----------|
| Hey Siri wake-word detection | **No** -- runs on Always-On Processor (AOP), not ANE ([Apple ML Research](https://machinelearning.apple.com/research/hey-siri)) | N/A | The AOP is even lower power than ANE |
| Face ID recognition & adaptation | Yes (inference + on-device k-NN personalization) | Yes, but higher power | Privacy + always-available + low power |
| Computational photography (subject detection, depth, HDR) | Yes | Yes, but slower and higher power | Real-time requirement during camera use |
| On-device text recognition (Live Text) | Yes | Yes | Speed + battery during sustained use |
| Apple Intelligence 3B on-device model | Hybrid (CoreML dispatches across ANE+CPU+GPU) | Yes -- MLX runs same model on GPU only | CoreML uses ANE for power-efficient portions |
| Apple Watch gesture detection (double tap) | Yes -- 4-core ANE in S9 chip | CPU could, but battery impact severe | Watch battery is tiny; ANE's efficiency is critical |
| QuickType keyboard predictions | Partially (on-device inference) | Yes | Low latency + privacy |
| Siri speech recognition (on-device) | Yes | Yes | Battery life during extended voice interaction |

**Key finding**: Hey Siri -- the canonical "always-on ML" example -- does NOT run on the Neural Engine. It runs on a dedicated Always-On Processor with a tiny 32-unit DNN. The ANE is used for *burst inference* tasks (camera, text recognition, intelligence features) where power efficiency matters but "always-on" is not the mode.

### Third-Party / Research Use

| Use Case | Status | Notes |
|----------|--------|-------|
| ANEMLL (LLM inference on ANE) | Beta, open-source | 47-62 tok/s for 1B models; 9.3 tok/s for 8B; uses 500MB vs 8GB on GPU ([InsiderLLM](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)) |
| Orion (ANE training via reverse-engineered APIs) | Research prototype | 110M model, 1000 steps in 22 min; uses backprop, not forward-only ([arXiv 2603.06728](https://arxiv.org/abs/2603.06728)) |
| AutoANE (MeZO training on ANE) | Research prototype | 1.71x faster than CPU; first forward-only ANE training result |
| CoreML on-device personalization | Shipped since iOS 13 | Limited to k-NN and simple transfer learning; Apple calls it "personalization" not "training" |

---

## Part 2: Use Cases That Genuinely Need Always-On, Low-Power, Independent ML

### Tier 1: ANE Provides a REAL Advantage (Not Just "Nice to Have")

**1. Wearable Health Monitoring (Apple Watch)**

- **The case**: Apple Watch Series 9+ has a 4-core ANE. Battery is ~308 mAh. Every milliwatt matters. Continuous heart rate classification, fall detection, and gesture recognition need ML inference running 24/7 alongside the user's actual watch usage.
- **Does it REQUIRE ANE?** On a watch, effectively yes. The CPU/GPU power envelope is too large for continuous ML inference while maintaining >18hr battery life. The ANE's ~2W active / ~0mW idle characteristic is not optional -- it is a hard constraint.
- **Real product?** Yes. Shipping on hundreds of millions of Apple Watches.
- **Verdict: GENUINE ANE advantage.** GPU would kill the battery. CPU is too slow for real-time sensor fusion at acceptable power.

**2. Background ML Agents That Must Not Interfere With User Work**

- **The case**: A background agent summarizing notifications, monitoring messages, or running continuous document analysis while the user is doing GPU-heavy work (video editing, gaming, ML development). ANE runs on a separate power domain and separate silicon -- it physically cannot compete with the GPU for compute or memory bandwidth.
- **Does it REQUIRE ANE?** Not strictly -- you could timeslice the GPU. But on a laptop with 8-16GB unified memory, running an 8B model on GPU (consuming ~8GB) while the user runs Final Cut Pro is impractical. ANE uses ~500MB and zero GPU bandwidth.
- **Real product?** Apple Intelligence notification summaries run in background via CoreML, which uses ANE. Third-party: not yet, limited by CoreML's inflexibility.
- **Verdict: GENUINE ANE advantage.** Resource isolation is a real architectural property, not just a performance optimization. You cannot get "zero GPU interference" from the GPU.

**3. Battery-Constrained Extended Inference (Phones, Laptops on Battery)**

- **The case**: Running a local LLM assistant all day on a MacBook unplugged. ANE at 2W vs GPU at 20W means the difference between 4 hours and 12+ hours of battery life with continuous inference.
- **Does it REQUIRE ANE?** No -- GPU works fine if plugged in. But for the specific scenario of "all-day mobile ML inference," ANE's 10x power efficiency is not marginal -- it changes whether the product is viable.
- **Real product?** Apple Intelligence features on iPhone/iPad are the closest. No third-party "all-day background agent" ships yet.
- **Verdict: STRONG ANE advantage for mobile.** On desktop (plugged in), GPU is usually better due to higher throughput.

### Tier 2: ANE Is Helpful But Not Required

**4. Privacy-Sensitive On-Device Inference**

- **The case**: Health data, financial data, personal communications -- data that legally or ethically cannot leave the device.
- **Does it REQUIRE ANE?** No. CPU and GPU can run inference on-device just as privately. ANE adds efficiency but privacy is about *where* compute happens, not *which silicon* does it.
- **Real product?** Apple's entire on-device ML stack. But it runs on CPU+GPU+ANE hybrid, not ANE exclusively.
- **Verdict: Nice to have.** Privacy requires on-device compute. It does not require ANE specifically.

**5. Real-Time Camera/Video ML Pipelines**

- **The case**: Object detection, segmentation, depth estimation during video recording.
- **Does it REQUIRE ANE?** No -- Metal GPU compute shaders do this well. But using ANE frees the GPU for rendering/encoding, enabling higher quality video output.
- **Verdict: Nice to have.** A legitimate optimization, but GPU-only solutions exist and work.

### Tier 3: ANE Adds No Unique Value

**6. One-Shot Inference Tasks (Single Image Classification, etc.)**

- GPU is faster (2-5x), has better tooling, and power draw is irrelevant for a 100ms task.
- **Verdict: GPU is better.**

**7. Large Context Window LLM Inference**

- ANE caps at 2048-4096 token context. GPU handles 32K+ routinely.
- **Verdict: GPU is strictly better.**

---

## Part 3: What About Training? Does ANE Uniquely Enable Anything?

### Forward-Only Training (MeZO, Forward-Forward)

**The theoretical appeal**: ANE is inference-only hardware. MeZO needs only forward passes. Perfect match?

**The honest assessment**:

| Claim | Reality |
|-------|---------|
| "MeZO on ANE is the only way to train on inference hardware" | True but misleading. You can train on CPU/GPU with backprop, which converges faster. MeZO on ANE is a *choice*, not a *requirement*. |
| "MeZO achieves comparable accuracy to backprop" | Within 1-5% on most tasks ([Princeton NLP](https://princeton-nlp.github.io/mezo/)). Not better -- comparable at best, often slightly worse. |
| "MeZO can optimize non-differentiable objectives" | **True and genuinely unique.** Directly maximizing accuracy/F1/reward-model scores is impossible with backprop. But this advantage applies equally to MeZO on CPU/GPU -- it has nothing to do with ANE. |
| "MeZO uses inference-only memory" | True. But on a Mac with 16GB+ unified memory, this rarely matters. It matters on phones (4-8GB shared with OS). |
| "ANE training is faster than CPU" | True: 1.71x measured (AutoANE). But GPU is still faster for training (MLX backprop). |
| "ANE training is more power-efficient" | **Unvalidated.** Our measurements show ~12.5-13.3W package power across all modes (ANE, CPU, GPU). The promised 2.8W ANE-only power draw has not been reproduced during training workloads. |

**Where MeZO is genuinely better than backprop (not just comparable):**

1. **Non-differentiable objectives**: MeZO can directly maximize accuracy, F1, or reward scores for models up to 66B parameters. Backprop cannot do this at all without surrogate losses. This is a *real* advantage but has nothing to do with ANE.

2. **Memory-constrained settings**: On a device with N bytes of memory, MeZO can fine-tune a model that is ~12x larger than what backprop can handle ([arXiv 2511.11362](https://arxiv.org/abs/2511.11362)). On an 8GB iPhone, this means MeZO could fine-tune a 7B model where backprop can only handle ~600M. **This is genuinely useful on mobile devices where ANE is the most efficient inference engine.**

3. **Convergence on memory-constrained hardware**: When forced to use the same memory budget, MeZO with a larger model (Llama2-7B) reaches 82% accuracy on BoolQ while backprop with a smaller model (GPT2-Medium) reaches only 75%. The advantage is from fitting a bigger model, not from a better optimizer.

### The Forward-Forward Algorithm

- Hinton's Forward-Forward (FF) replaces backprop with two forward passes (positive + negative data).
- **Real advantages**: Biologically plausible; suitable for analog/low-power hardware; can learn when forward computation details are unknown.
- **Real limitations**: Only demonstrated on MNIST/CIFAR-10 (1.4% test error on MNIST). No production deployment exists. No evidence it works on modern LLMs or transformers at scale.
- **ANE relevance**: FF could theoretically run on ANE (forward-only). But FF is not mature enough for any real product. This is a 5-10 year research direction, not a near-term use case.
- **Verdict: Academic exercise today.** Potentially interesting for future neuromorphic hardware, not for ANE in 2026.

### Orion's Approach (Full Backprop on ANE)

- Orion demonstrated actual backpropagation on ANE via reverse-engineered APIs.
- 110M model, 1000 steps in 22 minutes on M4 Max.
- **This proves ANE CAN do backward passes** -- but at 0.656 TFLOPS (vs 19 TFLOPS for forward inference), it's using only 3.4% of ANE's theoretical throughput.
- **Verdict: Proof of concept.** GPU is far better for backprop training. ANE's architecture is optimized for forward inference; backward passes are a hack.

---

## Part 4: Continuous Inference That Doubles As Training

**The most interesting theoretical use case for ANE + MeZO.**

### The Concept

Test-time training (TTT) and test-time adaptation (TTA) blur the line between inference and training. The model improves *while serving users*. MeZO makes this feasible with forward passes only.

### What Exists Today

| System | What It Does | Hardware | Real Product? |
|--------|-------------|----------|---------------|
| TTT-E2E ([VentureBeat](https://venturebeat.com/infrastructure/new-test-time-training-method-lets-ai-keep-learning-without-exploding)) | Continual learning during inference; 3B models; matches full-attention quality | GPU | No -- research |
| TinyTTA ([NeurIPS 2024](https://neurips.cc/virtual/2024/poster/94778)) | Test-time adaptation on edge devices via early-exit ensembles | MCUs/edge | Research |
| On-demand TTA ([arXiv 2505.00986](https://arxiv.org/abs/2505.00986)) | Triggers adaptation only on detected domain shift | Edge devices | Research |
| Corun ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487435/)) | Concurrent inference + training on single GPU | GPU | Research |

### Could ANE Enable This Better Than GPU?

**The pitch**: ANE runs inference at 2W. MeZO turns inference into training. Therefore ANE could continuously adapt a model while serving users, using minimal power, without touching the GPU.

**The reality check**:
- MeZO requires *two* forward passes per training step (for finite-difference gradient estimation). This halves your inference throughput.
- MeZO convergence is slow: typically 10-100x more steps than backprop.
- On ANE: 9.3 tok/s for 8B models. Training while serving would drop this to ~4.7 tok/s. That is painfully slow for interactive use.
- On ANE with 1B models: 47-62 tok/s inference, ~23-31 tok/s during training-while-serving. Usable but only for small models.

**Verdict: Theoretically interesting. Practically marginal.** The real bottleneck is MeZO's slow convergence, not the hardware. If MeZO needed 10 steps to adapt, this would be killer. It needs 1000+.

---

## Part 5: Privacy-Sensitive On-Device Personalization

### What Apple Actually Ships

Apple has deployed federated learning for:
- Keyboard predictions (QuickType) -- since iOS 13 (2019) ([MIT Technology Review](https://www.technologyreview.com/2019/12/11/131629/apple-ai-personalizes-siri-federated-learning/))
- Siri speaker recognition -- federated, on-device
- App prediction / behavior modeling ([Apple ML Research](https://machinelearning.apple.com/research/federated-personalization))
- Foundation Models LoRA adapters (WWDC 2025) -- on-device adapter fine-tuning

**Critical detail**: Apple's federated learning uses standard backpropagation on CPU/GPU, not MeZO, and not the Neural Engine. The training happens when the device is idle, plugged in, and on WiFi. ANE is not involved in the training step.

### Could MeZO+ANE Improve This?

| Dimension | Current (CPU/GPU backprop) | MeZO+ANE | Better? |
|-----------|---------------------------|----------|---------|
| Power during training | ~10-20W | ~2-3W (claimed) | Yes, if validated |
| Memory for training | 3-5x inference | 1x inference | Yes -- fit larger models |
| Training speed | Fast (backprop) | Slow (10-100x more steps) | No |
| Accuracy | Backprop optimum | Within 1-5% | No |
| Can train while device in use | No (scheduled for idle) | Potentially yes (low power, no GPU) | **Possibly** |

**The genuine value proposition**: MeZO+ANE could enable training while the device is in active use (not just when idle/charging), because it doesn't touch CPU/GPU and uses minimal power. Current federated learning can only train when the device is idle. This is a real but incremental improvement.

**Verdict: Incremental improvement, not a revolution.** The training-while-active capability is genuinely new, but the practical impact is limited because MeZO needs many steps to converge and the accuracy ceiling is slightly lower.

---

## Part 6: Federated Learning -- ANE on Millions of Devices

### The Concept

2B+ Apple devices each have an ANE. If each contributes forward-pass compute via MeZO, the aggregate is enormous:
- 2B devices x 19 TFLOPS = 38 exaFLOPS theoretical peak
- Even at 0.1% utilization: 38 petaFLOPS

### The Reality

1. **Apple already does federated learning** -- using CPU/GPU backprop, not ANE ([Apple ML Research](https://machinelearning.apple.com/research/learning-with-privacy-at-scale)).

2. **Communication, not compute, is the bottleneck** in federated learning. Devices send model updates (not raw data) to a server. The bandwidth and latency of this aggregation dominates, not the on-device compute speed.

3. **MeZO's gradient estimates are noisier** than backprop gradients. Federated MeZO would need more communication rounds to converge, which worsens the bottleneck.

4. **FwdLLM** (arXiv 2308.13894) demonstrated forward-only federated LLM fine-tuning with 14.6x memory reduction. This is the closest existing work. But it targets commodity mobile devices in general, not ANE specifically.

5. **Scheduling is the hard problem**: Apple's FL system only trains when devices are idle, plugged in, on WiFi. The "aggregate 2B devices" number is theoretical -- real participation is a small fraction at any time.

**Verdict: ANE adds nothing unique here.** Federated learning's constraints are communication, privacy, and scheduling -- not on-device compute speed. CPU/GPU backprop with differential privacy is what ships and works.

---

## Part 7: The Honest Bottom Line

### What ANE Genuinely Enables That CPU/GPU Cannot (Or Cannot Do Well)

1. **Sustained low-power inference on battery-constrained devices** (especially Apple Watch, iPhone on battery). 10x better TFLOPS/W than GPU. This is real, shipping, and matters.

2. **ML inference that physically cannot interfere with CPU/GPU work.** Separate silicon, separate power domain. No timeslicing, no memory bandwidth competition. Real architectural advantage for background agents.

3. **Inference-time memory efficiency.** 500MB for 8B model (via quantization + ANE memory management) vs 8GB on GPU. On devices with 4-8GB total RAM, this can be the difference between running a model and not.

### What ANE Does NOT Uniquely Enable

1. **Training of any kind.** CPU/GPU with backprop is faster and more accurate. MeZO on ANE is a valid research direction, but it is slower than GPU backprop and achieves comparable-or-slightly-worse accuracy. The 1.71x speedup over CPU is real but modest.

2. **Privacy-preserving ML.** Privacy is about where compute happens (on-device vs cloud), not which chip does it.

3. **Federated learning.** Communication and scheduling dominate, not on-device compute.

4. **Forward-only training superiority.** MeZO's advantages (memory efficiency, non-differentiable objectives) apply equally on CPU/GPU. The ANE hardware is irrelevant to MeZO's algorithmic advantages.

5. **Test-time adaptation.** The concept is hardware-agnostic. ANE could run it at lower power, but GPU can run it faster.

### The Uncomfortable Truth About ANE + Training

**ANE does not uniquely enable anything for training.**

The argument for MeZO+ANE is:
- *"It's the only way to train on inference-only hardware"* -- True, but you don't need inference-only hardware. GPU works better.
- *"It's lower power for training"* -- Unvalidated. Our measurements show similar package power across all modes.
- *"It enables training while the device is in active use"* -- The best argument, but MeZO's slow convergence makes the practical value limited.

The honest framing: **ANE is excellent at what it was designed for -- efficient inference. Training on ANE is a clever hack, not a killer feature.** The research value is in proving it's possible and characterizing the hardware. The product value is marginal until MeZO convergence improves by 10-100x.

### Where the Real Opportunity Lies

If you accept that ANE's unique value is **always-on, low-power, GPU-independent inference**, then the product opportunities are:

1. **Always-on background intelligence agents** that process notifications, emails, documents continuously without battery drain or GPU contention. No such product exists yet. This is genuinely new territory.

2. **Continuous health/context monitoring** on wearables and phones -- not just sensor processing, but running actual ML models 24/7 on ambient data.

3. **On-device model serving** that frees GPU entirely for user-facing tasks (rendering, gaming, creative apps). The model runs on ANE; the GPU does what GPUs are good at.

4. **MeZO+ANE personalization during active use** -- not because it's better than backprop, but because it can happen *right now* without waiting for the device to be idle and charging. The delta is small per session, but it compounds over weeks.

None of these require training. They require efficient, sustained, low-power inference -- which is exactly what ANE does well.

---

## Sources

### Apple Research & Documentation
- [Hey Siri: On-device DNN-powered Voice Trigger](https://machinelearning.apple.com/research/hey-siri)
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Apple Foundation Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)
- [Federated Evaluation and Tuning for On-Device Personalization](https://machinelearning.apple.com/research/federated-personalization)
- [Learning with Privacy at Scale](https://machinelearning.apple.com/research/learning-with-privacy-at-scale)
- [Introducing Apple's On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- [Personalizing a Model with On-Device Updates](https://developer.apple.com/documentation/CoreML/personalizing-a-model-with-on-device-updates)

### ANE Reverse Engineering & Training
- [Orion: Characterizing and Programming Apple's Neural Engine (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728)
- [Inside the M4 Apple Neural Engine - Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Inside the M4 Apple Neural Engine - Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [maderix/ANE on GitHub](https://github.com/maderix/ANE)
- [hollance/neural-engine on GitHub](https://github.com/hollance/neural-engine)

### Forward-Only Training Methods
- [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://princeton-nlp.github.io/mezo/)
- [On-Device Fine-Tuning via Backprop-Free Zeroth-Order Optimization (arXiv 2511.11362)](https://arxiv.org/abs/2511.11362)
- [MobiZO: Efficient LLM Fine-Tuning at the Edge (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1022/)
- [AGZO: Activation-Guided Zeroth-Order Optimization](https://arxiv.org/abs/2601.17261)
- [The Forward-Forward Algorithm (Hinton)](https://arxiv.org/abs/2212.13345)

### ANE Performance Comparisons
- [Apple Neural Engine for LLM Inference: What Actually Works](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
- [Defending the Apple Neural Engine](https://dennisforbes.ca/blog/microblog/2026/02/apple-neural-engine-and-you/)
- [Apple Neural Engine vs Google TPU vs NVIDIA Tensor Cores](https://pynomial.com/2025/03/apple-neural-engine-vs-google-tpu-vs-nvidia-tensor-cores/)

### Edge AI & Continuous Learning
- [Corun: Concurrent Inference and Continuous Training at the Edge](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487435/)
- [New Test-Time Training Method (VentureBeat)](https://venturebeat.com/infrastructure/new-test-time-training-method-lets-ai-keep-learning-without-exploding)
- [TinyTTA: Efficient Test-time Adaptation on Edge Devices](https://neurips.cc/virtual/2024/poster/94778)
- [On-demand Test-time Adaptation for Edge Devices](https://arxiv.org/abs/2505.00986)

### Federated Learning
- [How Apple Personalizes Siri Without Hoovering Up Your Data (MIT Tech Review)](https://www.technologyreview.com/2019/12/11/131629/apple-ai-personalizes-siri-federated-learning/)
- [Federated Learning in Practice: Reflections and Projections](https://arxiv.org/html/2410.08892v2)
- [FwdLLM: Forward-Only Federated LLM Fine-Tuning](https://arxiv.org/abs/2308.13894)
