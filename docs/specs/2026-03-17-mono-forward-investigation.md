# Investigation: Mono-Forward for Transformer LoRA Fine-Tuning

**Date**: 2026-03-17
**Status**: Investigation / Analysis
**Author**: Research team
**Context**: Evaluating whether the Mono-Forward algorithm (arXiv:2501.09238) can replace backpropagation for transformer LoRA fine-tuning on Apple Neural Engine.

---

## 0. Executive Summary

**Verdict: Mono-Forward for transformer LoRA is theoretically adaptable but practically risky. Probability of matching backprop on real LLM tasks: ~15-25%.**

Mono-Forward (MF) is a genuine advance over Hinton's Forward-Forward for MLPs -- it eliminates positive/negative data pairs, requires only a single forward pass, and independently replicated evaluations confirm it matches or slightly beats backprop on MLPs. However, the algorithm has never been tested on transformers, and there are fundamental architectural mismatches (attention non-locality, residual streams, normalization) that make direct transfer non-trivial. A concurrent paper (Contrastive Forward-Forward for ViTs, arXiv:2502.00571) shows that forward-only methods CAN work on transformers with 1-3% accuracy gaps, but that work uses a different algorithm (contrastive FF, not Mono-Forward) and full-parameter training (not LoRA).

We recommend a 2-day mini-experiment before committing to a full Obj-C implementation.

---

## 1. Mono-Forward Algorithm: Deep Technical Analysis

### 1.1 Source Paper

- **Title**: "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors"
- **Authors**: (arXiv:2501.09238, January 2025)
- **Venue**: Preprint (not peer-reviewed at a top venue as of March 2026)

### 1.2 Core Mechanism

Mono-Forward replaces backpropagation with layer-local supervised learning. Each layer has:

1. **Layer weights** W_i: The standard weight matrix for layer i.
2. **Projection matrix** M_i: An additional m x n matrix (m = number of classes, n = neurons in layer i) that maps activations to class-specific scores.

**Algorithm (per batch, per layer i):**

```
Input: X_batch, y_batch (labels)
For each layer i = 1 to num_layers:
    z_i = a_{i-1} @ W_i              # linear transform
    a_i = ReLU(z_i)                   # activation
    G_i = a_i @ M_i^T                 # goodness scores (batch x classes)
    L_i = CrossEntropy(y_batch, G_i)  # local loss per layer
    W_i -= lr * dL_i/dW_i            # update layer weights
    M_i -= lr * dL_i/dM_i            # update projection matrix
    pass a_i to next layer (detached -- no gradient flows between layers)
```

### 1.3 The "Goodness" Function

Unlike Hinton's Forward-Forward which uses the sum of squared activations as "goodness":

- **Hinton FF**: G_l = sum_j (h_{l,j})^2 (scalar, positive/negative classification)
- **Mono-Forward**: G_i = a_i @ M_i^T (vector of class scores, one per category)

The key innovation is that M_i provides an **explicit label-input connection** -- each layer directly predicts class labels through its projection matrix, eliminating the need for positive/negative data pairs. This is why MF needs only ONE forward pass instead of FF's two (or m) passes.

The loss is standard cross-entropy applied to softmax(G_i), which is well-understood and stable.

### 1.4 How Label Signal Reaches Each Layer

In backpropagation, the label signal reaches early layers through the chain rule. In Mono-Forward, each layer independently receives the label via the projection matrix M_i. Layer i does not need any information from layers i+1, ..., L to compute its gradient. The projection matrix acts as a "local classifier head" attached to each layer.

This is conceptually similar to auxiliary classifiers (e.g., GoogLeNet/Inception), but taken to the extreme: EVERY layer has its own classifier, and layers are trained completely independently.

### 1.5 Key Difference from Forward-Forward

| Aspect | Forward-Forward | Mono-Forward |
|--------|----------------|--------------|
| Passes required | 2 (positive + negative) or m (one per class) | 1 |
| Positive/negative data | Required (must construct corrupted examples) | Not needed |
| Loss function | Custom goodness threshold | Standard cross-entropy |
| Label encoding | One-hot embedded in input pixels | Via projection matrix M_i |
| Prediction | Requires m forward passes or separate decoder | Same as BP (argmax of G at final layer) |
| Additional parameters | None | m * n per layer (projection matrices) |

---

## 2. Experimental Results: What the Paper Actually Shows

### 2.1 Reported Results (Original Paper, Table 3 -- MLPs)

| Dataset | Backprop | Mono-Forward | Delta |
|---------|----------|-------------|-------|
| MNIST | 99.52% | 99.58% | +0.06% |
| Fashion-MNIST | 92.63% | 93.98% | +1.35% |
| CIFAR-10 | 77.80% | 82.39% | +4.59% |
| CIFAR-100 | 42.10% | 54.77% | +12.67% |

**MLP architectures**: MNIST/FMNIST: 2x1000 ReLU. CIFAR-10/100: 3x2000 ReLU.

### 2.2 Reported Results (Original Paper, Table 2 -- CNNs)

| Dataset | Backprop | Mono-Forward | Delta |
|---------|----------|-------------|-------|
| MNIST | 98.69% | 98.74% | +0.05% |
| Fashion-MNIST | 90.27% | 90.52% | +0.25% |
| CIFAR-10 | 54.25% | 56.99% | +2.74% |
| CIFAR-100 | 27.64% | 29.05% | +1.41% |

**CNN architecture**: 4 conv layers (64, 128, 256, 512 feature maps), 3x3 kernels, ReLU, average pooling.

### 2.3 Independent Replication (arXiv:2511.01061 -- Rigorous Evaluation)

A follow-up paper conducted hardware-validated comparisons with **fairly tuned** backprop baselines:

| Dataset (MLP) | Backprop (fair) | Mono-Forward | Delta | Training Time | Energy |
|---------------|----------------|--------------|-------|---------------|--------|
| MNIST | ~99.4% | +0.09% | Small | -12% | -13% |
| Fashion-MNIST | ~92.5% | +0.51% | Small | +18% | +10% |
| CIFAR-10 | 61.13% | 62.34% (+1.21%) | Modest | -34% | -41% |
| CIFAR-100 | ~42% | +0.37% | Small | -1% | -12% |

**Critical findings from the independent evaluation:**
- MF's accuracy advantage is **real but much smaller** than the original paper reports when BP is fairly tuned
- The original paper's CIFAR-10 MLP BP baseline (77.80%) vs the replication's (61.13%) suggests **the original used different architectures or hyperparameters** for BP vs MF
- MF genuinely saves energy on CIFAR-10 (41% reduction) but not consistently across tasks
- Memory savings are modest (~5%), not dramatic as theoretically expected

### 2.4 Red Flags in the Original Paper

1. **Hyperparameter tuning on test set**: The paper states tuning was "based on testing set performance" -- this is a methodological error that inflates reported numbers.
2. **No explicit limitations section**: The paper does not acknowledge any limitations.
3. **CNN results are weak**: Both BP (54.25%) and MF (56.99%) on CIFAR-10 CNN are far below state-of-the-art (~95%+), suggesting the CNN architecture and training setup are not well-optimized. This makes the comparison less informative.
4. **Not peer-reviewed at a top venue**: As of March 2026, the paper remains an arXiv preprint.
5. **CIFAR-100 MLP gap is suspicious**: A +12.67% gap (42.10% vs 54.77%) is enormous. The independent replication found only +0.37%. This strongly suggests the original BP baseline was under-tuned.

---

## 3. Transferability to Transformers: Detailed Analysis

### 3.1 Why Mono-Forward Works for MLPs

In an MLP, information flows strictly forward: layer i's output depends only on layer i's input and weights. The local objective makes sense because:
- Each layer independently transforms its input
- The projection matrix M_i acts as a local probe of "does this layer's representation predict the class?"
- Greedy layer-wise optimization encourages each layer to produce class-discriminative features
- The independent evaluation suggests this acts as a **regularizer**, preventing overfitting

### 3.2 Transformer Architecture Mismatches

A standard transformer layer consists of:

```
x_{l+1} = x_l + FFN(RMSNorm(x_l + Attention(RMSNorm(x_l))))
```

Where Attention involves:
```
Q = x @ W_Q,  K = x @ W_K,  V = x @ W_V
A = softmax(Q @ K^T / sqrt(d_k))
out = A @ V @ W_O
```

**Problem 1: Attention is non-local across sequence positions.**
- In an MLP, each layer operates on a fixed-size vector independently.
- In attention, token i's output depends on ALL tokens via Q_i @ K_j^T for all j.
- The projection matrix M_i would see the post-attention activations, but the attention pattern itself creates long-range dependencies WITHIN the layer.
- Mono-Forward's local loss can still work at the layer level (treating the whole attention block as one "layer"), but it cannot provide learning signal to individual attention heads about which tokens to attend to.

**Problem 2: Residual connections create information shortcuts.**
- x_{l+1} = x_l + delta_l means the next layer sees BOTH the current layer's contribution AND all previous layers' contributions via the residual stream.
- If layer l's projection M_l classifies well because x_l already contains good features from layers 0..l-1, layer l has no incentive to learn anything useful itself -- it can "free-ride" on the residual stream.
- This is a known problem with auxiliary classifiers and was partially addressed in the Contrastive FF ViT paper.

**Problem 3: Normalization (RMSNorm/LayerNorm) rescales activations.**
- MF's goodness G_i = a_i @ M_i^T depends on activation magnitudes.
- RMSNorm normalizes the activation vector, potentially washing out the goodness signal that MF relies on.
- The projection matrix would need to work with normalized activations, which have unit variance by design.

**Problem 4: Multi-head attention splits the representation.**
- Each head operates on a d_k-dimensional subspace (typically 64 or 128 dims).
- The projection matrix M_i would need to capture class-relevant information across all heads simultaneously.
- In standard transformers, individual heads specialize (some for syntax, some for semantics, etc.) -- a single projection matrix per layer may not capture this.

**Problem 5: Sequence length dimension.**
- MLP activations are (batch, hidden_dim) -- directly compatible with M_i of shape (classes, hidden_dim).
- Transformer activations are (batch, seq_len, hidden_dim). To apply M_i, we need to aggregate across the sequence dimension first (e.g., mean pooling, CLS token).
- This aggregation is an additional design choice not addressed by MF.

### 3.3 Evidence from Related Work: CFF on Vision Transformers

The Contrastive Forward-Forward for Vision Transformers paper (arXiv:2502.00571) provides the closest evidence:

**Setup**: ViT architectures (e.g., ViT[192, 6 heads, 6 layers]) on CIFAR-10/100 and Tiny ImageNet.

**Key results**:

| Dataset | BP+CE | CFF+M (forward-only) | Gap |
|---------|-------|----------------------|-----|
| CIFAR-10 | 86.95% | 85.17% | -1.78% |
| CIFAR-100 | 84.13% | 84.28% | +0.15% |
| Tiny ImageNet | 78.95% | 76.06% | -2.89% |

**How they handled attention**: They did NOT modify the attention mechanism. Each transformer encoder layer is treated as one opaque "layer" for the forward-forward objective. The loss is applied AFTER each encoder block, using average pooling over token positions to reduce (batch, seq, dim) to (batch, dim) before computing the local loss.

**Critical insight**: The CFF paper shows that treating attention blocks as black boxes for local learning works reasonably well, with 1-3% accuracy gaps. But this uses contrastive learning (positive/negative pairs), not Mono-Forward's projection matrices.

**Caveats**:
- These are ViTs (image patches as tokens, 32x32 images -> 4x4 or 8x8 patches -> 16-64 tokens), not language models with 512-2048 token sequences.
- CIFAR classification is much simpler than language generation.
- CFF requires data augmentation to generate positive/negative pairs.
- Without heavy augmentation, CFF actually outperforms BP. WITH heavy augmentation (RandAug), BP pulls ahead. This suggests forward-only methods may have weaker capacity to exploit data augmentation.

### 3.4 Assessment: Can Mono-Forward Handle Transformers?

**What could work:**
- Treating each transformer block as one "layer" and applying a local cross-entropy loss via projection matrix M_i
- Average-pooling over sequence positions to get a fixed-size activation vector for M_i
- For classification tasks, this is architecturally feasible

**What is uncertain:**
- Whether the projection matrix provides enough learning signal for attention weight learning
- Whether the residual stream free-riding problem kills learning in later layers
- Whether the approach works for generation tasks (next-token prediction) rather than classification

**What likely does NOT work:**
- Language generation tasks where the "class" is the next token (vocabulary 32K-128K classes -> M_i would be enormous)
- Long sequence tasks where attention patterns are crucial
- Tasks requiring cross-layer feature coordination (which backprop naturally provides)

---

## 4. Proposed Mono-Forward LoRA Adaptation

### 4.1 Architecture

For each transformer layer l, we have:
- **Frozen pretrained weights**: W_Q, W_K, W_V, W_O, W_up, W_gate, W_down (frozen)
- **LoRA adapters**: A_Q, B_Q, A_V, B_V (trainable, rank r)
- **Projection matrix**: M_l of shape (num_classes, hidden_dim) (trainable)

### 4.2 Forward Pass (Single Pass)

```python
for layer_l in transformer_layers:
    # Standard forward through frozen + LoRA
    x_l = layer_l.forward(x_{l-1})  # includes attention + FFN + residual

    # Local classification via projection
    x_pooled = x_l.mean(dim=1)         # (batch, dim) -- pool over sequence
    G_l = x_pooled @ M_l.T             # (batch, num_classes) -- goodness scores
    L_l = CrossEntropy(labels, G_l)    # local loss

    # Compute gradients ONLY for this layer's LoRA params and M_l
    # This requires a LOCAL backward through layer_l only
    grads = torch.autograd.grad(L_l, [A_Q, B_Q, A_V, B_V, M_l])

    # Update LoRA params
    A_Q -= lr * grads[0]
    B_Q -= lr * grads[1]
    # ... etc

    # Detach x_l before passing to next layer (no inter-layer gradients)
    x_{l-1} = x_l.detach()
```

### 4.3 Key Design Decisions

**Sequence aggregation**: Mean pooling over sequence positions is the simplest choice. Alternatives:
- Use only the last token position (causal LM style)
- Use CLS token (BERT style)
- Learned weighted sum over positions

**LoRA-specific goodness**: Instead of the generic projection G_l = a_l @ M_l^T, we could define a LoRA-specific goodness:
```
G_l = ||B_l @ (A_l @ x_l)||^2
```
This measures "how much signal passes through the LoRA adapter" but loses the class-conditional information. Not recommended for classification, but could be useful as an auxiliary regularizer.

**Label injection for generation tasks**: For next-token prediction, the "label" per position is the next token. With vocabulary V = 32K, M_l would be 32K x hidden_dim -- too large. Options:
- **Shared projection**: Use a single M shared across layers (reduces to adding auxiliary classifier heads)
- **Low-rank projection**: M_l = U_l @ V_l^T where U_l is (V, r) and V_l is (dim, r) with r << dim
- **Contrastive objective instead**: Abandon MF's cross-entropy and use contrastive loss (correct next-token embedding vs random embeddings)

### 4.4 The Local Backward Problem

A critical subtlety: Mono-Forward claims to be "backpropagation-free" but still requires computing dL_l/dW_l for each layer. For an MLP layer with W and M, this is trivial:
```
dL/dW = dL/dG * dG/da * da/dz * dz/dW
```
Each term is local and computable from quantities available during the forward pass.

For a transformer layer, the "local backward" through attention is still complex:
```
dL/d(A_Q) requires: dL/dG * dG/dx_l * dx_l/d(attn_out) * d(attn_out)/dQ * dQ/d(A_Q)
```

The term d(attn_out)/dQ involves the softmax Jacobian and depends on K -- this IS computable locally (K is within the same layer), but it is nontrivial and requires storing attention weights.

**This means**: Even though we do not backpropagate BETWEEN layers, we still need a full backward pass WITHIN each transformer layer. On the Apple Neural Engine, which does not support backward passes at all, this is a problem. We would need to either:
1. Approximate the within-layer gradient (e.g., with zeroth-order methods like MeZO)
2. Use a forward-mode autodiff approximation
3. Restrict LoRA to only the FFN sublayer (where the local gradient is simpler)

---

## 5. Show-Stoppers and Critical Issues

### 5.1 Show-Stopper 1: Intra-Layer Backward Pass Still Required

Mono-Forward eliminates INTER-layer backpropagation but still requires INTRA-layer gradient computation. For MLP layers, this is trivial (single matrix multiply). For transformer layers with attention, this requires a full backward pass through the attention mechanism, including:
- Softmax Jacobian computation
- Q/K/V gradient chain
- Multi-head concatenation gradients

On ANE (our target hardware), this is just as impossible as full backpropagation. This is arguably the biggest issue.

**Mitigation**: Use MeZO (zeroth-order) to estimate the within-layer gradient. This requires 2 forward passes per layer per step, but each pass is through a SINGLE layer (not the full model). For a 32-layer model, this is 64 single-layer forward passes vs 2 full-model passes for standard MeZO -- potentially faster if single-layer passes can be parallelized.

### 5.2 Show-Stopper 2: Residual Stream Free-Riding

With residual connections, layer l's projection M_l sees:
```
x_l = x_0 + delta_1 + delta_2 + ... + delta_l
```

If x_0 + delta_1 + ... + delta_{l-1} already predicts the class well, M_l can achieve low loss without delta_l contributing anything useful. Layer l's LoRA adapters receive zero gradient and learn nothing.

This is a fundamental problem for greedy layer-wise training with residual connections. The CFF ViT paper partially addresses this by using contrastive loss (which cares about relative distances, not absolute classification), but MF's cross-entropy is more susceptible.

**Mitigation options**:
- Normalize out the residual: Apply M_l only to delta_l (the layer's own contribution), not the full x_l.
- Use a "residual-aware" projection: M_l @ (x_l - x_{l-1}) instead of M_l @ x_l.
- Progressive learning: Train layer 1 first, then freeze it and train layer 2 on the residual, etc.

### 5.3 Show-Stopper 3: Generation Tasks Are Incompatible

Mono-Forward is designed for classification (small number of classes). Language model fine-tuning is typically next-token prediction with vocabulary sizes of 32K-128K. The projection matrix M_l for 32K classes and hidden_dim 4096 would be 32K * 4096 = 128M parameters PER LAYER -- more than the LoRA adapters by orders of magnitude.

**Mitigation**: Reformulate as a contrastive task. Instead of predicting the exact next token, define the local loss as:
```
L_l = -log(exp(sim(x_l, e_correct)) / sum_j exp(sim(x_l, e_j)))
```
where e_correct is the embedding of the correct next token and e_j are negative samples. This is similar to noise-contrastive estimation and avoids the huge projection matrix.

### 5.4 Show-Stopper 4: Attention Pattern Quality

In standard backprop training, the loss at the output propagates gradients that shape attention patterns across all layers. Each layer's attention learns what to attend to based on what is useful for the final task.

In Mono-Forward, each layer's attention must learn patterns that are useful for the LOCAL classification at that layer. There is no signal saying "this attention pattern in layer 3 is useful because layer 12 needs the information it extracts." Early layers may develop myopic attention patterns optimized for immediate classification rather than useful feature extraction for deeper layers.

**Mitigation**: This may not matter for fine-tuning (where attention patterns are already pretrained). LoRA only makes small adjustments, so the pretrained attention patterns remain largely intact.

### 5.5 Summary of Show-Stoppers

| Issue | Severity | Mitigation | Feasibility |
|-------|----------|-----------|-------------|
| Intra-layer backward | CRITICAL for ANE | MeZO within-layer | Moderate |
| Residual free-riding | HIGH | Residual subtraction | Untested |
| Generation vocab size | HIGH | Contrastive reformulation | Moderate |
| Attention pattern quality | MEDIUM | Pretrained patterns sufficient? | Likely OK for fine-tuning |
| Normalization interaction | LOW | Projection adapts | Likely OK |

---

## 6. Mini-Experiment Proposal

### 6.1 Objective

Validate whether Mono-Forward can train LoRA adapters on a tiny transformer for classification, and measure the accuracy gap vs backpropagation.

### 6.2 Setup

- **Model**: 2-layer transformer, hidden_dim=64, 4 attention heads, FFN_dim=256
- **Task**: MNIST classification (10 classes, treat 28x28 image as 49 tokens of dim 16 via 4x4 patches)
- **Training**: Compare 3 methods on identical architecture:
  1. Standard backprop (full model)
  2. Standard backprop (LoRA only, rank=4)
  3. Mono-Forward (LoRA only, rank=4)
- **Metrics**: Test accuracy, loss curve, training time

### 6.3 Proposed Code

```python
"""
Mono-Forward LoRA vs Backprop LoRA on a tiny transformer for MNIST.
Run: python mono_forward_test.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Tiny Transformer ---
class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=64, heads=4, ffn_dim=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )
        # LoRA adapters (rank 4) on Q and V projections
        self.lora_q_A = nn.Parameter(torch.randn(dim, 4) * 0.01)
        self.lora_q_B = nn.Parameter(torch.zeros(4, dim))
        self.lora_v_A = nn.Parameter(torch.randn(dim, 4) * 0.01)
        self.lora_v_B = nn.Parameter(torch.zeros(4, dim))

    def forward(self, x):
        h = self.norm1(x)
        lora_q_delta = h @ self.lora_q_A @ self.lora_q_B
        lora_v_delta = h @ self.lora_v_A @ self.lora_v_B
        attn_out, _ = self.attn(h + lora_q_delta, h, h + lora_v_delta)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, num_layers=2, dim=64, num_classes=10,
                 seq_len=49, patch_dim=16):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.layers = nn.ModuleList(
            [TinyTransformerBlock(dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 49, 16)
        x = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.head(x)

# --- Mono-Forward Training ---
def train_mono_forward(model, train_loader, epochs=5, lr=1e-3):
    for p in model.parameters():
        p.requires_grad = False
    lora_params = []
    projections = []
    for i, layer in enumerate(model.layers):
        for name in ['lora_q_A', 'lora_q_B', 'lora_v_A', 'lora_v_B']:
            getattr(layer, name).requires_grad = True
            lora_params.append(getattr(layer, name))
        M_i = nn.Parameter(
            torch.randn(10, 64) * 0.01
        ).to(next(model.parameters()).device)
        projections.append(M_i)

    all_params = lora_params + projections
    optimizer = torch.optim.Adam(all_params, lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            h = x.view(x.size(0), 49, 16)
            h = model.patch_embed(h) + model.pos_embed
            layer_loss = 0
            for i, layer in enumerate(model.layers):
                h = layer(h)
                h_pooled = h.mean(dim=1)
                G_i = h_pooled @ projections[i].T
                L_i = F.cross_entropy(G_i, y)
                layer_loss += L_i
                h = h.detach()  # stop gradient between layers
            layer_loss.backward()
            optimizer.step()
            total_loss += layer_loss.item()
        losses.append(total_loss / len(train_loader))
        print(f"  MF Epoch {epoch+1}: loss={losses[-1]:.4f}")
    return losses, projections

# --- Standard Backprop Training (LoRA only) ---
def train_backprop_lora(model, train_loader, epochs=5, lr=1e-3):
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.layers:
        for name in ['lora_q_A', 'lora_q_B', 'lora_v_A', 'lora_v_B']:
            getattr(layer, name).requires_grad = True
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        print(f"  BP Epoch {epoch+1}: loss={losses[-1]:.4f}")
    return losses

# --- Evaluation ---
def evaluate_bp(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate_mf(model, projections, loader):
    """Evaluate using last layer's projection matrix."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            h = x.view(x.size(0), 49, 16)
            h = model.patch_embed(h) + model.pos_embed
            for layer in model.layers:
                h = layer(h)
            h_pooled = h.mean(dim=1)
            G = h_pooled @ projections[-1].T
            correct += (G.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# --- Main ---
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256)

    torch.manual_seed(42)
    model_bp = TinyTransformer()
    torch.manual_seed(42)
    model_mf = TinyTransformer()

    print("=== Backprop LoRA ===")
    bp_losses = train_backprop_lora(model_bp, train_loader, epochs=10)
    bp_acc = evaluate_bp(model_bp, test_loader)
    print(f"  BP Test Accuracy: {bp_acc:.4f}")

    print("\n=== Mono-Forward LoRA ===")
    mf_losses, projections = train_mono_forward(
        model_mf, train_loader, epochs=10
    )
    mf_acc = evaluate_mf(model_mf, projections, test_loader)
    print(f"  MF Test Accuracy: {mf_acc:.4f}")

    print(f"\n=== Results ===")
    print(f"  Backprop LoRA:       {bp_acc:.4f}")
    print(f"  Mono-Forward LoRA:   {mf_acc:.4f}")
    print(f"  Gap:                 {(bp_acc - mf_acc)*100:.2f}%")
```

### 6.4 What to Watch For

1. **Does MF loss decrease?** If the layer-local losses do not decrease, the approach is fundamentally broken for transformers.
2. **Accuracy gap**: Expect MF to be 2-10% worse than BP on this simple task. If the gap is >15%, the approach is not viable.
3. **Layer loss pattern**: Do all layers learn (decreasing loss), or do later layers free-ride on the residual?
4. **Training speed**: MF should be slightly faster per step (no backward pass between layers), but may need more epochs.

### 6.5 Expected Outcome

Based on the CFF ViT results (1-3% gap on CIFAR), I predict:
- MF LoRA will achieve ~90-94% on MNIST (vs ~96-97% for BP LoRA)
- The gap will be larger on harder tasks
- Later layers will show less learning than earlier layers (residual free-riding)

---

## 7. Comparison with Our Existing FF-LoRA Design

We already have a Forward-Forward + LoRA design (see `2026-03-16-forward-forward-lora-design.md`). How does Mono-Forward compare?

| Aspect | FF-LoRA (our existing design) | MF-LoRA (proposed) |
|--------|-------------------------------|-------------------|
| Forward passes | 2 (positive + negative) | 1 |
| Negative data | Must construct corrupted examples | Not needed |
| Loss function | Binary goodness threshold | Cross-entropy via projection |
| Label injection | Embedded in input tokens | Via projection matrix M_l |
| Additional params | Threshold theta_l per layer | Projection M_l per layer (much larger) |
| Complexity | Higher (negative generation) | Lower (simpler pipeline) |
| Proven on transformers | CFF paper: ~1-3% gap on ViTs | Never tested |

**MF-LoRA advantages over FF-LoRA**:
- Single forward pass (no need to construct negative examples)
- Standard cross-entropy loss (well-understood optimization landscape)
- Simpler implementation

**MF-LoRA disadvantages**:
- Large projection matrices (10 * dim per layer for 10-class; V * dim for language modeling)
- Never tested on any transformer architecture
- Residual free-riding may be worse (FF-LoRA's goodness measures activation energy, which is more sensitive to layer contribution)

---

## 8. Honest Assessment

### 8.1 Probability Estimates

| Scenario | Probability | Reasoning |
|----------|------------|-----------|
| MF-LoRA works for transformer classification (within 3% of BP) | 40-50% | CFF ViT shows forward-only CAN work; MF is simpler than FF; but untested |
| MF-LoRA works for transformer generation/LLM fine-tuning | 10-15% | Vocabulary size issue, sequential generation mismatch |
| MF-LoRA matches or beats BP on transformers | 5-10% | The "beats backprop" claim barely holds on MLPs; transformers are harder |
| MF-LoRA is viable for ANE deployment specifically | 15-20% | Still needs within-layer gradients, which ANE cannot compute |

### 8.2 The "Surpasses Backprop" Claim: Verdict

**The claim is overstated.** Here is the evidence:

1. The original paper's backprop baselines appear under-tuned (hyperparameters tuned on test set, enormous gaps on CIFAR-100 that do not replicate).
2. Independent replication with fairly-tuned baselines shows MF is **within +/-1%** of backprop on MLPs -- matching, not surpassing.
3. The ~1% advantage on CIFAR-10 MLPs may be a real regularization effect (greedy layer-wise training acts as implicit regularization), but it is task-specific and architecture-specific.
4. CNN results are weak for both methods (54-57% on CIFAR-10 vs SOTA ~95%), suggesting the experimental setup does not represent practical use cases.
5. No independent validation of the "surpasses" claim exists for any architecture beyond simple MLPs.

**For transformers, MF is unlikely to match backprop**, let alone surpass it. The CFF ViT results (different algorithm, same family) show 1-3% gaps even on simple image classification.

### 8.3 Recommendation for AutoANE Project

1. **Do NOT proceed directly to Obj-C/ANE implementation of Mono-Forward.** The algorithm has never been tested on transformers and has fundamental architectural mismatches.

2. **Run the mini-experiment first** (Section 6). Estimated effort: 4-6 hours. This will tell us:
   - Whether MF-LoRA converges at all on a tiny transformer
   - The accuracy gap vs BP-LoRA
   - Whether residual free-riding is a fatal problem

3. **If the mini-experiment succeeds (gap < 5%)**:
   - Extend to a larger model (e.g., GPT-2 small, 12 layers)
   - Test on a language task (next-token prediction with contrastive reformulation)
   - Compare against our existing FF-LoRA and MeZO approaches

4. **If the mini-experiment fails (gap > 10%)**:
   - Investigate hybrid approaches: MeZO for gradient estimation + MF for layer-local loss
   - Consider the CFF (Contrastive Forward-Forward) approach instead, which has proven ViT results
   - Fall back to our existing MeZO or FF-LoRA designs

5. **For ANE specifically**: The within-layer gradient requirement is a critical blocker. Even if MF-LoRA works in PyTorch, deploying on ANE requires solving the within-layer gradient problem. The most promising path is **MF loss structure + MeZO gradient estimation within each layer**, which would give us:
   - Layer-local objective (MF's cross-entropy via projection)
   - Forward-only gradient estimation (MeZO's perturbation)
   - Single-layer forward passes (parallelizable across layers on ANE)

This hybrid "MF-MeZO" approach is the most promising path and warrants investigation after the mini-experiment.

---

## 9. References

1. **Mono-Forward original paper**: arXiv:2501.09238 (Jan 2025) -- "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors"
2. **Rigorous evaluation**: arXiv:2511.01061 (Nov 2025) -- "Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Evaluation of Forward-Only Algorithms"
3. **Contrastive FF for ViTs**: arXiv:2502.00571 (Feb 2025) -- "Contrastive Forward-Forward: A Training Algorithm of Vision Transformer"
4. **Self-Contrastive FF**: Nature Communications (2025) -- "Self-Contrastive Forward-Forward algorithm"
5. **Forward-Forward original**: arXiv:2212.13345 (Hinton, Dec 2022)
6. **Beyond Backpropagation survey**: arXiv:2509.19063 (Sep 2025) -- "Beyond Backpropagation: Exploring Innovative Algorithms for Energy-Efficient Deep Neural Network Training"
7. **Our FF-LoRA design**: `docs/specs/2026-03-16-forward-forward-lora-design.md`

---

## Appendix A: Mono-Forward Algorithm Pseudocode (from paper)

```
Algorithm 1: Layer-Wise Training for One Batch
Input: X_batch, y_batch
For i = 1 to num_layers:
    z_i <- a_{i-1} W_i          // linear
    a_i <- phi(z_i)             // activation (ReLU)
    G_i <- a_i M_i^T            // goodness scores
    L_i <- CrossEntropy(y, G_i) // local loss
    W_i <- W_i - eta * dL_i/dW_i
    M_i <- M_i - eta * dL_i/dM_i
    pass a_i to next layer (detached)
```

## Appendix B: Key Numbers at a Glance

**Mono-Forward on MLPs (independently replicated, fair baselines)**:
- MNIST: +0.09% vs BP
- Fashion-MNIST: +0.51% vs BP
- CIFAR-10: +1.21% vs BP
- CIFAR-100: +0.37% vs BP

**Contrastive FF on ViTs (closest transformer result)**:
- CIFAR-10: -1.78% vs BP
- CIFAR-100: +0.15% vs BP
- Tiny ImageNet: -2.89% vs BP

**Projection matrix overhead per layer**: num_classes * hidden_dim parameters
- 10-class task, dim=64: 640 params (negligible)
- 10-class task, dim=4096: 40K params (small)
- 32K-class LM, dim=4096: 128M params (BLOCKER -- larger than most LoRA adapters)
