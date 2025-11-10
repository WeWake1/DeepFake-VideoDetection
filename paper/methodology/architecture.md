# Combined Spatial-Temporal Multi-Quality Deepfake Detection Architecture

This document is a complete, end-to-end reference for the SOTA architecture we’ll train for deepfake detection on Celeb-DF. It covers the high-level flow, mathematical formulation, network modules, tensor dimensions, training loop, losses, optimization, and expected performance. It’s designed to be implementation-ready and aligned with the preprocessing you’ve completed.

- Dataset: Celeb-DF (590 real, 5,639 fake videos)
- Preprocessing: 224×224 aligned faces (MTCNN, GPU), frame skip = 3, output on F:\
- Hardware reference: RTX 4500 Ada (24GB VRAM), 72-core CPU, 256GB RAM (Windows)
- Repo context:
  - Face detection script: `co/face_detect_mtcnn_gpu(final).py`
  - Verification: `co/verify_face_extraction.py`
  - Mappings: `frame_mapping.csv`, `enhanced_mapping.csv`, `training_pairs.csv`

---

## 1. Architecture overview

```
INPUT: Video → Sample T frames (every 3rd) → Build HQ/LQ sequences
        │
        ├───────────────────────────┬───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
   HQ sequence (224×224)       LQ sequence (112×112)       (optional metadata)
        │                           │
        ▼                           ▼
     CNN_HQ (EffNet-B4)         CNN_LQ (EffNet-B0)
  Spatial features (T×7×7×1792)  Spatial features (T×7×7×1280)
        │                           │
        ▼                           ▼
  ConvLSTM_HQ (temporal)       ConvLSTM_LQ (temporal)
   Temporal vector (512)        Temporal vector (512)
        └───────────────┬───────────────┘
                        ▼
                  Fusion (attention)
                     (512)
                        ▼
              Classifier (Dense → Sigmoid)
                        ▼
                 p(fake | video) ∈ [0,1]
```

Why this works:
- Spatial CNNs detect pixel-level artifacts, blending seams, GAN fingerprints.
- ConvLSTM captures temporal inconsistencies (blink dynamics, lighting coherence).
- Multi-quality streams (HQ/LQ) make the model robust to compression and real-world re-encodes.
- Attention fusion learns which stream is more reliable per sample.

---

## 2. Mathematical foundation

### 2.1 Problem
- Video V with frames: $V = \{F_1, F_2, \ldots, F_n\}$
- Sample a sequence of T frames (stride s = 3): $S = \{F_{t_1}, \ldots, F_{t_T}\}$
- Build dual-quality inputs: $S^{HQ}$ (224×224), $S^{LQ}$ (112×112 with JPEG artifacts)
- Learn $f(V): V \to \{0,1\}$ (0 = real, 1 = fake)

### 2.2 Decomposition
$$
\begin{aligned}
\varphi_{HQ} &= g_{\text{spatial}}^{HQ}(S^{HQ}) \in \mathbb{R}^{T\times 7\times 7\times 1792} \\
\varphi_{LQ} &= g_{\text{spatial}}^{LQ}(S^{LQ}) \in \mathbb{R}^{T\times 7\times 7\times 1280} \\
\psi_{HQ} &= g_{\text{temporal}}^{HQ}(\varphi_{HQ}) \in \mathbb{R}^{512} \\
\psi_{LQ} &= g_{\text{temporal}}^{LQ}(\varphi_{LQ}) \in \mathbb{R}^{512} \\
\psi_{\text{fused}} &= g_{\text{fusion}}(\psi_{HQ},\psi_{LQ}) \in \mathbb{R}^{512} \\
\hat{y} &= \sigma(\mathbf{w}^\top \psi_{\text{fused}} + b) \in [0,1]
\end{aligned}
$$

Binary cross-entropy loss (with logits):
$$
\mathcal{L}_{BCE} = -\,y\,\log(\hat{y}) - (1-y)\,\log(1-\hat{y})
$$

Optional weight decay: $\mathcal{L} = \mathcal{L}_{BCE} + \lambda \lVert \theta \rVert_2^2$.

---

## 3. Network components deep dive

### 3.1 Dual-quality preprocessing
- HQ: 224×224 aligned faces (already produced by MTCNN pipeline).
- LQ: create via downsampling + JPEG artifacts.

Example (concept):
```python
# LQ transform
lq = cv2.resize(hq, (112,112), interpolation=cv2.INTER_AREA)
_, buf = cv2.imencode('.jpg', lq, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
lq = cv2.imdecode(buf, cv2.IMREAD_COLOR)
```

### 3.2 Spatial feature extractors
- HQ stream: EfficientNet-B4 → output feature map per frame ≈ (7×7×1792)
- LQ stream: EfficientNet-B0 → output feature map per frame ≈ (7×7×1280)
- Both use MBConv (inverted bottlenecks + depthwise conv + SE), Swish activation.

MBConv sketch:
$$
\text{MBConv}(x)=\text{BN}(\text{Conv}_{1\times1}(\text{SE}(\text{BN}(\text{DWConv}(\text{Swish}(\text{BN}(\text{Conv}_{1\times1}(x))))))))
$$

### 3.3 Temporal modeling: ConvLSTM
Treats each time step as a spatial tensor; replaces matrix multiplies with convolutions to preserve 2D structure.

ConvLSTM equations:
$$
\begin{aligned}
I_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) \\
F_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + b_f) \\
G_t &= \tanh  (W_{xg} * X_t + W_{hg} * H_{t-1} + b_g) \\
O_t &= \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + b_o) \\
C_t &= F_t \odot C_{t-1} + I_t \odot G_t \\
H_t &= O_t \odot \tanh(C_t)
\end{aligned}
$$

- 2 stacked ConvLSTM layers per stream (e.g., 256 filters → 128 filters)
- Global average pooling → 512-D temporal vector per stream.

Captures:
- Blink dynamics (smooth open→close→open)
- Head pose trajectories
- Lighting consistency over time
- Temporal jitter/flicker unique to fakes

### 3.4 Fusion: attention
Compute scalar confidence per stream and combine:
$$
\begin{aligned}
e_{HQ} &= \mathbf{w}_a^\top \psi_{HQ} + b_a,\quad e_{LQ}=\mathbf{w}_a^\top \psi_{LQ} + b_a \\
\alpha_{HQ} &= \dfrac{e^{e_{HQ}}}{e^{e_{HQ}}+e^{e_{LQ}}},\quad \alpha_{LQ} = 1-\alpha_{HQ} \\
\psi_{fused} &= \alpha_{HQ}\,\psi_{HQ} + \alpha_{LQ}\,\psi_{LQ}
\end{aligned}
$$

Alternate: 2×512 multi-head self-attention over concatenated streams.

### 3.5 Classifier
- Dense(512→256) + ReLU + Dropout(0.5)
- Dense(256→128) + ReLU + Dropout(0.3)
- Dense(128→1) → Sigmoid

---

## 4. Data flow & tensor shapes

Let T=10 frames sampled with stride=3.

- HQ input: $(T, 224, 224, 3)$ → CNN_B4 → $(T, 7, 7, 1792)$ → ConvLSTM → $(512)$
- LQ input: $(T, 112, 112, 3)$ → CNN_B0 → $(T, 7, 7, 1280)$ → ConvLSTM → $(512)$
- Fusion: $(512)+(512)→(512)$
- Classifier: $(512)→(1)$

End-to-end per-video FLOPs ≈ 48 GFLOPs; inference latency ≈ 0.5–1.0s on RTX 4500.

---

## 5. Training process

### 5.1 Sampling & batching
- Use `training_pairs.csv` to build balanced batches of real/fake samples.
- Each sample: two sequences (HQ/LQ) of length T, normalized with ImageNet stats.
- Recommended: batch size 8–16 (mixed precision if needed).

### 5.2 Optimizer & schedule
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau on val loss (factor=0.5, patience=3)
- Gradient clipping: 1.0 (stabilize ConvLSTM)
- Mixed precision: AMP for throughput and VRAM efficiency

### 5.3 Early stopping
- Monitor val loss; patience ~10 epochs; save `best_model.pth` on improvement.

### 5.4 Data augmentation
- Spatial: random horizontal flip, light color jitter, mild random crop/resize
- Temporal: random start index within valid range (keeps stride=3), optional temporal flip (reverse) for robustness

---

## 6. Losses & metrics

- Primary loss: Binary Cross-Entropy with logits
- Regularization: L2 weight decay, dropout in classifier
- Metrics: Accuracy, AUC-ROC, Precision/Recall, F1, PR-AUC

$$
\text{BCE}(y,\hat{y}) = -\big(y\log\hat{y} + (1-y)\log(1-\hat{y})\big)
$$

---

## 7. Implementation details

### 7.1 Memory & speed
- Use AMP (`torch.cuda.amp.autocast` + `GradScaler`)
- Gradient checkpointing for deep CNN blocks if VRAM-limited
- Precompute LQ sequences offline if I/O becomes bottleneck

### 7.2 Practical defaults (Celeb-DF)
- T = 10, stride = 3 (aligned with preprocessing)
- CNNs: B4 (HQ), B0 (LQ)
- ConvLSTM: 256→128 filters, kernel 3×3, padding same
- Fusion: scalar attention (simple, effective)
- Batch size: 12 (tune with VRAM)
- Epochs: 40–60

### 7.3 Failure modes to watch
- Temporal collapse (ConvLSTM ignoring time): add temporal augmentation
- Overfitting to identities: ensure balanced identities in splits
- Compression bias: ensure LQ stream present during training

---

## 8. Performance expectations & ablations

| Configuration                        | Acc. (Celeb-DF) |
|--------------------------------------|------------------|
| CNN_HQ only                          | 0.85–0.88        |
| CNN_HQ + ConvLSTM_HQ                 | 0.92–0.95        |
| CNN_LQ only                          | 0.80–0.83        |
| CNN_LQ + ConvLSTM_LQ                 | 0.88–0.91        |
| Both streams (concat, no attention)  | 0.94–0.95        |
| **Full (attention fusion)**          | **0.96–0.98**    |

Latency (per video): 0.5–1.0s (RTX 4500). Training time: ~1–2 days depending on batch size and AMP.

---

## 9. Input/Output contract (for training code)

- Input:
  - `F:/real/<video_name>/*.jpg`, `F:/fake/<video_name>/*.jpg` (224×224)
  - Sampling params: T=10, stride=3
  - Normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Output:
  - `p_fake ∈ [0,1]` per video
  - Checkpoints: `checkpoints/best_model.pth`
  - Metrics: CSV/JSON log (val loss/acc/AUC)

Edge cases:
- Short videos (< T·stride): fallback to uniform sampling with wrap or pad
- Missing faces for a tiny subset: skip or reduce T for those

---

## 10. Next steps

- Implement the training pipeline with this specification (PyTorch, AMP, DataLoader for pairs)
- Add a `train/` module with:
  - `dataset.py` (sequence builder from faces)
  - `models.py` (CNN backbones, ConvLSTM blocks, fusion, head)
  - `train.py` (loop, logging, checkpoints)
  - `inference.py` (per-video scoring)
- Provide a minimal config file for hyperparams and paths

If you want, I can scaffold the full training code in `train/` with runnable scripts and a readme.
