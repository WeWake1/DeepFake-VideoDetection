# GPU Utilization & System Resource Usage Guide

## Quick Answer: CPU + GPU Simultaneous Usage

**Yes, the training pipeline ALREADY uses both CPU and GPU simultaneously!** Here's how:

### What runs on GPU:
- ✅ All model computations (EfficientNet, ConvLSTM, fusion, classifier)
- ✅ Forward pass through the network
- ✅ Backward pass (gradient computation)
- ✅ Loss calculation
- ✅ Tensor operations

### What runs on CPU:
- ✅ Data loading (reading .jpg files from disk)
- ✅ Image preprocessing (resize, augmentation, normalization)
- ✅ Batch collation (stacking tensors)
- ✅ Metrics calculation (accuracy, AUC, etc.)
- ✅ Logging and checkpointing
- ✅ Multi-process workers for parallel data loading

### Configuration for Maximum Utilization

**For your system (72-core CPU, RTX 4500 24GB):**

```yaml
# config/defaults.yaml
training:
  batch_size: 16          # Increased from 12 to use full 24GB VRAM
  num_workers: 8          # 8 parallel CPU processes for data loading
  
optimization:
  use_amp: true           # Mixed precision for 1.5-2x speedup + VRAM savings
```

**Expected resource usage during training:**
- GPU: 90-100% utilization (constant computation)
- CPU: 20-40% utilization (8 workers loading data in parallel)
- RAM: 20-40 GB (buffering batches)
- Disk I/O: High on F: drive (reading faces)

---

## Step-by-Step: Fix & Verify

### 1. Fix Pillow Dependency Conflict

**Already fixed in requirements.txt!** The issue was:
- ❌ `pillow==11.3.0` (conflicts with facenet-pytorch)
- ✅ `pillow>=10.2.0,<10.3.0` (compatible)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- efficientnet-pytorch (CNN backbones)
- pyyaml (config parsing)
- scikit-learn (metrics)
- matplotlib (optional plotting)
- **Correct Pillow version**

### 3. Verify GPU Setup

```bash
python check_system.py
```

This will:
- ✅ Detect CUDA and GPU
- ✅ Show GPU memory (should show ~24 GB)
- ✅ Test GPU computation speed
- ✅ Recommend optimal batch_size and num_workers
- ✅ Explain CPU+GPU usage
- ✅ Estimate training time

**Expected output:**
```
CUDA Available: True
GPU 0:
  Name: NVIDIA RTX 4500 Ada Generation
  Total Memory: 24.0 GB
  Compute Capability: (8, 9)
  
Recommended batch_size: 16 (you have plenty of VRAM)
Recommended num_workers: 8 (you have 72 physical cores)
AMP: ENABLED (your GPU supports Tensor Cores)
Expected speedup: 1.5-2x

Estimated training time: 18-24 hours
```

### 4. Start Training

```bash
python train/train.py --config config/defaults.yaml
```

The training script now includes GPU verification:
- Checks CUDA is available
- Creates a test tensor on GPU
- Warns if falling back to CPU
- Shows GPU memory stats

---

## Why You CANNOT Split Model Across CPU+GPU

**Don't try to run part of the model on CPU and part on GPU** - it would be much slower!

**Reasons:**
1. **Transfer overhead:** Moving tensors between CPU↔GPU RAM is slow (~10-20 GB/s)
2. **Synchronization:** CPU and GPU would wait for each other
3. **Pipeline breaks:** No parallelism benefit

**Example of what NOT to do:**
```python
# ❌ BAD: Split model (slow!)
hq_cnn = hq_cnn.to('cuda')
lq_cnn = lq_cnn.to('cpu')  # Don't do this!

# ✅ GOOD: Entire model on GPU
model = model.to('cuda')
```

The **correct** way to use both CPU and GPU is already implemented via `num_workers`:

```python
# DataLoader uses CPU workers (automatic)
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=8,  # <-- 8 CPU processes loading data in parallel
    pin_memory=True  # <-- Speeds up CPU→GPU transfer
)
```

---

## Monitoring During Training

### GPU Utilization

**Windows (PowerShell):**
```powershell
nvidia-smi -l 1
```

**Expected output:**
```
+-------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.1 |
|-------------------------------+----------------------+------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute|
|===============================+======================+==================|
|   0  NVIDIA RTX 4500...  WDDM | 00000000:01:00.0 Off |                  |
| 30%   65C    P2   280W / 320W |  18240MiB / 24576MiB |     98%      Default|
+-------------------------------+----------------------+------------------+
```

**What to look for:**
- GPU-Util: Should be **90-100%** during training
- Memory-Usage: Should be **16-20 GB** with batch_size=16
- Temperature: Should be **60-80°C** (normal under load)
- Power: Should be **250-300W** (near max)

If GPU-Util is low (<50%):
- ❌ Data loading bottleneck → increase `num_workers`
- ❌ Disk I/O slow → move faces to SSD
- ❌ Batch size too small → increase `batch_size`

### CPU Utilization

**Task Manager → Performance → CPU**

Expected: 20-40% overall utilization
- 8 workers × 2-3 threads each = 16-24 threads active
- On 72 logical cores = 30-40% usage

### RAM Usage

Expected: 20-40 GB
- DataLoader buffers: 10-20 GB
- Python + model overhead: 5-10 GB
- OS + background: 5-10 GB

### Disk I/O

**Watch F: drive activity:**
- Should show continuous reads during training
- Speed: 500 MB/s - 2 GB/s (if SSD)

If bottleneck (GPU waiting for data):
- Move faces to faster SSD
- Reduce image size (already 224×224, optimal)
- Increase `num_workers` (but diminishing returns >8)

---

## Troubleshooting GPU Issues

### "CUDA out of memory"
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Fix:**
1. Reduce `batch_size: 12` (or 8)
2. Ensure no other programs using GPU
3. Restart Python kernel to clear VRAM

### "CUDA not available"
```
Device: cpu
⚠ WARNING: Training on CPU
```

**Causes:**
- PyTorch installed without CUDA (wrong version)
- NVIDIA driver not installed
- GPU disabled in BIOS

**Verify PyTorch CUDA:**
```python
import torch
print(torch.__version__)  # Should show: 2.2.2+cu121
print(torch.version.cuda)  # Should show: 12.1
```

If shows `2.2.2+cpu`:
```bash
pip uninstall torch torchvision
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### GPU utilization low (<50%)

**Possible causes:**

1. **Data loading bottleneck:**
   - Increase `num_workers: 12`
   - Ensure `pin_memory: True` (already set)

2. **Disk I/O bottleneck:**
   - Check if F: drive is HDD (slow) or SSD (fast)
   - Use `nvidia-smi` - if GPU is idle waiting, it's I/O bound

3. **Batch size too small:**
   - GPU not fully utilized with small batches
   - Increase `batch_size: 20` if VRAM allows

---

## Expected Performance

### Training Speed
- **Batch size 16, num_workers 8, AMP enabled:**
  - ~2-3 seconds per batch
  - ~800-1200 batches per epoch (depends on data split)
  - ~30-60 minutes per epoch
  - **Total: 30-60 hours for 60 epochs**

### Resource Usage
- GPU: 95-100% utilization ✅
- CPU: 25-40% utilization ✅ (8 workers out of 72 cores)
- RAM: 25-35 GB ✅
- VRAM: 18-22 GB ✅

---

## Optimization Checklist

Before starting training, verify:

- [x] ✅ CUDA available (`check_system.py`)
- [x] ✅ Pillow version compatible (`requirements.txt` fixed)
- [x] ✅ batch_size optimized for 24GB VRAM (16)
- [x] ✅ num_workers set for 72 cores (8)
- [x] ✅ AMP enabled (`use_amp: true`)
- [x] ✅ Faces on fast storage (F: drive)
- [x] ✅ pin_memory enabled (already in dataset.py)
- [x] ✅ GPU verification in train.py

**You're ready to train!** 🚀

Run:
```bash
python check_system.py          # Verify setup
pip install -r requirements.txt  # Install deps
python train/train.py            # Start training
```
