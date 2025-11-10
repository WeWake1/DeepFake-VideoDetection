# Installation Guide - CUDA Setup for Training

## ✅ Current Status (Verified Working)

Your system is now correctly configured:
- ✅ PyTorch 2.2.2+cu121 (CUDA 12.1)
- ✅ CUDA available: True
- ✅ GPU: NVIDIA RTX 4500 Ada (24 GB)
- ✅ NumPy 1.26.4 (compatible)
- ✅ Pillow 10.2.0 (compatible)
- ✅ OpenCV 4.10.0 (compatible)

## Quick Start - You're Ready to Train!

```cmd
cd J:\DF
.venv\Scripts\python.exe train\train.py --config config\defaults.yaml
```

---

## Fresh Installation Instructions (For Future Reference)

If you need to reinstall or set up on another machine:

### Step 1: Create Virtual Environment
```cmd
cd J:\DF
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install PyTorch with CUDA (CRITICAL - Do this first!)
```cmd
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision==0.17.2+cu121
```

### Step 3: Install facenet-pytorch (without letting it downgrade PyTorch)
```cmd
python -m pip install --no-deps facenet-pytorch==2.6.0
python -m pip install requests
```

### Step 4: Install remaining dependencies
```cmd
python -m pip install numpy==1.26.4 pillow==10.2.0 opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 tqdm pandas psutil pyyaml scikit-learn matplotlib efficientnet-pytorch
```

### Step 5: Verify GPU
```cmd
python verify_gpu.py
```

Expected output:
```
✓ GPU is working correctly!
torch: 2.2.2+cu121
CUDA available: True
GPU name: NVIDIA RTX 4500 Ada Generation
```

---

## System Specifications (Verified)

**Hardware:**
- GPU: NVIDIA RTX 4500 Ada Generation (24 GB VRAM)
- CPU: 36 physical cores / 72 logical cores
- RAM: 255 GB
- Storage: F: (11 TB), H: (954 GB), I: (954 GB), J: (954 GB)

**Performance:**
- GPU Compute: ~23,000 GFLOPS (1024x1024 matmul)
- GPU Memory Bandwidth: 140 GB/s
- Estimated training time: 18-36 hours for 60 epochs

**Recommended Settings (already in config/defaults.yaml):**
- batch_size: 16 (you can use 12-20 depending on VRAM)
- num_workers: 8 (for data loading parallelism)
- use_amp: true (Tensor Cores enabled, 1.5-2x speedup)

---

## Troubleshooting

### If CUDA becomes unavailable after pip install

**Symptom:** `torch.cuda.is_available()` returns False

**Cause:** Another package downgraded PyTorch to CPU version

**Fix:**
```cmd
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision==0.17.2+cu121
python verify_gpu.py
```

### If NumPy warnings appear

**Symptom:** "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"

**Fix:**
```cmd
python -m pip install --force-reinstall numpy==1.26.4
```

### If Pillow conflicts appear

**Symptom:** "facenet-pytorch requires Pillow<10.3.0 but you have 11.x"

**Fix:**
```cmd
python -m pip install --force-reinstall "pillow==10.2.0"
```

---

## Next Steps - Start Training

1. **Verify configuration:**
```cmd
python check_system.py
```

2. **Start training:**
```cmd
python train\train.py --config config\defaults.yaml
```

3. **Monitor GPU usage (separate terminal):**
```cmd
nvidia-smi -l 1
```

Expected GPU utilization during training: **90-100%**

4. **Monitor training progress:**
- Live console output shows batch loss/accuracy
- Logs saved to `logs/training_log.csv`
- Checkpoints saved to `checkpoints/best_model.pth`

---

## Training Tips

**If GPU utilization is low (<50%):**
- Increase `num_workers` in config (try 10-12)
- Ensure faces are on SSD, not HDD
- Increase `batch_size` if VRAM available

**If OOM (Out of Memory):**
- Reduce `batch_size` to 12, 10, or 8
- Reduce `num_workers` to 4

**To speed up training:**
- Already using AMP (enabled)
- Already using Tensor Cores (RTX 4500 supports them)
- Increase batch_size to 20 if VRAM allows

---

## Current Package Versions (Working)

```
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
numpy: 1.26.4
pillow: 10.2.0
opencv-python: 4.10.0.84
facenet-pytorch: 2.6.0
efficientnet-pytorch: 0.7.1
pyyaml: 6.0+
scikit-learn: 1.7.2
matplotlib: 3.10.7
```

All versions tested and verified compatible.
