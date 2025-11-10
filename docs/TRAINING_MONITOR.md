# Training Progress Monitoring Guide

## ✅ Training Started Successfully!

**Current Status:**
- Model: Dual-stream EfficientNet-B4/B0 + ConvLSTM (58.5M parameters)
- Dataset: 9,318 training samples, 1,646 validation samples
- Configuration: 582 train batches, 103 val batches per epoch
- GPU: NVIDIA RTX 4500 Ada (CUDA 12.1)
- AMP: Enabled (mixed precision training)

---

## What to Expect

### First Epoch (Currently Running)
The first epoch will be slower because:
1. **Data loading initialization** - Workers are spawning and caching
2. **Model compilation** - CUDA kernels are being compiled
3. **EfficientNet pretrained weights** - Already downloaded ✓

**Expected time for first epoch:** 60-90 minutes  
**Subsequent epochs:** 40-60 minutes

### Training Progress
You'll see output like this:
```
Training: 100%|███████████| 582/582 [45:23<00:00, 4.68s/it]
Train - Loss: 0.4523, Acc: 0.7845, AUC: 0.8234
Validation: 100%|███████| 103/103 [08:12<00:00, 4.78s/it]
Val   - Loss: 0.3891, Acc: 0.8123, AUC: 0.8567
✓ Best model saved (val_loss: 0.3891)
```

**Key metrics:**
- **Loss:** Should decrease over epochs (target: <0.1-0.2)
- **Accuracy:** Should increase (target: >0.95)
- **AUC:** Should increase (target: >0.97)

### Expected Timeline

| Epoch | Train Loss | Val Loss | Val Acc | Val AUC | Time   |
|-------|-----------|----------|---------|---------|--------|
| 1     | ~0.5-0.6  | ~0.4-0.5 | 0.70-0.80 | 0.75-0.85 | 60-90m |
| 5     | ~0.3-0.4  | ~0.3-0.4 | 0.80-0.85 | 0.85-0.90 | 40-60m |
| 10    | ~0.2-0.3  | ~0.2-0.3 | 0.85-0.90 | 0.90-0.95 | 40-60m |
| 20    | ~0.1-0.2  | ~0.15-0.25 | 0.90-0.95 | 0.95-0.97 | 40-60m |
| 40+   | ~0.05-0.1 | ~0.1-0.2 | 0.95-0.98 | 0.97-0.99 | 40-60m |

**Total estimated time:** 40-60 hours for 60 epochs

---

## Monitoring GPU Usage

### Open a second terminal and run:
```cmd
nvidia-smi -l 1
```

**Expected GPU stats during training:**
```
GPU-Util: 95-100%              ✓ Good - GPU fully utilized
Memory-Usage: 18000-22000 MiB  ✓ Good - Using most of 24GB
Temperature: 60-80°C           ✓ Normal under load
Power: 180-210W                ✓ Near max TDP
```

**Warning signs:**
- GPU-Util < 50% → Data loading bottleneck (disk I/O slow)
- Memory-Usage < 10GB → Batch size too small
- Temperature > 85°C → Check cooling/fans

---

## Output Files

### During Training

**Logs:**
```
logs/
  training_log.csv    # Epoch-level metrics (loss, acc, AUC, lr)
```

**Checkpoints:**
```
checkpoints/
  best_model.pth      # Best validation loss (auto-saved)
  last_model.pth      # Most recent epoch (auto-saved)
```

### Monitoring Progress

**View training log:**
```cmd
type logs\training_log.csv
```

**Check latest metrics:**
```cmd
J:\DF\.venv\Scripts\python.exe -c "import pandas as pd; df=pd.read_csv('logs/training_log.csv'); print(df.tail(10))"
```

---

## Training Controls

### Stop Training Gracefully
- Press `Ctrl+C` once
- Current epoch will finish
- Model will save to `checkpoints/last_model.pth`

### Resume Training (if interrupted)
The training script doesn't automatically resume, but you can modify it:
1. Load `last_model.pth` in `train.py`
2. Set `start_epoch` to the saved epoch
3. Re-run training

Or start fresh with saved knowledge (models learn fast from pretrained weights).

### Early Stopping
Training will auto-stop if validation loss doesn't improve for 10 epochs.

---

## Common Issues & Solutions

### Training is slow (<<1 it/s)
**Symptoms:**
- Progress bar shows < 0.5 iterations/second
- GPU utilization low (<50%)

**Causes & Fixes:**
1. **Disk I/O bottleneck** (faces on slow HDD)
   - Check if F: drive is HDD or SSD
   - Monitor disk activity in Task Manager
   - Solution: Move faces to SSD or increase `num_workers`

2. **Too many workers** (context switching overhead)
   - Solution: Reduce `num_workers` from 8 to 4 in config

3. **Batch size too small**
   - Solution: Increase `batch_size` from 16 to 20 (if VRAM allows)

### Out of Memory (OOM)
**Error:** `RuntimeError: CUDA out of memory`

**Fix:** Edit `config/defaults.yaml`:
```yaml
training:
  batch_size: 12  # Reduce from 16
  num_workers: 4  # Reduce workers if still OOM
```

### Loss becomes NaN
**Symptoms:**
- Loss shows "nan" instead of number
- Training crashes

**Fixes:**
1. Reduce learning rate: `learning_rate: 0.00005` in config
2. Check for corrupted images in dataset
3. Ensure gradient clipping is enabled (already set to 1.0)

### Accuracy stuck at ~50%
**Symptoms:**
- Validation accuracy doesn't improve past 50-60%
- Model is just guessing

**Possible causes:**
1. **Data imbalance** - Check train/val split has equal real/fake
2. **Learning rate too high** - Model not converging
3. **Bug in labels** - Real/fake labels reversed

**Debug:**
```cmd
J:\DF\.venv\Scripts\python.exe -c "import pandas as pd; pairs=pd.read_csv('training_pairs.csv'); print('Real videos:', len(pairs)); print('Fake videos:', len(pairs))"
```

---

## Performance Optimization

### If GPU utilization is low:

1. **Increase num_workers** (more data loading parallelism):
```yaml
training:
  num_workers: 10  # Try 10 or 12
```

2. **Increase batch_size** (better GPU saturation):
```yaml
training:
  batch_size: 20  # If VRAM allows
```

3. **Check disk speed:**
```cmd
# Run in PowerShell
Measure-Command { Get-Content F:\real\id16_0000\* | Out-Null }
```
If > 1 second for ~300 images, disk is slow.

### If training is working well:

You're already optimized! Just let it run.

Expected speed: **~2-3 seconds per batch** with current settings.

---

## When Training Completes

### Expected Final Results (Celeb-DF)
- **Validation Accuracy:** 95-98%
- **AUC-ROC:** 0.97-0.99
- **Best model:** `checkpoints/best_model.pth`

### Evaluate on Test Set
```cmd
J:\DF\.venv\Scripts\python.exe train\inference.py --checkpoint checkpoints\best_model.pth --test-pairs training_pairs.csv --output-dir results
```

This will:
- Score all videos
- Generate `results/predictions.csv`
- Generate `results/metrics.json` with final performance

### Next Steps After Training
1. Analyze errors (which videos are misclassified?)
2. Try inference on new videos
3. Export model for deployment
4. Fine-tune on additional data if needed

---

## Current Training Session

**Started:** November 3, 2025  
**Expected completion:** November 5-6, 2025 (40-60 hours)  
**Configuration:**
- Batch size: 16
- Num workers: 8
- Learning rate: 0.0001
- Scheduler: ReduceLROnPlateau
- Early stopping patience: 10 epochs

**You can leave this running overnight/over days. The script will save checkpoints automatically.**

Monitor progress anytime by checking:
- Terminal output (live)
- `logs/training_log.csv` (metrics per epoch)
- `nvidia-smi` (GPU usage)

Good luck with training! 🚀
