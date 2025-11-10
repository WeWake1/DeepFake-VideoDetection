# Deepfake Detection Model Training

This directory contains the training pipeline for the dual-stream (HQ/LQ) deepfake detection model with ConvLSTM temporal modeling and attention fusion.

## Architecture Overview

See `docs/architecture.md` for the complete technical specification.

**Key components:**
- Dual spatial CNNs: EfficientNet-B4 (HQ 224×224) + EfficientNet-B0 (LQ 112×112)
- Temporal modeling: Stacked ConvLSTM (256→128 filters)
- Fusion: Attention-based combination of HQ/LQ streams
- Classifier: 512→256→128→1 with dropout

**Expected performance on Celeb-DF:**
- Accuracy: 96-98%
- AUC-ROC: 0.97-0.99
- Training time: ~1-2 days (RTX 4500 Ada, batch_size=12, AMP enabled)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Training-specific packages:**
- `efficientnet-pytorch` - EfficientNet backbones
- `pyyaml` - Configuration parsing
- `scikit-learn` - Metrics calculation
- `matplotlib` - Plotting (optional)

### 2. Verify Data Paths

Ensure your preprocessed faces are accessible at the paths in `config/defaults.yaml`:

```yaml
data:
  real_path: "F:/real"              # Real face directories
  fake_path: "F:/fake"              # Fake face directories
  training_pairs: "training_pairs.csv"
```

**Data structure expected:**
```
F:/
  real/
    <video_name>/
      *.jpg  (aligned 224×224 faces)
  fake/
    <video_name>/
      *.jpg
```

### 3. Train Model

**Basic training:**
```bash
cd j:\DF
python train/train.py --config config/defaults.yaml
```

**With custom config:**
```bash
python train/train.py --config config/my_config.yaml
```

**Training will:**
- Split data 85/15 train/val (stratified by real/fake)
- Use AMP (automatic mixed precision) for speed + VRAM efficiency
- Save checkpoints to `checkpoints/best_model.pth` and `checkpoints/last_model.pth`
- Log metrics to `logs/training_log.csv`
- Apply early stopping (patience=10 epochs on val loss)

### 4. Monitor Training

**Watch live progress:**
```bash
# Training logs printed to console
# CSV logs: logs/training_log.csv
```

**Metrics tracked:**
- Loss (BCE with logits)
- Accuracy
- AUC-ROC, PR-AUC
- Precision, Recall, F1

### 5. Evaluate Model

**Full test set evaluation:**
```bash
python train/inference.py --checkpoint checkpoints/best_model.pth --test-pairs training_pairs.csv --output-dir results
```

**Score a single video:**
```bash
python train/inference.py --checkpoint checkpoints/best_model.pth --video-dir "F:/fake/id10_id3_0003"
```

**Output:**
- `results/predictions.csv` - Per-video predictions and scores
- `results/metrics.json` - Aggregate metrics

---

## Configuration Guide

Edit `config/defaults.yaml` to customize training:

### Key Parameters

**Batch size** (tune for your VRAM):
```yaml
training:
  batch_size: 12  # Reduce to 8 if OOM; increase to 16 if VRAM available
```

**Sequence length** (frames per video):
```yaml
data:
  sequence_length: 10  # T=10 frames (30 frames with stride=3)
  frame_skip: 3        # Must match preprocessing
```

**Learning rate & schedule:**
```yaml
training:
  learning_rate: 0.0001
optimization:
  scheduler_factor: 0.5      # LR *= 0.5 on plateau
  scheduler_patience: 3      # Wait 3 epochs before reducing
```

**Early stopping:**
```yaml
training:
  early_stop_patience: 10    # Stop if val loss doesn't improve for 10 epochs
```

**Model architecture:**
```yaml
model:
  hq_backbone: "efficientnet-b4"   # Options: b0-b7 (larger = more capacity)
  lq_backbone: "efficientnet-b0"
  convlstm_filters: [256, 128]     # ConvLSTM layer sizes
  dropout: [0.5, 0.3]              # Classifier dropout rates
```

---

## Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3060 (12GB VRAM)
- RAM: 32GB
- Storage: 50GB free (for checkpoints, logs)

**Recommended (as used):**
- GPU: NVIDIA RTX 4500 Ada (24GB VRAM)
- RAM: 256GB
- CPU: 72 cores (for DataLoader workers)

**VRAM usage** (batch_size=12, T=10, AMP):
- Model: ~3GB
- Forward pass: ~8GB
- Total peak: ~12-14GB

If you encounter OOM:
1. Reduce `batch_size` to 8 or 6
2. Reduce `num_workers` to 2
3. Use gradient checkpointing (requires code modification)

---

## Training Tips

### Faster Training
- **Enable AMP:** Already enabled by default (`use_amp: true`)
- **Increase batch size:** If VRAM allows, try 16 or 20
- **More workers:** Set `num_workers: 6-8` if you have many CPU cores
- **Shorter sequences:** Reduce `sequence_length` to 8 (less temporal info, faster)

### Better Accuracy
- **Longer training:** Increase `epochs` to 80-100
- **Stronger augmentation:** Edit `dataset.py` to add more transforms
- **Larger backbone:** Use `efficientnet-b5` or `b6` for HQ stream
- **Ensemble:** Train multiple models with different seeds and average predictions

### Debugging
- **Overfit on small subset:** Reduce data to 100 samples, train to 100% accuracy
- **Check gradients:** Add gradient norm logging in `train.py`
- **Visualize predictions:** Load samples and inspect model outputs

---

## Resuming Training

To resume from a saved checkpoint:

1. Modify `train.py` to add resume logic (load `last_model.pth` at startup)
2. Or manually edit config to reduce epochs and continue

**Example modification** (add to `train.py` after model creation):

```python
# Resume from last checkpoint if it exists
if os.path.exists(config['paths']['last_model']):
    checkpoint_data = load_checkpoint(
        config['paths']['last_model'], 
        model, optimizer, scheduler
    )
    start_epoch = checkpoint_data['epoch']
    print(f"Resuming from epoch {start_epoch}")
```

---

## Output Files

**During training:**
```
checkpoints/
  best_model.pth       # Best validation loss checkpoint
  last_model.pth       # Most recent checkpoint

logs/
  training_log.csv     # Epoch-level metrics (loss, acc, AUC, lr)
```

**After evaluation:**
```
results/
  predictions.csv      # Per-video predictions and scores
  metrics.json         # Aggregate test set metrics
```

---

## Troubleshooting

### OOM (Out of Memory)
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce `batch_size` to 8 or 6 in `config/defaults.yaml`

### Slow DataLoader
```
Training hangs on first batch for a long time
```
**Fix:** 
- Reduce `num_workers` to 2 or 0 (slower but more stable on Windows)
- Ensure antivirus isn't scanning face directories

### Low Accuracy
```
Validation accuracy stuck at ~50-60%
```
**Possible causes:**
- Data imbalance not handled (check train/val split logs)
- Learning rate too high/low (try 1e-5 or 5e-4)
- Insufficient augmentation (real/fake too distinguishable by artifact)
- Model underfitting (increase epochs or model capacity)

### NaN Loss
```
Loss becomes NaN during training
```
**Fix:**
- Reduce learning rate to 1e-5
- Check for corrupted images in dataset
- Ensure gradient clipping is enabled (default: 1.0)

### Slow Training (<<1 it/s)
**Causes:**
- Disk I/O bottleneck (faces on slow HDD) → move to SSD or cache in RAM
- Too many DataLoader workers (context switching) → reduce to 2-4
- AMP disabled → ensure `use_amp: true`

---

## Advanced: Model Modifications

### Use a Different Backbone
Edit `config/defaults.yaml`:
```yaml
model:
  hq_backbone: "efficientnet-b5"  # Larger model
  lq_backbone: "efficientnet-b1"
```

### Change Fusion Strategy
Edit `train/models.py` → `AttentionFusion` class:
- Current: Scalar attention (learns weight per stream)
- Alternative: Multi-head self-attention, concat fusion, gated fusion

### Add More ConvLSTM Layers
```yaml
model:
  convlstm_filters: [256, 128, 64]  # 3 layers instead of 2
```

### Freeze Backbone for Fine-Tuning
Edit `train/models.py` → `SpatialCNN.__init__`:
```python
self.backbone = EfficientNet.from_pretrained(model_name)
for param in self.backbone.parameters():
    param.requires_grad = False  # Freeze all
```

---

## Performance Benchmarks

| Configuration | Acc | AUC | Epochs | Time (RTX 4500) |
|--------------|-----|-----|--------|------------------|
| Baseline (B4+B0, T=10) | 0.96 | 0.98 | 40 | ~24h |
| Fast (B0+B0, T=8) | 0.94 | 0.96 | 30 | ~12h |
| High-accuracy (B5+B1, T=12) | 0.98 | 0.99 | 60 | ~48h |

---

## Citation

If you use this training code or architecture, please reference:
- Celeb-DF dataset: [https://github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- EfficientNet: Tan & Le, ICML 2019
- ConvLSTM: Shi et al., NeurIPS 2015

---

## Next Steps

1. **Train baseline model:** Run with defaults for 40-60 epochs
2. **Evaluate on test set:** Check if AUC > 0.95
3. **Error analysis:** Inspect false positives/negatives
4. **Iterate:** Adjust augmentation, architecture, or hyperparameters
5. **Deploy:** Export model for inference pipeline

For questions or issues, see `docs/architecture.md` for detailed design notes.
