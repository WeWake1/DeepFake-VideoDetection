# Face Detection, Cropping, and Alignment for Deepfake Detection

This directory contains scripts for preprocessing video frames by detecting, cropping, and aligning faces for deepfake detection training.

## 📋 Overview

Three scripts are provided with different performance characteristics:

1. **`face_detect_gpu.py`** - GPU-accelerated RetinaFace (FASTEST, MOST ACCURATE)
2. **`face_detect_cpu.py`** - CPU-optimized RetinaFace (ACCURATE, SLOWER)
3. **`face_detect_mtcnn.py`** - CPU-optimized MTCNN (FASTEST CPU, LESS ACCURATE)

## 🚀 Installation

### For GPU Version (RECOMMENDED):
```cmd
install_face_detection.bat
```

This installs:
- PyTorch with CUDA 12.1 support
- RetinaFace
- OpenCV and dependencies

### For CPU-Only Version:
```cmd
pip install retina-face mtcnn opencv-python pillow tqdm
```

## 📊 Performance Comparison

| Method | Speed (GPU) | Speed (CPU) | Accuracy | Use Case |
|--------|-------------|-------------|----------|----------|
| **RetinaFace GPU** | ⚡⚡⚡ Very Fast | - | ⭐⭐⭐ Excellent | Production, RTX GPU available |
| **RetinaFace CPU** | - | 🐌 Slow | ⭐⭐⭐ Excellent | No GPU, need accuracy |
| **MTCNN CPU** | - | 🐌🐌 Medium | ⭐⭐ Good | No GPU, need speed |

## 🎯 Features

### Face Detection
- **RetinaFace**: State-of-the-art face detector with landmark detection
- **MTCNN**: Lightweight cascade CNN for face detection
- Confidence thresholding (0.9 default)
- Minimum face size filtering (80px default)

### Face Alignment
- Eye-based alignment for consistent orientation
- Rotation correction based on eye positions
- Ensures horizontal eye alignment across all faces

### Face Cropping
- Standardized output size: 224x224 pixels
- High-quality resizing (LANCZOS interpolation)
- Preserves facial features for deepfake detection

## ⚙️ Configuration

Edit the configuration section in each script:

```python
# Face detection settings
FACE_SIZE = 224  # Output face size (224x224 for most models)
MIN_FACE_SIZE = 80  # Minimum face size to detect
CONFIDENCE_THRESHOLD = 0.9  # Detection confidence threshold

# Output format
OUTPUT_FORMAT = "jpg"  # jpg or png
JPEG_QUALITY = 95  # JPEG quality (1-100)
```

## 📁 Input/Output

### Input
- Reads from: `j:\DF\frame_mapping.csv`
- Processes frames from extracted videos

### Output
- Faces saved to 3 drives (round-robin distribution):
  - `H:\Celeb-DF FACES\`
  - `J:\DF\Celeb-DF FACES\`
  - `I:\Celeb-DF FACES\`
- Separate folders for real and fake faces
- Detection results: `j:\DF\face_detection_results.json`

## 🏃 Usage

### GPU Version (FASTEST):
```cmd
python j:\DF\co\face_detect_gpu.py
```

**Requirements:**
- NVIDIA GPU with CUDA support
- 24GB VRAM recommended (RTX 4500 Ada)
- Batch processing for maximum GPU utilization

### CPU Version (RetinaFace):
```cmd
python j:\DF\co\face_detect_cpu.py
```

**Features:**
- Uses all available CPU cores (64 workers on Xeon)
- Process-based parallelization
- Slower but maintains RetinaFace accuracy

### CPU Version (MTCNN):
```cmd
python j:\DF\co\face_detect_mtcnn.py
```

**Features:**
- Lighter weight than RetinaFace
- Faster CPU processing
- Good balance of speed vs. accuracy

## 📈 Expected Results

For Celeb-DF dataset (6,227 videos, ~2.3M frames):

### GPU Processing (RTX 4500):
- **Speed**: ~100-200 frames/second
- **Total time**: ~3-6 hours for full dataset
- **Face detection rate**: ~95-98%

### CPU Processing (72-core Xeon):
- **Speed**: ~10-30 frames/second
- **Total time**: ~20-60 hours for full dataset
- **Face detection rate**: ~95-98% (RetinaFace), ~90-95% (MTCNN)

## 🔄 Resume Capability

All scripts support automatic resume:
- Skips already processed videos
- Set `RESUME = True` in configuration
- Safe to interrupt and restart

## 📊 Output Statistics

After processing, you'll get:
```
Videos processed: 6227/6227
Total faces detected: 2,200,000+
Total frames processed: 2,338,957
Face detection rate: 95.8%
```

## 🎯 Why Face Alignment Matters

For deepfake detection:
1. **Consistency**: All faces aligned to same orientation
2. **Feature extraction**: Better CNN feature learning
3. **Comparison**: Easier to spot manipulation artifacts
4. **Training stability**: Reduces variance in input data

## 🛠️ Troubleshooting

### GPU Not Detected
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```
If False, reinstall PyTorch with CUDA support.

### Out of Memory (GPU)
Reduce `BATCH_SIZE` in configuration:
```python
BATCH_SIZE = 16  # Lower if OOM
```

### Out of Memory (CPU)
Reduce worker count:
```python
NUM_WORKERS = 32  # Lower if system freezes
```

### Low Detection Rate
Lower confidence threshold:
```python
CONFIDENCE_THRESHOLD = 0.85  # From 0.9
```

## 📝 Notes

- **GPU script** is HIGHLY RECOMMENDED for speed
- Face alignment is CRITICAL for deepfake detection
- 224x224 is standard for most deepfake detection models
- JPEG quality 95 balances file size and quality
- All scripts distribute output across 3 drives automatically

## 🔬 Next Steps

After face detection:
1. Verify face detection rate in results JSON
2. Spot-check aligned faces for quality
3. Proceed to feature extraction or training
4. Use aligned faces for temporal consistency analysis

## 💡 Tips

1. **Start with 10-20 videos** to test configuration
2. **Monitor disk space** - faces will take significant space
3. **Check detection rate** - should be >90% for good dataset
4. **GPU version** reduces processing from days to hours
5. **Set confidence threshold carefully** - too high misses faces, too low gets false positives
