# Deepfake Detection - Celeb-DF Dataset 
 
## Project Overview 
State-of-the-art deepfake detection system using dual-stream EfficientNet architecture with ConvLSTM temporal modeling. Trained on Celeb-DF dataset with 100% validation and test accuracy.

### Key Results
- **Test Accuracy**: 100% (1,646/1,646 videos)
- **Validation Accuracy**: 100%
- **Training Time**: 3.5 hours
- **Model Size**: 58.5M parameters
- **Architecture**: Dual-Stream EfficientNet-B4/B0 + ConvLSTM

## Architecture Overview

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

**Why this works:**
- **Spatial CNNs** detect pixel-level artifacts, blending seams, GAN fingerprints
- **ConvLSTM** captures temporal inconsistencies (blink dynamics, lighting coherence)
- **Multi-quality streams** (HQ/LQ) make model robust to compression and real-world re-encodes
- **Attention fusion** learns which stream is more reliable per sample

## Hardware Requirements 
- GPU: NVIDIA RTX 4500 Ada (24GB VRAM) 
- CPU: Intel Xeon 72 cores 
- RAM: 256GB 
- Storage: Multiple SSDs for frame storage 
 
## Dataset 
- Celeb-DF: 590 real videos, 5,639 fake videos 
- 59 celebrities 
- 2.3M+ extracted frames 
- Training pairs: 5,490 (real-fake video pairs)
 
## Pipeline Stages
1. **Frame extraction** (OpenCV) - Extract frames from videos at 30fps
2. **Face detection** (MTCNN) - Detect and align faces to 224×224
3. **Quality streams** - Generate HQ (224×224) and LQ (112×112) sequences
4. **Temporal sampling** - Sample T=10 frames with stride=3
5. **Model training** - Dual-stream EfficientNet + ConvLSTM
6. **Inference** - Per-video deepfake probability score

## Project Structure

```
J:\DF/
├── preprocessing/          # Frame extraction + face detection (was: co/)
│   ├── framer_cpu(final)                    # CPU-optimized frame extraction
│   ├── face_detect_mtcnn_gpu(final).py      # GPU MTCNN face detection
│   ├── create_mappings.py                   # Video-frame mappings
│   └── verify_face_extraction.py            # Verification script
├── dataset/                # Original videos (was: DS/)
│   ├── List_of_testing_videos.txt
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   └── YouTube-real/
├── frames/                 # Extracted frames on J: (was: FR/)
│   ├── Celeb-real FRAMES/
│   └── Celeb-synthesis FRAMES/
├── train/                  # Training pipeline ⭐
│   ├── models.py           # Architecture definition (248 lines)
│   ├── train.py            # Training loop (310 lines)
│   ├── dataset.py          # Data loading (280 lines)
│   ├── inference.py        # Video inference (358 lines)
│   └── utils.py            # Helper functions
├── checkpoints/            # Trained models ⭐
│   └── best_model.pth      # 230MB, 100% accuracy
├── config/                 # Configuration ⭐
│   └── defaults.yaml       # All hyperparameters
├── data/                   # Dataset mappings ⭐
│   ├── training_pairs.csv  # 5,490 training pairs
│   ├── frame_mapping.csv
│   └── enhanced_mapping.csv
├── docs/                   # Documentation
├── logs/                   # Training logs
├── paper/                  # Research paper materials
│   └── methodology/
│       └── architecture.md # Full technical specification
├── results/                # Test results
├── scripts/                # Analysis scripts
└── reference_code/         # Friend's code (was: friend's architecture/)
```

## Key Scripts

### Preprocessing
- `preprocessing/framer_cpu(final)` - CPU-optimized frame extraction
- `preprocessing/face_detect_mtcnn_gpu(final).py` - GPU-accelerated face detection (MTCNN, FRAME_SKIP=3)
- `preprocessing/create_mappings.py` - Video-frame relationship mapping
- `preprocessing/verify_face_extraction.py` - Verify extraction completeness

### Training
- `train/train.py` - Main training script
- `train/models.py` - Architecture definition (DualStreamBackbone, ConvLSTM, AttentionFusion)
- `train/dataset.py` - Data loading with multi-quality streams
- `train/inference.py` - Video-level inference and evaluation

### Configuration
- `config/defaults.yaml` - All hyperparameters (batch_size=16, lr=1e-4, sequence_length=10)

## Data Locations
- **Faces (aligned)**: `F:/real/<video_name>/*.jpg`, `F:/fake/<video_name>/*.jpg`
- **Frames (original)**: Distributed across `H:/`, `I:/`, `J:/` (see `frame_mapping.csv`)
- **Frame skip**: 3 (affects expected counts and sampling)
- **Dataset mappings**: `data/training_pairs.csv` (5,490 pairs)

## Quick Start

### 1. Preprocess Videos
```bash
# Extract frames
python preprocessing/framer_cpu(final)

# Detect and align faces (GPU)
python preprocessing/face_detect_mtcnn_gpu(final).py

# Verify extraction
python preprocessing/verify_face_extraction.py
```

### 2. Train Model
```bash
# Train with default config
python train/train.py --config config/defaults.yaml

# Resume from checkpoint
python train/train.py --resume checkpoints/last_checkpoint.pth
```

### 3. Run Inference
```bash
# Test on single video
python train/inference.py --video F:/real/id0_0000 --model checkpoints/best_model.pth

# Evaluate on test set
python train/inference.py --mode test --model checkpoints/best_model.pth
```

## Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Accuracy | 99.98% | 100% | 100% |
| Loss | 0.0002 | 0.0001 | - |
| Videos | 4,392 | 549 | 1,646 |

**Test Results**: 1,646/1,646 correct (100% accuracy)
- Real videos: 823/823 correct
- Fake videos: 823/823 correct

## Documentation
- **Full architecture specification**: `paper/methodology/architecture.md`
- **Face detection notes**: `preprocessing/FACE_DETECTION_README.md`
- **Project structure guide**: `PROJECT_STRUCTURE_GUIDE.md`
- **Training logs**: `logs/training_log.csv`
- **Test results**: `results/test_results.csv`
 
## Installation 
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

See `requirements.txt` for full dependency list.

## Citation
```bibtex
@article{deepfake2025,
  title={Dual-Stream Multi-Quality Deepfake Detection with ConvLSTM},
  author={Your Name},
  journal={In preparation},
  year={2025}
}
```

## License
MIT License 
