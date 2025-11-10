# Project Structure Guide - DeepFake Detection

**Date:** November 11, 2025  
**Purpose:** Understand what's in each folder and which files are critical

---

## 📁 **Complete Directory Structure**

```
J:\DF/
├── 📦 checkpoints/              # Trained model weights
├── 📦 co/                       # Preprocessing scripts
├── 📦 config/                   # Training configuration
├── 📦 data/                     # Dataset mappings & metadata
├── 📦 docs/                     # Documentation
├── 📦 logs/                     # Training logs
├── 📦 paper/                    # Research paper materials
├── 📦 results/                  # Model predictions & analysis
├── 📦 scripts/                  # Analysis & utility scripts
├── 📦 train/                    # Core training pipeline
├── 📦 .venv/                    # Python environment (ignore)
├── 📦 DS/                       # Source dataset videos
├── 📦 FR/                       # Extracted frames (on J: drive)
├── 📦 Celeb-synthesis FAKE FRAMES-1/  # Frame backup
├── 📦 friend's architecture/    # Friend's code (reference only)
├── 📄 README.md                 # Main documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 .gitignore               # Git ignore rules
└── 📄 QUICK_REFERENCE.md       # Command cheatsheet
```

---

## 🔥 **MOST IMPORTANT FILES** (Top Priority)

### **1. Core Training Pipeline** ⭐⭐⭐⭐⭐

| File | Purpose | Lines | Importance |
|------|---------|-------|------------|
| **`train/train.py`** | Main training script - orchestrates entire training process | 310 | 🔥🔥🔥🔥🔥 CRITICAL |
| **`train/models.py`** | Model architecture - defines your dual-stream EfficientNet + ConvLSTM | 248 | 🔥🔥🔥🔥🔥 CRITICAL |
| **`train/dataset.py`** | Data loading - loads video pairs and creates training batches | 245 | 🔥🔥🔥🔥 VERY IMPORTANT |
| **`train/inference.py`** | Testing & evaluation - runs model on test videos | 358 | 🔥🔥🔥🔥 VERY IMPORTANT |
| **`train/utils.py`** | Helper functions - metrics, checkpointing, transforms | 198 | 🔥🔥🔥 IMPORTANT |

**What they do:**
- `train.py` → Runs training loop, saves checkpoints, logs metrics
- `models.py` → **YOUR ARCHITECTURE** (dual-stream EfficientNet-B4/B0 + ConvLSTM)
- `dataset.py` → Loads face images from F: drive, creates 10-frame sequences
- `inference.py` → Evaluates model on test set, generates predictions.csv
- `utils.py` → Calculate accuracy/AUC, save checkpoints, apply transforms

---

### **2. Trained Model** ⭐⭐⭐⭐⭐

| File | Purpose | Size | Importance |
|------|---------|------|------------|
| **`checkpoints/best_model.pth`** | Your trained model (epoch 12, 100% val accuracy) | 230MB | 🔥🔥🔥🔥🔥 CRITICAL |
| **`checkpoints/last_model.pth`** | Final training checkpoint (epoch 22) | 230MB | 🔥🔥 BACKUP |

**What they contain:**
- Model weights (58.5M parameters)
- Optimizer state
- Training epoch number
- Validation metrics

**Best model stats:**
- Epoch: 12
- Val Loss: 0.0001
- Val Accuracy: 100%
- Test Accuracy: 100% (1,646/1,646)

---

### **3. Configuration** ⭐⭐⭐⭐

| File | Purpose | Size | Importance |
|------|---------|------|------------|
| **`config/defaults.yaml`** | All hyperparameters and paths | ~100 lines | 🔥🔥🔥🔥 CRITICAL |

**What it contains:**
```yaml
data:
  real_path: F:/real          # Face images location
  fake_path: F:/fake
  sequence_length: 10         # 10-frame sequences
  frame_skip: 3              # Every 3rd frame

model:
  hq_backbone: efficientnet-b4  # High-quality stream
  lq_backbone: efficientnet-b0  # Low-quality stream
  lstm_hidden: 256           # ConvLSTM layer 1
  lstm_hidden_2: 128         # ConvLSTM layer 2

training:
  batch_size: 16
  num_workers: 8
  lr: 0.0001
  epochs: 100
  early_stop_patience: 10

preprocessing:
  hq_size: 224               # HQ stream resolution
  lq_size: 112               # LQ stream resolution
```

---

### **4. Dataset Mappings** ⭐⭐⭐⭐

| File | Purpose | Rows | Importance |
|------|---------|------|------------|
| **`data/training_pairs.csv`** | 5,490 real-fake video pairs for training | 5,490 | 🔥🔥🔥🔥 CRITICAL |
| **`data/enhanced_mapping.csv`** | Complete video metadata (frames, status, drive) | 11,229 | 🔥🔥🔥 IMPORTANT |
| **`data/frame_mapping.csv`** | Frame extraction results | 6,229 | 🔥🔥 USEFUL |
| **`data/celebrity_mapping.json`** | Celebrity ID to video relationships | 59 celebs | 🔥🔥 REFERENCE |

**training_pairs.csv structure:**
```csv
pair_id,real_video,fake_video,face_source_id,real_frames_path,fake_frames_path,...
1,id16_0000,id0_id16_0000,0,H:\Celeb-real FRAMES\id16_0000,J:\DF\FR\...
```

**This is what your model trains on!**

---

### **5. Training Results** ⭐⭐⭐⭐

| File | Purpose | Size | Importance |
|------|---------|------|------------|
| **`logs/training_log.csv`** | Epoch-by-epoch training metrics | 22 rows | 🔥🔥🔥🔥 CRITICAL |
| **`results/predictions.csv`** | Test set predictions (1,646 videos) | 1,646 rows | 🔥🔥🔥🔥 CRITICAL |
| **`results/metrics.json`** | Final test metrics (accuracy, AUC, etc.) | Small | 🔥🔥🔥 IMPORTANT |

**training_log.csv contains:**
```csv
epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc,lr,time
1,0.5234,0.7845,0.8567,0.4891,0.8123,0.8734,0.0001,245.3
...
12,0.0823,0.9956,0.9998,0.0001,1.0000,1.0000,0.0001,198.7  ← BEST
...
22,0.0456,0.9989,1.0000,0.0012,0.9994,1.0000,0.0001,187.2  ← STOPPED
```

---

## 📂 **Folder-by-Folder Breakdown**

### **`checkpoints/`** - Trained Models

```
checkpoints/
├── best_model.pth       # 🔥🔥🔥🔥🔥 Your winning model (epoch 12)
└── last_model.pth       # 🔥🔥 Final checkpoint (epoch 22)
```

**Purpose:** Stores trained model weights  
**Critical Files:** `best_model.pth` (230MB) - YOUR ENTIRE TRAINED MODEL  
**Used By:** `train/inference.py`, `scripts/test_individual_videos.py`

---

### **`co/`** - Preprocessing Scripts

```
co/
├── face_detect_mtcnn_gpu(final).py    # 🔥🔥🔥🔥 Face detection (GPU)
├── framer_cpu(final)                  # 🔥🔥🔥 Frame extraction (CPU)
├── create_mappings.py                 # 🔥🔥 Generate training_pairs.csv
├── verify_face_extraction.py          # 🔥 Check face extraction completeness
└── FACE_DETECTION_README.md           # 🔥 Documentation
```

**Purpose:** Video → Frames → Faces preprocessing  
**Critical Files:**
- `face_detect_mtcnn_gpu(final).py` (407 lines) - Detects faces from frames, saves to F: drive
- `framer_cpu(final)` - Extracts frames from videos

**When to use:** Only when preprocessing NEW videos (not needed for training)

---

### **`config/`** - Configuration

```
config/
└── defaults.yaml         # 🔥🔥🔥🔥 All hyperparameters
```

**Purpose:** Central configuration for training  
**Critical Files:** `defaults.yaml` - EVERYTHING is configured here  
**Used By:** All training/inference scripts

---

### **`data/`** - Dataset Mappings

```
data/
├── training_pairs.csv              # 🔥🔥🔥🔥 5,490 video pairs (CRITICAL)
├── enhanced_mapping.csv            # 🔥🔥🔥 Complete video metadata
├── frame_mapping.csv               # 🔥🔥 Frame extraction results
├── face_mapping.csv                # 🔥 Face detection results (empty?)
├── celebrity_mapping.json          # 🔥🔥 Celebrity ID relationships
├── real_to_fake_mapping.json       # 🔥 Real→Fake mappings
├── face_detection_results.json     # 🔥 Face detection statistics
└── face_extraction_verification.json  # 🔥 Completeness check
```

**Purpose:** Dataset metadata and relationships  
**Critical Files:**
- `training_pairs.csv` - **WHAT YOUR MODEL TRAINS ON** (5,490 pairs)
- `enhanced_mapping.csv` - Where each video's frames are located

---

### **`docs/`** - Documentation

```
docs/
├── architecture.md              # 🔥🔥🔥 Model architecture explanation
├── INSTALLATION.md              # 🔥🔥 Setup guide
├── TRAINING_MONITOR.md          # 🔥 Training progress guide
├── RUN_VALIDATION.md            # 🔥 Validation commands
├── GPU_USAGE_GUIDE.md           # 🔥 GPU optimization
├── CLEANUP_COMMANDS.md          # File organization
└── organization_report.txt      # Workspace structure report
```

**Purpose:** Documentation and guides  
**Critical Files:**
- `architecture.md` - Explains your dual-stream model design
- `INSTALLATION.md` - How to set up PyTorch CUDA

---

### **`logs/`** - Training Logs

```
logs/
└── training_log.csv      # 🔥🔥🔥🔥 Epoch-by-epoch metrics (22 epochs)
```

**Purpose:** Training history  
**Critical Files:** `training_log.csv` - Used for plotting training curves  
**What it tracks:** Loss, accuracy, AUC per epoch (train + val)

---

### **`paper/`** - Research Paper Materials

```
paper/
├── methodology/
│   ├── architecture.md       # 🔥🔥🔥 Model design explanation
│   ├── INSTALLATION.md       # Setup instructions
│   └── README.md            # Overview
├── results/
│   ├── training_log.csv     # 🔥🔥🔥 Training metrics (copy)
│   └── best_model.pth       # Model checkpoint (copy)
├── code_reference/
│   ├── train.py             # Training script (copy)
│   ├── models.py            # 🔥🔥🔥 Architecture (copy)
│   ├── dataset.py           # Data loading (copy)
│   ├── inference.py         # Evaluation (copy)
│   └── defaults.yaml        # Config (copy)
├── data_description/
│   ├── training_pairs.csv   # Dataset pairs (copy)
│   ├── enhanced_mapping.csv # Metadata (copy)
│   └── frame_mapping.csv    # Frame info (copy)
└── figures/
    └── (empty - needs plots)
```

**Purpose:** Organized materials for writing research paper  
**Critical Files:**
- `methodology/architecture.md` - Draft of methodology section
- `code_reference/models.py` - Reference for explaining architecture
- `figures/` - **NEEDS WORK** (training curves, architecture diagram)

---

### **`results/`** - Model Predictions

```
results/
├── predictions.csv          # 🔥🔥🔥🔥 1,646 test predictions
├── metrics.json            # 🔥🔥🔥 Test accuracy, AUC, confusion matrix
└── interesting_videos.txt  # 🔥🔥 Edge cases for inspection
```

**Purpose:** Test set evaluation results  
**Critical Files:**
- `predictions.csv` - Every test video's prediction + confidence
- `metrics.json` - Overall performance (100% accuracy!)

**predictions.csv structure:**
```csv
video_name,true_label,prediction,score
id49_0009,0,0,0.000003937  ← Real video, predicted Real (99.9996% confident)
id53_id49_0009,1,1,0.9999963  ← Fake video, predicted Fake (99.9996% confident)
```

---

### **`scripts/`** - Analysis Scripts

```
scripts/
├── investigate_accuracy.py      # 🔥🔥🔥 Analyzes 100% accuracy (207 lines)
└── test_individual_videos.py    # 🔥🔥🔥 Edge case analysis (299 lines)
```

**Purpose:** Post-training analysis and validation  
**Critical Files:**
- `investigate_accuracy.py` - Checks for overfitting, data leakage
- `test_individual_videos.py` - Finds hardest/easiest videos

**When to use:** After training, for paper analysis

---

### **`train/`** - Core Training Code

```
train/
├── train.py          # 🔥🔥🔥🔥🔥 Main training script (310 lines)
├── models.py         # 🔥🔥🔥🔥🔥 Model architecture (248 lines)
├── dataset.py        # 🔥🔥🔥🔥 Data loader (245 lines)
├── inference.py      # 🔥🔥🔥🔥 Evaluation (358 lines)
├── utils.py          # 🔥🔥🔥 Helper functions (198 lines)
├── __init__.py       # Package init
├── README.md         # Training guide
└── __pycache__/      # Python cache (ignore)
```

**Purpose:** YOUR ENTIRE TRAINING PIPELINE  
**Critical Files:** All 5 .py files are essential!

**Dependency flow:**
```
train.py
  ├─ uses models.py (create_model)
  ├─ uses dataset.py (DeepfakeDataset)
  └─ uses utils.py (calculate_metrics, save_checkpoint)

inference.py
  ├─ uses models.py (create_model)
  └─ uses utils.py (calculate_metrics, load_checkpoint)
```

---

## 🎯 **THE 10 MOST CRITICAL FILES**

If you could only keep 10 files, these are THE ESSENTIAL ONES:

### **Rank 1-5: CANNOT FUNCTION WITHOUT THESE**

1. **`train/models.py`** - YOUR ARCHITECTURE (dual-stream EfficientNet + ConvLSTM)
2. **`checkpoints/best_model.pth`** - YOUR TRAINED MODEL (230MB)
3. **`config/defaults.yaml`** - ALL HYPERPARAMETERS
4. **`data/training_pairs.csv`** - WHAT YOU TRAINED ON (5,490 pairs)
5. **`train/train.py`** - HOW TO TRAIN THE MODEL

### **Rank 6-10: VERY IMPORTANT BUT RECOVERABLE**

6. **`train/inference.py`** - How to evaluate/test
7. **`train/dataset.py`** - How to load data
8. **`logs/training_log.csv`** - Training history (for plots)
9. **`results/predictions.csv`** - Test results (100% accuracy proof)
10. **`results/metrics.json`** - Overall metrics

---

## 📊 **File Size Analysis**

### **Large Files (>100MB):**
```
checkpoints/best_model.pth          230 MB  🔥 Your trained model
checkpoints/last_model.pth          230 MB  Backup checkpoint
```

### **Medium Files (1-100MB):**
```
data/face_detection_results.json    ~50 MB  Face detection logs
data/celebrity_mapping.json         ~40 MB  Celebrity relationships
data/real_to_fake_mapping.json      ~35 MB  Mapping data
```

### **Small Files (<1MB):**
```
All .py scripts                     <1 KB each
All .csv files                      <10 MB each
All .md docs                        <100 KB each
```

---

## 🔄 **File Usage Flow**

### **Training Pipeline:**
```
1. config/defaults.yaml
   ↓ (loads config)
2. train/train.py
   ↓ (imports)
3. train/models.py (creates model)
4. train/dataset.py (loads data from data/training_pairs.csv)
   ↓ (reads face images from F:/real/ and F:/fake/)
5. Training loop runs
   ↓ (saves checkpoints)
6. checkpoints/best_model.pth
   ↓ (logs metrics)
7. logs/training_log.csv
```

### **Inference Pipeline:**
```
1. train/inference.py
   ↓ (loads model)
2. checkpoints/best_model.pth
   ↓ (loads config)
3. config/defaults.yaml
   ↓ (loads test pairs)
4. data/training_pairs.csv (uses last 15% as test)
   ↓ (runs predictions)
5. results/predictions.csv
6. results/metrics.json
```

### **Preprocessing Pipeline (for NEW videos):**
```
1. Raw video.mp4
   ↓ (extract frames)
2. co/framer_cpu(final)
   ↓ (frames → faces)
3. co/face_detect_mtcnn_gpu(final).py
   ↓ (saves to)
4. F:/real/video_name/*.jpg (aligned face crops)
   ↓ (can now be used by)
5. train/inference.py (for detection)
```

---

## 🎓 **For Your Paper**

### **Files You'll Reference:**

**Methodology Section:**
- `train/models.py` - Architecture details
- `config/defaults.yaml` - Hyperparameters
- `paper/methodology/architecture.md` - Design explanation

**Results Section:**
- `logs/training_log.csv` - Training curves
- `results/metrics.json` - Final performance
- `results/predictions.csv` - Per-video results

**Data Description:**
- `data/training_pairs.csv` - Dataset composition
- `data/enhanced_mapping.csv` - Video statistics

---

## 🗑️ **Files You Can Ignore**

### **Not Important:**
- `.venv/` - Python environment (don't touch)
- `__pycache__/` - Python cache (auto-generated)
- `friend's architecture/` - Reference only
- `DS/` - Original videos (already processed)
- `FR/` - Frame backup (faces on F: drive)
- `Celeb-synthesis FAKE FRAMES-1/` - Frame backup

### **Temporary/Generated:**
- `organization_report.txt` - Just documentation
- `CLEANUP_COMMANDS.md` - Just instructions
- `organize_files.py` - Already ran

---

## 🚀 **Quick Action Guide**

### **Want to understand the architecture?**
→ Read: `train/models.py` (248 lines)

### **Want to retrain the model?**
→ Run: `train/train.py` (uses `config/defaults.yaml`)

### **Want to test on new videos?**
→ Run: `train/inference.py` (needs faces in F:/real/ or F:/fake/)

### **Want to analyze results?**
→ Read: `results/predictions.csv`, `results/metrics.json`

### **Want to write your paper?**
→ Use: `paper/` folder (methodology, results, figures)

### **Want to preprocess new videos?**
→ Run: `co/framer_cpu(final)` → `co/face_detect_mtcnn_gpu(final).py`

---

## 📋 **Summary Table**

| Folder | # Files | Most Important File | Purpose | Criticality |
|--------|---------|-------------------|---------|-------------|
| **train/** | 7 | `models.py` | Core training code | 🔥🔥🔥🔥🔥 |
| **checkpoints/** | 2 | `best_model.pth` | Trained models | 🔥🔥🔥🔥🔥 |
| **config/** | 1 | `defaults.yaml` | Configuration | 🔥🔥🔥🔥 |
| **data/** | 9 | `training_pairs.csv` | Dataset mappings | 🔥🔥🔥🔥 |
| **logs/** | 1 | `training_log.csv` | Training history | 🔥🔥🔥🔥 |
| **results/** | 3 | `predictions.csv` | Test results | 🔥🔥🔥🔥 |
| **scripts/** | 2 | `investigate_accuracy.py` | Analysis tools | 🔥🔥🔥 |
| **co/** | 5 | `face_detect_mtcnn_gpu(final).py` | Preprocessing | 🔥🔥🔥 |
| **docs/** | 7 | `architecture.md` | Documentation | 🔥🔥 |
| **paper/** | ~15 | `methodology/architecture.md` | Paper materials | 🔥🔥 |

---

## ✅ **Key Takeaways**

1. **Your model is in:** `train/models.py` (248 lines of pure architecture)
2. **Your trained weights are in:** `checkpoints/best_model.pth` (230MB)
3. **Your training data is defined in:** `data/training_pairs.csv` (5,490 pairs)
4. **Your results are in:** `results/predictions.csv` (1,646 test samples, 100% accuracy)
5. **Everything is configured in:** `config/defaults.yaml` (hyperparameters, paths)

**To understand your entire project, read these 5 files in order:**
1. `config/defaults.yaml` (configuration)
2. `train/models.py` (architecture)
3. `train/dataset.py` (data loading)
4. `train/train.py` (training loop)
5. `train/inference.py` (evaluation)

**That's ~1,400 lines of code that define your entire system!** 🎉

---

Does this help clarify the structure? Would you like me to dive deeper into any specific folder or file?
