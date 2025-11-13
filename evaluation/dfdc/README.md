# DFDC Evaluation Pipeline

Cross-dataset evaluation of trained Celeb-DF model on DFDC dataset.

## Overview

This folder contains scripts to evaluate your model (trained on Celeb-DF) on the DFDC dataset to test generalization.

## Dataset Info

- **Source**: Kaggle DFDC sample (~400 videos)
- **Location**: `C:\Users\Administrater\Downloads\deepfake-detection-challenge\train_sample_videos`
- **Metadata**: `metadata.json` (included with videos)
- **Purpose**: Cross-dataset validation

## Pipeline Steps

### Step 1: Extract Frames
```bash
python 01_extract_frames_dfdc.py
```
- Extracts every 3rd frame from DFDC videos (matching training setup)
- Output: `H:\EVAL-1\frames\`
- Time: ~15-30 minutes for 400 videos

### Step 2: Detect Faces
```bash
python 02_detect_faces_dfdc.py
```
- GPU-accelerated MTCNN face detection
- Output: `H:\EVAL-1\faces\`
- Time: ~30-45 minutes for 400 videos

### Step 3: Parse Metadata
```bash
python 03_parse_metadata.py
```
- Converts `metadata.json` to test CSV
- Output: `dfdc_test_videos.csv` and `dfdc_test_videos_ready.csv`
- Time: <1 minute

### Step 4: Run Inference
```bash
python 04_run_inference.py
```
- Runs trained model on DFDC videos
- Output: `dfdc_results.csv`
- Time: ~30-60 minutes for 400 videos

## Configuration

All paths are configured at the top of each script:

**Input:**
- Videos: `C:\Users\Administrater\Downloads\deepfake-detection-challenge\train_sample_videos`
- Metadata: Same folder, `metadata.json`

**Output:**
- Frames: `H:\EVAL-1\frames\`
- Faces: `H:\EVAL-1\faces\`
- Results: `j:\DF\evaluation\dfdc\dfdc_results.csv`

**Model:**
- Checkpoint: `j:\DF\checkpoints\best_model.pth`
- Config: `j:\DF\config\defaults.yaml`

## Expected Results

Based on your Celeb-DF performance (100%), expected DFDC results:

- **Best case**: 85-95% (DFDC is harder, more diverse)
- **Good case**: 75-85% (still shows generalization)
- **Acceptable**: 65-75% (cross-dataset is challenging)

Even 70% on DFDC would be strong evidence of generalization!

## Disk Space Requirements

- Frames: ~10-15 GB
- Faces: ~3-5 GB
- Total: ~20 GB on `H:\EVAL-1`

## Resume Capability

All scripts support resume:
- If interrupted, just re-run the same script
- Already processed videos/frames will be skipped
- No need to start from scratch

## Troubleshooting

**Problem**: "No faces detected" for many videos
- Solution: Lower `CONFIDENCE_THRESHOLD` in `02_detect_faces_dfdc.py` (try 0.90 instead of 0.95)

**Problem**: Out of memory during face detection
- Solution: Reduce `BATCH_SIZE` in `02_detect_faces_dfdc.py` (try 16 instead of 32)

**Problem**: Inference errors
- Solution: Check that faces exist in `H:\EVAL-1\faces\` for the video

## Output Files

After running all steps:

1. **dfdc_test_videos.csv** - All videos from metadata
2. **dfdc_test_videos_ready.csv** - Only videos with detected faces
3. **dfdc_results.csv** - Inference results with predictions

## Next Steps

After getting results:

1. Compare with Celeb-DF results
2. Analyze misclassified videos
3. Generate comparison plots
4. Add to paper as cross-dataset validation

## Notes

- This is a COPY of preprocessing scripts (won't affect originals)
- Frame skip = 3 (matches training data)
- Face size = 224x224 (matches training)
- Uses same model checkpoint (no retraining)
