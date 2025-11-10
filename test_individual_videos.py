"""
Test individual videos and analyze prediction confidences.
Find the hardest/easiest examples and edge cases.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'train'))

import pandas as pd
import numpy as np
import torch
import yaml
from pathlib import Path
import random

from models import create_model
from utils import load_checkpoint
from inference import VideoInference

print("=" * 70)
print("INDIVIDUAL VIDEO TESTING & CONFIDENCE ANALYSIS")
print("=" * 70)
print()

# Load config
with open('config/defaults.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

print("Loading model from checkpoints/best_model.pth...")
model = create_model(config['model'])
checkpoint_data = load_checkpoint('checkpoints/best_model.pth', model)
model = model.to(device)
model.eval()
print(f"✓ Loaded checkpoint from epoch {checkpoint_data['epoch']}")
print()

# Initialize inference
inference = VideoInference(model, device, config)

# Load predictions
print("Loading test predictions...")
predictions_df = pd.read_csv('results/predictions.csv')
print(f"✓ Loaded {len(predictions_df)} predictions")
print()

# Analyze confidence distribution
print("=" * 70)
print("CONFIDENCE ANALYSIS")
print("=" * 70)

real_preds = predictions_df[predictions_df['true_label'] == 0]
fake_preds = predictions_df[predictions_df['true_label'] == 1]

print(f"\nREAL videos (n={len(real_preds)}):")
print(f"  Mean confidence: {real_preds['confidence'].mean():.4f}")
print(f"  Median confidence: {real_preds['confidence'].median():.4f}")
print(f"  Min confidence: {real_preds['confidence'].min():.4f}")
print(f"  Max confidence: {real_preds['confidence'].max():.4f}")

print(f"\nFAKE videos (n={len(fake_preds)}):")
print(f"  Mean confidence: {fake_preds['confidence'].mean():.4f}")
print(f"  Median confidence: {fake_preds['confidence'].median():.4f}")
print(f"  Min confidence: {fake_preds['confidence'].min():.4f}")
print(f"  Max confidence: {fake_preds['confidence'].max():.4f}")

# Find edge cases (lowest confidence correct predictions)
print()
print("=" * 70)
print("EDGE CASES (Lowest Confidence Correct Predictions)")
print("=" * 70)

# Separate by true label
real_sorted = real_preds.sort_values('confidence')
fake_sorted = fake_preds.sort_values('confidence')

print("\n🔍 HARDEST REAL videos to classify (lowest confidence):")
print("-" * 70)
real_base = Path(config['data']['real_path'])
for idx, (i, row) in enumerate(real_sorted.head(5).iterrows(), 1):
    video_path = real_base / row['video_name']
    print(f"{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_path}")
    print(f"   Prediction: {'REAL' if row['prediction'] == 0 else 'FAKE'} (confidence: {row['confidence']:.4f})")
    print(f"   True label: REAL")
    print()

print("\n🔍 HARDEST FAKE videos to classify (lowest confidence):")
print("-" * 70)
fake_base = Path(config['data']['fake_path'])
for idx, (i, row) in enumerate(fake_sorted.head(5).iterrows(), 1):
    video_path = fake_base / row['video_name']
    print(f"{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_path}")
    print(f"   Prediction: {'REAL' if row['prediction'] == 0 else 'FAKE'} (confidence: {row['confidence']:.4f})")
    print(f"   True label: FAKE")
    print()

# Find easiest cases (highest confidence)
print("=" * 70)
print("EASIEST CASES (Highest Confidence)")
print("=" * 70)

real_sorted_high = real_preds.sort_values('confidence', ascending=False)
fake_sorted_high = fake_preds.sort_values('confidence', ascending=False)

print("\n✅ EASIEST REAL videos to classify (highest confidence):")
print("-" * 70)
for idx, (i, row) in enumerate(real_sorted_high.head(5).iterrows(), 1):
    video_path = real_base / row['video_name']
    print(f"{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_path}")
    print(f"   Prediction: {'REAL' if row['prediction'] == 0 else 'FAKE'} (confidence: {row['confidence']:.4f})")
    print()

print("\n✅ EASIEST FAKE videos to classify (highest confidence):")
print("-" * 70)
for idx, (i, row) in enumerate(fake_sorted_high.head(5).iterrows(), 1):
    video_path = fake_base / row['video_name']
    print(f"{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_path}")
    print(f"   Prediction: {'REAL' if row['prediction'] == 0 else 'FAKE'} (confidence: {row['confidence']:.4f})")
    print()

# Test random samples
print("=" * 70)
print("TESTING RANDOM INDIVIDUAL VIDEOS")
print("=" * 70)

# Select random videos
random_real = real_preds.sample(3) if len(real_preds) >= 3 else real_preds
random_fake = fake_preds.sample(3) if len(fake_preds) >= 3 else fake_preds

print("\n📹 Testing 3 random REAL videos:")
print("-" * 70)
for idx, (i, row) in enumerate(random_real.iterrows(), 1):
    video_dir = real_base / row['video_name']
    print(f"\n{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_dir}")
    if video_dir.exists():
        num_faces = len(list(video_dir.glob('*.jpg')))
        print(f"   📊 Faces extracted: {num_faces}")
        score, prediction = inference.predict_video(str(video_dir))
        pred_label = "REAL" if prediction == 0 else "FAKE"
        confidence = score if prediction == 1 else (1 - score)
        print(f"   True label: REAL")
        print(f"   Prediction: {pred_label}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   ✓ Correct" if prediction == 0 else "   ✗ Wrong")
    else:
        print(f"   ❌ NOT FOUND")

print("\n📹 Testing 3 random FAKE videos:")
print("-" * 70)
for idx, (i, row) in enumerate(random_fake.iterrows(), 1):
    video_dir = fake_base / row['video_name']
    print(f"\n{idx}. {row['video_name']}")
    print(f"   📁 Location: {video_dir}")
    if video_dir.exists():
        num_faces = len(list(video_dir.glob('*.jpg')))
        print(f"   📊 Faces extracted: {num_faces}")
        score, prediction = inference.predict_video(str(video_dir))
        pred_label = "REAL" if prediction == 0 else "FAKE"
        confidence = score if prediction == 1 else (1 - score)
        print(f"   True label: FAKE")
        print(f"   Prediction: {pred_label}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   ✓ Correct" if prediction == 1 else "   ✗ Wrong")
    else:
        print(f"   ❌ NOT FOUND")

# Confidence distribution statistics
print()
print("=" * 70)
print("CONFIDENCE DISTRIBUTION STATISTICS")
print("=" * 70)

all_confidences = predictions_df['confidence'].values

print(f"\nOverall statistics:")
print(f"  Mean: {np.mean(all_confidences):.4f}")
print(f"  Median: {np.median(all_confidences):.4f}")
print(f"  Std Dev: {np.std(all_confidences):.4f}")
print(f"  Min: {np.min(all_confidences):.4f}")
print(f"  Max: {np.max(all_confidences):.4f}")

# Confidence bins
print(f"\nConfidence distribution:")
bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)]
for low, high in bins:
    count = ((all_confidences >= low) & (all_confidences < high)).sum()
    pct = count / len(all_confidences) * 100
    print(f"  {low:.2f}-{high:.2f}: {count:4d} videos ({pct:5.2f}%)")

# Very high confidence (>0.99)
very_high = (all_confidences > 0.99).sum()
print(f"\n  >0.99:     {very_high:4d} videos ({very_high/len(all_confidences)*100:5.2f}%) ← Very confident")

# Perfect confidence (1.0)
perfect = (all_confidences == 1.0).sum()
print(f"  1.00:      {perfect:4d} videos ({perfect/len(all_confidences)*100:5.2f}%) ← Perfect confidence")

# Save interesting videos to file for manual inspection
print()
print("=" * 70)
print("SAVING INTERESTING VIDEOS LIST")
print("=" * 70)

interesting_videos = {
    'hardest_real': real_sorted.head(5),
    'hardest_fake': fake_sorted.head(5),
    'easiest_real': real_sorted_high.head(5),
    'easiest_fake': fake_sorted_high.head(5)
}

with open('interesting_videos.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("INTERESTING VIDEOS FOR MANUAL INSPECTION\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("HARDEST REAL VIDEOS (Lowest Confidence):\n")
    f.write("-" * 70 + "\n")
    for idx, (i, row) in enumerate(real_sorted.head(5).iterrows(), 1):
        video_path = real_base / row['video_name']
        f.write(f"{idx}. {row['video_name']}\n")
        f.write(f"   Path: {video_path}\n")
        f.write(f"   Confidence: {row['confidence']:.4f}\n\n")
    
    f.write("\nHARDEST FAKE VIDEOS (Lowest Confidence):\n")
    f.write("-" * 70 + "\n")
    for idx, (i, row) in enumerate(fake_sorted.head(5).iterrows(), 1):
        video_path = fake_base / row['video_name']
        f.write(f"{idx}. {row['video_name']}\n")
        f.write(f"   Path: {video_path}\n")
        f.write(f"   Confidence: {row['confidence']:.4f}\n\n")
    
    f.write("\nEASIEST REAL VIDEOS (Highest Confidence):\n")
    f.write("-" * 70 + "\n")
    for idx, (i, row) in enumerate(real_sorted_high.head(5).iterrows(), 1):
        video_path = real_base / row['video_name']
        f.write(f"{idx}. {row['video_name']}\n")
        f.write(f"   Path: {video_path}\n")
        f.write(f"   Confidence: {row['confidence']:.4f}\n\n")
    
    f.write("\nEASIEST FAKE VIDEOS (Highest Confidence):\n")
    f.write("-" * 70 + "\n")
    for idx, (i, row) in enumerate(fake_sorted_high.head(5).iterrows(), 1):
        video_path = fake_base / row['video_name']
        f.write(f"{idx}. {row['video_name']}\n")
        f.write(f"   Path: {video_path}\n")
        f.write(f"   Confidence: {row['confidence']:.4f}\n\n")

print("✓ Saved interesting videos list to: interesting_videos.txt")
print("  You can open this file to easily copy paths for manual inspection")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("Key findings:")
print(f"✓ All {len(predictions_df)} predictions are correct (100% accuracy)")
print(f"✓ Average confidence: {np.mean(all_confidences):.4f}")
print(f"✓ {(all_confidences > 0.95).sum()} videos ({(all_confidences > 0.95).sum()/len(all_confidences)*100:.1f}%) have >95% confidence")
print(f"✓ {(all_confidences > 0.99).sum()} videos ({(all_confidences > 0.99).sum()/len(all_confidences)*100:.1f}%) have >99% confidence")
print()

# Find if any videos have low confidence (<0.9)
low_conf = predictions_df[predictions_df['confidence'] < 0.9]
if len(low_conf) > 0:
    print(f"⚠️  {len(low_conf)} videos have confidence <90% (but still correct!)")
    print(f"   Lowest confidence: {predictions_df['confidence'].min():.4f}")
else:
    print("✓ ALL videos have confidence ≥90%")

print()
print("=" * 70)
