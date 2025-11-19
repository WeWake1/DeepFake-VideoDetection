"""
DFD Face Extraction Script (OPTIMIZED)
=======================================
Purpose: Extract faces from DFD frame directories using MTCNN GPU with BATCH PROCESSING

Input:
- Real frames: F:/DFD_preprocessed/frames/DFD real frames
- Fake frames: F:/DFD_preprocessed/frames/DFD fake frames

Output:
- Real faces: F:/DFD_preprocessed/faces/DFD real faces
- Fake faces: F:/DFD_preprocessed/faces/DFD fake faces

Author: DeepFake Detection Project
Date: November 18, 2025 (Optimized: November 19, 2025)

Optimizations:
- Batch processing (process 16 frames at once)
- Skip videos that already have faces extracted
- Better error handling
- Faster file I/O
"""

import os
import cv2
import torch
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import time
from tqdm import tqdm

# Paths from verified locations (November 18, 2025)
FRAMES_DIR_REAL = Path("F:/DFD_preprocessed/frames/DFD real frames")
FRAMES_DIR_FAKE = Path("F:/DFD_preprocessed/frames/DFD fake frames")
OUTPUT_DIR_REAL = Path("F:/DFD_preprocessed/faces/DFD real faces")
OUTPUT_DIR_FAKE = Path("F:/DFD_preprocessed/faces/DFD fake faces")

# Face detection parameters
TARGET_SIZE = (224, 224)
MIN_FACE_SIZE = 20
THRESHOLDS = [0.6, 0.7, 0.8]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16  # Process 16 frames at once for speed

def initialize_mtcnn():
    """Initialize MTCNN face detector on GPU"""
    print(f"\n🔧 Initializing MTCNN on device: {DEVICE}")
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        min_face_size=MIN_FACE_SIZE,
        thresholds=THRESHOLDS,
        factor=0.709,
        post_process=True,
        device=DEVICE,
        keep_all=False  # Keep only the largest face
    )
    print("✅ MTCNN initialized successfully")
    return mtcnn

def extract_faces_from_video_folder(video_folder, output_folder, mtcnn):
    """
    Extract faces from all frames in a video folder using BATCH PROCESSING
    
    Args:
        video_folder: Path to folder containing frame images
        output_folder: Path to save detected faces
        mtcnn: MTCNN detector instance
    
    Returns:
        dict with statistics
    """
    # Skip if already processed
    if output_folder.exists():
        existing_faces = list(output_folder.glob("*.jpg"))
        if len(existing_faces) > 0:
            # Already processed, skip
            return {"total": 0, "detected": 0, "saved": 0, "skipped": True}
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(video_folder.glob("*.png"))
    if not frame_files:
        frame_files = sorted(video_folder.glob("*.jpg"))
    
    if not frame_files:
        return {"total": 0, "detected": 0, "saved": 0, "skipped": False}
    
    stats = {"total": len(frame_files), "detected": 0, "saved": 0, "skipped": False}
    
    # Process in batches
    for i in range(0, len(frame_files), BATCH_SIZE):
        batch_files = frame_files[i:i + BATCH_SIZE]
        batch_images = []
        batch_paths = []
        
        # Load batch of images
        for frame_path in batch_files:
            try:
                img = Image.open(frame_path).convert('RGB')
                batch_images.append(img)
                batch_paths.append(frame_path)
            except Exception:
                continue
        
        if not batch_images:
            continue
        
        # Detect faces in batch
        try:
            faces = mtcnn(batch_images)
            
            # Save detected faces
            for idx, (face, frame_path) in enumerate(zip(faces, batch_paths)):
                if face is not None:
                    stats["detected"] += 1
                    
                    # Convert tensor to numpy and save
                    face_np = face.permute(1, 2, 0).cpu().numpy()
                    face_np = (face_np * 128 + 127.5).clip(0, 255).astype('uint8')
                    face_img = Image.fromarray(face_np)
                    
                    # Save face
                    output_path = output_folder / frame_path.name
                    face_img.save(output_path, quality=95)
                    stats["saved"] += 1
        except Exception:
            # Skip problematic batch
            continue
    
    return stats

def process_dataset(frames_dir, output_dir, label, mtcnn):
    """
    Process entire dataset (real or fake)
    
    Args:
        frames_dir: Root directory containing video frame folders
        output_dir: Root directory for output faces
        label: "real" or "fake"
        mtcnn: MTCNN detector instance
    """
    print(f"\n{'='*80}")
    print(f"Processing DFD {label.upper()} Videos")
    print(f"{'='*80}")
    print(f"Input:  {frames_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video folders
    video_folders = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    total_videos = len(video_folders)
    
    print(f"\n📊 Found {total_videos:,} video folders to process")
    
    # Statistics
    overall_stats = {
        "videos_processed": 0,
        "videos_skipped": 0,
        "videos_with_faces": 0,
        "total_frames": 0,
        "total_faces_detected": 0,
        "total_faces_saved": 0
    }
    
    start_time = time.time()
    
    # Process each video folder
    for idx, video_folder in enumerate(tqdm(video_folders, desc=f"DFD {label}")):
        video_name = video_folder.name
        output_folder = output_dir / video_name
        
        # Extract faces from this video
        stats = extract_faces_from_video_folder(video_folder, output_folder, mtcnn)
        
        if stats["skipped"]:
            overall_stats["videos_skipped"] += 1
            continue
        
        overall_stats["videos_processed"] += 1
        overall_stats["total_frames"] += stats["total"]
        overall_stats["total_faces_detected"] += stats["detected"]
        overall_stats["total_faces_saved"] += stats["saved"]
        
        if stats["saved"] > 0:
            overall_stats["videos_with_faces"] += 1
        
        # Progress update every 50 videos
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            processed = idx + 1 - overall_stats["videos_skipped"]
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_videos = total_videos - idx - 1
            remaining = remaining_videos / rate if rate > 0 else 0
            
            print(f"\n⏱️  Progress: {idx+1}/{total_videos} videos ({overall_stats['videos_skipped']} skipped)")
            print(f"   Faces saved: {overall_stats['total_faces_saved']:,}")
            print(f"   Processing rate: {rate:.2f} videos/sec")
            print(f"   Estimated time remaining: {remaining/60:.1f} minutes")
    
    elapsed_time = time.time() - start_time
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"✅ DFD {label.upper()} Processing Complete")
    print(f"{'='*80}")
    print(f"Videos processed:     {overall_stats['videos_processed']:,}")
    print(f"Videos skipped:       {overall_stats['videos_skipped']:,} (already extracted)")
    print(f"Videos with faces:    {overall_stats['videos_with_faces']:,}")
    print(f"Total frames:         {overall_stats['total_frames']:,}")
    print(f"Faces detected:       {overall_stats['total_faces_detected']:,}")
    print(f"Faces saved:          {overall_stats['total_faces_saved']:,}")
    if overall_stats['total_frames'] > 0:
        print(f"Detection rate:       {overall_stats['total_faces_detected']/overall_stats['total_frames']*100:.2f}%")
    print(f"Time elapsed:         {elapsed_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    return overall_stats

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("DFD FACE EXTRACTION - GPU ACCELERATED")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Target face size: {TARGET_SIZE}")
    print("="*80)
    
    # Verify input directories exist
    if not FRAMES_DIR_REAL.exists():
        print(f"❌ ERROR: Real frames directory not found: {FRAMES_DIR_REAL}")
        return
    
    if not FRAMES_DIR_FAKE.exists():
        print(f"❌ ERROR: Fake frames directory not found: {FRAMES_DIR_FAKE}")
        return
    
    # Initialize MTCNN
    mtcnn = initialize_mtcnn()
    
    # Process real videos
    real_stats = process_dataset(FRAMES_DIR_REAL, OUTPUT_DIR_REAL, "real", mtcnn)
    
    # Process fake videos
    fake_stats = process_dataset(FRAMES_DIR_FAKE, OUTPUT_DIR_FAKE, "fake", mtcnn)
    
    # Overall summary
    print("\n" + "="*80)
    print("🎯 OVERALL DFD EXTRACTION SUMMARY")
    print("="*80)
    print(f"\nReal Videos:")
    print(f"  Processed: {real_stats['videos_with_faces']:,} / {real_stats['videos_processed']:,}")
    print(f"  Faces saved: {real_stats['total_faces_saved']:,}")
    
    print(f"\nFake Videos:")
    print(f"  Processed: {fake_stats['videos_with_faces']:,} / {fake_stats['videos_processed']:,}")
    print(f"  Faces saved: {fake_stats['total_faces_saved']:,}")
    
    print(f"\nTotal:")
    print(f"  Videos: {real_stats['videos_processed'] + fake_stats['videos_processed']:,}")
    print(f"  Faces: {real_stats['total_faces_saved'] + fake_stats['total_faces_saved']:,}")
    
    print("\n✅ DFD face extraction complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
