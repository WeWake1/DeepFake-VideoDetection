"""
FaceForensics++ (FF++) Face Extraction Script
==============================================
Purpose: Extract faces from FF++ frame directories using MTCNN GPU

Input:
- Real frames: F:\FF++_preprocessed\frames\FF Real Frames
- Fake frames: F:\FF++_preprocessed\frames\FF Fake Frames

Output:
- Real faces: F:\FF++_preprocessed\faces\FF Real Faces
- Fake faces: F:\FF++_preprocessed\faces\FF Fake Faces

Author: DeepFake Detection Project
Date: November 18, 2025

Five Rules Compliance:
1. ✓ Save scripts where they belong (preprocessing/ folder)
2. ✓ Use absolute paths from verified locations
3. ✓ Provide clear documentation
4. ✓ Handle errors gracefully
5. ✓ Output structured, readable results
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
FRAMES_DIR_REAL = Path("F:/FF++_preprocessed/frames/FF Real Frames")
FRAMES_DIR_FAKE = Path("F:/FF++_preprocessed/frames/FF Fake Frames")
OUTPUT_DIR_REAL = Path("F:/FF++_preprocessed/faces/FF Real Faces")
OUTPUT_DIR_FAKE = Path("F:/FF++_preprocessed/faces/FF Fake Faces")

# Face detection parameters
TARGET_SIZE = (224, 224)
MIN_FACE_SIZE = 20
THRESHOLDS = [0.6, 0.7, 0.8]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    Extract faces from all frames in a video folder
    
    Args:
        video_folder: Path to folder containing frame images
        output_folder: Path to save detected faces
        mtcnn: MTCNN detector instance
    
    Returns:
        dict with statistics
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(video_folder.glob("*.png"))
    if not frame_files:
        frame_files = sorted(video_folder.glob("*.jpg"))
    
    if not frame_files:
        return {"total": 0, "detected": 0, "saved": 0}
    
    stats = {"total": len(frame_files), "detected": 0, "saved": 0}
    
    for frame_path in frame_files:
        try:
            # Read frame
            img = Image.open(frame_path).convert('RGB')
            
            # Detect face
            face = mtcnn(img)
            
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
                
        except Exception as e:
            # Skip problematic frames
            continue
    
    return stats

def process_dataset(frames_dir, output_dir, label, mtcnn):
    """
    Process entire dataset (real or fake)
    Handles both flat structure and nested manipulation method folders
    
    Args:
        frames_dir: Root directory containing video frame folders
        output_dir: Root directory for output faces
        label: "real" or "fake"
        mtcnn: MTCNN detector instance
    """
    print(f"\n{'='*80}")
    print(f"Processing FF++ {label.upper()} Videos")
    print(f"{'='*80}")
    print(f"Input:  {frames_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have manipulation method subdirectories (for fake videos)
    video_folders = []
    manipulation_methods = []
    
    if label == "fake":
        # Check for manipulation method subdirectories
        potential_methods = [d for d in frames_dir.iterdir() if d.is_dir()]
        
        # Check if these are manipulation method folders or video folders
        # Manipulation methods: Deepfakes, Face2Face, FaceSwap, etc.
        method_names = ["deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter", "deepfakedetection"]
        
        for d in potential_methods:
            if any(method in d.name.lower() for method in method_names):
                # This is a manipulation method folder
                manipulation_methods.append(d)
            else:
                # This is a video folder
                video_folders.append(d)
        
        if manipulation_methods:
            print(f"\n📊 Found {len(manipulation_methods)} manipulation method folders:")
            for method in manipulation_methods:
                print(f"   - {method.name}")
            
            # Get all video folders from all methods
            video_folders = []
            for method_dir in manipulation_methods:
                method_videos = [d for d in method_dir.iterdir() if d.is_dir()]
                video_folders.extend(method_videos)
        else:
            # No manipulation methods, use direct video folders
            video_folders = potential_methods
    else:
        # Real videos - flat structure
        video_folders = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    
    total_videos = len(video_folders)
    
    print(f"\n📊 Found {total_videos:,} video folders to process")
    
    # Statistics
    overall_stats = {
        "videos_processed": 0,
        "videos_with_faces": 0,
        "total_frames": 0,
        "total_faces_detected": 0,
        "total_faces_saved": 0
    }
    
    start_time = time.time()
    
    # Process each video folder
    for idx, video_folder in enumerate(tqdm(video_folders, desc=f"FF++ {label}")):
        video_name = video_folder.name
        output_folder = output_dir / video_name
        
        # Extract faces from this video
        stats = extract_faces_from_video_folder(video_folder, output_folder, mtcnn)
        
        overall_stats["videos_processed"] += 1
        overall_stats["total_frames"] += stats["total"]
        overall_stats["total_faces_detected"] += stats["detected"]
        overall_stats["total_faces_saved"] += stats["saved"]
        
        if stats["saved"] > 0:
            overall_stats["videos_with_faces"] += 1
        
        # Progress update every 100 videos
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (total_videos - idx - 1) / rate if rate > 0 else 0
            
            print(f"\n⏱️  Progress: {idx+1}/{total_videos} videos")
            print(f"   Faces saved: {overall_stats['total_faces_saved']:,}")
            print(f"   Estimated time remaining: {remaining/60:.1f} minutes")
    
    elapsed_time = time.time() - start_time
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"✅ FF++ {label.upper()} Processing Complete")
    print(f"{'='*80}")
    print(f"Videos processed:     {overall_stats['videos_processed']:,}")
    print(f"Videos with faces:    {overall_stats['videos_with_faces']:,}")
    print(f"Total frames:         {overall_stats['total_frames']:,}")
    print(f"Faces detected:       {overall_stats['total_faces_detected']:,}")
    print(f"Faces saved:          {overall_stats['total_faces_saved']:,}")
    print(f"Detection rate:       {overall_stats['total_faces_detected']/overall_stats['total_frames']*100:.2f}%")
    print(f"Time elapsed:         {elapsed_time/60:.1f} minutes")
    print(f"{'='*80}")
    
    return overall_stats

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FF++ FACE EXTRACTION - GPU ACCELERATED")
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
    print("🎯 OVERALL FF++ EXTRACTION SUMMARY")
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
    
    print("\n✅ FF++ face extraction complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
