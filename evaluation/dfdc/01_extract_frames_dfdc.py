"""
Frame Extractor for DFDC Dataset - Evaluation Version
Adapted from preprocessing/framer_cpu(final)

Extracts every 3rd frame from DFDC videos for evaluation.
"""
import cv2
import os
from multiprocessing import Pool, cpu_count, set_start_method
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import gc

# ===== DFDC CONFIGURATION =====
SOURCE_VIDEO_DIR = r"C:\Users\Administrater\Downloads\deepfake-detection-challenge\train_sample_videos"
OUTPUT_DIR = r"H:\EVAL-1\frames"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# CPU Configuration
NUM_WORKERS = min(32, cpu_count())  # Use 32 workers for extraction
CONCURRENT_VIDEOS = 8
PNG_COMPRESSION = 0  # No compression for speed

# Frame skip (extract every 3rd frame like training data)
FRAME_SKIP = 3

def write_frame_to_disk(frame_data):
    """Write a single frame to disk."""
    frame, filename = frame_data
    try:
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        return True
    except Exception:
        return False

def get_resume_info(output_subfolder):
    """Check for existing frames to enable resume."""
    existing_frame_numbers = set()
    
    if output_subfolder.exists():
        existing_frames = list(output_subfolder.glob('*.png'))
        for f in existing_frames:
            try:
                num = int(f.stem.split('_')[1])
                existing_frame_numbers.add(num)
            except (ValueError, IndexError):
                continue
    
    if not existing_frame_numbers:
        return 0, set()
    
    return max(existing_frame_numbers) + 1, existing_frame_numbers

def process_single_video(video_info):
    """Process a single DFDC video with frame skip."""
    video_path, output_dir = video_info
    
    video_name_without_ext = video_path.stem
    output_subfolder = output_dir / video_name_without_ext
    output_subfolder.mkdir(exist_ok=True, parents=True)
    
    # Check resume
    start_frame, existing_frame_nums = get_resume_info(output_subfolder)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"❌ Error: Could not open {video_path.name}", 0
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame >= total_video_frames - 1 and total_video_frames > 0:
        cap.release()
        return f"✅ {video_path.name} already complete", 0
    
    # Load frames into RAM (with frame skip)
    frames_in_ram = []
    work_items = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    current_frame_idx = 0
    extracted_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only extract every FRAME_SKIP frames
        if current_frame_idx % FRAME_SKIP == 0:
            if extracted_frame_idx not in existing_frame_nums:
                frames_in_ram.append(frame.copy())
                output_path = output_subfolder / f"frame_{extracted_frame_idx:05d}.png"
                work_items.append((len(frames_in_ram) - 1, output_path))
            extracted_frame_idx += 1
        
        current_frame_idx += 1
    
    cap.release()
    
    if not work_items:
        return f"✅ {video_path.name} - no new frames", 0
    
    # Write frames in parallel
    frames_written = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        all_write_tasks = [(frames_in_ram[idx], path) for idx, path in work_items]
        results = list(executor.map(write_frame_to_disk, all_write_tasks))
        frames_written = sum(1 for r in results if r)
    
    # Cleanup
    del frames_in_ram
    del work_items
    gc.collect()
    
    return f"✅ {video_path.name} - {frames_written} frames", frames_written

def main():
    """Main extraction logic for DFDC."""
    print("="*70)
    print("🎯 DFDC FRAME EXTRACTOR (EVALUATION)")
    print("="*70)
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Concurrent Videos: {CONCURRENT_VIDEOS}")
    print(f"   Frame Skip: Every {FRAME_SKIP} frames")
    print(f"   Source: {SOURCE_VIDEO_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print("="*70)
    
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    source_dir = Path(SOURCE_VIDEO_DIR)
    output_dir = Path(OUTPUT_DIR)
    
    if not source_dir.exists():
        print(f"❌ Error: Source directory not found: {source_dir}")
        return
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    video_files = [f for f in source_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not video_files:
        print("❌ No videos found to process.")
        return
    
    print(f"Found {len(video_files)} videos to process.\n")
    
    video_tasks = [(video_path, output_dir) for video_path in video_files]
    
    total_frames = 0
    with Pool(processes=CONCURRENT_VIDEOS, maxtasksperchild=1) as pool:
        with tqdm(total=len(video_tasks), desc="Processing Videos", unit="video") as pbar:
            for result_msg, frames in pool.imap_unordered(process_single_video, video_tasks, chunksize=1):
                total_frames += frames
                tqdm.write(result_msg)
                pbar.update(1)
    
    print(f"\n🎉 Extraction complete! Total frames: {total_frames:,}")
    print(f"📁 Frames saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
