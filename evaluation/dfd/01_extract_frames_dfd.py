"""
Frame Extractor for DFD (Google/Jigsaw) Dataset
Extracts frames from DFD videos (both original and manipulated)

DFD Structure:
- original_videos/ (363 videos)
- manipulated_videos/ (3,068 videos)
"""
import os
import cv2
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time

# ===== DFD CONFIGURATION =====
SOURCE_BASE_DIR = r"H:\DFD Dataset"
OUTPUT_BASE_DIR = r"F:\DFD_preprocessed"  # HDD with 2,536 videos already here

# Folders to process (nested structure: DFD_manipulated_sequences/DFD_manipulated_sequences)
FOLDERS_TO_PROCESS = [
    ('DFD_manipulated_sequences', 'DFD_manipulated_sequences'),  # (parent_folder, subfolder) - 3,068 videos
    ('DFD_original sequences', None)  # (folder, None) - 363 videos
]

# Frame extraction settings
FRAME_SKIP = 3  # Extract every 3rd frame (matching training data)
OUTPUT_FORMAT = "png"

# Processing settings (optimized for 72-core Xeon + HDD writes)
CONCURRENT_VIDEOS = 12  # Increased for better CPU utilization
FRAME_BUFFER_SIZE = 100

# Resume capability
RESUME = True

# ============================================================================
# FRAME EXTRACTION LOGIC
# ============================================================================

def write_frame_to_disk(frame, output_path):
    """Write a single frame to disk"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)

def get_resume_info(output_dir, frame_skip):
    """Check if video already processed and return resume frame number"""
    if not output_dir.exists():
        return 0, 0
    
    existing_frames = list(output_dir.glob(f'*.{OUTPUT_FORMAT}'))
    if not existing_frames:
        return 0, 0
    
    # Get highest frame number
    frame_nums = []
    for f in existing_frames:
        try:
            num = int(f.stem.split('_')[1])
            frame_nums.append(num)
        except:
            continue
    
    if frame_nums:
        last_frame = max(frame_nums)
        next_frame = last_frame + frame_skip
        return len(frame_nums), next_frame
    
    return 0, 0

def process_single_video(video_path, output_base, folder_name):
    """Process a single video file"""
    video_name = video_path.stem
    # Output: H:\DFD\frames\<folder_name>\<video_name>\
    output_dir = output_base / 'frames' / folder_name / video_name
    
    # Check if already processed
    if RESUME:
        existing_count, resume_frame = get_resume_info(output_dir, FRAME_SKIP)
        if existing_count > 10:  # Skip if has reasonable number of frames
            return f"⏭️  [{folder_name}] {video_name} - already processed ({existing_count} frames)"
    else:
        resume_frame = 0
        existing_count = 0
    
    # Open video
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"❌ [{folder_name}] {video_name} - failed to open"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Skip to resume position
        if resume_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
        
        frame_idx = resume_frame
        extracted = existing_count
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every FRAME_SKIP frames
            if frame_idx % FRAME_SKIP == 0:
                output_path = output_dir / f"frame_{frame_idx:06d}.{OUTPUT_FORMAT}"
                write_frame_to_disk(frame, output_path)
                extracted += 1
            
            frame_idx += 1
        
        cap.release()
        
        return f"✅ [{folder_name}] {video_name} - {extracted} frames extracted ({total_frames} total)"
    
    except Exception as e:
        return f"❌ [{folder_name}] {video_name} - error: {str(e)}"

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main frame extraction logic for DFD"""
    print("="*70)
    print("🎬 DFD FRAME EXTRACTOR")
    print("="*70)
    print(f"   Source: {SOURCE_BASE_DIR}")
    print(f"   Output: {OUTPUT_BASE_DIR}")
    print(f"   Frame Skip: Every {FRAME_SKIP} frames")
    print(f"   Concurrent: {CONCURRENT_VIDEOS} videos")
    print(f"   Folders: {FOLDERS_TO_PROCESS}")
    print("="*70)
    
    source_base = Path(SOURCE_BASE_DIR)
    output_base = Path(OUTPUT_BASE_DIR)
    
    if not source_base.exists():
        print(f"❌ Error: Source directory not found: {source_base}")
        return
    
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Collect all video files
    all_videos = []
    folder_stats = {}
    
    for folder_info in FOLDERS_TO_PROCESS:
        if isinstance(folder_info, tuple):
            parent_folder, subfolder = folder_info
            if subfolder:
                folder_path = source_base / parent_folder / subfolder
                display_name = f"{parent_folder}/{subfolder}"
            else:
                folder_path = source_base / parent_folder
                display_name = parent_folder
        else:
            # Legacy support for simple string folder names
            folder_path = source_base / folder_info
            display_name = folder_info
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {display_name}")
            continue
        
        # Get all .mp4 files (including in nested folders)
        video_files = list(folder_path.rglob('*.mp4'))
        folder_stats[display_name] = len(video_files)
        
        for video_file in video_files:
            all_videos.append((video_file, display_name))
        
        print(f"   {display_name}: {len(video_files)} videos")
    
    if not all_videos:
        print("❌ No videos found.")
        return
    
    print(f"\n📊 Total videos to process: {len(all_videos)}\n")
    
    # Process videos with ThreadPoolExecutor
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_VIDEOS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_video, video_path, output_base, folder_name): (video_path, folder_name)
            for video_path, folder_name in all_videos
        }
        
        # Progress bar
        with tqdm(total=len(futures), desc="Extracting Frames", unit="video") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                tqdm.write(result)
                pbar.update(1)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 EXTRACTION SUMMARY")
    print("="*70)
    print(f"   Total videos processed: {len(all_videos)}")
    print(f"   Time elapsed: {elapsed/60:.1f} minutes")
    print(f"   Avg time per video: {elapsed/len(all_videos):.1f} seconds")
    print("="*70)
    print(f"📁 Frames saved to: {OUTPUT_BASE_DIR}\\frames\\")
    print("\n📊 Folder breakdown:")
    for folder, count in folder_stats.items():
        print(f"   {folder}: {count} videos")

if __name__ == "__main__":
    main()
