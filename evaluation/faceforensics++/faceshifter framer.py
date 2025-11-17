"""
Frame Extractor for FaceForensics++ Dataset
Extracts every 3rd frame from all FF++ videos (7,000 videos total)

Processes 7 folders:
- original/ (1,000 real videos)
- Deepfakes/ (1,000 fake)
- Face2Face/ (1,000 fake)
- FaceSwap/ (1,000 fake)
- NeuralTextures/ (1,000 fake)
- FaceShifter/ (1,000 fake)
- DeepFakeDetection/ (1,000 fake)
"""
import cv2
import os
from multiprocessing import Pool, cpu_count, set_start_method
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# ===== FACEFORENSICS++ CONFIGURATION =====
SOURCE_BASE_DIR = r"F:\F++ Dataset\FaceForensics++_C23"
OUTPUT_BASE_DIR = r"F:\FF++_preprocessed"

# Folders to process
FOLDERS_TO_PROCESS = ['FaceShifter']

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# CPU Configuration (optimized for HDD writes)
NUM_WORKERS = min(72, cpu_count())
CONCURRENT_VIDEOS = 4  # Process 4 videos simultaneously (HDD-friendly)
PNG_COMPRESSION = 0

# Frame skip (extract every 3rd frame)
FRAME_SKIP = 3

# ============================================================================
# DIRECT HDD WRITING
# ============================================================================

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

def write_frame_direct(frame_data):
    """Write frame directly to disk"""
    frame, output_path = frame_data
    try:
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        return True
    except:
        return False

def process_single_video(video_info):
    """Process a single FF++ video - direct HDD write"""
    video_path, output_dir, folder_name = video_info
    
    video_name_without_ext = video_path.stem
    output_subfolder = output_dir / folder_name / video_name_without_ext
    output_subfolder.mkdir(exist_ok=True, parents=True)
    
    # Check resume
    start_frame, existing_frame_nums = get_resume_info(output_subfolder)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"❌ [{folder_name}] Error: Could not open {video_path.name}", 0
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame >= total_video_frames - 1 and total_video_frames > 0:
        cap.release()
        return f"✅ [{folder_name}] {video_path.name} already complete ({total_video_frames} total frames)", 0
    
    # Load frames into RAM for this video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    current_frame_idx = 0
    extracted_frame_idx = 0
    frames_to_write = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only extract every FRAME_SKIP frames
        if current_frame_idx % FRAME_SKIP == 0:
            if extracted_frame_idx not in existing_frame_nums:
                output_path = output_subfolder / f"frame_{extracted_frame_idx:05d}.png"
                frames_to_write.append((frame, output_path))
            extracted_frame_idx += 1
        
        current_frame_idx += 1
    
    cap.release()
    
    if not frames_to_write:
        return f"✅ [{folder_name}] {video_path.name} - no new frames", 0
    
    # Write directly to HDD (32 threads)
    frames_written = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(write_frame_direct, frames_to_write))
        frames_written = sum(1 for r in results if r)
    
    return f"✅ [{folder_name}] {video_path.name} - {frames_written} frames written ({total_video_frames} total)", frames_written

def main():
    """Main extraction logic for FaceForensics++."""
    print("="*70)
    print("🎯 FACEFORENSICS++ FRAME EXTRACTOR")
    print("="*70)
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Concurrent Videos: {CONCURRENT_VIDEOS}")
    print(f"   Frame Skip: Every {FRAME_SKIP} frames")
    print(f"   Source: {SOURCE_BASE_DIR}")
    print(f"   Output: {OUTPUT_BASE_DIR}\\frames\\")
    print(f"   Folders: {len(FOLDERS_TO_PROCESS)}")
    print("="*70)
    
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    source_base = Path(SOURCE_BASE_DIR)
    output_base = Path(OUTPUT_BASE_DIR) / "frames"
    
    if not source_base.exists():
        print(f"❌ Error: Source directory not found: {source_base}")
        return
    
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Collect all videos from all folders
    all_video_tasks = []
    folder_stats = {}
    
    for folder in FOLDERS_TO_PROCESS:
        folder_path = source_base / folder
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        # Find videos recursively
        video_files = list(folder_path.rglob('*.mp4'))
        folder_stats[folder] = len(video_files)
        
        for video_path in video_files:
            all_video_tasks.append((video_path, output_base, folder))
        
        print(f"   {folder}: {len(video_files)} videos")
    
    total_videos = len(all_video_tasks)
    if total_videos == 0:
        print("❌ No videos found to process.")
        return
    
    print(f"\n📊 Total videos to process: {total_videos}")
    print(f"💪 Using 4 video workers × 32 write threads = 128 parallel HDD writes\n")
    
    total_frames = 0
    with Pool(processes=CONCURRENT_VIDEOS, maxtasksperchild=1) as pool:
        with tqdm(total=total_videos, desc="Processing Videos", unit="video") as pbar:
            for result_msg, frames in pool.imap_unordered(process_single_video, all_video_tasks, chunksize=1):
                total_frames += frames
                tqdm.write(result_msg)
                pbar.update(1)
    
    print(f"\n🎉 Extraction complete!")
    print(f"📁 Total frames written: {total_frames:,}")
    print(f"📁 Frames saved to: {output_base}")
    print("\n📊 Folder breakdown:")
    for folder, count in folder_stats.items():
        print(f"   {folder}: {count} videos")

if __name__ == "__main__":
    main()
