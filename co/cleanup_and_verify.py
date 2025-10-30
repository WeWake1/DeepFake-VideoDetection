"""
================================================================================
    FRAME FOLDERS CLEANUP AND VERIFICATION SCRIPT
================================================================================

This script will:
1. Scan all 3 fake frame directories for empty/incomplete folders
2. Delete folders with 0 frames or <10 frames (incomplete)
3. Verify all videos have their frames extracted
4. Create a master mapping file showing where each video's frames are located
5. Generate a list of videos that need to be processed

"""
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ==================== CONFIGURATION ====================

# Real video frames (single directory)
REAL_FRAMES_DIR = r"H:\Celeb-real FRAMES"

# Fake video frames (3 directories)
FAKE_FRAMES_DIRS = [
    r"J:\DF\FR\Celeb-synthesis FRAMES",
    r"I:\Caleb S Rem FRAMES 2",
    r"H:\Caleb S FRAMES 3"
]

# Original video directories
REAL_VIDEOS_DIR = r"J:\DF\DS\Celeb-real"
FAKE_VIDEOS_DIR = r"H:\Celeb-synthesis"

# Output files
OUTPUT_MAPPING_CSV = r"J:\DF\frame_mapping.csv"
OUTPUT_MAPPING_JSON = r"J:\DF\frame_mapping.json"
OUTPUT_MISSING_VIDEOS = r"J:\DF\videos_to_process.txt"
OUTPUT_REPORT = r"J:\DF\cleanup_report.txt"

# Thresholds
MIN_FRAMES_THRESHOLD = 20  # Folders with fewer frames are considered incomplete

# ==================== FUNCTIONS ====================

def count_frames_in_folder(folder_path):
    """Count PNG files in a folder."""
    try:
        return len(list(folder_path.glob('*.png')))
    except Exception as e:
        return 0

def scan_frame_directory(frame_dir):
    """
    Scan a frame directory and categorize folders.
    Returns: (valid_folders, empty_folders, incomplete_folders)
    """
    frame_dir = Path(frame_dir)
    
    if not frame_dir.exists():
        print(f"⚠️  Directory not found: {frame_dir}")
        return {}, [], []
    
    valid_folders = {}  # {folder_name: (path, frame_count)}
    empty_folders = []
    incomplete_folders = []
    
    print(f"\n📁 Scanning: {frame_dir}")
    
    subfolders = [f for f in frame_dir.iterdir() if f.is_dir()]
    
    for folder in tqdm(subfolders, desc="  Scanning folders", unit="folder"):
        frame_count = count_frames_in_folder(folder)
        
        if frame_count == 0:
            empty_folders.append(folder)
        elif frame_count < MIN_FRAMES_THRESHOLD:
            incomplete_folders.append((folder, frame_count))
        else:
            valid_folders[folder.name] = (str(folder), frame_count)
    
    return valid_folders, empty_folders, incomplete_folders

def delete_folders(folders, folder_type="empty"):
    """Delete a list of folders."""
    if not folders:
        print(f"✅ No {folder_type} folders to delete.")
        return 0
    
    print(f"\n🗑️  Found {len(folders)} {folder_type} folders:")
    for folder in folders[:5]:  # Show first 5
        if isinstance(folder, tuple):
            print(f"   - {folder[0].name} ({folder[1]} frames)")
        else:
            print(f"   - {folder.name}")
    
    if len(folders) > 5:
        print(f"   ... and {len(folders) - 5} more")
    
    response = input(f"\n⚠️  Delete all {len(folders)} {folder_type} folders? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print(f"❌ Skipping deletion of {folder_type} folders.")
        return 0
    
    deleted = 0
    for item in tqdm(folders, desc=f"  Deleting {folder_type} folders", unit="folder"):
        try:
            folder_path = item[0] if isinstance(item, tuple) else item
            
            # Delete all files in folder first
            for file in folder_path.iterdir():
                file.unlink()
            
            # Delete the folder
            folder_path.rmdir()
            deleted += 1
        except Exception as e:
            print(f"   ⚠️  Could not delete {folder_path.name}: {e}")
    
    print(f"✅ Deleted {deleted} {folder_type} folders.")
    return deleted

def get_videos_from_directory(video_dir):
    """Get list of video files from directory."""
    video_dir = Path(video_dir)
    
    if not video_dir.exists():
        print(f"⚠️  Video directory not found: {video_dir}")
        return []
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f.stem for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]
    
    return videos

def verify_and_create_mapping():
    """
    Verify all videos have frames and create mapping.
    """
    print("\n" + "="*60)
    print("📋 VERIFICATION AND MAPPING")
    print("="*60)
    
    # Get all original videos
    print("\n🎬 Scanning original video directories...")
    real_videos = get_videos_from_directory(REAL_VIDEOS_DIR)
    fake_videos = get_videos_from_directory(FAKE_VIDEOS_DIR)
    
    print(f"   Found {len(real_videos)} real videos")
    print(f"   Found {len(fake_videos)} fake videos")
    
    # Scan all frame directories
    print("\n🔍 Scanning frame directories...")
    
    # Real frames
    real_frames_path = Path(REAL_FRAMES_DIR)
    real_frame_folders = {}
    if real_frames_path.exists():
        valid, _, _ = scan_frame_directory(real_frames_path)
        real_frame_folders = valid
    
    # Fake frames (combine from all 3 drives)
    fake_frame_folders = {}
    for fake_dir in FAKE_FRAMES_DIRS:
        valid, _, _ = scan_frame_directory(fake_dir)
        fake_frame_folders.update(valid)
    
    # Create mapping and find missing
    mapping = []
    missing_videos = []
    
    # Check real videos
    print("\n✅ Verifying real videos...")
    for video_name in tqdm(real_videos, desc="  Checking real videos", unit="video"):
        if video_name in real_frame_folders:
            path, count = real_frame_folders[video_name]
            mapping.append({
                'video_name': video_name,
                'type': 'real',
                'frame_folder_path': path,
                'num_frames': count,
                'status': 'complete',
                'drive': Path(path).drive
            })
        else:
            mapping.append({
                'video_name': video_name,
                'type': 'real',
                'frame_folder_path': 'MISSING',
                'num_frames': 0,
                'status': 'missing',
                'drive': 'N/A'
            })
            missing_videos.append(f"{video_name}.mp4 (real)")
    
    # Check fake videos
    print("\n✅ Verifying fake videos...")
    for video_name in tqdm(fake_videos, desc="  Checking fake videos", unit="video"):
        if video_name in fake_frame_folders:
            path, count = fake_frame_folders[video_name]
            mapping.append({
                'video_name': video_name,
                'type': 'fake',
                'frame_folder_path': path,
                'num_frames': count,
                'status': 'complete',
                'drive': Path(path).drive
            })
        else:
            mapping.append({
                'video_name': video_name,
                'type': 'fake',
                'frame_folder_path': 'MISSING',
                'num_frames': 0,
                'status': 'missing',
                'drive': 'N/A'
            })
            missing_videos.append(f"{video_name}.mp4 (fake)")
    
    return mapping, missing_videos

def save_mapping(mapping):
    """Save mapping to CSV and JSON files."""
    print("\n💾 Saving mapping files...")
    
    # Save CSV
    with open(OUTPUT_MAPPING_CSV, 'w', newline='', encoding='utf-8') as f:
        if mapping:
            writer = csv.DictWriter(f, fieldnames=mapping[0].keys())
            writer.writeheader()
            writer.writerows(mapping)
    
    print(f"   ✅ CSV saved: {OUTPUT_MAPPING_CSV}")
    
    # Save JSON
    with open(OUTPUT_MAPPING_JSON, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"   ✅ JSON saved: {OUTPUT_MAPPING_JSON}")

def save_missing_videos(missing_videos):
    """Save list of missing videos."""
    if not missing_videos:
        print("\n🎉 All videos have frames! No missing videos.")
        return
    
    with open(OUTPUT_MISSING_VIDEOS, 'w', encoding='utf-8') as f:
        f.write("# Videos that need frame extraction\n")
        f.write(f"# Total: {len(missing_videos)}\n\n")
        for video in missing_videos:
            f.write(f"{video}\n")
    
    print(f"\n📝 Missing videos list saved: {OUTPUT_MISSING_VIDEOS}")
    print(f"   Total missing: {len(missing_videos)}")

def generate_report(stats):
    """Generate a summary report."""
    report = []
    report.append("="*60)
    report.append("     CLEANUP AND VERIFICATION REPORT")
    report.append("="*60)
    report.append("")
    report.append(f"Real Videos: {stats['total_real_videos']}")
    report.append(f"  - With frames: {stats['real_with_frames']}")
    report.append(f"  - Missing: {stats['real_missing']}")
    report.append("")
    report.append(f"Fake Videos: {stats['total_fake_videos']}")
    report.append(f"  - With frames: {stats['fake_with_frames']}")
    report.append(f"  - Missing: {stats['fake_missing']}")
    report.append("")
    report.append(f"Empty folders deleted: {stats['empty_deleted']}")
    report.append(f"Incomplete folders deleted: {stats['incomplete_deleted']}")
    report.append("")
    report.append("Frame Distribution Across Drives:")
    for drive, count in stats['drive_distribution'].items():
        report.append(f"  {drive}: {count} videos")
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    # Print to console
    print("\n" + report_text)
    
    # Save to file
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📄 Report saved: {OUTPUT_REPORT}")

def main():
    """Main execution function."""
    print("="*60)
    print("  🚀 FRAME FOLDERS CLEANUP AND VERIFICATION")
    print("="*60)
    
    stats = {
        'empty_deleted': 0,
        'incomplete_deleted': 0,
        'total_real_videos': 0,
        'real_with_frames': 0,
        'real_missing': 0,
        'total_fake_videos': 0,
        'fake_with_frames': 0,
        'fake_missing': 0,
        'drive_distribution': defaultdict(int)
    }
    
    # Step 1: Clean up fake frame directories
    print("\n" + "="*60)
    print("STEP 1: CLEANUP FAKE FRAME DIRECTORIES")
    print("="*60)
    
    all_empty = []
    all_incomplete = []
    
    for fake_dir in FAKE_FRAMES_DIRS:
        valid, empty, incomplete = scan_frame_directory(fake_dir)
        all_empty.extend(empty)
        all_incomplete.extend(incomplete)
    
    # Delete empty folders
    stats['empty_deleted'] = delete_folders(all_empty, "empty")
    
    # Delete incomplete folders
    stats['incomplete_deleted'] = delete_folders(all_incomplete, "incomplete")
    
    # Step 2: Verify and create mapping
    print("\n" + "="*60)
    print("STEP 2: VERIFICATION AND MAPPING")
    print("="*60)
    
    mapping, missing_videos = verify_and_create_mapping()
    
    # Calculate stats
    for entry in mapping:
        if entry['type'] == 'real':
            stats['total_real_videos'] += 1
            if entry['status'] == 'complete':
                stats['real_with_frames'] += 1
                stats['drive_distribution'][entry['drive']] += 1
            else:
                stats['real_missing'] += 1
        else:
            stats['total_fake_videos'] += 1
            if entry['status'] == 'complete':
                stats['fake_with_frames'] += 1
                stats['drive_distribution'][entry['drive']] += 1
            else:
                stats['fake_missing'] += 1
    
    # Step 3: Save results
    print("\n" + "="*60)
    print("STEP 3: SAVING RESULTS")
    print("="*60)
    
    save_mapping(mapping)
    save_missing_videos(missing_videos)
    generate_report(stats)
    
    print("\n" + "="*60)
    print("✅ CLEANUP AND VERIFICATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
