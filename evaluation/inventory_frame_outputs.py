"""
Comprehensive Frame Output Inventory
Checks all extracted frames for DFD and FaceForensics++ datasets
Generates detailed CSV reports with per-video statistics
"""

import csv
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directories to check
FF_FRAMES_BASE = r"F:\FF++_preprocessed\frames"
DFD_FRAMES_BASE = r"F:\DFD_preprocessed\frames"

# FF++ manipulation folders
FF_FOLDERS = [
    'original',
    'Deepfakes',
    'Face2Face',
    'FaceSwap',
    'NeuralTextures',
    'FaceShifter',
    'DeepFakeDetection'
]

# DFD folders
DFD_FOLDERS = [
    'DFD_original sequences',
    'DFD_manipulated_sequences'
]

# Output files
OUTPUT_DIR = Path(r"j:\DF\evaluation")
FF_INVENTORY_CSV = OUTPUT_DIR / "ff_frame_inventory.csv"
DFD_INVENTORY_CSV = OUTPUT_DIR / "dfd_frame_inventory.csv"
SUMMARY_TXT = OUTPUT_DIR / "frame_inventory_summary.txt"
PROBLEMS_CSV = OUTPUT_DIR / "frame_problems.csv"

# Thresholds for flagging issues
MIN_FRAMES_THRESHOLD = 10  # Flag videos with fewer than 10 frames
EXPECTED_FRAME_SKIP = 3  # We extract every 3rd frame

# ============================================================================
# INVENTORY FUNCTIONS
# ============================================================================

def check_video_folder(video_folder):
    """
    Check a single video folder and return statistics
    
    Returns:
        dict with keys: video_name, png_count, first_file, last_file, 
                       has_gaps, file_pattern
    """
    video_name = video_folder.name
    
    # Get all PNG files
    png_files = sorted(list(video_folder.glob('*.png')))
    png_count = len(png_files)
    
    if png_count == 0:
        return {
            'video_name': video_name,
            'png_count': 0,
            'first_file': '',
            'last_file': '',
            'has_gaps': False,
            'file_pattern': 'no_files',
            'status': 'empty'
        }
    
    first_file = png_files[0].name
    last_file = png_files[-1].name
    
    # Check for gaps in numbering
    frame_numbers = []
    file_pattern = 'unknown'
    
    for f in png_files[:min(10, len(png_files))]:  # Sample first 10 to detect pattern
        try:
            # Try to extract number from filename (e.g., frame_00123.png -> 123)
            parts = f.stem.split('_')
            if len(parts) >= 2:
                num = int(parts[1])
                frame_numbers.append(num)
                # Detect padding length
                num_str = parts[1]
                if file_pattern == 'unknown':
                    file_pattern = f"frame_{len(num_str)}digit"
        except (ValueError, IndexError):
            pass
    
    # Check for gaps
    has_gaps = False
    if len(frame_numbers) > 1:
        # Check if sequential (accounting for frame skip)
        diffs = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
        # If all differences are 1, it's sequential extracted indices
        # If all differences are 3, it's absolute frame indices with skip=3
        if diffs:
            common_diff = max(set(diffs), key=diffs.count)
            if common_diff == 1:
                file_pattern = file_pattern + '_sequential'
            elif common_diff == EXPECTED_FRAME_SKIP:
                file_pattern = file_pattern + '_absolute'
            
            # Flag if we see unexpected gaps
            has_gaps = any(d != common_diff for d in diffs)
    
    status = 'ok'
    if png_count < MIN_FRAMES_THRESHOLD:
        status = 'low_count'
    elif has_gaps:
        status = 'has_gaps'
    
    return {
        'video_name': video_name,
        'png_count': png_count,
        'first_file': first_file,
        'last_file': last_file,
        'has_gaps': has_gaps,
        'file_pattern': file_pattern,
        'status': status
    }


def inventory_dataset(base_path, folder_list, dataset_name):
    """
    Inventory all videos in a dataset
    
    Returns:
        list of dicts with video statistics
    """
    print(f"\n{'='*80}")
    print(f"INVENTORYING {dataset_name}")
    print(f"{'='*80}\n")
    
    base = Path(base_path)
    
    if not base.exists():
        print(f"❌ Base path does not exist: {base_path}")
        return []
    
    all_results = []
    folder_stats = {}
    
    for folder_name in folder_list:
        folder_path = base / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            folder_stats[folder_name] = {'count': 0, 'exists': False}
            continue
        
        # Get all video subfolders
        video_folders = [f for f in folder_path.iterdir() if f.is_dir()]
        folder_stats[folder_name] = {'count': len(video_folders), 'exists': True}
        
        print(f"Processing {folder_name}: {len(video_folders)} videos...")
        
        # Check each video folder
        for video_folder in tqdm(video_folders, desc=f"  {folder_name}", leave=False):
            result = check_video_folder(video_folder)
            result['dataset'] = dataset_name
            result['folder'] = folder_name
            result['full_path'] = str(video_folder)
            all_results.append(result)
    
    # Print folder summary
    print(f"\n{dataset_name} Folder Summary:")
    for folder_name, stats in folder_stats.items():
        if stats['exists']:
            print(f"  ✓ {folder_name:30s}: {stats['count']:4d} video folders")
        else:
            print(f"  ✗ {folder_name:30s}: not found")
    
    return all_results


def save_inventory_csv(results, output_path, dataset_name):
    """Save inventory results to CSV"""
    if not results:
        print(f"No results to save for {dataset_name}")
        return
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['dataset', 'folder', 'video_name', 'png_count', 
                     'first_file', 'last_file', 'file_pattern', 
                     'has_gaps', 'status', 'full_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in results:
            writer.writerow(row)
    
    print(f"✓ Saved {len(results)} entries to: {output_path}")


def generate_summary(ff_results, dfd_results):
    """Generate comprehensive summary report"""
    
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY REPORT")
    print(f"{'='*80}\n")
    
    # Combine all results
    all_results = ff_results + dfd_results
    
    # Count by status
    status_counts = defaultdict(int)
    for r in all_results:
        status_counts[r['status']] += 1
    
    # Find problem videos
    problems = [r for r in all_results if r['status'] != 'ok']
    
    # Calculate statistics
    total_videos = len(all_results)
    total_frames = sum(r['png_count'] for r in all_results)
    videos_ok = status_counts['ok']
    videos_low = status_counts['low_count']
    videos_empty = status_counts['empty']
    videos_gaps = status_counts['has_gaps']
    
    # Write summary to file
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FRAME EXTRACTION INVENTORY SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total videos checked: {total_videos}\n")
        f.write(f"Total frames extracted: {total_frames:,}\n")
        f.write(f"  ✓ OK: {videos_ok} ({videos_ok/total_videos*100:.1f}%)\n")
        f.write(f"  ⚠ Low count (<{MIN_FRAMES_THRESHOLD}): {videos_low} ({videos_low/total_videos*100:.1f}%)\n")
        f.write(f"  ⚠ Has gaps: {videos_gaps} ({videos_gaps/total_videos*100:.1f}%)\n")
        f.write(f"  ✗ Empty: {videos_empty} ({videos_empty/total_videos*100:.1f}%)\n\n")
        
        # FaceForensics++ breakdown
        f.write("FACEFORENSICS++ BREAKDOWN\n")
        f.write("-"*80 + "\n")
        ff_by_folder = defaultdict(lambda: {'total': 0, 'ok': 0, 'problems': 0, 'frames': 0})
        for r in ff_results:
            folder = r['folder']
            ff_by_folder[folder]['total'] += 1
            ff_by_folder[folder]['frames'] += r['png_count']
            if r['status'] == 'ok':
                ff_by_folder[folder]['ok'] += 1
            else:
                ff_by_folder[folder]['problems'] += 1
        
        for folder, stats in sorted(ff_by_folder.items()):
            f.write(f"  {folder:30s}: {stats['ok']:4d}/{stats['total']:4d} OK, "
                   f"{stats['problems']:4d} problems, {stats['frames']:,} frames\n")
        f.write(f"\n  Total FF++ videos: {len(ff_results)}\n")
        f.write(f"  Total FF++ frames: {sum(r['png_count'] for r in ff_results):,}\n\n")
        
        # DFD breakdown
        f.write("DFD BREAKDOWN\n")
        f.write("-"*80 + "\n")
        dfd_by_folder = defaultdict(lambda: {'total': 0, 'ok': 0, 'problems': 0, 'frames': 0})
        for r in dfd_results:
            folder = r['folder']
            dfd_by_folder[folder]['total'] += 1
            dfd_by_folder[folder]['frames'] += r['png_count']
            if r['status'] == 'ok':
                dfd_by_folder[folder]['ok'] += 1
            else:
                dfd_by_folder[folder]['problems'] += 1
        
        for folder, stats in sorted(dfd_by_folder.items()):
            f.write(f"  {folder:30s}: {stats['ok']:4d}/{stats['total']:4d} OK, "
                   f"{stats['problems']:4d} problems, {stats['frames']:,} frames\n")
        f.write(f"\n  Total DFD videos: {len(dfd_results)}\n")
        f.write(f"  Total DFD frames: {sum(r['png_count'] for r in dfd_results):,}\n\n")
        
        # File pattern analysis
        f.write("FILE NAMING PATTERNS DETECTED\n")
        f.write("-"*80 + "\n")
        pattern_counts = defaultdict(int)
        for r in all_results:
            if r['png_count'] > 0:
                pattern_counts[r['file_pattern']] += 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {pattern:40s}: {count:4d} videos\n")
        f.write("\n")
        
        # Problem summary
        if problems:
            f.write("PROBLEM VIDEOS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total problem videos: {len(problems)}\n\n")
            
            # Show top 20 lowest frame counts
            problems_sorted = sorted(problems, key=lambda x: x['png_count'])
            f.write("Top 20 videos with lowest frame counts:\n")
            for r in problems_sorted[:20]:
                f.write(f"  {r['video_name']:40s} ({r['folder']:20s}): {r['png_count']:4d} frames - {r['status']}\n")
            
            if len(problems) > 20:
                f.write(f"  ... and {len(problems)-20} more (see {PROBLEMS_CSV.name})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILES GENERATED:\n")
        f.write(f"  - {FF_INVENTORY_CSV.name}\n")
        f.write(f"  - {DFD_INVENTORY_CSV.name}\n")
        f.write(f"  - {PROBLEMS_CSV.name}\n")
        f.write(f"  - {SUMMARY_TXT.name}\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary saved to: {SUMMARY_TXT}")
    
    # Save problems CSV
    if problems:
        with open(PROBLEMS_CSV, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['dataset', 'folder', 'video_name', 'png_count', 
                         'status', 'first_file', 'last_file', 'file_pattern',
                         'has_gaps', 'full_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in sorted(problems, key=lambda x: x['png_count']):
                writer.writerow(row)
        
        print(f"✓ Problem videos saved to: {PROBLEMS_CSV}")
        print(f"  ({len(problems)} videos flagged)")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {total_videos}")
    print(f"Total frames: {total_frames:,}")
    print(f"  ✓ OK: {videos_ok} ({videos_ok/total_videos*100:.1f}%)")
    print(f"  ⚠ Problems: {len(problems)} ({len(problems)/total_videos*100:.1f}%)")
    print(f"\nDetailed reports saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("FRAME EXTRACTION COMPREHENSIVE INVENTORY")
    print("="*80)
    print(f"\nChecking:")
    print(f"  - FaceForensics++: {FF_FRAMES_BASE}")
    print(f"  - DFD: {DFD_FRAMES_BASE}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Inventory FaceForensics++
    ff_results = inventory_dataset(FF_FRAMES_BASE, FF_FOLDERS, "FaceForensics++")
    save_inventory_csv(ff_results, FF_INVENTORY_CSV, "FaceForensics++")
    
    # Inventory DFD
    dfd_results = inventory_dataset(DFD_FRAMES_BASE, DFD_FOLDERS, "DFD")
    save_inventory_csv(dfd_results, DFD_INVENTORY_CSV, "DFD")
    
    # Generate summary
    generate_summary(ff_results, dfd_results)
    
    print("\n✓ Inventory complete!\n")


if __name__ == "__main__":
    main()
