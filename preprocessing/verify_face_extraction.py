"""
Verify Face Extraction Completeness
Checks if faces have been extracted for all videos
Generates detailed report of missing/incomplete videos
"""

import csv
from pathlib import Path
from collections import defaultdict
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input files
FRAME_MAPPING_CSV = r"j:\DF\frame_mapping.csv"
OUTPUT_BASE_DIR = r"F:"  # Where faces are stored

# Frame skip setting used during extraction
FRAME_SKIP = 3  # Must match what you used in face_detect_mtcnn_gpu.py

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_face_extraction():
    """Check if all videos have faces extracted"""
    
    print("=" * 80)
    print("FACE EXTRACTION VERIFICATION")
    print("=" * 80)
    print()
    
    # Load video list
    print(f"Loading video list from: {FRAME_MAPPING_CSV}")
    videos = []
    with open(FRAME_MAPPING_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] in ['completed', 'complete']:
                videos.append(row)
    
    print(f"✓ Found {len(videos)} videos to verify")
    print()
    
    # Statistics
    stats = {
        'total_videos': len(videos),
        'complete': 0,
        'partial': 0,
        'missing': 0,
        'total_frames_expected': 0,
        'total_faces_found': 0,
        'videos_by_status': defaultdict(list)
    }
    
    # Check each video
    print("Checking face extraction status...")
    print()
    
    for idx, video_info in enumerate(videos, 1):
        video_name = video_info['video_name']
        video_type = video_info['type']
        num_frames = int(video_info['num_frames'])
        
        # Calculate expected number of faces (with frame skip)
        expected_faces = (num_frames + FRAME_SKIP - 1) // FRAME_SKIP
        stats['total_frames_expected'] += expected_faces
        
        # Check if face folder exists
        face_folder = Path(OUTPUT_BASE_DIR) / video_type / video_name
        
        if not face_folder.exists():
            # No faces extracted at all
            stats['missing'] += 1
            stats['videos_by_status']['missing'].append({
                'video_name': video_name,
                'type': video_type,
                'expected_faces': expected_faces,
                'found_faces': 0,
                'completion': 0.0
            })
            continue
        
        # Count extracted faces
        face_files = list(face_folder.glob('*.jpg')) + list(face_folder.glob('*.png'))
        num_faces = len(face_files)
        stats['total_faces_found'] += num_faces
        
        # Calculate completion percentage
        completion = (num_faces / expected_faces * 100) if expected_faces > 0 else 0
        
        # Categorize
        if num_faces >= expected_faces * 0.95:  # 95% threshold for "complete"
            stats['complete'] += 1
            stats['videos_by_status']['complete'].append({
                'video_name': video_name,
                'type': video_type,
                'expected_faces': expected_faces,
                'found_faces': num_faces,
                'completion': completion
            })
        elif num_faces > 0:
            stats['partial'] += 1
            stats['videos_by_status']['partial'].append({
                'video_name': video_name,
                'type': video_type,
                'expected_faces': expected_faces,
                'found_faces': num_faces,
                'completion': completion
            })
        else:
            stats['missing'] += 1
            stats['videos_by_status']['missing'].append({
                'video_name': video_name,
                'type': video_type,
                'expected_faces': expected_faces,
                'found_faces': 0,
                'completion': 0.0
            })
        
        # Progress indicator
        if idx % 100 == 0:
            print(f"  Checked {idx}/{len(videos)} videos...")
    
    print(f"✓ Verification complete!")
    print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total videos: {stats['total_videos']}")
    print(f"  ✓ Complete (≥95%): {stats['complete']} ({stats['complete']/stats['total_videos']*100:.1f}%)")
    print(f"  ⚠ Partial (<95%):  {stats['partial']} ({stats['partial']/stats['total_videos']*100:.1f}%)")
    print(f"  ✗ Missing (0%):    {stats['missing']} ({stats['missing']/stats['total_videos']*100:.1f}%)")
    print()
    print(f"Faces extracted: {stats['total_faces_found']:,} / {stats['total_frames_expected']:,} expected")
    print(f"Overall completion: {stats['total_faces_found']/stats['total_frames_expected']*100:.1f}%")
    print()
    
    # Breakdown by type
    print("Breakdown by type:")
    type_stats = defaultdict(lambda: {'complete': 0, 'partial': 0, 'missing': 0, 'total': 0})
    
    for status in ['complete', 'partial', 'missing']:
        for video in stats['videos_by_status'][status]:
            vtype = video['type']
            type_stats[vtype][status] += 1
            type_stats[vtype]['total'] += 1
    
    for vtype, counts in sorted(type_stats.items()):
        print(f"  {vtype}:")
        print(f"    Complete: {counts['complete']:4d} | Partial: {counts['partial']:4d} | Missing: {counts['missing']:4d} | Total: {counts['total']:4d}")
    print()
    
    # Save detailed report
    report_file = Path(FRAME_MAPPING_CSV).parent / 'face_extraction_verification.json'
    with open(report_file, 'w') as f:
        json.dump({
            'summary': {
                'total_videos': stats['total_videos'],
                'complete': stats['complete'],
                'partial': stats['partial'],
                'missing': stats['missing'],
                'total_faces_found': stats['total_faces_found'],
                'total_frames_expected': stats['total_frames_expected'],
                'completion_rate': stats['total_faces_found']/stats['total_frames_expected']*100 if stats['total_frames_expected'] > 0 else 0
            },
            'videos': {
                'complete': stats['videos_by_status']['complete'],
                'partial': stats['videos_by_status']['partial'],
                'missing': stats['videos_by_status']['missing']
            }
        }, f, indent=2)
    
    print(f"Detailed report saved to: {report_file}")
    print()
    
    # Show missing videos if any
    if stats['missing'] > 0:
        print("=" * 80)
        print(f"MISSING VIDEOS ({stats['missing']} total)")
        print("=" * 80)
        print()
        for video in stats['videos_by_status']['missing'][:20]:  # Show first 20
            print(f"  • {video['video_name']} ({video['type']}) - Expected: {video['expected_faces']} faces")
        
        if stats['missing'] > 20:
            print(f"  ... and {stats['missing'] - 20} more (see JSON report for full list)")
        print()
    
    # Show partial videos if any
    if stats['partial'] > 0:
        print("=" * 80)
        print(f"PARTIAL VIDEOS ({stats['partial']} total)")
        print("=" * 80)
        print()
        # Sort by completion percentage
        partial_sorted = sorted(stats['videos_by_status']['partial'], key=lambda x: x['completion'])
        
        for video in partial_sorted[:20]:  # Show first 20
            print(f"  • {video['video_name']} ({video['type']}) - {video['found_faces']}/{video['expected_faces']} ({video['completion']:.1f}%)")
        
        if stats['partial'] > 20:
            print(f"  ... and {stats['partial'] - 20} more (see JSON report for full list)")
        print()
    
    # Final status
    print("=" * 80)
    if stats['missing'] == 0 and stats['partial'] == 0:
        print("✓ ALL VIDEOS HAVE FACES EXTRACTED!")
        print("✓ Ready for model training!")
    elif stats['missing'] == 0:
        print(f"⚠ Face extraction mostly complete, but {stats['partial']} videos are partial.")
        print("  You can proceed with training or re-run face detection on partial videos.")
    else:
        print(f"✗ Face extraction incomplete!")
        print(f"  {stats['missing']} videos have no faces")
        print(f"  {stats['partial']} videos are partial")
        print()
        print("To complete extraction, run:")
        print("  python co\\face_detect_mtcnn_gpu.py")
        print("  (It will automatically skip completed videos)")
    print("=" * 80)
    print()

if __name__ == "__main__":
    verify_face_extraction()
