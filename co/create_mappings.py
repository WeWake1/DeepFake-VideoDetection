"""
Create comprehensive mappings for Celeb-DF dataset
Generates 4 mapping files:
1. enhanced_mapping.csv - Original mapping + relationship columns
2. real_to_fake_mapping.json - Real videos → their fakes with metadata
3. training_pairs.csv - All (real, fake) pairs for training (Approach A)
4. celebrity_mapping.json - Celebrity-centric view
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Input/Output paths
FRAME_MAPPING_CSV = r"j:\DF\frame_mapping.csv"
OUTPUT_DIR = r"j:\DF"

# Output files
ENHANCED_MAPPING_CSV = Path(OUTPUT_DIR) / "enhanced_mapping.csv"
REAL_TO_FAKE_JSON = Path(OUTPUT_DIR) / "real_to_fake_mapping.json"
TRAINING_PAIRS_CSV = Path(OUTPUT_DIR) / "training_pairs.csv"
CELEBRITY_MAPPING_JSON = Path(OUTPUT_DIR) / "celebrity_mapping.json"
MAPPING_STATS_TXT = Path(OUTPUT_DIR) / "mapping_statistics.txt"


def parse_fake_video_name(fake_name: str) -> Tuple[str, str, str]:
    """
    Parse fake video name to extract face source, video source, and sequence.
    
    Format: id{FACE_SOURCE}_id{VIDEO_SOURCE}_{SEQUENCE}.mp4
    Example: id0_id1_0003.mp4 → face_id=0, video_id=1, sequence=0003
    Returns: (face_source_id, real_video_name, original_fake_name)
    
    For real videos (single id pattern): returns (None, None, video_name)
    """
    # Remove .mp4 extension if present
    name = fake_name.replace('.mp4', '')
    
    # Pattern for fake videos: id{X}_id{Y}_{Z}
    fake_pattern = r'id(\d+)_id(\d+)_(\d+)'
    match = re.match(fake_pattern, name)
    
    if match:
        face_id = match.group(1)
        video_id = match.group(2)
        sequence = match.group(3)
        
        # Construct the real video name
        real_video_name = f"id{video_id}_{sequence}"
        
        return (face_id, real_video_name, name)
    
    # If no match, it's a real video
    return (None, None, name)


def load_frame_mapping() -> Dict[str, Dict]:
    """Load existing frame mapping CSV into a dictionary."""
    mapping = {}
    
    with open(FRAME_MAPPING_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video_name']
            mapping[video_name] = {
                'type': row['type'],
                'frame_folder_path': row['frame_folder_path'],
                'num_frames': int(row['num_frames']) if row['num_frames'].isdigit() else 0,
                'status': row['status'],
                'drive': row['drive']
            }
    
    return mapping


def build_relationships(frame_mapping: Dict[str, Dict]) -> Tuple[Dict, Dict, Dict]:
    """
    Build relationship dictionaries from frame mapping.
    
    Returns:
        - real_to_fakes: {real_video: [list of fake videos using it]}
        - fake_to_real: {fake_video: real_source_video}
        - celebrity_videos: {celebrity_id: {'real': [...], 'fakes': [...]}}
    """
    real_to_fakes = defaultdict(list)
    fake_to_real = {}
    celebrity_videos = defaultdict(lambda: {'real': [], 'fakes': []})
    
    for video_name, data in frame_mapping.items():
        if data['type'] == 'real':
            # Extract celebrity ID from real video name (id{X}_{SEQUENCE})
            match = re.match(r'id(\d+)_\d+', video_name)
            if match:
                celeb_id = match.group(1)
                celebrity_videos[celeb_id]['real'].append(video_name)
        
        elif data['type'] == 'fake':
            # Parse fake video to get relationships
            face_id, real_video, _ = parse_fake_video_name(video_name)
            
            if face_id and real_video:
                # Map fake to its source real video
                fake_to_real[video_name] = real_video
                
                # Map real video to this fake
                real_to_fakes[real_video].append(video_name)
                
                # Add to celebrity's fake videos
                celebrity_videos[face_id]['fakes'].append(video_name)
    
    return dict(real_to_fakes), fake_to_real, dict(celebrity_videos)


def create_enhanced_mapping(frame_mapping: Dict, fake_to_real: Dict, real_to_fakes: Dict):
    """Create enhanced mapping CSV with relationship columns."""
    print(f"Creating enhanced mapping: {ENHANCED_MAPPING_CSV}")
    
    with open(ENHANCED_MAPPING_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'video_name', 'type', 'frame_folder_path', 'num_frames', 
            'status', 'drive', 'source_real_video', 'face_source_id', 
            'num_related_fakes', 'num_related_reals'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for video_name, data in sorted(frame_mapping.items()):
            row = {
                'video_name': video_name,
                'type': data['type'],
                'frame_folder_path': data['frame_folder_path'],
                'num_frames': data['num_frames'],
                'status': data['status'],
                'drive': data['drive'],
                'source_real_video': '',
                'face_source_id': '',
                'num_related_fakes': '',
                'num_related_reals': ''
            }
            
            if data['type'] == 'fake':
                # Add fake-specific info
                face_id, real_video, _ = parse_fake_video_name(video_name)
                row['source_real_video'] = real_video if real_video else 'unknown'
                row['face_source_id'] = face_id if face_id else 'unknown'
                row['num_related_reals'] = 1 if real_video in frame_mapping else 0
            
            elif data['type'] == 'real':
                # Add real-specific info
                row['num_related_fakes'] = len(real_to_fakes.get(video_name, []))
            
            writer.writerow(row)
    
    print(f"✓ Enhanced mapping saved: {len(frame_mapping)} videos")


def create_real_to_fake_mapping(frame_mapping: Dict, real_to_fakes: Dict):
    """Create JSON mapping of real videos to their fakes with full metadata."""
    print(f"Creating real-to-fake mapping: {REAL_TO_FAKE_JSON}")
    
    result = {}
    
    for real_video in sorted(real_to_fakes.keys()):
        if real_video not in frame_mapping:
            continue  # Skip if real video not found
        
        real_data = frame_mapping[real_video]
        fake_list = []
        
        for fake_video in sorted(real_to_fakes[real_video]):
            if fake_video in frame_mapping:
                fake_data = frame_mapping[fake_video]
                face_id, _, _ = parse_fake_video_name(fake_video)
                
                fake_list.append({
                    'fake_video_name': fake_video,
                    'face_source_id': face_id,
                    'frame_folder_path': fake_data['frame_folder_path'],
                    'num_frames': fake_data['num_frames'],
                    'status': fake_data['status'],
                    'drive': fake_data['drive']
                })
        
        result[real_video] = {
            'real_video_metadata': {
                'frame_folder_path': real_data['frame_folder_path'],
                'num_frames': real_data['num_frames'],
                'status': real_data['status'],
                'drive': real_data['drive']
            },
            'num_fakes_using_this_video': len(fake_list),
            'fakes': fake_list
        }
    
    with open(REAL_TO_FAKE_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    total_fakes = sum(len(v['fakes']) for v in result.values())
    print(f"✓ Real-to-fake mapping saved: {len(result)} real videos → {total_fakes} fakes")


def create_training_pairs(frame_mapping: Dict, fake_to_real: Dict):
    """Create CSV with all (real, fake) training pairs (Approach A: ALL pairs)."""
    print(f"Creating training pairs: {TRAINING_PAIRS_CSV}")
    
    with open(TRAINING_PAIRS_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'pair_id', 'real_video', 'fake_video', 'face_source_id',
            'real_frames_path', 'fake_frames_path',
            'real_num_frames', 'fake_num_frames',
            'real_drive', 'fake_drive',
            'real_status', 'fake_status',
            'both_complete'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        pair_id = 0
        complete_pairs = 0
        incomplete_pairs = 0
        
        # Iterate through all fakes and create pairs
        for fake_video, real_video in sorted(fake_to_real.items()):
            if fake_video not in frame_mapping or real_video not in frame_mapping:
                continue
            
            fake_data = frame_mapping[fake_video]
            real_data = frame_mapping[real_video]
            face_id, _, _ = parse_fake_video_name(fake_video)
            
            both_complete = (fake_data['status'] == 'complete' and 
                           real_data['status'] == 'complete')
            
            if both_complete:
                complete_pairs += 1
            else:
                incomplete_pairs += 1
            
            pair_id += 1
            writer.writerow({
                'pair_id': pair_id,
                'real_video': real_video,
                'fake_video': fake_video,
                'face_source_id': face_id,
                'real_frames_path': real_data['frame_folder_path'],
                'fake_frames_path': fake_data['frame_folder_path'],
                'real_num_frames': real_data['num_frames'],
                'fake_num_frames': fake_data['num_frames'],
                'real_drive': real_data['drive'],
                'fake_drive': fake_data['drive'],
                'real_status': real_data['status'],
                'fake_status': fake_data['status'],
                'both_complete': 'yes' if both_complete else 'no'
            })
    
    print(f"✓ Training pairs saved: {pair_id} total pairs")
    print(f"  - Complete pairs (both videos extracted): {complete_pairs}")
    print(f"  - Incomplete pairs (one or both missing): {incomplete_pairs}")


def create_celebrity_mapping(frame_mapping: Dict, celebrity_videos: Dict):
    """Create JSON with celebrity-centric view of videos."""
    print(f"Creating celebrity mapping: {CELEBRITY_MAPPING_JSON}")
    
    result = {}
    
    for celeb_id in sorted(celebrity_videos.keys(), key=lambda x: int(x)):
        celeb_data = celebrity_videos[celeb_id]
        
        # Get real video metadata
        real_list = []
        for real_video in sorted(celeb_data['real']):
            if real_video in frame_mapping:
                data = frame_mapping[real_video]
                real_list.append({
                    'video_name': real_video,
                    'frame_folder_path': data['frame_folder_path'],
                    'num_frames': data['num_frames'],
                    'status': data['status'],
                    'drive': data['drive']
                })
        
        # Get fake video metadata (where this celebrity's face was used)
        fake_list = []
        for fake_video in sorted(celeb_data['fakes']):
            if fake_video in frame_mapping:
                data = frame_mapping[fake_video]
                _, real_source, _ = parse_fake_video_name(fake_video)
                fake_list.append({
                    'video_name': fake_video,
                    'real_source_video': real_source,
                    'frame_folder_path': data['frame_folder_path'],
                    'num_frames': data['num_frames'],
                    'status': data['status'],
                    'drive': data['drive']
                })
        
        result[f"id{celeb_id}"] = {
            'celebrity_id': celeb_id,
            'num_real_videos': len(real_list),
            'num_fakes_using_face': len(fake_list),
            'real_videos': real_list,
            'fakes_using_this_face': fake_list
        }
    
    with open(CELEBRITY_MAPPING_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    total_celebs = len(result)
    total_real = sum(v['num_real_videos'] for v in result.values())
    total_fakes = sum(v['num_fakes_using_face'] for v in result.values())
    print(f"✓ Celebrity mapping saved: {total_celebs} celebrities")
    print(f"  - Total real videos: {total_real}")
    print(f"  - Total fakes (by face source): {total_fakes}")


def create_statistics(frame_mapping: Dict, real_to_fakes: Dict, 
                     fake_to_real: Dict, celebrity_videos: Dict):
    """Create comprehensive statistics report."""
    print(f"Creating statistics report: {MAPPING_STATS_TXT}")
    
    # Count videos by type and status
    real_complete = sum(1 for v in frame_mapping.values() 
                       if v['type'] == 'real' and v['status'] == 'complete')
    real_missing = sum(1 for v in frame_mapping.values() 
                      if v['type'] == 'real' and v['status'] == 'missing')
    fake_complete = sum(1 for v in frame_mapping.values() 
                       if v['type'] == 'fake' and v['status'] == 'complete')
    fake_missing = sum(1 for v in frame_mapping.values() 
                      if v['type'] == 'fake' and v['status'] == 'missing')
    
    # Count by drive
    drive_counts = defaultdict(int)
    for data in frame_mapping.values():
        drive_counts[data['drive']] += 1
    
    # Frame statistics
    total_frames = sum(v['num_frames'] for v in frame_mapping.values())
    real_frames = sum(v['num_frames'] for v in frame_mapping.values() if v['type'] == 'real')
    fake_frames = sum(v['num_frames'] for v in frame_mapping.values() if v['type'] == 'fake')
    
    # Relationship statistics
    real_with_fakes = len(real_to_fakes)
    real_without_fakes = real_complete - real_with_fakes
    avg_fakes_per_real = (sum(len(fakes) for fakes in real_to_fakes.values()) / 
                          real_with_fakes if real_with_fakes > 0 else 0)
    
    # Celebrity statistics
    num_celebrities = len(celebrity_videos)
    avg_real_per_celeb = (sum(len(v['real']) for v in celebrity_videos.values()) / 
                          num_celebrities if num_celebrities > 0 else 0)
    avg_fakes_per_celeb = (sum(len(v['fakes']) for v in celebrity_videos.values()) / 
                           num_celebrities if num_celebrities > 0 else 0)
    
    with open(MAPPING_STATS_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CELEB-DF MAPPING STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total videos: {len(frame_mapping)}\n")
        f.write(f"  - Real videos: {real_complete + real_missing} ({real_complete} complete, {real_missing} missing)\n")
        f.write(f"  - Fake videos: {fake_complete + fake_missing} ({fake_complete} complete, {fake_missing} missing)\n")
        f.write(f"Total frames extracted: {total_frames:,}\n")
        f.write(f"  - From real videos: {real_frames:,}\n")
        f.write(f"  - From fake videos: {fake_frames:,}\n\n")
        
        f.write("STORAGE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for drive in sorted(drive_counts.keys()):
            f.write(f"Drive {drive}: {drive_counts[drive]} videos\n")
        f.write("\n")
        
        f.write("RELATIONSHIP STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Real videos with fakes: {real_with_fakes}\n")
        f.write(f"Real videos without fakes: {real_without_fakes}\n")
        f.write(f"Average fakes per real video: {avg_fakes_per_real:.2f}\n")
        f.write(f"Total fake-to-real mappings: {len(fake_to_real)}\n\n")
        
        f.write("CELEBRITY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total celebrities: {num_celebrities}\n")
        f.write(f"Average real videos per celebrity: {avg_real_per_celeb:.2f}\n")
        f.write(f"Average fakes per celebrity (by face): {avg_fakes_per_celeb:.2f}\n\n")
        
        f.write("OUTPUT FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"1. {ENHANCED_MAPPING_CSV.name} - Enhanced mapping with relationships\n")
        f.write(f"2. {REAL_TO_FAKE_JSON.name} - Real videos → their fakes (JSON)\n")
        f.write(f"3. {TRAINING_PAIRS_CSV.name} - All (real, fake) pairs for training\n")
        f.write(f"4. {CELEBRITY_MAPPING_JSON.name} - Celebrity-centric view (JSON)\n")
        f.write(f"5. {MAPPING_STATS_TXT.name} - This statistics file\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Mapping creation completed successfully!\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Statistics report saved")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("CELEB-DF COMPREHENSIVE MAPPING GENERATION")
    print("=" * 80 + "\n")
    
    # Step 1: Load existing frame mapping
    print(f"Loading frame mapping from: {FRAME_MAPPING_CSV}")
    frame_mapping = load_frame_mapping()
    print(f"✓ Loaded {len(frame_mapping)} videos\n")
    
    # Step 2: Build relationships
    print("Building relationship dictionaries...")
    real_to_fakes, fake_to_real, celebrity_videos = build_relationships(frame_mapping)
    print(f"✓ Found {len(real_to_fakes)} real videos with fakes")
    print(f"✓ Found {len(fake_to_real)} fake-to-real mappings")
    print(f"✓ Found {len(celebrity_videos)} celebrities\n")
    
    # Step 3: Create all mapping files
    print("Generating mapping files...\n")
    
    create_enhanced_mapping(frame_mapping, fake_to_real, real_to_fakes)
    print()
    
    create_real_to_fake_mapping(frame_mapping, real_to_fakes)
    print()
    
    create_training_pairs(frame_mapping, fake_to_real)
    print()
    
    create_celebrity_mapping(frame_mapping, celebrity_videos)
    print()
    
    create_statistics(frame_mapping, real_to_fakes, fake_to_real, celebrity_videos)
    print()
    
    # Summary
    print("=" * 80)
    print("ALL MAPPING FILES GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  1. {ENHANCED_MAPPING_CSV.name}")
    print(f"  2. {REAL_TO_FAKE_JSON.name}")
    print(f"  3. {TRAINING_PAIRS_CSV.name}")
    print(f"  4. {CELEBRITY_MAPPING_JSON.name}")
    print(f"  5. {MAPPING_STATS_TXT.name}")
    print("\nReady for deepfake detection model training!\n")


if __name__ == "__main__":
    main()
