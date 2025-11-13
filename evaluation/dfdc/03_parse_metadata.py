"""
Parse DFDC metadata.json and create test CSV
Converts DFDC metadata into format compatible with inference pipeline
"""
import json
import pandas as pd
from pathlib import Path

# ===== CONFIGURATION =====
METADATA_PATH = r"C:\Users\Administrater\Downloads\deepfake-detection-challenge\train_sample_videos\metadata.json"
FACES_DIR = r"H:\EVAL-1\faces"
OUTPUT_CSV = r"j:\DF\evaluation\dfdc\dfdc_test_videos.csv"

def parse_metadata():
    """Parse DFDC metadata.json and create test CSV"""
    print("="*70)
    print("📋 DFDC METADATA PARSER")
    print("="*70)
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   Faces Dir: {FACES_DIR}")
    print(f"   Output CSV: {OUTPUT_CSV}")
    print("="*70)
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n✓ Loaded metadata for {len(metadata)} videos")
    
    # Parse metadata
    video_data = []
    faces_dir = Path(FACES_DIR)
    
    for video_file, info in metadata.items():
        video_name = Path(video_file).stem  # Remove .mp4 extension
        label = info.get('label', 'UNKNOWN').lower()  # FAKE or REAL -> fake or real
        
        # Check if faces exist
        face_folder = faces_dir / video_name
        
        if face_folder.exists():
            face_count = len(list(face_folder.glob('*.jpg')))
            status = 'exists'
        else:
            face_count = 0
            status = 'missing'
        
        video_data.append({
            'video_name': video_name,
            'label': label,
            'faces_path': str(face_folder),
            'face_count': face_count,
            'status': status
        })
    
    # Create DataFrame
    df = pd.DataFrame(video_data)
    
    # Statistics
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    print(f"   Total videos: {len(df)}")
    print(f"   Real videos: {(df['label'] == 'real').sum()}")
    print(f"   Fake videos: {(df['label'] == 'fake').sum()}")
    print(f"   Videos with faces: {(df['status'] == 'exists').sum()}")
    print(f"   Videos missing faces: {(df['status'] == 'missing').sum()}")
    print(f"   Total faces detected: {df['face_count'].sum():,}")
    print(f"   Avg faces per video: {df[df['face_count'] > 0]['face_count'].mean():.1f}")
    print("="*70)
    
    # Save CSV
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Test CSV saved to: {OUTPUT_CSV}")
    
    # Show sample
    print("\n📋 Sample entries:")
    print(df.head(10).to_string(index=False))
    
    # Filter only videos with faces for testing
    df_testable = df[df['status'] == 'exists'].copy()
    testable_path = output_path.parent / "dfdc_test_videos_ready.csv"
    df_testable.to_csv(testable_path, index=False)
    
    print(f"\n✅ Testable videos CSV saved to: {testable_path}")
    print(f"   ({len(df_testable)} videos ready for testing)")
    
    return df

if __name__ == "__main__":
    parse_metadata()
