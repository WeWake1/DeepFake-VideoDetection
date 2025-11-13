"""
Parse FaceForensics++ metadata and create test CSV
Converts FF++ folder structure into format compatible with inference pipeline
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== FACEFORENSICS++ CONFIGURATION =====
FACES_DIR = r"F:\FF++_preprocessed\faces"
OUTPUT_CSV = r"j:\DF\evaluation\faceforensics++\ff_test_videos.csv"

# Folders to process
FOLDERS_TO_PROCESS = {
    'original': 'real',
    'Deepfakes': 'fake',
    'Face2Face': 'fake',
    'FaceSwap': 'fake',
    'NeuralTextures': 'fake',
    'FaceShifter': 'fake',
    'DeepFakeDetection': 'fake'
}

def parse_faceforensics_metadata():
    """Parse FF++ folder structure and create test CSV"""
    print("="*70)
    print("📋 FACEFORENSICS++ METADATA PARSER")
    print("="*70)
    print(f"   Faces Dir: {FACES_DIR}")
    print(f"   Output CSV: {OUTPUT_CSV}")
    print(f"   Folders: {len(FOLDERS_TO_PROCESS)}")
    print("="*70)
    
    faces_base = Path(FACES_DIR)
    
    if not faces_base.exists():
        print(f"❌ Error: Faces directory not found: {faces_base}")
        return
    
    video_data = []
    
    # Process each folder
    for folder_name, label in tqdm(FOLDERS_TO_PROCESS.items(), desc="Scanning folders"):
        folder_path = faces_base / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder_name}")
            continue
        
        # Get all video subfolders
        video_folders = [f for f in folder_path.iterdir() if f.is_dir()]
        
        print(f"\n   {folder_name}: {len(video_folders)} videos")
        
        for video_folder in tqdm(video_folders, desc=f"  {folder_name}", leave=False):
            video_name = video_folder.name
            
            # Count faces
            face_count = len(list(video_folder.glob('*.jpg')))
            
            if face_count > 0:
                video_data.append({
                    'video_name': video_name,
                    'label': label,
                    'manipulation': folder_name,
                    'faces_path': str(video_folder),
                    'face_count': face_count,
                    'status': 'exists'
                })
            else:
                video_data.append({
                    'video_name': video_name,
                    'label': label,
                    'manipulation': folder_name,
                    'faces_path': str(video_folder),
                    'face_count': 0,
                    'status': 'no_faces'
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
    print(f"   Videos without faces: {(df['status'] == 'no_faces').sum()}")
    print(f"   Total faces detected: {df['face_count'].sum():,}")
    print(f"   Avg faces per video: {df[df['face_count'] > 0]['face_count'].mean():.1f}")
    print("\n   By manipulation method:")
    for method in FOLDERS_TO_PROCESS.keys():
        method_df = df[df['manipulation'] == method]
        count = len(method_df)
        with_faces = (method_df['status'] == 'exists').sum()
        print(f"      {method:20s}: {with_faces:4d}/{count:4d} videos with faces")
    print("="*70)
    
    # Save full CSV
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Test CSV saved to: {OUTPUT_CSV}")
    
    # Show sample
    print("\n📋 Sample entries:")
    print(df.head(10).to_string(index=False))
    
    # Filter only videos with faces for testing
    df_testable = df[df['status'] == 'exists'].copy()
    testable_path = output_path.parent / "ff_test_videos_ready.csv"
    df_testable.to_csv(testable_path, index=False)
    
    print(f"\n✅ Testable videos CSV saved to: {testable_path}")
    print(f"   ({len(df_testable)} videos ready for testing)")
    
    return df

if __name__ == "__main__":
    parse_faceforensics_metadata()
