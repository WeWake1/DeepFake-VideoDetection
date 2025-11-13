"""
Parse DFD metadata and create test CSV
Converts DFD folder structure into format compatible with inference pipeline
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== DFD CONFIGURATION =====
FACES_DIR = r"F:\DFD_preprocessed\faces"
OUTPUT_CSV = r"j:\DF\evaluation\dfd\dfd_test_videos.csv"

# Folders to process
FOLDERS_TO_PROCESS = {
    'DFD_original sequences': 'real',
    'DFD_manipulated_sequences': 'fake'
}

def parse_dfd_metadata():
    """Parse DFD folder structure and create test CSV"""
    print("="*70)
    print("📋 DFD METADATA PARSER")
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
            
            # Extract actor info from filename if possible
            # DFD naming: Actor_Target__Scene__Hash.mp4
            parts = video_name.split('__')
            if len(parts) >= 2:
                actors = parts[0]  # Actor_Target
                scene = parts[1] if len(parts) > 1 else 'unknown'
            else:
                actors = 'unknown'
                scene = 'unknown'
            
            if face_count > 0:
                video_data.append({
                    'video_name': video_name,
                    'label': label,
                    'folder': folder_name,
                    'actors': actors,
                    'scene': scene,
                    'faces_path': str(video_folder),
                    'face_count': face_count,
                    'status': 'exists'
                })
            else:
                video_data.append({
                    'video_name': video_name,
                    'label': label,
                    'folder': folder_name,
                    'actors': actors,
                    'scene': scene,
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
    print("\n   By folder:")
    for folder in FOLDERS_TO_PROCESS.keys():
        folder_df = df[df['folder'] == folder]
        count = len(folder_df)
        with_faces = (folder_df['status'] == 'exists').sum()
        print(f"      {folder:25s}: {with_faces:4d}/{count:4d} videos with faces")
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
    testable_path = output_path.parent / "dfd_test_videos_ready.csv"
    df_testable.to_csv(testable_path, index=False)
    
    print(f"\n✅ Testable videos CSV saved to: {testable_path}")
    print(f"   ({len(df_testable)} videos ready for testing)")
    
    return df

if __name__ == "__main__":
    parse_dfd_metadata()
