"""
Delete all the badly detected faces from RetinaFace attempt
Removes all FACES folders created during the failed run
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import csv

# Drives where faces might have been created
DRIVES = ['H:', 'J:', 'I:']

def find_and_delete_face_folders():
    """Find and delete all FACES folders"""
    
    deleted_count = 0
    deleted_size = 0
    
    print("Searching for FACES folders to delete...")
    print()
    
    for drive in DRIVES:
        # Search for FACES folders
        drive_path = Path(drive + '\\')
        
        if not drive_path.exists():
            print(f"⚠ Drive {drive} not found, skipping...")
            continue
        
        print(f"Scanning {drive}...")
        
        # Find all FACES folders
        face_folders = list(drive_path.rglob('*FACES*'))
        
        for folder in tqdm(face_folders, desc=f"Deleting from {drive}"):
            if folder.is_dir() and 'FACES' in folder.name:
                try:
                    # Calculate size before deletion
                    size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                    deleted_size += size
                    
                    # Delete folder and all contents
                    shutil.rmtree(folder)
                    deleted_count += 1
                    
                except Exception as e:
                    print(f"Error deleting {folder}: {e}")
    
    print()
    print("=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    print(f"Folders deleted: {deleted_count}")
    print(f"Space freed: {deleted_size / 1e9:.2f} GB")
    print()

if __name__ == "__main__":
    print("=" * 80)
    print("DELETE BAD RETINAFACE RESULTS")
    print("=" * 80)
    print()
    print("This will delete ALL 'FACES' folders from drives H:, J:, and I:")
    print()
    
    confirm = input("Type 'DELETE' to confirm: ")
    
    if confirm == 'DELETE':
        find_and_delete_face_folders()
        print("✓ Cleanup complete!")
    else:
        print("Cancelled.")
