"""
Dataset Location Verification Script
=====================================
Purpose: Verify all dataset locations and identify what preprocessing is needed

Author: DeepFake Detection Project
Date: November 18, 2025

Five Rules Compliance:
1. ✓ Save scripts where they belong (scripts/ folder)
2. ✓ Use absolute paths from user-provided locations
3. ✓ Provide clear documentation
4. ✓ Handle errors gracefully
5. ✓ Output structured, readable results
"""

from pathlib import Path
import json
from collections import defaultdict

# Dataset locations from user verification (November 18, 2025)
DATASET_LOCATIONS = {
    "DFD": {
        "raw_videos": {
            "fake": Path("H:/DFD Dataset/DFD_manipulated_sequences/DFD_manipulated_sequences"),
            "real": Path("H:/DFD Dataset/DFD_original sequences")
        },
        "frames": {
            "fake": Path("F:/DFD_preprocessed/frames/DFD fake frames"),
            "real": Path("F:/DFD_preprocessed/frames/DFD real frames")
        },
        "faces": {
            "fake": None,  # NOT EXTRACTED YET
            "real": None   # NOT EXTRACTED YET
        }
    },
    "Celeb-DF": {
        "raw_videos": {
            "real": Path("H:/CELEB DataSet/Celeb-real"),
            "fake": Path("H:/CELEB DataSet/Celeb-synthesis")
        },
        "frames": None,  # Skipped - went straight to faces
        "faces": {
            "fake": Path("F:/Celeb_preprocessed/Faces/Celeb Fake Faces"),
            "real": Path("F:/Celeb_preprocessed/Faces/Celeb Real Faces")
        }
    },
    "DFDC": {
        "raw_videos": None,  # External location
        "frames": None,  # Not needed
        "faces": {
            "mixed": Path("F:/DFDC_preprocessed/faces")
        },
        "metadata": Path("J:/DF/evaluation/dfdc/dfdcmetadata.json")
    },
    "FF++": {
        "raw_videos": {
            "mixed": Path("F:/FF++ Dataset/FaceForensics++_C23")
        },
        "frames": {
            "real": Path("F:/FF++_preprocessed/frames/FF Real Frames"),
            "fake_root": Path("F:/FF++_preprocessed/frames/FF Fake Frames"),
            "fake_methods": {
                "DeepFakeDetection": Path("F:/FF++_preprocessed/frames/FF Fake Frames/DeepFakeDetection"),
                "Deepfakes": Path("F:/FF++_preprocessed/frames/FF Fake Frames/Deepfakes"),
                "Face2Face": Path("F:/FF++_preprocessed/frames/FF Fake Frames/Face2Face"),
                "FaceShifter": Path("F:/FF++_preprocessed/frames/FF Fake Frames/FaceShifter"),
                "FaceSwap": Path("F:/FF++_preprocessed/frames/FF Fake Frames/FaceSwap"),
                "NeuralTextures": Path("F:/FF++_preprocessed/frames/FF Fake Frames/NeuralTextures")
            }
        },
        "faces": {
            "fake": None,  # NOT EXTRACTED YET
            "real": None   # NOT EXTRACTED YET
        },
        "metadata": Path("J:/DF/evaluation/faceforensics++/csv provided along with the dataset")
    }
}

def check_path_exists(path, path_type="directory"):
    """Check if a path exists and return status"""
    if path is None:
        return "❌ NOT SET", None
    
    if not path.exists():
        return "❌ NOT FOUND", None
    
    if path_type == "directory":
        if path.is_dir():
            # Count items in directory
            try:
                items = list(path.iterdir())
                count = len(items)
                return "✅ EXISTS", count
            except PermissionError:
                return "⚠️ NO ACCESS", None
        else:
            return "❌ NOT A DIRECTORY", None
    else:  # file
        if path.is_file():
            return "✅ EXISTS", path.stat().st_size
        else:
            return "❌ NOT A FILE", None

def verify_all_locations():
    """Verify all dataset locations and generate report"""
    
    print("\n" + "="*80)
    print("DATASET LOCATION VERIFICATION")
    print("="*80)
    print("Date: November 18, 2025")
    print("="*80)
    
    verification_results = {}
    preprocessing_needed = defaultdict(list)
    
    for dataset_name, locations in DATASET_LOCATIONS.items():
        print(f"\n{'='*80}")
        print(f"📊 {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        dataset_results = {}
        
        # Check raw videos
        if locations.get("raw_videos"):
            print("\n🎥 Raw Videos:")
            for label, path in locations["raw_videos"].items():
                status, count = check_path_exists(path)
                print(f"   {label:10} → {status:15} {f'({count:,} items)' if count else ''}")
                print(f"              {path}")
                dataset_results[f"raw_{label}"] = {"status": status, "path": str(path), "count": count}
        
        # Check frames
        if locations.get("frames"):
            print("\n🖼️  Frames:")
            if isinstance(locations["frames"], dict):
                for label, path in locations["frames"].items():
                    if label == "fake_methods":
                        # Handle FF++ manipulation methods
                        print(f"   fake methods:")
                        for method, method_path in path.items():
                            status, count = check_path_exists(method_path)
                            print(f"      - {method:20} → {status:15} {f'({count:,} items)' if count else ''}")
                    elif label != "fake_root":
                        status, count = check_path_exists(path)
                        print(f"   {label:10} → {status:15} {f'({count:,} items)' if count else ''}")
                        print(f"              {path}")
                        dataset_results[f"frames_{label}"] = {"status": status, "path": str(path), "count": count}
            else:
                status, count = check_path_exists(locations["frames"])
                print(f"   frames     → {status:15} {f'({count:,} items)' if count else ''}")
                print(f"              {locations['frames']}")
                dataset_results["frames"] = {"status": status, "path": str(locations["frames"]), "count": count}
        
        # Check faces
        if locations.get("faces"):
            print("\n👤 Faces:")
            for label, path in locations["faces"].items():
                status, count = check_path_exists(path)
                print(f"   {label:10} → {status:15} {f'({count:,} items)' if count else ''}")
                if path:
                    print(f"              {path}")
                dataset_results[f"faces_{label}"] = {"status": status, "path": str(path) if path else None, "count": count}
                
                # Track if face extraction is needed
                if status == "❌ NOT SET" and locations.get("frames"):
                    preprocessing_needed[dataset_name].append(f"Extract {label} faces")
        
        # Check metadata
        if locations.get("metadata"):
            print("\n📋 Metadata:")
            status, size = check_path_exists(locations["metadata"], "file")
            print(f"   metadata   → {status:15}")
            print(f"              {locations['metadata']}")
            dataset_results["metadata"] = {"status": status, "path": str(locations["metadata"]), "size": size}
        
        verification_results[dataset_name] = dataset_results
    
    # Summary report
    print("\n" + "="*80)
    print("📊 PREPROCESSING STATUS SUMMARY")
    print("="*80)
    
    datasets_status = {
        "Celeb-DF": "✅ COMPLETE (Raw → Faces extracted)",
        "DFDC": "✅ COMPLETE (Faces extracted, metadata ready)",
        "DFD": "⚠️ PARTIAL (Frames extracted, faces NOT extracted)",
        "FF++": "⚠️ PARTIAL (Frames extracted, faces NOT extracted)"
    }
    
    for dataset, status in datasets_status.items():
        print(f"\n   {dataset:15} → {status}")
    
    # What needs to be done
    print("\n" + "="*80)
    print("🔧 PREPROCESSING NEEDED")
    print("="*80)
    
    print("\n1️⃣  DFD Dataset:")
    print("   ❌ Face extraction NOT done")
    print("   📍 Input (frames):")
    print(f"      - Real: {DATASET_LOCATIONS['DFD']['frames']['real']}")
    print(f"      - Fake: {DATASET_LOCATIONS['DFD']['frames']['fake']}")
    print("   📍 Output (faces) - TO BE CREATED:")
    print("      - Real: F:/DFD_preprocessed/faces/DFD real faces")
    print("      - Fake: F:/DFD_preprocessed/faces/DFD fake faces")
    
    print("\n2️⃣  FF++ Dataset:")
    print("   ❌ Face extraction NOT done")
    print("   📍 Input (frames):")
    print(f"      - Real: {DATASET_LOCATIONS['FF++']['frames']['real']}")
    print("      - Fake (6 methods):")
    print(f"        * DeepFakeDetection: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['DeepFakeDetection']}")
    print(f"        * Deepfakes: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['Deepfakes']}")
    print(f"        * Face2Face: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['Face2Face']}")
    print(f"        * FaceShifter: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['FaceShifter']}")
    print(f"        * FaceSwap: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['FaceSwap']}")
    print(f"        * NeuralTextures: {DATASET_LOCATIONS['FF++']['frames']['fake_methods']['NeuralTextures']}")
    print("   📍 Output (faces) - TO BE CREATED:")
    print("      - Real: F:/FF++_preprocessed/faces/FF Real Faces")
    print("      - Fake: F:/FF++_preprocessed/faces/FF Fake Faces (all methods combined)")
    
    # Count verification
    print("\n" + "="*80)
    print("📈 VIDEO/FRAME COUNT VERIFICATION")
    print("="*80)
    
    print("\n✅ Datasets with Faces Extracted:")
    
    # Celeb-DF
    celeb_real_faces = DATASET_LOCATIONS['Celeb-DF']['faces']['real']
    celeb_fake_faces = DATASET_LOCATIONS['Celeb-DF']['faces']['fake']
    if celeb_real_faces and celeb_real_faces.exists():
        real_count = len(list(celeb_real_faces.iterdir()))
        print(f"   Celeb-DF Real: {real_count:,} folders")
    if celeb_fake_faces and celeb_fake_faces.exists():
        fake_count = len(list(celeb_fake_faces.iterdir()))
        print(f"   Celeb-DF Fake: {fake_count:,} folders")
    
    # DFDC
    dfdc_faces = DATASET_LOCATIONS['DFDC']['faces']['mixed']
    if dfdc_faces and dfdc_faces.exists():
        dfdc_count = len(list(dfdc_faces.iterdir()))
        print(f"   DFDC Mixed: {dfdc_count:,} folders")
    
    print("\n⚠️ Datasets with Frames Only (Need Face Extraction):")
    
    # DFD
    dfd_real_frames = DATASET_LOCATIONS['DFD']['frames']['real']
    dfd_fake_frames = DATASET_LOCATIONS['DFD']['frames']['fake']
    if dfd_real_frames and dfd_real_frames.exists():
        real_count = len(list(dfd_real_frames.iterdir()))
        print(f"   DFD Real Frames: {real_count:,} folders")
    if dfd_fake_frames and dfd_fake_frames.exists():
        fake_count = len(list(dfd_fake_frames.iterdir()))
        print(f"   DFD Fake Frames: {fake_count:,} folders")
    
    # FF++
    ff_real_frames = DATASET_LOCATIONS['FF++']['frames']['real']
    if ff_real_frames and ff_real_frames.exists():
        real_count = len(list(ff_real_frames.iterdir()))
        print(f"   FF++ Real Frames: {real_count:,} folders")
    
    ff_fake_methods = DATASET_LOCATIONS['FF++']['frames']['fake_methods']
    total_fake_count = 0
    for method, method_path in ff_fake_methods.items():
        if method_path and method_path.exists():
            count = len(list(method_path.iterdir()))
            total_fake_count += count
    if total_fake_count > 0:
        print(f"   FF++ Fake Frames: {total_fake_count:,} folders (across 6 methods)")
    
    # Next steps
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    
    print("\n1. Create face extraction scripts for:")
    print("   • DFD dataset (extract faces from frames)")
    print("   • FF++ dataset (extract faces from frames)")
    
    print("\n2. Run face extraction (GPU accelerated with MTCNN):")
    print("   • Estimated time: 4-8 hours for both datasets")
    print("   • Output: ~200-300 GB of face images")
    
    print("\n3. After face extraction completes:")
    print("   • Parse metadata for DFD and FF++")
    print("   • Create combined training CSV")
    print("   • Calculate class weights")
    print("   • Ready for Model B training!")
    
    # Save results to JSON
    output_path = Path("J:/DF/dataset_verification_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "verification_date": "2025-11-18",
            "datasets": verification_results,
            "preprocessing_needed": dict(preprocessing_needed),
            "next_steps": [
                "Extract faces from DFD frames",
                "Extract faces from FF++ frames",
                "Parse metadata for both datasets",
                "Create combined training dataset"
            ]
        }, f, indent=2)
    
    print(f"\n💾 Verification results saved to: {output_path}")
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    verify_all_locations()
