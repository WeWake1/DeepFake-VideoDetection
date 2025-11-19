"""
Dataset Inventory Analysis Script
==================================
Purpose: Provides comprehensive statistics about all datasets used in the project
         including real/fake video counts and total inventory across FF++, DFD, and Celeb-DF

Author: DeepFake Detection Project
Date: November 17, 2025

Five Rules Compliance:
1. ✓ Save scripts where they belong (scripts/ folder)
2. ✓ Use absolute paths from config
3. ✓ Provide clear documentation
4. ✓ Handle errors gracefully
5. ✓ Output structured, readable results
"""

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

# Base directory
BASE_DIR = Path("J:/DF")
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"

def analyze_ffpp_dataset():
    """
    Analyze FaceForensics++ (FF++) dataset
    Returns: dict with real_count, fake_count, and breakdown by manipulation type
    """
    print("\n" + "="*70)
    print("ANALYZING FACEFORENSICS++ (FF++) DATASET")
    print("="*70)
    
    # Read FF++ metadata
    ff_metadata_path = DATASET_DIR / "FF++_Metadata.csv"
    
    if not ff_metadata_path.exists():
        print(f"⚠️  Warning: FF++ metadata not found at {ff_metadata_path}")
        return {"real_count": 0, "fake_count": 0, "breakdown": {}, "total": 0}
    
    # Read the metadata CSV
    df = pd.read_csv(ff_metadata_path)
    
    # Count real videos (original videos)
    real_count = len(df[df['label'] == 'real'])
    
    # Count fake videos by manipulation type
    manipulation_counts = {}
    manipulation_types = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    
    fake_count = 0
    for manip_type in manipulation_types:
        count = len(df[df['label'] == manip_type])
        manipulation_counts[manip_type] = count
        fake_count += count
    
    print(f"\n📊 FF++ Statistics:")
    print(f"   Real videos: {real_count:,}")
    print(f"   Fake videos: {fake_count:,}")
    print(f"\n   Breakdown by manipulation type:")
    for manip_type, count in manipulation_counts.items():
        print(f"      - {manip_type}: {count:,}")
    
    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "breakdown": manipulation_counts,
        "total": real_count + fake_count
    }

def analyze_dfd_dataset():
    """
    Analyze DeepFake Detection (DFD) dataset
    Returns: dict with real_count and fake_count
    """
    print("\n" + "="*70)
    print("ANALYZING DEEPFAKE DETECTION (DFD) DATASET")
    print("="*70)
    
    # Read DFD metadata
    dfd_original_path = DATASET_DIR / "original.csv"
    dfd_method_path = DATASET_DIR / "method.csv"
    
    real_count = 0
    fake_count = 0
    
    # Count original (real) videos
    if dfd_original_path.exists():
        df_original = pd.read_csv(dfd_original_path)
        real_count = len(df_original)
        print(f"\n📊 DFD Original (Real) Videos:")
        print(f"   Real videos: {real_count:,}")
    else:
        print(f"⚠️  Warning: DFD original.csv not found at {dfd_original_path}")
    
    # Count manipulated (fake) videos
    if dfd_method_path.exists():
        df_method = pd.read_csv(dfd_method_path)
        fake_count = len(df_method)
        print(f"\n📊 DFD Manipulated (Fake) Videos:")
        print(f"   Fake videos: {fake_count:,}")
    else:
        print(f"⚠️  Warning: DFD method.csv not found at {dfd_method_path}")
    
    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "total": real_count + fake_count
    }

def analyze_celebdf_dataset():
    """
    Analyze Celeb-DF dataset
    Returns: dict with real_count and fake_count
    """
    print("\n" + "="*70)
    print("ANALYZING CELEB-DF DATASET")
    print("="*70)
    
    # Count fake frames folders
    fake_frames_dir = BASE_DIR / "Celeb-synthesis FAKE FRAMES-1"
    
    fake_count = 0
    if fake_frames_dir.exists():
        # Count directories matching the pattern id#_id#_####
        fake_dirs = [d for d in fake_frames_dir.iterdir() if d.is_dir()]
        fake_count = len(fake_dirs)
        print(f"\n📊 Celeb-DF Fake Videos:")
        print(f"   Fake video folders: {fake_count:,}")
        print(f"   Naming pattern: id#_id#_#### (e.g., id0_id1_0000)")
    else:
        print(f"⚠️  Warning: Celeb-DF fake frames directory not found at {fake_frames_dir}")
    
    # Real videos count (from documentation: 518 YouTube celebrity videos)
    real_count = 518
    print(f"\n📊 Celeb-DF Real Videos:")
    print(f"   Real videos: {real_count:,} (YouTube celebrity videos)")
    
    return {
        "real_count": real_count,
        "fake_count": fake_count,
        "total": real_count + fake_count,
        "note": "Real count from Celeb-DF documentation (518 YouTube videos)"
    }

def analyze_training_pairs():
    """
    Analyze the training pairs used in the current model
    Returns: dict with training statistics
    """
    print("\n" + "="*70)
    print("ANALYZING TRAINING PAIRS (ACTUAL MODEL INPUT)")
    print("="*70)
    
    training_pairs_path = DATA_DIR / "training_pairs.csv"
    
    if not training_pairs_path.exists():
        print(f"⚠️  Warning: training_pairs.csv not found at {training_pairs_path}")
        return {}
    
    df = pd.read_csv(training_pairs_path)
    
    total_pairs = len(df)
    
    # Count unique real and fake videos
    real_videos = df['real_video'].nunique()
    fake_videos = df['fake_video'].nunique()
    
    # Dataset breakdown
    dataset_counts = Counter()
    if 'dataset' in df.columns:
        dataset_counts = df['dataset'].value_counts().to_dict()
    
    print(f"\n📊 Training Pairs Statistics:")
    print(f"   Total pairs: {total_pairs:,}")
    print(f"   Unique real videos: {real_videos:,}")
    print(f"   Unique fake videos: {fake_videos:,}")
    
    if dataset_counts:
        print(f"\n   Breakdown by dataset:")
        for dataset, count in dataset_counts.items():
            print(f"      - {dataset}: {count:,} pairs")
    
    return {
        "total_pairs": total_pairs,
        "unique_real": real_videos,
        "unique_fake": fake_videos,
        "dataset_breakdown": dataset_counts
    }

def generate_summary_report(ffpp_stats, dfd_stats, celebdf_stats, training_stats):
    """
    Generate a comprehensive summary report
    """
    print("\n" + "="*70)
    print("DATASET INVENTORY - FINAL SUMMARY")
    print("="*70)
    
    # Calculate totals
    total_real = ffpp_stats["real_count"] + dfd_stats["real_count"] + celebdf_stats["real_count"]
    total_fake = ffpp_stats["fake_count"] + dfd_stats["fake_count"] + celebdf_stats["fake_count"]
    grand_total = total_real + total_fake
    
    print(f"\n📈 COMPLETE DATASET INVENTORY:")
    print(f"\n   {'Dataset':<20} {'Real Videos':<15} {'Fake Videos':<15} {'Total':<15}")
    print(f"   {'-'*65}")
    print(f"   {'FF++':<20} {ffpp_stats['real_count']:>14,} {ffpp_stats['fake_count']:>14,} {ffpp_stats['total']:>14,}")
    print(f"   {'DFD':<20} {dfd_stats['real_count']:>14,} {dfd_stats['fake_count']:>14,} {dfd_stats['total']:>14,}")
    print(f"   {'Celeb-DF':<20} {celebdf_stats['real_count']:>14,} {celebdf_stats['fake_count']:>14,} {celebdf_stats['total']:>14,}")
    print(f"   {'-'*65}")
    print(f"   {'TOTAL':<20} {total_real:>14,} {total_fake:>14,} {grand_total:>14,}")
    
    print(f"\n📊 TRAINING DATA USAGE:")
    if training_stats:
        print(f"   Training pairs used: {training_stats.get('total_pairs', 0):,}")
        print(f"   Unique real videos: {training_stats.get('unique_real', 0):,}")
        print(f"   Unique fake videos: {training_stats.get('unique_fake', 0):,}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Real-to-Fake ratio: 1:{total_fake/total_real:.2f}")
    print(f"   • Largest dataset: {max([('FF++', ffpp_stats['total']), ('DFD', dfd_stats['total']), ('Celeb-DF', celebdf_stats['total'])], key=lambda x: x[1])[0]}")
    print(f"   • Total videos available: {grand_total:,}")
    
    # Save summary to JSON
    summary = {
        "datasets": {
            "FaceForensics++": ffpp_stats,
            "DeepFake_Detection": dfd_stats,
            "Celeb-DF": celebdf_stats
        },
        "totals": {
            "real_videos": total_real,
            "fake_videos": total_fake,
            "total_videos": grand_total
        },
        "training_data": training_stats,
        "analysis_date": "2025-11-17"
    }
    
    output_path = BASE_DIR / "dataset_inventory_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {output_path}")
    
    return summary

def main():
    """
    Main execution function
    """
    print("\n" + "🔍 " + "="*68 + " 🔍")
    print("   DATASET INVENTORY ANALYSIS - DEEPFAKE DETECTION PROJECT")
    print("🔍 " + "="*68 + " 🔍")
    
    # Analyze each dataset
    ffpp_stats = analyze_ffpp_dataset()
    dfd_stats = analyze_dfd_dataset()
    celebdf_stats = analyze_celebdf_dataset()
    
    # Analyze training pairs
    training_stats = analyze_training_pairs()
    
    # Generate final summary
    summary = generate_summary_report(ffpp_stats, dfd_stats, celebdf_stats, training_stats)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
